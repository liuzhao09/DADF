"""
WeChat21 数据集预处理脚本
来源：微信视频推荐数据集（WeChatBigData Challenge 2021）

运行方式（从项目根目录）:
  cd dataset/wechat21 && python wechat21_process.py

原始数据文件（raw_data/ 目录）:
  user_action.csv  — userid, feedid, play, date_, device, ...
  feed_info.csv    — feedid, authorid, videoplayseconds, manual_keyword_list, ...

输出:
  wechat21_data.pkl       — 10% 采样版（默认使用，约 122 万条）
  wechat21_data_full.pkl  — 全量版
  d2q_duration_bucket_ranges.csv
  d2q_duration_bucket_playtime_quantiles.csv
  d2q_duration_bucket_playtime_quantiles_10pct.csv
"""

import pandas as pd
import numpy as np
import pickle

# ── 1. 读取原始数据 ────────────────────────────────────────────────────────────
print("Step 1: 读取原始数据...")
df_action = pd.read_csv("raw_data/user_action.csv")
df_feed   = pd.read_csv("raw_data/feed_info.csv")
print("  user_action: {} 行".format(len(df_action)))
print("  feed_info:   {} 行".format(len(df_feed)))

# ── 2. JOIN + 预处理 ───────────────────────────────────────────────────────────
print("Step 2: JOIN + 预处理...")
df = pd.merge(df_action, df_feed, on="feedid", how="left")
df = df.fillna('12345')

# 过滤：play > 0 AND videoplayseconds > 0
df['play']             = pd.to_numeric(df['play'],             errors='coerce').fillna(0)
df['videoplayseconds'] = pd.to_numeric(df['videoplayseconds'], errors='coerce').fillna(0)
df = df[df['play'] > 0]
df = df[df['videoplayseconds'] > 0]
print("  过滤后行数: {}".format(len(df)))

# 固定种子打散。特征词表跨划分共享，标签统计在 split 后仅由训练集计算。
df = df.sample(frac=1.0, random_state=312).reset_index(drop=True)
print("  打散完成")

# ── 3. 删除不使用的列 ──────────────────────────────────────────────────────────
print("Step 3: 删除不使用的列...")
drop_cols = [
    # 文本字段（太长，不用于 embedding）
    'description', 'ocr', 'asr',
    'description_char', 'ocr_char', 'asr_char',
    # 行为结果（非输入特征，是标签而非特征）
    'read_comment', 'comment', 'like', 'stay',
    'click_avatar', 'forward', 'follow', 'favorite',
]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# ── 4. 特征定义 ────────────────────────────────────────────────────────────────
label            = "play_time"
sequence_feature = "manual_keyword_list"

sparse_feature_list = [
    'manual_keyword_list',   # 序列特征
    'user_id',
    'video_id',
    'author_id',
    'device',
    'date_',
    'bgm_song_id',
    'bgm_singer_id',
]
dense_feature_list = ['duration']

# ── 5. 构建 label 和 duration ─────────────────────────────────────────────────
print("Step 5: 构建 play_time / duration...")
desc = [('play_time', -1, 'label'), ('duration', -1, 'ctn')]

df['play_time'] = df['play'].astype(float)                         # ms（直接使用原始 play 字段）
df['duration']  = df['videoplayseconds'].astype(float) * 1000.0   # 秒 → ms，与 play_time 单位对齐

print("  play_time 范围: [{:.1f}, {:.1f}] ms".format(df['play_time'].min(), df['play_time'].max()))
print("  duration  范围: [{:.1f}, {:.1f}] ms".format(df['duration'].min(),  df['duration'].max()))

# 统一列名（与 KuaiRec 对齐，方便复用 DataLoader 和模型代码）
df = df.rename(columns={'feedid': 'video_id', 'authorid': 'author_id', 'userid': 'user_id'})

# ── 6. 稀疏 / 序列特征编码（向量化，vocab 基于全量）────────────────────────────
print("Step 6: 稀疏 / 序列特征编码...")

for sparse_feature_name in sparse_feature_list:

    if sparse_feature_name not in df.columns:
        print("  [SKIP] {} 列不存在".format(sparse_feature_name))
        continue

    if sparse_feature_name == sequence_feature:
        # 序列特征：manual_keyword_list，分号分隔的整数 ID，空值填 '0'
        df[sparse_feature_name] = df[sparse_feature_name].astype(str).replace('12345', '0')
        df[sparse_feature_name] = df[sparse_feature_name].apply(
            lambda x: x if x not in ('', 'nan', 'NaN', 'None') else '0'
        )

        # explode 获取全局词表
        series = df[sparse_feature_name].str.split(';')
        all_tokens = sorted(set(series.explode().unique()))
        sparse_feature_voc = {word: idx + 1 for idx, word in enumerate(all_tokens)}
        vocab_size = len(sparse_feature_voc)
        print("  sparse feature {} size: {}".format(sparse_feature_name, vocab_size))

        def encode_and_pad(seq_str, voc=sparse_feature_voc):
            tokens = [voc[t] for t in seq_str.split(';') if t in voc]
            tokens = tokens[:10]
            mask   = [1.0] * len(tokens) + [0.0] * (10 - len(tokens))
            tokens = tokens + [0] * (10 - len(tokens))
            return tokens, mask

        encoded = df[sparse_feature_name].astype(str).apply(encode_and_pad)
        df[sparse_feature_name]         = encoded.apply(lambda x: x[0])
        df['manual_keyword_listmask']   = encoded.apply(lambda x: x[1])

        desc.append((sparse_feature_name, vocab_size + 1, 'seq'))
        desc.append(('manual_keyword_listmask', -1, 'seqm'))

    else:
        # 普通稀疏特征：全列转 str → 建 vocab → map（纯向量化）
        col_str     = df[sparse_feature_name].astype(str)
        unique_vals = sorted(col_str.unique())
        sparse_feature_voc = {word: idx + 1 for idx, word in enumerate(unique_vals)}
        vocab_size  = len(sparse_feature_voc)
        print("  sparse feature {} size: {}".format(sparse_feature_name, vocab_size))

        if vocab_size == 1:
            df = df.drop(sparse_feature_name, axis=1)
            print("  remove the {} feature !!!!!".format(sparse_feature_name))
        else:
            df[sparse_feature_name] = col_str.map(sparse_feature_voc)
            desc.append((sparse_feature_name, vocab_size + 1, 'spr'))

# ── 7. 过滤异常 play_time（split 前过滤，不涉及统计量）─────────────────────────
print("Step 7: 过滤 play_time >= duration * 10...")
before = len(df)
df = df[df['play_time'] < df['duration'] * 10]
print("  过滤掉 {} 行，剩余 {} 行".format(before - len(df), len(df)))

# ── 8. 只保留 description 中的列（丢弃多余字符串列）──────────────────────────
keep_cols = [name for name, _, _ in desc]
df = df[[c for c in keep_cols if c in df.columns]]
print("  保留列: {}".format(list(df.columns)))


def normalize_split(df_in, max_val):
    df_out = df_in.copy()
    df_out['play_time'] = df_out['play_time'] / max_val
    df_out['duration']  = df_out['duration'].clip(upper=max_val) / max_val
    return df_out


# ── 9. 全量版：先 split（80/10/10 train/val/test），再从 train 计算所有统计量（无穿越）──
print("Step 9: 全量版 80/10/10 split（先 split，再计算统计量）...")
df_full           = df.sample(frac=1, random_state=1234).reset_index(drop=True)
n_full            = len(df_full)
n_train_full      = int(0.8 * n_full)
n_val_full        = int(0.9 * n_full) - n_train_full
df_train_full_raw = df_full.iloc[:n_train_full].copy()
df_val_full_raw   = df_full.iloc[n_train_full:n_train_full + n_val_full].copy()
df_test_full_raw  = df_full.iloc[n_train_full + n_val_full:].copy()

# max_val：只用 train（修复穿越）
max_val            = float(df_train_full_raw['play_time'].max())
video_duration_max = float(df_train_full_raw['duration'].max())
print("  play_duration_max (train-only, no leakage): {:.2f} ms".format(max_val))
print("  video_duration_max (train-only):             {:.2f} ms".format(video_duration_max))

# duration_bucket：pd.qcut 只在 train 上建桶（修复穿越）
# test 用 pd.cut + extended_bins（±inf）确保所有 test 值都有合法桶
print("Step 10: 生成 duration_bucket（train-only qcut，无穿越）...")
n_bins = 50
df_train_full_raw['duration_bucket'], bins = pd.qcut(
    df_train_full_raw['duration'], q=n_bins, labels=False,
    retbins=True, duplicates='drop'
)
df_train_full_raw['duration_bucket'] = df_train_full_raw['duration_bucket'].astype(int)

extended_bins     = bins.copy()
extended_bins[0]  = -np.inf
extended_bins[-1] =  np.inf
df_test_full_raw['duration_bucket'] = pd.cut(
    df_test_full_raw['duration'], bins=extended_bins, labels=False, include_lowest=True
).astype(int)
df_val_full_raw['duration_bucket'] = pd.cut(
    df_val_full_raw['duration'], bins=extended_bins, labels=False, include_lowest=True
).astype(int)

n_actual_bins = len(bins) - 1
bucket_ranges = pd.DataFrame({
    'bucket_index': range(n_actual_bins),
    'min_duration': bins[:-1],
    'max_duration': bins[1:],
})
bucket_ranges.to_csv('d2q_duration_bucket_ranges.csv', index=False)
print("  实际桶数: {}，已保存 d2q_duration_bucket_ranges.csv".format(n_actual_bins))
desc.append(('duration_bucket', n_actual_bins, 'spr'))

# 归一化（均用 train max_val）
df_train_full = normalize_split(df_train_full_raw, max_val)
df_val_full   = normalize_split(df_val_full_raw,   max_val)
df_test_full  = normalize_split(df_test_full_raw,  max_val)
print("  全量 train={} val={} test={}".format(len(df_train_full), len(df_val_full), len(df_test_full)))

# D2Q 分位数：只用归一化后的 train（无穿越，与原逻辑一致）
quantile_num = 100
quantiles    = np.linspace(0, 1, quantile_num + 1)
quantile_df  = (df_train_full.groupby('duration_bucket')['play_time']
                .quantile(quantiles).reset_index()
                .rename(columns={'level_1': 'quantile'}))
quantile_pivot = quantile_df.pivot(index='duration_bucket', columns='quantile', values='play_time')
quantile_pivot.to_csv('d2q_duration_bucket_playtime_quantiles.csv')
print("  已保存 d2q_duration_bucket_playtime_quantiles.csv（train-only）")

data_full = {
    "train":              df_train_full,
    "val":                df_val_full,
    "test":               df_test_full,
    "description":        desc,
    "play_duration_max":  max_val,
    "video_duration_max": video_duration_max,
}
with open('./wechat21_data_full.pkl', 'wb+') as f:
    pickle.dump(data_full, f)
print("saved wechat21_data_full.pkl  train={} val={} test={}".format(len(df_train_full), len(df_val_full), len(df_test_full)))

# ── 11. 10% 采样版：完全独立计算所有统计量（无穿越）────────────────────────────
print("Step 11: 10% 采样版 80/10/10 split（独立计算统计量，无穿越）...")
df_10pct           = df.sample(frac=0.1, random_state=312).reset_index(drop=True)
df_10pct           = df_10pct.sample(frac=1, random_state=1234).reset_index(drop=True)
n_10pct            = len(df_10pct)
n_train_10pct      = int(0.8 * n_10pct)
n_val_10pct        = int(0.9 * n_10pct) - n_train_10pct
df_train_10pct_raw = df_10pct.iloc[:n_train_10pct].copy()
df_val_10pct_raw   = df_10pct.iloc[n_train_10pct:n_train_10pct + n_val_10pct].copy()
df_test_10pct_raw  = df_10pct.iloc[n_train_10pct + n_val_10pct:].copy()

# 10pct max_val 和 video_duration_max：只用 10pct train（修复穿越）
max_val_10pct            = float(df_train_10pct_raw['play_time'].max())
video_duration_max_10pct = float(df_train_10pct_raw['duration'].max())
print("  play_duration_max (10pct train-only): {:.2f} ms".format(max_val_10pct))
print("  video_duration_max (10pct train-only): {:.2f} ms".format(video_duration_max_10pct))

# 10pct duration_bucket：pd.qcut 在 10pct train 上建桶，test 用 pd.cut
df_train_10pct_raw['duration_bucket'], bins_10pct = pd.qcut(
    df_train_10pct_raw['duration'], q=n_bins, labels=False,
    retbins=True, duplicates='drop'
)
df_train_10pct_raw['duration_bucket'] = df_train_10pct_raw['duration_bucket'].astype(int)

extended_bins_10pct     = bins_10pct.copy()
extended_bins_10pct[0]  = -np.inf
extended_bins_10pct[-1] =  np.inf
df_test_10pct_raw['duration_bucket'] = pd.cut(
    df_test_10pct_raw['duration'], bins=extended_bins_10pct, labels=False, include_lowest=True
).astype(int)
df_val_10pct_raw['duration_bucket'] = pd.cut(
    df_val_10pct_raw['duration'], bins=extended_bins_10pct, labels=False, include_lowest=True
).astype(int)

n_actual_bins_10pct = len(bins_10pct) - 1
# 10pct 的 description 独立
desc_10pct = [item for item in desc if item[0] != 'duration_bucket']
desc_10pct.append(('duration_bucket', n_actual_bins_10pct, 'spr'))
print("  10pct bucket 数: {}（全量版: {}）".format(n_actual_bins_10pct, n_actual_bins))

# 归一化（均用 10pct train max_val）
df_train_10pct = normalize_split(df_train_10pct_raw, max_val_10pct)
df_val_10pct   = normalize_split(df_val_10pct_raw,   max_val_10pct)
df_test_10pct  = normalize_split(df_test_10pct_raw,  max_val_10pct)
print("  10pct  train={} val={} test={}".format(len(df_train_10pct), len(df_val_10pct), len(df_test_10pct)))

# D2Q 分位数：只用 10pct train
quantile_df_10pct = (df_train_10pct.groupby('duration_bucket')['play_time']
                     .quantile(quantiles).reset_index()
                     .rename(columns={'level_1': 'quantile'}))
quantile_pivot_10pct = quantile_df_10pct.pivot(
    index='duration_bucket', columns='quantile', values='play_time'
)
quantile_pivot_10pct.to_csv('d2q_duration_bucket_playtime_quantiles_10pct.csv')
print("  已保存 d2q_duration_bucket_playtime_quantiles_10pct.csv（10pct train-only）")

data_10pct = {
    "train":              df_train_10pct,
    "val":                df_val_10pct,
    "test":               df_test_10pct,
    "description":        desc_10pct,              # 10pct 独立 description
    "play_duration_max":  max_val_10pct,           # 10pct 独立归一化分母
    "video_duration_max": video_duration_max_10pct,
}
with open('./wechat21_data.pkl', 'wb+') as f:
    pickle.dump(data_10pct, f)
print("saved wechat21_data.pkl       train={} val={} test={}".format(len(df_train_10pct), len(df_val_10pct), len(df_test_10pct)))

# ── 12. 打印 description ──────────────────────────────────────────────────────
print("\n=== description ===")
for item in desc:
    print("  {}".format(item))
