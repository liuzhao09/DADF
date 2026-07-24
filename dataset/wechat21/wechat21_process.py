
import pandas as pd
import numpy as np
import pickle

print("Step 1: 读取原始数据...")
df_action = pd.read_csv("raw_data/user_action.csv")
df_feed   = pd.read_csv("raw_data/feed_info.csv")
print("  user_action: {} 行".format(len(df_action)))
print("  feed_info:   {} 行".format(len(df_feed)))

print("Step 2: JOIN + 预处理...")
df = pd.merge(df_action, df_feed, on="feedid", how="left")
df = df.fillna('12345')

df['play']             = pd.to_numeric(df['play'],             errors='coerce').fillna(0)
df['videoplayseconds'] = pd.to_numeric(df['videoplayseconds'], errors='coerce').fillna(0)
df = df[df['play'] > 0]
df = df[df['videoplayseconds'] > 0]
print("  过滤后行数: {}".format(len(df)))

df = df.sample(frac=1.0, random_state=312).reset_index(drop=True)
print("  打散完成")

print("Step 3: 删除不使用的列...")
drop_cols = [

    'description', 'ocr', 'asr',
    'description_char', 'ocr_char', 'asr_char',

    'read_comment', 'comment', 'like', 'stay',
    'click_avatar', 'forward', 'follow', 'favorite',
]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

label            = "play_time"
sequence_feature = "manual_keyword_list"

sparse_feature_list = [
    'manual_keyword_list',
    'user_id',
    'video_id',
    'author_id',
    'device',
    'date_',
    'bgm_song_id',
    'bgm_singer_id',
]
dense_feature_list = ['duration']

print("Step 5: 构建 play_time / duration...")
desc = [('play_time', -1, 'label'), ('duration', -1, 'ctn')]

df['play_time'] = df['play'].astype(float)
df['duration']  = df['videoplayseconds'].astype(float) * 1000.0

print("  play_time 范围: [{:.1f}, {:.1f}] ms".format(df['play_time'].min(), df['play_time'].max()))
print("  duration  范围: [{:.1f}, {:.1f}] ms".format(df['duration'].min(),  df['duration'].max()))

df = df.rename(columns={'feedid': 'video_id', 'authorid': 'author_id', 'userid': 'user_id'})

print("Step 6: 稀疏 / 序列特征编码...")

for sparse_feature_name in sparse_feature_list:

    if sparse_feature_name not in df.columns:
        print("  [SKIP] {} 列不存在".format(sparse_feature_name))
        continue

    if sparse_feature_name == sequence_feature:

        df[sparse_feature_name] = df[sparse_feature_name].astype(str).replace('12345', '0')
        df[sparse_feature_name] = df[sparse_feature_name].apply(
            lambda x: x if x not in ('', 'nan', 'NaN', 'None') else '0'
        )

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

print("Step 7: 过滤 play_time >= duration * 10...")
before = len(df)
df = df[df['play_time'] < df['duration'] * 10]
print("  过滤掉 {} 行，剩余 {} 行".format(before - len(df), len(df)))

keep_cols = [name for name, _, _ in desc]
df = df[[c for c in keep_cols if c in df.columns]]
print("  保留列: {}".format(list(df.columns)))

def normalize_split(df_in, max_val):
    df_out = df_in.copy()
    df_out['play_time'] = df_out['play_time'] / max_val
    df_out['duration']  = df_out['duration'].clip(upper=max_val) / max_val
    return df_out

print("Step 9: 全量版 80/10/10 split（先 split，再计算统计量）...")
df_full           = df.sample(frac=1, random_state=1234).reset_index(drop=True)
n_full            = len(df_full)
n_train_full      = int(0.8 * n_full)
n_val_full        = int(0.9 * n_full) - n_train_full
df_train_full_raw = df_full.iloc[:n_train_full].copy()
df_val_full_raw   = df_full.iloc[n_train_full:n_train_full + n_val_full].copy()
df_test_full_raw  = df_full.iloc[n_train_full + n_val_full:].copy()

max_val            = float(df_train_full_raw['play_time'].max())
video_duration_max = float(df_train_full_raw['duration'].max())
print("  play_duration_max (train-only, no leakage): {:.2f} ms".format(max_val))
print("  video_duration_max (train-only):             {:.2f} ms".format(video_duration_max))

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

df_train_full = normalize_split(df_train_full_raw, max_val)
df_val_full   = normalize_split(df_val_full_raw,   max_val)
df_test_full  = normalize_split(df_test_full_raw,  max_val)
print("  全量 train={} val={} test={}".format(len(df_train_full), len(df_val_full), len(df_test_full)))

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

print("Step 11: 10% 采样版 80/10/10 split（独立计算统计量，无穿越）...")
df_10pct           = df.sample(frac=0.1, random_state=312).reset_index(drop=True)
df_10pct           = df_10pct.sample(frac=1, random_state=1234).reset_index(drop=True)
n_10pct            = len(df_10pct)
n_train_10pct      = int(0.8 * n_10pct)
n_val_10pct        = int(0.9 * n_10pct) - n_train_10pct
df_train_10pct_raw = df_10pct.iloc[:n_train_10pct].copy()
df_val_10pct_raw   = df_10pct.iloc[n_train_10pct:n_train_10pct + n_val_10pct].copy()
df_test_10pct_raw  = df_10pct.iloc[n_train_10pct + n_val_10pct:].copy()

max_val_10pct            = float(df_train_10pct_raw['play_time'].max())
video_duration_max_10pct = float(df_train_10pct_raw['duration'].max())
print("  play_duration_max (10pct train-only): {:.2f} ms".format(max_val_10pct))
print("  video_duration_max (10pct train-only): {:.2f} ms".format(video_duration_max_10pct))

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

desc_10pct = [item for item in desc if item[0] != 'duration_bucket']
desc_10pct.append(('duration_bucket', n_actual_bins_10pct, 'spr'))
print("  10pct bucket 数: {}（全量版: {}）".format(n_actual_bins_10pct, n_actual_bins))

df_train_10pct = normalize_split(df_train_10pct_raw, max_val_10pct)
df_val_10pct   = normalize_split(df_val_10pct_raw,   max_val_10pct)
df_test_10pct  = normalize_split(df_test_10pct_raw,  max_val_10pct)
print("  10pct  train={} val={} test={}".format(len(df_train_10pct), len(df_val_10pct), len(df_test_10pct)))

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
    "description":        desc_10pct,
    "play_duration_max":  max_val_10pct,
    "video_duration_max": video_duration_max_10pct,
}
with open('./wechat21_data.pkl', 'wb+') as f:
    pickle.dump(data_10pct, f)
print("saved wechat21_data.pkl       train={} val={} test={}".format(len(df_train_10pct), len(df_val_10pct), len(df_test_10pct)))

print("\n=== description ===")
for item in desc:
    print("  {}".format(item))
