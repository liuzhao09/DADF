import pandas as pd
import numpy as np
import pickle

with open("raw_data/kuairec_caption_category.csv", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

with open("raw_data/cleaned_kuairec_caption_category.csv", "w", encoding="utf-8") as f:
    f.writelines(lines)

df5 = pd.read_csv("raw_data/cleaned_kuairec_caption_category.csv", encoding="utf-8")
df5["video_id"] = pd.to_numeric(df5["video_id"], errors="coerce")
df5 = df5.dropna(subset=["video_id"])
df5["video_id"] = df5["video_id"].astype(int)
df5.to_csv("raw_data/cleaned_kuairec_caption_category.csv", index=False, encoding="utf-8")

#join feature files to one file
df1 = pd.read_csv("raw_data/big_matrix.csv")
df2 = pd.read_csv("raw_data/user_features.csv")
df3 = pd.read_csv("raw_data/item_daily_features.csv").loc[:, ['video_id', 'video_type','music_id','video_tag_id']]
df3 = df3.drop_duplicates(subset=['video_id'])
df4 = pd.read_csv("raw_data/item_categories.csv")
df5 = pd.read_csv("raw_data/cleaned_kuairec_caption_category.csv", encoding="utf-8").loc[:,['video_id', 'first_level_category_id', 'second_level_category_id', 'third_level_category_id']]

df_merged_1_2 = pd.merge(df1, df2, on="user_id", how="left")
df_merged_1_2_3 = pd.merge(df_merged_1_2, df3, on="video_id", how="left")
df_merged_1_2_3_4 = pd.merge(df_merged_1_2_3, df4, on="video_id", how="left")
df_final = pd.merge(df_merged_1_2_3_4, df5, on="video_id", how="left")

df = df_final.fillna('12345')

#filter AD video
df = df[df['video_type'] != 'AD']
df = df[df['play_duration'] > 0]
df = df[df['video_duration'] > 0]

# 固定种子打散。特征词表跨划分共享，标签统计在 split 后仅由训练集计算。
df = df.sample(frac=1.0, random_state=312).reset_index(drop=True)

#delete unnecessary features
df = df.drop(columns=['video_type','time','date','watch_ratio','follow_user_num','fans_user_num','friend_user_num','register_days'])


# indentify different types of feature
label="play_duration"
sequence_feature="feat"
sparse_feature_list=['feat','user_id','video_id','music_id','video_tag_id','user_active_degree','is_lowactive_period','is_live_streamer','is_video_author','follow_user_num_range','fans_user_num_range','friend_user_num_range','register_days_range','onehot_feat0','onehot_feat1','onehot_feat2','onehot_feat3','onehot_feat4','onehot_feat5','onehot_feat6','onehot_feat7','onehot_feat8','onehot_feat9','onehot_feat10','onehot_feat11','onehot_feat12','onehot_feat13','onehot_feat14','onehot_feat15','onehot_feat16','onehot_feat17','first_level_category_id','second_level_category_id','third_level_category_id']
dense_feature_list=['video_duration','timestamp']

# create feature desc and process play_time_ms/duration_ms
desc=[('play_time', -1, 'label'), ('duration', -1, 'ctn')]
df['play_time'] = df['play_duration'].astype(float)
df['duration'] = df['video_duration'].astype(float)

# ── 向量化稀疏特征编码（替换原来的 iterrows 循环，速度提升 100x+）──────────────
for sparse_feature_name in sparse_feature_list:

    if sparse_feature_name == sequence_feature:
        # 序列特征：explode 获取全局词表，再 apply 做 padding
        series = df[sparse_feature_name].astype(str).str.split(',')
        all_tokens = sorted(set(series.explode().unique()))
        sparse_feature_voc = {word: idx + 1 for idx, word in enumerate(all_tokens)}
        vocab_size = len(sparse_feature_voc)
        print("sparse feature {} size: {}".format(sparse_feature_name, vocab_size))

        def encode_and_pad(seq_str):
            tokens = [sparse_feature_voc[t] for t in seq_str.split(',') if t in sparse_feature_voc]
            tokens = tokens[:10]
            mask = [1.0] * len(tokens) + [0.0] * (10 - len(tokens))
            tokens = tokens + [0] * (10 - len(tokens))
            return tokens, mask

        encoded = df[sparse_feature_name].astype(str).apply(encode_and_pad)
        df[sparse_feature_name] = encoded.apply(lambda x: x[0])
        df['featmask'] = encoded.apply(lambda x: x[1])

        desc.append((sparse_feature_name, vocab_size + 1, 'seq'))
        desc.append(('featmask', -1, 'seqm'))

    else:
        # 普通稀疏特征：全列转 str → 建 vocab → map（纯向量化）
        col_str = df[sparse_feature_name].astype(str)
        unique_vals = sorted(col_str.unique())
        sparse_feature_voc = {word: idx + 1 for idx, word in enumerate(unique_vals)}
        vocab_size = len(sparse_feature_voc)
        print("sparse feature {} size: {}".format(sparse_feature_name, vocab_size))

        if vocab_size == 1:
            df = df.drop(sparse_feature_name, axis=1)
            print("remove the {} feature !!!!!".format(sparse_feature_name))
        else:
            df[sparse_feature_name] = col_str.map(sparse_feature_voc)
            desc.append((sparse_feature_name, vocab_size + 1, 'spr'))

# ── 过滤异常 play_time（split 前过滤，不涉及统计量）─────────────────────────────
df = df[df['play_time'] < df['duration'] * 10]

def normalize_split(df_in, max_val):
    df_out = df_in.copy()
    df_out['play_time'] = df_out['play_time'] / max_val
    df_out['duration'] = df_out['duration'].clip(upper=max_val) / max_val
    return df_out

# ── 全量版：先 split（80/10/10 train/val/test），再从 train 计算所有统计量（无穿越）──
df_full          = df.sample(frac=1, random_state=1234).reset_index(drop=True)
n_full           = len(df_full)
n_train_full     = int(0.8 * n_full)
n_val_full       = int(0.9 * n_full) - n_train_full
df_train_full_raw = df_full.iloc[:n_train_full].copy()
df_val_full_raw   = df_full.iloc[n_train_full:n_train_full + n_val_full].copy()
df_test_full_raw  = df_full.iloc[n_train_full + n_val_full:].copy()

# 1. max_val：只用 train（修复穿越：原来用全量）
max_val            = float(df_train_full_raw['play_time'].max())
video_duration_max = float(df_train_full_raw['video_duration'].max())
print("play_duration_max (train-only, no leakage): {}".format(max_val))
print("video_duration_max (train-only):             {}".format(video_duration_max))

# 2. duration_bucket：pd.qcut 只在 train 上建桶（修复穿越：原来用全量）
#    test 用 pd.cut + extended_bins（±inf）确保所有 test 值都有合法桶，不产生 NaN
n_bins = 50
df_train_full_raw['duration_bucket'], bins = pd.qcut(
    df_train_full_raw['duration'], q=n_bins, labels=False,
    retbins=True, duplicates='drop'
)
df_train_full_raw['duration_bucket'] = df_train_full_raw['duration_bucket'].astype(int)

extended_bins        = bins.copy()
extended_bins[0]     = -np.inf  # 覆盖 test 中低于 train 最小值的极端情况
extended_bins[-1]    =  np.inf  # 覆盖 test 中高于 train 最大值的极端情况
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
desc.append(('duration_bucket', n_actual_bins, 'spr'))

# 3. 归一化（均用 train max_val）
df_train_full = normalize_split(df_train_full_raw, max_val)
df_val_full   = normalize_split(df_val_full_raw,   max_val)
df_test_full  = normalize_split(df_test_full_raw,  max_val)

# 4. D2Q 分位数：只用归一化后的 train（无穿越，与原逻辑一致）
quantile_num = 100
quantiles    = np.linspace(0, 1, quantile_num + 1)
quantile_df  = (df_train_full.groupby('duration_bucket')['play_time']
                .quantile(quantiles).reset_index()
                .rename(columns={'level_1': 'quantile'}))
quantile_pivot = quantile_df.pivot(index='duration_bucket', columns='quantile', values='play_time')
quantile_pivot.to_csv('d2q_duration_bucket_playtime_quantiles.csv')
print("  saved d2q_duration_bucket_playtime_quantiles.csv (train-only)")

data_full = {
    "train":              df_train_full,
    "val":                df_val_full,
    "test":               df_test_full,
    "description":        desc,
    "play_duration_max":  max_val,
    "video_duration_max": video_duration_max,
}
with open('./kuairec_data_full.pkl', 'wb+') as f:
    pickle.dump(data_full, f)
print("saved kuairec_data_full.pkl  train={} val={} test={}".format(len(df_train_full), len(df_val_full), len(df_test_full)))

# ── 10% 采样版：完全独立计算所有统计量（无穿越）────────────────────────────────
# 从 split 前的全量 df 抽取 10%，再做 80/10/10 split，并仅由 10pct train 计算统计量
# 两份 pkl 的 description / play_duration_max / bins 各自独立（允许微小差异）
df_10pct          = df.sample(frac=0.1, random_state=312).reset_index(drop=True)
df_10pct          = df_10pct.sample(frac=1, random_state=1234).reset_index(drop=True)
n_10pct           = len(df_10pct)
n_train_10pct     = int(0.8 * n_10pct)
n_val_10pct       = int(0.9 * n_10pct) - n_train_10pct
df_train_10pct_raw = df_10pct.iloc[:n_train_10pct].copy()
df_val_10pct_raw   = df_10pct.iloc[n_train_10pct:n_train_10pct + n_val_10pct].copy()
df_test_10pct_raw  = df_10pct.iloc[n_train_10pct + n_val_10pct:].copy()

# 10pct max_val 和 video_duration_max：只用 10pct train（修复穿越）
max_val_10pct            = float(df_train_10pct_raw['play_time'].max())
video_duration_max_10pct = float(df_train_10pct_raw['video_duration'].max())
print("play_duration_max (10pct train-only): {}".format(max_val_10pct))

# 10pct duration_bucket：pd.qcut 在 10pct train 上建桶，test 用 pd.cut
df_train_10pct_raw['duration_bucket'], bins_10pct = pd.qcut(
    df_train_10pct_raw['duration'], q=n_bins, labels=False,
    retbins=True, duplicates='drop'
)
df_train_10pct_raw['duration_bucket'] = df_train_10pct_raw['duration_bucket'].astype(int)

extended_bins_10pct        = bins_10pct.copy()
extended_bins_10pct[0]     = -np.inf
extended_bins_10pct[-1]    =  np.inf
df_test_10pct_raw['duration_bucket'] = pd.cut(
    df_test_10pct_raw['duration'], bins=extended_bins_10pct, labels=False, include_lowest=True
).astype(int)
df_val_10pct_raw['duration_bucket'] = pd.cut(
    df_val_10pct_raw['duration'], bins=extended_bins_10pct, labels=False, include_lowest=True
).astype(int)

n_actual_bins_10pct = len(bins_10pct) - 1

# 10pct 的 description 独立，duration_bucket 桶数可能与全量版不同
# 基础 desc（除 duration_bucket 外的字段）与全量版共享 vocab（来自全量 df 编码）
desc_10pct = [item for item in desc if item[0] != 'duration_bucket']
desc_10pct.append(('duration_bucket', n_actual_bins_10pct, 'spr'))
print("  10pct bucket 数: {}（全量版: {}）".format(n_actual_bins_10pct, n_actual_bins))

# 归一化（均用 10pct train max_val）
df_train_10pct = normalize_split(df_train_10pct_raw, max_val_10pct)
df_val_10pct   = normalize_split(df_val_10pct_raw,   max_val_10pct)
df_test_10pct  = normalize_split(df_test_10pct_raw,  max_val_10pct)

# D2Q 分位数：只用 10pct train
quantile_df_10pct = (df_train_10pct.groupby('duration_bucket')['play_time']
                     .quantile(quantiles).reset_index()
                     .rename(columns={'level_1': 'quantile'}))
quantile_pivot_10pct = quantile_df_10pct.pivot(
    index='duration_bucket', columns='quantile', values='play_time'
)
quantile_pivot_10pct.to_csv('d2q_duration_bucket_playtime_quantiles_10pct.csv')
print("  saved d2q_duration_bucket_playtime_quantiles_10pct.csv (10pct train-only)")

data_10pct = {
    "train":              df_train_10pct,
    "val":                df_val_10pct,
    "test":               df_test_10pct,
    "description":        desc_10pct,             # 10pct 独立 description（可能 bucket 数不同）
    "play_duration_max":  max_val_10pct,          # 10pct 独立归一化分母（无穿越）
    "video_duration_max": video_duration_max_10pct,
}
with open('./kuairec_data.pkl', 'wb+') as f:
    pickle.dump(data_10pct, f)
print("saved kuairec_data.pkl       train={} val={} test={}".format(len(df_train_10pct), len(df_val_10pct), len(df_test_10pct)))
