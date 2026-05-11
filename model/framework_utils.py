"""
框架共用工具函数（TranSUN 和 V2 Debias 框架共享）
"""

import sys, os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataloader import KUAIRECDataLoader


def mae_rescale_to_second(dataset, mae, play_duration_max=None):
    if dataset == 'wechat21':
        # play_time 单位为 ms，归一化分母约 1200000ms=1200s
        scale = play_duration_max if play_duration_max is not None else 1200000.0
        return mae * scale / 1000.0
    else:
        scale = play_duration_max if play_duration_max is not None else 1000.0
        return mae * scale / 1000.0


def get_loaders(name, dataset_path, device, bsz, full_data=False, pkl_suffix=None):
    """pkl_suffix: 非空时覆盖默认 suffix 逻辑（如 '_bins30' 加载 kuairec_data_bins30.pkl）"""
    if pkl_suffix is not None and pkl_suffix != '':
        suffix = pkl_suffix
    else:
        suffix = '_full' if full_data else ''
    path = _os.path.join(dataset_path, name, '{}_data{}.pkl'.format(name, suffix))
    if name == 'kuairec':
        return KUAIRECDataLoader(name, path, device, bsz=bsz)
    elif name == 'wechat21':
        from dataloader.wechat21 import WeChat21DataLoader
        return WeChat21DataLoader(name, path, device, bsz=bsz)
    else:
        raise ValueError('Unknown dataset: {}'.format(name))


def get_vocab_size(description, name):
    for n, size, type_ in description:
        if n == name:
            return size
    raise ValueError("Feature '{}' not found in description".format(name))


# ---------------------------------------------------------------------------
# D2Q utilities
# ---------------------------------------------------------------------------

import torch as _torch
import numpy as _np
import pandas as _pd


def label_norm(label, max_value):
    return _torch.clamp(label, max=max_value) / max_value


def from_value_to_quantile(bucket_quantiles, bucket_index, value):
    quantile = bucket_quantiles[bucket_index]
    if len(quantile) < 2:
        return 0.0
    if value <= quantile[0]:
        return 0.0
    if value >= quantile[-1]:
        return len(quantile) - 1
    idx = _np.searchsorted(quantile, value) - 1
    left_val = quantile[idx]
    right_val = quantile[idx + 1]
    if right_val == left_val:
        return float(idx)
    fraction = (value - left_val) / (right_val - left_val)
    quantile = idx + fraction
    return quantile


def from_quantile_to_value(bucket_quantiles, bucket_index, quantile):
    quantiles = bucket_quantiles[bucket_index]
    num_quantiles = len(quantiles)
    quantile_steps = _np.linspace(0.0, 1.0, num_quantiles)
    if quantile <= 0.0:
        return quantiles[0]
    elif quantile >= 1.0:
        return quantiles[-1]
    idx = _np.searchsorted(quantile_steps, quantile) - 1
    if quantile_steps[idx + 1] == quantile:
        return quantiles[idx + 1]
    q1, q2 = quantile_steps[idx], quantile_steps[idx + 1]
    v1, v2 = quantiles[idx], quantiles[idx + 1]
    interpolated_value = v1 + (v2 - v1) * (quantile - q1) / (q2 - q1)
    return interpolated_value


def get_buckets_infor(buckets_quantiles_path):
    df = _pd.read_csv(buckets_quantiles_path)
    bucket_quantiles = {}
    for idx, row in df.iterrows():
        bucket_index = int(row[0])
        quantiles = row[1:].values.tolist()
        bucket_quantiles[bucket_index] = quantiles
    return bucket_quantiles


# ---------------------------------------------------------------------------
# CREAD utilities
# ---------------------------------------------------------------------------


def discretize_time_label(playtime, split_nodes):
    playtime = playtime.reshape([-1, 1, 1])
    split_nodes = split_nodes.reshape([1, 1, -1])
    cmp_tensor = playtime > split_nodes
    binary_labels = _torch.where(cmp_tensor, _torch.ones_like(cmp_tensor), _torch.zeros_like(cmp_tensor))  # [bsz, 1, M]
    return binary_labels.squeeze(1)  # [bsz, M] — squeeze only the middle dim, not batch dim


def restore_time_label(preds, split_nodes):
    append_split_nodes = _torch.concat((_torch.tensor([0]).to(_torch.float32).to(split_nodes.device), split_nodes))
    left_split_nodes, right_split_nodes = append_split_nodes[:-1], append_split_nodes[1:]
    bkt_size_list = right_split_nodes - left_split_nodes
    return _torch.sum(preds * bkt_size_list.view([1, -1]), dim=1)  # [bsz]


def get_ord_criterion(preds):
    left_preds, right_preds = preds[:, :-1], preds[:, 1:]
    return _torch.sum(_torch.clamp(right_preds - left_preds, min=0.0))


def get_split_nodes(all_labels, M, alpha):
    split_nodes = []
    cdf_list = []
    for m in range(1, M + 1):
        z = m / M
        gamma = (1 - _np.exp(-alpha * z)) / (1 - _np.exp(-alpha))
        split_nodes.append(_torch.quantile(all_labels, gamma))
        cdf_list.append(gamma)
    return _torch.tensor(split_nodes), _torch.tensor(cdf_list)


def cread_grid_search(dataloader_train, M):
    all_labels = []
    for (_, label) in dataloader_train:
        all_labels.append(label)
    all_labels = _torch.concat(all_labels).to(_torch.float32)
    alpha_search_space = list(_np.arange(0.001, 5.0, 0.1))
    beta_search_space = [50]
    best_loss, best_alpha, best_beta, best_split = None, None, None, None
    print("Strat Cread Split Nodes Search....l")
    for alpha in alpha_search_space:
        for beta in beta_search_space:
            split_nodes, cdf_list = get_split_nodes(all_labels, M, alpha)
            split_nodes_left, split_nodes_right = _torch.cat([_torch.tensor([0]), split_nodes[:-1]]), split_nodes
            cdf_list_left, cdf_list_right = _torch.cat([_torch.tensor([0]), cdf_list[:-1]]), cdf_list
            A_w = _torch.sum(_torch.pow(cdf_list_right - cdf_list_left, 2)) * _torch.sum(_torch.pow(split_nodes_right - split_nodes_left, 2) / (cdf_list_right - cdf_list_left))
            A_b = _torch.sum(_torch.pow(cdf_list_right - cdf_list_left, 2)) * _torch.sum(_torch.pow(split_nodes_right - split_nodes_left, 2))
            A_loss = A_w + beta * A_b
            print("Searching | alpha={:.7f}, beta={:.7f}: A_loss={:.7f},A_w={:.7f},A_b={:.7f} ".format(alpha, beta, A_loss, A_w, A_b))
            if (best_loss is None) or (best_loss > A_loss):
                best_loss, best_alpha, best_beta, best_split = A_loss, alpha, beta, split_nodes
    print("Cread Search Complete! Best Loss is {:.7f}, Best Alpha is {:.7f}, Best Beta is {:.7f}.".format(best_loss, best_alpha, best_beta))
    return best_split


# ---------------------------------------------------------------------------
# D2CO utilities
# ---------------------------------------------------------------------------


def get_gmm_label(label, idx, nega_GMM_mean, posi_GMM_mean, alpha=1.0):
    p = nega_GMM_mean[idx]
    q = posi_GMM_mean[idx]
    denom = _np.exp(alpha * p) - _np.exp(alpha * q)
    if abs(denom) < 1e-9:  # p ≈ q (degenerate bucket): return midpoint
        return 0.5
    gmm_label = (_np.exp(alpha * label) - _np.exp(alpha * q)) / denom
    return _np.clip(gmm_label, 0, 1)


def get_real_value(y, idx, nega_GMM_mean, posi_GMM_mean, alpha=1.0):
    p = nega_GMM_mean[idx]
    q = posi_GMM_mean[idx]
    denom = _np.exp(alpha * p) - _np.exp(alpha * q)
    if abs(denom) < 1e-9:  # p ≈ q (degenerate bucket): return midpoint label
        return q
    real_y = _np.log(y * denom + _np.exp(alpha * q)) / alpha
    return real_y


def get_gmm_mean(df_train, n_bins=None):
    from sklearn.mixture import GaussianMixture
    from collections import Counter
    gmmMeanList = []
    durationBucketList = []
    playTimeList = []
    for _, (features, label) in enumerate(df_train):
        durationBucketList.extend(features['duration_bucket'].cpu().numpy().flatten())
        playTimeList.extend(label.cpu().numpy().flatten())
    durationBucketList = _np.array(durationBucketList, dtype=int)
    playTimeList = _np.array(playTimeList)

    actual_buckets = sorted(_np.unique(durationBucketList).astype(int))
    for d in actual_buckets:
        playTimeInBucket = playTimeList[durationBucketList == d].reshape(-1, 1)
        if len(playTimeInBucket) == 0:
            continue
        elif len(playTimeInBucket) < 2:
            single_val = float(playTimeInBucket[0, 0])
            means = _np.array([single_val, single_val])
        else:
            gm = GaussianMixture(
                n_components=2, init_params='kmeans',
                covariance_type='spherical', max_iter=500, random_state=61
            ).fit(playTimeInBucket)
            means = _np.sort(gm.means_.T[0])
        gmmMeanList.append([d, means[0], means[1]])

    processed_bucket_set = {int(row[0]) for row in gmmMeanList}
    numInEachBucket = Counter(durationBucketList)
    numInEachBucket = [numInEachBucket[d] for d in actual_buckets if d in processed_bucket_set]

    def freq_moving_ave(ls_v, ls_w, windows_size=5):
        ls_mul = _np.array(ls_v) * _np.array(ls_w)
        amount = _pd.Series(ls_mul)
        amount_sum = amount.rolling(2 * windows_size - 1, min_periods=1, center=True).agg(lambda x: _np.sum(x))
        weight = _pd.Series(ls_w)
        weight_sum = weight.rolling(2 * windows_size - 1, min_periods=1, center=True).agg(lambda x: _np.sum(x))
        return amount_sum / weight_sum

    gmmMeanList = _np.array(gmmMeanList)
    nega_GMM_mean = dict(zip(gmmMeanList[:, 0], freq_moving_ave(gmmMeanList[:, 1], numInEachBucket, windows_size=5)))
    posi_GMM_mean = dict(zip(gmmMeanList[:, 0], freq_moving_ave(gmmMeanList[:, 2], numInEachBucket, windows_size=5)))
    return nega_GMM_mean, posi_GMM_mean
