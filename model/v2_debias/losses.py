"""
V2 纠偏框架 Loss 函数
参照 online_code/tf_graph.py: normal_regularization_loss / v2_abs_predtime_loss
"""

import torch
from .transforms import duration_to_onehot


def normal_regularization_loss(x_trans, duration, thresholds, weight_valid,
                               bucket_freq=None, eps=1e-8):
    """
    正态化正则 loss：强制各 bucket 内 Box-Cox 变换后的值分布
    均值 → 0，方差 → 1，偏度 → 0，峰度 → 3（正态峰度）
    参照 tf_graph.py: normal_regularization_loss，增加峰度项

    各项权重（与 tf_graph.py 一致）：均值 1.0，方差 0.8，偏度 0.6
    增加：超额峰度 0.3（约束 excess_kurtosis → 0，即 kurtosis → 3）
    分母动态 = len(thresholds) + 1（支持 3 或 4 个时长 bucket）
    var 使用 unbiased=False（匹配 TF reduce_variance / N）

    bucket_freq（可选）：每 bucket 的样本频率 [bucket_num]。
      提供时按 sqrt(freq) 加权各 bucket NR 贡献（样本多的桶信号更可靠）。
      不提供时退回均匀 1/N 权重（向后兼容）。
    """
    d = duration.view(-1)
    # 动态生成 bucket_masks，支持任意 len(thresholds) = N-1
    # K=1 edge case: no thresholds -> single global bucket (all-True mask)
    if len(thresholds) == 0:
        bucket_masks = [torch.ones_like(d, dtype=torch.bool)]
    else:
        bucket_masks = []
        prev = None
        for t in thresholds:
            if prev is None:
                bucket_masks.append(d < t)
            else:
                bucket_masks.append((d >= prev) & (d < t))
            prev = t
        bucket_masks.append(d >= prev)

    x_flat = x_trans.view(-1)
    valid   = weight_valid.view(-1).bool()

    # 预计算 bucket 权重：sqrt(freq) 归一化，样本多的桶权重更大（信号更可靠）
    if bucket_freq is not None:
        bw = (bucket_freq.clamp(min=1e-6)).sqrt()
        bw = bw / bw.sum()  # 归一化使权重和=1
    else:
        n = float(len(thresholds) + 1)
        bw = [1.0 / n] * (len(thresholds) + 1)

    reg_loss = torch.tensor(0.0, device=x_trans.device, dtype=torch.float32)

    for i, mask in enumerate(bucket_masks):
        group_mask = mask & valid
        x_group    = x_flat[group_mask]
        if x_group.shape[0] < 4:  # 需要至少 4 个样本计算峰度
            continue

        mean     = x_group.mean()
        var      = x_group.var(unbiased=False).clamp(min=eps)
        std      = var.sqrt()
        centered = x_group - mean
        skewness = centered.pow(3).mean() / (std.pow(3) + eps)
        # 超额峰度：N(0,1) 的峰度为 3，excess_kurtosis → 0
        kurtosis      = centered.pow(4).mean() / (var.pow(2) + eps)
        excess_kurt   = kurtosis - 3.0

        bucket_nr = (
            1.0 * mean.pow(2)
            + 0.8 * (var - 1.0).pow(2)
            + 0.6 * skewness.abs()
            + 0.3 * excess_kurt.abs()   # 新增峰度约束
        )
        w = bw[i] if bucket_freq is None else bw[i]
        reg_loss = reg_loss + w * bucket_nr

    return reg_loss


def weighted_huber_loss(pred, target, weight, delta=0.2):
    """
    带样本权重的 Huber loss（参照 tf_graph.py: v2_abs_predtime_loss）

    与线上实现的差异（有意）：
      - 线上：delta=20 在 [0, scale_watch_time≈100s] 空间；
              由于 delta=20 > max_value≈10，所有样本都在二次区 → 退化为纯 MSE
      - 离线：delta=0.2 在 [0,1] 归一化空间；
              |diff| > 0.2 的样本切换到线性区（L1 保护），引入真实 Huber 行为
    delta=0.2 不是线上的等价值，而是主动引入大误差 L1 保护的设计选择。
    对两个数据集一致（均归一化到 [0,1]），delta=0.2 均代表 "20% range 内二次，超出线性"。
    """
    diff  = torch.abs(pred - target)
    huber = torch.where(
        diff < delta,
        0.5 * diff.pow(2),
        delta * (diff - 0.5 * delta),
    )
    denom = weight.sum().clamp(min=1.0)
    return (huber * weight).sum() / denom
