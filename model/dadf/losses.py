
import torch
from .transforms import duration_to_onehot

def normal_regularization_loss(x_trans, duration, thresholds, weight_valid,
                               bucket_freq=None, kurtosis_weight=0.0, eps=1e-8):
    d = duration.view(-1)

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

    if bucket_freq is not None:
        bw = (bucket_freq.clamp(min=1e-6)).sqrt()
        bw = bw / bw.sum()
    else:
        n = float(len(thresholds) + 1)
        bw = [1.0 / n] * (len(thresholds) + 1)

    reg_loss = torch.tensor(0.0, device=x_trans.device, dtype=torch.float32)

    for i, mask in enumerate(bucket_masks):
        group_mask = mask & valid
        x_group    = x_flat[group_mask]
        if x_group.shape[0] < 4:
            continue

        mean     = x_group.mean()
        var      = x_group.var(unbiased=False).clamp(min=eps)
        std      = var.sqrt()
        centered = x_group - mean
        skewness = centered.pow(3).mean() / (std.pow(3) + eps)
        bucket_nr = (
            1.0 * mean.pow(2)
            + 0.8 * (var - 1.0).pow(2)
            + 0.6 * skewness.abs()
        )
        if kurtosis_weight > 0.0:
            kurtosis = centered.pow(4).mean() / (var.pow(2) + eps)
            bucket_nr = bucket_nr + kurtosis_weight * (kurtosis - 3.0).abs()
        w = bw[i] if bucket_freq is None else bw[i]
        reg_loss = reg_loss + w * bucket_nr

    return reg_loss

def weighted_huber_loss(pred, target, weight, delta=0.2):
    diff  = torch.abs(pred - target)
    huber = torch.where(
        diff < delta,
        0.5 * diff.pow(2),
        delta * (diff - 0.5 * delta),
    )
    denom = weight.sum().clamp(min=1.0)
    return (huber * weight).sum() / denom
