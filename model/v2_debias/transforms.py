"""
Box-Cox 变换工具函数
参照 online_code/tf_graph.py: debias_factor_trans / debias_factor_reverse / duration_bucket_to_onehot

设计要点：
  - 使用 lam_safe 防止 torch.where 幽灵梯度（λ≈0 时梯度量级可达 1e8）
  - 分母不加 eps（is_nonzero 已保护 λ≠0 分支）
  - boxcox_inverse 输出 clamp 到 [0.1, +∞)
"""

import torch


def boxcox_transform(x, lambda_tensor, eps=1e-8):
    """
    正向 Box-Cox 变换
      λ≠0: (x^λ - 1) / λ
      λ=0: log(x)
    x 和 lambda_tensor 形状须可广播，通常均为 [B, 1]
    """
    x_safe     = torch.clamp(x, min=eps)
    is_nonzero = lambda_tensor.abs() > eps

    lam_safe   = torch.where(is_nonzero, lambda_tensor, torch.ones_like(lambda_tensor))
    x_pow      = torch.pow(x_safe, lam_safe)
    bc_nz      = (x_pow - 1.0) / lam_safe
    bc_z       = torch.log(x_safe)

    return torch.where(is_nonzero, bc_nz, bc_z)


def boxcox_inverse(y, lambda_tensor, eps=1e-8):
    """
    逆 Box-Cox 变换
      λ≠0: (λy + 1)^(1/λ)
      λ=0: exp(y)
    输出 clamp 到 [0.1, +∞)
    """
    is_nonzero = lambda_tensor.abs() > eps

    lam_safe   = torch.where(is_nonzero, lambda_tensor, torch.ones_like(lambda_tensor))
    inner      = torch.clamp(lam_safe * y + 1.0, min=eps)
    inv_lam    = 1.0 / lam_safe
    inv_nz     = torch.pow(inner, inv_lam)
    inv_z      = torch.exp(y)

    result = torch.where(is_nonzero, inv_nz, inv_z)
    return torch.clamp(result, min=0.1)


def duration_to_onehot(duration, thresholds):
    """
    将归一化时长 [B, 1] 映射为 N 维 one-hot [B, N]，N = len(thresholds) + 1
    支持 3 或 4 个专家（通过传入 2 或 3 个 threshold）。
    参照 tf_graph.py: duration_bucket_to_onehot

    thresholds: [t0, ..., t_{N-2}]，将 [0, 1] 区间分成 N 段
      bucket 0: [0,        t0)
      bucket k: [t_{k-1},  t_k)   for k in 1..N-2
      bucket N-1: [t_{N-2}, ∞)
    """
    masks = []
    # K=1 edge case: no thresholds -> single global bucket (all samples in bucket 0)
    if len(thresholds) == 0:
        return torch.ones_like(duration, dtype=torch.float32)
    prev = None
    for t in thresholds:
        if prev is None:
            masks.append((duration < t).float())
        else:
            masks.append(((duration >= prev) & (duration < t)).float())
        prev = t
    masks.append((duration >= prev).float())
    return torch.cat(masks, dim=1)   # [B, N]
