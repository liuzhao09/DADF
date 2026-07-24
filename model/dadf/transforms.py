
import torch

def boxcox_transform(x, lambda_tensor, eps=1e-8):
    x_safe     = torch.clamp(x, min=eps)
    is_nonzero = lambda_tensor.abs() > eps

    lam_safe   = torch.where(is_nonzero, lambda_tensor, torch.ones_like(lambda_tensor))
    x_pow      = torch.pow(x_safe, lam_safe)
    bc_nz      = (x_pow - 1.0) / lam_safe
    bc_z       = torch.log(x_safe)

    return torch.where(is_nonzero, bc_nz, bc_z)

def boxcox_inverse(y, lambda_tensor, eps=1e-8):
    is_nonzero = lambda_tensor.abs() > eps

    lam_safe   = torch.where(is_nonzero, lambda_tensor, torch.ones_like(lambda_tensor))
    inner      = torch.clamp(lam_safe * y + 1.0, min=eps)
    inv_lam    = 1.0 / lam_safe
    inv_nz     = torch.pow(inner, inv_lam)
    inv_z      = torch.exp(y)

    result = torch.where(is_nonzero, inv_nz, inv_z)
    return torch.clamp(result, min=0.1)

def duration_to_onehot(duration, thresholds):
    masks = []

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
    return torch.cat(masks, dim=1)
