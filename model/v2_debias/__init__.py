"""model.v2_debias — 通用 V2 纠偏框架"""

from .transforms import boxcox_transform, boxcox_inverse, duration_to_onehot
from .losses     import normal_regularization_loss, weighted_huber_loss
from .network    import DebiasNetV2
from .adapter    import (
    BaseAdapter,
    VRAdapter,
    WLRAdapter,
    WideAndDeepAdapter,   # alias for WLRAdapter
    EGMNAdapter,
    build_adapter,
    list_supported_models,
)

__all__ = [
    'boxcox_transform', 'boxcox_inverse', 'duration_to_onehot',
    'normal_regularization_loss', 'weighted_huber_loss',
    'DebiasNetV2',
    'BaseAdapter', 'VRAdapter', 'WLRAdapter', 'WideAndDeepAdapter', 'EGMNAdapter',
    'build_adapter', 'list_supported_models',
]
