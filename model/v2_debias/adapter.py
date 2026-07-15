"""
通用纠偏框架 - 基础模型适配层

每个 Adapter 封装一种基础模型，对外统一暴露两个接口：

  stage1_loss(features, label) -> Tensor
      基础模型的 Stage1 训练 loss

  get_base_pred(features) -> (p, proxy)
      p     : [B, 1] ∈ (0, 1)，sigmoid-like，传入 DebiasNetV2 内部做 logit 转换
      proxy : [B, 1] ∈ (0, +∞)，watch_time 代理值
              debias_factor = play_time / proxy

支持的基础模型:

  vr   — WideAndDeep + MSE（Vanilla Regression）
         proxy = sigmoid（MSE 训练使 sigmoid ≈ play_time）
         对应 run_vr.py

  wlr  — WideAndDeep + WBCE（Weighted Logistic Regression）
         proxy = odds = p/(1-p)，最优解处 odds(p*) = play_time
         对应 run_wlr.py
         别名: 'wd'（向后兼容）

  egmn — EGMN + NLL（分布建模）
         proxy = E[X] = Σ π_k * 各分量均值，直接在时长空间
         对应 run_egmn.py

  tpm  — TPM + tree_classify + MSE + variance（树概率建模）
         proxy = E[X] from get_tree_encoded_value，直接在时长空间
         对应 run_tpm.py

  d2q  — D2Q + MSE on quantile space（分位数回归）
         proxy = from_quantile_to_value，还原到时长空间
         对应 run_d2q.py

  cread — Cread + BCE + Huber + ord（有序回归）
          proxy = restore_time_label，还原到时长空间
          对应 run_cread.py

  d2co — WideAndDeep + MSE on GMM label（GMM 分布映射）
         proxy = get_real_value，还原到时长空间
         对应 run_d2co.py

扩展新模型：继承 BaseAdapter，实现 stage1_loss 和 get_base_pred，注册到 _REGISTRY 即可。
"""

import torch
import torch.nn.functional as F


# ====================================================================
# 抽象基类
# ====================================================================

class BaseAdapter:
    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def set_log_mean(self, log_mean: float):
        """
        动态设置 p_calibrated 重心常数（替代硬编码 -4.6）。
        由 train.py 在加载训练数据后调用：log_mean = mean(log(play_time_train))。
        不同数据集的标签均值不同，动态计算避免 p_calibrated 饱和。
        """
        self._LOG_MEAN = log_mean

    def stage1_loss(self, features, label):
        raise NotImplementedError

    def get_base_pred(self, features):
        """
        返回 (p, proxy)：
          p     : [B, 1] ∈ (0,1)，sigmoid-like，传入 Debias 网络作 logit 转换
          proxy : [B, 1] ∈ (0,+∞)，watch_time 代理，debias_factor = play_time / proxy
        调用方需在 torch.no_grad() 内使用。
        """
        raise NotImplementedError

    def get_base_hidden(self, features):
        """
        返回 base model 最后一个 hidden layer 的特征向量（debias_v2 专用上下文）。
        默认返回 None（debias_net 收到 None 时退回纯 logit 变换模式）。
        WideAndDeep 系列（VR/WLR/D2CO）override 此方法返回 64-dim hidden。
        调用方无需 .detach()，各 override 内部已处理。
        """
        return None

    def get_base_pred_and_hidden(self, features):
        """
        合并 get_base_pred + get_base_hidden 为单次前向（减少 BN 重复更新）。
        默认实现：两次调用（非 WD 模型，hidden=None，无额外前向）。
        WD 系列（VR/WLR/D2CO）override 此方法用 forward_with_hidden 只走一次 MLP。
        返回 (p, proxy, hidden)，hidden 可为 None。
        """
        p, proxy = self.get_base_pred(features)
        hidden   = self.get_base_hidden(features)
        return p, proxy, hidden


# ====================================================================
# VR Adapter — WideAndDeep + MSE
# ====================================================================

class VRAdapter(BaseAdapter):
    """
    WideAndDeep + MSE（Vanilla Regression）

    Stage1 loss: MSELoss(sigmoid, play_time)
    MSE 使得 sigmoid ≈ E[play_time|x]
    → proxy = sigmoid，直接作为 watch_time 的点估计
    → 对应 run_vr.py（去掉 20x scale trick，直接在 [0,1] 空间）
    """

    def stage1_loss(self, features, label):
        p = self.model(features)                              # sigmoid [B]
        return F.mse_loss(p, label.float())

    def get_base_pred(self, features):
        p      = self.model(features).unsqueeze(1)            # [B, 1] sigmoid
        p_safe = p.clamp(1e-6, 1.0 - 1e-6)
        # proxy = sigmoid（MSE 已将 sigmoid 校准到 play_time 量纲）
        return p_safe, torch.clamp(p_safe, min=0.001)

    def get_base_hidden(self, features):
        _, hidden = self.model.forward_with_hidden(features)  # [B, 32]
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        """单次 forward_with_hidden，避免 VR 额外前向导致 BN 重复更新。"""
        pred, hidden = self.model.forward_with_hidden(features)
        p = pred.unsqueeze(1).clamp(1e-6, 1.0 - 1e-6)
        return p, torch.clamp(p, min=0.001), hidden.detach()


# ====================================================================
# WLR Adapter — WideAndDeep + WBCE
# ====================================================================

class WLRAdapter(BaseAdapter):
    """
    WideAndDeep + 加权二分类交叉熵（Weighted Logistic Regression）

    Stage1 loss: -play_time * log(p) - log(1-p)
    数学性质：最优解 p* = play_time/(1+play_time)，odds(p*) = play_time
    → proxy = odds = p/(1-p)，最优解处等于 play_time
    → 对应 run_wlr.py
    """

    def stage1_loss(self, features, label):
        p = self.model(features).clamp(1e-8, 1.0 - 1e-8)    # sigmoid [B]
        w = label.float()
        return -(w * torch.log(p) + torch.log(1.0 - p)).mean()

    def get_base_pred(self, features):
        p      = self.model(features).unsqueeze(1)            # [B, 1] sigmoid
        p_safe = p.clamp(1e-6, 1.0 - 1e-6)
        proxy  = p_safe / (1.0 - p_safe)                     # odds ≈ play_time at optimum
        return p_safe, torch.clamp(proxy, min=0.001)

    def get_base_hidden(self, features):
        _, hidden = self.model.forward_with_hidden(features)  # [B, 32]
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        """单次 forward_with_hidden，避免 WLR 额外前向导致 BN 重复更新。"""
        pred, hidden = self.model.forward_with_hidden(features)
        p = pred.unsqueeze(1).clamp(1e-6, 1.0 - 1e-6)
        proxy = p / (1.0 - p)                                 # odds ≈ play_time at optimum
        return p, torch.clamp(proxy, min=0.001), hidden.detach()


# WideAndDeepAdapter 保留为 WLRAdapter 的别名（向后兼容）
WideAndDeepAdapter = WLRAdapter


# ====================================================================
# EGMN Adapter — EGMN + NLL
# ====================================================================

class EGMNAdapter(BaseAdapter):
    """
    EGMN（指数 + 高斯混合分布）+ NLL 训练

    Stage1 loss: NLL + 0.1*entropy + 1.0*reg（对应 run_egmn.py 默认超参）
    → proxy = E[X] = Σ π_k * 各分量均值，直接在时长空间，无需变换
    → 对应 run_egmn.py
    """

    _ALPHA_ENTROPY = 0.1
    _BETA_REG      = 1.0
    _LOG_MEAN      = -4.6   # re-centering constant; overridden by set_log_mean()

    def stage1_loss(self, features, label):
        pi, lambda_, mu, sigma = self.model(features)
        nll, reg, entropy = self.model.loss(
            label.float(), pi, lambda_, mu, sigma
        )
        # 对应 run_egmn.py: loss = nll_loss - alpha * entropy_loss + beta * reg_loss
        # entropy_loss = H（正熵），最大化熵（-） 防止混合分量坍缩
        return nll - self._ALPHA_ENTROPY * entropy + self._BETA_REG * reg

    def get_base_pred(self, features):
        pi, lambda_, mu, sigma = self.model(features)
        pi_soft = torch.softmax(pi, dim=1)
        mu_all  = torch.cat([1.0 / lambda_, mu], dim=1)
        pred    = (pi_soft * mu_all).sum(dim=1, keepdim=True)   # E[X] [B, 1]

        # Map E[X] → calibrated probability for DebiasNet input.
        # Re-center using log(E[X]/mean) so the network sees a zero-centered
        # logit with natural spread ~1-2 units.
        # _LOG_MEAN is set dynamically per dataset via set_log_mean() (default -4.6 for KuaiRec)
        p_calibrated = torch.sigmoid(
            torch.log(pred.clamp(min=1e-7)) - self._LOG_MEAN
        )  # ≈0.5 at mean, spread over (0,1) with natural variance

        return p_calibrated, torch.clamp(pred, min=0.001)

    def get_base_hidden(self, features):
        _, _, _, _, hidden = self.model.forward_with_hidden(features)
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        """Single forward pass returning (p, proxy, hidden) to avoid duplicate BN updates."""
        pi, lambda_, mu, sigma, hidden = self.model.forward_with_hidden(features)
        pi_soft = torch.softmax(pi, dim=1)
        mu_all  = torch.cat([1.0 / lambda_, mu], dim=1)
        pred    = (pi_soft * mu_all).sum(dim=1, keepdim=True)
        p_calibrated = torch.sigmoid(
            torch.log(pred.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(pred, min=0.001), hidden.detach()

    def get_mixture_entropy(self, features):
        """
        EGMN 专用：返回混合模型分布熵 [B, 1]。
        熵高 = 模型不确定，debias 应保守；熵低 = 模型确信，debias 可大胆。
        """
        pi, _, _, _ = self.model(features)
        pi_soft = torch.softmax(pi, dim=1)
        entropy = -(pi_soft * torch.log(pi_soft + 1e-8)).sum(dim=1, keepdim=True)
        return entropy.detach()

    def get_mixture_confidence(self, features):
        """
        EGMN 专用：返回置信度 [B, 1] ∈ (0, 1]。
        confidence = exp(-entropy)，熵越低，置信度越高。
        用于置信度加权 BC 损失：不确定样本的纠偏目标更嘈杂，应降低其权重。
        """
        entropy = self.get_mixture_entropy(features)
        # exp(-entropy): entropy=0→1.0, entropy=log(K)→1/K; clamp prevents overaggression
        return torch.exp(-entropy).clamp(min=0.1, max=1.0).detach()


# ====================================================================
# TPM Adapter — TPM + tree_classify + MSE + variance
# ====================================================================

class TPMAdapter(BaseAdapter):
    """
    TPM（树概率建模）+ tree_classify_loss + MSE + variance

    Stage1 loss: tree_classify_loss * 1.0 + MSE(encoded_y, label) * 1.0 + variance * 0.0001
    → proxy = encoded_y from get_tree_encoded_value，E[X] 在时长空间
    → 对应 run_tpm.py 默认超参 tree_cla_weight=1, mse_weight=1, variance_weight=0.0001
    """

    _TREE_CLA_WEIGHT  = 1.0
    _MSE_WEIGHT       = 1.0
    _VARIANCE_WEIGHT  = 0.0001
    _LOG_MEAN         = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.wr_bucknum    = 32
        self.bucket_begins = None
        self.bucket_ends   = None

    def set_buckets(self, bucket_begins, bucket_ends, wr_bucknum=32):
        """设置树分桶辅助数据（来自 get_playtime_percentiles_range）"""
        self.bucket_begins = bucket_begins
        self.bucket_ends   = bucket_ends
        self.wr_bucknum    = wr_bucknum

    def stage1_loss(self, features, label):
        from utils import get_tree_encoded_value, get_tree_encoded_label, get_tree_classify_loss
        y = self.model(features)
        encoded_y, variance = get_tree_encoded_value(
            y, self.wr_bucknum, self.bucket_begins, self.bucket_ends
        )
        encoded_label, bucket_weights = get_tree_encoded_label(
            label, self.wr_bucknum, self.bucket_begins, self.bucket_ends
        )
        tree_classify_loss = get_tree_classify_loss(
            encoded_label, bucket_weights, y, self.wr_bucknum
        )
        mse_loss = F.mse_loss(encoded_y, label.view(-1, 1).float())
        return (
            tree_classify_loss * self._TREE_CLA_WEIGHT
            + mse_loss * self._MSE_WEIGHT
            + variance * self._VARIANCE_WEIGHT
        )

    def get_base_pred(self, features):
        from utils import get_tree_encoded_value
        y = self.model(features)
        encoded_y, _ = get_tree_encoded_value(
            y, self.wr_bucknum, self.bucket_begins, self.bucket_ends
        )
        proxy = encoded_y.clamp(min=1e-6)                         # [B, 1]
        p_calibrated = torch.sigmoid(
            torch.log(proxy.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(proxy, min=0.001)


# ====================================================================
# D2Q Adapter — D2Q + MSE on quantile space
# ====================================================================

class D2QAdapter(BaseAdapter):
    """
    D2Q（分位数回归）+ MSE on quantile space

    Stage1 loss: MSE(y, label_norm(mapped_quantile_label, quantile_max))
                 Python loop over batch per run_d2q.py
    → proxy = from_quantile_to_value per sample，还原到时长空间
    → 对应 run_d2q.py 默认超参 quantile_max=100
    """

    _LOG_MEAN    = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.buckets_quantiles = None
        self.quantile_max      = 100.0

    def set_quantiles(self, buckets_quantiles, quantile_max=100.0):
        """设置分桶分位数辅助数据（来自 get_buckets_infor）"""
        self.buckets_quantiles = buckets_quantiles
        self.quantile_max      = quantile_max

    def stage1_loss(self, features, label):
        from model.framework_utils import from_value_to_quantile, label_norm
        y            = self.model(features)
        bucket_index = features['duration_bucket']
        device       = y.device
        mapped_label = []
        for idx, label_origin in zip(bucket_index, label.tolist()):
            mapped_label_value = from_value_to_quantile(
                self.buckets_quantiles, idx.item(), label_origin
            )
            mapped_label.append(mapped_label_value)
        mapped_label = torch.tensor(mapped_label, device=device, dtype=torch.float32)
        mapped_label = label_norm(mapped_label, self.quantile_max)
        return F.mse_loss(y, mapped_label)

    def get_base_pred(self, features):
        from model.framework_utils import from_quantile_to_value
        y            = self.model(features)
        bucket_index = features['duration_bucket']
        mapped_y     = []
        for idx, quantile in zip(bucket_index, y.tolist()):
            mapped_y_value = from_quantile_to_value(
                self.buckets_quantiles, idx.item(), quantile
            )
            mapped_y.append(mapped_y_value)
        proxy = torch.tensor(mapped_y, device=y.device, dtype=torch.float32).view(-1, 1).clamp(min=1e-6)
        p_calibrated = torch.sigmoid(
            torch.log(proxy.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(proxy, min=0.001)


# ====================================================================
# CREAD Adapter — Cread + BCE + Huber + ord
# ====================================================================

class CREADAdapter(BaseAdapter):
    """
    Cread（有序回归）+ BCE + Huber + ord

    Stage1 loss: BCE(preds, binary_labels) + Huber(restore_pred, label) + 0.00002 * ord_loss
                 对应 run_cread.py 默认 restore_w=1.0, ord_w=0.00002
    → proxy = restore_time_label(preds, split_nodes)，还原到时长空间
    → 对应 run_cread.py
    """

    _RESTORE_W = 1.0
    _ORD_W     = 0.00002
    _LOG_MEAN  = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.split_nodes = None
        self._bce_criterion   = torch.nn.BCELoss()
        self._huber_criterion = torch.nn.HuberLoss()

    def set_split_nodes(self, split_nodes, device=None):
        """设置有序分割点（来自 cread_grid_search），可选 device 转移"""
        if device is not None:
            self.split_nodes = split_nodes.to(device)
        else:
            self.split_nodes = split_nodes

    def to(self, device):
        super().to(device)
        if self.split_nodes is not None:
            self.split_nodes = self.split_nodes.to(device)
        return self

    def stage1_loss(self, features, label):
        from model.framework_utils import discretize_time_label, restore_time_label, get_ord_criterion
        preds         = self.model(features)
        binary_labels = discretize_time_label(label, self.split_nodes)
        restore_pred  = restore_time_label(preds, self.split_nodes)
        loss_bce      = self._bce_criterion(preds, binary_labels.float())
        loss_restore  = self._huber_criterion(restore_pred, label.float())
        loss_ord      = get_ord_criterion(preds)
        return loss_bce + self._RESTORE_W * loss_restore + self._ORD_W * loss_ord

    def get_base_pred(self, features):
        from model.framework_utils import restore_time_label
        preds = self.model(features)
        proxy = restore_time_label(preds, self.split_nodes).view(-1, 1).clamp(min=1e-6)
        p_calibrated = torch.sigmoid(
            torch.log(proxy.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(proxy, min=0.001)


# ====================================================================
# D2CO Adapter — WideAndDeep + MSE on GMM label
# ====================================================================

class D2COAdapter(BaseAdapter):
    """
    WideAndDeep + MSE on GMM label（GMM 分布映射）

    Stage1 loss: MSE(y, mapped_gmm_label)
                 Python loop over batch per run_d2co.py
    → proxy = get_real_value per sample，还原到时长空间
    → 对应 run_d2co.py
    """

    _LOG_MEAN = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.nega_GMM_mean = None
        self.posi_GMM_mean = None

    def set_gmm(self, nega_GMM_mean, posi_GMM_mean):
        """设置 GMM 均值字典（来自 get_gmm_mean）"""
        self.nega_GMM_mean = nega_GMM_mean
        self.posi_GMM_mean = posi_GMM_mean

    def stage1_loss(self, features, label):
        from model.framework_utils import get_gmm_label
        y            = self.model(features)
        bucket_index = features['duration_bucket']
        device       = y.device
        mapped_label = []
        for idx, label_origin in zip(bucket_index, label.tolist()):
            mapped_label_value = get_gmm_label(
                label_origin, idx.item(), self.nega_GMM_mean, self.posi_GMM_mean
            )
            mapped_label.append(mapped_label_value)
        mapped_label = torch.tensor(mapped_label, device=device, dtype=torch.float32)
        return F.mse_loss(y, mapped_label)

    def get_base_pred(self, features):
        from model.framework_utils import get_real_value
        y            = self.model(features)
        bucket_index = features['duration_bucket']
        mapped_y     = []
        for idx, gmm_y in zip(bucket_index, y.tolist()):
            mapped_y_value = get_real_value(
                gmm_y, idx.item(), self.nega_GMM_mean, self.posi_GMM_mean
            )
            mapped_y.append(mapped_y_value)
        proxy = torch.tensor(mapped_y, device=y.device, dtype=torch.float32).view(-1, 1).clamp(min=1e-6)
        p_calibrated = torch.sigmoid(
            torch.log(proxy.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(proxy, min=0.001)

    def get_base_hidden(self, features):
        _, hidden = self.model.forward_with_hidden(features)  # [B, 32]
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        """单次 forward_with_hidden，D2CO GMM 映射逻辑与 get_base_pred 一致。"""
        from model.framework_utils import get_real_value
        y, hidden    = self.model.forward_with_hidden(features)
        bucket_index = features['duration_bucket']
        mapped_y     = []
        for idx, gmm_y in zip(bucket_index, y.tolist()):
            mapped_y.append(get_real_value(
                gmm_y, idx.item(), self.nega_GMM_mean, self.posi_GMM_mean
            ))
        proxy = torch.tensor(mapped_y, device=y.device, dtype=torch.float32).view(-1, 1).clamp(min=1e-6)
        p_calibrated = torch.sigmoid(torch.log(proxy.clamp(min=1e-7)) - self._LOG_MEAN)
        return p_calibrated, torch.clamp(proxy, min=0.001), hidden.detach()


# ====================================================================
# 工厂函数
# ====================================================================

_REGISTRY = {
    'vr':    VRAdapter,
    'wlr':   WLRAdapter,
    'wd':    WLRAdapter,    # 向后兼容别名
    'egmn':  EGMNAdapter,
    'tpm':   TPMAdapter,
    'd2q':   D2QAdapter,
    'cread': CREADAdapter,
    'd2co':  D2COAdapter,
}

# 对外展示的规范名（不含别名）
_CANONICAL = ['vr', 'wlr', 'egmn', 'tpm', 'd2q', 'cread', 'd2co']


def build_adapter(base_model_name: str, model) -> BaseAdapter:
    """工厂函数：根据名称创建 Adapter"""
    if base_model_name not in _REGISTRY:
        raise ValueError(
            "Unknown base model '{}'. Supported: {}".format(
                base_model_name, _CANONICAL
            )
        )
    return _REGISTRY[base_model_name](model)


def list_supported_models():
    """返回规范模型名称列表（用于 argparse choices）"""
    return _CANONICAL
