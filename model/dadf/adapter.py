
import torch
import torch.nn.functional as F

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
        self._LOG_MEAN = log_mean

    def stage1_loss(self, features, label):
        raise NotImplementedError

    def get_base_pred(self, features):
        raise NotImplementedError

    def get_base_hidden(self, features):
        return None

    def get_base_pred_and_hidden(self, features):
        p, proxy = self.get_base_pred(features)
        hidden   = self.get_base_hidden(features)
        return p, proxy, hidden

class VRAdapter(BaseAdapter):

    def stage1_loss(self, features, label):
        p = self.model(features)
        return F.mse_loss(p, label.float())

    def get_base_pred(self, features):
        p      = self.model(features).unsqueeze(1)
        p_safe = p.clamp(1e-6, 1.0 - 1e-6)

        return p_safe, torch.clamp(p_safe, min=0.001)

    def get_base_hidden(self, features):
        _, hidden = self.model.forward_with_hidden(features)
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        pred, hidden = self.model.forward_with_hidden(features)
        p = pred.unsqueeze(1).clamp(1e-6, 1.0 - 1e-6)
        return p, torch.clamp(p, min=0.001), hidden.detach()

class WLRAdapter(BaseAdapter):

    def stage1_loss(self, features, label):
        p = self.model(features).clamp(1e-8, 1.0 - 1e-8)
        w = label.float()
        return -(w * torch.log(p) + torch.log(1.0 - p)).mean()

    def get_base_pred(self, features):
        p      = self.model(features).unsqueeze(1)
        p_safe = p.clamp(1e-6, 1.0 - 1e-6)
        proxy  = p_safe / (1.0 - p_safe)
        return p_safe, torch.clamp(proxy, min=0.001)

    def get_base_hidden(self, features):
        _, hidden = self.model.forward_with_hidden(features)
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        pred, hidden = self.model.forward_with_hidden(features)
        p = pred.unsqueeze(1).clamp(1e-6, 1.0 - 1e-6)
        proxy = p / (1.0 - p)
        return p, torch.clamp(proxy, min=0.001), hidden.detach()

WideAndDeepAdapter = WLRAdapter

class EGMNAdapter(BaseAdapter):

    _ALPHA_ENTROPY = 0.1
    _BETA_REG      = 1.0
    _LOG_MEAN      = -4.6

    def stage1_loss(self, features, label):
        pi, lambda_, mu, sigma = self.model(features)
        nll, reg, entropy = self.model.loss(
            label.float(), pi, lambda_, mu, sigma
        )

        return nll - self._ALPHA_ENTROPY * entropy + self._BETA_REG * reg

    def get_base_pred(self, features):
        pi, lambda_, mu, sigma = self.model(features)
        pi_soft = torch.softmax(pi, dim=1)
        mu_all  = torch.cat([1.0 / lambda_, mu], dim=1)
        pred    = (pi_soft * mu_all).sum(dim=1, keepdim=True)

        p_calibrated = torch.sigmoid(
            torch.log(pred.clamp(min=1e-7)) - self._LOG_MEAN
        )

        return p_calibrated, torch.clamp(pred, min=0.001)

    def get_base_hidden(self, features):
        _, _, _, _, hidden = self.model.forward_with_hidden(features)
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
        pi, lambda_, mu, sigma, hidden = self.model.forward_with_hidden(features)
        pi_soft = torch.softmax(pi, dim=1)
        mu_all  = torch.cat([1.0 / lambda_, mu], dim=1)
        pred    = (pi_soft * mu_all).sum(dim=1, keepdim=True)
        p_calibrated = torch.sigmoid(
            torch.log(pred.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(pred, min=0.001), hidden.detach()

    def get_mixture_entropy(self, features):
        pi, _, _, _ = self.model(features)
        pi_soft = torch.softmax(pi, dim=1)
        entropy = -(pi_soft * torch.log(pi_soft + 1e-8)).sum(dim=1, keepdim=True)
        return entropy.detach()

    def get_mixture_confidence(self, features):
        entropy = self.get_mixture_entropy(features)

        return torch.exp(-entropy).clamp(min=0.1, max=1.0).detach()

class TPMAdapter(BaseAdapter):

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
        proxy = encoded_y.clamp(min=1e-6)
        p_calibrated = torch.sigmoid(
            torch.log(proxy.clamp(min=1e-7)) - self._LOG_MEAN
        )
        return p_calibrated, torch.clamp(proxy, min=0.001)

class D2QAdapter(BaseAdapter):

    _LOG_MEAN    = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.buckets_quantiles = None
        self.quantile_max      = 100.0

    def set_quantiles(self, buckets_quantiles, quantile_max=100.0):
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

class CREADAdapter(BaseAdapter):

    _RESTORE_W = 1.0
    _ORD_W     = 0.00002
    _LOG_MEAN  = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.split_nodes = None
        self._bce_criterion   = torch.nn.BCELoss()
        self._huber_criterion = torch.nn.HuberLoss()

    def set_split_nodes(self, split_nodes, device=None):
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

class D2COAdapter(BaseAdapter):

    _LOG_MEAN = -4.6

    def __init__(self, model):
        super().__init__(model)
        self.nega_GMM_mean = None
        self.posi_GMM_mean = None

    def set_gmm(self, nega_GMM_mean, posi_GMM_mean):
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
        _, hidden = self.model.forward_with_hidden(features)
        return hidden.detach()

    def get_base_pred_and_hidden(self, features):
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

_REGISTRY = {
    'vr':    VRAdapter,
    'wlr':   WLRAdapter,
    'wd':    WLRAdapter,
    'egmn':  EGMNAdapter,
    'tpm':   TPMAdapter,
    'd2q':   D2QAdapter,
    'cread': CREADAdapter,
    'd2co':  D2COAdapter,
}

_CANONICAL = ['vr', 'wlr', 'egmn', 'tpm', 'd2q', 'cread', 'd2co']

def build_adapter(base_model_name: str, model) -> BaseAdapter:
    if base_model_name not in _REGISTRY:
        raise ValueError(
            "Unknown base model '{}'. Supported: {}".format(
                base_model_name, _CANONICAL
            )
        )
    return _REGISTRY[base_model_name](model)

def list_supported_models():
    return _CANONICAL
