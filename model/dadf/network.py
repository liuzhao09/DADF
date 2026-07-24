
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import duration_to_onehot

class DADF(nn.Module):

    def __init__(self, user_vocab_size, video_vocab_size,
                 embed_dim=16, bucket_num=4, hidden_dim=64,
                 lambda_init=None, hidden_dim_base=0, shared_correction=False,
                 use_bucket_emb=True, bucket_mean_proxy=None,
                 shared_user_emb=None, shared_video_emb=None,
                 use_uncertainty=False,
                 use_uv_interaction=False, debias_dropout=0.0,
                 aux_target_names=()):
        super().__init__()
        self.bucket_num      = bucket_num
        self.hidden_dim_base = hidden_dim_base
        self.shared_correction = shared_correction
        self.embed_dim       = embed_dim
        self.use_bucket_emb  = use_bucket_emb
        self.aux_target_names = tuple(aux_target_names or ())
        self.use_aux_targets = len(self.aux_target_names) > 0

        if bucket_mean_proxy is not None:
            bmp = torch.tensor(bucket_mean_proxy, dtype=torch.float32).clamp(min=1e-7)
        else:
            bmp = torch.ones(bucket_num, dtype=torch.float32)
        self.register_buffer('bucket_mean_proxy', bmp)

        self.use_uncertainty = use_uncertainty

        if lambda_init is not None:
            assert len(lambda_init) == bucket_num, "lambda_init length must match bucket_num"
            init_tensor = torch.tensor(lambda_init, dtype=torch.float32)
        else:

            init_tensor = torch.full((bucket_num,), 0.1)
        self.lambda_params = nn.Parameter(init_tensor)

        self._shared_emb = (shared_user_emb is not None and shared_video_emb is not None)
        if self._shared_emb:
            self.user_emb  = shared_user_emb
            self.video_emb = shared_video_emb
            embed_dim = shared_user_emb.embedding_dim
            self.embed_dim = embed_dim
        else:
            self.user_emb  = nn.Embedding(user_vocab_size, embed_dim)
            self.video_emb = nn.Embedding(video_vocab_size, embed_dim)

        self.duration_bucket_emb = nn.Embedding(bucket_num, embed_dim)

        self.use_uv_interaction = use_uv_interaction
        dur_input_dim = embed_dim if use_bucket_emb else 1
        uncertainty_dim = 1 if use_uncertainty else 0
        uv_dim = embed_dim if use_uv_interaction else 0
        input_dim = 8 + 2 + 1 + uncertainty_dim + hidden_dim_base + dur_input_dim + embed_dim + embed_dim + uv_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(p=debias_dropout),
            nn.Linear(128, hidden_dim), nn.ReLU(), nn.Dropout(p=debias_dropout),
        )

        if self.use_aux_targets:
            aux_hidden_dim = max(16, hidden_dim // 4)
            self.aux_towers = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(hidden_dim, 32), nn.ReLU(),
                    nn.Linear(32, aux_hidden_dim), nn.ReLU(),
                )
                for name in self.aux_target_names
            })
            self.aux_heads = nn.ModuleDict({
                name: nn.Linear(aux_hidden_dim, 1)
                for name in self.aux_target_names
            })
            aux_logits_dim = len(self.aux_target_names) * 8
            aux_embed_dim = len(self.aux_target_names) * aux_hidden_dim
            self.auxiliary_fusion = nn.Sequential(
                nn.Linear(hidden_dim + aux_logits_dim + aux_embed_dim, hidden_dim),
                nn.ReLU(),
            )

        expert_count = 1 if shared_correction else bucket_num
        self.regime_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(),
                nn.Linear(32, 16),         nn.ReLU(),
                nn.Linear(16, 1),
            )
            for _ in range(expert_count)
        ])

    def _nonlinear_features(self, pred, proxy=None, duration=None, bucket_idx=None, uncertainty=None):
        eps    = 1e-6
        pred_c = pred.clamp(eps, 1.0 - eps)
        logit  = torch.log(pred_c / (1.0 - pred_c))

        f1 = logit
        f2 = torch.relu(logit)
        f3 = torch.sigmoid(logit)
        f4 = (torch.tanh(logit) + 1.0) * 0.5
        f5 = logit.pow(2)
        f6 = torch.sqrt(torch.relu(logit) + eps)
        f7 = torch.log1p(torch.relu(logit))
        f8 = F.softplus(logit)
        feats = [f1, f2, f3, f4, f5, f6, f7, f8]

        if proxy is not None:
            if getattr(self, 'disable_proxy_features', False):
                zero_col = torch.zeros_like(logit)
                feats.extend([zero_col, zero_col.clone()])
                if bucket_idx is not None:
                    feats.append(zero_col.clone())
            else:
                proxy_safe = proxy.clamp(min=eps)
                f9  = torch.log(proxy_safe)
                if duration is not None:
                    dur_safe = duration.clamp(min=eps)
                    f10 = torch.log(proxy_safe / dur_safe)
                else:
                    f10 = f9
                feats.extend([f9, f10])

                if bucket_idx is not None:
                    bmp = self.bucket_mean_proxy[bucket_idx].view(-1, 1).clamp(min=eps)
                    f11 = torch.log(proxy_safe / bmp)
                    feats.append(f11)

        if uncertainty is not None and self.use_uncertainty:

            max_entropy = np.log(self.bucket_num + 1e-8)
            entropy_norm = uncertainty / max_entropy - 0.5
            feats.append(entropy_norm)

        return torch.cat(feats, dim=1)

    def _routing_weights(self, duration, thresholds):
        return duration_to_onehot(duration, thresholds)

    def get_routed_lambda(self, duration, thresholds):
        routing_w      = self._routing_weights(duration, thresholds)
        lambda_clipped = torch.clamp(self.lambda_params, -1.0, 1.0)
        return (routing_w * lambda_clipped.unsqueeze(0)).sum(dim=1, keepdim=True)

    def _expand_aux_logits(self, aux_logits):
        eps = 1e-6
        f1 = aux_logits
        f2 = torch.relu(aux_logits)
        f3 = torch.sigmoid(aux_logits)
        f4 = (torch.tanh(aux_logits) + 1.0) * 0.5
        f5 = aux_logits.pow(2)
        f6 = torch.sqrt(torch.relu(aux_logits) + eps)
        f7 = torch.log1p(torch.relu(aux_logits))
        f8 = F.softplus(aux_logits)
        stacked = torch.stack([f1, f2, f3, f4, f5, f6, f7, f8], dim=2)
        return stacked.reshape(aux_logits.shape[0], -1)

    def forward(self, base_pred, user_id, video_id, duration, thresholds,
                base_hidden=None, proxy=None, uncertainty=None, return_aux=False):

        bucket_idx = duration_to_onehot(duration, thresholds).argmax(dim=1)

        nonlinear = self._nonlinear_features(
            base_pred, proxy=proxy, duration=duration, bucket_idx=bucket_idx,
            uncertainty=uncertainty
        )

        if self._shared_emb:
            with torch.no_grad():
                u = self.user_emb(user_id.view(-1))
                v = self.video_emb(video_id.view(-1))
        else:
            u = self.user_emb(user_id.view(-1))
            v = self.video_emb(video_id.view(-1))

        if self.use_bucket_emb:
            dur_input  = self.duration_bucket_emb(bucket_idx)
        else:
            dur_input  = duration

        uv = (u * v) if self.use_uv_interaction else None

        if base_hidden is not None and self.hidden_dim_base > 0:
            assert base_hidden.shape[1] == self.hidden_dim_base, (
                "base_hidden dim {} != hidden_dim_base {}".format(
                    base_hidden.shape[1], self.hidden_dim_base))
            parts = [nonlinear, base_hidden, dur_input, u, v]
        else:
            parts = [nonlinear, dur_input, u, v]
        if uv is not None:
            parts.append(uv)
        x = torch.cat(parts, dim=1)

        shared_hidden = self.input_mlp(x)
        aux_logits_dict = {}

        if self.use_aux_targets:
            aux_embed_list = []
            aux_logits_list = []
            for name in self.aux_target_names:
                aux_embed = self.aux_towers[name](shared_hidden)
                aux_logit = self.aux_heads[name](aux_embed)
                aux_logits_dict[name] = aux_logit
                aux_embed_list.append(aux_embed)
                aux_logits_list.append(aux_logit)

            aux_logits = torch.cat(aux_logits_list, dim=1)
            aux_logits_nl = self._expand_aux_logits(aux_logits).detach()
            aux_embed_flat = torch.cat(aux_embed_list, dim=1).detach()
            hidden = self.auxiliary_fusion(
                torch.cat([shared_hidden, aux_logits_nl, aux_embed_flat], dim=1)
            )
        else:
            hidden = shared_hidden

        if self.shared_correction:
            transformed_prediction = self.regime_experts[0](hidden)
        else:
            expert_outputs = torch.cat(
                [expert(hidden) for expert in self.regime_experts], dim=1
            )
            routing_weights = self._routing_weights(duration, thresholds)
            transformed_prediction = (
                expert_outputs * routing_weights
            ).sum(dim=1, keepdim=True)
        if return_aux:
            return transformed_prediction, aux_logits_dict
        return transformed_prediction
