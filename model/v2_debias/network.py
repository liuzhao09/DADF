"""
DebiasNetV2：Box-Cox 变换 + 时长感知分桶专家网络
参照 online_code/tf_graph.py vtr_debias_aware scope（简化版）

与原版的主要差异（离线简化版）：
  - 默认无多任务辅助 logit（原版用 fpr/svr/lvr/evr 等 8 个任务的
    logit × 8 种变换 × attention）
  - 默认用 base_pred → logit 空间 → 8 种非线性变换替代
  - 可选启用 watch-time auxiliary heads（svr/fpr/evr/lvr/evr_p60/lvr_p80/lvr_p90）
  - 时长 bucket 阈值由离线配置决定（原版硬编码 18/32/120s）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import duration_to_onehot


class DebiasNetV2(nn.Module):
    """
    V2 纠偏网络：Box-Cox 变换 + 时长感知分桶专家

    输入（通过 forward）：
      base_pred : [B, 1] sigmoid-like，调用处先 .detach()（stop_gradient）
                  内部转换到 logit 空间做非线性变换
      user_id   : [B, 1] long
      video_id  : [B, 1] long
      duration  : [B, 1] float，归一化时长
      thresholds: 长度 = bucket_num - 1 的列表，各数据集分桶切点

    输出: [B, 1]，Box-Cox 空间的纠偏因子预测值

    整体结构（参照 tf_graph.py debias_net_tower + debias_net_bucket_{i}_tower）：
      [nonlinear(8), duration(1), user_emb, video_emb]
        → input_mlp (128 → 64)
        → bucket_num 个 bucket 专家头各出 [B, 1]
        → duration one-hot 加权选择
    """

    def __init__(self, user_vocab_size, video_vocab_size,
                 embed_dim=16, bucket_num=4, hidden_dim=64,
                 lambda_init=None, hidden_dim_base=0, hard_routing=False,
                 use_bucket_emb=True, bucket_mean_proxy=None,
                 shared_user_emb=None, shared_video_emb=None,
                 bucket_mean_factor=None, use_uncertainty=False,
                 use_uv_interaction=False, debias_dropout=0.0,
                 aux_target_names=()):
        super().__init__()
        self.bucket_num      = bucket_num
        self.hidden_dim_base = hidden_dim_base
        self.hard_routing    = hard_routing
        self.embed_dim       = embed_dim
        self.use_bucket_emb  = use_bucket_emb
        self.aux_target_names = tuple(aux_target_names or ())
        self.use_aux_targets = len(self.aux_target_names) > 0

        # bucket_mean_proxy: 每 bucket 内 proxy 均值，用于 f11=log(proxy/mean) 特征
        # 注册为 buffer（不参与梯度，但随模型保存/迁移设备）
        if bucket_mean_proxy is not None:
            bmp = torch.tensor(bucket_mean_proxy, dtype=torch.float32).clamp(min=1e-7)
        else:
            bmp = torch.ones(bucket_num, dtype=torch.float32)
        self.register_buffer('bucket_mean_proxy', bmp)

        # bucket_mean_factor：两阶段纠偏 — E[play_time/proxy | bucket]
        # warmup 后计算，debias_net 只学残差；inference 时乘回系统因子
        # 值为 1.0 时退化为单阶段（默认）
        if bucket_mean_factor is not None:
            bmf2 = torch.tensor(bucket_mean_factor, dtype=torch.float32).clamp(min=1e-4)
        else:
            bmf2 = torch.ones(bucket_num, dtype=torch.float32)
        self.register_buffer('bucket_mean_factor', bmf2)
        self.use_two_stage = (bucket_mean_factor is not None)

        # 用户级修正：三阶段纠偏（桶级 → 用户级 → 残差）
        # 用 set_user_factors() 在 warmup 后设置；默认 None = 禁用
        self.user_mean_factor   = None      # [user_vocab_size], set post-warmup
        self.use_user_two_stage = False

        # 视频级修正：类似用户级，捕获视频内容质量偏差
        self.video_mean_factor   = None     # [video_vocab_size], set post-warmup
        self.use_video_two_stage = False

        # Uncertainty feature (EGMN only): mixture entropy as debias input
        self.use_uncertainty = use_uncertainty

        # 可学习 Box-Cox λ（每 bucket 一个），对应 tf_graph.py vtr_boxcox_lambda scope
        if lambda_init is not None:
            assert len(lambda_init) == bucket_num, "lambda_init length must match bucket_num"
            init_tensor = torch.tensor(lambda_init, dtype=torch.float32)
        else:
            # 默认初始化为 0.1：避开 λ=0 的梯度平坦区，加速早期收敛
            init_tensor = torch.full((bucket_num,), 0.1)
        self.lambda_params = nn.Parameter(init_tensor)

        # ID embedding：优先使用 base model 的共享 embedding（充分训练），否则独立 embedding
        self._shared_emb = (shared_user_emb is not None and shared_video_emb is not None)
        if self._shared_emb:
            self.user_emb  = shared_user_emb
            self.video_emb = shared_video_emb
            embed_dim = shared_user_emb.embedding_dim  # 与 base model 对齐
            self.embed_dim = embed_dim
        else:
            self.user_emb  = nn.Embedding(user_vocab_size, embed_dim)
            self.video_emb = nn.Embedding(video_vocab_size, embed_dim)

        # 时长 bucket 嵌入：将 duration bucket 映射为 embed_dim 维向量
        # 替代原来的标量 duration 输入，提升时长感知能力
        self.duration_bucket_emb = nn.Embedding(bucket_num, embed_dim)

        # Adaptive per-sample lambda：根据 user/video context 调整 lambda 强度
        # 输出 (-1,1) 标量，通过 lambda_adapter_scale 限制修正幅度
        self.lambda_adapter = nn.Sequential(
            nn.Linear(2 * embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Tanh(),
        )
        self.lambda_adapter_scale = 0.05  # 小扰动：NR loss 要求桶内 lambda 近似恒定，大幅修正会破坏正态化约束

        # 共享输入 MLP：8（nonlinear logit 变换）+ 2（proxy 比值特征）+ 1（bucket-relative proxy）
        #               + hidden_dim_base（base hidden，可选）
        #               + embed_dim or 1（duration: bucket emb or scalar）
        #               + 2×embed_dim（user/video embedding）
        #               + embed_dim（user×video 交叉特征，可选）
        # f11 = log(proxy/bucket_mean_proxy): proxy 相对于桶均值的偏差，直接捕获系统性时长偏差
        # f_uv = u ⊙ v: 用户-视频 Hadamard 积，捕获个性化偏好交互
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
            self.debias_fusion = nn.Sequential(
                nn.Linear(hidden_dim + aux_logits_dim + aux_embed_dim, hidden_dim),
                nn.ReLU(),
            )

        # bucket_num 个时长 bucket 专家头，对应 tf_graph.py debias_net_bucket_{i}_tower
        self.bucket_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(),
                nn.Linear(32, 16),         nn.ReLU(),
                nn.Linear(16, 1),
            )
            for _ in range(bucket_num)
        ])

        # Soft routing gate：将 duration 映射到 bucket_num 维 soft 权重
        # 用轻量 2 层 MLP（duration→32→bucket_num），温度参数 temperature 可学习
        # 保持 expert output 和 lambda selection 语义一致（同一套 soft_weights）
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, bucket_num),
        )
        # 可学习温度：训练初期接近 1（较软），随训练梯度自然收敛；clamp(min=0.1) 防止无限尖锐
        self.log_temperature = nn.Parameter(torch.zeros(1))  # exp(0)=1.0 初始温度

    # ─────────────────────────────────────────────────────────
    # 内部方法
    # ─────────────────────────────────────────────────────────

    def _nonlinear_features(self, pred, proxy=None, duration=None, bucket_idx=None, uncertainty=None):
        """
        sigmoid 输出 → logit 空间 → 8 种非线性变换 [B, 8]
        + 2 个增量 proxy 比值特征 [B, 2]（如果 proxy 不为 None）
        + 1 个 bucket-relative proxy 特征 [B, 1]（f11，如果 bucket_idx 不为 None）

        f11 = log(proxy / bucket_mean_proxy[bucket]):
          proxy 相对于当前时长桶均值的偏差
          正值 = 模型对该样本的预测高于桶均值（过度预测）
          负值 = 低于桶均值（预测不足）
          直接捕获系统性时长暴露偏差，是强纠偏信号
        """
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

        # §4.2 anchor ablation: disable_proxy_features=True -> 用 0 占位 f9/f10/f11,
        # 保持 input_dim 不变（Linear(123,128) 权重形状不动），只消融特征信号本身
        if proxy is not None:
            if getattr(self, 'disable_proxy_features', False):
                zero_col = torch.zeros_like(logit)   # [B, 1]
                feats.extend([zero_col, zero_col.clone()])   # f9=0, f10=0
                if bucket_idx is not None:
                    feats.append(zero_col.clone())           # f11=0
            else:
                proxy_safe = proxy.clamp(min=eps)
                f9  = torch.log(proxy_safe)
                if duration is not None:
                    dur_safe = duration.clamp(min=eps)
                    f10 = torch.log(proxy_safe / dur_safe)
                else:
                    f10 = f9
                feats.extend([f9, f10])

                # f11: bucket-relative proxy（捕获各桶系统偏差）
                if bucket_idx is not None:
                    bmp = self.bucket_mean_proxy[bucket_idx].view(-1, 1).clamp(min=eps)
                    f11 = torch.log(proxy_safe / bmp)
                    feats.append(f11)

        if uncertainty is not None and self.use_uncertainty:
            # Normalize entropy to ~N(0,1) range by dividing by log(bucket_num)
            max_entropy = np.log(self.bucket_num + 1e-8)  # max entropy for uniform
            entropy_norm = uncertainty / max_entropy - 0.5  # center at 0
            feats.append(entropy_norm)

        return torch.cat(feats, dim=1)

    def set_user_factors(self, user_mean_tensor):
        """
        Warmup 后设置用户级纠偏因子（三阶段纠偏第二阶段）。
        user_mean_tensor: [user_vocab_size] float tensor (CPU)
        """
        device = next(self.parameters()).device
        self.user_mean_factor   = user_mean_tensor.to(device)
        self.use_user_two_stage = True
        print('User-level two-stage debias enabled. '
              'Mean factor={:.4f}, std={:.4f}'.format(
                  float(user_mean_tensor.mean()), float(user_mean_tensor.std())))

    def set_video_factors(self, video_mean_tensor):
        """
        视频级纠偏因子：捕获视频内容质量偏差，和用户级独立叠加。
        video_mean_tensor: [video_vocab_size] float tensor (CPU)
        """
        device = next(self.parameters()).device
        self.video_mean_factor   = video_mean_tensor.to(device)
        self.use_video_two_stage = True
        print('Video-level two-stage debias enabled. '
              'Mean factor={:.4f}, std={:.4f}'.format(
                  float(video_mean_tensor.mean()), float(video_mean_tensor.std())))

    def to(self, *args, **kwargs):
        """Override to() so user/video mean factors follow device moves (multi-GPU safety)."""
        result = super().to(*args, **kwargs)
        device = next(result.parameters()).device
        if result.user_mean_factor is not None:
            result.user_mean_factor = result.user_mean_factor.to(device)
        if result.video_mean_factor is not None:
            result.video_mean_factor = result.video_mean_factor.to(device)
        return result

    # ─────────────────────────────────────────────────────────
    # 公共方法
    # ─────────────────────────────────────────────────────────

    def _soft_weights(self, duration, thresholds=None):
        """
        Soft routing gate：将 duration 映射到 [B, bucket_num] soft 权重。
        hard_routing=True 时退回 duration_to_onehot（消融实验用）。
        """
        if self.hard_routing and thresholds is not None:
            return duration_to_onehot(duration, thresholds)
        temperature = torch.exp(self.log_temperature).clamp(min=0.1)
        gate_logits = self.gate_mlp(duration)                  # [B, bucket_num]
        return torch.softmax(gate_logits / temperature, dim=1) # [B, bucket_num]

    def get_lambda_per_sample(self, duration, thresholds, user_id=None, video_id=None):
        """
        根据时长 bucket 为每个样本选取对应 λ，返回 [B, 1]。
        使用 soft routing（与 forward 一致），使 lambda 选择可微分。
        hard_routing=True 时使用 hard one-hot（保持与 forward 语义一致）。

        user_id, video_id（可选）：提供时激活 adaptive lambda，
        根据 user/video embedding 对 base_lambda 做小幅修正（±lambda_adapter_scale）。
        """
        soft_w         = self._soft_weights(duration, thresholds)         # [B, bucket_num]
        lambda_clipped = torch.clamp(self.lambda_params, -1.0, 1.0)      # [bucket_num]
        base_lambda    = (soft_w * lambda_clipped.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B, 1]

        # §4.2 anchor ablation: skip per-sample adapter when disable_lambda_adapter=True
        if (user_id is not None and video_id is not None
                and not getattr(self, 'disable_lambda_adapter', False)):
            with torch.no_grad():
                u = self.user_emb(user_id.view(-1))   # [B, embed_dim]
                v = self.video_emb(video_id.view(-1)) # [B, embed_dim]
            adapt_delta = self.lambda_adapter(torch.cat([u, v], dim=1))  # [B, 1]
            return torch.clamp(base_lambda + self.lambda_adapter_scale * adapt_delta, -1.0, 1.0)

        return base_lambda

    def _expand_aux_logits(self, aux_logits):
        """
        复用线上 debias_aware 的 8 种 logit 非线性映射。

        aux_logits: [B, T]
        return:     [B, T * 8]
        """
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
        """
        base_pred   : [B, 1] sigmoid-like，调用处先 .detach()
        duration    : [B, 1] float，归一化时长
        base_hidden : [B, hidden_dim_base] float，base model 最后隐层（可选）
                      WD 系列（VR/WLR/D2CO）传入 32-dim hidden；其他模型传 None
        proxy       : [B, 1] float，base model 的 watch-time 代理值（可选）
                      传入时激活 log(proxy) 和 log(proxy/duration) 两个增量特征
        return      : [B, 1] Box-Cox 空间预测值
        """
        # Duration input: bucket index (used for both embedding and f11 feature)
        bucket_idx = duration_to_onehot(duration, thresholds).argmax(dim=1)  # [B]

        nonlinear = self._nonlinear_features(
            base_pred, proxy=proxy, duration=duration, bucket_idx=bucket_idx,
            uncertainty=uncertainty
        )  # [B, 8+2+1 or 8+2 or 8]
        # ID embedding lookup (no_grad if shared to prevent gradient leaking to base model)
        if self._shared_emb:
            with torch.no_grad():
                u = self.user_emb(user_id.view(-1))
                v = self.video_emb(video_id.view(-1))
        else:
            u = self.user_emb(user_id.view(-1))                   # [B, embed_dim]
            v = self.video_emb(video_id.view(-1))                 # [B, embed_dim]

        # Duration input: bucket embedding or scalar (reuse bucket_idx computed above)
        if self.use_bucket_emb:
            dur_input  = self.duration_bucket_emb(bucket_idx)   # [B, embed_dim]
        else:
            dur_input  = duration                                 # [B, 1]

        # User-video Hadamard 交叉特征：捕获个性化偏好交互
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

        shared_hidden = self.input_mlp(x)                     # [B, hidden_dim]
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
            hidden = self.debias_fusion(
                torch.cat([shared_hidden, aux_logits_nl, aux_embed_flat], dim=1)
            )
        else:
            hidden = shared_hidden

        bucket_outputs = torch.cat(
            [head(hidden) for head in self.bucket_heads], dim=1
        )                                                      # [B, bucket_num]

        # Soft routing：soft_weights 同时用于 expert output 和 lambda，保持语义一致
        soft_w = self._soft_weights(duration, thresholds)     # [B, bucket_num]

        debias_output = (bucket_outputs * soft_w).sum(dim=1, keepdim=True)  # [B, 1]
        if return_aux:
            return debias_output, aux_logits_dict
        return debias_output
