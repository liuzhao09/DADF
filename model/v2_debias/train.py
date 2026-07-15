"""
V2 Debias 框架训练脚本
对齐 online_code/tf_graph.py vtr_debias_aware scope

本项目方法：Box-Cox 变换 + 时长感知分桶专家

两层架构:
  Layer 1: 基础模型（--base_model wd/egmn）
  Layer 2: DebiasNetV2（Box-Cox + 时长分桶专家）

训练流程:
  Warmup（warmup_epoch 轮）: 仅 base_loss → proxy 先收敛
  Correction（epoch 轮）   : 默认冻结 base，仅优化论文中的四项 DADF objective
  --joint_finetune_base 仅用于显式消融

运行方式（从项目根目录）:
  python model/v2_debias/train.py --base_model wd   --device cuda:0
  python model/v2_debias/train.py --base_model egmn --device cuda:0
"""

# ── 项目根目录注入（确保从任意位置均可运行）──────────────────
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ─────────────────────────────────────────────────────────────

import copy
import torch
import numpy as np
import argparse
import torch.nn.functional as F

from model import WideAndDeep, EGMN
from model.v2_debias import (
    DebiasNetV2,
    build_adapter,
    list_supported_models,
    boxcox_transform,
    boxcox_inverse,
    normal_regularization_loss,
    weighted_huber_loss,
    duration_to_onehot,
)
from model.framework_utils import mae_rescale_to_second, get_loaders, get_vocab_size
from utils import eval_mae, eval_xauc, eval_by_duration_bucket
from logger import setup_logger

_DEFAULT_DATASET  = os.path.join(_ROOT, 'dataset')
_DEFAULT_LOG_BASE = os.path.join(_ROOT, 'logs', 'debias_v2')
DEFAULT_WATCH_AUX_TARGETS = (
    'svr', 'fpr', 'evr', 'lvr', 'evr_p60', 'lvr_p80', 'lvr_p90'
)


# ────────────────────────────────────────────────────────────
# 参数
# ────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='V2 Debias 框架（Box-Cox + 时长分桶专家）'
    )
    parser.add_argument('--base_model',   default='wlr',
                        choices=list_supported_models(),
                        help='基础模型: ' + str(list_supported_models()))
    parser.add_argument('--dataset_name', default='kuairec')
    parser.add_argument('--dataset_path', default=_DEFAULT_DATASET)
    parser.add_argument('--device',       default='cuda:0')
    parser.add_argument('--bsz',          type=int,   default=2048)
    parser.add_argument('--log_interval', type=int,   default=10)

    parser.add_argument('--warmup_epoch', type=int,   default=3,
                        help='仅 base_loss 的 warmup 轮数（默认 3；warmup 期 base 收敛后 debias 再开始训练）')
    parser.add_argument('--epoch',        type=int,   default=30,
                        help='联合训练轮数')
    parser.add_argument('--patience',     type=int,   default=5,
                        help='early stopping patience（联合阶段 XAUC 不提升的容忍轮数）')

    parser.add_argument('--base_lr',         type=float, default=0.1)
    parser.add_argument('--debias_lr',       type=float, default=0.01)
    parser.add_argument('--debias_lr_scale', type=float, default=1.0,
                        help='Multiplier for debias_lr (e.g. 0.3 → effective lr=0.003). '
                             'Slow down debias head in early joint epochs to preserve XAUC.')
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--abs_time_weight', type=float, default=0.8)
    parser.add_argument('--nr_weight',       type=float, default=0.05)
    parser.add_argument('--huber_delta',     type=float, default=0.2,
                        help='abs_predtime_loss Huber delta（[0,1] 归一化空间，0.2=20%%范围内L2，超出L1；线上等效是纯MSE，此处有意引入L1保护）')

    # TPM-specific
    parser.add_argument('--wr_bucknum',  type=int,   default=32,
                        help='TPM 树桶数（对应 run_tpm.py --wr_bucknum）')
    # CREAD-specific
    parser.add_argument('--bkt_num',     type=int,   default=50,
                        help='CREAD 分桶数（对应 run_cread.py --bkt_num）')
    # D2Q-specific
    parser.add_argument('--quantile_max', type=float, default=100.0,
                        help='D2Q 分位数最大值（对应 run_d2q.py --quantile_max）')
    parser.add_argument('--label_debias_weight', type=float, default=0.0,
                        help='debias loss 中高label样本上采样权重系数，w=1+coeff*(play_time/batch_max)；0=均匀')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='保存预测结果到 .npz 文件（用于详细分析）')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--full-data', action='store_true',
                        help='使用全量数据 (kuairec_data_full.pkl)')
    parser.add_argument('--pkl_suffix', type=str, default='',
                        help='覆盖默认 pkl 后缀（如 _bins30 → kuairec_data_bins30.pkl；覆盖 --full-data）')
    parser.add_argument('--soft_routing', dest='hard_routing', action='store_false',
                        default=True,
                        help='使用 learned soft gate（仅用于消融；默认与论文一致采用 hard one-hot routing）')
    parser.add_argument('--lambda_init', type=float, nargs='+', default=None,
                        help='手动指定 lambda_init（覆盖数据集默认值），格式: 0.30 0.49 0.67')
    # ── Ablation flags ──────────────────────────────────────────────────────
    parser.add_argument('--no_base_hidden', action='store_true',
                        help='禁用 base model hidden layer 输入（ablation：验证 base_hidden 贡献）')
    parser.add_argument('--no_data_lambda', action='store_true',
                        help='禁用数据驱动 lambda 初始化，回退到默认 0.1（ablation：验证 lambda_init 贡献）')
    parser.add_argument('--gradual_warmup', action='store_true',
                        help='实验选项：在联合训练开始后逐步增加 DADF loss 权重；默认 alpha=1')
    parser.add_argument('--backbone_autotune', action='store_true',
                        help='实验选项：启用针对特定 backbone 的自动超参数覆盖；默认关闭')
    # ── Debias architecture flags ────────────────────────────────────────────
    parser.add_argument('--no_bucket_emb', action='store_true',
                        help='禁用时长 bucket embedding，退回标量 duration 输入')
    parser.add_argument('--debias_embed_dim', type=int, default=16,
                        help='debias_net user/video embedding 维度（默认 16）')
    parser.add_argument('--debias_hidden_dim', type=int, default=64,
                        help='debias_net shared MLP hidden 维度（默认 64）')
    parser.add_argument('--share_base_emb', action='store_true',
                        help='共享 base model 的 user/video embedding（WD 系列专用；无需额外训练，'
                             '减少稀疏数据下的过拟合）')
    parser.add_argument('--use_aux_targets', dest='use_aux_targets', action='store_true',
                        default=True,
                        help='启用由 play_time/duration 构造的 watch-time 辅助目标，'
                             '并将其 logits 注入 debias head（近似线上 multi-label aware）')
    parser.add_argument('--no_aux_targets', dest='use_aux_targets', action='store_false',
                        help='禁用辅助目标与 multi-label-aware 表征（消融）')
    parser.add_argument('--aux_targets', type=str,
                        default=','.join(DEFAULT_WATCH_AUX_TARGETS),
                        help='逗号分隔的辅助目标子集，可选: {}'
                             .format(','.join(DEFAULT_WATCH_AUX_TARGETS)))
    parser.add_argument('--aux_target_weight', type=float, default=0.10,
                        help='辅助目标 BCE loss 权重；各辅助目标 BCE 先求均值，再乘该权重')
    parser.add_argument('--two_stage_debias', action='store_true',
                        help='两阶段纠偏：warmup后用MLE Box-Cox拟合每桶lambda初始化，debias_net从更好起点学习（替代旧BMF方案）')
    parser.add_argument('--use_uncertainty', action='store_true',
                        help='使用EGMN混合模型不确定性作为debias特征（仅EGMN有效）')
    parser.add_argument('--confidence_weighted_loss', action='store_true',
                        help='置信度加权 BC 损失（EGMN专用）：不确定样本降权，使纠偏更专注可靠信号')
    parser.add_argument('--debias_bucket_num', type=int, default=4,
                        help='纠偏网络的时长专家数；阈值由 duration_thresh_mode 决定')
    parser.add_argument('--duration_thresh_mode',
                        choices=['auto', 'physical', 'quantile'], default='quantile',
                        help='分桶阈值计算策略。quantile=论文默认的等频分桶；'
                             'auto=K∈白名单走 physical，其他 quantile；'
                             'physical=强制 _PHYS_BUCKET_TABLE 白名单（K 不在表里会报错）；'
                             'quantile=跳过 physical 表强制走 np.quantile fallback。'
                             'K=1 无论哪种 mode 都是空 thresholds（global Box-Cox）。')
    parser.add_argument('--joint_finetune_base', dest='freeze_base', action='store_false',
                        default=True,
                        help='联合阶段继续微调 base model（仅用于消融；默认按论文冻结 first-stage predictor）')
    # ── 实验/消融控制参数 ─────────────────────────────────────────────────────
    parser.add_argument('--alpha_max', type=float, default=1.0,
                        help='渐变 warmup 最大 alpha 上界（default 1.0=无限制；<1.0=限制 debias loss 贡献比例）')
    parser.add_argument('--debias_factor_max', type=float, default=10.0,
                        help='神经网络纠偏因子上界（boxcox_inverse 后，two_stage 之前裁剪，default 10.0）')
    parser.add_argument('--debias_factor_min', type=float, default=0.1,
                        help='神经网络纠偏因子下界（default 0.1）')
    parser.add_argument('--final_debias_factor_max', type=float, default=None,
                        help='DADF correction factor 的最终上界（default None=不裁剪）'
                             '1.1=最多允许 10%% 修正；D2Q final_clip 实验（D_R5）')
    parser.add_argument('--final_debias_factor_min', type=float, default=None,
                        help='最终因子下界（default None=不裁剪）')
    parser.add_argument('--nr_pred_weight', type=float, default=0.0,
                        help='实验选项：预测侧 NR loss 权重（默认 0，与论文目标一致）')
    parser.add_argument('--lambda_smooth_weight', type=float, default=0.0,
                        help='实验选项：相邻 duration bucket 的 lambda 平滑权重（默认 0）')
    parser.add_argument('--kurtosis_weight', type=float, default=0.0,
                        help='实验选项：L_reg 中的超额峰度权重（默认 0）')
    parser.add_argument('--bucket_reweighting', action='store_true',
                        help='实验选项：按 duration bucket 的 CV/sqrt(freq) 重加权（默认关闭）')
    parser.add_argument('--disable_proxy_features', action='store_true',
                        help='§4.2 anchor ablation: drop f9/f10/f11 proxy-distribution features '
                             'from debias net input (log(proxy), log(proxy/dur), log(proxy/bucket_mean))')
    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

_PHYS_BUCKET_TABLE = {
    'kuairec': {
        4: [6000.0, 10000.0, 18000.0],
        5: [6000.0, 10000.0, 18000.0, 32000.0],
        8: [3000.0, 6000.0, 10000.0, 18000.0, 30000.0, 60000.0, 120000.0],
    },
    'wechat21': {
        4: [6000.0, 10000.0, 18000.0],
        8: [3000.0, 6000.0, 10000.0, 18000.0, 30000.0, 60000.0, 120000.0],
    },
}


def _quantile_fallback_thresholds(dataloader_train, bucket_num):
    """通用 quantile-based fallback：np.quantile(durations, linspace(0,1,K+1)[1:-1])."""
    durations = []
    for features, _ in dataloader_train:
        durations.extend(features['duration'].view(-1).cpu().tolist())
    durations = np.array(durations)
    qs = np.linspace(0.0, 1.0, bucket_num + 1)[1:-1]
    thresholds = [float(np.quantile(durations, q)) for q in qs]
    print('Quantile fallback thresholds (bucket_num={}): {}'.format(
        bucket_num, '/'.join('{:.4f}'.format(t) for t in thresholds)))
    return thresholds


def compute_duration_thresholds(dataloader_train, dataset_name='kuairec',
                                play_duration_max=None, bucket_num=4,
                                mode='auto'):
    """从训练集计算时长分桶 thresholds。

    Arguments:
        mode: one of {'auto', 'physical', 'quantile'}
            auto     : K∈_PHYS_BUCKET_TABLE → physical; 否则 → quantile（现行行为）
            physical : 强制 physical 表；K 不在表里则 ValueError
            quantile : 跳过 physical 表，直接走 _quantile_fallback_thresholds

    K=1  → []            (global Box-Cox, no bucketing)  — mode 无关
    """
    if bucket_num == 1:
        print('bucket_num=1: global Box-Cox (no bucketing), thresholds=[] (mode={})'.format(mode))
        return []

    phys_table = _PHYS_BUCKET_TABLE.get(dataset_name, {})

    if mode == 'physical':
        if bucket_num not in phys_table:
            raise ValueError(
                '--duration_thresh_mode=physical but bucket_num={} not in '
                '_PHYS_BUCKET_TABLE[{!r}]={} (available datasets: {}).'.format(
                    bucket_num, dataset_name, sorted(phys_table.keys()),
                    sorted(_PHYS_BUCKET_TABLE.keys())))
        phys_ms = phys_table[bucket_num]
        thresholds = [p / play_duration_max for p in phys_ms]
        phys_tag = '/'.join('{:.0f}s'.format(p / 1000.0) for p in phys_ms)
        print('[mode=physical] {} {}-bucket thresholds {} (normalized): {}'.format(
            dataset_name, bucket_num, phys_tag,
            '/'.join('{:.4f}'.format(t) for t in thresholds)))
        return thresholds

    if mode == 'quantile':
        print('[mode=quantile] skipping _PHYS_BUCKET_TABLE, K={}'.format(bucket_num))
        return _quantile_fallback_thresholds(dataloader_train, bucket_num)

    # mode == 'auto' : 保持现行行为（向后兼容主表）
    if play_duration_max is not None and bucket_num in phys_table:
        phys_ms = phys_table[bucket_num]
        thresholds = [p / play_duration_max for p in phys_ms]
        phys_tag = '/'.join('{:.0f}s'.format(p / 1000.0) for p in phys_ms)
        print('[mode=auto] {} {}-bucket thresholds {} (normalized): {}'.format(
            dataset_name, bucket_num, phys_tag,
            '/'.join('{:.4f}'.format(t) for t in thresholds)))
        return thresholds

    print('[mode=auto] fallback to quantile (K={} not in _PHYS_BUCKET_TABLE[{!r}])'.format(
        bucket_num, dataset_name))
    return _quantile_fallback_thresholds(dataloader_train, bucket_num)


def parse_aux_target_names(raw_names):
    if raw_names is None:
        return tuple()
    names = tuple(x.strip() for x in raw_names.split(',') if x.strip())
    unknown = sorted(set(names) - set(DEFAULT_WATCH_AUX_TARGETS))
    if unknown:
        raise ValueError('Unknown auxiliary targets: {}. Available: {}'.format(
            ', '.join(unknown), ', '.join(DEFAULT_WATCH_AUX_TARGETS)))
    return names


def build_watch_aux_labels(play_time, duration, play_duration_max_ms, selected_names):
    """
    基于 `play_time + duration` 构造 watch-time 辅助目标。

    规则对齐 `online_code/label_extract_flow.py` 中与主目标最相关的 7 个标签：
      svr / fpr / evr / lvr / evr_p60 / lvr_p80 / lvr_p90
    """
    scale = float(play_duration_max_ms)
    play_ms = play_time.float() * scale
    dur_ms = duration.float() * scale
    labels = {}

    if 'svr' in selected_names:
        labels['svr'] = ((play_ms > 0.0) & (play_ms <= 3000.0)).float()

    if 'fpr' in selected_names:
        labels['fpr'] = ((dur_ms > 3000.0) & (play_ms >= dur_ms)).float()

    if 'evr' in selected_names:
        labels['evr'] = (
            ((play_ms >= 7000.0) & (play_ms >= dur_ms)) | (play_ms >= 18000.0)
        ).float()

    if 'lvr' in selected_names:
        lvr_thr = torch.where(
            dur_ms <= 3000.0,
            torch.full_like(dur_ms, 18000.0),
            torch.where(
                dur_ms <= 40000.0,
                dur_ms * 0.9 + 4000.0,
                torch.full_like(dur_ms, 40000.0),
            ),
        )
        labels['lvr'] = (play_ms > lvr_thr).float()

    if 'evr_p60' in selected_names:
        evr_p60_thr = torch.where(
            dur_ms <= 3000.0,
            torch.full_like(dur_ms, 7000.0),
            torch.where(
                dur_ms <= 15000.0,
                dur_ms * 0.78 + 4610.0,
                torch.where(
                    dur_ms <= 80000.0,
                    -0.0046 * dur_ms * (dur_ms / 1000.0) + 0.69 * dur_ms + 7000.0,
                    torch.full_like(dur_ms, 32760.0),
                ),
            ),
        )
        labels['evr_p60'] = (play_ms > evr_p60_thr).float()

    if 'lvr_p80' in selected_names:
        lvr_p80_thr = torch.where(
            dur_ms <= 3000.0,
            torch.full_like(dur_ms, 18000.0),
            torch.where(
                dur_ms <= 15000.0,
                dur_ms * 0.73 + 7080.0,
                torch.where(
                    dur_ms <= 80000.0,
                    dur_ms * 0.97 + 2400.0,
                    torch.full_like(dur_ms, 80000.0),
                ),
            ),
        )
        labels['lvr_p80'] = (play_ms > lvr_p80_thr).float()

    if 'lvr_p90' in selected_names:
        lvr_p90_thr = torch.where(
            dur_ms <= 3000.0,
            torch.full_like(dur_ms, 18000.0),
            torch.where(
                dur_ms <= 40000.0,
                dur_ms * 0.69 + 15640.0,
                dur_ms * 0.97 + 4430.0,
            ),
        )
        labels['lvr_p90'] = (play_ms > lvr_p90_thr).float()

    return labels


def compute_bucket_mean_proxy(dataloader_train, adapter, thresholds, device, n_samples=50000):
    """
    从训练集计算每个时长 bucket 内的 proxy（play_time label）均值。
    用于 f11 = log(proxy/bucket_mean_proxy) 特征，捕获系统性时长偏差。

    Note: 用 label（play_time）而非模型 proxy 计算，不需要 adapter 收敛后再调用，
          在 warmup 前即可初始化 DebiasNetV2 的 bucket_mean_proxy。
          最优时 proxy ≈ E[play_time]，此初始化近似合理。
    """
    bucket_sum   = [0.0] * (len(thresholds) + 1)
    bucket_count = [0]   * (len(thresholds) + 1)
    total = 0

    for features, label in dataloader_train:
        duration  = features['duration']
        play_time = label.float().view(-1)
        onehot    = duration_to_onehot(duration, thresholds)   # [B, N]
        bkt_idx   = onehot.argmax(dim=1).cpu()                 # [B]
        pt_cpu    = play_time.cpu()
        for b in range(len(thresholds) + 1):
            mask = (bkt_idx == b)
            bucket_sum[b]   += pt_cpu[mask].sum().item()
            bucket_count[b] += int(mask.sum().item())
        total += play_time.shape[0]
        if total >= n_samples:
            break

    bucket_mean = [
        bucket_sum[b] / max(bucket_count[b], 1)
        for b in range(len(thresholds) + 1)
    ]
    print('Bucket mean play_time (proxy init): {}'.format(
        ['{:.6f}'.format(v) for v in bucket_mean]))
    return bucket_mean


def compute_lambda_from_residuals(adapter, dataloader_train, thresholds, device, n_samples=100000):
    """
    After warmup, compute optimal Box-Cox lambda per bucket using MLE.
    Replaces bucket_mean_debias_factor.

    For each duration bucket, fits lambda such that BoxCox(play_time/proxy, lambda) ~ N(0,1).
    Uses scipy MLE estimation on the bias = play_time / proxy ratio per bucket.
    """
    from scipy.stats import boxcox as scipy_boxcox
    import numpy as np

    adapter.eval()
    n_buckets = len(thresholds) + 1
    bucket_residuals = [[] for _ in range(n_buckets)]
    total = 0

    with torch.no_grad():
        for features, label in dataloader_train:
            _, proxy = adapter.get_base_pred(features)
            play_time = label.float().view(-1)
            valid = (play_time > 1e-7) & (proxy.view(-1) > 1e-7)

            bias = (play_time / proxy.view(-1).clamp(min=1e-7)).clamp(0.001, 200.0)
            bkt_idx = duration_to_onehot(features['duration'].to(device), thresholds).argmax(dim=1)

            for b in range(n_buckets):
                mask = (bkt_idx == b) & valid
                if mask.sum() > 0:
                    bucket_residuals[b].extend(bias[mask].cpu().numpy().tolist())

            total += label.shape[0]
            if total >= n_samples:
                break

    lambda_init = []
    for b in range(n_buckets):
        vals = np.array(bucket_residuals[b], dtype=np.float32)
        if len(vals) >= 50:
            vals = np.clip(vals, 1e-4, 500.0)
            try:
                _, lam = scipy_boxcox(vals)
                lam = float(np.clip(lam, -2.0, 3.0))
            except Exception:
                lam = 0.3
        else:
            lam = 0.3
        lambda_init.append(lam)
        print('  Bucket {} lambda_init (from warmup residuals): {:.4f} (n={})'.format(
            b, lam, len(vals)))

    adapter.train()
    return lambda_init


def compute_log_proxy_mean(dataloader_train):
    log_vals = []
    for _, label in dataloader_train:
        vals = label.float().view(-1)
        vals = vals[vals > 1e-7]          # 排除零值（未播放样本）
        if len(vals) > 0:
            log_vals.extend(torch.log(vals).cpu().tolist())
    log_mean = float(np.mean(log_vals)) if log_vals else -4.6
    print('log_proxy_mean (p_calibrated re-centering): {:.4f}'.format(log_mean))
    return log_mean


def build_model_and_adapter(args, description, device, dataloaders=None):
    if args.base_model in ('vr', 'wlr', 'wd'):
        model = WideAndDeep(
            description, embed_dim=16,
            mlp_dims=(256, 128, 64), dropout=0.0,
        ).to(device)
        adapter = build_adapter(args.base_model, model)

    elif args.base_model == 'egmn':
        model = EGMN(
            description, embed_dim=16,
            share_mlp_dims=(256, 128, 64), dropout=0.2,
        ).to(device)
        adapter = build_adapter(args.base_model, model)

    elif args.base_model == 'tpm':
        from model import TPM
        from utils import get_playtime_percentiles_range
        model = TPM(
            description, class_num=args.wr_bucknum - 1,
            embed_dim=16, mlp_dims=(256, 128, 64), dropout=0.0,
        ).to(device)
        adapter = build_adapter('tpm', model)
        if dataloaders is not None:
            bucket_begins, bucket_ends = get_playtime_percentiles_range(
                dataloaders['train'], args.wr_bucknum, str(device)
            )
            adapter.set_buckets(bucket_begins, bucket_ends, args.wr_bucknum)

    elif args.base_model == 'd2q':
        from model import D2Q
        from model.framework_utils import get_buckets_infor
        import os
        model = D2Q(
            description, embed_dim=16,
            mlp_dims=(256, 128, 64), dropout=0.0,
        ).to(device)
        adapter = build_adapter('d2q', model)
        _pkl_suffix = getattr(args, 'pkl_suffix', '')
        if _pkl_suffix:
            _q_suffix = _pkl_suffix  # e.g., '_full_bins40' → d2q_duration_bucket_playtime_quantiles_full_bins40.csv
        else:
            _q_suffix = '' if (getattr(args, 'full_data', False) or args.dataset_name == 'wechat21') else '_10pct'
        buckets_quantiles_path = os.path.join(
            args.dataset_path, args.dataset_name,
            'd2q_duration_bucket_playtime_quantiles{}.csv'.format(_q_suffix)
        )
        buckets_quantiles = get_buckets_infor(buckets_quantiles_path)
        adapter.set_quantiles(buckets_quantiles, quantile_max=args.quantile_max)

    elif args.base_model == 'cread':
        from model import Cread
        from model.framework_utils import cread_grid_search
        if dataloaders is not None:
            print('Running CREAD grid search for split nodes...')
            split_nodes = cread_grid_search(dataloaders['train'], args.bkt_num)
            M = split_nodes.shape[0]
        else:
            M = args.bkt_num
            split_nodes = None
        model = Cread(
            description, embed_dim=16,
            share_mlp_dims=(256, 128, 64),
            output_mlp_dims=(32,),
            head_num=M, dropout=0.0,
        ).to(device)
        adapter = build_adapter('cread', model)
        if split_nodes is not None:
            adapter.set_split_nodes(split_nodes, device=device)

    elif args.base_model == 'd2co':
        from model.framework_utils import get_gmm_mean
        model = WideAndDeep(
            description, embed_dim=16,
            mlp_dims=(256, 128, 64), dropout=0.0,
        ).to(device)
        adapter = build_adapter('d2co', model)
        if dataloaders is not None:
            print('Computing GMM means for D2CO...')
            nega_GMM_mean, posi_GMM_mean = get_gmm_mean(dataloaders['train'])
            adapter.set_gmm(nega_GMM_mean, posi_GMM_mean)

    else:
        raise ValueError('Unknown base_model: {}'.format(args.base_model))
    return model, adapter


# ────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────

def test_base(args, adapter, dataloaders, bucket_json_path=None, split='test'):
    """消融基线：仅 base model proxy（无 V2 纠偏）"""
    adapter.eval()
    labels, scores, durations = [], [], []
    with torch.no_grad():
        for features, label in dataloaders[split]:
            _, proxy = adapter.get_base_pred(features)
            labels.extend(label.tolist())
            scores.extend(proxy.squeeze(1).tolist())
            durations.extend(features['duration'].view(-1).tolist())

    labels    = np.array(labels)
    scores    = np.array(scores)
    durations = np.array(durations)
    mae    = mae_rescale_to_second(args.dataset_name, eval_mae(labels, scores), getattr(dataloaders, 'play_duration_max', None))
    xauc   = eval_xauc(labels, scores)
    print('[Base only] MAE: {:.4f} | XAUC: {:.4f}'.format(mae, xauc))
    if bucket_json_path is not None:
        eval_by_duration_bucket(labels, scores, durations,
                                getattr(dataloaders, 'play_duration_max', None),
                                args.dataset_name, save_path=bucket_json_path)
    return mae, xauc


def test(args, adapter, debias_net, dataloaders, thresholds, bucket_json_path=None, split='test'):
    """完整两层评估（base + V2 纠偏）"""
    adapter.eval()
    debias_net.eval()
    labels, scores, durations = [], [], []

    with torch.no_grad():
        for features, label in dataloaders[split]:
            duration                    = features['duration']
            p, proxy, base_hidden       = adapter.get_base_pred_and_hidden(features)

            # EGMN uncertainty feature
            uncertainty = None
            if hasattr(adapter, 'get_mixture_entropy') and getattr(debias_net, 'use_uncertainty', False):
                uncertainty = adapter.get_mixture_entropy(features)

            debias_v2_output = debias_net(
                p, features['user_id'], features['video_id'],
                duration, thresholds,
                base_hidden=base_hidden,
                proxy=proxy,
                uncertainty=uncertainty,
            )
            lambda_tensor = debias_net.get_routed_lambda(duration, thresholds)
            _dfmin = getattr(args, 'debias_factor_min', 0.1)
            _dfmax = getattr(args, 'debias_factor_max', 10.0)
            debias_factor = boxcox_inverse(
                torch.clamp(debias_v2_output, -6.0, 6.0), lambda_tensor
            ).clamp(_dfmin, _dfmax)
            final_pred = debias_factor * proxy
            # Final factor clip (after all stages): optional narrow range for conservative models
            # Useful for D2Q where proxy is well-calibrated; prevents two_stage over-correction
            _fdfmax = getattr(args, 'final_debias_factor_max', None)
            _fdfmin = getattr(args, 'final_debias_factor_min', None)
            if _fdfmax is not None or _fdfmin is not None:
                final_pred = (debias_factor.clamp(
                    _fdfmin if _fdfmin is not None else 0.05,
                    _fdfmax if _fdfmax is not None else 20.0,
                ) * proxy)
            # NaN/Inf fallback: if debias produces invalid values (e.g. near-zero lambda
            # with large BC output), fall back to proxy (no correction) for that sample
            final_pred = torch.where(torch.isfinite(final_pred), final_pred, proxy)

            labels.extend(label.tolist())
            scores.extend(final_pred.squeeze(1).tolist())
            durations.extend(duration.view(-1).tolist())

    labels    = np.array(labels)
    scores    = np.array(scores)
    durations = np.array(durations)
    mae    = mae_rescale_to_second(args.dataset_name, eval_mae(labels, scores), getattr(dataloaders, 'play_duration_max', None))
    xauc   = eval_xauc(labels, scores)
    print('{} | MAE: {:.4f} | XAUC: {:.4f}'.format(split, mae, xauc))
    if bucket_json_path is not None:
        eval_by_duration_bucket(labels, scores, durations,
                                getattr(dataloaders, 'play_duration_max', None),
                                args.dataset_name, save_path=bucket_json_path)
    return mae, xauc


# ────────────────────────────────────────────────────────────
# 联合训练（对齐 tf_graph.py）
# ────────────────────────────────────────────────────────────

def train_joint(args, adapter, debias_net, dataloader_train, thresholds, dataloaders):
    """
    联合训练，对齐 tf_graph.py 的 total_loss 设计。

    Warmup（warmup_epoch 轮）: loss = base_loss
    纠偏（epoch 轮，默认冻结 base）:
      loss = debias_trans_loss                       (Box-Cox 空间 MSE)
           + abs_time_weight * abs_predtime_loss     (Huber)
           + nr_weight * nr_loss                     (矩正则)
           + aux_target_weight * aux_loss            (辅助 BCE)

    stop_gradient: p.detach() 传入 debias_net（阻断 debias_loss 流回 base 表示层）
    含 early stopping（patience 轮 final XAUC 不提升则停止）
    """
    # optimizer 类型与对应 standalone baseline 一致
    # vr/wlr → Adagrad；tpm/cread → Adam；其余 → Adagrad
    opt_cls   = torch.optim.Adam if args.base_model in ('tpm', 'cread') else torch.optim.Adagrad
    # 若 debias_net 共享 base model 的 embedding，排除重复参数防止 optimizer 报错
    _base_param_ids = set(id(p) for p in adapter.parameters())
    _debias_params  = [p for p in debias_net.parameters() if id(p) not in _base_param_ids]
    optimizer = opt_cls([
        {'params': adapter.parameters(), 'lr': args.base_lr},
        {'params': _debias_params,       'lr': args.debias_lr * args.debias_lr_scale},
    ], weight_decay=args.weight_decay)

    # 仅为显式 bucket_reweighting 实验估计 CV/frequency 权重。
    # CV（变异系数）= 该桶内 play_time 的标准差 / 均值，衡量"该桶的 debias 修正空间大小"
    # 数据分析：KuaiRec B3(18s+) CV=1.647 >> B0(0-6s) CV=0.653
    # 当前 1/sqrt(freq) 只考虑频率不平衡，忽略了 B3 信号最强的事实
    # CV × 1/sqrt(freq) 让信号强的桶得到更多梯度关注
    device = next(debias_net.parameters()).device
    bucket_counts    = torch.zeros(debias_net.bucket_num, device=device)
    bucket_pt_sum    = torch.zeros(debias_net.bucket_num, device=device)  # Σ play_time
    bucket_pt_sq_sum = torch.zeros(debias_net.bucket_num, device=device)  # Σ play_time²
    _sample_count = 0
    for features, label in dataloader_train:
        duration  = features['duration']
        play_time = label.float().view(-1, 1)
        onehot    = duration_to_onehot(duration, thresholds)  # [B, N]
        bucket_counts    += onehot.sum(dim=0)
        bucket_pt_sum    += (onehot * play_time).sum(dim=0)
        bucket_pt_sq_sum += (onehot * play_time.pow(2)).sum(dim=0)
        _sample_count += label.shape[0]
        if _sample_count >= 50000:  # 采样 5 万条估算即可，不用全量
            break
    bucket_freq = bucket_counts / bucket_counts.sum().clamp(min=1.0)
    # CV per bucket
    bucket_mean = bucket_pt_sum    / bucket_counts.clamp(min=1.0)
    bucket_var  = bucket_pt_sq_sum / bucket_counts.clamp(min=1.0) - bucket_mean.pow(2)
    bucket_cv   = bucket_var.clamp(min=0.0).sqrt() / bucket_mean.clamp(min=1e-8)
    # The manuscript default gives every valid sample and duration group equal
    # weight. CV/frequency weighting is retained only as an explicit experiment.
    if getattr(args, 'bucket_reweighting', False):
        bucket_weight = bucket_cv / bucket_freq.sqrt().clamp(min=1e-3)
        bucket_weight = bucket_weight / bucket_weight.mean()
        reg_bucket_freq = bucket_freq
    else:
        bucket_weight = torch.ones_like(bucket_freq)
        reg_bucket_freq = None
    print('Bucket frequencies: {}'.format(['{:.3f}'.format(f.item()) for f in bucket_freq]))
    print('Bucket CV (play_time std/mean): {}'.format(['{:.3f}'.format(c.item()) for c in bucket_cv]))
    print('Bucket sample weights: {}'.format(['{:.3f}'.format(w.item()) for w in bucket_weight]))

    total_epochs = args.warmup_epoch + args.epoch
    best_xauc    = -1.0
    no_improve   = 0
    best_state   = None
    play_duration_max_ms = float(getattr(dataloaders, 'play_duration_max', 1.0))
    use_aux_targets = getattr(debias_net, 'use_aux_targets', False)
    aux_target_names = getattr(debias_net, 'aux_target_names', tuple())

    for epoch in range(1, total_epochs + 1):
        is_warmup  = epoch <= args.warmup_epoch
        phase_tag  = 'Warmup' if is_warmup else 'Joint '

        # Detect warmup→joint transition: initialize lambda_params from warmup residuals
        if not is_warmup and (epoch == args.warmup_epoch + 1) and getattr(args, 'two_stage_debias', False):
            print('Computing lambda_init from warmup residuals (MLE Box-Cox per bucket)...')
            lambda_init = compute_lambda_from_residuals(
                adapter, dataloader_train, thresholds,
                next(debias_net.parameters()).device
            )
            with torch.no_grad():
                debias_net.lambda_params.data = torch.tensor(
                    lambda_init, dtype=torch.float32,
                    device=debias_net.lambda_params.device
                )
            print('Lambda initialized from warmup residuals: {}'.format(
                ['{:.4f}'.format(v) for v in lambda_init]))

        # Freeze base model after warmup: only update debias_net during joint training
        if not is_warmup and (epoch == args.warmup_epoch + 1) and getattr(args, 'freeze_base', False):
            for p in adapter.parameters():
                p.requires_grad_(False)
            _debias_params_only = [p for p in debias_net.parameters() if id(p) not in set(id(q) for q in adapter.parameters())]
            opt_cls2 = torch.optim.Adam if args.base_model in ('tpm', 'cread') else torch.optim.Adagrad
            optimizer = opt_cls2(_debias_params_only, lr=args.debias_lr * args.debias_lr_scale,
                                  weight_decay=args.weight_decay)
            print('Freeze base: base model frozen, optimizer rebuilt with debias_net only (lr={:.5f})'.format(
                args.debias_lr * args.debias_lr_scale))

        adapter.train()
        debias_net.train()

        total_iters  = len(dataloader_train)
        epoch_base   = 0.0
        epoch_debias = 0.0
        epoch_aux    = 0.0

        for i, (features, label) in enumerate(dataloader_train):
            play_time = label.float().view(-1, 1)
            duration  = features['duration']

            base_loss = adapter.stage1_loss(features, label)

            if is_warmup:
                loss = base_loss
            else:
                with torch.no_grad():
                    # 单次前向同时取 p/proxy/hidden，WD 系列避免 BN 被额外更新
                    p, proxy, base_hidden = adapter.get_base_pred_and_hidden(features)

                debias_factor_v2   = play_time / proxy
                lambda_tensor      = debias_net.get_routed_lambda(duration, thresholds)
                debias_trans_label = boxcox_transform(debias_factor_v2, lambda_tensor)

                mask_label_valid  = (debias_trans_label >= -6.0) & (debias_trans_label <= 6.0)
                mask_factor_valid = (debias_factor_v2 >= 0.001) & (debias_factor_v2 <= 100.0)
                mask_play         = play_time > 0.0
                weight_valid_nr   = (mask_label_valid & mask_factor_valid & mask_play).float()
                mask_tight        = (debias_trans_label >= -4.0) & (debias_trans_label <= 4.0)
                weight_tight      = (mask_tight & mask_factor_valid & mask_play).float()

                # 获取每个样本的桶权重，乘入 weight_tight / weight_valid_nr
                with torch.no_grad():
                    sample_onehot = duration_to_onehot(duration, thresholds)  # [B, N]
                    sample_bucket_w = (sample_onehot * bucket_weight.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B, 1]
                weight_tight_balanced = weight_tight  * sample_bucket_w
                weight_valid_balanced = weight_valid_nr * sample_bucket_w

                # 置信度加权：EGMN 不确定的样本纠偏目标噪声大，降低其 loss 权重
                if getattr(args, 'confidence_weighted_loss', False) and hasattr(adapter, 'get_mixture_confidence'):
                    with torch.no_grad():
                        conf = adapter.get_mixture_confidence(features)  # [B, 1]
                    weight_tight_balanced = weight_tight_balanced * conf
                    weight_valid_balanced = weight_valid_balanced * conf

                # label-debias 高价值样本上采样：w = 1 + coeff * (play_time / batch_max)
                _ldw = getattr(args, 'label_debias_weight', 0.0)
                if _ldw > 0.0:
                    with torch.no_grad():
                        pt_norm = play_time.view(-1, 1) / (play_time.max().clamp(min=1e-8))
                        label_mult = 1.0 + _ldw * pt_norm
                    weight_tight_balanced = weight_tight_balanced * label_mult

                # EGMN uncertainty feature
                uncertainty = None
                if hasattr(adapter, 'get_mixture_entropy') and getattr(debias_net, 'use_uncertainty', False):
                    with torch.no_grad():
                        uncertainty = adapter.get_mixture_entropy(features)

                if use_aux_targets:
                    debias_v2_output, aux_logits = debias_net(
                        p.detach(), features['user_id'], features['video_id'],
                        duration, thresholds,
                        base_hidden=base_hidden,
                        proxy=proxy.detach(),
                        uncertainty=uncertainty,
                        return_aux=True,
                    )
                else:
                    debias_v2_output = debias_net(
                        p.detach(), features['user_id'], features['video_id'],
                        duration, thresholds,
                        base_hidden=base_hidden,
                        proxy=proxy.detach(),
                        uncertainty=uncertainty,
                    )
                    aux_logits = {}

                diff_bc = (debias_trans_label - debias_v2_output).pow(2) * weight_tight_balanced
                debias_trans_loss = diff_bc.sum() / weight_tight_balanced.sum().clamp(min=1.0)

                debias_pred_factor = boxcox_inverse(
                    torch.clamp(debias_v2_output, -6.0, 6.0), lambda_tensor
                ).clamp(0.1, 10.0)   # safety: debias factor max 10x in either direction
                pred_time = debias_pred_factor * proxy
                abs_predtime_loss = weighted_huber_loss(
                    pred_time, play_time, weight_tight_balanced, delta=args.huber_delta
                )

                nr_loss = normal_regularization_loss(
                    debias_trans_label, duration, thresholds, weight_valid_balanced,
                    bucket_freq=reg_bucket_freq,
                    kurtosis_weight=args.kurtosis_weight,
                )
                if args.nr_pred_weight > 0.0:
                    nr_pred_loss = normal_regularization_loss(
                        debias_v2_output, duration, thresholds, weight_valid_balanced,
                        bucket_freq=reg_bucket_freq,
                        kurtosis_weight=args.kurtosis_weight,
                    )
                else:
                    nr_pred_loss = torch.tensor(0.0, device=play_time.device)

                if use_aux_targets:
                    aux_labels = build_watch_aux_labels(
                        play_time, duration, play_duration_max_ms, aux_target_names
                    )
                    aux_mask = mask_play.float()
                    aux_denom = aux_mask.sum().clamp(min=1.0)
                    aux_terms = []
                    for name in aux_target_names:
                        bce = F.binary_cross_entropy_with_logits(
                            aux_logits[name], aux_labels[name], reduction='none'
                        )
                        aux_terms.append((bce * aux_mask).sum() / aux_denom)
                    aux_loss = torch.stack(aux_terms).mean() if aux_terms else torch.tensor(
                        0.0, device=play_time.device
                    )
                else:
                    aux_loss = torch.tensor(0.0, device=play_time.device)

                if args.lambda_smooth_weight > 0.0:
                    lambda_c = torch.clamp(debias_net.lambda_params, -1.0, 1.0)
                    lambda_smooth_loss = (
                        (lambda_c[1:] - lambda_c[:-1]).pow(2).mean()
                        if lambda_c.shape[0] > 1
                        else torch.tensor(0.0, device=lambda_c.device)
                    )
                else:
                    lambda_smooth_loss = torch.tensor(0.0, device=play_time.device)

                # 渐变 warmup：在前 warmup_epoch 内 debias loss 线性从 0 增长到 1
                # 让 debias head 缓慢进入联合训练，避免突变引起训练不稳
                # EGMN 实验验证：gradual warmup 对提升效果至关重要（+0.008 vs +0.004 无 warmup）
                if getattr(args, 'gradual_warmup', False):
                    joint_epoch = epoch - args.warmup_epoch
                    _alpha_max  = getattr(args, 'alpha_max', 1.0)
                    alpha = min(_alpha_max, joint_epoch / max(args.warmup_epoch, 1))
                else:
                    alpha = 1.0

                debias_loss = alpha * (
                    debias_trans_loss
                    + args.abs_time_weight * abs_predtime_loss
                    + args.nr_weight       * nr_loss
                    + args.nr_pred_weight  * nr_pred_loss
                    + args.aux_target_weight * aux_loss
                    + args.lambda_smooth_weight * lambda_smooth_loss
                )
                loss = debias_loss if args.freeze_base else base_loss + debias_loss
                epoch_debias += debias_loss.item()
                epoch_aux += aux_loss.item()

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止 debias_net 在少样本桶上产生爆梯度（CIKM16 bucket2/3 尤其少）
            torch.nn.utils.clip_grad_norm_(debias_net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_base += base_loss.item()

            if (i + 1) % args.log_interval == 0:
                if use_aux_targets and not is_warmup:
                    print('  [{}] Epoch {}/{} Iter {}/{} '
                          'base={:.5f} debias={:.5f} aux={:.5f}'.format(
                        phase_tag, epoch, total_epochs, i + 1, total_iters,
                        epoch_base / (i + 1),
                        epoch_debias / (i + 1),
                        epoch_aux / (i + 1),
                    ), end='\r')
                else:
                    print('  [{}] Epoch {}/{} Iter {}/{} '
                          'base={:.5f} debias={:.5f}'.format(
                        phase_tag, epoch, total_epochs, i + 1, total_iters,
                        epoch_base / (i + 1),
                        epoch_debias / (i + 1) if not is_warmup else 0.0,
                    ), end='\r')

        lam_str = ''
        if not is_warmup:
            lam_vals = torch.clamp(
                debias_net.lambda_params, -1.0, 1.0
            ).detach().cpu().numpy().round(4)
            lam_str = ' | λ={}'.format(lam_vals)

        if use_aux_targets and not is_warmup:
            print('  [{}] Epoch {}/{} base={:.5f} debias={:.5f} aux={:.5f}{}'.format(
                phase_tag, epoch, total_epochs,
                epoch_base / total_iters,
                epoch_debias / total_iters,
                epoch_aux / total_iters,
                lam_str,
            ), end='  ')
        else:
            print('  [{}] Epoch {}/{} base={:.5f} debias={:.5f}{}'.format(
                phase_tag, epoch, total_epochs,
                epoch_base  / total_iters,
                epoch_debias / total_iters if not is_warmup else 0.0,
                lam_str,
            ), end='  ')

        # 联合阶段：每 epoch 评估 + early stopping
        if not is_warmup:
            _, xauc = test(args, adapter, debias_net, dataloaders, thresholds, split='val')
            if xauc > best_xauc:
                best_xauc  = xauc
                no_improve = 0
                best_state = {
                    'adapter':    copy.deepcopy(adapter.model.state_dict()),
                    'debias_net': copy.deepcopy(debias_net.state_dict()),
                }
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print('Early stopping at epoch {}/{} '
                          '(XAUC no improvement for {} epochs)'.format(
                              epoch, total_epochs, args.patience))
                    break
        else:
            print()

    # Restore best weights before final evaluation
    if best_state is not None:
        adapter.model.load_state_dict(best_state['adapter'])
        debias_net.load_state_dict(best_state['debias_net'])
        print('Restored best model (best val XAUC={:.7f})'.format(best_xauc))

    return best_xauc


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()
    aux_target_names = parse_aux_target_names(args.aux_targets) if args.use_aux_targets else tuple()
    if args.use_aux_targets and not aux_target_names:
        raise ValueError('--use_aux_targets is set but --aux_targets is empty')
    if args.use_aux_targets:
        print('Watch-time auxiliary targets enabled: {}'.format(', '.join(aux_target_names)))
    _autotune = getattr(args, 'backbone_autotune', False)
    if _autotune and getattr(args, 'full_data', False) and args.patience == 5:
        args.patience = 6  # full data debias_v2: 每 epoch 长，patience=6 防过早停止（跨规模实验结论）
        print('full_data: patience auto-set to 6 (longer epochs, prevent premature stop)')
    if _autotune and args.base_model == 'wlr' and args.epoch == 30:
        args.epoch = 50   # wlr converges slower, needs more epochs
    # KuaiRec：lambda 由 tf_graph.py 专门调优，BC 分布已接近正态，过强 NR 约束反而阻碍修正
    # 两阶段纠偏后残差 BC 更接近正态，NR 约束可进一步降低；实验：two_stage+nr=0.05 > nr=0.1
    if _autotune and args.dataset_name == 'kuairec' and args.nr_weight == 1.0:
        if getattr(args, 'two_stage_debias', False):
            args.nr_weight = 0.05
            print('KuaiRec+two_stage: nr_weight auto-set to 0.05 (residual BC more normal; ablation: 0.05 > 0.1)')
        else:
            args.nr_weight = 0.1
            print('KuaiRec: nr_weight auto-set to 0.1 (lambda pre-tuned; ablation: 0.1 > 0.3 > 1.0 on XAUC)')
    # WeChat21：play_time 单位与 KuaiRec 相同（ms 归一化），BC 分布特性类似。
    # nr_weight=1.0 同样会使 NR 正则项远超回归损失，用 0.1 重新平衡（与 KuaiRec 一致）
    if _autotune and args.dataset_name == 'wechat21' and args.nr_weight == 1.0:
        args.nr_weight = 0.1
        print('WeChat21: nr_weight auto-set to 0.1 (same BC distribution as KuaiRec; default 1.0 overwhelms regression)')
    # KuaiRec debias_lr：WD 系列（vr/wlr/wd/d2co）从 0.02 > 0.01，分布类模型（egmn 等）反之
    # iter-9 WLR lr=0.02: ΔXAUC +0.027；EGMN lr=0.02: ΔXAUC +0.0045（退步）
    _wd_models = ('vr', 'wlr', 'wd', 'd2co')
    if (_autotune and args.dataset_name == 'kuairec'
            and args.base_model in _wd_models and args.debias_lr == 0.01):
        args.debias_lr = 0.02
        print('KuaiRec {}: debias_lr auto-set to 0.02 (WD proxy benefits from faster debias convergence)'.format(args.base_model))
    # KuaiRec patience：扩大到 6 给 debias_net 更多收敛时间（default 5 可能过早停止）
    if _autotune and args.dataset_name == 'kuairec' and args.patience == 5:
        args.patience = 6
        print('KuaiRec: patience auto-set to 6 (debias_net with lr=0.02 needs more epochs to plateau)')
    # KuaiRec+EGMN: epoch=15 + patience=10 — 30轮实验最优配置 (2026-04-20)
    # E_P3_1 (ep=15, seed=42): XAUC=0.6145 (best); 3-seed avg=0.6131±0.0014
    # ep=20 overfits (0.6127 < 0.6145); ep=12 sufficient but suboptimal
    # patience=10 prevents early stop before ep=15 peak (default 6 → may stop at ep=10)
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and args.epoch == 30 and args.patience == 6):
        args.epoch = 12
        args.patience = 8
        print('KuaiRec+EGMN: epoch auto-set to 12, patience to 8')
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and getattr(args, 'label_debias_weight', 0.0) == 0.0):
        args.label_debias_weight = 3.0
        print('KuaiRec+EGMN: label_debias_weight auto-set to 3.0')
    # KuaiRec+EGMN debias_lr=0.0001：更慢的 debias 学习率（Phase 9 val-split 跨 seed 验证）
    # seed=0:  baseline XAUC=0.6036 MAE=4.238 → lr=1e-4 XAUC=0.6128 MAE=4.086 (+0.0092/-0.152)
    # seed=42: baseline XAUC=0.6002 MAE=4.236 → lr=1e-4 XAUC=0.6094 MAE=4.074 (+0.0092/-0.162)
    # 两 seed Δ 完全一致 +0.0092 XAUC，稳定 MAE↓ + XAUC↑ 双赢
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and args.debias_lr == 0.01):
        args.debias_lr = 0.0001
        print('KuaiRec+EGMN: debias_lr auto-set to 0.0001 (Phase 9 cross-seed verified: XAUC +0.0092, MAE -0.155 on seeds 0/42)')
    # KuaiRec+EGMN：启用论文中的 per-bucket Box-Cox 初始化流程
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and not getattr(args, 'two_stage_debias', False)):
        args.two_stage_debias = True
        print('KuaiRec+EGMN: two_stage_debias auto-set to True')
    # KuaiRec+EGMN: confidence_weighted_loss — EGMN mixture confidence用于加权debias loss
    # VALIDATED (2026-04-19, E_R2 experiment):
    #   E_R2 (conf_weighted+auto_factor): Best=0.6130, ΔXAUC=+0.0013, beats E_R4a(0.6128)
    #   Mechanism: high-confidence EGMN predictions get more debias gradient →
    #   reduces correction-stage oscillation on the internal ablation runs
    #   Note: slightly reduces base XAUC (0.6117 vs E_R4a 0.6126) but improves final debias XAUC
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and not getattr(args, 'confidence_weighted_loss', False)):
        args.confidence_weighted_loss = True
        print('KuaiRec+EGMN: confidence_weighted_loss auto-set')
    # ══ KuaiRec+EGMN full-data 专属覆盖（Phase 13 验证）══════════════════════════
    # 10pct 的最优超参在 full-data 上 MAE 爆（F13_2: base 4.12 → final 4.29，+4.1%）
    # Full-data: N_train 10×, Adagrad 有效 lr ~1/√10，lr=1e-4 debias head 饿死
    # Phase 13 F13_5 最优组合：lr=0.01, ldw=0, alpha_max=0.15, clip=[0.97,1.03]
    #   → base MAE 4.112→final 4.106 (-0.006, ✅ MAE 下降)
    #   → base XAUC 0.6136→final 0.6137 (+0.0001, ✅ XAUC 微升)
    if (_autotune and getattr(args, 'full_data', False)
            and args.dataset_name == 'kuairec' and args.base_model == 'egmn'):
        args.debias_lr = 0.01             # 覆盖 10pct 的 0.0001（Adagrad 饿死）
        args.label_debias_weight = 0.0    # 覆盖 10pct 的 3.0（full 信号足，不需加权）
        args.alpha_max = 0.15             # 10pct 默认 1.0，full 上 debias 过强
        if getattr(args, 'final_debias_factor_max', None) is None:
            args.final_debias_factor_max = 1.03  # EGMN 之前无 clip，full 上需要
            args.final_debias_factor_min = 0.97
        print('KuaiRec+EGMN+full_data: debias_lr=0.01, ldw=0, alpha_max=0.15, clip=[0.97,1.03] '
              '(Phase 13 F13_5: MAE 4.112→4.106, XAUC 0.6136→0.6137)')
    # ═══════════════════════════════════════════════════════════════════════════
    # KuaiRec+D2Q 最优配置：two_stage_debias + debias_lr=0.001
    # 实验(10%): two_stage+lr=0.001 XAUC=0.6131 > 旧默认0.6124（+0.0007）
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and not getattr(args, 'two_stage_debias', False)):
        args.two_stage_debias = True
        print('KuaiRec+D2Q: two_stage_debias auto-set to True')
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and args.debias_lr == 0.01):
        args.debias_lr = 0.001
        print('KuaiRec+D2Q: debias_lr auto-set to 0.001 (two_stage+lr=0.001: XAUC=0.6131 > 0.6124)')
    # KuaiRec+D2Q: alpha_max=0.3 — 限制 debias loss 贡献防止扰乱已精准的 D2Q proxy
    # VALIDATED (2026-04-19, D_R2 experiment):
    #   D_R2 trajectory: ep1=0.6116→ep2=0.6123→ep3=0.6127→ep4=0.6131(peak, stable)
    #   Best debias XAUC=0.6131, Base XAUC=0.6135, ΔXAUC=-0.0004 (near-neutral!)
    #   vs default (alpha→1.0): ep1=0.5976, final=0.6040, ΔXAUC=-0.0095 (catastrophic!)
    # TRIPLE VALIDATED (D_R7 experiment 2026-04-19):
    #   D_R7 (no alpha_max, warmup=10): trajectory α=0.1→0.2→...→1.0:
    #     α=0.2: 0.6112(peak) → α=0.5: 0.6056 → α=1.0: 0.5962 (COLLAPSE!)
    #   D_R2(0.6131) - D_R7_at_full_alpha(0.5962) = +0.0169 = saved by alpha_max!
    #   Without alpha_max: base model also degrades (0.6122 vs D_R2's 0.6135)
    #   because high alpha starves base model of gradient in later epochs
    # Mechanism: alpha=0.3 cap ensures: 70% gradient for base, 30% for debias,
    # preventing both proxy disruption AND base model degradation
    # Note: Phase 7 N_R4 发现: final_clip=[0.98,1.02] → XAUC=0.6141 > baseline 0.6137! (单seed=42)
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and getattr(args, 'alpha_max', 1.0) == 1.0
            and getattr(args, 'gradual_warmup', False)):
        args.alpha_max = 0.3
        print('KuaiRec+D2Q: alpha_max auto-set to 0.3 (conservative debias preserves D2Q proxy XAUC)')
    # KuaiRec+D2Q: final_debias_factor=[0.98, 1.02] — 推理时修正幅度限制至±2%
    # Phase 7 N_R4 发现（2026-04-21）: clip=[0.98,1.02] → XAUC=0.6141 > baseline 0.6137
    # 机制：ep4时debias_net随机初始化+clip截断大修正→近似base XAUC(0.6135+微弱有益修正)
    # 已验证：seed=42(XAUC=0.6141) + seed=123(XAUC=0.6138) 均 > baseline 0.6137
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and getattr(args, 'final_debias_factor_max', None) is None):
        args.final_debias_factor_max = 1.02
        args.final_debias_factor_min = 0.98
        print('KuaiRec+D2Q: final_debias_factor=[0.98,1.02] auto-set (Phase 7 N_R4: XAUC=0.6141 > 0.6137, NOTE: single seed, N_R16 validation pending)')
    log_dir = os.path.join(_ROOT, 'logs', args.dataset_name, 'debias_v2', args.base_model)
    setup_logger(log_dir, args, 'model/v2_debias/train.py')

    if args.seed > -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    device    = torch.device(args.device)
    full_data = getattr(args, 'full_data', False)
    torch.cuda.empty_cache()

    dataloaders = get_loaders(
        args.dataset_name, args.dataset_path, device, args.bsz, full_data,
        pkl_suffix=getattr(args, 'pkl_suffix', ''),
    )

    # Auto warmup：保证 base model 见到足够梯度步数再开始联合训练
    # 只在使用默认值（3）时自动扩展；显式传 --warmup_epoch N 完全尊重用户设定
    # CIKM16（小数据，base 收敛快）：上界 epoch//10，WLR=5/VR=3，避免过长 warmup
    # KuaiRec（大数据）：上界 epoch//3，但 auto 通常 < 默认 3，不触发
    _DEFAULT_WARMUP = 3
    _iters_per_epoch = len(dataloaders['train'])
    _target_warmup_iters = 10000
    _auto_warmup = _target_warmup_iters // max(_iters_per_epoch, 1)
    _warmup_cap = max(3, args.epoch // 3)    # 其他数据集保持 1/3 上界
    _auto_warmup = min(_auto_warmup, _warmup_cap)
    if args.warmup_epoch == _DEFAULT_WARMUP and _auto_warmup > _DEFAULT_WARMUP:
        args.warmup_epoch = _auto_warmup
        print('Auto warmup: {} epochs ({} iters/epoch, cap={})'.format(
            args.warmup_epoch, _iters_per_epoch, _warmup_cap))
    else:
        print('Warmup: {} epochs ({} iters/epoch)'.format(
            args.warmup_epoch, _iters_per_epoch))

    description = dataloaders.description
    thresholds  = compute_duration_thresholds(
        dataloaders['train'],
        dataset_name=args.dataset_name,
        play_duration_max=getattr(dataloaders, 'play_duration_max', None),
        bucket_num=getattr(args, 'debias_bucket_num', 4),
        mode=getattr(args, 'duration_thresh_mode', 'auto'),
    )

    model, adapter = build_model_and_adapter(args, description, device, dataloaders)

    # 动态计算 log_proxy_mean，替代 adapter 中硬编码的 -4.6（KuaiRec 专用常数）
    log_mean = compute_log_proxy_mean(dataloaders['train'])
    adapter.set_log_mean(log_mean)

    # bucket_num = thresholds 数量 + 1（动态对齐）
    bucket_num = len(thresholds) + 1

    # KuaiRec 对齐线上 lambda 初始化（tf_graph.py:1302-1305）
    if args.no_data_lambda:
        lambda_init = None   # ablation: 回退到默认 0.1，验证 lambda_init 的贡献
        print('Ablation: no_data_lambda=True, using default lambda_init=0.1')
    elif args.dataset_name == 'kuairec':
        if bucket_num == 4:
            lambda_init = [0.145, 0.121, 0.057, -0.023]   # 4 桶，对齐线上
        else:
            lambda_init = None   # 非标准桶数用默认 0.1
    else:
        lambda_init = None
    # --lambda_init 命令行参数覆盖数据集默认值（用于消融实验）
    if getattr(args, 'lambda_init', None) is not None:
        lambda_init = args.lambda_init
        print('Lambda init overridden by --lambda_init: {}'.format(lambda_init))

    # WD 系列（VR/WLR/D2CO）暴露 64-dim hidden；EGMN 暴露 64-dim（share_mlp_dims[-1]=64）；其他返回 None
    _HIDDEN_DIMS = {'vr': 64, 'wlr': 64, 'wd': 64, 'd2co': 64, 'egmn': 64}
    if args.no_base_hidden:
        hidden_dim_base = 0   # ablation: 禁用 base_hidden，验证其贡献
        print('Ablation: no_base_hidden=True, hidden_dim_base=0')
    else:
        hidden_dim_base = _HIDDEN_DIMS.get(args.base_model, 0)

    # 共享 base model embeddings（WD 系列）：避免 debias_net 在稀疏数据上过拟合独立 embedding
    _share = getattr(args, 'share_base_emb', False)
    _emb_layer = getattr(adapter.model, 'emb_layer', None)
    if _share and _emb_layer is not None:
        _shared_user_emb  = _emb_layer['user_id']  if 'user_id'  in _emb_layer else None
        _shared_video_emb = _emb_layer['video_id'] if 'video_id' in _emb_layer else None
    else:
        _shared_user_emb = _shared_video_emb = None

    debias_net = DebiasNetV2(
        user_vocab_size  = get_vocab_size(description, 'user_id'),
        video_vocab_size = get_vocab_size(description, 'video_id'),
        embed_dim=getattr(args, 'debias_embed_dim', 16),
        bucket_num=bucket_num,
        hidden_dim=getattr(args, 'debias_hidden_dim', 64),
        lambda_init=lambda_init,
        hidden_dim_base=hidden_dim_base,
        hard_routing=getattr(args, 'hard_routing', True),
        use_bucket_emb=not getattr(args, 'no_bucket_emb', False),
        bucket_mean_proxy=compute_bucket_mean_proxy(
            dataloaders['train'], None, thresholds, device
        ),
        shared_user_emb=_shared_user_emb,
        shared_video_emb=_shared_video_emb,
        use_uncertainty=getattr(args, 'use_uncertainty', False),
        aux_target_names=aux_target_names,
    ).to(device)

    # Optional proxy-feature ablation.
    debias_net.disable_proxy_features = getattr(args, 'disable_proxy_features', False)
    if debias_net.disable_proxy_features:
        print('Ablation: --disable_proxy_features=True, f9/f10/f11 dropped from debias input')

    # 两阶段训练（base warmup → correction；base 默认冻结）
    train_joint(args, adapter, debias_net, dataloaders['train'], thresholds, dataloaders)

    # 消融评估（base only）
    print('\n[Ablation] Base model alone:')
    _log_dir = os.path.join(_ROOT, 'logs', args.dataset_name, 'debias_v2', args.base_model)
    test_base(args, adapter, dataloaders,
              bucket_json_path=os.path.join(_log_dir, 'bucket_{}_debias_base.json'.format(args.base_model)))

    # 完整评估（base + V2）
    print('[Final]    Base + V2 Debias:')
    test(args, adapter, debias_net, dataloaders, thresholds,
         bucket_json_path=os.path.join(_log_dir, 'bucket_{}_debias_final.json'.format(args.base_model)))
