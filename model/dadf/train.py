

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import copy
import torch
import numpy as np
import argparse
import torch.nn.functional as F

from model import WideAndDeep, EGMN
from model.dadf import (
    DADF,
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
_DEFAULT_LOG_BASE = os.path.join(_ROOT, 'logs', 'dadf')
DEFAULT_WATCH_AUX_TARGETS = (
    'svr', 'fpr', 'evr', 'lvr', 'evr_p60', 'lvr_p80', 'lvr_p90'
)

def get_args():
    parser = argparse.ArgumentParser(
        description='DADF 框架（Box-Cox + 时长分桶专家）'
    )
    parser.add_argument('--base_model',   default='wlr',
                        choices=list_supported_models(),
                        help='基础模型: ' + str(list_supported_models()))
    parser.add_argument('--base_mlp_dims', type=int, nargs='+',
                        default=[256, 128, 64],
                        help='Backbone 主干 MLP 维度，例如: --base_mlp_dims 512 256 128')
    parser.add_argument('--base_only', action='store_true',
                        help='仅训练和评估 backbone，跳过全部 DADF 相关计算')
    parser.add_argument('--base_epoch', type=int, default=30,
                        help='--base_only 模式下 backbone 的最大训练轮数')
    parser.add_argument('--dataset_name', default='kuairec')
    parser.add_argument('--dataset_path', default=_DEFAULT_DATASET)
    parser.add_argument('--device',       default='cuda:0')
    parser.add_argument('--bsz',          type=int,   default=2048)
    parser.add_argument('--log_interval', type=int,   default=10)

    parser.add_argument('--warmup_epoch', type=int,   default=3,
                        help='仅 base_loss 的 warmup 轮数（默认 3；warmup 期 base 收敛后 debias 再开始训练）')
    parser.add_argument('--epoch',        type=int,   default=30,
                        help='联合训练轮数')
    parser.add_argument('--patience',     type=int,   default=6,
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

    parser.add_argument('--wr_bucknum',  type=int,   default=32,
                        help='TPM 树桶数（对应 run_tpm.py --wr_bucknum）')

    parser.add_argument('--bkt_num',     type=int,   default=50,
                        help='CREAD 分桶数（对应 run_cread.py --bkt_num）')

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
    parser.add_argument('--shared_correction', action='store_true',
                        help='使用共享 correction mapping 替代 duration-indexed experts')
    parser.add_argument('--lambda_init', type=float, nargs='+', default=None,
                        help='手动指定 lambda_init（覆盖数据集默认值），格式: 0.30 0.49 0.67')

    parser.add_argument('--no_base_hidden', action='store_true',
                        help='禁用 base model hidden layer 输入（ablation：验证 base_hidden 贡献）')
    parser.add_argument('--no_data_lambda', action='store_true',
                        help='禁用数据驱动 lambda 初始化，回退到默认 0.1（ablation：验证 lambda_init 贡献）')
    parser.add_argument('--gradual_warmup', action='store_true',
                        help='实验选项：在联合训练开始后逐步增加 DADF loss 权重；默认 alpha=1')
    parser.add_argument('--backbone_autotune', action='store_true',
                        help='实验选项：启用针对特定 backbone 的自动超参数覆盖；默认关闭')

    parser.add_argument('--no_bucket_emb', action='store_true',
                        help='禁用时长 bucket embedding，退回标量 duration 输入')
    parser.add_argument('--debias_embed_dim', type=int, default=16,
                        help='dadf_model user/video embedding 维度（默认 16）')
    parser.add_argument('--debias_hidden_dim', type=int, default=64,
                        help='dadf_model shared MLP hidden 维度（默认 64）')
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
                        help='两阶段纠偏：warmup后用MLE Box-Cox拟合每桶lambda初始化，dadf_model从更好起点学习（替代旧BMF方案）')
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
    bucket_sum   = [0.0] * (len(thresholds) + 1)
    bucket_count = [0]   * (len(thresholds) + 1)
    total = 0

    for features, label in dataloader_train:
        duration  = features['duration']
        play_time = label.float().view(-1)
        onehot    = duration_to_onehot(duration, thresholds)
        bkt_idx   = onehot.argmax(dim=1).cpu()
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
        vals = vals[vals > 1e-7]
        if len(vals) > 0:
            log_vals.extend(torch.log(vals).cpu().tolist())
    log_mean = float(np.mean(log_vals)) if log_vals else -4.6
    print('log_proxy_mean (p_calibrated re-centering): {:.4f}'.format(log_mean))
    return log_mean

def build_model_and_adapter(args, description, device, dataloaders=None):
    base_mlp_dims = tuple(args.base_mlp_dims)
    if args.base_model in ('vr', 'wlr', 'wd'):
        model = WideAndDeep(
            description, embed_dim=16,
            mlp_dims=base_mlp_dims, dropout=0.0,
        ).to(device)
        adapter = build_adapter(args.base_model, model)

    elif args.base_model == 'egmn':
        model = EGMN(
            description, embed_dim=16,
            share_mlp_dims=base_mlp_dims, dropout=0.2,
        ).to(device)
        adapter = build_adapter(args.base_model, model)

    elif args.base_model == 'tpm':
        from model import TPM
        from utils import get_playtime_percentiles_range
        model = TPM(
            description, class_num=args.wr_bucknum - 1,
            embed_dim=16, mlp_dims=base_mlp_dims, dropout=0.0,
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
            mlp_dims=base_mlp_dims, dropout=0.0,
        ).to(device)
        adapter = build_adapter('d2q', model)
        _pkl_suffix = getattr(args, 'pkl_suffix', '')
        if _pkl_suffix:
            _q_suffix = _pkl_suffix
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
            share_mlp_dims=base_mlp_dims,
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
            mlp_dims=base_mlp_dims, dropout=0.0,
        ).to(device)
        adapter = build_adapter('d2co', model)
        if dataloaders is not None:
            print('Computing GMM means for D2CO...')
            nega_GMM_mean, posi_GMM_mean = get_gmm_mean(dataloaders['train'])
            adapter.set_gmm(nega_GMM_mean, posi_GMM_mean)

    else:
        raise ValueError('Unknown base_model: {}'.format(args.base_model))
    return model, adapter


def _unique_parameters(*modules):
    parameters = {}
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameters[id(parameter)] = parameter
    return parameters


def _embedding_parameter_ids(*modules):
    parameter_ids = set()
    for module in modules:
        if module is None:
            continue
        for child in module.modules():
            if isinstance(child, torch.nn.Embedding):
                parameter_ids.update(id(parameter) for parameter in child.parameters())
    return parameter_ids


def _parameter_counts(*modules):
    parameters = _unique_parameters(*modules)
    embedding_ids = _embedding_parameter_ids(*modules)
    total = sum(parameter.numel() for parameter in parameters.values())
    dense = sum(
        parameter.numel()
        for parameter_id, parameter in parameters.items()
        if parameter_id not in embedding_ids
    )
    return total, dense


def report_parameter_counts(backbone, dadf_model=None):
    backbone_total, backbone_dense = _parameter_counts(backbone)
    print('Parameters | backbone total={:,} dense={:,}'.format(
        backbone_total, backbone_dense
    ))

    if dadf_model is None:
        return

    combined_total, combined_dense = _parameter_counts(backbone, dadf_model)
    additional_total = combined_total - backbone_total
    additional_dense = combined_dense - backbone_dense
    print('Parameters | DADF additional total={:,} dense={:,}'.format(
        additional_total, additional_dense
    ))
    print('Parameters | backbone+DADF total={:,} dense={:,}'.format(
        combined_total, combined_dense
    ))

def test_base(args, adapter, dataloaders, bucket_json_path=None, split='test'):
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
    print('{} base | MAE: {:.4f} | XAUC: {:.4f}'.format(split, mae, xauc))
    if bucket_json_path is not None:
        eval_by_duration_bucket(labels, scores, durations,
                                getattr(dataloaders, 'play_duration_max', None),
                                args.dataset_name, save_path=bucket_json_path)
    return mae, xauc


def train_base_only(args, adapter, dataloaders):
    opt_cls = torch.optim.Adam if args.base_model in ('tpm', 'cread') else torch.optim.Adagrad
    optimizer = opt_cls(
        adapter.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    best_xauc = -1.0
    best_state = None
    no_improve = 0
    total_iters = len(dataloaders['train'])

    for epoch in range(1, args.base_epoch + 1):
        adapter.train()
        epoch_loss = 0.0

        for i, (features, label) in enumerate(dataloaders['train']):
            loss = adapter.stage1_loss(features, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                print(
                    '  [Base] Epoch {}/{} Iter {}/{} loss={:.5f}'.format(
                        epoch, args.base_epoch, i + 1, total_iters,
                        epoch_loss / (i + 1),
                    ),
                    end='\r',
                )

        print(
            '  [Base] Epoch {}/{} loss={:.5f}'.format(
                epoch, args.base_epoch, epoch_loss / total_iters
            ),
            end='  ',
        )
        _, xauc = test_base(args, adapter, dataloaders, split='val')

        if xauc > best_xauc:
            best_xauc = xauc
            no_improve = 0
            best_state = copy.deepcopy(adapter.model.state_dict())
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(
                    'Base early stopping at epoch {}/{} '
                    '(XAUC no improvement for {} epochs)'.format(
                        epoch, args.base_epoch, args.patience
                    )
                )
                break

    if best_state is not None:
        adapter.model.load_state_dict(best_state)
        print('Restored best backbone (best val XAUC={:.7f})'.format(best_xauc))

    return best_xauc


def test(args, adapter, dadf_model, dataloaders, thresholds, bucket_json_path=None, split='test'):
    adapter.eval()
    dadf_model.eval()
    labels, scores, durations = [], [], []

    with torch.no_grad():
        for features, label in dataloaders[split]:
            duration                    = features['duration']
            p, proxy, base_hidden       = adapter.get_base_pred_and_hidden(features)

            uncertainty = None
            if hasattr(adapter, 'get_mixture_entropy') and getattr(dadf_model, 'use_uncertainty', False):
                uncertainty = adapter.get_mixture_entropy(features)

            transformed_prediction = dadf_model(
                p, features['user_id'], features['video_id'],
                duration, thresholds,
                base_hidden=base_hidden,
                proxy=proxy,
                uncertainty=uncertainty,
            )
            lambda_tensor = dadf_model.get_routed_lambda(duration, thresholds)
            _dfmin = getattr(args, 'debias_factor_min', 0.1)
            _dfmax = getattr(args, 'debias_factor_max', 10.0)
            debias_factor = boxcox_inverse(
                torch.clamp(transformed_prediction, -6.0, 6.0), lambda_tensor
            ).clamp(_dfmin, _dfmax)
            final_pred = debias_factor * proxy

            _fdfmax = getattr(args, 'final_debias_factor_max', None)
            _fdfmin = getattr(args, 'final_debias_factor_min', None)
            if _fdfmax is not None or _fdfmin is not None:
                final_pred = (debias_factor.clamp(
                    _fdfmin if _fdfmin is not None else 0.05,
                    _fdfmax if _fdfmax is not None else 20.0,
                ) * proxy)

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

def train_joint(args, adapter, dadf_model, dataloader_train, thresholds, dataloaders):

    opt_cls   = torch.optim.Adam if args.base_model in ('tpm', 'cread') else torch.optim.Adagrad

    _base_param_ids = set(id(p) for p in adapter.parameters())
    _debias_params  = [p for p in dadf_model.parameters() if id(p) not in _base_param_ids]
    optimizer = opt_cls([
        {'params': adapter.parameters(), 'lr': args.base_lr},
        {'params': _debias_params,       'lr': args.debias_lr * args.debias_lr_scale},
    ], weight_decay=args.weight_decay)

    device = next(dadf_model.parameters()).device
    bucket_counts    = torch.zeros(dadf_model.bucket_num, device=device)
    bucket_pt_sum    = torch.zeros(dadf_model.bucket_num, device=device)
    bucket_pt_sq_sum = torch.zeros(dadf_model.bucket_num, device=device)
    _sample_count = 0
    for features, label in dataloader_train:
        duration  = features['duration']
        play_time = label.float().view(-1, 1)
        onehot    = duration_to_onehot(duration, thresholds)
        bucket_counts    += onehot.sum(dim=0)
        bucket_pt_sum    += (onehot * play_time).sum(dim=0)
        bucket_pt_sq_sum += (onehot * play_time.pow(2)).sum(dim=0)
        _sample_count += label.shape[0]
        if _sample_count >= 50000:
            break
    bucket_freq = bucket_counts / bucket_counts.sum().clamp(min=1.0)

    bucket_mean = bucket_pt_sum    / bucket_counts.clamp(min=1.0)
    bucket_var  = bucket_pt_sq_sum / bucket_counts.clamp(min=1.0) - bucket_mean.pow(2)
    bucket_cv   = bucket_var.clamp(min=0.0).sqrt() / bucket_mean.clamp(min=1e-8)

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
    use_aux_targets = getattr(dadf_model, 'use_aux_targets', False)
    aux_target_names = getattr(dadf_model, 'aux_target_names', tuple())

    for epoch in range(1, total_epochs + 1):
        is_warmup  = epoch <= args.warmup_epoch
        phase_tag  = 'Warmup' if is_warmup else 'Joint '

        if not is_warmup and (epoch == args.warmup_epoch + 1) and getattr(args, 'two_stage_debias', False):
            print('Computing lambda_init from warmup residuals (MLE Box-Cox per bucket)...')
            lambda_init = compute_lambda_from_residuals(
                adapter, dataloader_train, thresholds,
                next(dadf_model.parameters()).device
            )
            with torch.no_grad():
                dadf_model.lambda_params.data = torch.tensor(
                    lambda_init, dtype=torch.float32,
                    device=dadf_model.lambda_params.device
                )
            print('Lambda initialized from warmup residuals: {}'.format(
                ['{:.4f}'.format(v) for v in lambda_init]))

        if not is_warmup and (epoch == args.warmup_epoch + 1) and getattr(args, 'freeze_base', False):
            for p in adapter.parameters():
                p.requires_grad_(False)
            _debias_params_only = [p for p in dadf_model.parameters() if id(p) not in set(id(q) for q in adapter.parameters())]
            opt_cls2 = torch.optim.Adam if args.base_model in ('tpm', 'cread') else torch.optim.Adagrad
            optimizer = opt_cls2(_debias_params_only, lr=args.debias_lr * args.debias_lr_scale,
                                  weight_decay=args.weight_decay)
            print('Freeze base: base model frozen, optimizer rebuilt with dadf_model only (lr={:.5f})'.format(
                args.debias_lr * args.debias_lr_scale))

        adapter.train()
        dadf_model.train()

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

                    p, proxy, base_hidden = adapter.get_base_pred_and_hidden(features)

                debias_factor_v2   = play_time / proxy
                lambda_tensor      = dadf_model.get_routed_lambda(duration, thresholds)
                debias_trans_label = boxcox_transform(debias_factor_v2, lambda_tensor)

                mask_label_valid  = (debias_trans_label >= -6.0) & (debias_trans_label <= 6.0)
                mask_factor_valid = (debias_factor_v2 >= 0.001) & (debias_factor_v2 <= 100.0)
                mask_play         = play_time > 0.0
                weight_valid_nr   = (mask_label_valid & mask_factor_valid & mask_play).float()
                mask_tight        = (debias_trans_label >= -4.0) & (debias_trans_label <= 4.0)
                weight_tight      = (mask_tight & mask_factor_valid & mask_play).float()

                with torch.no_grad():
                    sample_onehot = duration_to_onehot(duration, thresholds)
                    sample_bucket_w = (sample_onehot * bucket_weight.unsqueeze(0)).sum(dim=1, keepdim=True)
                weight_tight_balanced = weight_tight  * sample_bucket_w
                weight_valid_balanced = weight_valid_nr * sample_bucket_w

                if getattr(args, 'confidence_weighted_loss', False) and hasattr(adapter, 'get_mixture_confidence'):
                    with torch.no_grad():
                        conf = adapter.get_mixture_confidence(features)
                    weight_tight_balanced = weight_tight_balanced * conf
                    weight_valid_balanced = weight_valid_balanced * conf

                _ldw = getattr(args, 'label_debias_weight', 0.0)
                if _ldw > 0.0:
                    with torch.no_grad():
                        pt_norm = play_time.view(-1, 1) / (play_time.max().clamp(min=1e-8))
                        label_mult = 1.0 + _ldw * pt_norm
                    weight_tight_balanced = weight_tight_balanced * label_mult

                uncertainty = None
                if hasattr(adapter, 'get_mixture_entropy') and getattr(dadf_model, 'use_uncertainty', False):
                    with torch.no_grad():
                        uncertainty = adapter.get_mixture_entropy(features)

                if use_aux_targets:
                    transformed_prediction, aux_logits = dadf_model(
                        p.detach(), features['user_id'], features['video_id'],
                        duration, thresholds,
                        base_hidden=base_hidden,
                        proxy=proxy.detach(),
                        uncertainty=uncertainty,
                        return_aux=True,
                    )
                else:
                    transformed_prediction = dadf_model(
                        p.detach(), features['user_id'], features['video_id'],
                        duration, thresholds,
                        base_hidden=base_hidden,
                        proxy=proxy.detach(),
                        uncertainty=uncertainty,
                    )
                    aux_logits = {}

                diff_bc = (debias_trans_label - transformed_prediction).pow(2) * weight_tight_balanced
                debias_trans_loss = diff_bc.sum() / weight_tight_balanced.sum().clamp(min=1.0)

                debias_pred_factor = boxcox_inverse(
                    torch.clamp(transformed_prediction, -6.0, 6.0), lambda_tensor
                ).clamp(0.1, 10.0)
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
                        transformed_prediction, duration, thresholds, weight_valid_balanced,
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
                    lambda_c = torch.clamp(dadf_model.lambda_params, -1.0, 1.0)
                    lambda_smooth_loss = (
                        (lambda_c[1:] - lambda_c[:-1]).pow(2).mean()
                        if lambda_c.shape[0] > 1
                        else torch.tensor(0.0, device=lambda_c.device)
                    )
                else:
                    lambda_smooth_loss = torch.tensor(0.0, device=play_time.device)

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

            torch.nn.utils.clip_grad_norm_(dadf_model.parameters(), max_norm=1.0)
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
                dadf_model.lambda_params, -1.0, 1.0
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

        if not is_warmup:
            _, xauc = test(args, adapter, dadf_model, dataloaders, thresholds, split='val')
            if xauc > best_xauc:
                best_xauc  = xauc
                no_improve = 0
                best_state = {
                    'adapter':    copy.deepcopy(adapter.model.state_dict()),
                    'dadf_model': copy.deepcopy(dadf_model.state_dict()),
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

    if best_state is not None:
        adapter.model.load_state_dict(best_state['adapter'])
        dadf_model.load_state_dict(best_state['dadf_model'])
        print('Restored best model (best val XAUC={:.7f})'.format(best_xauc))

    return best_xauc

if __name__ == '__main__':
    args = get_args()
    if not args.base_mlp_dims or any(dim <= 0 for dim in args.base_mlp_dims):
        raise ValueError('--base_mlp_dims must contain positive integers')
    if args.base_epoch <= 0:
        raise ValueError('--base_epoch must be positive')

    use_aux_targets = args.use_aux_targets and not args.base_only
    args.use_aux_targets = use_aux_targets
    aux_target_names = parse_aux_target_names(args.aux_targets) if use_aux_targets else tuple()
    if use_aux_targets and not aux_target_names:
        raise ValueError('--use_aux_targets is set but --aux_targets is empty')
    if use_aux_targets:
        print('Watch-time auxiliary targets enabled: {}'.format(', '.join(aux_target_names)))
    _autotune = getattr(args, 'backbone_autotune', False) and not args.base_only
    if _autotune and getattr(args, 'full_data', False) and args.patience == 5:
        args.patience = 6
        print('full_data: patience auto-set to 6 (longer epochs, prevent premature stop)')
    if _autotune and args.base_model == 'wlr' and args.epoch == 30:
        args.epoch = 50

    if _autotune and args.dataset_name == 'kuairec' and args.nr_weight == 1.0:
        if getattr(args, 'two_stage_debias', False):
            args.nr_weight = 0.05
            print('KuaiRec+two_stage: nr_weight auto-set to 0.05 (residual BC more normal; ablation: 0.05 > 0.1)')
        else:
            args.nr_weight = 0.1
            print('KuaiRec: nr_weight auto-set to 0.1 (lambda pre-tuned; ablation: 0.1 > 0.3 > 1.0 on XAUC)')

    if _autotune and args.dataset_name == 'wechat21' and args.nr_weight == 1.0:
        args.nr_weight = 0.1
        print('WeChat21: nr_weight auto-set to 0.1 (same BC distribution as KuaiRec; default 1.0 overwhelms regression)')

    _wd_models = ('vr', 'wlr', 'wd', 'd2co')
    if (_autotune and args.dataset_name == 'kuairec'
            and args.base_model in _wd_models and args.debias_lr == 0.01):
        args.debias_lr = 0.02
        print('KuaiRec {}: debias_lr auto-set to 0.02 (WD proxy benefits from faster debias convergence)'.format(args.base_model))

    if _autotune and args.dataset_name == 'kuairec' and args.patience == 5:
        args.patience = 6
        print('KuaiRec: patience auto-set to 6 (dadf_model with lr=0.02 needs more epochs to plateau)')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and args.epoch == 30 and args.patience == 6):
        args.epoch = 12
        args.patience = 8
        print('KuaiRec+EGMN: epoch auto-set to 12, patience to 8')
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and getattr(args, 'label_debias_weight', 0.0) == 0.0):
        args.label_debias_weight = 3.0
        print('KuaiRec+EGMN: label_debias_weight auto-set to 3.0')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and args.debias_lr == 0.01):
        args.debias_lr = 0.0001
        print('KuaiRec+EGMN: debias_lr auto-set to 0.0001 (Phase 9 cross-seed verified: XAUC +0.0092, MAE -0.155 on seeds 0/42)')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and not getattr(args, 'two_stage_debias', False)):
        args.two_stage_debias = True
        print('KuaiRec+EGMN: two_stage_debias auto-set to True')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'egmn'
            and not getattr(args, 'confidence_weighted_loss', False)):
        args.confidence_weighted_loss = True
        print('KuaiRec+EGMN: confidence_weighted_loss auto-set')

    if (_autotune and getattr(args, 'full_data', False)
            and args.dataset_name == 'kuairec' and args.base_model == 'egmn'):
        args.debias_lr = 0.01
        args.label_debias_weight = 0.0
        args.alpha_max = 0.15
        if getattr(args, 'final_debias_factor_max', None) is None:
            args.final_debias_factor_max = 1.03
            args.final_debias_factor_min = 0.97
        print('KuaiRec+EGMN+full_data: debias_lr=0.01, ldw=0, alpha_max=0.15, clip=[0.97,1.03] '
              '(Phase 13 F13_5: MAE 4.112→4.106, XAUC 0.6136→0.6137)')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and not getattr(args, 'two_stage_debias', False)):
        args.two_stage_debias = True
        print('KuaiRec+D2Q: two_stage_debias auto-set to True')
    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and args.debias_lr == 0.01):
        args.debias_lr = 0.001
        print('KuaiRec+D2Q: debias_lr auto-set to 0.001 (two_stage+lr=0.001: XAUC=0.6131 > 0.6124)')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and getattr(args, 'alpha_max', 1.0) == 1.0
            and getattr(args, 'gradual_warmup', False)):
        args.alpha_max = 0.3
        print('KuaiRec+D2Q: alpha_max auto-set to 0.3 (conservative debias preserves D2Q proxy XAUC)')

    if (_autotune and args.dataset_name == 'kuairec' and args.base_model == 'd2q'
            and getattr(args, 'final_debias_factor_max', None) is None):
        args.final_debias_factor_max = 1.02
        args.final_debias_factor_min = 0.98
        print('KuaiRec+D2Q: final_debias_factor=[0.98,1.02] auto-set (Phase 7 N_R4: XAUC=0.6141 > 0.6137, NOTE: single seed, N_R16 validation pending)')
    run_mode = 'base' if args.base_only else 'dadf'
    log_dir = os.path.join(_ROOT, 'logs', args.dataset_name, run_mode, args.base_model)
    setup_logger(log_dir, args, 'model/dadf/train.py')

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

    description = dataloaders.description
    if args.base_only:
        model, adapter = build_model_and_adapter(
            args, description, device, dataloaders
        )
        report_parameter_counts(model)
        train_base_only(args, adapter, dataloaders)
        test_base(
            args,
            adapter,
            dataloaders,
            bucket_json_path=os.path.join(
                log_dir, 'bucket_{}_base_only.json'.format(args.base_model)
            ),
        )
        sys.exit(0)

    _DEFAULT_WARMUP = 3
    _iters_per_epoch = len(dataloaders['train'])
    _target_warmup_iters = 10000
    _auto_warmup = _target_warmup_iters // max(_iters_per_epoch, 1)
    _warmup_cap = max(3, args.epoch // 3)
    _auto_warmup = min(_auto_warmup, _warmup_cap)
    if args.warmup_epoch == _DEFAULT_WARMUP and _auto_warmup > _DEFAULT_WARMUP:
        args.warmup_epoch = _auto_warmup
        print('Auto warmup: {} epochs ({} iters/epoch, cap={})'.format(
            args.warmup_epoch, _iters_per_epoch, _warmup_cap))
    else:
        print('Warmup: {} epochs ({} iters/epoch)'.format(
            args.warmup_epoch, _iters_per_epoch))

    thresholds  = compute_duration_thresholds(
        dataloaders['train'],
        dataset_name=args.dataset_name,
        play_duration_max=getattr(dataloaders, 'play_duration_max', None),
        bucket_num=getattr(args, 'debias_bucket_num', 4),
        mode=getattr(args, 'duration_thresh_mode', 'auto'),
    )

    model, adapter = build_model_and_adapter(args, description, device, dataloaders)

    log_mean = compute_log_proxy_mean(dataloaders['train'])
    adapter.set_log_mean(log_mean)

    bucket_num = len(thresholds) + 1

    if args.no_data_lambda:
        lambda_init = None
        print('Ablation: no_data_lambda=True, using default lambda_init=0.1')
    elif args.dataset_name == 'kuairec':
        if bucket_num == 4:
            lambda_init = [0.145, 0.121, 0.057, -0.023]
        else:
            lambda_init = None
    else:
        lambda_init = None

    if getattr(args, 'lambda_init', None) is not None:
        lambda_init = args.lambda_init
        print('Lambda init overridden by --lambda_init: {}'.format(lambda_init))

    if args.no_base_hidden:
        hidden_dim_base = 0
        print('Ablation: no_base_hidden=True, hidden_dim_base=0')
    elif args.base_model in ('vr', 'wlr', 'wd', 'd2co', 'egmn'):
        hidden_dim_base = args.base_mlp_dims[-1]
    else:
        hidden_dim_base = 0

    _share = getattr(args, 'share_base_emb', False)
    _emb_layer = getattr(adapter.model, 'emb_layer', None)
    if _share and _emb_layer is not None:
        _shared_user_emb  = _emb_layer['user_id']  if 'user_id'  in _emb_layer else None
        _shared_video_emb = _emb_layer['video_id'] if 'video_id' in _emb_layer else None
    else:
        _shared_user_emb = _shared_video_emb = None

    dadf_model = DADF(
        user_vocab_size  = get_vocab_size(description, 'user_id'),
        video_vocab_size = get_vocab_size(description, 'video_id'),
        embed_dim=getattr(args, 'debias_embed_dim', 16),
        bucket_num=bucket_num,
        hidden_dim=getattr(args, 'debias_hidden_dim', 64),
        lambda_init=lambda_init,
        hidden_dim_base=hidden_dim_base,
        shared_correction=getattr(args, 'shared_correction', False),
        use_bucket_emb=not getattr(args, 'no_bucket_emb', False),
        bucket_mean_proxy=compute_bucket_mean_proxy(
            dataloaders['train'], None, thresholds, device
        ),
        shared_user_emb=_shared_user_emb,
        shared_video_emb=_shared_video_emb,
        use_uncertainty=getattr(args, 'use_uncertainty', False),
        aux_target_names=aux_target_names,
    ).to(device)

    dadf_model.disable_proxy_features = getattr(args, 'disable_proxy_features', False)
    if dadf_model.disable_proxy_features:
        print('Ablation: --disable_proxy_features=True, f9/f10/f11 dropped from debias input')

    report_parameter_counts(model, dadf_model)
    train_joint(args, adapter, dadf_model, dataloaders['train'], thresholds, dataloaders)

    print('\n[Ablation] Base model alone:')
    _log_dir = os.path.join(_ROOT, 'logs', args.dataset_name, 'dadf', args.base_model)
    test_base(args, adapter, dataloaders,
              bucket_json_path=os.path.join(_log_dir, 'bucket_{}_debias_base.json'.format(args.base_model)))

    print('[Final]    Base + DADF:')
    test(args, adapter, dadf_model, dataloaders, thresholds,
         bucket_json_path=os.path.join(_log_dir, 'bucket_{}_debias_final.json'.format(args.base_model)))
