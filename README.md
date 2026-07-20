# DADF: Distribution-Aware Debiasing Framework for Watch-Time Regression

[![Manuscript](https://img.shields.io/badge/status-manuscript-blue)](#scope-and-reproducibility)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

Reference implementation accompanying **DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems**.

## Scope and Reproducibility

This repository provides a research reference implementation of the DADF structure used for the public-benchmark study. It is aligned with the method described in the manuscript, but it is **not the complete production implementation**. Proprietary production feature pipelines, first-stage multi-task towers, serving infrastructure, and system optimizations are not included. The public code reconstructs the method with public-dataset features and auxiliary labels so that the core architecture can be inspected and evaluated.

Unless an experimental flag is explicitly supplied, the public implementation follows the manuscript: the first-stage predictor is frozen during correction training, duration groups use hard routing, Box-Cox parameters are group-level rather than adapted per sample, and no additional user-level or video-level correction stage is applied. The default objective contains transformed-space fitting, absolute-time, moment-regularization, and auxiliary-task terms. Prediction-side regularization, lambda smoothing, kurtosis regularization, bucket reweighting, backbone-specific auto-tuning, and narrow inference clipping are disabled by default.

## Overview

Watch-time prediction is a central regression task in short-video recommender systems. In production, watch-time labels are highly long-tailed, and prediction bias varies systematically across different watch-time intervals. A model may appear globally well-calibrated while severely overestimating short views and underestimating long views — a phenomenon we call **pseudo-balance** caused by error cancellation across intervals.

**DADF** addresses this by performing second-stage multiplicative residual correction on top of any deployed first-stage predictor. Rather than replacing the base model, DADF learns a lightweight correction factor:

```
y_hat = y_hat_base * b_hat
```

where `b_hat` is the predicted correction factor estimated from ranking features, the base prediction, and auxiliary engagement signals.

### Key Components

1. **Dynamic Distribution-Aware Transformation** — Applies a group-specific Box-Cox transformation to stabilize long-tailed multiplicative correction targets, together with transformed-space and moment regularization losses.

2. **Debias-Factor-Aware Correction** — Assigns each sample to a duration group and selects the corresponding expert through hard one-hot routing, matching the manuscript definition.

3. **Multi-Label-Aware Representation** — Applies fixed nonlinear projections to auxiliary logits, concatenates them with auxiliary tower representations and the shared correction context, and uses an MLP to estimate correction factors.

### Two-Stage Training

- **Stage 1 (Warmup)**: Train only the base model to convergence.
- **Stage 2 (Correction)**: Freeze the first-stage predictor and optimize the DADF correction module using transformed-space fitting, absolute-time loss, moment regularization, and auxiliary-task losses.

### Supported Base Models

| Model | Description |
|-------|-------------|
| `wlr` | Weighted Logistic Regression (Wide&Deep + WBCE) |
| `vr`  | Vanilla Regression (Wide&Deep + MSE) |
| `egmn`| Exponential-Gaussian Mixture Network |
| `tpm` | Tree Probability Model |
| `d2q` | Duration-to-Quantile regression |
| `cread`| Cumulative Regression with Ordinal Decoding |
| `d2co`| Duration-to-Conditional Output (GMM mapping) |

## Repository Structure

```
DADF/
├── model/
│   ├── __init__.py
│   ├── wd.py                   # Wide&Deep base model
│   ├── egmn.py                 # EGMN model
│   ├── cread.py                # CREAD model
│   ├── d2q.py                  # D2Q model
│   ├── tpm.py                  # TPM model
│   ├── layers.py               # Shared network layers
│   ├── framework_utils.py      # Shared utility functions
│   └── v2_debias/
│       ├── __init__.py
│       ├── train.py            # Main DADF training script
│       ├── network.py          # DebiasNetV2 architecture
│       ├── adapter.py          # Base model adapter layer
│       ├── transforms.py       # Box-Cox transform utilities
│       └── losses.py           # DADF loss functions
├── dataloader/
│   ├── __init__.py
│   ├── kuairec.py              # KuaiRec data loader
│   └── wechat21.py             # WeChat21 data loader
├── dataset/
│   ├── kuairec/
│   │   ├── kuairec_process.py  # KuaiRec preprocessing script
│   │   └── raw_data/           # Place raw KuaiRec files here
│   └── wechat21/
│       ├── wechat21_process.py # WeChat21 preprocessing script
│       └── raw_data/           # Place raw WeChat21 files here
├── utils.py                    # Evaluation metrics
├── logger.py                   # Logging utilities
├── run_DADF_wlr.sh             # Run DADF on WLR backbone
└── README.md
```

## Setup

### Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn
```

Tested with Python 3.8+, PyTorch 1.12+.

### Datasets

#### KuaiRec

KuaiRec is a fully-observed recommendation dataset from Kuaishou short-video platform.

1. Download from the official source: [KuaiRec](https://kuairec.com/)
2. Place the following files under `dataset/kuairec/raw_data/`:
   - `big_matrix.csv` — main interaction records
   - `user_features.csv` — user features
   - `item_daily_features.csv` — item features
   - `item_categories.csv` — item category features
   - `kuairec_caption_category.csv` — item caption categories
3. Run preprocessing:

```bash
cd dataset/kuairec
python kuairec_process.py
```

This generates `kuairec_data.pkl` (10% sample) and `kuairec_data_full.pkl` (full data) in `dataset/kuairec/`.

#### WeChat21

WeChat21 is from the WeChatBigData Challenge 2021.

1. Download from: [WeChatBigData Challenge 2021](https://algo.weixin.qq.com/)
2. Place the following files under `dataset/wechat21/raw_data/`:
   - `user_action.csv` — user interaction records
   - `feed_info.csv` — video metadata
3. Run preprocessing:

```bash
cd dataset/wechat21
python wechat21_process.py
```

This generates `wechat21_data.pkl` (10% sample) and `wechat21_data_full.pkl` (full data) in `dataset/wechat21/`.

**Preprocessing details:**
- Feature vocabularies are shared across splits.
- Train/validation/test examples use a fixed-seed random 80% / 10% / 10% split.
- Label-derived normalization statistics (max value and duration buckets) are computed on the training set only.
- Duration bucket quantiles for D2Q are computed on the training set only.

## Running DADF

### Quick Start: DADF on WLR Backbone

```bash
# Run on both KuaiRec and WeChat21 (parallel)
bash run_DADF_wlr.sh

# Run on KuaiRec only
DATASET=kuairec bash run_DADF_wlr.sh

# Run on WeChat21 only
DATASET=wechat21 bash run_DADF_wlr.sh

# Specify GPU
DEVICE=cuda:1 bash run_DADF_wlr.sh

# Sequential mode (useful for debugging)
SEQUENTIAL=1 DATASET=kuairec bash run_DADF_wlr.sh
```

### Manual Training

```bash
# DADF + WLR on KuaiRec (K=4 duration buckets, quantile thresholds)
python model/v2_debias/train.py \
    --base_model wlr \
    --dataset_name kuairec \
    --dataset_path dataset \
    --full-data \
    --two_stage_debias \
    --debias_bucket_num 4 \
    --duration_thresh_mode quantile \
    --epoch 25 \
    --warmup_epoch 3 \
    --patience 6 \
    --base_lr 0.1 \
    --debias_lr 0.02 \
    --weight_decay 1e-6 \
    --abs_time_weight 0.8 \
    --nr_weight 0.05 \
    --use_aux_targets \
    --aux_targets svr,fpr,evr,lvr,evr_p60,lvr_p80,lvr_p90 \
    --aux_target_weight 0.10 \
    --device cuda:0

# DADF + WLR on WeChat21 (K=3 duration buckets)
python model/v2_debias/train.py \
    --base_model wlr \
    --dataset_name wechat21 \
    --dataset_path dataset \
    --full-data \
    --two_stage_debias \
    --debias_bucket_num 3 \
    --duration_thresh_mode quantile \
    --epoch 30 \
    --warmup_epoch 3 \
    --patience 6 \
    --base_lr 0.1 \
    --debias_lr 0.01 \
    --weight_decay 1e-6 \
    --abs_time_weight 0.8 \
    --nr_weight 0.05 \
    --use_aux_targets \
    --aux_targets svr,fpr,evr,lvr,evr_p60,lvr_p80,lvr_p90 \
    --aux_target_weight 0.10 \
    --device cuda:0
```

### Using Other Base Models

Replace `--base_model wlr` with any of: `vr`, `egmn`, `tpm`, `d2q`, `cread`.

```bash
# DADF + EGMN on KuaiRec
python model/v2_debias/train.py \
    --base_model egmn \
    --dataset_name kuairec \
    --dataset_path dataset \
    --full-data \
    --two_stage_debias \
    --debias_bucket_num 4 \
    --duration_thresh_mode quantile \
    --device cuda:0
```

## Evaluation Metrics

DADF is evaluated on the following metrics:

| Metric | Description |
|--------|-------------|
| **MAE** (seconds) | Mean Absolute Error in watch-time prediction |
| **XAUC** | Strict order agreement over all unordered sample pairs; label or prediction ties receive zero credit |

## Main Results

The values below are copied from the current manuscript table so that the repository and paper use one reporting source of truth. MAE is in seconds; lower is better. Higher XAUC is better.

| Backbone | Method | KuaiRec MAE | KuaiRec XAUC | WeChat21 MAE | WeChat21 XAUC |
|---|---|---:|---:|---:|---:|
| VR | Base | 4.584 | 0.5578 | 18.681 | 0.6766 |
| VR | w/ TranSUN | 4.478 | 0.5693 | 18.571 | 0.6787 |
| VR | w/ DADF | **4.235** | **0.6125** | **17.912** | **0.6902** |
| WLR | Base | 4.414 | 0.5941 | 18.215 | 0.6861 |
| WLR | w/ TranSUN | 4.364 | 0.5965 | 18.133 | 0.6876 |
| WLR | w/ DADF | **4.172** | **0.6227** | **17.838** | **0.6934** |
| TPM | Base | 4.459 | 0.5495 | 19.545 | 0.6570 |
| TPM | w/ TranSUN | 4.361 | 0.5971 | 18.529 | 0.6814 |
| TPM | w/ DADF | **4.166** | **0.6233** | **18.109** | **0.6898** |
| D2Q | Base | 4.123 | 0.6319 | 17.544 | 0.6935 |
| D2Q | w/ TranSUN | 4.323 | 0.6082 | 17.855 | 0.6925 |
| D2Q | w/ DADF | **4.106** | **0.6345** | **17.534** | **0.6946** |
| CREAD | Base | 4.346 | 0.5927 | 19.128 | 0.6679 |
| CREAD | w/ TranSUN | 4.395 | 0.5958 | 18.515 | 0.6824 |
| CREAD | w/ DADF | **4.189** | **0.6211** | **18.164** | **0.6903** |
| D²CO | Base | 4.613 | 0.5687 | 18.558 | 0.6861 |
| D²CO | w/ TranSUN | 4.300 | 0.6097 | 18.080 | 0.6868 |
| D²CO | w/ DADF | **4.168** | **0.6233** | **17.683** | **0.6952** |
| EGMN | Base | 4.081 | 0.6245 | 18.330 | 0.6896 |
| EGMN | w/ TranSUN | 4.255 | 0.6120 | 18.099 | 0.6892 |
| EGMN | w/ DADF | **4.002** | **0.6257** | **17.955** | **0.6911** |

The study repeated offline comparisons with matched random seeds to check stability. The compact tables retain the manuscript point estimates rather than per-run variance; reproducibility studies should use the same seed list for paired methods and compute uncertainty from the resulting logs.

### Baseline Quality and Matched Comparison

Each reproduced first-stage backbone was tuned on the validation split and evaluated at a competitive operating point before correction. All backbones use the same feature preprocessing and data split, 16-dimensional sparse-feature embeddings, and comparable MLP capacity (hidden dimensions 256, 128, and 64). In the reported comparison, Base, w/ TranSUN, and w/ DADF within each backbone group start from the same trained first-stage checkpoint, which remains frozen during correction training. Differences within a backbone group therefore reflect the correction method rather than base-model quality.

As an external sanity check, the original [RecSys 2025 EGMN paper](https://arxiv.org/pdf/2508.12665) reports EGMN results of 4.204 MAE / 0.6093 XAUC on KuaiRec and 18.88 MAE / 0.6692 XAUC on WeChat. Our tuned EGMN baseline reaches 4.081 / 0.6245 and 18.330 / 0.6896, respectively. Thus, the DADF comparison starts from an EGMN baseline that is stronger on both reported metrics, rather than from a degraded reproduction. Because preprocessing details and evaluation splits may differ between repositories, this cross-paper comparison should be treated as a reference sanity check, not as a strictly controlled head-to-head experiment.

### WLR Ablation

| Variant | KuaiRec MAE | KuaiRec XAUC | WeChat21 MAE | WeChat21 XAUC |
|---|---:|---:|---:|---:|
| Full DADF | 4.1723 | 0.6227 | 17.8376 | 0.6934 |
| w/o Distribution-Aware Transformation | 4.1901 | 0.6210 | 17.8748 | 0.6930 |
| w/o Debias-Factor-Aware Correction | 4.1823 | 0.6212 | 17.8454 | 0.6931 |
| w/o Multi-Label-Aware Representation | 4.1865 | 0.6204 | 17.9137 | 0.6920 |

## Key Hyperparameters

| Parameter | Description | KuaiRec | WeChat21 |
|-----------|-------------|---------|---------|
| `--debias_bucket_num` | Number of duration expert buckets | 4 | 3 |
| `--duration_thresh_mode` | Bucket threshold mode (`quantile`/`physical`) | quantile | quantile |
| `--warmup_epoch` | Warmup epochs (base only) | 3 | 3 |
| `--epoch` | Joint training epochs | 25 | 30 |
| `--debias_lr` | DADF module learning rate | 0.02 | 0.01 |
| `--nr_weight` | Normal regularization loss weight | 0.05 | 0.05 |
| `--abs_time_weight` | Absolute time Huber loss weight | 0.8 | 0.8 |
| `--aux_target_weight` | Auxiliary target loss weight | 0.10 | 0.10 |

Hard duration routing, equal-frequency buckets, auxiliary heads, and a frozen first-stage predictor are defaults. `--soft_routing`, `--joint_finetune_base`, `--nr_pred_weight`, `--lambda_smooth_weight`, `--kurtosis_weight`, `--bucket_reweighting`, and `--backbone_autotune` are explicit experimental options and do not affect the manuscript-aligned default.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yang2026dadf,
  title     = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author    = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang},
  year      = {2026},
  note      = {Manuscript}
}
```

## License

This project is released under the MIT License.
