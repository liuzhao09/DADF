# DADF: Distribution-Aware Debiasing Framework for Watch-Time Regression

[![RecSys 2026](https://img.shields.io/badge/RecSys-2026-blue)](https://recsys.acm.org/recsys26/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

Official implementation of **DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems** (RecSys 2026).

## Overview

Watch-time prediction is a central regression task in short-video recommender systems. In production, watch-time labels are highly long-tailed, and prediction bias varies systematically across different watch-time intervals. A model may appear globally well-calibrated while severely overestimating short views and underestimating long views — a phenomenon we call **pseudo-balance** caused by error cancellation across intervals.

**DADF** addresses this by performing second-stage multiplicative residual correction on top of any deployed first-stage predictor. Rather than replacing the base model, DADF learns a lightweight correction factor:

```
y_hat = y_hat_base * b_hat
```

where `b_hat` is the predicted correction factor estimated from ranking features, the base prediction, and auxiliary engagement signals.

### Key Components

1. **Box-Cox Transformation** — Transforms the correction target into an approximately Gaussian space, enabling stable regression and variance-aware normalization.

2. **Duration-Aware Bucket Experts** — Multiple expert heads specialized for different video duration intervals, with learned soft routing to capture duration-specific bias patterns.

3. **Normal Regularization Loss** — Enforces the transformed distribution to be approximately Gaussian (zero mean, unit variance, zero skewness, kurtosis ≈ 3) within each bucket, providing distributional constraints during training.

4. **Auxiliary Watch-Time Targets** — Multi-task learning with auxiliary engagement signals (short-view rate, finish-play rate, long-view rate, etc.) providing side information for more accurate correction factor estimation.

### Two-Stage Training

- **Stage 1 (Warmup)**: Train only the base model to convergence.
- **Stage 2 (Joint)**: Fine-tune jointly with the DADF correction module using combined base loss + Box-Cox MSE loss + absolute time Huber loss + normal regularization loss.

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
- Features are encoded with vocabularies built on the full dataset (no data leakage).
- Train/val/test split: 80% / 10% / 10%, stratified by time order.
- Normalization statistics (max value, duration buckets) computed on training set only.
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
| **XAUC** | Ranking quality metric for continuous labels (AUC generalization) |

## Main Results

DADF consistently improves over base models across all 7 backbones on both datasets.

### KuaiRec (MAE in seconds, lower is better)

| Model | Base | + DADF | Improvement |
|-------|------|--------|-------------|
| WLR   | ~5.2 | ~4.17  | -19.8% |
| EGMN  | ~5.0 | ~4.00  | -20.0% |
| D2Q   | ~5.1 | ~4.11  | -19.4% |

### WeChat21 (MAE in seconds, lower is better)

| Model | Base | + DADF | Improvement |
|-------|------|--------|-------------|
| WLR   | ~22  | ~17.84 | -18.9% |
| EGMN  | ~22  | ~17.96 | -18.4% |
| D2Q   | ~23  | ~17.53 | -23.8% |

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

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yang2026dadf,
  title     = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author    = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys)},
  year      = {2026},
  address   = {Minneapolis, MN, USA}
}
```

## License

This project is released under the MIT License.
