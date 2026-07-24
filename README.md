# DADF: Distribution-Aware Debiasing for Watch-Time Regression

Reference implementation for **DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems**.

[中文说明](README_zh.md)

## Overview

DADF is a lightweight second-stage correction framework for watch-time regression. It keeps a trained first-stage predictor frozen and estimates the inference-time-predictable part of its conditional residual. The corrected prediction preserves the original scalar interface:

```text
corrected_watch_time = base_watch_time * correction_factor
```

Here, debiasing means correcting predictable conditional residuals rather than globally recalibrating the first-stage model. Video duration indexes heterogeneous regimes; it is not treated as the sole cause of prediction error.

## Method

The implementation follows the three components in the paper:

1. **Regime-Specific Target Transformation** applies a learnable group-level Box-Cox transform to the multiplicative correction target.
2. **Duration-Indexed Expert Routing** assigns each sample to a duration regime and uses deterministic hard routing to evaluate one correction expert.
3. **Auxiliary Behavioral Representation** expands auxiliary logits with fixed nonlinear responses and fuses them with auxiliary tower states and the shared correction context.

The correction objective contains four terms:

- transformed-space MSE;
- original watch-time Huber loss;
- regime-level moment regularization;
- auxiliary-task BCE loss.

The first-stage prediction is detached when the correction target is constructed. Auxiliary labels provide training supervision only; inference uses predicted auxiliary logits, tower representations, and other inference-time features.

## Public Implementation Scope

This repository provides the public-benchmark implementation used to study the DADF architecture on KuaiRec and WeChat21. It includes data preprocessing, seven first-stage predictors, the DADF correction module, training, and MAE/XAUC evaluation.

The repository focuses on the reproducible research path. Dataset files and deployment-specific infrastructure are not distributed here. The public auxiliary heads are trained from labels available in the two public datasets and implement the same representation pattern described in the paper.

## Fair Comparison Protocol

The offline comparison controls the first-stage model and correction protocol:

- all backbones use the same feature preprocessing and the same 80%/10%/10% data split;
- sparse feature embeddings use 16 dimensions;
- backbone MLP widths are 256, 128, and 64 where applicable;
- each first-stage backbone is selected on the validation split before correction training;
- Base and DADF within a backbone group start from the same first-stage checkpoint;
- the first-stage checkpoint is frozen during DADF correction training;
- matched runs use the same split and random seed.

This setup keeps model capacity and data treatment comparable, so the within-backbone difference reflects the correction stage rather than a weaker base model.

## Supported First-Stage Predictors

| Name | First-stage formulation |
|---|---|
| `vr` | Direct value regression |
| `wlr` | Weighted logistic regression |
| `tpm` | Tree-based ordinal watch-time modeling |
| `d2q` | Duration-aware quantile regression |
| `cread` | Error-adaptive discretization and restoration |
| `d2co` | Duration-related component correction |
| `egmn` | Exponential-Gaussian mixture modeling |

## Manuscript-Aligned Defaults

| Setting | KuaiRec | WeChat21 |
|---|---:|---:|
| Duration regimes | 4 | 3 |
| Regime construction | Equal-frequency | Equal-frequency |
| Batch size | 2048 | 2048 |
| Correction hidden size | 64 | 64 |
| Maximum correction epochs | 25 | 30 |
| Early-stopping patience | 6 | 6 |

Common defaults:

- hard duration routing;
- group-level learnable Box-Cox parameters;
- uniform regime weighting;
- frozen first-stage predictor;
- seven auxiliary watch-time targets;
- validation XAUC for early stopping;
- loss weights `(transformed, absolute, regularization, auxiliary) = (1.0, 0.8, 0.05, 0.10)`.

The inverse path clips the transformed prediction and recovered correction factor to the numerical ranges reported in the paper.

## Installation

Requirements:

- Python 3.8+
- PyTorch 1.12+

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the datasets from their official sources:

- [KuaiRec](https://kuairec.com/)
- [WeChatBigData Challenge 2021](https://algo.weixin.qq.com/)

Follow [`dataset/README.md`](dataset/README.md) to place and preprocess the raw files. Raw and processed dataset artifacts are excluded from version control.

## Running DADF

Run the manuscript-aligned WLR configuration on KuaiRec:

```bash
DATASET=kuairec DEVICE=cuda:0 bash run_DADF_wlr.sh
```

Run on WeChat21 or both datasets:

```bash
DATASET=wechat21 DEVICE=cuda:0 bash run_DADF_wlr.sh
DATASET=all DEVICE=cuda:0 bash run_DADF_wlr.sh
```

The direct training entry point supports all first-stage predictors:

```bash
python model/dadf/train.py --help
python model/dadf/train.py --base_model egmn --dataset_name kuairec --full-data --device cuda:0
```

## Evaluation and Reported Results

The runner reports:

- **MAE**, where lower is better;
- **XAUC**, the strict pairwise ordering agreement used in the paper, where higher is better.

Across the 14 backbone-dataset settings reported in the manuscript, DADF reduces MAE by **4.33%** and improves XAUC by **4.01%** on average relative to the corresponding frozen backbone. Per-run seeds and metrics are written to the local log directory.

The WLR ablation evaluates the three paper components:

| Variant | Removed component |
|---|---|
| `w/o Dist.` | Regime-specific target transformation |
| `w/o Factor` | Duration-indexed routing, replaced by a shared correction mapping |
| `w/o Aux.` | Auxiliary behavioral representation |

All three removals reduce MAE/XAUC performance on both public datasets in the reported study. The code exposes `--shared_correction` for the routing ablation and `--no_aux_targets` for the auxiliary ablation.

## Repository Layout

```text
model/dadf/        DADF network, transforms, losses, adapters, and training
model/             First-stage predictors and shared layers
dataloader/        Dataset loaders
dataset/           Public-dataset preprocessing
tests/             Public method-contract tests
run_DADF_wlr.sh    WLR experiment entry point
```

## Citation

```bibtex
@misc{yang2026dadf,
  title  = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang and Kun Gai},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

MIT
