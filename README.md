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

The repository focuses on the reproducible research path. Dataset files and deployment-specific infrastructure are not distributed here. The public auxiliary heads are trained from labels available in the two public datasets and follow the same auxiliary representation design.

## Experimental Setup

The experiments use a consistent first-stage model and correction setup:

- all backbones use the same feature preprocessing and the same 80%/10%/10% data split;
- sparse feature embeddings use 16 dimensions;
- backbone MLP widths are 256, 128, and 64 where applicable;
- each first-stage backbone is selected on the validation split before correction training;
- Base and DADF within a backbone group start from the same first-stage checkpoint;
- the first-stage checkpoint is frozen during DADF correction training;
- matched runs use the same split and random seed.

These settings keep model capacity and data treatment consistent across methods within each backbone group.

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

## Default Configuration

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

Run the standard WLR+DADF configuration on KuaiRec:

```bash
BASE_MODEL=wlr MODE=dadf DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
```

Select any supported backbone with `BASE_MODEL`:

```bash
BASE_MODEL=egmn MODE=dadf DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
BASE_MODEL=cread MODE=dadf DATASET=wechat21 DEVICE=cuda:0 bash run_DADF.sh
```

Run a backbone without constructing or training DADF:

```bash
BASE_MODEL=wlr MODE=base DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
```

Run all seven backbones concurrently in the background. This launcher hardcodes
`MODE=base`, so it does not construct or train DADF. By default, it uses the
original `256 128 64` backbone widths and distributes the seven jobs round-robin
across two GPUs:

```bash
bash run_all_backbone.sh
```

The default assignment is `vr/tpm/cread/egmn` on `cuda:0` and
`wlr/d2q/d2co` on `cuda:1`. All seven jobs start immediately rather than being
queued. Each job runs for at most 100 epochs and stops when validation XAUC has
not improved for 6 consecutive epochs. Logs and process IDs are written to:

```text
logs/all_backbones_<timestamp>/base_earlystop_<backbone>.log
logs/all_backbones_<timestamp>/backbone_pids.txt
```

The GPU list, training limit, patience, and other shared settings can be
overridden without editing the script:

```bash
DEVICES="cuda:0 cuda:1" BASE_EPOCH=100 PATIENCE=6 \
  bash run_all_backbone.sh
```

Run the dense-capacity control for all seven backbones with:

```bash
CAPACITY_MATCHED=1 bash run_all_backbone.sh
```

This remains a backbone-only experiment but selects the matched MLP dimensions
for each model automatically: `354 128 64` for VR, WLR, D2CO, and EGMN, and
`342 128 64` for TPM, D2Q, and CREAD. Its logs are isolated under
`logs/all_backbones_capacity_matched_<timestamp>/capacity_matched_<backbone>.log`.

Monitor a launch with:

```bash
RUN_DIR=$(ls -dt logs/all_backbones_* | head -1)
cat "${RUN_DIR}/backbone_pids.txt"
tail -f "${RUN_DIR}/base_earlystop_wlr.log"
watch -n 2 nvidia-smi
```

With full-data mode, every process materializes its own CPU copy of the
processed dataset. Check host memory before using the concurrent launcher; use
`run_DADF.sh` separately if the machine cannot hold seven copies.

Override the backbone MLP dimensions with a space-separated list:

```bash
BASE_MODEL=wlr MODE=base BASE_MLP_DIMS="354 128 64" \
  DATASET=kuairec DEVICE=cuda:0 bash run_DADF.sh
```

The direct training entry point provides the same controls:

```bash
python model/dadf/train.py --help
python model/dadf/train.py --base_model egmn --base_mlp_dims 256 128 64 \
  --dataset_name kuairec --full-data --device cuda:0
python model/dadf/train.py --base_model egmn --base_only --base_epoch 30 \
  --base_mlp_dims 354 128 64 --dataset_name kuairec --full-data --device cuda:0
```

### Dense-Capacity Control

The training entry point reports unique total parameters and dense parameters.
Dense parameters exclude embedding tables. Under the default KuaiRec feature
schema and DADF configuration, the following enlarged first-layer dimensions
match the dense capacity of the corresponding backbone+DADF model within 0.1%:

| Backbone | Backbone dense | Backbone+DADF dense | Matched `BASE_MLP_DIMS` | Matched dense |
|---|---:|---:|---|---:|
| VR | 185,730 | 253,649 | `354 128 64` | 253,448 |
| WLR | 185,730 | 253,649 | `354 128 64` | 253,448 |
| TPM | 187,936 | 247,663 | `342 128 64` | 247,448 |
| D2Q | 185,986 | 245,713 | `342 128 64` | 245,498 |
| CREAD | 294,771 | 354,498 | `342 128 64` | 354,283 |
| D2CO | 185,730 | 253,649 | `354 128 64` | 253,448 |
| EGMN | 188,001 | 255,920 | `354 128 64` | 255,817 |

The exact count printed by a run is authoritative because preprocessing may
drop constant fields. Use the same split, seed, optimization budget, and
validation protocol when comparing a capacity-matched backbone with DADF.

#### Why control for backbone capacity?

DADF introduces additional dense parameters through its correction module. A
natural question is whether its gains come from the proposed debiasing design
or merely from increased model capacity. We therefore construct a strict
capacity control that removes DADF entirely and enlarges only the backbone MLP
until its dense parameter count matches Backbone+DADF within 0.1%.

Crucially, the original backbone, the capacity-matched backbone, and DADF use
exactly the same preprocessed examples, train/validation/test split, and raw
input feature fields. DADF introduces no external dataset or additional
inference-time feature field; its auxiliary targets are deterministically
constructed from labels already present in the same training records. The
capacity control therefore holds both information access and model size nearly
constant: it uses the same data and features as DADF while matching its dense
parameter count within 0.1%. The remaining distinction is how DADF structures
and supervises that capacity for distribution-aware debiasing.

For each random seed, run the default-capacity and capacity-matched backbones
with the same data split, optimization budget, and validation-XAUC checkpoint
selection protocol:

```bash
# Original-capacity backbones
SEED=42 CAPACITY_MATCHED=0 DEVICES="cuda:0 cuda:1" \
  BASE_EPOCH=100 PATIENCE=6 bash run_all_backbone.sh

# Capacity-matched backbones; run after the jobs above have finished
SEED=42 CAPACITY_MATCHED=1 DEVICES="cuda:0 cuda:1" \
  BASE_EPOCH=100 PATIENCE=6 bash run_all_backbone.sh
```

Replace `42` with each seed used in the experiment and repeat this paired
protocol for all 10 random seeds. Because
`run_all_backbone.sh` starts seven background jobs and returns immediately,
wait for each launch to finish before starting the next seed.

#### Ten-seed capacity-control results

The table reports test XAUC averaged over 10 random seeds after restoring the
checkpoint with the best validation XAUC for each run. `Delta XAUC` is
capacity-matched minus original, so a positive value favors the larger
backbone.

| Backbone | Original dense | Matched dense | Parameter increase | Original XAUC | Matched XAUC | Delta XAUC |
|---|---:|---:|---:|---:|---:|---:|
| VR | 185,730 | 253,448 | +36.46% | **0.5612** | 0.5583 | -0.0029 |
| WLR | 185,730 | 253,448 | +36.46% | **0.5947** | 0.5807 | -0.0140 |
| TPM | 187,936 | 247,448 | +31.67% | 0.5517 | **0.5534** | +0.0017 |
| D2Q | 185,986 | 245,498 | +32.00% | **0.6317** | 0.6316 | -0.0001 |
| CREAD | 294,771 | 354,283 | +20.19% | 0.5954 | **0.6013** | +0.0059 |
| D2CO | 185,730 | 253,448 | +36.46% | **0.5708** | 0.5706 | -0.0002 |
| EGMN | 188,001 | 255,817 | +36.07% | 0.6263 | **0.6268** | +0.0005 |
| **Average** | **201,983** | **266,199** | **+31.79%** | **0.5903** | **0.5890** | **-0.0013** |

Increasing dense capacity by 31.79% on average does not yield consistent
improvements: mean XAUC decreases by 0.0013, and four of the seven backbones do
not improve. The capacity-matched backbone uses the same training records and
raw features as DADF and has nearly the same dense parameter count, yet adding
this capacity to the backbone alone does not reproduce DADF's gains. Therefore,
the gains cannot be attributed to extra data, extra features, or parameter count
alone; they depend on how DADF structures and trains that capacity for
distribution-aware debiasing.

## Evaluation and Reported Results

The runner reports:

- **MAE**, where lower is better;
- **XAUC**, a strict pairwise ordering agreement metric, where higher is better.

Across 14 backbone-dataset settings, DADF reduces MAE by **4.33%** and improves XAUC by **4.01%** on average relative to the corresponding frozen backbone. Per-run seeds and metrics are written to the local log directory.

### EGMN Reference

The original [EGMN paper](https://arxiv.org/pdf/2508.12665) provides a useful reference point for the reproduced EGMN backbone:

| Source | KuaiRec MAE | KuaiRec XAUC | WeChat MAE | WeChat XAUC |
|---|---:|---:|---:|---:|
| Original EGMN | 4.204 | 0.6093 | 18.880 | 0.6692 |
| EGMN baseline in this repository | 4.081 | 0.6245 | 18.330 | 0.6896 |

The reproduced EGMN reaches a competitive operating point on both datasets. Since preprocessing and evaluation splits may differ between repositories, these values are a reference rather than a controlled head-to-head comparison.

The WLR ablation includes the three core components:

| Variant | Removed component |
|---|---|
| `w/o Dist.` | Regime-specific target transformation |
| `w/o Factor` | Duration-indexed routing, replaced by a shared correction mapping |
| `w/o Aux.` | Auxiliary behavioral representation |

Removing any of the three components reduces MAE/XAUC performance on both public datasets. The code exposes `--shared_correction` for the routing ablation and `--no_aux_targets` for the auxiliary ablation.

## Repository Layout

```text
model/dadf/        DADF network, transforms, losses, adapters, and training
model/             First-stage predictors and shared layers
dataloader/        Dataset loaders
dataset/           Public-dataset preprocessing
tests/             Public method-contract tests
run_DADF.sh        Generic backbone and DADF experiment launcher
run_all_backbone.sh Concurrent backbone-only launcher
```

## Citation

```bibtex
@misc{yang2026dadf,
  title  = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang and Kun Gai},
  year   = {2026}
}
```

## License

MIT
