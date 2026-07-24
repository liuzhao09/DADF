# DADF: Distribution-Aware Debiasing for Watch-Time Regression

Reference implementation for **DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems**.

## Method

DADF applies a second-stage multiplicative correction to a frozen watch-time predictor:

\[
\hat{Y}=\hat{Y}_0\hat{B}.
\]

The implementation contains the three components described in the paper:

1. **Regime-Specific Target Transformation** uses a learnable group-level Box–Cox transform for the correction target.
2. **Duration-Indexed Expert Routing** uses deterministic hard routing to select one correction expert per sample.
3. **Auxiliary Behavioral Representation** fuses auxiliary logits and tower representations with the shared correction context.

Training combines transformed-space MSE, original-space Huber loss, transformation moment regularization, and auxiliary BCE loss. Auxiliary labels are training targets only; inference uses model outputs and inference-time features.

## Requirements

- Python 3.8+
- PyTorch 1.12+

```bash
pip install -r requirements.txt
```

## Data

Download KuaiRec or WeChat21 from their official sources and follow [`dataset/README.md`](dataset/README.md). Raw and processed data are excluded from version control.

## Run

The standard WLR experiment is:

```bash
DATASET=kuairec DEVICE=cuda:0 bash run_DADF_wlr.sh
```

Use `DATASET=wechat21` for WeChat21 or `DATASET=all` for both datasets. The direct entry point is:

```bash
python model/dadf/train.py --help
```

The manuscript-aligned defaults freeze the first-stage predictor during correction training, use equal-frequency duration regimes, group-level transformation parameters, hard routing, auxiliary representations, and validation-XAUC early stopping.

## Supported First-Stage Predictors

`vr`, `wlr`, `tpm`, `d2q`, `cread`, `d2co`, and `egmn`.

## Evaluation

The runner reports MAE and XAUC using the definitions in the paper. Random seeds and per-run metrics are written to the local log directory.

## Citation

```bibtex
@misc{yang2026dadf,
  title  = {DADF: A Distribution-Aware Debiasing Framework for Watch-Time Regression in Recommender Systems},
  author = {Yiqing Yang and Xinlong Zhao and Zhao Liu and Xiao Lv and Ruiming Tang},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

MIT
