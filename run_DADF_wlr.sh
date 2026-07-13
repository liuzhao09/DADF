#!/usr/bin/env bash
# =============================================================================
# run_DADF_wlr.sh — DADF on WLR backbone (KuaiRec + WeChat21)
#
# DADF: Distribution-Aware Debiasing Framework for Watch-Time Regression
# Reference implementation accompanying the DADF manuscript
#
# This script runs DADF with WLR (Weighted Logistic Regression) as the base
# model on both KuaiRec and WeChat21 datasets.
#
# Usage:
#   bash run_DADF_wlr.sh                         # default: both datasets, parallel
#   DATASET=kuairec bash run_DADF_wlr.sh          # KuaiRec only
#   DATASET=wechat21 bash run_DADF_wlr.sh         # WeChat21 only
#   DEVICE=cuda:0 bash run_DADF_wlr.sh            # specify GPU
#   SEQUENTIAL=1 bash run_DADF_wlr.sh             # sequential mode
#   SEED=42 bash run_DADF_wlr.sh                  # set random seed
#
# Estimated wall time: 30-60 min per dataset (depends on GPU)
# =============================================================================

set -u
set -m

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DADF_ENTRY="${SCRIPT_DIR}/model/v2_debias/train.py"
DATASET_PATH="${SCRIPT_DIR}/dataset"

DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
DATASET=${DATASET:-all}
SEQUENTIAL=${SEQUENTIAL:-0}

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/DADF_wlr_${RUN_TS}"
mkdir -p "$LOG_DIR"
SUMMARY="${LOG_DIR}/_summary.txt"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

echo "============================================================" | tee "$SUMMARY"
echo " DADF with WLR backbone"                                       | tee -a "$SUMMARY"
echo " Paper: DADF: A Distribution-Aware Debiasing Framework"        | tee -a "$SUMMARY"
echo "        for Watch-Time Regression in Recommender Systems"      | tee -a "$SUMMARY"
echo " DEVICE=$DEVICE  SEED=$SEED  DATASET=$DATASET"                 | tee -a "$SUMMARY"
echo " log_dir: $LOG_DIR"                                            | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

[ ! -f "${DADF_ENTRY}" ] && { echo "ERROR: ${DADF_ENTRY} not found"; exit 1; }

# ---------------------------------------------------------------------------
# Data pre-check: make sure processed pkl files exist
# ---------------------------------------------------------------------------
check_dataset() {
    local ds=$1
    local pkl="${DATASET_PATH}/${ds}/${ds}_data_full.pkl"
    if [ ! -f "$pkl" ]; then
        echo ""                                                          | tee -a "$SUMMARY"
        echo "  [INFO] ${ds}_data_full.pkl not found. Running preprocessing..."  | tee -a "$SUMMARY"
        local proc="${DATASET_PATH}/${ds}/${ds}_process.py"
        if [ -f "$proc" ]; then
            ( cd "${DATASET_PATH}/${ds}" && python3 "${ds}_process.py" ) \
                2>&1 | tee "${LOG_DIR}/preprocess_${ds}.log"
            if [ $? -ne 0 ]; then
                echo "  [ERROR] Preprocessing failed for ${ds}."         | tee -a "$SUMMARY"
                echo "          Please check dataset/README.md for setup instructions." | tee -a "$SUMMARY"
                exit 1
            fi
        else
            echo "  [ERROR] ${proc} not found."                           | tee -a "$SUMMARY"
            echo "          Please follow dataset/README.md to prepare raw data first." | tee -a "$SUMMARY"
            exit 1
        fi
    else
        echo "  [OK] ${ds}_data_full.pkl found."                         | tee -a "$SUMMARY"
    fi
}

if [[ "$DATASET" == "kuairec" || "$DATASET" == "all" ]]; then
    check_dataset kuairec
fi
if [[ "$DATASET" == "wechat21" || "$DATASET" == "all" ]]; then
    check_dataset wechat21
fi

# ---------------------------------------------------------------------------
# DADF reference configuration (aligned with paper Table 1 main results)
#
# WLR + DADF configuration:
#   - Two-stage training: base warmup + correction training with the base frozen
#   - Box-Cox transformation for distribution normalization
#   - Duration-aware bucket experts (K=4 for KuaiRec, K=3 for WeChat21)
#   - Hard one-hot duration routing (default; matches the manuscript)
#   - Frozen first-stage predictor during correction training (default)
#   - Auxiliary watch-time targets for multi-task learning
#   - Normal regularization loss to enforce Gaussian-like distribution
# ---------------------------------------------------------------------------

DADF_BASE="python3 ${DADF_ENTRY} --dataset_path ${DATASET_PATH} --seed ${SEED}"
AUX_TARGETS="svr,fpr,evr,lvr,evr_p60,lvr_p80,lvr_p90"

# ---------------------------------------------------------------------------
# KuaiRec: WLR + DADF (K=4 quantile buckets, epoch=25)
#   Main table: MAE ~4.172s, XAUC ~0.6227
# ---------------------------------------------------------------------------
CMD_KUAIREC="${DADF_BASE} \
--base_model wlr \
--dataset_name kuairec \
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
--aux_targets ${AUX_TARGETS} \
--aux_target_weight 0.10 \
--device ${DEVICE}"

# ---------------------------------------------------------------------------
# WeChat21: WLR + DADF (K=3 quantile buckets, epoch=30)
#   Main table: MAE ~17.838s, XAUC ~0.6934
# ---------------------------------------------------------------------------
CMD_WECHAT21="${DADF_BASE} \
--base_model wlr \
--dataset_name wechat21 \
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
--aux_targets ${AUX_TARGETS} \
--aux_target_weight 0.10 \
--device ${DEVICE}"

# ---------------------------------------------------------------------------
# Launch runs
# ---------------------------------------------------------------------------
LAUNCHED_PIDS=()

cleanup_signal() {
    echo ""
    echo ">>> Interrupted, terminating DADF runs..."
    for pid in "${LAUNCHED_PIDS[@]:-}"; do
        [[ -z "$pid" || "$pid" == "0" ]] && continue
        kill -TERM -- -"$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${LAUNCHED_PIDS[@]:-}"; do
        [[ -z "$pid" || "$pid" == "0" ]] && continue
        kill -KILL -- -"$pid" 2>/dev/null || true
    done
    exit 130
}
trap cleanup_signal INT TERM

T_START=$(date +%s)
echo "" | tee -a "$SUMMARY"

run_one() {
    local tag=$1
    local cmd=$2
    local log="${LOG_DIR}/${tag}.log"

    echo "  [launch] ${tag} on ${DEVICE}" | tee -a "$SUMMARY"
    echo "           log: ${log}"         | tee -a "$SUMMARY"

    if [[ "$SEQUENTIAL" == "1" ]]; then
        eval "$cmd" > "$log" 2>&1
        echo "  [${tag}] exit=$?" | tee -a "$SUMMARY"
    else
        (
            echo "[${tag}] started $(date)"
            echo "[${tag}] cmd: $cmd"
            echo "---"
            eval "$cmd"
            rc=$?
            echo "---"
            echo "[${tag}] exit=$rc at $(date)"
            exit $rc
        ) > "$log" 2>&1 &
        pid=$!
        LAUNCHED_PIDS+=("$pid")
        sleep 2
    fi
}

if [[ "$DATASET" == "kuairec" || "$DATASET" == "all" ]]; then
    run_one "DADF_wlr_kuairec" "$CMD_KUAIREC"
fi
if [[ "$DATASET" == "wechat21" || "$DATASET" == "all" ]]; then
    run_one "DADF_wlr_wechat21" "$CMD_WECHAT21"
fi

# ---------------------------------------------------------------------------
# Wait for completion
# ---------------------------------------------------------------------------
if [[ "$SEQUENTIAL" != "1" && ${#LAUNCHED_PIDS[@]} -gt 0 ]]; then
    echo "" | tee -a "$SUMMARY"
    echo "  Waiting for ${#LAUNCHED_PIDS[@]} run(s) to complete..." | tee -a "$SUMMARY"

    TAGS=()
    [[ "$DATASET" == "kuairec" || "$DATASET" == "all" ]] && TAGS+=("DADF_wlr_kuairec")
    [[ "$DATASET" == "wechat21" || "$DATASET" == "all" ]] && TAGS+=("DADF_wlr_wechat21")

    OK_CNT=0; FAIL_CNT=0
    for idx in "${!LAUNCHED_PIDS[@]}"; do
        tag=${TAGS[$idx]}
        if wait "${LAUNCHED_PIDS[$idx]}"; then
            echo "  [OK]   ${tag}" | tee -a "$SUMMARY"
            OK_CNT=$((OK_CNT+1))
        else
            echo "  [FAIL] ${tag}" | tee -a "$SUMMARY"
            FAIL_CNT=$((FAIL_CNT+1))
        fi
    done
fi

T_END=$(date +%s)
DUR=$((T_END - T_START))

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo " Done. ok=${OK_CNT:-N/A} fail=${FAIL_CNT:-N/A}"              | tee -a "$SUMMARY"
echo " elapsed=${DUR}s ($((DUR/60))min)"                           | tee -a "$SUMMARY"
echo " log_dir: ${LOG_DIR}"                                        | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# ---------------------------------------------------------------------------
# Quick results summary
# ---------------------------------------------------------------------------
echo "" | tee -a "$SUMMARY"
echo " Results:" | tee -a "$SUMMARY"
printf "  %-25s %-10s %-10s\n" "Run" "MAE(s)" "XAUC" | tee -a "$SUMMARY"
printf "  %-25s %-10s %-10s\n" "-------------------------" "------" "------" | tee -a "$SUMMARY"

for log_file in "${LOG_DIR}"/*.log; do
    [ -f "$log_file" ] || continue
    tag=$(basename "$log_file" .log)
    [[ "$tag" == "_summary" ]] && continue
    final_line=$(grep -E "^test \| MAE:" "$log_file" 2>/dev/null | tail -1)
    if [[ -z "$final_line" ]]; then
        printf "  %-25s %-10s %-10s\n" "$tag" "N/A" "N/A" | tee -a "$SUMMARY"
    else
        mae=$(echo "$final_line"  | sed -E 's/.*MAE: ([0-9.]+) .*/\1/')
        xauc=$(echo "$final_line" | sed -E 's/.*XAUC: ([0-9.]+) .*/\1/')
        printf "  %-25s %-10s %-10s\n" "$tag" "$mae" "$xauc" | tee -a "$SUMMARY"
    fi
done
