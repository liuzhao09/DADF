#!/usr/bin/env bash

set -u
set -m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DADF_ENTRY="${SCRIPT_DIR}/model/dadf/train.py"
DATASET_PATH="${SCRIPT_DIR}/dataset"

BASE_MODEL=${BASE_MODEL:-wlr}
MODE=${MODE:-dadf}
BASE_MLP_DIMS=${BASE_MLP_DIMS:-"256 128 64"}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}
DATASET=${DATASET:-kuairec}
SEQUENTIAL=${SEQUENTIAL:-0}
BASE_EPOCH=${BASE_EPOCH:-30}
BASE_LR=${BASE_LR:-0.1}
PATIENCE=${PATIENCE:-6}
BACKBONE_AUTOTUNE=${BACKBONE_AUTOTUNE:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

SUPPORTED_MODELS="vr wlr tpm d2q cread d2co egmn"
if [[ " ${SUPPORTED_MODELS} " != *" ${BASE_MODEL} "* ]]; then
    echo "ERROR: unsupported BASE_MODEL=${BASE_MODEL}"
    echo "Supported models: ${SUPPORTED_MODELS}"
    exit 2
fi
if [[ "${MODE}" != "base" && "${MODE}" != "dadf" ]]; then
    echo "ERROR: MODE must be 'base' or 'dadf', got '${MODE}'"
    exit 2
fi
if [[ "${DATASET}" != "kuairec" && "${DATASET}" != "wechat21" && "${DATASET}" != "all" ]]; then
    echo "ERROR: DATASET must be 'kuairec', 'wechat21', or 'all'"
    exit 2
fi

read -r -a MLP_DIMS_ARRAY <<< "${BASE_MLP_DIMS}"
read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${MODE}_${BASE_MODEL}_${RUN_TS}"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/_summary.txt"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

{
    echo "============================================================"
    echo " DADF experiment launcher"
    echo " MODE=${MODE}  BASE_MODEL=${BASE_MODEL}"
    echo " BASE_MLP_DIMS=${BASE_MLP_DIMS}"
    echo " BACKBONE_AUTOTUNE=${BACKBONE_AUTOTUNE}"
    echo " DEVICE=${DEVICE}  SEED=${SEED}  DATASET=${DATASET}"
    echo " log_dir: ${LOG_DIR}"
    echo "============================================================"
} | tee "${SUMMARY}"

if [[ ! -f "${DADF_ENTRY}" ]]; then
    echo "ERROR: ${DADF_ENTRY} not found"
    exit 1
fi

check_dataset() {
    local dataset_name=$1
    local data_file="${DATASET_PATH}/${dataset_name}/${dataset_name}_data_full.pkl"
    if [[ -f "${data_file}" ]]; then
        echo "  [OK] ${dataset_name}_data_full.pkl found." | tee -a "${SUMMARY}"
        return
    fi

    echo "  [INFO] ${dataset_name}_data_full.pkl not found. Running preprocessing..." | tee -a "${SUMMARY}"
    local process_script="${DATASET_PATH}/${dataset_name}/${dataset_name}_process.py"
    if [[ ! -f "${process_script}" ]]; then
        echo "  [ERROR] ${process_script} not found." | tee -a "${SUMMARY}"
        exit 1
    fi

    (
        cd "${DATASET_PATH}/${dataset_name}" || exit 1
        python3 "${dataset_name}_process.py"
    ) > "${LOG_DIR}/preprocess_${dataset_name}.log" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "  [ERROR] Preprocessing failed for ${dataset_name}." | tee -a "${SUMMARY}"
        echo "          See ${LOG_DIR}/preprocess_${dataset_name}.log" | tee -a "${SUMMARY}"
        exit 1
    fi
}

if [[ "${DATASET}" == "kuairec" || "${DATASET}" == "all" ]]; then
    check_dataset kuairec
fi
if [[ "${DATASET}" == "wechat21" || "${DATASET}" == "all" ]]; then
    check_dataset wechat21
fi

AUX_TARGETS="svr,fpr,evr,lvr,evr_p60,lvr_p80,lvr_p90"
LAUNCHED_PIDS=()
LAUNCHED_TAGS=()

cleanup_signal() {
    echo
    echo ">>> Interrupted, terminating experiment runs..."
    for pid in "${LAUNCHED_PIDS[@]:-}"; do
        [[ -z "${pid}" || "${pid}" == "0" ]] && continue
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    exit 130
}
trap cleanup_signal INT TERM

build_command() {
    local dataset_name=$1
    COMMAND=(
        python3 "${DADF_ENTRY}"
        --dataset_path "${DATASET_PATH}"
        --seed "${SEED}"
        --base_model "${BASE_MODEL}"
        --base_mlp_dims "${MLP_DIMS_ARRAY[@]}"
        --dataset_name "${dataset_name}"
        --full-data
        --bsz 2048
        --base_lr "${BASE_LR}"
        --weight_decay 1e-6
        --patience "${PATIENCE}"
        --device "${DEVICE}"
    )

    if [[ "${MODE}" == "base" ]]; then
        COMMAND+=(--base_only --base_epoch "${BASE_EPOCH}")
    else
        local bucket_num=4
        local dadf_epoch=30
        local debias_lr=0.01
        if [[ "${dataset_name}" == "kuairec" ]]; then
            bucket_num=4
            if [[ "${BASE_MODEL}" == "wlr" ]]; then
                dadf_epoch=25
                debias_lr=0.02
            else
                dadf_epoch=30
                debias_lr=0.01
            fi
        else
            bucket_num=3
            dadf_epoch=30
            debias_lr=0.01
        fi

        COMMAND+=(
            --debias_bucket_num "${bucket_num}"
            --duration_thresh_mode quantile
            --epoch "${dadf_epoch}"
            --warmup_epoch 3
            --debias_lr "${debias_lr}"
            --abs_time_weight 0.8
            --nr_weight 0.05
            --use_aux_targets
            --aux_targets "${AUX_TARGETS}"
            --aux_target_weight 0.10
        )
        if [[ "${BACKBONE_AUTOTUNE}" == "1" ]]; then
            COMMAND+=(--backbone_autotune)
        fi
    fi

    if [[ ${#EXTRA_ARGS_ARRAY[@]} -gt 0 ]]; then
        COMMAND+=("${EXTRA_ARGS_ARRAY[@]}")
    fi
}

run_one() {
    local dataset_name=$1
    local tag="${MODE}_${BASE_MODEL}_${dataset_name}"
    local log_file="${LOG_DIR}/${tag}.log"
    build_command "${dataset_name}"

    echo "  [launch] ${tag} on ${DEVICE}" | tee -a "${SUMMARY}"
    printf "  [command]" >> "${SUMMARY}"
    printf " %q" "${COMMAND[@]}" >> "${SUMMARY}"
    printf "\n" >> "${SUMMARY}"

    if [[ "${SEQUENTIAL}" == "1" ]]; then
        "${COMMAND[@]}" > "${log_file}" 2>&1
        local exit_code=$?
        echo "  [${tag}] exit=${exit_code}" | tee -a "${SUMMARY}"
        return "${exit_code}"
    fi

    (
        echo "[${tag}] started $(date)"
        printf "[${tag}] command:"
        printf " %q" "${COMMAND[@]}"
        printf "\n---\n"
        "${COMMAND[@]}"
        exit_code=$?
        printf "%s\n" "---"
        echo "[${tag}] exit=${exit_code} at $(date)"
        exit "${exit_code}"
    ) > "${log_file}" 2>&1 &

    LAUNCHED_PIDS+=("$!")
    LAUNCHED_TAGS+=("${tag}")
}

START_TIME=$(date +%s)
OK_COUNT=0
FAIL_COUNT=0
if [[ "${DATASET}" == "kuairec" || "${DATASET}" == "all" ]]; then
    if run_one kuairec; then
        [[ "${SEQUENTIAL}" == "1" ]] && OK_COUNT=$((OK_COUNT + 1))
    else
        [[ "${SEQUENTIAL}" == "1" ]] && FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi
if [[ "${DATASET}" == "wechat21" || "${DATASET}" == "all" ]]; then
    if run_one wechat21; then
        [[ "${SEQUENTIAL}" == "1" ]] && OK_COUNT=$((OK_COUNT + 1))
    else
        [[ "${SEQUENTIAL}" == "1" ]] && FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

if [[ "${SEQUENTIAL}" != "1" ]]; then
    for index in "${!LAUNCHED_PIDS[@]}"; do
        if wait "${LAUNCHED_PIDS[$index]}"; then
            echo "  [OK] ${LAUNCHED_TAGS[$index]}" | tee -a "${SUMMARY}"
            OK_COUNT=$((OK_COUNT + 1))
        else
            echo "  [FAIL] ${LAUNCHED_TAGS[$index]}" | tee -a "${SUMMARY}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
{
    echo
    echo "============================================================"
    echo " Done. ok=${OK_COUNT} fail=${FAIL_COUNT}"
    echo " elapsed=${ELAPSED}s ($((ELAPSED / 60))min)"
    echo " log_dir: ${LOG_DIR}"
    echo "============================================================"
    echo
    printf "  %-30s %-12s %-12s %-12s\n" "Run" "DenseParams" "MAE(s)" "XAUC"
} | tee -a "${SUMMARY}"

for log_file in "${LOG_DIR}"/*.log; do
    [[ -f "${log_file}" ]] || continue
    tag=$(basename "${log_file}" .log)
    parameter_line=$(grep -E '^Parameters \| (backbone\+DADF total|backbone total)=' "${log_file}" | tail -1)
    dense_params=$(echo "${parameter_line}" | sed -E 's/.*dense=([0-9,]+).*/\1/')
    result_line=$(grep -E '^(test base|test) \| MAE:' "${log_file}" | tail -1)
    mae=$(echo "${result_line}" | sed -E 's/.*MAE: ([0-9.]+).*/\1/')
    xauc=$(echo "${result_line}" | sed -E 's/.*XAUC: ([0-9.]+).*/\1/')
    printf "  %-30s %-12s %-12s %-12s\n" \
        "${tag}" "${dense_params:-N/A}" "${mae:-N/A}" "${xauc:-N/A}" | tee -a "${SUMMARY}"
done
