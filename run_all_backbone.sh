#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_DADF.sh"

MODELS=(vr wlr tpm d2q cread d2co egmn)

DEVICES=${DEVICES:-"cuda:0 cuda:1"}
BASE_MLP_DIMS=${BASE_MLP_DIMS:-"256 128 64"}
CAPACITY_MATCHED=${CAPACITY_MATCHED:-0}
BASE_EPOCH=${BASE_EPOCH:-100}
PATIENCE=${PATIENCE:-6}
BASE_LR=${BASE_LR:-0.1}
DATASET=${DATASET:-kuairec}
SEED=${SEED:-42}

if [[ "${CAPACITY_MATCHED}" != "0" && "${CAPACITY_MATCHED}" != "1" ]]; then
    echo "ERROR: CAPACITY_MATCHED must be 0 or 1."
    exit 2
fi

read -r -a DEVICE_ARRAY <<< "${DEVICES}"
if [[ ${#DEVICE_ARRAY[@]} -eq 0 ]]; then
    echo "ERROR: DEVICES must contain at least one device, e.g. 'cuda:0 cuda:1'."
    exit 2
fi

if [[ ! -x "${RUNNER}" ]]; then
    echo "ERROR: ${RUNNER} is missing or not executable."
    exit 1
fi

model_mlp_dims() {
    local model=$1
    if [[ "${CAPACITY_MATCHED}" == "0" ]]; then
        echo "${BASE_MLP_DIMS}"
        return
    fi

    case "${model}" in
        vr|wlr|d2co|egmn)
            echo "354 128 64"
            ;;
        tpm|d2q|cread)
            echo "342 128 64"
            ;;
        *)
            echo "ERROR: no capacity-matched dimensions for ${model}" >&2
            return 1
            ;;
    esac
}

RUN_TS=$(date +%Y%m%d_%H%M%S)
if [[ "${CAPACITY_MATCHED}" == "1" ]]; then
    RUN_LABEL="capacity_matched"
    LOG_DIR="${SCRIPT_DIR}/logs/all_backbones_capacity_matched_${RUN_TS}"
else
    RUN_LABEL="base_earlystop"
    LOG_DIR="${SCRIPT_DIR}/logs/all_backbones_${RUN_TS}"
fi
PID_FILE="${LOG_DIR}/backbone_pids.txt"
mkdir -p "${LOG_DIR}"

{
    echo "============================================================"
    echo " Backbone-only parallel launcher"
    echo " MODE=base"
    echo " MODELS=${MODELS[*]}"
    echo " DEVICES=${DEVICE_ARRAY[*]}"
    echo " CAPACITY_MATCHED=${CAPACITY_MATCHED}"
    if [[ "${CAPACITY_MATCHED}" == "1" ]]; then
        echo " BASE_MLP_DIMS=per-backbone capacity-matched dimensions"
    else
        echo " BASE_MLP_DIMS=${BASE_MLP_DIMS}"
    fi
    echo " BASE_EPOCH=${BASE_EPOCH}  PATIENCE=${PATIENCE}"
    echo " DATASET=${DATASET}  SEED=${SEED}"
    echo " LOG_DIR=${LOG_DIR}"
    echo "============================================================"
} | tee "${LOG_DIR}/launch_summary.log"

for index in "${!MODELS[@]}"; do
    model="${MODELS[$index]}"
    device="${DEVICE_ARRAY[$((index % ${#DEVICE_ARRAY[@]}))]}"
    mlp_dims=$(model_mlp_dims "${model}") || exit 1
    log_file="${LOG_DIR}/${RUN_LABEL}_${model}.log"

    nohup env -u LD_LIBRARY_PATH \
        BASE_MODEL="${model}" \
        MODE=base \
        BASE_MLP_DIMS="${mlp_dims}" \
        BASE_EPOCH="${BASE_EPOCH}" \
        PATIENCE="${PATIENCE}" \
        BASE_LR="${BASE_LR}" \
        BACKBONE_AUTOTUNE=0 \
        DATASET="${DATASET}" \
        DEVICE="${device}" \
        SEED="${SEED}" \
        SEQUENTIAL=1 \
        bash "${RUNNER}" \
        > "${log_file}" 2>&1 < /dev/null &

    pid=$!
    printf "%-8s pid=%-8s device=%-8s mlp=%-14s log=%s\n" \
        "${model}" "${pid}" "${device}" "${mlp_dims}" "${log_file}" \
        | tee -a "${PID_FILE}"
done

echo
echo "All ${#MODELS[@]} backbone jobs are running in the background."
echo "PID file: ${PID_FILE}"
echo "Logs:     ${LOG_DIR}/${RUN_LABEL}_<backbone>.log"
