#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_DADF.sh"

MODELS=(vr wlr tpm d2q cread d2co egmn)

DEVICES=${DEVICES:-"cuda:0 cuda:1"}
BASE_MLP_DIMS=${BASE_MLP_DIMS:-"256 128 64"}
BASE_EPOCH=${BASE_EPOCH:-100}
PATIENCE=${PATIENCE:-6}
BASE_LR=${BASE_LR:-0.1}
DATASET=${DATASET:-kuairec}
SEED=${SEED:-42}

read -r -a DEVICE_ARRAY <<< "${DEVICES}"
if [[ ${#DEVICE_ARRAY[@]} -eq 0 ]]; then
    echo "ERROR: DEVICES must contain at least one device, e.g. 'cuda:0 cuda:1'."
    exit 2
fi

if [[ ! -x "${RUNNER}" ]]; then
    echo "ERROR: ${RUNNER} is missing or not executable."
    exit 1
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/all_backbones_${RUN_TS}"
PID_FILE="${LOG_DIR}/backbone_pids.txt"
mkdir -p "${LOG_DIR}"

{
    echo "============================================================"
    echo " Backbone-only parallel launcher"
    echo " MODE=base"
    echo " MODELS=${MODELS[*]}"
    echo " DEVICES=${DEVICE_ARRAY[*]}"
    echo " BASE_MLP_DIMS=${BASE_MLP_DIMS}"
    echo " BASE_EPOCH=${BASE_EPOCH}  PATIENCE=${PATIENCE}"
    echo " DATASET=${DATASET}  SEED=${SEED}"
    echo " LOG_DIR=${LOG_DIR}"
    echo "============================================================"
} | tee "${LOG_DIR}/launch_summary.log"

for index in "${!MODELS[@]}"; do
    model="${MODELS[$index]}"
    device="${DEVICE_ARRAY[$((index % ${#DEVICE_ARRAY[@]}))]}"
    log_file="${LOG_DIR}/base_earlystop_${model}.log"

    nohup env -u LD_LIBRARY_PATH \
        BASE_MODEL="${model}" \
        MODE=base \
        BASE_MLP_DIMS="${BASE_MLP_DIMS}" \
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
    printf "%-8s pid=%-8s device=%-8s log=%s\n" \
        "${model}" "${pid}" "${device}" "${log_file}" | tee -a "${PID_FILE}"
done

echo
echo "All ${#MODELS[@]} backbone jobs are running in the background."
echo "PID file: ${PID_FILE}"
echo "Logs:     ${LOG_DIR}/base_earlystop_<backbone>.log"
