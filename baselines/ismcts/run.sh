#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export CUDA_VISIBLE_DEVICES=2
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="heart"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Fail if CUDA is unavailable; CPU remains available for the real match engine.
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1

SEEDS=(2000000)
ITERATIONS=1024
BATCH_SIZE=32
EXPLORATION=1.4142135623730951
GAMES=96
OPPONENTS=("medium" "medium" "medium")
LOG_DIR="logs/$(date +%m%d)"
mkdir -p "${LOG_DIR}"

for SEED in "${SEEDS[@]}"; do
    JOB_NAME="Heart_$(date +%Y%m%d_%H%M%S_%N)_ismcts_iter${ITERATIONS}_seed${SEED}"
    OUT_FILE="${LOG_DIR}/${JOB_NAME}.out"
    LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"
    CMD=(
        python -u -m baselines.ismcts.evaluate
        --device gpu --iterations "${ITERATIONS}" --batch-size "${BATCH_SIZE}"
        --exploration "${EXPLORATION}" --games "${GAMES}" --seed "${SEED}"
        --opponents "${OPPONENTS[@]}"
    )
    echo "Starting job: ${JOB_NAME}" | tee -a "${LOG_FILE}"
    printf 'Command:' | tee -a "${LOG_FILE}"
    printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
    printf '\n' | tee -a "${LOG_FILE}"
    nohup "${CMD[@]}" < /dev/null \
        > >(tee -a "${OUT_FILE}") \
        2> >(tee -a "${LOG_FILE}" >&2) &
    PID=$!
    echo "  -> PID: ${PID}" | tee -a "${LOG_FILE}"
    if wait "${PID}"; then
        echo "Completed job: ${JOB_NAME}" | tee -a "${LOG_FILE}"
    else
        STATUS=$?
        echo "Failed job: ${JOB_NAME} (exit ${STATUS})" | tee -a "${LOG_FILE}" >&2
        exit "${STATUS}"
    fi
done
