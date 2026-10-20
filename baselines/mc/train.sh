#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# GPU and conda environment
export CUDA_VISIBLE_DEVICES=2
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="heart"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# Keep the JAX simulator on CPU so it does not reserve the learner's GPU memory.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1

# Training settings: each iteration collects NUM_ENVS complete matches.
SEEDS=(1)
ITERATIONS=1000
NUM_ENVS=16
BATCH_SIZE=256
HIDDEN=256
LR=0.0003
EPSILON=0.1
EVAL_SEED=1000000
EVAL_EVERY=25

# Set a last.pt path to resume; keep one seed and the original training settings.
RESUME=""
OUTPUT_ROOT=".local_training/mc"
LOG_ROOT="logs"
LOG_DATE="$(date +%m%d)"
LOG_DIR="${LOG_ROOT}/${LOG_DATE}"
mkdir -p "${LOG_DIR}"

if [[ -n "${RESUME}" && "${#SEEDS[@]}" -ne 1 ]]; then
    echo "RESUME requires exactly one seed." >&2
    exit 2
fi

run_job() {
    local JOB_NAME="$1"
    shift
    local CMD=("$@")
    local OUT_FILE="${LOG_DIR}/${JOB_NAME}.out"
    local LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"

    echo "Starting job: ${JOB_NAME}" | tee -a "${LOG_FILE}"
    printf 'Command:' | tee -a "${LOG_FILE}"
    printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
    printf '\n' | tee -a "${LOG_FILE}"

    nohup "${CMD[@]}" < /dev/null \
        > >(tee -a "${OUT_FILE}") \
        2> >(tee -a "${LOG_FILE}" >&2) &

    local PID=$!
    echo "  -> PID: ${PID}" | tee -a "${LOG_FILE}"

    if wait "${PID}"; then
        echo "Completed job: ${JOB_NAME}" | tee -a "${LOG_FILE}"
    else
        local STATUS=$?
        echo "Failed job: ${JOB_NAME} (exit ${STATUS})" | tee -a "${LOG_FILE}" >&2
        return "${STATUS}"
    fi
}

for SEED in "${SEEDS[@]}"; do
    TIMESTAMP="$(date +%Y%m%d_%H%M%S_%N)"
    JOB_NAME="Heart_${TIMESTAMP}_mc_medium_seed${SEED}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${JOB_NAME}"
    RESUME_ARGS=()
    if [[ -n "${RESUME}" ]]; then
        OUTPUT_DIR="$(dirname -- "${RESUME}")"
        RESUME_ARGS=(--resume "${RESUME}")
    fi

    run_job "${JOB_NAME}" \
        python -u -m baselines.mc.train \
        --device cuda \
        --seed "${SEED}" \
        --iterations "${ITERATIONS}" \
        --num-envs "${NUM_ENVS}" \
        --batch-size "${BATCH_SIZE}" \
        --hidden "${HIDDEN}" \
        --lr "${LR}" \
        --epsilon "${EPSILON}" \
        --eval-seed "${EVAL_SEED}" \
        --eval-every "${EVAL_EVERY}" \
        --output "${OUTPUT_DIR}" \
        "${RESUME_ARGS[@]}"
done
