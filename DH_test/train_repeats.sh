#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  DH_test/train_repeats.sh [--gpu ID] [--repeats N] -- [train.py options]

Options:
  --gpu ID          CUDA GPU index (default: 2; use "cpu" for CPU runs)
  --repeats N       Number of independent runs with random seeds (default: 1)
  -h, --help        Show this help

Example:
  DH_test/train_repeats.sh --gpu 1 --repeats 5 -- --steps 100000 --eval-games 20
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GPU_ID="2"
REPEATS=1
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
TRAIN_ARGS=()

while (($#)); do
  case "$1" in
    --gpu)
      (($# >= 2)) || { echo "error: --gpu needs an ID" >&2; exit 2; }
      GPU_ID="$2"
      shift 2
      ;;
    --repeats)
      (($# >= 2)) || { echo "error: --repeats needs a count" >&2; exit 2; }
      REPEATS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_ARGS=("$@")
      break
      ;;
    *)
      echo "error: training options must follow -- (unexpected: $1)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --repeats must be a positive integer" >&2; exit 2; }
if [[ "$GPU_ID" == "cpu" ]]; then
  DEVICE="cpu"
else
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "error: --gpu must be a nonnegative integer or cpu" >&2; exit 2; }
  DEVICE="cuda:${GPU_ID}"
fi

mkdir -p -- "$CHECKPOINT_DIR"
mapfile -t SEEDS < <(python -c 'import secrets, sys; print("\n".join(map(str, secrets.SystemRandom().sample(range(2**31), int(sys.argv[1])))))' "$REPEATS")

for ((i = 0; i < REPEATS; i++)); do
  run_number=$((i + 1))
  echo "[${run_number}/${REPEATS}] GPU=${GPU_ID} seed=${SEEDS[$i]} checkpoint=${CHECKPOINT_DIR}/best.pt"
  (
    cd -- "$PROJECT_ROOT"
    python -m DH_test.train \
      "${TRAIN_ARGS[@]}" \
      --device "$DEVICE" \
      --seed "${SEEDS[$i]}"
  )
done

echo "Completed ${REPEATS} run(s)."
