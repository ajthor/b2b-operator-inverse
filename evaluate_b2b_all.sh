#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────

# Parse optional --base_dir argument
B2B_RESULTS_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --base_dir)
      B2B_RESULTS_DIR_OVERRIDE="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# Hierarchy: script arg > B2B_RESULTS_DIR env var > default ./results
B2B_RESULTS_DIR="${B2B_RESULTS_DIR_OVERRIDE:-${B2B_RESULTS_DIR:-}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BASE_DIR="$SCRIPT_DIR/results"
BASE_DIR="${B2B_RESULTS_DIR:-$DEFAULT_BASE_DIR}"
if [[ "$BASE_DIR" != /* ]]; then
  BASE_DIR="$SCRIPT_DIR/$BASE_DIR"
fi
mkdir -p "$BASE_DIR"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
RUNS_BASE_DIR="$BASE_DIR/runs"
mkdir -p "$RUNS_BASE_DIR"
export B2B_RESULTS_DIR BASE_DIR
echo "Resolved base dir: $BASE_DIR"

# List of GPUs to use for evaluation
GPUS=(0)
ALL_GPUS=("${GPUS[@]}")
if [ ${#ALL_GPUS[@]} -eq 0 ]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi

echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1
LOCK_FILE=/tmp/gpu_lock_file_eval
STATUS_DIR=/tmp/gpu_status_eval

# DATASETS=(burgers_1d darcy_1d elastic_plate wave_scattering fwi chladni_2d)
DATASETS=(burgers_1d darcy_1d elastic_plate wave_scattering fwi chladni_2d)
FORWARD_MODELS=(b2b_linear b2b_nonlinear)
SEEDS=(1 2 3 4 5)

# Batch size for evaluation (match evaluate_b2b.py default)
BATCH_SIZE=4

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── EVALUATION WORKER ─────────────────────────────────────
evaluate_b2b_dataset() {
  local dataset gpu count exit_code

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)  dataset="$2";  shift 2;;
      --gpu)      gpu="$2";      shift 2;;
      --count)    count="$2";    shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  echo "  → [$count/$TOTAL_JOBS] Evaluating B2B models: $dataset → cuda:$gpu"

  local log_dir="$RUNS_BASE_DIR/$dataset/shared"
  local log_file="$log_dir/evaluate_b2b.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  # Run Python evaluation script - Python handles all path construction
  if ! python inverse_neural_operator/evaluate_b2b.py \
    --dataset "$dataset" \
    --seeds "${SEEDS[@]}" \
    --forward_models "${FORWARD_MODELS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda:$gpu" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_JOBS] Evaluation failed for $dataset (exit code: $exit_code)" | tee -a "$log_file"
  fi

  # free the GPU slot
  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "

  return 0
}

export -f evaluate_b2b_dataset
export LOCK_FILE
export STATUS_DIR
export SEEDS
export FORWARD_MODELS
export BATCH_SIZE

#── ARGUMENT PARSING ──────────────────────────────────────
DATASET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --dataset DATASET_NAME    Evaluate only specified dataset (optional)"
      echo "  --batch_size SIZE         Batch size for evaluation (default: 32)"
      echo "  --help, -h                Show this help message"
      echo ""
      echo "Available datasets: ${DATASETS[*]}"
      echo "Seeds: ${SEEDS[*]}"
      echo "Forward models: ${FORWARD_MODELS[*]}"
      echo "GPUs: ${ALL_GPUS[*]}"
      echo ""
      echo "Examples:"
      echo "  $0                         # Evaluate all datasets"
      echo "  $0 --dataset burgers_1d    # Evaluate only burgers_1d"
      echo "  $0 --batch_size 64         # Use larger batch size"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Override datasets if specified
if [[ -n "$DATASET" ]]; then
  DATASETS=("$DATASET")
fi

# Calculate total number of evaluation jobs
TOTAL_JOBS=${#DATASETS[@]}
export TOTAL_JOBS

echo "═══════════════════════════════════════════════════════════════"
echo "  B2B Model Evaluation"
echo "═══════════════════════════════════════════════════════════════"
echo "  Datasets:       ${DATASETS[*]}"
echo "  Seeds:          ${SEEDS[*]}"
echo "  Forward Models: ${FORWARD_MODELS[*]}"
echo "  GPUs:           ${ALL_GPUS[*]}"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Total Jobs:     $TOTAL_JOBS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

#── MAIN SCHEDULER ────────────────────────────────────────
count=0

echo "───────────────────────────────────────────────────────────────"
echo "  Evaluating B2B models across datasets"
echo "───────────────────────────────────────────────────────────────"

for dataset in "${DATASETS[@]}"; do
  count=$((count+1))

  # wait for a free GPU slot
  while :; do
    for gpu_idx in "${!ALL_GPUS[@]}"; do
      gpu="${ALL_GPUS[$gpu_idx]}"
      if flock $LOCK_FILE bash -c "[ \$(< $STATUS_DIR/gpu_$gpu) -lt $PROCS_PER_GPU ]"; then

        # claim it
        flock $LOCK_FILE bash -c "
          c=\$(< $STATUS_DIR/gpu_$gpu)
          echo \$((c+1)) > $STATUS_DIR/gpu_$gpu
        "

        evaluate_b2b_dataset \
          --dataset "$dataset" \
          --gpu     "$gpu" \
          --count   "$count" &

        sleep 1

        # break out so we move on to the next dataset
        break 2
      fi
    done
    sleep 2
  done
done

# Wait for all evaluation jobs to complete
wait
echo ""
echo "  ✓ All B2B model evaluations completed"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluation Complete"
echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluated $TOTAL_JOBS dataset(s)"
if [ -n "$B2B_RESULTS_DIR" ]; then
  echo "  Results saved to: $B2B_RESULTS_DIR/runs/*/shared/"
else
echo "  Results saved under: $RUNS_BASE_DIR/*/shared/"
fi
echo "═══════════════════════════════════════════════════════════════"
