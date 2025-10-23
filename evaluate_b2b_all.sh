#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Datasets and models to evaluate - matches run_all.sh

# List of GPUs to use for evaluation
GPUS=(3)
ALL_GPUS=("${GPUS[@]}")
if [ ${#ALL_GPUS[@]} -eq 0 ]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi

echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1
LOCK_FILE=/tmp/gpu_lock_file_eval
STATUS_DIR=/tmp/gpu_status_eval

# DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d)
DATASETS=(burgers_1d darcy_1d)
FORWARD_MODELS=(b2b_linear b2b_nonlinear)
SEEDS=(1 2 3 4 5)

# Base directories - same as run_all.sh and plot_all.sh
LOG_BASE_DIR="/store/at46867/b2b_operator_inverse"
RESULTS_BASE_DIR="results"

# Batch size for evaluation
BATCH_SIZE=32

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── EVALUATION WORKER ─────────────────────────────────────
evaluate_b2b_dataset() {
  local dataset gpu count

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)  dataset="$2";  shift 2;;
      --gpu)      gpu="$2";      shift 2;;
      --count)    count="$2";    shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  # Construct paths
  local dataset_log_dir="$LOG_BASE_DIR/$dataset"
  local output_dir="$RESULTS_BASE_DIR/$dataset/shared"

  # Check if shared directory exists
  local shared_dir="$dataset_log_dir/shared"
  if [ ! -d "$shared_dir" ]; then
    echo "  ⚠ [$count/$TOTAL_JOBS] Skipping $dataset: Shared directory not found at $shared_dir"
    # free the GPU slot
    flock "$LOCK_FILE" bash -c "
      c=\$(< $STATUS_DIR/gpu_$gpu)
      echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
    "
    return 1
  fi

  # Check if at least one seed directory exists
  local seed_exists=false
  for seed in "${SEEDS[@]}"; do
    if [ -d "$shared_dir/seed_$seed" ]; then
      seed_exists=true
      break
    fi
  done

  if [ "$seed_exists" = false ]; then
    echo "  ⚠ [$count/$TOTAL_JOBS] Skipping $dataset: No seed directories found in $shared_dir"
    # free the GPU slot
    flock "$LOCK_FILE" bash -c "
      c=\$(< $STATUS_DIR/gpu_$gpu)
      echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
    "
    return 1
  fi

  # Create output directory if it doesn't exist
  mkdir -p "$output_dir"

  echo "  → [$count/$TOTAL_JOBS] Evaluating B2B models: $dataset → cuda:$gpu"

  # Run Python evaluation script
  python inverse_neural_operator/evaluate_b2b.py \
    --dataset "$dataset" \
    --log_base_dir "$dataset_log_dir" \
    --seeds "${SEEDS[@]}" \
    --forward_models "${FORWARD_MODELS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda:$gpu" \
    --output_dir "$output_dir" \
    > "$output_dir/evaluation_log.txt" 2>&1 \
    || echo "  ✗ [$count/$TOTAL_JOBS] Evaluation failed for $dataset (exit code: $?)"

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
echo "  Results saved to: $RESULTS_BASE_DIR/*/shared/"
echo "═══════════════════════════════════════════════════════════════"
