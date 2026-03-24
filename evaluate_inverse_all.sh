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

# GPUs available for evaluation
GPUS=(0)
ALL_GPUS=("${GPUS[@]}")
if [ ${#ALL_GPUS[@]} -eq 0 ]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi

echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1
LOCK_FILE=/tmp/gpu_lock_file_inverse_eval
STATUS_DIR=/tmp/gpu_status_inverse_eval

# Datasets and inverse models to evaluate
# DATASETS=(burgers_1d darcy_1d elastic_plate wave_scattering fwi chladni_2d)
DATASETS=(burgers_1d darcy_1d elastic_plate wave_scattering fwi chladni_2d)
MODELS=(linear linear_inverse nonlinear variational_autoencoder inn_additive cinn_additive cinn_additive_probabilistic inn_affine cinn_affine mixture_density_network conditional_realnvp)
SEEDS=(1 2 3 4 5)

# Batch size for evaluation DataLoader
BATCH_SIZE=1

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── INVERSE EVALUATION WORKER ─────────────────────────────
evaluate_inverse_dataset() {
  local dataset gpu count exit_code

  while (( $# )); do
    case "$1" in
      --dataset) dataset="$2"; shift 2;;
      --gpu)     gpu="$2";     shift 2;;
      --count)   count="$2";   shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  echo "  → [$count/$TOTAL_JOBS] Evaluating inverse models: $dataset → cuda:$gpu"

  local log_dir="$RUNS_BASE_DIR/$dataset"
  local log_file="$log_dir/evaluate.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  # Run Python evaluation script - Python handles all path construction
  if ! python inverse_neural_operator/evaluate_models.py \
    --dataset "$dataset" \
    --models "${MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda:$gpu" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_JOBS] Evaluation failed for $dataset (exit code: $exit_code)" | tee -a "$log_file"
  fi

  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "

  return 0
}

export -f evaluate_inverse_dataset
export LOCK_FILE
export STATUS_DIR
export BATCH_SIZE

#── ARGUMENT PARSING ──────────────────────────────────────
SELECTED_MODELS=()
SELECTED_SEEDS=()
DATASET_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_FILTER="$2"
      shift 2
      ;;
    --models)
      SELECTED_MODELS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        SELECTED_MODELS+=("$1")
        shift
      done
      ;;
    --seeds)
      SELECTED_SEEDS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        SELECTED_SEEDS+=("$1")
        shift
      done
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --dataset NAME        Evaluate only the specified dataset
  --models M1 [M2 ...]  Override the default model list
  --seeds S1 [S2 ...]   Override the default seed list
  --batch_size N        Evaluation batch size (default: $BATCH_SIZE)
  --help, -h            Show this help message

Defaults:
  Datasets: ${DATASETS[*]}
  Models:   ${MODELS[*]}
  Seeds:    ${SEEDS[*]}
  GPUs:     ${ALL_GPUS[*]}

Examples:
  $0 --dataset burgers_1d
  $0 --dataset darcy_1d --models linear_inverse nonlinear
  $0 --seeds 1 2 3 --batch_size 4
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help for usage information" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$DATASET_FILTER" ]]; then
  DATASETS=("$DATASET_FILTER")
fi

if [[ ${#SELECTED_MODELS[@]} -gt 0 ]]; then
  MODELS=("${SELECTED_MODELS[@]}")
fi

if [[ ${#SELECTED_SEEDS[@]} -gt 0 ]]; then
  SEEDS=("${SELECTED_SEEDS[@]}")
fi

TOTAL_JOBS=${#DATASETS[@]}
export TOTAL_JOBS

echo "═══════════════════════════════════════════════════════════════"
echo "  Inverse Model Evaluation"
echo "═══════════════════════════════════════════════════════════════"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Models:     ${MODELS[*]}"
echo "  Seeds:      ${SEEDS[*]}"
echo "  GPUs:       ${ALL_GPUS[*]}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Total Jobs: $TOTAL_JOBS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

#── MAIN SCHEDULER ────────────────────────────────────────
count=0

echo "───────────────────────────────────────────────────────────────"
echo "  Evaluating inverse models across datasets"
echo "───────────────────────────────────────────────────────────────"

for dataset in "${DATASETS[@]}"; do
  count=$((count+1))

  while :; do
    for gpu_idx in "${!ALL_GPUS[@]}"; do
      gpu="${ALL_GPUS[$gpu_idx]}"
      if flock "$LOCK_FILE" bash -c "[ \$(< $STATUS_DIR/gpu_$gpu) -lt $PROCS_PER_GPU ]"; then
        flock "$LOCK_FILE" bash -c "
          c=\$(< $STATUS_DIR/gpu_$gpu)
          echo \$((c+1)) > $STATUS_DIR/gpu_$gpu
        "

        evaluate_inverse_dataset \
          --dataset "$dataset" \
          --gpu     "$gpu" \
          --count   "$count" &

        sleep 1
        break 2
      fi
    done
    sleep 2
  done
done

wait
echo ""
echo "  ✓ All inverse model evaluations completed"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluation Complete"
echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluated $TOTAL_JOBS dataset(s)"
if [ -n "$B2B_RESULTS_DIR" ]; then
  echo "  Results saved to: $B2B_RESULTS_DIR/runs/*/<model>/"
else
echo "  Results saved under: $RUNS_BASE_DIR/*/<model>/"
fi
echo "═══════════════════════════════════════════════════════════════"
