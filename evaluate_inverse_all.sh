#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Update these defaults to match the environments used in run_all.sh

# GPUs available for evaluation (use "cpu" to force CPU execution)
GPUS=(4)
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
# DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d)
DATASETS=(darcy_1d wave_scattering chladni_2d)
MODELS=(linear linear_inverse nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine mixture_density_network conditional_realnvp)
SEEDS=(1 2 3 4 5)

# Base directories (aligned with run_all.sh / plot scripts)
LOG_BASE_DIR="/store/at46867/b2b_operator_inverse"
RESULTS_BASE_DIR="results"

# Batch size for evaluation DataLoader
BATCH_SIZE=1

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── INVERSE EVALUATION WORKER ─────────────────────────────
evaluate_inverse_dataset() {
  local dataset gpu count

  while (( $# )); do
    case "$1" in
      --dataset) dataset="$2"; shift 2;;
      --gpu)     gpu="$2";     shift 2;;
      --count)   count="$2";   shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  local dataset_log_dir="$LOG_BASE_DIR/$dataset"
  local output_dir="$RESULTS_BASE_DIR/$dataset/inverse"

  if [ ! -d "$dataset_log_dir" ]; then
    echo "  ⚠ [$count/$TOTAL_JOBS] Skipping $dataset: log directory not found at $dataset_log_dir"
    flock "$LOCK_FILE" bash -c "
      c=\$(< $STATUS_DIR/gpu_$gpu)
      echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
    "
    return 1
  fi

  local has_runs=false
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [ -f "$dataset_log_dir/$model/seed_$seed/params.pth" ]; then
        has_runs=true
        break 2
      fi
    done
  done

  if [ "$has_runs" = false ]; then
    echo "  ⚠ [$count/$TOTAL_JOBS] Skipping $dataset: no trained inverse models found"
    flock "$LOCK_FILE" bash -c "
      c=\$(< $STATUS_DIR/gpu_$gpu)
      echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
    "
    return 1
  fi

  mkdir -p "$output_dir"

  echo "  → [$count/$TOTAL_JOBS] Evaluating inverse models: $dataset → cuda:$gpu"

  python inverse_neural_operator/evaluate_models.py \
    --dataset "$dataset" \
    --log_base_dir "$dataset_log_dir" \
    --models "${MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda:$gpu" \
    --output_dir "$output_dir" \
    > "$output_dir/evaluation_log.txt" 2>&1 \
    || echo "  ✗ [$count/$TOTAL_JOBS] Evaluation failed for $dataset (exit code: $?)"

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
echo "  Results saved to: $RESULTS_BASE_DIR/*/inverse/"
echo "═══════════════════════════════════════════════════════════════"
