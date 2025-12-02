#!/usr/bin/env bash
set -euo pipefail

# ── CONFIGURATION (edit like run_all.sh) ───────────────────────────

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

# GPUs to use (round-robin scheduling across these ids)
GPUS=(1 2 3 4 5 6)
ALL_GPUS=("${GPUS[@]}")
if [[ ${#ALL_GPUS[@]} -eq 0 ]]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi
echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1               # concurrent jobs per GPU
LOCK_FILE=/tmp/ifno_gpu_lock
STATUS_DIR=/tmp/ifno_gpu_status

# Datasets / seeds to iterate over
# DATASETS=(darcy_1d burgers_1d wave_scattering chladni_2d)
DATASETS=(darcy_1d burgers_1d)
SEEDS=(1 2 3 4 5)

echo "Datasets: ${DATASETS[*]}"
echo "Seeds   : ${SEEDS[*]}"

# Capture any extra CLI args (passed straight to train_ifno_standalone.py)
PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  echo "Extra python args → ${PY_ARGS[*]}"
else
  echo "Extra python args → <none>"
fi

# ── INITIALIZE GPU STATUS ─────────────────────────────────────────
mkdir -p "$STATUS_DIR"
> "$LOCK_FILE"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

# ── WORKER FUNCTION ───────────────────────────────────────────────
train_ifno_job() {
  local dataset seed gpu count exit_code

  while (( $# )); do
    case "$1" in
      --dataset) dataset="$2"; shift 2;;
      --seed)    seed="$2";    shift 2;;
      --gpu)     gpu="$2";     shift 2;;
      --count)   count="$2";   shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  echo "  → [$count/$TOTAL_IFNO_JOBS] Training IFNO: ${dataset} | seed=${seed} → cuda:${gpu}"

  local log_dir="$RUNS_BASE_DIR/$dataset/ifno/seed_$seed"
  local log_file="$log_dir/train.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  # Train IFNO - Python handles all path construction
  if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
    if ! python inverse_neural_operator/train_ifno_standalone.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "cuda:${gpu}" \
      --base_dir "$BASE_DIR" \
      "${PY_ARGS[@]}" \
      >>"$log_file" 2>&1
    then
      exit_code=$?
      echo "  ✗ [$count/$TOTAL_IFNO_JOBS] IFNO training failed (${dataset}, seed=${seed}) with exit code $exit_code" | tee -a "$log_file"
    fi
  else
    if ! python inverse_neural_operator/train_ifno_standalone.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "cuda:${gpu}" \
      --base_dir "$BASE_DIR" \
      >>"$log_file" 2>&1
    then
      exit_code=$?
      echo "  ✗ [$count/$TOTAL_IFNO_JOBS] IFNO training failed (${dataset}, seed=${seed}) with exit code $exit_code" | tee -a "$log_file"
    fi
  fi

  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
}

export -f train_ifno_job
export LOCK_FILE STATUS_DIR B2B_RESULTS_DIR

# ── MAIN SCHEDULER ────────────────────────────────────────────────
TOTAL_IFNO_JOBS=$((${#DATASETS[@]} * ${#SEEDS[@]}))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Launching IFNO training sweep"
echo "  Total jobs: ${TOTAL_IFNO_JOBS}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

count=0
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    count=$((count+1))

    while :; do
      for gpu_idx in "${!ALL_GPUS[@]}"; do
        gpu="${ALL_GPUS[$gpu_idx]}"
        if flock "$LOCK_FILE" bash -c "[ \$(< $STATUS_DIR/gpu_$gpu) -lt $PROCS_PER_GPU ]"; then
          flock "$LOCK_FILE" bash -c "
            c=\$(< $STATUS_DIR/gpu_$gpu)
            echo \$((c+1)) > $STATUS_DIR/gpu_$gpu
          "

          train_ifno_job \
            --dataset "$dataset" \
            --seed "$seed" \
            --gpu "$gpu" \
            --count "$count" &

          sleep 1
          break 2
        fi
      done
      sleep 2
    done

  done
done

wait
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ All IFNO jobs completed"
echo "  Logs: ${RUNS_BASE_DIR}"
echo "═══════════════════════════════════════════════════════════════"
