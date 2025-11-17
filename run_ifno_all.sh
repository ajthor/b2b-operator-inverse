#!/usr/bin/env bash
set -euo pipefail

# ── CONFIGURATION (edit like run_all.sh) ───────────────────────────

# GPUs to use (round-robin scheduling across these ids)
GPUS=(5 6)
ALL_GPUS=("${GPUS[@]}")
if [[ ${#ALL_GPUS[@]} -eq 0 ]]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi
echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1               # concurrent jobs per GPU
LOCK_FILE=/tmp/ifno_gpu_lock
STATUS_DIR=/tmp/ifno_gpu_status

# Base directory for IFNO experiment logs / checkpoints
LOG_BASE_DIR="/workspaces/b2b-operator-inverse/logs_ifno"

# Datasets / seeds to iterate over
DATASETS=(darcy_1d burgers_1d chladni_2d wave_scattering)
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

# timestamp="$(date +%Y%m%d_%H%M%S)"
# RUN_ROOT="${LOG_BASE_DIR}/${timestamp}"
RUN_ROOT="${LOG_BASE_DIR}"
mkdir -p "${RUN_ROOT}"

# ── WORKER FUNCTION ───────────────────────────────────────────────
train_ifno_job() {
  local dataset seed gpu count

  while (( $# )); do
    case "$1" in
      --dataset) dataset="$2"; shift 2;;
      --seed)    seed="$2";    shift 2;;
      --gpu)     gpu="$2";     shift 2;;
      --count)   count="$2";   shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  local run_dir="${RUN_ROOT}/${dataset}/seed_${seed}"
  mkdir -p "${run_dir}/checkpoints"
  local logfile="${run_dir}/train.log"
  : > "$logfile"

  echo "  → [$count/$TOTAL_IFNO_JOBS] Training IFNO: ${dataset} | seed=${seed} → cuda:${gpu}"

  if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
    python train_ifno_standalone.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "cuda:${gpu}" \
      --log_dir "${run_dir}" \
      --checkpoint_dir "${run_dir}/checkpoints" \
      "${PY_ARGS[@]}" \
      >>"$logfile" 2>&1
  else
    python train_ifno_standalone.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "cuda:${gpu}" \
      --log_dir "${run_dir}" \
      --checkpoint_dir "${run_dir}/checkpoints" \
      >>"$logfile" 2>&1
  fi || echo "  ✗ [$count/$TOTAL_IFNO_JOBS] IFNO training failed (${dataset}, seed=${seed})"

  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
}

export -f train_ifno_job
export LOCK_FILE STATUS_DIR RUN_ROOT

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
echo "  Logs: ${RUN_ROOT}"
echo "═══════════════════════════════════════════════════════════════"
