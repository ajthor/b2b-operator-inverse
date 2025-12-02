#!/usr/bin/env bash
set -euo pipefail

# ── CONFIGURATION ────────────────────────────────────────────────

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

# Datasets / seeds to iterate over
# DATASETS=(darcy_1d burgers_1d wave_scattering chladni_2d)
DATASETS=(darcy_1d burgers_1d wave_scattering chladni_2d)
SEEDS=(1 2 3 4 5)
echo "Datasets: ${DATASETS[*]}"
echo "Seeds   : ${SEEDS[*]}"

# GPUs to use for scheduling
GPUS=(1 2 3 4 5 6)
ALL_GPUS=("${GPUS[@]}")
if [[ ${#ALL_GPUS[@]} -eq 0 ]]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi
echo "Using GPUs: ${ALL_GPUS[*]}"

# GPUs consumed per IFNO job
IFNO_GPUS_PER_JOB=${IFNO_GPUS_PER_JOB:-1}

# Capture any extra CLI args (passed straight to train_ifno_standalone.py)
PY_ARGS=("$@")
if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
  echo "Extra python args → ${PY_ARGS[*]}"
else
  echo "Extra python args → <none>"
fi

# Scheduler helpers
LOCK_FILE=/tmp/ifno_gpu_lock
STATUS_DIR=/tmp/ifno_gpu_status
mkdir -p "$STATUS_DIR"
> "$LOCK_FILE"
trap 'rm -rf "$STATUS_DIR"' EXIT
CURRENT_GPU_GROUP_DIR=""
GPU_GROUPS=()

build_gpu_groups() {
  local stage="$1"
  local per_job="$2"
  if (( per_job <= 0 )); then
    echo "Error: GPUs per job for IFNO must be >= 1" >&2
    exit 1
  fi
  GPU_GROUPS=()
  local total=${#ALL_GPUS[@]}
  local idx=0
  while (( idx < total )); do
    local group=()
    local count=0
    while (( count < per_job && idx < total )); do
      group+=("${ALL_GPUS[$idx]}")
      idx=$((idx + 1))
      count=$((count + 1))
    done
    if [[ ${#group[@]} -gt 0 ]]; then
      GPU_GROUPS+=("$(IFS=,; echo "${group[*]}")")
    fi
  done
  if [[ ${#GPU_GROUPS[@]} -eq 0 ]]; then
    echo "Error: Unable to build GPU groups" >&2
    exit 1
  fi
  CURRENT_GPU_GROUP_DIR="$STATUS_DIR/${stage}_groups"
  rm -rf "$CURRENT_GPU_GROUP_DIR"
  mkdir -p "$CURRENT_GPU_GROUP_DIR"
  for idx in "${!GPU_GROUPS[@]}"; do
    echo 0 > "$CURRENT_GPU_GROUP_DIR/group_$idx"
  done
  export CURRENT_GPU_GROUP_DIR
}

try_claim_group() {
  local idx="$1"
  if flock "$LOCK_FILE" bash -c "[ \$(< $CURRENT_GPU_GROUP_DIR/group_$idx) -eq 0 ]"; then
    flock "$LOCK_FILE" bash -c "echo 1 > $CURRENT_GPU_GROUP_DIR/group_$idx"
    return 0
  fi
  return 1
}

# ── WORKER FUNCTION ───────────────────────────────────────────────

train_ifno_job() {
  local dataset seed cuda_devices group_idx count exit_code world_size printable
  while (( $# )); do
    case "$1" in
      --dataset) dataset="$2"; shift 2;;
      --seed)    seed="$2";    shift 2;;
      --cuda_devices) cuda_devices="$2"; shift 2;;
      --group_idx) group_idx="$2"; shift 2;;
      --count)   count="$2";   shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  IFS=',' read -ra _gpu_array <<< "$cuda_devices"
  world_size=${#_gpu_array[@]}
  printable="${cuda_devices//,/ }"
  echo "  → [$count/$TOTAL_IFNO_JOBS] Training IFNO: ${dataset} | seed=${seed} → cuda:{${printable}}"

  local log_dir="$RUNS_BASE_DIR/$dataset/ifno/seed_$seed"
  local log_file="$log_dir/train.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  local master_port=$(( (RANDOM % 20000) + 10000 ))
  if [[ ${#PY_ARGS[@]} -gt 0 ]]; then
    if ! CUDA_VISIBLE_DEVICES="$cuda_devices" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$master_port \
      torchrun --standalone --nproc_per_node="$world_size" \
      inverse_neural_operator/train_ifno_standalone.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "cuda" \
      --base_dir "$BASE_DIR" \
      "${PY_ARGS[@]}" \
      >>"$log_file" 2>&1
    then
      exit_code=$?
      echo "  ✗ [$count/$TOTAL_IFNO_JOBS] IFNO training failed (${dataset}, seed=${seed}) with exit code $exit_code" | tee -a "$log_file"
    fi
  else
    if ! CUDA_VISIBLE_DEVICES="$cuda_devices" \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$master_port \
      torchrun --standalone --nproc_per_node="$world_size" \
      inverse_neural_operator/train_ifno_standalone.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --device "cuda" \
      --base_dir "$BASE_DIR" \
      >>"$log_file" 2>&1
    then
      exit_code=$?
      echo "  ✗ [$count/$TOTAL_IFNO_JOBS] IFNO training failed (${dataset}, seed=${seed}) with exit code $exit_code" | tee -a "$log_file"
    fi
  fi

  flock "$LOCK_FILE" bash -c "echo 0 > $CURRENT_GPU_GROUP_DIR/group_$group_idx"
}

export -f train_ifno_job
export LOCK_FILE CURRENT_GPU_GROUP_DIR BASE_DIR RUNS_BASE_DIR

# ── MAIN SCHEDULER ────────────────────────────────────────────────

TOTAL_IFNO_JOBS=$((${#DATASETS[@]} * ${#SEEDS[@]}))
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Launching IFNO training sweep"
echo "  Total jobs: ${TOTAL_IFNO_JOBS}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

build_gpu_groups "ifno" "$IFNO_GPUS_PER_JOB"
count=0
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    count=$((count + 1))
    while :; do
      for group_idx in "${!GPU_GROUPS[@]}"; do
        if try_claim_group "$group_idx"; then
          cuda_devices="${GPU_GROUPS[$group_idx]}"
          train_ifno_job \
            --dataset "$dataset" \
            --seed "$seed" \
            --cuda_devices "$cuda_devices" \
            --group_idx "$group_idx" \
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
