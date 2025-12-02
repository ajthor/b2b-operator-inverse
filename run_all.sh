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

# DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d elastic_plate)
# MODELS=(linear linear_inverse nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine cinn_additive_probabilistic mixture_density_network conditional_realnvp)
# FORWARD_MODELS=(b2b_linear b2b_nonlinear)
DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d)
MODELS=(linear linear_inverse nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine cinn_additive_probabilistic mixture_density_network conditional_realnvp)
FORWARD_MODELS=(b2b_linear b2b_nonlinear)
SEEDS=(1 2 3 4 5)   # add more seeds if you like

# Available GPUs (ids on your machine)
GPUS=(1 2 3 4 5 6)
ALL_GPUS=("${GPUS[@]}")
if [[ ${#ALL_GPUS[@]} -eq 0 ]]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi
echo "Using GPUs: ${ALL_GPUS[*]}"

# Per-phase GPU allocation (set >1 to run DDP on multiple GPUs per job)
# Use GPUS_PER_JOB as a single global knob, but allow per-phase overrides if desired.
DEFAULT_GPUS_PER_JOB=${GPUS_PER_JOB:-1}
ENCODER_GPUS_PER_JOB=${ENCODER_GPUS_PER_JOB:-$DEFAULT_GPUS_PER_JOB}
FORWARD_GPUS_PER_JOB=${FORWARD_GPUS_PER_JOB:-$DEFAULT_GPUS_PER_JOB}
INVERSE_GPUS_PER_JOB=${INVERSE_GPUS_PER_JOB:-$DEFAULT_GPUS_PER_JOB}

# Scheduler state
LOCK_FILE=/tmp/b2b_gpu_lock
STATUS_DIR=/tmp/b2b_gpu_status
mkdir -p "$STATUS_DIR"
> "$LOCK_FILE"
trap 'rm -rf "$STATUS_DIR"' EXIT
CURRENT_GPU_GROUP_DIR=""
GPU_GROUPS=()

build_gpu_groups() {
  local stage="$1"
  local per_job="$2"
  if (( per_job <= 0 )); then
    echo "Error: GPUs per job for $stage must be >= 1" >&2
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
    echo "Error: Unable to build GPU groups for $stage" >&2
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

#── WORKERS ──────────────────────────────────────────────

train_function_encoder() {
  local dataset seed cuda_devices group_idx count encoder_type exit_code world_size printable
  while (( $# )); do
    case "$1" in
      --dataset)       dataset="$2";       shift 2;;
      --seed)          seed="$2";          shift 2;;
      --cuda_devices)  cuda_devices="$2";  shift 2;;
      --group_idx)     group_idx="$2";     shift 2;;
      --count)         count="$2";         shift 2;;
      --encoder_type)  encoder_type="$2";  shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done
  IFS=',' read -ra _gpu_array <<< "$cuda_devices"
  world_size=${#_gpu_array[@]}
  printable="${cuda_devices//,/ }"
  echo "  → [$count/$TOTAL_ENCODER_JOBS] Training ${encoder_type^^} function encoder: $dataset | seed=$seed → cuda:{${printable}}"

  local log_dir="$RUNS_BASE_DIR/$dataset/shared/seed_$seed"
  local log_file="$log_dir/function_encoder_${encoder_type}.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  local master_port=$(( (RANDOM % 20000) + 10000 ))
  if ! CUDA_VISIBLE_DEVICES="$cuda_devices" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$master_port \
    torchrun --standalone --nproc_per_node="$world_size" \
    inverse_neural_operator/train_function_encoder.py \
    --encoder_type "$encoder_type" \
    --dataset "$dataset" \
    --model "shared" \
    --seed "$seed" \
    --device "cuda" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_ENCODER_JOBS] Training $encoder_type function encoder failed with exit code $exit_code" | tee -a "$log_file"
  fi

  flock "$LOCK_FILE" bash -c "echo 0 > $CURRENT_GPU_GROUP_DIR/group_$group_idx"
}

train_forward_model() {
  local dataset model seed cuda_devices group_idx count exit_code world_size printable
  while (( $# )); do
    case "$1" in
      --dataset)     dataset="$2";     shift 2;;
      --model)       model="$2";       shift 2;;
      --seed)        seed="$2";        shift 2;;
      --cuda_devices) cuda_devices="$2"; shift 2;;
      --group_idx)   group_idx="$2";   shift 2;;
      --count)       count="$2";       shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done
  IFS=',' read -ra _gpu_array <<< "$cuda_devices"
  world_size=${#_gpu_array[@]}
  printable="${cuda_devices//,/ }"
  echo "  → [$count/$TOTAL_FORWARD_JOBS] Training forward model: $dataset | $model | seed=$seed → cuda:{${printable}}"

  local log_dir="$RUNS_BASE_DIR/$dataset/shared/seed_$seed"
  local log_file="$log_dir/forward_${model}.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  local master_port=$(( (RANDOM % 20000) + 10000 ))
  if ! CUDA_VISIBLE_DEVICES="$cuda_devices" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$master_port \
    torchrun --standalone --nproc_per_node="$world_size" \
    inverse_neural_operator/train_forward_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_FORWARD_JOBS] Training forward model $model failed with exit code $exit_code" | tee -a "$log_file"
  fi

  flock "$LOCK_FILE" bash -c "echo 0 > $CURRENT_GPU_GROUP_DIR/group_$group_idx"
}

train_model() {
  local dataset model seed cuda_devices group_idx count exit_code world_size printable
  while (( $# )); do
    case "$1" in
      --dataset)     dataset="$2";     shift 2;;
      --model)       model="$2";       shift 2;;
      --seed)        seed="$2";        shift 2;;
      --cuda_devices) cuda_devices="$2"; shift 2;;
      --group_idx)   group_idx="$2";   shift 2;;
      --count)       count="$2";       shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done
  IFS=',' read -ra _gpu_array <<< "$cuda_devices"
  world_size=${#_gpu_array[@]}
  printable="${cuda_devices//,/ }"
  echo "  → [$count/$TOTAL_INVERSE_JOBS] Training inverse model: $dataset | $model | seed=$seed → cuda:{${printable}}"

  local log_dir="$RUNS_BASE_DIR/$dataset/$model/seed_$seed"
  local log_file="$log_dir/train.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  local master_port=$(( (RANDOM % 20000) + 10000 ))
  if ! CUDA_VISIBLE_DEVICES="$cuda_devices" \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$master_port \
    torchrun --standalone --nproc_per_node="$world_size" \
    inverse_neural_operator/train_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_INVERSE_JOBS] Training inverse model $model failed with exit code $exit_code" | tee -a "$log_file"
  fi

  flock "$LOCK_FILE" bash -c "echo 0 > $CURRENT_GPU_GROUP_DIR/group_$group_idx"
}

export -f train_function_encoder
export -f train_forward_model
export -f train_model
export LOCK_FILE CURRENT_GPU_GROUP_DIR BASE_DIR RUNS_BASE_DIR

#── MAIN SCHEDULER ────────────────────────────────────────

TOTAL_ENCODER_JOBS=$((2 * ${#DATASETS[@]} * ${#SEEDS[@]}))
TOTAL_FORWARD_JOBS=$((${#FORWARD_MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))
TOTAL_INVERSE_JOBS=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#SEEDS[@]}))

echo "═══════════════════════════════════════════════════════════════"
echo "  Starting training pipeline"
echo "  Datasets: ${DATASETS[*]}"
echo "  Forward models: ${FORWARD_MODELS[*]}"
echo "  Inverse models: ${MODELS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Phase 1 jobs: $TOTAL_ENCODER_JOBS function encoders"
echo "  Phase 2 jobs: $TOTAL_FORWARD_JOBS forward models"
echo "  Phase 3 jobs: $TOTAL_INVERSE_JOBS inverse models"
echo "═══════════════════════════════════════════════════════════════"
echo ""

count=0

# Phase 1: Function encoders
echo "───────────────────────────────────────────────────────────────"
echo "  Phase 1: Training function encoders"
echo "───────────────────────────────────────────────────────────────"
build_gpu_groups "phase1" "$ENCODER_GPUS_PER_JOB"
for encoder_type in input output; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      count=$((count + 1))
      while :; do
        for group_idx in "${!GPU_GROUPS[@]}"; do
          if try_claim_group "$group_idx"; then
            cuda_devices="${GPU_GROUPS[$group_idx]}"
            train_function_encoder \
              --dataset "$dataset" \
              --seed "$seed" \
              --cuda_devices "$cuda_devices" \
              --group_idx "$group_idx" \
              --count "$count" \
              --encoder_type "$encoder_type" &
            sleep 1
            break 2
          fi
        done
        sleep 2
      done
    done
  done
done
wait
echo ""
echo "  ✓ Phase 1 completed: All function encoders trained"
echo ""

# Phase 2: Forward models
echo "───────────────────────────────────────────────────────────────"
echo "  Phase 2: Training forward models"
echo "───────────────────────────────────────────────────────────────"
count=0
if [[ ${#FORWARD_MODELS[@]} -gt 0 ]]; then
  build_gpu_groups "phase2" "$FORWARD_GPUS_PER_JOB"
  for dataset in "${DATASETS[@]}"; do
    for model in "${FORWARD_MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        count=$((count + 1))
        while :; do
          for group_idx in "${!GPU_GROUPS[@]}"; do
            if try_claim_group "$group_idx"; then
              cuda_devices="${GPU_GROUPS[$group_idx]}"
              train_forward_model \
                --dataset "$dataset" \
                --model "$model" \
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
  done
  wait
  echo ""
  echo "  ✓ Phase 2 completed: All forward models trained"
  echo ""
else
  echo "  ⚠ Phase 2 skipped: No forward models specified"
  echo ""
fi

# Phase 3: Inverse models
echo "───────────────────────────────────────────────────────────────"
echo "  Phase 3: Training inverse models"
echo "───────────────────────────────────────────────────────────────"
build_gpu_groups "phase3" "$INVERSE_GPUS_PER_JOB"
count=0
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      count=$((count + 1))
      while :; do
        for group_idx in "${!GPU_GROUPS[@]}"; do
          if try_claim_group "$group_idx"; then
            cuda_devices="${GPU_GROUPS[$group_idx]}"
            train_model \
              --dataset "$dataset" \
              --model "$model" \
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
done
wait

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Training pipeline completed"
echo "  Phase 1: $TOTAL_ENCODER_JOBS function encoders trained"
echo "  Phase 2: $TOTAL_FORWARD_JOBS forward models trained"
echo "  Phase 3: $TOTAL_INVERSE_JOBS inverse models trained"
echo "═══════════════════════════════════════════════════════════════"
