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

# List of GPUs to use
GPUS=(1 2 3 4 5 6)
ALL_GPUS=("${GPUS[@]}")
if [ ${#ALL_GPUS[@]} -eq 0 ]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi

echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1
LOCK_FILE=/tmp/gpu_lock_file
STATUS_DIR=/tmp/gpu_status

# DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d elastic_plate)
# MODELS=(linear linear_inverse nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine cinn_additive_probabilistic mixture_density_network conditional_realnvp)
# FORWARD_MODELS=(b2b_linear b2b_nonlinear)
DATASETS=(burgers_1d darcy_1d)
MODELS=(linear linear_inverse nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine cinn_additive_probabilistic mixture_density_network conditional_realnvp)
FORWARD_MODELS=(b2b_linear b2b_nonlinear)
SEEDS=(1 2 3 4 5)   # add more seeds if you like

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── FUNCTION ENCODER WORKER ──────────────────────────────
train_function_encoder() {
  local dataset seed gpu count encoder_type exit_code

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)       dataset="$2";       shift 2;;
      --seed)          seed="$2";          shift 2;;
      --gpu)           gpu="$2";           shift 2;;
      --count)         count="$2";         shift 2;;
      --encoder_type)  encoder_type="$2";  shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  echo "  → [$count/$TOTAL_ENCODER_JOBS] Training ${encoder_type^^} function encoder: $dataset | seed=$seed → cuda:$gpu"

  local log_dir="$RUNS_BASE_DIR/$dataset/shared/seed_$seed"
  local log_file="$log_dir/function_encoder_${encoder_type}.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  # Train function encoder - Python handles all path construction
  if ! python inverse_neural_operator/train_function_encoder.py \
    --encoder_type "$encoder_type" \
    --dataset "$dataset" \
    --model "shared" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_ENCODER_JOBS] Training $encoder_type function encoder failed with exit code $exit_code" | tee -a "$log_file"
  fi

  # free the GPU slot
  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "

  return 0
}

#── FORWARD MODEL TRAINING WORKER ────────────────────────
train_forward_model() {
  local dataset model seed gpu count
  local exit_code=0

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)     dataset="$2";   shift 2;;
      --model)       model="$2";     shift 2;;
      --seed)        seed="$2";      shift 2;;
      --gpu)         gpu="$2";       shift 2;;
      --count)       count="$2";     shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  echo "  → [$count/$TOTAL_FORWARD_JOBS] Training forward model: $dataset | $model | seed=$seed → cuda:$gpu"

  local log_dir="$RUNS_BASE_DIR/$dataset/shared/seed_$seed"
  local log_file="$log_dir/forward_${model}.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  # Train the forward model - Python handles all path construction
  if ! python inverse_neural_operator/train_forward_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_FORWARD_JOBS] Training forward model $model failed with exit code $exit_code" | tee -a "$log_file"
  fi

  sleep 1

  # free the GPU slot
  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
    
  return 0
}

#── INVERSE MODEL TRAINING WORKER ────────────────────────────────
train_model() {
  local dataset model seed gpu count
  local exit_code=0

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)     dataset="$2";   shift 2;;
      --model)       model="$2";     shift 2;;
      --seed)        seed="$2";      shift 2;;
      --gpu)         gpu="$2";       shift 2;;
      --count)       count="$2";     shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  echo "  → [$count/$TOTAL_INVERSE_JOBS] Training inverse model: $dataset | $model | seed=$seed → cuda:$gpu"

  local log_dir="$RUNS_BASE_DIR/$dataset/$model/seed_$seed"
  local log_file="$log_dir/train.log"
  mkdir -p "$log_dir"
  : > "$log_file"

  # Train the model - Python handles all path construction and loads function encoders from shared location
  if ! python inverse_neural_operator/train_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --base_dir "$BASE_DIR" \
    >>"$log_file" 2>&1
  then
    exit_code=$?
    echo "  ✗ [$count/$TOTAL_INVERSE_JOBS] Training inverse model $model failed with exit code $exit_code" | tee -a "$log_file"
  fi

  sleep 1

  # free the GPU slot
  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
    
  return 0
}

export -f train_function_encoder
export -f train_forward_model
export -f train_model
export LOCK_FILE

#── MAIN SCHEDULER ────────────────────────────────────────

# Calculate total jobs
TOTAL_ENCODER_JOBS=$((2 * ${#DATASETS[@]} * ${#SEEDS[@]}))  # input + output
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

# Phase 1: Train function encoders for each dataset-seed combination
echo "───────────────────────────────────────────────────────────────"
echo "  Phase 1: Training function encoders"
echo "───────────────────────────────────────────────────────────────"
for encoder_type in input output; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
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

            train_function_encoder \
              --dataset      "$dataset" \
              --seed         "$seed"  \
              --gpu          "$gpu"  \
              --count        "$count" \
              --encoder_type "$encoder_type" &

            sleep 1

            # break out so we move on to the next (encoder_type, dataset, seed)
            break 2
          fi
        done
        sleep 2
      done

    done
  done
done

# Wait for all function encoder training to complete
wait
echo ""
echo "  ✓ Phase 1 completed: All function encoders trained"
echo ""

# Reset count for Phase 2
count=0

# Phase 2: Train forward models using the pre-trained function encoders
echo "───────────────────────────────────────────────────────────────"
echo "  Phase 2: Training forward models"
echo "───────────────────────────────────────────────────────────────"
if [ ${#FORWARD_MODELS[@]} -gt 0 ]; then
  for dataset in "${DATASETS[@]}"; do
    for model in "${FORWARD_MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
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

              train_forward_model \
                --dataset "$dataset" \
                --model   "$model" \
                --seed    "$seed"  \
                --gpu     "$gpu"  \
                --count   "$count" &

              sleep 1

              # break out so we move on to the next (dataset, model, seed)
              break 2
            fi
          done
          sleep 2
        done

      done
    done
  done

  # Wait for all forward model training to complete
  wait
  echo ""
  echo "  ✓ Phase 2 completed: All forward models trained"
  echo ""
else
  echo "  ⚠ Phase 2 skipped: No forward models specified"
  echo ""
fi

# Reset count for Phase 3
count=0

# Phase 3: Train inverse models using the pre-trained function encoders
echo "───────────────────────────────────────────────────────────────"
echo "  Phase 3: Training inverse models"
echo "───────────────────────────────────────────────────────────────"
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
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

            train_model \
              --dataset "$dataset" \
              --model   "$model" \
              --seed    "$seed"  \
              --gpu     "$gpu"  \
              --count   "$count" &

            sleep 1

            # break out so we move on to the next (dataset, model, seed)
            break 2
          fi
        done
        sleep 2
      done

    done
  done
done

# Wait for all inverse model training to complete
wait
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Training pipeline completed"
echo "  Phase 1: $TOTAL_ENCODER_JOBS function encoders trained"
echo "  Phase 2: $TOTAL_FORWARD_JOBS forward models trained"
echo "  Phase 3: $TOTAL_INVERSE_JOBS inverse models trained"
echo "═══════════════════════════════════════════════════════════════"
