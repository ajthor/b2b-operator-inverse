#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────

# List of GPUs to use
GPUS=(0 1)
ALL_GPUS=("${GPUS[@]}")
if [ ${#ALL_GPUS[@]} -eq 0 ]; then
  echo "Error: No GPUs specified" >&2
  exit 1
fi

echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1
LOCK_FILE=/tmp/gpu_lock_file
STATUS_DIR=/tmp/gpu_status

# Base directory for experiment logs
LOG_BASE_DIR="/geoelements/Stepan/b2b-operator-inverse/logs"
#LOG_BASE_DIR="/store/at46867/b2b_operator_inverse"

# DATASETS=(burgers_1d darcy_1d parametric_heat wave_scattering fwi chladni_2d elastic_plate)
# MODELS=(b2b_linear b2b_nonlinear variational_autoencoder deeponet inn_additive cinn_additive inn_affine cinn_affine mixture_density_network)
DATASETS=(elastic_plate)
# MODELS=(b2b_linear b2b_nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine mixture_density_network)
MODELS=(b2b_nonlinear)
FORWARD_MODELS=(b2b_nonlinear_fwd)  # Currently only b2b_nonlinear_fwd is supported
SEEDS=(1)   # add more seeds if you like

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── FUNCTION ENCODER WORKERS ──────────────────────────────
train_input_function_encoder() {
  local dataset seed gpu count encoder_type="input"

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)     dataset="$2";   shift 2;;
      --seed)        seed="$2";      shift 2;;
      --gpu)         gpu="$2";       shift 2;;
      --count)       count="$2";     shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  # prepare shared logdir for this dataset-seed combination
  local shared_logdir="$LOG_BASE_DIR/$dataset/shared/seed_$seed"
  mkdir -p "$shared_logdir"
  local logfile="$shared_logdir/input_function_encoder_log.txt"

  # clear the log file
  : > "$logfile"

  echo "[$count] Training INPUT function encoder for $dataset | seed=$seed → cuda:$gpu"
  
  # Train input function encoder
  python inverse_neural_operator/train_function_encoder.py \
    --encoder_type "input" \
    --dataset "$dataset" \
    --model "shared" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$shared_logdir" \
    >>"$logfile" 2>&1 \
    || echo "[$count] Training input function encoder failed with exit code $?"

  # free the GPU slot
  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
    
  return 0
}

train_output_function_encoder() {
  local dataset seed gpu count encoder_type="output"

  # parse named args
  while (( $# )); do
    case "$1" in
      --dataset)     dataset="$2";   shift 2;;
      --seed)        seed="$2";      shift 2;;
      --gpu)         gpu="$2";       shift 2;;
      --count)       count="$2";     shift 2;;
      *) echo "Unknown option: $1" >&2; exit 1;;
    esac
  done

  # prepare shared logdir for this dataset-seed combination
  local shared_logdir="$LOG_BASE_DIR/$dataset/shared/seed_$seed"
  mkdir -p "$shared_logdir"
  local logfile="$shared_logdir/output_function_encoder_log.txt"

  # clear the log file
  : > "$logfile"

  echo "[$count] Training OUTPUT function encoder for $dataset | seed=$seed → cuda:$gpu"
  
  # Train output function encoder
  python inverse_neural_operator/train_function_encoder.py \
    --encoder_type "output" \
    --dataset "$dataset" \
    --model "shared" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$shared_logdir" \
    >>"$logfile" 2>&1 \
    || echo "[$count] Training output function encoder failed with exit code $?"

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

  # use shared logdir for forward model (same as function encoders)
  local shared_logdir="$LOG_BASE_DIR/$dataset/shared/seed_$seed"
  mkdir -p "$shared_logdir"
  local logfile="$shared_logdir/forward_${model}_log.txt"

  # clear the log file
  : > "$logfile"

  echo "[$count] Training forward model: $dataset | $model | seed=$seed → cuda:$gpu"
  
  # Train the forward model using pre-trained function encoders (save to shared dir)
  python inverse_neural_operator/train_forward_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$shared_logdir" \
    >>"$logfile" 2>&1 \
    || echo "[$count] Training forward model $model failed with exit code $?"

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

  # prepare model-specific logdir
  local model_logdir="$LOG_BASE_DIR/$dataset/$model/seed_$seed"
  local shared_logdir="$LOG_BASE_DIR/$dataset/shared/seed_$seed"
  mkdir -p "$model_logdir"
  local logfile="$model_logdir/log.txt"

  # clear the log file
  : > "$logfile"

  # Copy the pre-trained function encoders to the model directory
  cp "$shared_logdir/input_function_encoder.pth" "$model_logdir/" || {
    echo "[$count] Failed to copy input function encoder for $dataset | $model | seed=$seed"
    return 1
  }
  cp "$shared_logdir/output_function_encoder.pth" "$model_logdir/" || {
    echo "[$count] Failed to copy output function encoder for $dataset | $model | seed=$seed"
    return 1
  }
  cp "$shared_logdir/input_function_encoder_params.pth" "$model_logdir/" || {
    echo "[$count] Failed to copy input function encoder params for $dataset | $model | seed=$seed"
    return 1
  }
  cp "$shared_logdir/output_function_encoder_params.pth" "$model_logdir/" || {
    echo "[$count] Failed to copy output function encoder params for $dataset | $model | seed=$seed"
    return 1
  }

  # Copy any trained forward models from shared directory to the inverse model directory
  for forward_model_file in "$shared_logdir"/forward_*.pth; do
    if [ -f "$forward_model_file" ]; then
      cp "$forward_model_file" "$model_logdir/" || {
        echo "[$count] Warning: Failed to copy forward model $(basename "$forward_model_file") for $dataset | $model | seed=$seed"
      }
    fi
  done

  echo "[$count] Training model: $dataset | $model | seed=$seed → cuda:$gpu"
  
  # Train the model using pre-trained function encoders
  python inverse_neural_operator/train_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$model_logdir" \
    >>"$logfile" 2>&1 \
    || echo "[$count] Training model $model failed with exit code $?"

  sleep 1

  # free the GPU slot
  flock "$LOCK_FILE" bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
    
  return 0
}

export -f train_input_function_encoder
export -f train_output_function_encoder
export -f train_forward_model
export -f train_model
export LOCK_FILE

#── MAIN SCHEDULER ────────────────────────────────────────
count=0

# Phase 1: Train function encoders for each dataset-seed combination
echo "=== Phase 1: Training function encoders in parallel ==="
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    count=$((count+1))
    
    # We need 2 GPUs for parallel training (input and output encoders)
    input_gpu=""
    output_gpu=""
    
    # Wait until we have 2 free GPU slots
    while [[ -z "$input_gpu" || -z "$output_gpu" ]]; do
      for gpu_idx in "${!ALL_GPUS[@]}"; do
        gpu="${ALL_GPUS[$gpu_idx]}"
        if flock $LOCK_FILE bash -c "[ \$(< $STATUS_DIR/gpu_$gpu) -lt $PROCS_PER_GPU ]"; then
          if [[ -z "$input_gpu" ]]; then
            input_gpu="$gpu"
            # claim it for input encoder
            flock $LOCK_FILE bash -c "
              c=\$(< $STATUS_DIR/gpu_$gpu)
              echo \$((c+1)) > $STATUS_DIR/gpu_$gpu
            "
          elif [[ -z "$output_gpu" && "$gpu" != "$input_gpu" ]]; then
            output_gpu="$gpu"
            # claim it for output encoder
            flock $LOCK_FILE bash -c "
              c=\$(< $STATUS_DIR/gpu_$gpu)
              echo \$((c+1)) > $STATUS_DIR/gpu_$gpu
            "
          fi
        fi
      done
      
      # If we don't have 2 different GPUs, wait and try again
      if [[ -z "$input_gpu" || -z "$output_gpu" ]]; then
        sleep 2
      fi
    done

    echo "=== [$count] Training function encoders for $dataset | seed=$seed → input:cuda:$input_gpu, output:cuda:$output_gpu ==="

    # Launch both encoders in parallel
    train_input_function_encoder \
      --dataset "$dataset" \
      --seed    "$seed"  \
      --gpu     "$input_gpu"  \
      --count   "${count}a" &
    
    train_output_function_encoder \
      --dataset "$dataset" \
      --seed    "$seed"  \
      --gpu     "$output_gpu"  \
      --count   "${count}b" &

    # Wait for both encoders to complete before moving to next dataset-seed
    wait

    sleep 1

  done
done

echo "=== Phase 1 completed: All function encoders trained ==="

# Reset count for Phase 2
count=0

# Phase 2: Train forward models using the pre-trained function encoders
echo "=== Phase 2: Training forward models ==="
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
  echo "=== Phase 2 completed: All forward models trained ==="
else
  echo "=== Phase 2 skipped: No forward models specified ==="
fi

# Reset count for Phase 3
count=0

# Phase 3: Train inverse models using the pre-trained function encoders
echo "=== Phase 3: Training inverse models ==="
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
echo "=== Phase 3 completed: All inverse models trained ==="
