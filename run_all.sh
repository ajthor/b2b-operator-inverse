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
LOG_BASE_DIR="/workspaces/b2b-operator-inverse/logs"

# DATASETS=(burgers_1d darcy_1d parametric_heat wave_scattering fwi_flat fwi_curve chladni_2d)
# MODELS=(b2b_linear b2b_nonlinear variational_autoencoder invertible_network)
DATASETS=(chladni_2d)

MODELS=(b2b_linear b2b_nonlinear variational_autoencoder invertible_network)
SEEDS=(1)   # add more seeds if you like

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── EXPERIMENT WORKER ─────────────────────────────────────
run_experiment() {
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

  # prepare logdir
  local logdir="$LOG_BASE_DIR/$dataset/$model/seed_$seed"
  mkdir -p "$logdir"
  # echo "$logdir"
  local logfile="$logdir/log.txt"

  # clear the log file
  : > "$logfile"

  # run and capture exit code
  echo "[$count] $dataset | $model | seed=$seed → cuda:$gpu"
  
  # Step 1: Train input function encoder
  python inverse_neural_operator/train_function_encoder.py \
    --encoder_type "input" \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$logdir" \
    >>"$logfile" 2>&1 \
    || echo "[$count] Training input function encoder failed with exit code $?"

  sleep 1
  
  # Step 2: Train output function encoder
  python inverse_neural_operator/train_function_encoder.py \
    --encoder_type "output" \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$logdir" \
    >>"$logfile" 2>&1 \
    || echo "[$count] Training output function encoder failed with exit code $?"

  sleep 1
  
  # Step 3: Train the model if both function encoders succeeded
  python inverse_neural_operator/train_model.py \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$seed" \
    --device "cuda:$gpu" \
    --log_dir "$logdir" \
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

export -f run_experiment
export LOCK_FILE

#── MAIN SCHEDULER ────────────────────────────────────────
count=0

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

            run_experiment \
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

wait
