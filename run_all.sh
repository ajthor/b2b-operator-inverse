#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Default: use all available GPUs if not specified
GPU_LIST=${GPU_LIST:-}  # Can be set externally, e.g., GPU_LIST="0 2 3"

if [ -z "$GPU_LIST" ]; then
  # Auto-detect all available GPUs
  ALL_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
else
  # Use user-specified GPUs
  ALL_GPUS=($GPU_LIST)
fi

NUM_GPUS=${#ALL_GPUS[@]}
if [ $NUM_GPUS -eq 0 ]; then
  echo "Error: No GPUs specified or detected" >&2
  exit 1
fi

echo "Using GPUs: ${ALL_GPUS[*]}"

PROCS_PER_GPU=1
LOCK_FILE=/tmp/gpu_lock_file
STATUS_DIR=/tmp/gpu_status

# Base directory for experiment logs
LOG_BASE_DIR="/store/at46867"

# DATASETS=(burgers_1d darcy_1d parametric_heat wave_scattering)
# MODELS=(b2b_linear b2b_nonlinear variational_autoencoder invertible_network)
DATASETS=(burgers_1d darcy_1d)
MODELS=(b2b_linear b2b_nonlinear variational_autoencoder invertible_network)
SEEDS=(1 2 3)   # add more seeds if you like

#── INITIALIZE GPU STATUS ─────────────────────────────────
mkdir -p "$STATUS_DIR"
for gpu in "${ALL_GPUS[@]}"; do
  echo 0 > "$STATUS_DIR/gpu_$gpu"
done

#── EXPERIMENT WORKER ─────────────────────────────────────
run_experiment() {
  local dataset model seed gpu count

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
  echo "$logdir"
  local logfile="$logdir/log.txt"

  # run and capture exit code
  echo "[$count] $dataset | $model | seed=$seed → cuda:$gpu"
  python inverse_neural_operator/train.py \
    --dataset        "$dataset" \
    --model          "$model" \
    --seed           "$seed" \
    --device         "cuda:$gpu" \
    --log_dir        "$logdir" \
    >"$logfile" 2>&1 \
    || echo "Experiment #$count failed (exit $?)"

  # release the GPU slot
  flock $LOCK_FILE bash -c "
    c=\$(< $STATUS_DIR/gpu_$gpu)
    echo \$((c-1)) > $STATUS_DIR/gpu_$gpu
  "
  sleep 1
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
            # launch
            run_experiment \
              --dataset      "$dataset" \
              --model        "$model" \
              --seed         "$seed" \
              --gpu          "$gpu" \
              --count        "$count" &
            break 2
          fi
        done
        sleep 2
      done

    done
  done
done

wait
echo "done"
