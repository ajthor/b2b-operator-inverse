#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Same datasets and models as in run_all.sh
DATASETS=(burgers_1d darcy_1d)
MODELS=(b2b_linear b2b_nonlinear variational_autoencoder invertible_network)
# DATASETS=(burgers_1d darcy_1d)
# MODELS=(invertible_network)

# Base directory for experiment logs - same as in run_all.sh
LOG_BASE_DIR="/store/at46867"
RESULTS_BASE_DIR="results"

echo "Starting plots generation for all datasets and models..."

#── MAIN LOOP ────────────────────────────────────────────
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "Plotting results for dataset: $dataset, model: $model"
    
    # Create results directory if it doesn't exist
    RESULTS_DIR="$RESULTS_BASE_DIR/$dataset/$model"
    if [ ! -d "$RESULTS_DIR" ]; then
      echo "Creating results directory: $RESULTS_DIR"
      mkdir -p "$RESULTS_DIR"
    fi
    
    # Run the plotting script
    python inverse_neural_operator/plot_results.py \
      --dataset "$dataset" \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
    
    echo "✓ Completed plots for $dataset/$model"
  done
done

echo "done"