#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Same datasets and models as in run_all.sh

# DATASETS=(burgers_1d darcy_1d parametric_heat wave_scattering fwi chladni_2d)
# MODELS=(b2b_linear b2b_nonlinear variational_autoencoder deeponet inn_additive cinn_additive inn_affine cinn_affine mixture_density_network)
DATASETS=(burgers_1d)
MODELS=(variational_autoencoder)

# Base directory for experiment logs - same as in run_all.sh
LOG_BASE_DIR="/store/at46867/b2b_operator_inverse"
RESULTS_BASE_DIR="results"

#── ARGUMENT PARSING ──────────────────────────────────────
MODEL=""
DATASET=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--model MODEL_NAME] [--dataset DATASET_NAME]"
      echo "  --model MODEL_NAME    Specify a single model to plot (optional)"
      echo "  --dataset DATASET_NAME Specify a single dataset to plot (optional)"
      echo "  --help                Show this help message"
      echo ""
      echo "Available models: ${MODELS[*]}"
      echo "Available datasets: ${DATASETS[*]}"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Set datasets and models based on arguments
if [[ -n "$DATASET" ]]; then
  DATASETS=("$DATASET")
fi

if [[ -n "$MODEL" ]]; then
  MODELS=("$MODEL")
fi

echo "Starting plots generation for datasets: ${DATASETS[*]} and models: ${MODELS[*]}..."

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
    # Select plotting function based on dataset
    case "$dataset" in
      burgers_1d)
      python inverse_neural_operator/plots/plot_burgers.py \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
      ;;
      chladni)
      python inverse_neural_operator/plots/plot_chladni.py \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
      ;;
      darcy_1d)
      python inverse_neural_operator/plots/plot_darcy.py \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
      ;;
      fwi)
      python inverse_neural_operator/plots/plot_fwi.py \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
      ;;
      parametric_heat)
      python inverse_neural_operator/plots/plot_parametric_heat.py \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
      ;;
      wave_scattering)
      python inverse_neural_operator/plots/plot_wave_scattering.py \
      --model "$model" \
      --log_dir "$LOG_BASE_DIR" \
      --results_dir "$RESULTS_DIR"
      ;;
      *)
      echo "Unknown dataset: $dataset"
      exit 1
      ;;
    esac

    
    echo "Completed plots for $dataset/$model"
  done
done

echo "done"