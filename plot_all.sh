#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Same datasets and models as in run_all.sh

# DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d)
# MODELS=(linear_inverse linear nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine cinn_additive_probabilistic cinn_affine_probabilistic mixture_density_network)
DATASETS=(burgers_1d darcy_1d wave_scattering fwi)
MODELS=(linear_inverse linear nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine cinn_additive_probabilistic cinn_affine_probabilistic mixture_density_network)

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

# Calculate total number of plot jobs
TOTAL_JOBS=$((${#DATASETS[@]} * ${#MODELS[@]}))
CURRENT_JOB=0

echo "═══════════════════════════════════════════════════════════════"
echo "  Starting plot generation"
echo "  Datasets: ${DATASETS[*]}"
echo "  Models: ${MODELS[*]}"
echo "  Total jobs: $TOTAL_JOBS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

#── MAIN LOOP ────────────────────────────────────────────
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    CURRENT_JOB=$((CURRENT_JOB + 1))

    echo "───────────────────────────────────────────────────────────────"
    echo "[$CURRENT_JOB/$TOTAL_JOBS] Processing: $dataset/$model"
    echo "───────────────────────────────────────────────────────────────"

    # Construct complete paths for model logs and results
    MODEL_LOG_DIR="$LOG_BASE_DIR/$dataset/$model/seed_1"
    RESULTS_DIR="$RESULTS_BASE_DIR/$dataset/$model"

    # Create results directory if it doesn't exist
    if [ ! -d "$RESULTS_DIR" ]; then
      mkdir -p "$RESULTS_DIR"
      echo "  ✓ Created results directory: $RESULTS_DIR"
    fi

    # Check if model exists
    if [ ! -f "$MODEL_LOG_DIR/params.pth" ]; then
      echo "  ⚠ Skipping: Model not found at $MODEL_LOG_DIR"
      echo ""
      continue
    fi

    # Select plotting function based on dataset
    case "$dataset" in
      burgers_1d)
      echo "  → Generating standard plots..."
      if python inverse_neural_operator/plots/plot_burgers.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Standard plots completed"
      else
        echo "  ✗ Standard plots failed"
      fi

      # Also generate probabilistic plots
      echo "  → Generating probabilistic plots..."
      if python inverse_neural_operator/plots/plot_burgers_probabilistic.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR/probabilistic" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Probabilistic plots completed"
      else
        echo "  ✗ Probabilistic plots failed"
      fi
      
      # # Generate publication-quality plots
      # python inverse_neural_operator/plots/plot_burgers_publication.py \
      # --model "$model" \
      # --log_dir "$MODEL_LOG_DIR" \
      # --results_dir "$RESULTS_DIR/publication"
      
      # # Generate comparison plot (only for first model to avoid duplicates)
      # if [ "$model" = "${MODELS[0]}" ]; then
      #   python inverse_neural_operator/plots/plot_burgers_comparison.py \
      #   --log_dir "$LOG_BASE_DIR" \
      #   --results_dir "$RESULTS_DIR/comparison"
      # fi
      ;;
      chladni_2d)
      echo "  → Generating standard plots..."
      if python inverse_neural_operator/plots/plot_chladni.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Standard plots completed"
      else
        echo "  ✗ Standard plots failed"
      fi
      ;;
      darcy_1d)
      echo "  → Generating standard plots..."
      if python inverse_neural_operator/plots/plot_darcy.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Standard plots completed"
      else
        echo "  ✗ Standard plots failed"
      fi

      # Also generate probabilistic plots
      echo "  → Generating probabilistic plots..."
      if python inverse_neural_operator/plots/plot_darcy_probabilistic.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR/probabilistic" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Probabilistic plots completed"
      else
        echo "  ✗ Probabilistic plots failed"
      fi

      # # Generate publication-quality plots
      # python inverse_neural_operator/plots/plot_darcy_publication.py \
      # --model "$model" \
      # --log_dir "$MODEL_LOG_DIR" \
      # --results_dir "$RESULTS_DIR/publication"
      ;;
      fwi)
      echo "  → Generating standard plots..."
      if python inverse_neural_operator/plots/plot_fwi.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Standard plots completed"
      else
        echo "  ✗ Standard plots failed"
      fi
      ;;
      parametric_heat)
      echo "  → Generating standard plots..."
      if python inverse_neural_operator/plots/plot_parametric_heat.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Standard plots completed"
      else
        echo "  ✗ Standard plots failed"
      fi
      ;;
      wave_scattering)
      echo "  → Generating standard plots..."
      if python inverse_neural_operator/plots/plot_wave_scattering.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Standard plots completed"
      else
        echo "  ✗ Standard plots failed"
      fi

      # Also generate probabilistic plots
      echo "  → Generating probabilistic plots..."
      if python inverse_neural_operator/plots/plot_wave_scattering_probabilistic.py \
        --model "$model" \
        --log_dir "$MODEL_LOG_DIR" \
        --results_dir "$RESULTS_DIR/probabilistic" 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Probabilistic plots completed"
      else
        echo "  ✗ Probabilistic plots failed"
      fi

      # # Generate publication-quality plots
      # python inverse_neural_operator/plots/plot_wave_scattering_publication.py \
      # --model "$model" \
      # --log_dir "$MODEL_LOG_DIR" \
      # --results_dir "$RESULTS_DIR/publication"
      ;;
      *)
      echo "  ✗ Unknown dataset: $dataset"
      exit 1
      ;;
    esac

    echo ""
  done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Plot generation completed"
echo "  Processed $CURRENT_JOB/$TOTAL_JOBS jobs"
echo "  Results saved to: $RESULTS_BASE_DIR"
echo "═══════════════════════════════════════════════════════════════"