#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Same datasets and models as in run_all.sh

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
LOG_BASE_DIR="$BASE_DIR/models"
RESULTS_BASE_DIR="$BASE_DIR/runs"
mkdir -p "$LOG_BASE_DIR" "$RESULTS_BASE_DIR"
export B2B_RESULTS_DIR BASE_DIR
echo "Resolved base dir: $BASE_DIR"

# DATASETS=(burgers_1d darcy_1d wave_scattering fwi chladni_2d)
# MODELS=(linear_inverse linear nonlinear variational_autoencoder inn_additive cinn_additive inn_affine cinn_affine conditional_realnvp mixture_density_network)
DATASETS=(burgers_1d darcy_1d wave_scattering chladni_2d)
MODELS=(linear_inverse linear nonlinear variational_autoencoder inn_additive cinn_additive cinn_additive_probabilistic inn_affine cinn_affine conditional_realnvp mixture_density_network)

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

#── B2B PERFORMANCE PLOTS ────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Generating B2B model performance plots"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Generate B2B performance plots for each dataset (once per dataset, not per model)
for dataset in "${DATASETS[@]}"; do
  echo "───────────────────────────────────────────────────────────────"
  echo "Processing B2B plots for: $dataset"
  echo "───────────────────────────────────────────────────────────────"

  # Check if shared directory exists
  SHARED_LOG_DIR="$LOG_BASE_DIR/$dataset/shared/seed_1"
  if [ ! -f "$SHARED_LOG_DIR/input_function_encoder.pth" ]; then
    echo "  ⚠ Skipping: Function encoders not found at $SHARED_LOG_DIR"
    echo ""
    continue
  fi

  # Construct paths
  SHARED_RESULTS_DIR="$RESULTS_BASE_DIR/$dataset/shared"

  # Create results directory if it doesn't exist
  if [ ! -d "$SHARED_RESULTS_DIR" ]; then
    mkdir -p "$SHARED_RESULTS_DIR"
    echo "  ✓ Created shared results directory: $SHARED_RESULTS_DIR"
  fi

  # Select plotting script based on dataset
  case "$dataset" in
    burgers_1d)
      echo "  → Generating B2B performance plots..."
      if python inverse_neural_operator/plots/plot_burgers_b2b.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_BASE_DIR" \
        --forward_model all 2>&1 | sed 's/^/    /'; then
        echo "  ✓ B2B plots completed"
      else
        echo "  ✗ B2B plots failed"
      fi
      ;;
    chladni_2d)
      echo "  → Generating B2B performance plots..."
      if python inverse_neural_operator/plots/plot_chladni_b2b.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_BASE_DIR" \
        --forward_model all 2>&1 | sed 's/^/    /'; then
        echo "  ✓ B2B plots completed"
      else
        echo "  ✗ B2B plots failed"
      fi
      ;;
    darcy_1d)
      echo "  → Generating B2B performance plots..."
      if python inverse_neural_operator/plots/plot_darcy_b2b.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_BASE_DIR" \
        --forward_model all 2>&1 | sed 's/^/    /'; then
        echo "  ✓ B2B plots completed"
      else
        echo "  ✗ B2B plots failed"
      fi
      ;;
    fwi)
      echo "  → Generating B2B performance plots..."
      if python inverse_neural_operator/plots/plot_fwi_b2b.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_BASE_DIR" \
        --forward_model all 2>&1 | sed 's/^/    /'; then
        echo "  ✓ B2B plots completed"
      else
        echo "  ✗ B2B plots failed"
      fi
      ;;
    parametric_heat)
      echo "  → Generating B2B performance plots..."
      if python inverse_neural_operator/plots/plot_parametric_heat_b2b.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_BASE_DIR" \
        --forward_model all 2>&1 | sed 's/^/    /'; then
        echo "  ✓ B2B plots completed"
      else
        echo "  ✗ B2B plots failed"
      fi
      ;;
    wave_scattering)
      echo "  → Generating B2B performance plots..."
      if python inverse_neural_operator/plots/plot_wave_scattering_b2b.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_BASE_DIR" \
        --forward_model all 2>&1 | sed 's/^/    /'; then
        echo "  ✓ B2B plots completed"
      else
        echo "  ✗ B2B plots failed"
      fi
      ;;
    *)
      echo "  ⚠ B2B plots not available for dataset: $dataset"
      ;;
  esac

  echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Generating model-specific plots"
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
echo "  All plot generation completed"
echo "  Generated B2B plots for ${#DATASETS[@]} dataset(s)"
echo "  Processed $CURRENT_JOB model-specific jobs"
echo "  Results saved to: $RESULTS_BASE_DIR"
echo "═══════════════════════════════════════════════════════════════"
