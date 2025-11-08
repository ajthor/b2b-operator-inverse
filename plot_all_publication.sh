#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────
# Publication plots compare all models for a dataset, so we only need to specify datasets

# DATASETS=(burgers_1d darcy_1d chladni_2d wave_scattering fwi)
DATASETS=(burgers_1d darcy_1d chladni_2d)

# Base directory for experiment logs
LOG_BASE_DIR="/store/at46867/b2b_operator_inverse"
RESULTS_BASE_DIR="results"

#── ARGUMENT PARSING ──────────────────────────────────────
DATASET=""
SEED=1
SAMPLE_INDEX=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --sample_index)
      SAMPLE_INDEX="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--dataset DATASET_NAME] [--seed SEED] [--sample_index INDEX]"
      echo "  --dataset DATASET_NAME  Specify a single dataset to plot (optional)"
      echo "  --seed SEED             Specify random seed (default: 1)"
      echo "  --sample_index INDEX    Specify sample index (optional, auto-selects median if not provided)"
      echo "  --help                  Show this help message"
      echo ""
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

# Set datasets based on argument
if [[ -n "$DATASET" ]]; then
  DATASETS=("$DATASET")
fi

# Build sample_index argument if provided
SAMPLE_ARG=""
if [[ -n "$SAMPLE_INDEX" ]]; then
  SAMPLE_ARG="--sample_index $SAMPLE_INDEX"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Starting publication figure generation"
echo "  Datasets: ${DATASETS[*]}"
echo "  Seed: $SEED"
if [[ -n "$SAMPLE_INDEX" ]]; then
  echo "  Sample index: $SAMPLE_INDEX"
else
  echo "  Sample index: auto-select median-performing"
fi
echo "═══════════════════════════════════════════════════════════════"
echo ""

#── MAIN LOOP ────────────────────────────────────────────
for dataset in "${DATASETS[@]}"; do
  echo "───────────────────────────────────────────────────────────────"
  echo "Processing publication figure: $dataset"
  echo "───────────────────────────────────────────────────────────────"

  # Construct paths
  DATASET_LOG_DIR="$LOG_BASE_DIR/$dataset"
  RESULTS_DIR="$RESULTS_BASE_DIR/$dataset"

  # Create results directory if it doesn't exist
  if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "  ✓ Created results directory: $RESULTS_DIR"
  fi

  # Check if dataset directory exists
  if [ ! -d "$DATASET_LOG_DIR" ]; then
    echo "  ⚠ Skipping: Dataset not found at $DATASET_LOG_DIR"
    echo ""
    continue
  fi

  # Select plotting script based on dataset
  case "$dataset" in
    burgers_1d)
      echo "  → Generating publication figure..."
      if python inverse_neural_operator/plots/plot_burgers_publication.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_DIR" \
        --seed "$SEED" \
        $SAMPLE_ARG 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Publication figure completed"
      else
        echo "  ✗ Publication figure failed"
      fi
      ;;
    darcy_1d)
      echo "  → Generating publication figure..."
      if python inverse_neural_operator/plots/plot_darcy_publication.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_DIR" \
        --seed "$SEED" \
        $SAMPLE_ARG 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Publication figure completed"
      else
        echo "  ✗ Publication figure failed"
      fi
      ;;
    chladni_2d)
      echo "  → Generating publication figure..."
      if python inverse_neural_operator/plots/plot_chladni_publication.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_DIR" \
        --seed "$SEED" \
        $SAMPLE_ARG 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Publication figure completed"
      else
        echo "  ✗ Publication figure failed"
      fi
      ;;
    wave_scattering)
      echo "  → Generating publication figure..."
      if python inverse_neural_operator/plots/plot_wave_scattering_publication.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_DIR" \
        --seed "$SEED" \
        $SAMPLE_ARG 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Publication figure completed"
      else
        echo "  ✗ Publication figure failed"
      fi
      ;;
    fwi)
      echo "  → Generating publication figure..."
      if python inverse_neural_operator/plots/plot_fwi_publication.py \
        --log_dir "$LOG_BASE_DIR" \
        --results_dir "$RESULTS_DIR" \
        --seed "$SEED" \
        $SAMPLE_ARG 2>&1 | sed 's/^/    /'; then
        echo "  ✓ Publication figure completed"
      else
        echo "  ✗ Publication figure failed"
      fi
      ;;
    *)
      echo "  ✗ Unknown dataset: $dataset"
      ;;
  esac

  echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Publication figure generation completed"
echo "  Processed ${#DATASETS[@]} dataset(s)"
echo "  Results saved to: $RESULTS_BASE_DIR"
echo "═══════════════════════════════════════════════════════════════"
