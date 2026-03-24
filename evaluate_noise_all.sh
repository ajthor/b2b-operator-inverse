#!/usr/bin/env bash
set -euo pipefail

#── CONFIGURATION ────────────────────────────────────────

# Parse optional --base_dir argument (matches other automation scripts)
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

# DEFAULT_DATASETS=(burgers_1d darcy_1d elastic_plate)
DEFAULT_DATASETS=(burgers_1d darcy_1d elastic_plate)
DEFAULT_MODELS=(linear_inverse linear nonlinear variational_autoencoder inn_additive cinn_additive cinn_additive_probabilistic inn_affine cinn_affine conditional_realnvp mixture_density_network ifno)
DEFAULT_SEEDS=(1)
DEFAULT_NOISE_LEVELS=(0 0.02 0.04 0.06 0.08 0.1)
DEVICE="cpu"
DRAW_COUNT=10

DATASETS=("${DEFAULT_DATASETS[@]}")
MODELS=("${DEFAULT_MODELS[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")
NOISE_LEVELS=("${DEFAULT_NOISE_LEVELS[@]}")

#── ARGUMENT PARSING ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASETS=("$2")
      shift 2
      ;;
    --datasets)
      DATASETS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --models)
      MODELS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --seeds)
      SEEDS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --noise_levels)
      NOISE_LEVELS=()
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        NOISE_LEVELS+=("$1")
        shift
      done
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --noise_draws)
      DRAW_COUNT="$2"
      shift 2
      ;;
    --help|-h)
      cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --base_dir PATH        Override results base directory (default: ./results)
  --dataset NAME         Evaluate a single dataset
  --datasets D1 [D2 ..]  Evaluate specific datasets
  --models M1 [M2 ..]    Override model list
  --seeds S1 [S2 ..]     Override seed list
  --noise_levels N1 [...] Noise std values
  --device DEVICE        Torch device (default: cpu)
  --help                 Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Measurement-noise evaluation"
echo "  Base dir:    $BASE_DIR"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Models:      ${MODELS[*]}"
echo "  Seeds:       ${SEEDS[*]}"
echo "  Noise lvls:  ${NOISE_LEVELS[*]}"
echo "  Device:      $DEVICE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

#── MAIN LOOP ────────────────────────────────────────────
for dataset in "${DATASETS[@]}"; do
  echo "───────────────────────────────────────────────────────────────"
  echo "Evaluating noise sensitivity for dataset: $dataset"
  echo "───────────────────────────────────────────────────────────────"

  DATASET_RUN_DIR="$RUNS_BASE_DIR/$dataset"
  mkdir -p "$DATASET_RUN_DIR"
  LOG_PATH="$DATASET_RUN_DIR/noise_evaluate.log"
  : > "$LOG_PATH"

  if python inverse_neural_operator/evaluate_noise_all.py \
    --base_dir "$BASE_DIR" \
    --datasets "$dataset" \
    --models "${MODELS[@]}" \
    --seeds "${SEEDS[@]}" \
    --noise_levels "${NOISE_LEVELS[@]}" \
    --noise_draws "$DRAW_COUNT" \
    --device "$DEVICE" \
    >>"$LOG_PATH" 2>&1; then
    echo "  ✓ Completed noise evaluation for $dataset (log: $LOG_PATH)"
  else
    echo "  ✗ Noise evaluation failed for $dataset (see $LOG_PATH)"
  fi

  echo ""
done
