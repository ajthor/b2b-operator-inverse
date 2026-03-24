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

# DATASETS=(burgers_1d darcy_1d elastic_plate)
DATASETS=(darcy_1d elastic_plate)

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
    --help|-h)
      cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --base_dir PATH        Override results base directory (default: ./results)
  --dataset NAME         Generate plots for a single dataset
  --datasets D1 [D2 ..]  Generate plots for specific datasets
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

PLOT_SCRIPT="inverse_neural_operator/plots/plot_noise_comparison.py"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Noise sensitivity plotting"
echo "  Base dir: $BASE_DIR"
echo "  Requested datasets: ${DATASETS[*]}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

#── PREPARE DATASETS ─────────────────────────────────────
AVAILABLE_DATASETS=()
for dataset in "${DATASETS[@]}"; do
  SUMMARY_PATH="$RUNS_BASE_DIR/$dataset/measurement_noise_summary.json"
  if [[ -f "$SUMMARY_PATH" ]]; then
    AVAILABLE_DATASETS+=("$dataset")
  else
    echo "  ⚠ Skipping $dataset (summary not found at $SUMMARY_PATH)"
  fi
done

if [[ ${#AVAILABLE_DATASETS[@]} -eq 0 ]]; then
  echo "✗ No datasets available for plotting."
  exit 1
fi

echo "  → Using datasets: ${AVAILABLE_DATASETS[*]}"
echo ""

#── RUN PLOT SCRIPT ──────────────────────────────────────
if [[ ${#AVAILABLE_DATASETS[@]} -eq 1 ]]; then
  dataset="${AVAILABLE_DATASETS[0]}"
  SUMMARY_PATH="$RUNS_BASE_DIR/$dataset/measurement_noise_summary.json"
  OUTPUT_DIR="$RUNS_BASE_DIR/$dataset"
  echo "Generating measurement-noise plot for: $dataset"
  if python "$PLOT_SCRIPT" \
    --base_dir "$BASE_DIR" \
    --dataset "$dataset" \
    --summary_path "$SUMMARY_PATH" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | sed 's/^/    /'; then
    echo "  ✓ Plot completed → $OUTPUT_DIR"
  else
    echo "  ✗ Plotting failed for $dataset"
    exit 1
  fi
else
  OUTPUT_DIR="$RUNS_BASE_DIR"
  echo "Generating combined measurement-noise plot for datasets: ${AVAILABLE_DATASETS[*]}"
  if python "$PLOT_SCRIPT" \
    --base_dir "$BASE_DIR" \
    --datasets "${AVAILABLE_DATASETS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | sed 's/^/    /'; then
    echo "  ✓ Combined plot completed → $OUTPUT_DIR"
  else
    echo "  ✗ Combined plotting failed"
    exit 1
  fi
fi
