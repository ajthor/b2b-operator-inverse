"""
Compare Darcy inverse/forward MSE across noise levels using precomputed metrics.

Usage example:
    python -m inverse_neural_operator.plots.plot_darcy_noise_comparison \
        --results_root results/darcy_1d --noises 0 0.005 0.01 0.02 0.04
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

DEFAULT_RESULTS_ROOT = Path("results") / "darcy_1d"
DEFAULT_NOISES = [0.0, 0.005, 0.01, 0.02, 0.04]


def format_noise_value(std: float) -> str:
    trimmed = f"{std:.6f}".rstrip("0").rstrip(".")
    return trimmed or format(std, "g")


def discover_noise_levels(results_root: Path) -> List[float]:
    noises = []
    metrics_path = results_root / "metrics.json"
    if metrics_path.exists():
        noises.append(0.0)
    for entry in results_root.iterdir():
        if not entry.is_dir():
            continue
        metrics_file = entry / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            noise = float(entry.name)
        except ValueError:
            continue
        noises.append(noise)
    return sorted(set(noises))


def load_metrics(results_root: Path, noise: float) -> Optional[Dict[str, Dict]]:
    label = format_noise_value(noise) if noise > 0 else ""
    metrics_dir = results_root if noise == 0 else results_root / label
    metrics_path = metrics_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_data(
    results_root: Path, noises: List[float], models_filter: Optional[set[str]]
) -> Dict[float, Dict[str, Dict]]:
    comparison = {}
    for noise in noises:
        metrics = load_metrics(results_root, noise)
        if metrics is None:
            print(f"⚠️  Skipping noise {format_noise_value(noise)} (metrics.json not found).")
            continue
        filtered = {
            model: data
            for model, data in metrics.items()
            if not models_filter or model in models_filter
        }
        if not filtered:
            print(
                f"⚠️  No matching models for noise {format_noise_value(noise)} "
                f"in {results_root}"
            )
            continue
        comparison[noise] = filtered
    return comparison


def plot_comparison(comparison: Dict[float, Dict[str, Dict]], output_path: Path) -> None:
    sorted_noises = sorted(comparison.keys())
    if not sorted_noises:
        print("✗ No comparison data available; nothing to plot.")
        return

    all_models = sorted({model for data in comparison.values() for model in data.keys()})
    if not all_models:
        print("✗ No models found in metrics; nothing to plot.")
        return

    plt.figure(figsize=(10, 5))
    for model_name in all_models:
        noise_vals = []
        metric_vals = []
        for noise in sorted_noises:
            model_metrics = comparison[noise].get(model_name)
            if model_metrics is None:
                continue
            inverse_mse = model_metrics.get("inverse_mse")
            if inverse_mse is None or inverse_mse <= 0:
                continue
            metric_vals.append(inverse_mse)
            noise_vals.append(noise)
        if not metric_vals:
            continue
        plt.plot(noise_vals, metric_vals, marker="o", linewidth=2, label=model_name)

    plt.title("Inverse Prediction MSE vs. Inverse Noise Std")
    plt.xlabel("Inverse noise std")
    plt.ylabel("Inverse MSE (log scale)")
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted_noises)
    plt.yscale("log")
    plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison plot → {output_path}")


def save_summary(
    comparison: Dict[float, Dict[str, Dict]],
    output_path: Path,
) -> None:
    summary = {
        (format_noise_value(noise) if noise > 0 else "0"): metrics
        for noise, metrics in sorted(comparison.items())
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved metrics summary → {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Darcy inverse/forward MSE across noise levels using existing metrics.json files."
    )
    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT), help="Root directory containing per-noise metrics.json files (default: results/darcy_1d)")
    parser.add_argument("--noises", type=float, nargs="*", help="Noise std values to include (defaults to detected values or preset list if none found)")
    parser.add_argument("--models", type=str, nargs="*", help="Optional list of model names to include in the comparison")
    parser.add_argument("--output", type=str, default=None, help="Path to save the comparison plot (default: <results_root>/noise_comparison.png)")
    parser.add_argument("--summary", type=str, default=None, help="Path to save aggregated metrics JSON (default: <results_root>/metrics_summary.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    if not results_root.exists():
        print(f"✗ Results root not found: {results_root}")
        return

    discovered_noises = discover_noise_levels(results_root)
    if args.noises:
        target_noises = sorted(set(args.noises))
    elif discovered_noises:
        target_noises = discovered_noises
    else:
        target_noises = DEFAULT_NOISES

    models_filter = set(args.models) if args.models else None

    comparison = build_comparison_data(results_root, target_noises, models_filter)
    if not comparison:
        print("✗ No metrics available for requested configuration.")
        return

    summary_path = (
        Path(args.summary).resolve()
        if args.summary
        else results_root / "metrics_summary.json"
    )
    save_summary(comparison, summary_path)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else results_root / "noise_comparison.png"
    )
    plot_comparison(comparison, output_path)


if __name__ == "__main__":
    main()
