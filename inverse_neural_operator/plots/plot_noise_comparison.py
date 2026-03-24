"""
Plot measurement-noise sensitivity curves for inverse models.

This script reads the aggregated metrics produced by evaluate_noise_all.py and
generates publication-style comparison plots. Run evaluate_noise_all.sh (or the
Python CLI) beforehand so that measurement_noise_summary.json exists.

Multiple datasets can be plotted simultaneously. Each dataset occupies a panel
in a shared figure, and all models share one legend so that the color/marker
mapping remains consistent across panels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from plots.utils.plot_utils import (
    display_name,
    get_model_color,
    get_model_marker,
    setup_publication_style,
)

DEFAULT_DATASET = "darcy_1d"
DEFAULT_DATASETS = [DEFAULT_DATASET]
FIG_WIDTH = 5.5
FIG_HEIGHT = 2.6
EXCLUDED_MODELS = {
    # "elastic_plate": {"linear"},
}
DATASET_NAME_OVERRIDES = {"darcy_1d": "Darcy"}


def load_summary(
    path: Path, model_filter: Optional[str], exclude: set[str]
) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Measurement-noise summary not found at {path}. "
            "Run evaluate_noise_all.py first."
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    models = payload.get("models", {})
    if model_filter:
        models = {model_filter: models.get(model_filter, {})}
    return {k: v for k, v in models.items() if v and k not in exclude}


def dataset_display_name(name: str) -> str:
    return DATASET_NAME_OVERRIDES.get(name, name.replace("_", " ").title())


def _collect_model_order(
    dataset_models: Sequence[Tuple[str, Dict[str, Dict]]],
    metric_key: str,
) -> List[str]:
    baseline: Dict[str, float] = {}
    for _, models_data in dataset_models:
        for model_name, entries in models_data.items():
            zero_entry = entries.get("0")
            value = zero_entry.get(metric_key) if zero_entry else None
            candidate = value if value is not None else float("inf")
            if model_name not in baseline or candidate < baseline[model_name]:
                baseline[model_name] = candidate
    if not baseline:
        return []
    return sorted(baseline.keys(), key=lambda m: baseline.get(m, float("inf")))


def plot_metric_panels(
    dataset_models: Sequence[Tuple[str, Dict[str, Dict]]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not dataset_models:
        print("⚠️  No datasets available for plotting.")
        return

    ordered_models = _collect_model_order(dataset_models, metric_key)
    if not ordered_models:
        print(f"⚠️  No baseline values for {metric_key}; skipping plot.")
        return
    model_indices = {name: idx for idx, name in enumerate(ordered_models)}

    num_panels = len(dataset_models)
    fig, axes = plt.subplots(
        1, num_panels, sharey=True, figsize=(FIG_WIDTH, FIG_HEIGHT)
    )
    if num_panels == 1:
        axes = [axes]  # type: ignore[list-item]

    all_y_values = []
    legend_handles = []
    legend_labels = []
    plotted_models = set()
    for ax, (dataset_name, models_data) in zip(axes, dataset_models):
        axis_has_lines = False
        for model_name in ordered_models:
            if model_name not in models_data:
                continue
            entries = []
            for data in models_data[model_name].values():
                value = data.get(metric_key)
                noise_std = data.get("noise_std")
                if value is None or noise_std is None or value <= 0:
                    continue
                entries.append((noise_std, max(value, 1e-12)))
            entries.sort(key=lambda item: item[0])
            if not entries:
                continue
            xs, ys = zip(*entries)
            all_y_values.extend(ys)
            idx = model_indices[model_name]
            line = ax.plot(
                xs,
                ys,
                linewidth=1.0,
                label=display_name(model_name),
                color=get_model_color(model_name, idx),
                marker=get_model_marker(model_name, idx),
            )[0]
            axis_has_lines = True
            if model_name not in plotted_models:
                legend_handles.append(line)
                legend_labels.append(display_name(model_name))
                plotted_models.add(model_name)

        ax.set_title(dataset_display_name(dataset_name))
        ax.grid(True, alpha=0.3)
        # ax.set_yscale("log")
        if not axis_has_lines:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=5,
            )

    axes[0].set_ylabel(ylabel)
    if not all_y_values:
        print(f"⚠️  No positive {metric_key} values available for plotting.")
        plt.close(fig)
        return

    # ymin = min(all_y_values)
    # y_lower = max(ymin * 0.8, 1e-3)
    # y_upper = 2.0
    # if y_lower >= y_upper:
    #     y_lower = max(1e-3, y_upper * 0.5)
    # for ax in axes:
    #     ax.set_ylim(bottom=y_lower, top=y_upper)

    ax.set_ylim(bottom=0.0, top=1.0)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=6,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.0,
        )

    fig.supxlabel("Measurement noise std")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved plot → {output_path}")
    print(f"✓ Saved plot → {pdf_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot measurement-noise sensitivity curves for inverse models."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Base directory containing runs/ (default: results)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to include as panels (overrides --dataset)",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Path to measurement_noise_summary.json "
        "(defaults to <base_dir>/runs/<dataset>/measurement_noise_summary.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to summary directory or runs/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name filter",
    )
    parser.add_argument(
        "--exclude_models",
        nargs="*",
        default=None,
        help="Model names to exclude from plotting (defaults depend on dataset)",
    )
    return parser.parse_args()


def main():
    setup_publication_style(figsize=(FIG_WIDTH, FIG_HEIGHT))
    args = parse_args()
    dataset_list = (
        args.datasets
        if args.datasets is not None
        else ([args.dataset] if args.dataset else DEFAULT_DATASETS)
    )
    dataset_list = [d for d in dataset_list if d]
    if not dataset_list:
        print("✗ No datasets specified.")
        return

    dataset_models: List[Tuple[str, Dict[str, Dict]]] = []
    summary_dirs: List[Path] = []
    for dataset in dataset_list:
        summary_path = (
            Path(args.summary_path).resolve()
            if args.summary_path and len(dataset_list) == 1
            else Path(args.base_dir)
            / "runs"
            / dataset
            / "measurement_noise_summary.json"
        )
        exclude = (
            set(args.exclude_models)
            if args.exclude_models is not None
            else EXCLUDED_MODELS.get(dataset, set())
        )
        try:
            models_data = load_summary(summary_path, args.model, exclude)
        except FileNotFoundError as exc:
            print(f"✗ {exc}")
            continue

        if not models_data:
            print(
                f"✗ No models available to plot for dataset {dataset}. "
                "Ensure evaluation has been run."
            )
            continue
        dataset_models.append((dataset, models_data))
        summary_dirs.append(summary_path.parent)

    if not dataset_models:
        print("✗ No datasets available for plotting.")
        return

    if args.summary_path and len(dataset_list) > 1:
        print(
            "⚠️  Provided --summary_path ignored for multi-dataset plots; "
            "using dataset-specific summaries."
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (
            summary_dirs[0] if len(summary_dirs) == 1 else Path(args.base_dir) / "runs"
        )
    )

    plot_metric_panels(
        dataset_models,
        metric_key="inverse_rel_l2_mean",
        ylabel="Inverse relative L2 (mean)",
        output_path=output_dir / "measurement_noise_plot.png",
    )


if __name__ == "__main__":
    main()
