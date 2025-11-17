"""
Full Elastic Plate coefficient-noise evaluation pipeline with plotting.

For each trained model found in the logs, the script applies Gaussian noise to
the predicted inverse coefficients (alpha) at several preset levels, recomputes
inverse/forward/coefficient MSEs, saves per-model and aggregated metrics, and
generates publication-style comparison plots using shared utilities.

Example:
    python -m inverse_neural_operator.plots.plot_elastic_noise_comparison
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import math
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "inverse_neural_operator"
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from inverse_neural_operator.data.load_dataset import load_dataset
from inverse_neural_operator.models.load_model import load_models
from inverse_neural_operator.plots.utils.plot_utils import (
    display_name,
    get_model_color,
    setup_publication_style,
)


DEFAULT_DATASET = "elastic_plate"
DEFAULT_SEED = 1
DEFAULT_NOISE_LEVELS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
DEFAULT_MODEL_NAME = "model"
EXCLUDED_PLOT_MODELS = {"linear"}
EXCLUDED_METRIC_MODELS = {"linear", "linear_inverse", "inn_affine"}

device = "cpu"


@dataclass
class NoiseStats:
    inverse_sq: float = 0.0
    inverse_count: int = 0
    coeff_sq: float = 0.0
    coeff_count: int = 0

    def update_inverse(self, u_pred, u_true) -> None:
        self.inverse_sq += torch.sum((u_pred - u_true) ** 2).item()
        self.inverse_count += u_true.numel()

    def update_coeff(self, alpha_noisy, alpha_true) -> None:
        self.coeff_sq += torch.sum((alpha_noisy - alpha_true) ** 2).item()
        self.coeff_count += alpha_true.numel()

    def to_metrics(self, noise_std: float) -> Dict[str, float]:
        def mse(total_sq: float, count: int) -> float:
            return total_sq / count if count else 0.0

        return {
            "noise_std": noise_std,
            "inverse_mse": mse(self.inverse_sq, self.inverse_count),
            "inverse_count": self.inverse_count,
            "coeff_mse": mse(self.coeff_sq, self.coeff_count),
            "coeff_count": self.coeff_count,
        }


def _prepare_sample(sample: Iterable[torch.Tensor]):
    dev_tensors = [t.to(device) for t in sample]
    batched = [t.unsqueeze(0) for t in dev_tensors]
    return dev_tensors, batched


def format_noise_value(std: float) -> str:
    trimmed = f"{std:.6f}".rstrip("0").rstrip(".")
    return trimmed or format(std, "g")


def infer_dataset_name(log_path: Path) -> str:
    parents = list(log_path.parents)
    if len(parents) > 1 and parents[1].name:
        return parents[1].name
    return log_path.parent.name or DEFAULT_DATASET


def collect_model_log_dirs(base_path: Path, seed: int, model_filter: Optional[set[str]]) -> Dict[str, Path]:
    seed_dir_name = f"seed_{seed}"
    include = (lambda name: not model_filter or name in model_filter)

    if (base_path / "params.pth").exists():
        fallback = next(iter(model_filter)) if model_filter else DEFAULT_MODEL_NAME
        model_name = base_path.parent.name or fallback
        return {model_name: base_path} if include(model_name) else {}

    seed_candidate = base_path / seed_dir_name
    if (seed_candidate / "params.pth").exists():
        model_name = base_path.name
        return {model_name: seed_candidate} if include(model_name) else {}

    if not base_path.is_dir():
        return {}

    model_dirs = {}
    for entry in base_path.iterdir():
        if entry.name == "shared" or not entry.is_dir():
            continue
        seed_dir = entry / seed_dir_name
        if (seed_dir / "params.pth").exists() and include(entry.name):
            model_dirs[entry.name] = seed_dir
    return model_dirs


def evaluate_coeff_noise(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    dataset,
    noise_levels: List[float],
) -> Dict[str, Dict[str, float]]:
    stats_per_noise = {level: NoiseStats() for level in noise_levels}

    model.eval()

    with torch.no_grad():
        for sample in dataset:
            (X, u_true, Y, s_true), (X_b, u_b, Y_b, s_b) = _prepare_sample(sample)

            eval_out = evaluate_fn(
                model, (X_b, u_b, Y_b, s_b), input_function_encoder, output_function_encoder
            )
            if isinstance(eval_out, (tuple, list)):
                _, alpha_pred = eval_out
            else:
                alpha_pred = None

            if alpha_pred is None:
                raise ValueError(
                    "Model evaluate() must return alpha coefficients for coefficient noise analysis."
                )

            alpha_pred = alpha_pred.squeeze(0)
            alpha_true, _ = input_function_encoder.compute_coefficients(X_b, u_b)
            alpha_true = alpha_true.squeeze(0)

            for noise_std, stats in stats_per_noise.items():
                # Inverse noise: perturb observed outputs before feeding to inverse model
                noise_tensor = (
                    torch.zeros_like(s_b)
                    if noise_std == 0.0
                    else noise_std * torch.randn_like(s_b)
                )
                noisy_batch = (X_b, u_b, Y_b, s_b + noise_tensor)
                inv_eval = evaluate_fn(
                    model, noisy_batch, input_function_encoder, output_function_encoder
                )
                if isinstance(inv_eval, (tuple, list)):
                    u_pred_noise = inv_eval[0]
                else:
                    u_pred_noise = inv_eval
                stats.update_inverse(u_pred_noise.squeeze(0), u_true)

                # Coefficient noise: perturb latent coefficients for coeff/forward metrics
                alpha_noisy = (
                    alpha_pred
                    if noise_std == 0.0
                    else alpha_pred + noise_std * torch.randn_like(alpha_pred)
                )
                stats.update_coeff(alpha_noisy, alpha_true)

    return {
        (format_noise_value(level) if level > 0 else "0"): stats.to_metrics(level)
        for level, stats in stats_per_noise.items()
    }


def plot_aggregated_metric(
    aggregated: Dict[str, Dict[str, Dict]],
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    xlabel: str,
) -> None:
    noise_map: Dict[float, Dict[str, Dict]] = defaultdict(dict)
    for model_name, metrics in aggregated.items():
        for label, data in metrics.items():
            try:
                noise = float(label)
            except ValueError:
                continue
            noise_map[noise][model_name] = data

    if not noise_map:
        print(f"⚠️  No aggregated metrics found for coefficient plotting at {output_path}")
        return

    sorted_noises = sorted(noise_map.keys())
    noise_baseline = noise_map.get(0.0, {})
    models = sorted(
        {
            model
            for data in noise_map.values()
            for model in data.keys()
            if model not in EXCLUDED_METRIC_MODELS
        }
    )
    if noise_baseline:
        models = sorted(
            models,
            key=lambda m: noise_baseline.get(m, {}).get(metric_key, float("inf")),
        )
    if not models:
        print(f"⚠️  No models present in aggregated coefficient metrics at {output_path}")
        return
    legend_cols = len(models)

    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2)
    for model_name in models:
        points = [
            (noise, noise_map[noise][model_name][metric_key])
            for noise in sorted_noises
            if model_name in noise_map[noise]
            and noise_map[noise][model_name].get(metric_key, 0) > 0
        ]
        if points:
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=display_name(model_name),
                color=get_model_color(model_name),
            )

    if not ax.lines:
        plt.close(fig)
        print(f"⚠️  No positive {metric_key} values to plot in aggregate at {output_path}")
        return

    ax.set_title("")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted_noises)
    ax.set_yscale("log")
    fig.suptitle(title, y=1.01)
    legend_axes = fig.add_axes([0.10, 0.86, 0.90, 0.10], frameon=False)
    legend_axes.axis("off")
    legend = legend_axes.legend(
        [line for line in ax.lines],
        [display_name(m) for m in models],
        loc="center",
        ncol=legend_cols,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.2,
        borderpad=0.3,
    )
    ax.legend().remove()
    output_path = Path(output_path)
    base_path = output_path.with_suffix("") if output_path.suffix else output_path
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    for save_path in (png_path, pdf_path):
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved aggregated {metric_key} plot → {output_path}")


def resolve_results_root(dataset_name: str, override: Optional[str]) -> Path:
    return (
        Path(override).resolve()
        if override
        else PROJECT_ROOT / "results" / dataset_name
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Elastic Plate inverse models under coefficient noise and save MSE metrics."
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Path to logs root (defaults to logs/elastic_plate; accepts model/seed paths too)")
    parser.add_argument("--results_dir", type=str, default=None, help="Root directory to store coefficient-noise metrics (default mirrors results/dataset)")
    parser.add_argument("--noise_levels", type=float, nargs="*", help="Noise std values applied to inverse coefficients (default: 0,0.005,0.01,0.02,0.04)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default=None, help="Optional model name filter")
    return parser.parse_args()


def main():
    args = parse_args()

    setup_publication_style(figsize=(6.5, 3))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_log_dir = (
        Path(args.log_dir).resolve()
        if args.log_dir
        else PROJECT_ROOT / "logs" / DEFAULT_DATASET
    )
    if not base_log_dir.exists():
        print(f"✗ Log directory not found: {base_log_dir}")
        return

    model_filter = {args.model} if args.model else None
    model_log_dirs = collect_model_log_dirs(base_log_dir, args.seed, model_filter)
    if not model_log_dirs:
        target = args.model or "any model"
        print(f"✗ No models found at {base_log_dir} for {target}")
        return

    noise_levels = (
        sorted(set(args.noise_levels)) if args.noise_levels else DEFAULT_NOISE_LEVELS
    )

    aggregated: Dict[str, Dict[str, Dict]] = {}

    for model_name, model_log_dir in sorted(model_log_dirs.items()):
        dataset_name = infer_dataset_name(model_log_dir)
        results_root = resolve_results_root(dataset_name, args.results_dir)
        model_results_dir = results_root / model_name
        os.makedirs(model_results_dir, exist_ok=True)

        params = torch.load(model_log_dir / "params.pth", weights_only=False)
        test_dataset, dataset_info = load_dataset(
            params.dataset, params, device, split="test", return_info=True
        )

        try:
            input_encoder, output_encoder, model, evaluate_fn = load_models(
                log_dir=str(model_log_dir),
                dataset_info=dataset_info,
                params=params,
                device=device,
            )
        except Exception as exc:
            print(f"✗ {model_name}: failed to load model ({exc}); skipping.")
            continue

        metrics = evaluate_coeff_noise(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_encoder,
            output_function_encoder=output_encoder,
            dataset=test_dataset,
            noise_levels=noise_levels,
        )

        metrics_path = model_results_dir / "coefficient_noise_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ {model_name}: saved coefficient-noise metrics → {metrics_path}")

        aggregated.setdefault(dataset_name, {})[model_name] = metrics

    for dataset_name, data in aggregated.items():
        summary_root = resolve_results_root(dataset_name, args.results_dir)
        os.makedirs(summary_root, exist_ok=True)
        summary_path = summary_root / "coefficient_noise_metrics.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved aggregated metrics → {summary_path}")
        plot_aggregated_metric(
            data,
            metric_key="inverse_mse",
            title="Noise Sensitivity (Input Space)",
            ylabel="Inverse MSE (log scale)",
            xlabel="Observation noise std",
            output_path=summary_root / "coefficient_noise_plot.png",
        )
        plot_aggregated_metric(
            data,
            metric_key="coeff_mse",
            title="Coefficient Noise Sensitivity (Aggregated Coefficients)",
            ylabel="Coefficient MSE (log scale)",
            output_path=summary_root / "coefficient_noise_coeff_plot.png",
            xlabel="Coefficient noise std",
        )


if __name__ == "__main__":
    main()
