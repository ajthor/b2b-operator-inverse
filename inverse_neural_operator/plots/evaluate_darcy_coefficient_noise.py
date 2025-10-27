"""
Evaluate Darcy inverse models under coefficient noise.

For each trained model found in the logs, the script applies Gaussian noise to the
predicted inverse coefficients (alpha) at several preset levels, recomputes the
inverse and forward MSEs, and stores the aggregated results as JSON files.

Example:
    python -m inverse_neural_operator.plots.evaluate_darcy_coefficient_noise
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "inverse_neural_operator"
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from inverse_neural_operator.data.load_dataset import load_dataset
from inverse_neural_operator.models.load_model import load_models
from inverse_neural_operator.b2b.load_model import load_forward_model

DEFAULT_DATASET = "darcy_1d"
DEFAULT_SEED = 1
DEFAULT_NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.04]
DEFAULT_MODEL_NAME = "model"

device = "cpu"


def to_device(*tensors):
    return tuple(t.to(device) for t in tensors)


def add_batch_dim(*tensors):
    return tuple(t.unsqueeze(0) for t in tensors)


def format_noise_value(std: float) -> str:
    trimmed = f"{std:.6f}".rstrip("0").rstrip(".")
    return trimmed or format(std, "g")


def infer_dataset_name(log_path: Path) -> str:
    if len(log_path.parents) > 1 and log_path.parents[1].name:
        return log_path.parents[1].name
    if log_path.parent.name:
        return log_path.parent.name
    return DEFAULT_DATASET


def collect_model_log_dirs(base_path: Path, seed: int, model_filter: Optional[set[str]]) -> Dict[str, Path]:
    seed_dir_name = f"seed_{seed}"
    model_dirs = {}

    if (base_path / "params.pth").exists():
        fallback_model = next(iter(model_filter)) if model_filter else DEFAULT_MODEL_NAME
        model_name = base_path.parent.name or fallback_model
        if not model_filter or model_name in model_filter:
            model_dirs[model_name] = base_path
        return model_dirs

    seed_candidate = base_path / seed_dir_name
    if (seed_candidate / "params.pth").exists():
        model_name = base_path.name
        if not model_filter or model_name in model_filter:
            model_dirs[model_name] = seed_candidate
        return model_dirs

    if not base_path.is_dir():
        return model_dirs

    for entry in base_path.iterdir():
        if not entry.is_dir() or entry.name == "shared":
            continue
        seed_dir = entry / seed_dir_name
        if not (seed_dir / "params.pth").exists():
            continue
        if model_filter and entry.name not in model_filter:
            continue
        model_dirs[entry.name] = seed_dir

    return model_dirs


def evaluate_coeff_noise(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    dataset,
    noise_levels: List[float],
) -> Dict[str, Dict[str, float]]:
    metrics = {
        noise: {
            "inverse_sq": 0.0,
            "forward_sq": 0.0,
            "inverse_count": 0,
            "forward_count": 0,
            "coeff_sq": 0.0,
            "coeff_count": 0,
        }
        for noise in noise_levels
    }

    model.eval()
    forward_model.eval()

    with torch.no_grad():
        for sample in dataset:
            X, u_true, Y, s_observed = to_device(*sample)
            X_b, u_b, Y_b, s_b = add_batch_dim(X, u_true, Y, s_observed)

            u_pred_base, alpha_pred = evaluate_fn(
                model, (X_b, u_b, Y_b, s_b), input_function_encoder, output_function_encoder
            )
            alpha_pred = alpha_pred.squeeze(0)
            alpha_true, _ = input_function_encoder.compute_coefficients(X_b, u_b)
            alpha_true = alpha_true.squeeze(0)

            for noise_std in noise_levels:
                alpha_noisy = (
                    alpha_pred
                    if noise_std == 0.0
                    else alpha_pred + noise_std * torch.randn_like(alpha_pred)
                )

                alpha_noisy_b = alpha_noisy.unsqueeze(0)
                u_noisy = input_function_encoder(X_b, alpha_noisy_b).squeeze(0)
                beta_pred = forward_model.forward(alpha_noisy_b)
                s_resim = output_function_encoder(Y_b, beta_pred).squeeze(0)

                stats = metrics[noise_std]
                stats["inverse_sq"] += torch.sum((u_noisy - u_true) ** 2).item()
                stats["forward_sq"] += torch.sum((s_resim - s_observed) ** 2).item()
                stats["inverse_count"] += u_true.numel()
                stats["forward_count"] += s_observed.numel()
                stats["coeff_sq"] += torch.sum((alpha_noisy - alpha_true) ** 2).item()
                stats["coeff_count"] += alpha_true.numel()

    results = {}
    for noise_std in noise_levels:
        stats = metrics[noise_std]
        noise_label = format_noise_value(noise_std) if noise_std > 0 else "0"
        inverse_mse = (
            stats["inverse_sq"] / stats["inverse_count"] if stats["inverse_count"] else 0.0
        )
        forward_mse = (
            stats["forward_sq"] / stats["forward_count"] if stats["forward_count"] else 0.0
        )
        coeff_mse = (
            stats["coeff_sq"] / stats["coeff_count"] if stats["coeff_count"] else 0.0
        )
        results[noise_label] = {
            "noise_std": noise_std,
            "inverse_mse": inverse_mse,
            "forward_mse": forward_mse,
            "inverse_count": stats["inverse_count"],
            "forward_count": stats["forward_count"],
            "coeff_mse": coeff_mse,
            "coeff_count": stats["coeff_count"],
        }
    return results


def plot_aggregated_metrics(
    aggregated: Dict[str, Dict[str, Dict]], output_path: Path
) -> None:
    noise_map = defaultdict(dict)
    for model_name, metrics in aggregated.items():
        for noise_label, data in metrics.items():
            try:
                noise = float(noise_label)
            except ValueError:
                continue
            noise_map[noise][model_name] = data

    if not noise_map:
        print(f"⚠️  No aggregated metrics found for plotting at {output_path}")
        return

    sorted_noises = sorted(noise_map.keys())
    models = sorted({model for data in noise_map.values() for model in data.keys()})
    if not models:
        print(f"⚠️  No models present in aggregated metrics at {output_path}")
        return

    plt.figure(figsize=(10, 5))
    for model_name in models:
        noise_vals, inverse_vals = [], []
        for noise in sorted_noises:
            model_metrics = noise_map[noise].get(model_name)
            if not model_metrics:
                continue
            inv = model_metrics.get("inverse_mse")
            if inv is None or inv <= 0:
                continue
            noise_vals.append(noise)
            inverse_vals.append(inv)
        if not inverse_vals:
            continue
        plt.plot(noise_vals, inverse_vals, marker="o", linewidth=2, label=model_name)

    plt.title("Coefficient Noise Sensitivity (Aggregated)")
    plt.xlabel("Coefficient noise std")
    plt.ylabel("Inverse MSE (log scale)")
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted_noises)
    plt.yscale("log")
    plt.tight_layout()
    plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved aggregated coefficient-noise plot → {output_path}")


def plot_aggregated_coeff_metrics(
    aggregated: Dict[str, Dict[str, Dict]], output_path: Path
) -> None:
    noise_map = defaultdict(dict)
    for model_name, metrics in aggregated.items():
        for noise_label, data in metrics.items():
            try:
                noise = float(noise_label)
            except ValueError:
                continue
            noise_map[noise][model_name] = data

    if not noise_map:
        print(f"⚠️  No aggregated metrics found for coefficient plotting at {output_path}")
        return

    sorted_noises = sorted(noise_map.keys())
    models = sorted({model for data in noise_map.values() for model in data.keys()})
    if not models:
        print(f"⚠️  No models present in aggregated coefficient metrics at {output_path}")
        return

    plt.figure(figsize=(10, 5))
    for model_name in models:
        noise_vals, coeff_vals = [], []
        for noise in sorted_noises:
            model_metrics = noise_map[noise].get(model_name)
            if not model_metrics:
                continue
            coeff = model_metrics.get("coeff_mse")
            if coeff is None or coeff <= 0:
                continue
            noise_vals.append(noise)
            coeff_vals.append(coeff)
        if not coeff_vals:
            continue
        plt.plot(noise_vals, coeff_vals, marker="o", linewidth=2, label=model_name)

    if plt.gca().lines:
        plt.title("Coefficient Noise Sensitivity (Aggregated Coefficients)")
        plt.xlabel("Coefficient noise std")
        plt.ylabel("Coefficient MSE (log scale)")
        plt.grid(True, alpha=0.3)
        plt.xticks(sorted_noises)
        plt.yscale("log")
        plt.tight_layout()
        plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        os.makedirs(output_path.parent, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved aggregated coefficient-space noise plot → {output_path}")
    else:
        plt.close()
        print(f"⚠️  No positive coefficient MSE values to plot in aggregate at {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Darcy inverse models under coefficient noise and save MSE metrics."
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Path to logs root (defaults to logs/darcy_1d; accepts model/seed paths too)")
    parser.add_argument("--results_dir", type=str, default=None, help="Root directory to store coefficient-noise metrics (default mirrors results/dataset)")
    parser.add_argument("--noise_levels", type=float, nargs="*", help="Noise std values applied to inverse coefficients (default: 0,0.005,0.01,0.02,0.04)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default=None, help="Optional model name filter")
    return parser.parse_args()


def main():
    args = parse_args()

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
        results_root = (
            Path(args.results_dir).resolve()
            if args.results_dir
            else PROJECT_ROOT / "results" / dataset_name
        )
        model_results_dir = results_root / model_name
        os.makedirs(model_results_dir, exist_ok=True)

        params = torch.load(model_log_dir / "params.pth", weights_only=False)
        test_dataset, dataset_info = load_dataset(
            params.dataset, params, device, split="test", return_info=True
        )

        input_encoder, output_encoder, model, evaluate_fn = load_models(
            log_dir=str(model_log_dir),
            dataset_info=dataset_info,
            params=params,
            device=device,
        )
        forward_model = load_forward_model(
            log_dir=str(model_log_dir), forward_model_name="b2b_nonlinear", device=device
        )

        metrics = evaluate_coeff_noise(
            model=model,
            evaluate_fn=evaluate_fn,
            input_function_encoder=input_encoder,
            output_function_encoder=output_encoder,
            forward_model=forward_model,
            dataset=test_dataset,
            noise_levels=noise_levels,
        )

        model_metrics_path = model_results_dir / "coefficient_noise_metrics.json"
        with model_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ {model_name}: saved coefficient-noise metrics → {model_metrics_path}")

        aggregated.setdefault(dataset_name, {})[model_name] = metrics

    for dataset_name, data in aggregated.items():
        summary_root = (
            Path(args.results_dir).resolve()
            if args.results_dir
            else PROJECT_ROOT / "results" / dataset_name
        )
        os.makedirs(summary_root, exist_ok=True)
        summary_path = summary_root / "coefficient_noise_metrics.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved aggregated metrics → {summary_path}")
        plot_aggregated_metrics(data, summary_root / "coefficient_noise_plot.png")
        plot_aggregated_coeff_metrics(
            data, summary_root / "coefficient_noise_coeff_plot.png"
        )


if __name__ == "__main__":
    main()
