"""
Evaluate inverse models across multiple seeds and summarize performance.

This script mirrors the workflow of `evaluate_b2b.py`, but focuses on the
inverse models that map observed outputs back to input parameters. For each
model/seed combination it
  * loads the trained inverse model together with the shared function encoders,
  * evaluates reconstruction quality on the configured test split,
  * optionally computes forward re-simulation diagnostics when a forward model
    checkpoint is available,
  * aggregates dataset-wide error statistics, and
  * exports per-seed results alongside aggregate summaries to CSV and TXT files.

Example usage:
    python inverse_neural_operator/evaluate_models.py \
        --dataset burgers_1d \
        --log_base_dir /store/at46867/b2b_operator_inverse/burgers_1d \
        --models nonlinear_inverse cinn_affine \
        --seeds 1 2 3 4 5 \
        --output_dir results/evaluations/burgers_1d/inverse
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from b2b.load_model import load_forward_model
from data.load_dataset import load_dataset
from models.load_model import load_models


# Order in which metrics are reported (key, human readable label)
METRIC_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("inverse_rel_l2", "Inverse Rel L2"),
    ("inverse_mse", "Inverse MSE"),
    ("alpha_rel_l2", "Alpha Rel L2"),
    ("alpha_mse", "Alpha MSE"),
    ("resim_coeff_rel_l2", "Re-sim Coeff Rel L2"),
    ("resim_coeff_mse", "Re-sim Coeff MSE"),
    ("resim_pred_rel_l2", "Re-sim Output Rel L2"),
    ("resim_pred_mse", "Re-sim Output MSE"),
)


def format_value(value: float | None, precision: int = 6) -> str:
    """Format numeric values, keeping blanks for missing entries."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:.{precision}e}"


def compute_statistics(values: Iterable[float]) -> Dict[str, float]:
    """Compute standard statistics for a collection of floats."""
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_reference_dataset(
    log_base_dir: str,
    dataset_name: str,
    model_names: Iterable[str],
    seeds: Iterable[int],
    device: str,
    batch_size: int,
):
    """
    Locate the first available trained run and load the corresponding test dataset.
    """
    for model_name in model_names:
        for seed in seeds:
            run_dir = os.path.join(log_base_dir, model_name, f"seed_{seed}")
            params_path = os.path.join(run_dir, "params.pth")
            if not os.path.exists(params_path):
                continue

            params = torch.load(params_path, weights_only=False)
            params_copy = copy.deepcopy(params)
            params_copy.dataset = dataset_name
            params_copy.batch_size = batch_size
            params_copy.device = device

            test_dataset, dataset_info = load_dataset(
                dataset_name=dataset_name,
                params=params_copy,
                device=device,
                split="test",
                return_info=True,
            )
            return test_dataset, dataset_info

    raise FileNotFoundError(
        f"Could not locate params.pth under {log_base_dir} "
        f"for any of the requested models {list(model_names)} and seeds {list(seeds)}."
    )


def call_evaluate_fn(
    evaluate_fn,
    model,
    sample,
    input_encoder,
    output_encoder,
):
    """
    Invoke a model-specific evaluate() helper while handling signature quirks.
    """
    try:
        return evaluate_fn(model, sample, input_encoder, output_encoder)
    except TypeError:
        X, u, Y, s = sample
        point_dict = {"X": X, "u": u, "Y": Y, "s": s}
        return evaluate_fn(model, point_dict, input_encoder, output_encoder)


def accumulate_sample_metrics(
    accumulators: Dict[str, float],
    sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model,
    evaluate_fn,
    input_encoder,
    output_encoder,
    forward_model,
) -> None:
    """
    Evaluate a single (batched) sample and update running error totals.
    """
    X, u, Y, s = sample

    result = call_evaluate_fn(evaluate_fn, model, sample, input_encoder, output_encoder)

    if isinstance(result, tuple):
        pred_u = result[0]
        alpha_pred = result[1] if len(result) > 1 else None
    else:
        pred_u = result
        alpha_pred = None

    if isinstance(pred_u, (tuple, list)):
        pred_u = pred_u[0]
    if isinstance(alpha_pred, (tuple, list)):
        alpha_pred = alpha_pred[0]

    if not isinstance(pred_u, torch.Tensor):
        raise TypeError("Expected evaluate() to return a tensor prediction for u.")

    # Inverse reconstruction metrics (u space)
    accumulators["u_error_sq"] += torch.sum((pred_u - u) ** 2).item()
    accumulators["u_target_sq"] += torch.sum(u ** 2).item()
    accumulators["u_elements"] += u.numel()

    # Coefficient reconstruction metrics (alpha space)
    if alpha_pred is not None and isinstance(alpha_pred, torch.Tensor):
        alpha_target, _ = input_encoder.compute_coefficients(X, u)
        accumulators["alpha_error_sq"] += torch.sum((alpha_pred - alpha_target) ** 2).item()
        accumulators["alpha_target_sq"] += torch.sum(alpha_target ** 2).item()
        accumulators["alpha_elements"] += alpha_target.numel()

    # Forward re-simulation diagnostics (beta and s space)
    if forward_model is not None and isinstance(alpha_pred, torch.Tensor):
        beta_target, _ = output_encoder.compute_coefficients(Y, s)
        beta_resim = forward_model(alpha_pred)
        s_resim = output_encoder(Y, beta_resim)

        accumulators["beta_error_sq"] += torch.sum((beta_resim - beta_target) ** 2).item()
        accumulators["beta_target_sq"] += torch.sum(beta_target ** 2).item()
        accumulators["beta_elements"] += beta_target.numel()

        accumulators["s_error_sq"] += torch.sum((s_resim - s) ** 2).item()
        accumulators["s_target_sq"] += torch.sum(s ** 2).item()
        accumulators["s_elements"] += s.numel()


def finalize_metrics(accumulators: Dict[str, float]) -> Dict[str, float | None]:
    """Convert accumulated sums into interpretable metrics."""
    def safe_ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 0.0:
            return None
        return numerator / denominator

    def safe_relative(error_sq: float, target_sq: float) -> float | None:
        if target_sq <= 0.0 or error_sq < 0.0:
            return None
        return math.sqrt(error_sq / target_sq)

    metrics = {
        "inverse_mse": safe_ratio(accumulators["u_error_sq"], accumulators["u_elements"]),
        "inverse_rel_l2": safe_relative(accumulators["u_error_sq"], accumulators["u_target_sq"]),
        "alpha_mse": safe_ratio(accumulators["alpha_error_sq"], accumulators["alpha_elements"]),
        "alpha_rel_l2": safe_relative(accumulators["alpha_error_sq"], accumulators["alpha_target_sq"]),
        "resim_coeff_mse": safe_ratio(accumulators["beta_error_sq"], accumulators["beta_elements"]),
        "resim_coeff_rel_l2": safe_relative(accumulators["beta_error_sq"], accumulators["beta_target_sq"]),
        "resim_pred_mse": safe_ratio(accumulators["s_error_sq"], accumulators["s_elements"]),
        "resim_pred_rel_l2": safe_relative(accumulators["s_error_sq"], accumulators["s_target_sq"]),
        "num_samples": accumulators["num_samples"],
    }

    # Ensure consistent ordering
    return metrics


def evaluate_model_for_seed(
    model_name: str,
    seed: int,
    run_dir: str,
    dataset,
    dataset_info,
    batch_size: int,
    device: torch.device,
    device_str: str,
):
    """
    Load a specific model/seed run and compute dataset-wide metrics.
    """
    params_path = os.path.join(run_dir, "params.pth")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing params.pth for {model_name} seed {seed} under {run_dir}")

    params = torch.load(params_path, weights_only=False)
    params.dataset = getattr(params, "dataset", dataset_info.get("dataset", None))

    input_encoder, output_encoder, model, evaluate_fn = load_models(
        log_dir=run_dir,
        dataset_info=dataset_info,
        params=params,
        device=device_str,
    )

    forward_model = None
    forward_model_name = getattr(params, "forward_model", None)
    if forward_model_name:
        try:
            forward_model = load_forward_model(run_dir, forward_model_name, device=device_str)
        except FileNotFoundError:
            print(
                f"  ⚠ Forward model '{forward_model_name}' not found for {model_name} seed {seed}. "
                "Skipping re-simulation diagnostics."
            )
        except Exception as exc:
            print(
                f"  ⚠ Could not load forward model '{forward_model_name}' for {model_name} seed {seed}: {exc}. "
                "Skipping re-simulation diagnostics."
            )

    model.eval()
    input_encoder.eval()
    output_encoder.eval()
    if forward_model is not None:
        forward_model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    accumulators = defaultdict(float)
    accumulators.update(
        {
            "u_error_sq": 0.0,
            "u_target_sq": 0.0,
            "u_elements": 0.0,
            "alpha_error_sq": 0.0,
            "alpha_target_sq": 0.0,
            "alpha_elements": 0.0,
            "beta_error_sq": 0.0,
            "beta_target_sq": 0.0,
            "beta_elements": 0.0,
            "s_error_sq": 0.0,
            "s_target_sq": 0.0,
            "s_elements": 0.0,
            "num_samples": 0.0,
        }
    )

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                raise ValueError("Expected dataloader to yield (X, u, Y, s) tuples.")

            X, u, Y, s = batch
            batch_size_actual = X.shape[0]

            for idx in range(batch_size_actual):
                sample = (
                    X[idx : idx + 1].to(device),
                    u[idx : idx + 1].to(device),
                    Y[idx : idx + 1].to(device),
                    s[idx : idx + 1].to(device),
                )

                accumulate_sample_metrics(
                    accumulators=accumulators,
                    sample=sample,
                    model=model,
                    evaluate_fn=evaluate_fn,
                    input_encoder=input_encoder,
                    output_encoder=output_encoder,
                    forward_model=forward_model,
                )
                accumulators["num_samples"] += 1

    return finalize_metrics(accumulators)


def compute_model_statistics(seed_results: Dict[int, Dict[str, float | None]]):
    """Gather aggregate statistics across seeds for each metric."""
    stats: Dict[str, Dict[str, float]] = {}
    for metric_key, _ in METRIC_FIELDS:
        values = [
            metrics[metric_key]
            for metrics in seed_results.values()
            if metrics.get(metric_key) is not None
        ]
        if values:
            stats[metric_key] = compute_statistics(values)
    return stats


def print_console_summary(all_results: Dict[str, Dict[int, Dict[str, float | None]]]):
    """Print a concise summary to stdout."""
    for model_name, seed_metrics in all_results.items():
        if not seed_metrics:
            print(f"\nModel: {model_name}\n  (no valid runs found)")
            continue

        print(f"\nModel: {model_name}")
        headers = ["Seed"] + [label for _, label in METRIC_FIELDS]
        table_rows = []

        for seed in sorted(seed_metrics.keys()):
            metrics = seed_metrics[seed]
            row = [seed]
            for metric_key, _ in METRIC_FIELDS:
                row.append(format_value(metrics.get(metric_key)))
            table_rows.append(row)

        print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        stats = compute_model_statistics(seed_metrics)
        if stats:
            stat_rows = []
            for metric_key, label in METRIC_FIELDS:
                if metric_key not in stats:
                    continue
                metric_stats = stats[metric_key]
                stat_rows.append(
                    [
                        label,
                        format_value(metric_stats["mean"]),
                        format_value(metric_stats["median"]),
                        format_value(metric_stats["std"]),
                        format_value(metric_stats["min"]),
                        format_value(metric_stats["max"]),
                    ]
                )
            print(
                tabulate(
                    stat_rows,
                    headers=["Metric", "Mean", "Median", "Std", "Min", "Max"],
                    tablefmt="grid",
                )
            )


def save_csv_results(
    all_results: Dict[str, Dict[int, Dict[str, float | None]]],
    output_dir: str,
):
    """Write per-seed metrics to a single CSV file."""
    csv_path = os.path.join(output_dir, "inverse_evaluation.csv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ["model", "seed", "num_samples"] + [key for key, _ in METRIC_FIELDS]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, seed_metrics in all_results.items():
            for seed, metrics in sorted(seed_metrics.items()):
                row = {"model": model_name, "seed": seed}
                for key in fieldnames[2:]:
                    row[key] = metrics.get(key)
                writer.writerow(row)

    print(f"  ✓ Saved CSV summary to {csv_path}")


def save_text_summary(
    all_results: Dict[str, Dict[int, Dict[str, float | None]]],
    output_dir: str,
):
    """Write a readable text summary with per-seed tables and aggregate stats."""
    txt_path = os.path.join(output_dir, "inverse_evaluation_summary.txt")
    os.makedirs(output_dir, exist_ok=True)

    lines: List[str] = []
    for model_name, seed_metrics in all_results.items():
        lines.append(f"Model: {model_name}")
        if not seed_metrics:
            lines.append("  (no valid runs found)\n")
            continue

        headers = ["Seed"] + [label for _, label in METRIC_FIELDS]
        table_rows = []
        for seed in sorted(seed_metrics.keys()):
            metrics = seed_metrics[seed]
            row = [str(seed)]
            for metric_key, _ in METRIC_FIELDS:
                row.append(format_value(metrics.get(metric_key)))
            table_rows.append(row)
        lines.append(
            tabulate(table_rows, headers=headers, tablefmt="github")
        )

        stats = compute_model_statistics(seed_metrics)
        if stats:
            stat_rows = []
            for metric_key, label in METRIC_FIELDS:
                if metric_key not in stats:
                    continue
                metric_stats = stats[metric_key]
                stat_rows.append(
                    [
                        label,
                        format_value(metric_stats["mean"]),
                        format_value(metric_stats["median"]),
                        format_value(metric_stats["std"]),
                        format_value(metric_stats["min"]),
                        format_value(metric_stats["max"]),
                    ]
                )
            lines.append(
                tabulate(
                    stat_rows,
                    headers=["Metric", "Mean", "Median", "Std", "Min", "Max"],
                    tablefmt="github",
                )
            )
        lines.append("")  # spacer

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"  ✓ Saved TXT summary to {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inverse models across multiple seeds and export summary statistics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., burgers_1d, darcy_1d, wave_scattering).",
    )
    parser.add_argument(
        "--log_base_dir",
        type=str,
        required=True,
        help="Base directory containing model subdirectories (e.g., /logs/burgers_1d).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of inverse models to evaluate (e.g., nonlinear_inverse cinn_affine).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Seeds to evaluate (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation dataloader (default: 1 to ensure compatibility).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (default: cpu).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store CSV and TXT summaries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("INVERSE MODEL EVALUATION")
    print("=" * 80)
    print(f"Dataset:      {args.dataset}")
    print(f"Log Base Dir: {args.log_base_dir}")
    print(f"Models:       {args.models}")
    print(f"Seeds:        {args.seeds}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Device:       {args.device}")
    print(f"Output Dir:   {args.output_dir}")
    print("=" * 80)

    torch_device = torch.device(args.device)

    print("Loading reference dataset...")
    test_dataset, dataset_info = load_reference_dataset(
        log_base_dir=args.log_base_dir,
        dataset_name=args.dataset,
        model_names=args.models,
        seeds=args.seeds,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"  ✓ Loaded test dataset with {len(test_dataset)} samples.")

    all_results: Dict[str, Dict[int, Dict[str, float | None]]] = {model_name: {} for model_name in args.models}

    for model_name in args.models:
        print(f"\nEvaluating model '{model_name}'...")
        for seed in args.seeds:
            run_dir = os.path.join(args.log_base_dir, model_name, f"seed_{seed}")
            if not os.path.exists(run_dir):
                print(f"  ⚠ Skipping seed {seed}: directory not found at {run_dir}")
                continue

            try:
                metrics = evaluate_model_for_seed(
                    model_name=model_name,
                    seed=seed,
                    run_dir=run_dir,
                    dataset=test_dataset,
                    dataset_info=dataset_info,
                    batch_size=args.batch_size,
                    device=torch_device,
                    device_str=args.device,
                )
                all_results[model_name][seed] = metrics
                print(
                    "  ✓ Seed {seed}: inverse_rel_l2={irl}, resim_pred_rel_l2={srl}".format(
                        seed=seed,
                        irl=format_value(metrics.get("inverse_rel_l2")),
                        srl=format_value(metrics.get("resim_pred_rel_l2")),
                    )
                )
            except FileNotFoundError as err:
                print(f"  ⚠ Skipping seed {seed}: {err}")
            except Exception as exc:
                print(f"  ✗ Error evaluating seed {seed} for model {model_name}: {exc}")

    print_console_summary(all_results)

    os.makedirs(args.output_dir, exist_ok=True)
    print("\nSaving summaries...")
    save_csv_results(all_results, args.output_dir)
    save_text_summary(all_results, args.output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
