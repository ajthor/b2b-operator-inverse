"""
Evaluate iFNO models across multiple seeds and summarize performance.

This script evaluates the iFNO (Invertible Fourier Neural Operator) baseline
across multiple seeds. For each model/seed combination it:
  * loads the trained iFNO model,
  * evaluates inverse reconstruction quality (s → u) on the test split,
  * computes forward re-simulation diagnostics (u_pred → s_resim),
  * aggregates dataset-wide error statistics, and
  * exports per-seed results alongside aggregate summaries to CSV and TXT files.

Unlike the standard inverse model evaluation, iFNO works directly on function
values without coefficient encoders, so all metrics are computed in function space.

Example usage:
    python inverse_neural_operator/evaluate_ifno.py \
        --dataset burgers_1d \
        --base_dir /store/b2b-operator-inverse-results \
        --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from data.load_dataset import load_dataset
from inverse_neural_operator.plots.utils.ifno_utils import load_ifno_model
from config.paths import resolve_base_dir


# Order in which metrics are reported (key, human readable label)
METRIC_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("inverse_rel_l2", "Inverse Rel L2"),
    ("inverse_mse", "Inverse MSE"),
    ("forward_rel_l2", "Forward Rel L2"),
    ("forward_mse", "Forward MSE"),
)


def format_value(value: float | None, precision: int = 6) -> str:
    """Format numeric values, keeping blanks for missing entries."""
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "—"
    return f"{value:.{precision}e}"


def compute_statistics(values: Iterable[float]) -> Dict[str, float]:
    """Compute standard statistics for a collection of floats."""
    values = [float(v) for v in values if v is not None]
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_test_dataset(dataset_name: str, device: str, batch_size: int):
    """Load the test dataset for the specified dataset name."""

    # Create minimal params object for dataset loading
    class Params:
        pass

    params = Params()
    params.dataset = dataset_name
    params.batch_size = batch_size
    params.device = device

    test_dataset, dataset_info = load_dataset(
        dataset_name=dataset_name,
        params=params,
        device=device,
        split="test",
        return_info=True,
    )
    return test_dataset, dataset_info


def _trim_function_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Remove coordinate channels so shapes match the target function tensor."""
    if pred.shape[-1] > target.shape[-1]:
        return pred[..., -target.shape[-1] :]
    return pred


def _reshape_to_spatial(tensor: torch.Tensor, spatial_dims):
    """Ensure tensor matches iFNO's expected spatial shape."""
    if spatial_dims is None:
        return tensor

    if tensor.dim() == 0:
        return tensor
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    spatial_rank = len(spatial_dims)
    spatial_size = int(np.prod(spatial_dims))

    if tensor.dim() == spatial_rank and tensor.shape == tuple(spatial_dims):
        tensor = tensor.unsqueeze(-1)

    if tensor.dim() == spatial_rank + 1 and tuple(tensor.shape[:spatial_rank]) == tuple(
        spatial_dims
    ):
        return tensor

    feature_dim = tensor.shape[-1]

    flattened = tensor
    if tensor.dim() != 2:
        flattened = tensor.reshape(-1, feature_dim)

    if flattened.shape[0] == spatial_size:
        return flattened.reshape(*spatial_dims, feature_dim)

    return tensor


def accumulate_sample_metrics(
    accumulators: Dict[str, float],
    sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    input_spatial_dims,
    output_spatial_dims,
) -> None:
    """
    Evaluate a single (batched) sample and update running error totals.

    This follows the exact pattern used in collect_ifno_predictions from ifno_utils.py.
    """
    X, u_true, Y, s_true = sample

    # Remove batch dimension first - tensors come as [1, N, C]
    X = X.squeeze(0)  # [N, C]
    u_true = u_true.squeeze(0)  # [N, C]
    Y = Y.squeeze(0)  # [M, C]
    s_true = s_true.squeeze(0)  # [M, C]

    # Reshape to spatial dimensions expected by iFNO
    X_spatial = _reshape_to_spatial(X, input_spatial_dims)
    u_true_spatial = _reshape_to_spatial(u_true, input_spatial_dims)
    Y_spatial = _reshape_to_spatial(Y, output_spatial_dims)
    s_true_spatial = _reshape_to_spatial(s_true, output_spatial_dims)

    with torch.no_grad():
        try:
            # Inverse pass: s → u_pred (add batch dimension before model call)
            s_input = torch.cat(
                [Y_spatial.unsqueeze(0), s_true_spatial.unsqueeze(0)], dim=-1
            )
            inverse_result = model.inverse(s_input)
            if isinstance(inverse_result, (tuple, list)):
                u_pred = inverse_result[0]
            else:
                u_pred = inverse_result
            u_pred = _trim_function_channels(u_pred, u_true_spatial.unsqueeze(0))
        except Exception as e:
            raise RuntimeError(
                f"Inverse pass failed:\n"
                f"  Y_spatial shape: {Y_spatial.shape}\n"
                f"  s_true_spatial shape: {s_true_spatial.shape}\n"
                f"  Y_spatial.unsqueeze(0) shape: {Y_spatial.unsqueeze(0).shape}\n"
                f"  s_true_spatial.unsqueeze(0) shape: {s_true_spatial.unsqueeze(0).shape}\n"
                f"  Error: {e}"
            ) from e

        # Forward pass: u_pred → s_resim
        # First check if u_pred contains NaN
        if torch.isnan(u_pred).any():
            raise RuntimeError(
                f"u_pred contains NaN values after inverse pass!\n"
                f"  u_pred shape: {u_pred.shape}\n"
                f"  NaN count: {torch.isnan(u_pred).sum().item()}\n"
                f"  u_pred stats: min={u_pred.min().item()}, max={u_pred.max().item()}, mean={u_pred.mean().item()}"
            )

        u_input = torch.cat([X_spatial.unsqueeze(0), u_pred], dim=-1)
        forward_result = model(u_input)
        if isinstance(forward_result, (tuple, list)):
            s_resim = forward_result[0]
        else:
            s_resim = forward_result

        s_resim = _trim_function_channels(s_resim, s_true_spatial.unsqueeze(0))

        # For asymmetric problems, reshape s_resim to match output spatial dimensions
        # The model may return [1, input_spatial_size, output_spatial_size, channels]
        # but we need [1, output_spatial_dims..., channels]
        if output_spatial_dims and s_resim.shape != s_true_spatial.unsqueeze(0).shape:
            batch_size = s_resim.shape[0]
            output_channels = s_resim.shape[-1]
            expected_spatial_size = int(np.prod(output_spatial_dims))

            # Flatten all intermediate dimensions
            s_resim_flat = s_resim.reshape(batch_size, -1, output_channels)

            # Reshape to proper spatial format
            if s_resim_flat.shape[1] == expected_spatial_size:
                s_resim = s_resim_flat.reshape(
                    batch_size, *output_spatial_dims, output_channels
                )
            else:
                raise RuntimeError(
                    f"Cannot reshape s_resim to match output spatial dims:\n"
                    f"  s_resim shape: {s_resim.shape}\n"
                    f"  s_resim_flat shape: {s_resim_flat.shape}\n"
                    f"  Expected spatial size: {expected_spatial_size}\n"
                    f"  output_spatial_dims: {output_spatial_dims}\n"
                    f"  s_true_spatial.unsqueeze(0) shape: {s_true_spatial.unsqueeze(0).shape}"
                )

    # Inverse reconstruction metrics (u space)
    # u_pred has batch dim [1, ...], u_true_spatial doesn't, so add it for comparison
    u_true_with_batch = u_true_spatial.unsqueeze(0)
    accumulators["u_error_sq"] += torch.sum((u_pred - u_true_with_batch) ** 2).item()
    accumulators["u_target_sq"] += torch.sum(u_true_with_batch**2).item()
    accumulators["u_elements"] += u_true_with_batch.numel()

    # Forward re-simulation metrics (s space)
    # Check if forward pass produced NaN or inf - if so, skip this sample entirely
    s_true_with_batch = s_true_spatial.unsqueeze(0)
    if not torch.isfinite(s_resim).all():
        accumulators["skipped_samples"] += 1
        return

    # Compute error and check if it's finite
    s_error = (s_resim - s_true_with_batch) ** 2
    s_error_sum = torch.sum(s_error).item()

    if not math.isfinite(s_error_sum):
        accumulators["skipped_samples"] += 1
        return

    accumulators["s_error_sq"] += s_error_sum
    accumulators["s_target_sq"] += torch.sum(s_true_with_batch**2).item()
    accumulators["s_elements"] += s_true_with_batch.numel()


def finalize_metrics(accumulators: Dict[str, float]) -> Dict[str, float | None]:
    """Convert accumulated sums into interpretable metrics."""

    def safe_ratio(numerator: float, denominator: float, metric_name: str) -> float:
        if denominator <= 0.0:
            raise RuntimeError(
                f"Cannot compute {metric_name}: denominator is {denominator}\n"
                f"Accumulators: {accumulators}"
            )
        result = numerator / denominator
        if math.isnan(result) or math.isinf(result):
            raise RuntimeError(
                f"Invalid {metric_name}: result is {result}\n"
                f"  numerator={numerator}, denominator={denominator}\n"
                f"Accumulators: {accumulators}"
            )
        return result

    def safe_relative(error_sq: float, target_sq: float, metric_name: str) -> float:
        if target_sq <= 0.0:
            raise RuntimeError(
                f"Cannot compute {metric_name}: target_sq is {target_sq}\n"
                f"Accumulators: {accumulators}"
            )
        if error_sq < 0.0:
            raise RuntimeError(
                f"Cannot compute {metric_name}: error_sq is {error_sq}\n"
                f"Accumulators: {accumulators}"
            )
        if math.isnan(error_sq) or math.isnan(target_sq):
            raise RuntimeError(
                f"Cannot compute {metric_name}: NaN values detected\n"
                f"  error_sq={error_sq}, target_sq={target_sq}\n"
                f"Accumulators: {accumulators}"
            )
        result = math.sqrt(error_sq / target_sq)
        if math.isnan(result) or math.isinf(result):
            raise RuntimeError(
                f"Invalid {metric_name}: result is {result}\n"
                f"  error_sq={error_sq}, target_sq={target_sq}\n"
                f"Accumulators: {accumulators}"
            )
        return result

    metrics = {
        "inverse_mse": safe_ratio(
            accumulators["u_error_sq"], accumulators["u_elements"], "inverse_mse"
        ),
        "inverse_rel_l2": safe_relative(
            accumulators["u_error_sq"], accumulators["u_target_sq"], "inverse_rel_l2"
        ),
        "forward_mse": safe_ratio(
            accumulators["s_error_sq"], accumulators["s_elements"], "forward_mse"
        ),
        "forward_rel_l2": safe_relative(
            accumulators["s_error_sq"], accumulators["s_target_sq"], "forward_rel_l2"
        ),
        "num_samples": accumulators["num_samples"],
        "skipped_samples": accumulators.get("skipped_samples", 0.0),
    }

    return metrics


def evaluate_ifno_for_seed(
    seed: int,
    checkpoint_path: str,
    dataset,
    dataset_info,
    batch_size: int,
    device: torch.device,
    device_str: str,
):
    """
    Load a specific iFNO model/seed run and compute dataset-wide metrics.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint for seed {seed} at {checkpoint_path}"
        )

    model = load_ifno_model(dataset_info, checkpoint_path, device=device_str)
    model.eval()

    input_spatial_dims = dataset_info.get("input_spatial_dims")
    output_spatial_dims = dataset_info.get("output_spatial_dims")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    accumulators = defaultdict(float)
    accumulators.update(
        {
            "u_error_sq": 0.0,
            "u_target_sq": 0.0,
            "u_elements": 0.0,
            "s_error_sq": 0.0,
            "s_target_sq": 0.0,
            "s_elements": 0.0,
            "num_samples": 0.0,
        }
    )

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
                device=device,
                input_spatial_dims=input_spatial_dims,
                output_spatial_dims=output_spatial_dims,
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


def print_console_summary(seed_metrics: Dict[int, Dict[str, float | None]]):
    """Print a concise summary to stdout."""
    if not seed_metrics:
        print("\n(no valid runs found)")
        return

    print("\niFNO Model Evaluation")
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
    seed_metrics: Dict[int, Dict[str, float | None]],
    output_dir: str,
):
    """Write per-seed metrics to a CSV file."""
    import csv

    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = ["seed", "num_samples"] + [key for key, _ in METRIC_FIELDS]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed, metrics in sorted(seed_metrics.items()):
            row = {"seed": seed}
            for key in fieldnames[1:]:
                row[key] = metrics.get(key)
            writer.writerow(row)

    print(f"  ✓ Saved CSV summary to {csv_path}")


def save_text_summary(
    seed_metrics: Dict[int, Dict[str, float | None]],
    output_dir: str,
):
    """Write a readable text summary with per-seed tables and aggregate stats."""
    txt_path = os.path.join(output_dir, "evaluation_summary.txt")
    os.makedirs(output_dir, exist_ok=True)

    lines: List[str] = []
    lines.append("iFNO Model Evaluation")
    if not seed_metrics:
        lines.append("  (no valid runs found)\n")
    else:
        headers = ["Seed"] + [label for _, label in METRIC_FIELDS]
        table_rows = []
        for seed in sorted(seed_metrics.keys()):
            metrics = seed_metrics[seed]
            row = [str(seed)]
            for metric_key, _ in METRIC_FIELDS:
                row.append(format_value(metrics.get(metric_key)))
            table_rows.append(row)
        lines.append(tabulate(table_rows, headers=headers, tablefmt="github"))

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

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"  ✓ Saved TXT summary to {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate iFNO models across multiple seeds and export summary statistics."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., burgers_1d, darcy_1d, wave_scattering).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory for models/results/logs (overrides B2B_RESULTS_DIR / ./results fallback).",
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
        help="Batch size for evaluation dataloader (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (default: cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Construct paths from base_dir or config
    base_dir = str(resolve_base_dir(args.base_dir))
    models_base_dir = os.path.join(base_dir, "models", args.dataset, "ifno")
    output_dir = os.path.join(base_dir, "runs", args.dataset, "ifno")

    print("=" * 80)
    print("iFNO MODEL EVALUATION")
    print("=" * 80)
    print(f"Dataset:        {args.dataset}")
    print(f"Models Dir:     {models_base_dir}")
    print(f"Seeds:          {args.seeds}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Device:         {args.device}")
    print(f"Output Dir:     {output_dir}")
    print("=" * 80)

    torch_device = torch.device(args.device)

    print("\nLoading test dataset...")
    test_dataset, dataset_info = load_test_dataset(
        dataset_name=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"  ✓ Loaded test dataset with {len(test_dataset)} samples.")

    seed_metrics: Dict[int, Dict[str, float | None]] = {}

    print(f"\nEvaluating iFNO model...")
    for seed in args.seeds:
        checkpoint_path = os.path.join(
            models_base_dir, f"seed_{seed}", "ifno_model.safetensors"
        )
        if not os.path.exists(checkpoint_path):
            print(
                f"  ⚠ Skipping seed {seed}: checkpoint not found at {checkpoint_path}"
            )
            continue

        try:
            metrics = evaluate_ifno_for_seed(
                seed=seed,
                checkpoint_path=checkpoint_path,
                dataset=test_dataset,
                dataset_info=dataset_info,
                batch_size=args.batch_size,
                device=torch_device,
                device_str=args.device,
            )
            seed_metrics[seed] = metrics
            skipped = int(metrics.get("skipped_samples", 0))
            skip_msg = ""
            if skipped > 0:
                skip_msg = f" (skipped {skipped} samples with NaN/inf)"
            print(
                "  ✓ Seed {seed}: inverse_rel_l2={irl}, forward_rel_l2={frl}{skip_msg}".format(
                    seed=seed,
                    irl=format_value(metrics.get("inverse_rel_l2")),
                    frl=format_value(metrics.get("forward_rel_l2")),
                    skip_msg=skip_msg,
                )
            )
        except FileNotFoundError as err:
            print(f"  ⚠ Skipping seed {seed}: {err}")
        except Exception as exc:
            import traceback

            print(f"  ✗ Error evaluating seed {seed}: {exc}")
            print("  Full traceback:")
            traceback.print_exc()

    print_console_summary(seed_metrics)

    # Save results to dataset-specific output directory
    os.makedirs(output_dir, exist_ok=True)
    print("\nSaving summaries...")
    save_csv_results(seed_metrics, output_dir)
    save_text_summary(seed_metrics, output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
