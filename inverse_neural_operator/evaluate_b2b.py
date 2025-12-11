"""
Evaluate B2B models (function encoders and forward models) across multiple seeds.

This script loads all trained function encoders and forward B2B models from multiple
seed directories, computes test errors, and generates comprehensive CSV/TXT reports.

Usage:
    python inverse_neural_operator/evaluate_b2b.py \
        --dataset burgers_1d \
        --base_dir /store/b2b-operator-inverse-results \
        --seeds 1 2 3 4 5 \
        --forward_models b2b_linear b2b_nonlinear
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tabulate import tabulate
from collections import defaultdict
import math

from data.load_dataset import load_dataset
from b2b.load_model import load_function_encoders, load_forward_model
from data.process_data import InputFunctionEncoderDataset, OutputFunctionEncoderDataset
from b2b.function_encoder import evaluate as evaluate_function_encoder
from config.paths import resolve_base_dir
from report_utils import build_summary_rows


def evaluate_function_encoders(
    dataset_name,
    model_base_dir,
    seeds,
    batch_size=32,
    device="cpu",
    dataset_device="cpu",
):
    """
    Evaluate input and output function encoders across multiple seeds.

    Args:
        dataset_name: Name of the dataset
        model_base_dir: Base directory containing model files (models/dataset/)
        seeds: List of seed numbers to evaluate
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        dataset_device: Device to keep dataset tensors on before batching

    Returns:
        dict: {
            'input': {seed: test_loss, ...},
            'output': {seed: test_loss, ...}
        }
    """
    print("\n" + "=" * 80)
    print("Evaluating Function Encoders")
    print("=" * 80)

    # Load params from the first available seed directory
    params = None
    for seed in seeds:
        seed_model_dir = os.path.join(model_base_dir, "shared", f"seed_{seed}")
        params_path = os.path.join(seed_model_dir, "params.pth")
        if os.path.exists(params_path):
            params = torch.load(params_path, weights_only=False)
            print(f"  ✓ Loaded params from seed {seed}")
            break

    if params is None:
        raise FileNotFoundError(
            f"Could not find params.pth in any seed directory under {model_base_dir}/shared/"
        )

    # Load dataset using the actual params from training
    test_dataset, dataset_info = load_dataset(
        dataset_name=dataset_name,
        params=params,
        device=dataset_device,
        split="test",
        return_info=True,
    )

    # Create function encoder test datasets (match training procedure)
    input_test_dataset = InputFunctionEncoderDataset(test_dataset, device=device)
    output_test_dataset = OutputFunctionEncoderDataset(test_dataset, device=device)

    input_test_dataloader = DataLoader(
        input_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    output_test_dataloader = DataLoader(
        output_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    results = {
        "input": {"rel_l2": {}, "mse": {}},
        "output": {"rel_l2": {}, "mse": {}},
    }

    for seed in seeds:
        seed_model_dir = os.path.join(model_base_dir, "shared", f"seed_{seed}")

        # Check if function encoders exist
        input_encoder_path = os.path.join(
            seed_model_dir, "input_function_encoder.safetensors"
        )
        output_encoder_path = os.path.join(
            seed_model_dir, "output_function_encoder.safetensors"
        )

        if not os.path.exists(input_encoder_path):
            print(
                f"  ⚠ Skipping seed {seed}: input encoder not found at {input_encoder_path}"
            )
            continue

        if not os.path.exists(output_encoder_path):
            print(
                f"  ⚠ Skipping seed {seed}: output encoder not found at {output_encoder_path}"
            )
            continue

        print(f"  → Evaluating seed {seed}...")

        try:
            # Load function encoders
            input_encoder, output_encoder = load_function_encoders(
                model_dir=seed_model_dir,
                dataset_info=dataset_info,
                params=params,
                device=device,
            )

            # Evaluate encoders on full test set with global relative L2 + MSE
            input_encoder.eval()
            output_encoder.eval()
            total_input_sq_error = 0.0
            total_input_sq_target = 0.0
            total_input_elements = 0
            total_output_sq_error = 0.0
            total_output_sq_target = 0.0
            total_output_elements = 0

            with torch.no_grad():
                for batch in input_test_dataloader:
                    example_xs, example_ys, xs, ys = batch
                    example_xs = example_xs.to(device)
                    example_ys = example_ys.to(device)
                    xs = xs.to(device)
                    ys = ys.to(device)

                    u_pred = evaluate_function_encoder(
                        input_encoder, (example_xs, example_ys, xs, ys)
                    )
                    total_input_sq_error += torch.sum((u_pred - ys) ** 2).item()
                    total_input_sq_target += torch.sum(ys**2).item()
                    total_input_elements += ys.numel()

                for batch in output_test_dataloader:
                    example_xs, example_ys, xs, ys = batch
                    example_xs = example_xs.to(device)
                    example_ys = example_ys.to(device)
                    xs = xs.to(device)
                    ys = ys.to(device)

                    s_pred = evaluate_function_encoder(
                        output_encoder, (example_xs, example_ys, xs, ys)
                    )
                    total_output_sq_error += torch.sum((s_pred - ys) ** 2).item()
                    total_output_sq_target += torch.sum(ys**2).item()
                    total_output_elements += ys.numel()

            input_l2_error = (
                math.sqrt(total_input_sq_error / total_input_sq_target)
                if total_input_sq_target > 0
                else float("nan")
            )
            input_mse = (
                total_input_sq_error / total_input_elements
                if total_input_elements > 0
                else float("nan")
            )
            output_l2_error = (
                math.sqrt(total_output_sq_error / total_output_sq_target)
                if total_output_sq_target > 0
                else float("nan")
            )
            output_mse = (
                total_output_sq_error / total_output_elements
                if total_output_elements > 0
                else float("nan")
            )

            results["input"]["rel_l2"][seed] = input_l2_error
            results["input"]["mse"][seed] = input_mse
            print(f"    Input Encoder Relative L2: {input_l2_error:.6e}")
            print(f"    Input Encoder MSE:        {input_mse:.6e}")
            results["output"]["rel_l2"][seed] = output_l2_error
            results["output"]["mse"][seed] = output_mse
            print(f"    Output Encoder Relative L2: {output_l2_error:.6e}")
            print(f"    Output Encoder MSE:        {output_mse:.6e}")

        except Exception as e:
            print(f"  ✗ Error evaluating seed {seed}: {e}")
            continue

    return results


def evaluate_forward_models(
    dataset_name,
    model_base_dir,
    seeds,
    forward_models,
    batch_size=32,
    device="cpu",
    dataset_device="cpu",
):
    """
    Evaluate forward B2B models across multiple seeds.

    Args:
        dataset_name: Name of the dataset
        model_base_dir: Base directory containing model files (models/dataset/)
        seeds: List of seed numbers to evaluate
        forward_models: List of forward model names (e.g., ['b2b_linear', 'b2b_nonlinear'])
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        dataset_device: Device to keep dataset tensors on before batching

    Returns:
        dict: {model_name: {seed: test_loss, ...}, ...}
    """
    print("\n" + "=" * 80)
    print("Evaluating Forward Models")
    print("=" * 80)

    # Load params from the first available seed directory
    params = None
    for seed in seeds:
        seed_model_dir = os.path.join(model_base_dir, "shared", f"seed_{seed}")
        params_path = os.path.join(seed_model_dir, "params.pth")
        if os.path.exists(params_path):
            params = torch.load(params_path, weights_only=False)
            print(f"  ✓ Loaded params from seed {seed}")
            break

    if params is None:
        raise FileNotFoundError(
            f"Could not find params.pth in any seed directory under {model_base_dir}/shared/"
        )

    # Load dataset using the actual params from training
    test_dataset, dataset_info = load_dataset(
        dataset_name=dataset_name,
        params=params,
        device=dataset_device,
        split="test",
        return_info=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    def _new_metric_dict():
        return {"rel_l2": {}, "mse": {}}

    results = defaultdict(_new_metric_dict)

    for model_name in forward_models:
        print(f"\n  Model: {model_name}")
        print("  " + "-" * 76)

        for seed in seeds:
            seed_model_dir = os.path.join(model_base_dir, "shared", f"seed_{seed}")

            # Check if forward model exists
            forward_model_path = os.path.join(
                seed_model_dir, f"forward_{model_name}.safetensors"
            )

            if not os.path.exists(forward_model_path):
                print(
                    f"    ⚠ Skipping seed {seed}: forward model not found at {forward_model_path}"
                )
                continue

            print(f"    → Evaluating seed {seed}...")

            try:
                # Load function encoders
                input_encoder, output_encoder = load_function_encoders(
                    model_dir=seed_model_dir,
                    dataset_info=dataset_info,
                    params=params,
                    device=device,
                )

                # Load forward model
                forward_model = load_forward_model(
                    model_dir=seed_model_dir,
                    forward_model_name=model_name,
                    device=device,
                )

                # Evaluate forward model on full test set with global relative L2 error
                forward_model.eval()
                input_encoder.eval()
                output_encoder.eval()
                total_sq_error = 0.0
                total_sq_target = 0.0
                total_elements = 0

                with torch.no_grad():
                    for batch in test_dataloader:
                        X, u, Y, s = batch
                        X = X.to(device, non_blocking=True)
                        u = u.to(device, non_blocking=True)
                        Y = Y.to(device, non_blocking=True)
                        s = s.to(device, non_blocking=True)
                        # Compute input coefficients
                        alpha, _ = input_encoder.compute_coefficients(X, u)
                        # Get predicted output coefficients
                        beta_pred = forward_model(alpha)
                        # Decode to output
                        s_pred = output_encoder(Y, beta_pred)
                        total_sq_error += torch.sum((s_pred - s) ** 2).item()
                        total_sq_target += torch.sum(s**2).item()
                        total_elements += s.numel()

                if total_sq_target > 0:
                    test_l2_error = math.sqrt(total_sq_error / total_sq_target)
                else:
                    test_l2_error = float("nan")
                test_mse = (
                    total_sq_error / total_elements
                    if total_elements > 0
                    else float("nan")
                )
                results[model_name]["rel_l2"][seed] = test_l2_error
                results[model_name]["mse"][seed] = test_mse
                print(f"      Test Relative L2 Error: {test_l2_error:.6e}")
                print(f"      Test MSE:             {test_mse:.6e}")

            except Exception as e:
                print(f"    ✗ Error evaluating seed {seed}: {e}")
                continue

    return dict(results)


def compute_statistics(results_dict):
    """
    Compute statistics (mean, median, std) across seeds.

    Args:
        results_dict: Dictionary mapping seeds to test losses

    Returns:
        dict: {'mean': float, 'median': float, 'std': float, 'min': float, 'max': float}
    """
    if not results_dict:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}

    values = list(results_dict.values())
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
    }


METRIC_LABELS = {
    "rel_l2": "Relative L2 Error",
    "mse": "Mean Squared Error",
}


def print_results_tables(function_encoder_results, forward_model_results):
    """
    Print evaluation results in formatted tables.

    Args:
        function_encoder_results: Results from evaluate_function_encoders
        forward_model_results: Results from evaluate_forward_models
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    def _print_component(component_name, metrics_dict):
        has_data = False
        for metric_key, metric_label in METRIC_LABELS.items():
            values = metrics_dict.get(metric_key, {})
            if not values:
                continue
            has_data = True
            table_data = []
            for seed in sorted(values.keys()):
                loss = values[seed]
                table_data.append([seed, f"{loss:.6e}"])

            stats = compute_statistics(values)
            table_data.append(["---", "---"])
            table_data.append(["Mean", f"{stats['mean']:.6e}"])
            table_data.append(["Median", f"{stats['median']:.6e}"])
            table_data.append(["Std", f"{stats['std']:.6e}"])
            table_data.append(["Min", f"{stats['min']:.6e}"])
            table_data.append(["Max", f"{stats['max']:.6e}"])

            print(f"\n--- {component_name} ({metric_label}) ---")
            print(tabulate(table_data, headers=["Seed", metric_label], tablefmt="grid"))

        if not has_data:
            print(f"\n--- {component_name} ---")
            print("  No results available")

    _print_component("Input Function Encoder", function_encoder_results["input"])
    _print_component("Output Function Encoder", function_encoder_results["output"])

    for model_name in sorted(forward_model_results.keys()):
        _print_component(
            f"Forward Model: {model_name}", forward_model_results[model_name]
        )

    print("\n" + "=" * 80)


def save_csv_results(function_encoder_results, forward_model_results, output_dir):
    """
    Save evaluation results to CSV files.

    Args:
        function_encoder_results: Results from evaluate_function_encoders
        forward_model_results: Results from evaluate_forward_models
        output_dir: Directory to save CSV files
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Saving CSV Results")
    print("=" * 80)

    metric_order = list(METRIC_LABELS.keys())
    stat_fields = [
        ("Mean", "mean"),
        ("Median", "median"),
        ("Std", "std"),
        ("Min", "min"),
        ("Max", "max"),
    ]

    def _has_values(metrics_dict):
        return any(metrics_dict.get(key) for key in metric_order)

    def _write_component_csv(csv_path, metrics_dict):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Seed"] + [METRIC_LABELS[key] for key in metric_order]
            writer.writerow(header)

            seeds = set()
            for key in metric_order:
                seeds.update(metrics_dict.get(key, {}).keys())
            for seed in sorted(seeds):
                row = [seed]
                for key in metric_order:
                    value = metrics_dict.get(key, {}).get(seed)
                    row.append("" if value is None else value)
                writer.writerow(row)

            writer.writerow([])
            writer.writerow(
                ["Statistic"] + [METRIC_LABELS[key] for key in metric_order]
            )

            metric_stats = {
                key: compute_statistics(metrics_dict.get(key, {}))
                for key in metric_order
            }
            for label, field in stat_fields:
                row = [label]
                for key in metric_order:
                    row.append(metric_stats[key][field])
                writer.writerow(row)

    # Input encoder CSV
    if _has_values(function_encoder_results["input"]):
        csv_path = os.path.join(output_dir, "input_encoder_results.csv")
        _write_component_csv(csv_path, function_encoder_results["input"])
        print(f"  ✓ Saved: {csv_path}")

    # Output encoder CSV
    if _has_values(function_encoder_results["output"]):
        csv_path = os.path.join(output_dir, "output_encoder_results.csv")
        _write_component_csv(csv_path, function_encoder_results["output"])
        print(f"  ✓ Saved: {csv_path}")

    # Forward model CSVs
    for model_name in sorted(forward_model_results.keys()):
        metrics_dict = forward_model_results[model_name]
        if not _has_values(metrics_dict):
            continue
        csv_path = os.path.join(output_dir, f"{model_name}_results.csv")
        _write_component_csv(csv_path, metrics_dict)
        print(f"  ✓ Saved: {csv_path}")

    print("=" * 80)


def collect_component_statistics(function_encoder_results, forward_model_results):
    """Aggregate statistics across all evaluated components."""
    stats = {}

    def _add_component_stats(component_name, metrics_dict):
        for metric_key, metric_label in METRIC_LABELS.items():
            values = metrics_dict.get(metric_key, {})
            if not values:
                continue
            stats[f"{component_name} ({metric_label})"] = compute_statistics(values)

    _add_component_stats("Input Function Encoder", function_encoder_results["input"])
    _add_component_stats("Output Function Encoder", function_encoder_results["output"])

    for model_name, model_results in forward_model_results.items():
        _add_component_stats(f"Forward Model: {model_name}", model_results)

    return stats


def save_summary_reports(component_stats, output_dir):
    """
    Persist aggregate statistics to human-readable TXT and machine-readable CSV files.
    """
    if not component_stats:
        return

    import csv

    os.makedirs(output_dir, exist_ok=True)

    headers = [
        "Component",
        "Mean ± Std (Rel L2)",
        "Median",
        "Min",
        "Max",
    ]
    rows = list(build_summary_rows(component_stats))

    txt_path = os.path.join(output_dir, "b2b_evaluation_summary.txt")
    table = tabulate(rows, headers=headers, tablefmt="github")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("B2B Evaluation Summary\n")
        f.write(table + "\n")
    print(f"  ✓ Saved summary: {txt_path}")

    csv_path = os.path.join(output_dir, "b2b_evaluation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Component", "Mean", "Std", "Median", "Min", "Max"])
        for component in sorted(component_stats.keys()):
            stats = component_stats[component]
            writer.writerow(
                [
                    component,
                    stats.get("mean"),
                    stats.get("std"),
                    stats.get("median"),
                    stats.get("min"),
                    stats.get("max"),
                ]
            )
    print(f"  ✓ Saved summary CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate B2B models (function encoders and forward models) across multiple seeds"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., burgers_1d, darcy_1d, wave_scattering)",
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
        help="List of seeds to evaluate (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--forward_models",
        type=str,
        nargs="+",
        default=["b2b_linear", "b2b_nonlinear"],
        help="List of forward models to evaluate (default: b2b_linear b2b_nonlinear)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (default: cpu)",
    )
    parser.add_argument(
        "--dataset_device",
        type=str,
        default="cpu",
        help="Device to keep dataset tensors on before loading batches (default: cpu)",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Skip saving CSV files",
    )

    args = parser.parse_args()

    # Construct paths from base_dir or config
    base_dir = str(resolve_base_dir(args.base_dir))
    model_base_dir = os.path.join(base_dir, "models", args.dataset)
    output_dir = os.path.join(base_dir, "runs", args.dataset, "shared")

    print("=" * 80)
    print("B2B MODEL EVALUATION")
    print("=" * 80)
    print(f"Dataset:        {args.dataset}")
    print(f"Base Dir:       {base_dir}")
    print(f"Model Base Dir: {model_base_dir}")
    print(f"Seeds:          {args.seeds}")
    print(f"Forward Models: {args.forward_models}")
    print(f"Batch Size:     {args.batch_size}")
    print(f"Device:         {args.device}")
    print(f"Dataset Device: {args.dataset_device}")
    print(f"Output Dir:     {output_dir}")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate function encoders
    function_encoder_results = evaluate_function_encoders(
        dataset_name=args.dataset,
        model_base_dir=model_base_dir,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
        dataset_device=args.dataset_device,
    )

    # Evaluate forward models
    forward_model_results = evaluate_forward_models(
        dataset_name=args.dataset,
        model_base_dir=model_base_dir,
        seeds=args.seeds,
        forward_models=args.forward_models,
        batch_size=args.batch_size,
        device=args.device,
        dataset_device=args.dataset_device,
    )

    # Print results tables
    print_results_tables(function_encoder_results, forward_model_results)

    # Save CSV results
    if not args.no_csv:
        save_csv_results(
            function_encoder_results=function_encoder_results,
            forward_model_results=forward_model_results,
            output_dir=output_dir,
        )

    component_stats = collect_component_statistics(
        function_encoder_results, forward_model_results
    )
    save_summary_reports(component_stats, output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
