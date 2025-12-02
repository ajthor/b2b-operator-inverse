"""
Evaluate B2B models (function encoders and forward models) across multiple seeds.

This script loads all trained function encoders and forward B2B models from multiple
seed directories, computes test errors, extracts TensorBoard training curves, and
generates comprehensive statistics and visualizations.

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
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import math

from data.load_dataset import load_dataset
from b2b.load_model import load_function_encoders, load_forward_model
from data.process_data import InputFunctionEncoderDataset, OutputFunctionEncoderDataset
from b2b.function_encoder import evaluate as evaluate_function_encoder
from config.paths import resolve_base_dir

# Try to import tensorboard for reading event files
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Loss curves will not be extracted.")


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

    results = {"input": {}, "output": {}}

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

            # Evaluate encoders on full test set with global relative L2 errors
            input_encoder.eval()
            output_encoder.eval()
            total_input_sq_error = 0.0
            total_input_sq_target = 0.0
            total_output_sq_error = 0.0
            total_output_sq_target = 0.0

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

            input_l2_error = (
                math.sqrt(total_input_sq_error / total_input_sq_target)
                if total_input_sq_target > 0
                else float("nan")
            )
            output_l2_error = (
                math.sqrt(total_output_sq_error / total_output_sq_target)
                if total_output_sq_target > 0
                else float("nan")
            )

            results["input"][seed] = input_l2_error
            print(f"    Input Encoder Test Relative L2 Error: {input_l2_error:.6e}")
            results["output"][seed] = output_l2_error
            print(f"    Output Encoder Test Relative L2 Error: {output_l2_error:.6e}")

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

    results = defaultdict(dict)

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

                if total_sq_target > 0:
                    test_l2_error = math.sqrt(total_sq_error / total_sq_target)
                else:
                    test_l2_error = float("nan")
                results[model_name][seed] = test_l2_error
                print(f"      Test Relative L2 Error: {test_l2_error:.6e}")

            except Exception as e:
                print(f"    ✗ Error evaluating seed {seed}: {e}")
                continue

    return dict(results)


def extract_tensorboard_data(log_base_dir, seeds, model_types):
    """
    Extract training curves from TensorBoard event files.

    Args:
        log_base_dir: Base directory containing seed subdirectories
        seeds: List of seed numbers to process
        model_types: List of model types to extract (e.g., ['input_function_encoder', 'output_function_encoder'])

    Returns:
        dict: {
            model_type: {
                seed: {
                    'train': [(step, value), ...],
                    'test': [(step, value), ...]
                }
            }
        }
    """
    if not TENSORBOARD_AVAILABLE:
        return {}

    print("\n" + "=" * 80)
    print("Extracting TensorBoard Data")
    print("=" * 80)

    results = defaultdict(lambda: defaultdict(dict))

    for seed in seeds:
        seed_log_dir = os.path.join(log_base_dir, "shared", f"seed_{seed}")

        if not os.path.exists(seed_log_dir):
            print(f"  ⚠ Skipping seed {seed}: directory not found")
            continue

        print(f"  → Processing seed {seed}...")

        total_events_found = 0

        for model_type in model_types:
            try:
                # Look for subdirectories with train and test logs
                # Pattern: loss_train_{model_type}/ and loss_test_{model_type}/
                train_dir = os.path.join(seed_log_dir, f"loss_train_{model_type}")
                test_dir = os.path.join(seed_log_dir, f"loss_test_{model_type}")

                # Extract train data
                if os.path.exists(train_dir):
                    try:
                        event_acc = EventAccumulator(train_dir)
                        event_acc.Reload()
                        available_tags = event_acc.Tags().get("scalars", [])

                        # The tag is usually just the model name or a simple key
                        for tag in available_tags:
                            train_events = event_acc.Scalars(tag)
                            results[model_type][seed]["train"] = [
                                (e.step, e.value) for e in train_events
                            ]
                            total_events_found += len(train_events)
                            break  # Usually just one tag per directory
                    except Exception as e:
                        print(f"    ⚠ Could not read train data for {model_type}: {e}")

                # Extract test data
                if os.path.exists(test_dir):
                    try:
                        event_acc = EventAccumulator(test_dir)
                        event_acc.Reload()
                        available_tags = event_acc.Tags().get("scalars", [])

                        # The tag is usually just the model name or a simple key
                        for tag in available_tags:
                            test_events = event_acc.Scalars(tag)
                            results[model_type][seed]["test"] = [
                                (e.step, e.value) for e in test_events
                            ]
                            total_events_found += len(test_events)
                            break  # Usually just one tag per directory
                    except Exception as e:
                        print(f"    ⚠ Could not read test data for {model_type}: {e}")

            except Exception as e:
                print(f"    ✗ Error processing {model_type}: {e}")
                continue

        print(f"    ✓ Extracted {total_events_found} scalar values")

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

    # Input Function Encoder Table
    if function_encoder_results["input"]:
        print("\n--- Input Function Encoder ---")
        table_data = []
        for seed in sorted(function_encoder_results["input"].keys()):
            loss = function_encoder_results["input"][seed]
            table_data.append([seed, f"{loss:.6e}"])

        # Add statistics
        stats = compute_statistics(function_encoder_results["input"])
        table_data.append(["---", "---"])
        table_data.append(["Mean", f"{stats['mean']:.6e}"])
        table_data.append(["Median", f"{stats['median']:.6e}"])
        table_data.append(["Std", f"{stats['std']:.6e}"])
        table_data.append(["Min", f"{stats['min']:.6e}"])
        table_data.append(["Max", f"{stats['max']:.6e}"])

        print(
            tabulate(table_data, headers=["Seed", "Relative L2 Error"], tablefmt="grid")
        )
    else:
        print("\n--- Input Function Encoder ---")
        print("  No results available")

    # Output Function Encoder Table
    if function_encoder_results["output"]:
        print("\n--- Output Function Encoder ---")
        table_data = []
        for seed in sorted(function_encoder_results["output"].keys()):
            loss = function_encoder_results["output"][seed]
            table_data.append([seed, f"{loss:.6e}"])

        # Add statistics
        stats = compute_statistics(function_encoder_results["output"])
        table_data.append(["---", "---"])
        table_data.append(["Mean", f"{stats['mean']:.6e}"])
        table_data.append(["Median", f"{stats['median']:.6e}"])
        table_data.append(["Std", f"{stats['std']:.6e}"])
        table_data.append(["Min", f"{stats['min']:.6e}"])
        table_data.append(["Max", f"{stats['max']:.6e}"])

        print(
            tabulate(table_data, headers=["Seed", "Relative L2 Error"], tablefmt="grid")
        )
    else:
        print("\n--- Output Function Encoder ---")
        print("  No results available")

    # Forward Model Tables
    for model_name in sorted(forward_model_results.keys()):
        if forward_model_results[model_name]:
            print(f"\n--- Forward Model: {model_name} ---")
            table_data = []
            for seed in sorted(forward_model_results[model_name].keys()):
                loss = forward_model_results[model_name][seed]
                table_data.append([seed, f"{loss:.6e}"])

            # Add statistics
            stats = compute_statistics(forward_model_results[model_name])
            table_data.append(["---", "---"])
            table_data.append(["Mean", f"{stats['mean']:.6e}"])
            table_data.append(["Median", f"{stats['median']:.6e}"])
            table_data.append(["Std", f"{stats['std']:.6e}"])
            table_data.append(["Min", f"{stats['min']:.6e}"])
            table_data.append(["Max", f"{stats['max']:.6e}"])

            print(
                tabulate(
                    table_data, headers=["Seed", "Relative L2 Error"], tablefmt="grid"
                )
            )
        else:
            print(f"\n--- Forward Model: {model_name} ---")
            print("  No results available")

    print("\n" + "=" * 80)


def generate_plots(
    function_encoder_results, forward_model_results, tensorboard_data, output_dir
):
    """
    Generate visualization plots for evaluation results.

    Args:
        function_encoder_results: Results from evaluate_function_encoders
        forward_model_results: Results from evaluate_forward_models
        tensorboard_data: TensorBoard time series data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80)

    if not tensorboard_data:
        print("  ⚠ No TensorBoard data available for plotting")
        return

    # Group models for combined plots (excluding b2b_linear)
    model_groups = {
        "Function Encoders": ["input_function_encoder", "output_function_encoder"],
        "Forward Model": ["b2b_nonlinear"],
    }

    for group_name, model_types in model_groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))

        any_data_plotted = False

        for idx, model_type in enumerate(model_types):
            if model_type not in tensorboard_data:
                continue

            seed_data = tensorboard_data[model_type]
            if not seed_data:
                continue

            # Extract test loss data from all seeds
            test_data_by_step = defaultdict(list)
            for data in seed_data.values():
                if "test" not in data or not data["test"]:
                    continue
                for step, value in data["test"]:
                    test_data_by_step[step].append(value)

            if not test_data_by_step:
                continue

            # Compute median and std across seeds for each step
            steps = sorted(test_data_by_step.keys())
            medians = []
            stds = []
            for s in steps:
                values = test_data_by_step[s]
                if len(values) > 0:
                    medians.append(np.median(values))
                    stds.append(np.std(values))
                else:
                    medians.append(np.nan)
                    stds.append(np.nan)

            medians_array = np.array(medians)
            stds_array = np.array(stds)

            # Plot median line
            label = model_type.replace("_", " ").title()
            ax.plot(
                steps,
                medians_array,
                label=label,
                color=colors[idx],
                linewidth=2.5,
                alpha=0.9,
                zorder=10,
            )

            # Plot std as shaded region (use log space for better visibility)
            # Convert to log space, add/subtract std, then convert back
            log_medians = np.log10(
                medians_array + 1e-12
            )  # Add small epsilon to avoid log(0)
            log_stds = stds_array / (medians_array * np.log(10) + 1e-12)

            upper = 10 ** (log_medians + log_stds)
            lower = 10 ** (log_medians - log_stds)

            ax.fill_between(
                steps, lower, upper, color=colors[idx], alpha=0.25, zorder=5
            )

            any_data_plotted = True

        if not any_data_plotted:
            plt.close()
            print(f"  ⚠ Skipping {group_name}: No data available")
            continue

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Test Loss", fontsize=12)
        ax.set_title(
            f"{group_name}\nTest Loss (Median ± Std across seeds)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=10, loc="best", framealpha=0.9)

        plt.tight_layout()
        filename = group_name.lower().replace(" ", "_") + "_test_loss.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {plot_path}")

    print("=" * 80)


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

    # Input encoder CSV
    if function_encoder_results["input"]:
        csv_path = os.path.join(output_dir, "input_encoder_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Seed", "Relative L2 Error"])
            for seed in sorted(function_encoder_results["input"].keys()):
                writer.writerow([seed, function_encoder_results["input"][seed]])

            stats = compute_statistics(function_encoder_results["input"])
            writer.writerow([])
            writer.writerow(["Statistic", "Value"])
            writer.writerow(["Mean", stats["mean"]])
            writer.writerow(["Median", stats["median"]])
            writer.writerow(["Std", stats["std"]])
            writer.writerow(["Min", stats["min"]])
            writer.writerow(["Max", stats["max"]])
        print(f"  ✓ Saved: {csv_path}")

    # Output encoder CSV
    if function_encoder_results["output"]:
        csv_path = os.path.join(output_dir, "output_encoder_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Seed", "Relative L2 Error"])
            for seed in sorted(function_encoder_results["output"].keys()):
                writer.writerow([seed, function_encoder_results["output"][seed]])

            stats = compute_statistics(function_encoder_results["output"])
            writer.writerow([])
            writer.writerow(["Statistic", "Value"])
            writer.writerow(["Mean", stats["mean"]])
            writer.writerow(["Median", stats["median"]])
            writer.writerow(["Std", stats["std"]])
            writer.writerow(["Min", stats["min"]])
            writer.writerow(["Max", stats["max"]])
        print(f"  ✓ Saved: {csv_path}")

    # Forward model CSVs
    for model_name in sorted(forward_model_results.keys()):
        if forward_model_results[model_name]:
            csv_path = os.path.join(output_dir, f"{model_name}_results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Seed", "Relative L2 Error"])
                for seed in sorted(forward_model_results[model_name].keys()):
                    writer.writerow([seed, forward_model_results[model_name][seed]])

                stats = compute_statistics(forward_model_results[model_name])
                writer.writerow([])
                writer.writerow(["Statistic", "Value"])
                writer.writerow(["Mean", stats["mean"]])
                writer.writerow(["Median", stats["median"]])
                writer.writerow(["Std", stats["std"]])
                writer.writerow(["Min", stats["min"]])
                writer.writerow(["Max", stats["max"]])
            print(f"  ✓ Saved: {csv_path}")

    print("=" * 80)


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
        "--no_plots",
        action="store_true",
        help="Skip generating plots",
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
    log_base_dir = os.path.join(base_dir, "runs", args.dataset)
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

    # Extract TensorBoard data
    model_types = [
        "input_function_encoder",
        "output_function_encoder",
    ] + args.forward_models
    tensorboard_data = extract_tensorboard_data(
        log_base_dir=log_base_dir, seeds=args.seeds, model_types=model_types
    )

    # Print results tables
    print_results_tables(function_encoder_results, forward_model_results)

    # Generate plots
    if not args.no_plots:
        generate_plots(
            function_encoder_results=function_encoder_results,
            forward_model_results=forward_model_results,
            tensorboard_data=tensorboard_data,
            output_dir=output_dir,
        )

    # Save CSV results
    if not args.no_csv:
        save_csv_results(
            function_encoder_results=function_encoder_results,
            forward_model_results=forward_model_results,
            output_dir=output_dir,
        )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
