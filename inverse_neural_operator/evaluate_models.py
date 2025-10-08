"""
Evaluate all trained models and compute resimulation loss statistics over the entire test dataset.

This script loads all trained models from a specified log directory and computes both
coefficient resimulation loss and prediction resimulation loss over all test batches,
presenting the results in a table format.

To run:
cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.evaluate_models --dataset burgers --log_dir logs/burgers
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tabulate import tabulate

from data.load_dataset import load_dataset
from models.load_model import load_models, load_forward_model


# Available models to evaluate
MODELS = [
    "b2b_linear",
    "b2b_linear_deterministic",
    "b2b_nonlinear",
    "variational_autoencoder",
    "mixture_density_network",
    "inn_additive",
    "inn_affine",
    "cinn_additive",
    "cinn_affine",
]


def compute_resimulation_loss_all_batches(
    model,
    model_name,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    device,
):
    """
    Compute resimulation loss over all batches in the test dataset.

    Args:
        model: The trained inverse model
        model_name: Name of the model
        test_dataloader: DataLoader for test dataset
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model
        device: Device to use

    Returns:
        tuple: (avg_coeff_loss, avg_pred_loss)
    """
    # Import the appropriate resimulation_loss function based on model type
    if model_name == "b2b_linear":
        from models.b2b_operator_linear import resimulation_loss
    elif model_name == "b2b_linear_deterministic":
        from models.b2b_operator_linear_deterministic import resimulation_loss
    elif model_name == "b2b_nonlinear":
        from models.b2b_operator_nonlinear import resimulation_loss
    elif model_name == "variational_autoencoder":
        from models.variational_autoencoder import resimulation_loss
    elif model_name == "mixture_density_network":
        from models.mixture_density_network import resimulation_loss
    elif model_name == "inn_additive":
        from models.inn_additive import resimulation_loss
    elif model_name == "inn_affine":
        from models.inn_affine import resimulation_loss
    elif model_name == "cinn_additive":
        from models.cinn_additive import resimulation_loss
    elif model_name == "cinn_affine":
        from models.cinn_affine import resimulation_loss
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    forward_model.eval()

    total_coeff_loss = 0.0
    total_pred_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            X, u, Y, s = batch
            X = X.to(device)
            u = u.to(device)
            Y = Y.to(device)
            s = s.to(device)
            batch = (X, u, Y, s)

            # Compute resimulation loss
            coeff_loss, pred_loss = resimulation_loss(
                model=model,
                batch=batch,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                forward_model=forward_model,
                n_samples=1,  # Use deterministic evaluation
            )

            total_coeff_loss += coeff_loss
            total_pred_loss += pred_loss
            n_batches += 1

    avg_coeff_loss = total_coeff_loss / max(n_batches, 1)
    avg_pred_loss = total_pred_loss / max(n_batches, 1)

    return avg_coeff_loss, avg_pred_loss


def evaluate_all_models(log_dir, dataset_name, batch_size=32, device="cpu"):
    """
    Evaluate all trained models in the log directory.

    Args:
        log_dir: Directory containing trained models
        dataset_name: Name of the dataset
        batch_size: Batch size for evaluation
        device: Device to use for evaluation

    Returns:
        dict: Dictionary mapping model names to (coeff_loss, pred_loss) tuples
    """
    # Load dataset
    dataset = load_dataset(dataset_name=dataset_name)
    dataset_info = dataset.get_info()
    _, test_dataset = dataset.get_datasets()

    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    results = {}

    # Evaluate each model
    for model_name in MODELS:
        model_log_dir = os.path.join(log_dir, model_name)

        # Check if model exists
        if not os.path.exists(os.path.join(model_log_dir, "params.pth")):
            print(f"Skipping {model_name}: params.pth not found")
            continue

        if not os.path.exists(os.path.join(model_log_dir, "model.pth")):
            print(f"Skipping {model_name}: model.pth not found")
            continue

        print(f"Evaluating {model_name}...")

        try:
            # Load model parameters
            params = torch.load(
                os.path.join(model_log_dir, "params.pth"),
                weights_only=False
            )

            # Load models
            input_function_encoder, output_function_encoder, model, _ = load_models(
                log_dir=model_log_dir,
                dataset_info=dataset_info,
                params=params,
                device=device,
            )

            # Load forward model
            forward_model = load_forward_model(log_dir=model_log_dir, device=device)

            # Compute resimulation loss over all batches
            coeff_loss, pred_loss = compute_resimulation_loss_all_batches(
                model=model,
                model_name=model_name,
                test_dataloader=test_dataloader,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                forward_model=forward_model,
                device=device,
            )

            results[model_name] = (coeff_loss, pred_loss)
            print(f"  Coefficient Loss: {coeff_loss:.6e}")
            print(f"  Prediction Loss:  {pred_loss:.6e}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    return results


def print_results_table(results):
    """
    Print evaluation results in a formatted table.

    Args:
        results: Dictionary mapping model names to (coeff_loss, pred_loss) tuples
    """
    # Prepare table data
    table_data = []
    for model_name in sorted(results.keys()):
        coeff_loss, pred_loss = results[model_name]
        table_data.append([
            model_name,
            f"{coeff_loss:.6e}",
            f"{pred_loss:.6e}",
        ])

    # Print table
    headers = ["Model", "Coefficient Loss", "Prediction Loss"]
    print("\n" + "=" * 80)
    print("Model Evaluation Results")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models and compute resimulation loss statistics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., burgers, darcy, wave_scattering)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file to save results (CSV format)",
    )

    args = parser.parse_args()

    # Evaluate all models
    results = evaluate_all_models(
        log_dir=args.log_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Print results table
    print_results_table(results)

    # Save results to file if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Coefficient Loss", "Prediction Loss"])
            for model_name in sorted(results.keys()):
                coeff_loss, pred_loss = results[model_name]
                writer.writerow([model_name, coeff_loss, pred_loss])
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
