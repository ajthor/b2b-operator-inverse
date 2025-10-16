"""
Utility functions for plotting best and worst case samples.

This module provides functions to identify the best and worst performing samples
across all test data based on re-simulation error.
"""

import torch
import numpy as np
from typing import Tuple


def find_best_worst_samples(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    test_dataset,
    device="cpu",
) -> Tuple[int, int, float, float]:
    """
    Find the best and worst performing samples based on re-simulation MSE.

    Evaluates the model on all test samples and computes the MSE between
    observed output and re-simulated output. Returns the indices of the
    best (lowest MSE) and worst (highest MSE) samples.

    Args:
        model: The trained inverse model
        evaluate_fn: Evaluation function for the model
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model for re-simulation
        test_dataset: Test dataset to evaluate
        device: Device to use for computation (default: "cpu")

    Returns:
        Tuple of (best_idx, worst_idx, best_mse, worst_mse)
    """
    model.eval()
    forward_model.eval()

    sample_errors = []

    print(f"  Evaluating {len(test_dataset)} samples to find best/worst cases...")

    with torch.no_grad():
        for idx, sample in enumerate(test_dataset):
            X, u_true, Y, s_observed = sample

            # Ensure tensors are on correct device
            X = X.to(device)
            u_true = u_true.to(device)
            Y = Y.to(device)
            s_observed = s_observed.to(device)

            # Get model prediction for input
            point = (
                X.unsqueeze(0),
                u_true.unsqueeze(0),
                Y.unsqueeze(0),
                s_observed.unsqueeze(0),
            )
            u_pred, _ = evaluate_fn(
                model, point, input_function_encoder, output_function_encoder
            )
            u_pred = u_pred.squeeze(0)

            # Re-simulate using forward model
            X_batch = X.unsqueeze(0)
            u_pred_batch = u_pred.unsqueeze(0)
            Y_batch = Y.unsqueeze(0)

            # Compute alpha coefficients from predicted input
            alpha, _ = input_function_encoder.compute_coefficients(X_batch, u_pred_batch)

            # Forward pass through model to get beta coefficients
            beta_pred = forward_model.forward(alpha)

            # Reconstruct re-simulation output
            s_resim = output_function_encoder(Y_batch, beta_pred)
            s_resim = s_resim.squeeze(0)

            # Compute MSE for this sample
            mse = torch.mean((s_observed - s_resim) ** 2).item()
            sample_errors.append((idx, mse))

            # Print progress every 100 samples
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(test_dataset)} samples...")

    # Sort by error to find best and worst
    sample_errors.sort(key=lambda x: x[1])

    best_idx, best_mse = sample_errors[0]
    worst_idx, worst_mse = sample_errors[-1]

    print(f"  Best sample: idx={best_idx}, MSE={best_mse:.6e}")
    print(f"  Worst sample: idx={worst_idx}, MSE={worst_mse:.6e}")

    return best_idx, worst_idx, best_mse, worst_mse
