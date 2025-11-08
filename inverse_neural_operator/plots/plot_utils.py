"""
Utility functions for plotting best and worst case samples.

This module provides functions to identify the best and worst performing samples
across all test data based on re-simulation error.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from skimage.metrics import structural_similarity as compute_ssim


def find_best_worst_samples(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    test_dataset,
    device="cpu",
    metric: str = "mse",
    output_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, int, float, float]:
    """
    Find the best and worst performing samples based on a chosen metric.

    Evaluates the model on all test samples and computes the similarity/error
    between observed output and re-simulated output. Returns the indices of the
    best and worst samples according to the selected metric.

    Args:
        model: The trained inverse model
        evaluate_fn: Evaluation function for the model
        input_function_encoder: Input function encoder
        output_function_encoder: Output function encoder
        forward_model: Forward model for re-simulation
        test_dataset: Test dataset to evaluate
        device: Device to use for computation (default: "cpu")
        metric: Metric to use ("mse" or "ssim")
        output_shape: Desired reshape for outputs when computing SSIM

    Returns:
        Tuple of (best_idx, worst_idx, best_score, worst_score)
    """
    model.eval()
    forward_model.eval()

    sample_errors = []

    metric = metric.lower()
    if metric not in {"mse", "ssim"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'mse' or 'ssim'.")

    print(f"  Evaluating {len(test_dataset)} samples to find best/worst cases ({metric.upper()})...")

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

            if metric == "mse":
                score = torch.mean((s_observed - s_resim) ** 2).item()
            else:  # metric == "ssim"
                s_obs_np = s_observed.squeeze(-1).detach().cpu().numpy()
                s_resim_np = s_resim.squeeze(-1).detach().cpu().numpy()

                if output_shape is not None:
                    s_obs_np = s_obs_np.reshape(output_shape)
                    s_resim_np = s_resim_np.reshape(output_shape)
                else:
                    # Fallback to 2D reshape if possible, else keep 1D
                    if s_obs_np.ndim == 1:
                        s_obs_np = s_obs_np.reshape(1, -1)
                        s_resim_np = s_resim_np.reshape(1, -1)

                data_range = max(s_obs_np.max(), s_resim_np.max()) - min(
                    s_obs_np.min(), s_resim_np.min()
                )
                if data_range == 0:
                    data_range = 1.0
                score = compute_ssim(
                    s_obs_np,
                    s_resim_np,
                    data_range=data_range,
                    channel_axis=None,
                )

            sample_errors.append((idx, score))

            # Print progress every 100 samples
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{len(test_dataset)} samples...")

    # Sort to find best and worst based on metric
    if metric == "mse":
        sample_errors.sort(key=lambda x: x[1])  # lower is better
    else:  # SSIM
        sample_errors.sort(key=lambda x: x[1], reverse=True)  # higher is better

    best_idx, best_score = sample_errors[0]
    worst_idx, worst_score = sample_errors[-1]

    metric_label = "MSE" if metric == "mse" else "SSIM"
    score_fmt = "{:.6e}" if metric == "mse" else "{:.4f}"
    print(f"  Best sample: idx={best_idx}, {metric_label}={score_fmt.format(best_score)}")
    print(f"  Worst sample: idx={worst_idx}, {metric_label}={score_fmt.format(worst_score)}")

    return best_idx, worst_idx, best_score, worst_score
