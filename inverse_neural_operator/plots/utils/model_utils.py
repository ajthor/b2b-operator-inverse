"""
Simple utilities for loading and evaluating inverse models in publication plotting scripts.
"""

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from models.load_model import load_models


def load_all_models(
    log_dir: str,
    dataset_info,
    model_names: Iterable[str],
    seed: int = 1,
    device: str = "cpu",
) -> Tuple[Dict[str, Tuple[torch.nn.Module, callable]], object, object]:
    """Load inverse models and shared encoders from a results directory.

    Args:
        log_dir: Base directory containing model subdirectories
        dataset_info: Dataset configuration object
        model_names: Names of models to load
        seed: Random seed used during training
        device: Device to load models onto

    Returns:
        models_dict: {model_name: (model, evaluate_fn)}
        input_function_encoder: Shared input encoder
        output_function_encoder: Shared output encoder
    """
    models_dict = {}
    input_function_encoder = None
    output_function_encoder = None

    for model_name in model_names:
        model_log_dir = os.path.join(log_dir, model_name, f"seed_{seed}")
        params_path = os.path.join(model_log_dir, "params.pth")

        if not os.path.exists(params_path):
            print(f"  Skipping {model_name} - not found")
            continue

        try:
            params = torch.load(params_path, weights_only=False)
            inp_enc, out_enc, model, evaluate_fn = load_models(
                log_dir=model_log_dir,
                dataset_info=dataset_info,
                params=params,
                device=device,
            )

            if input_function_encoder is None:
                input_function_encoder = inp_enc
                output_function_encoder = out_enc

            models_dict[model_name] = (model, evaluate_fn)
            print(f"  Loaded {model_name}")
        except Exception as exc:
            print(f"  Error loading {model_name}: {exc}")

    return models_dict, input_function_encoder, output_function_encoder


def evaluate_models_on_subset(
    test_dataset,
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    max_samples: int = 32,
    device: str = "cpu",
) -> Tuple[Dict[str, List[float]], List[Tuple[int, float]]]:
    """Evaluate models on a subset of the test dataset.

    Args:
        test_dataset: Test dataset to evaluate on
        models_dict: Dictionary of loaded models
        input_function_encoder: Input encoder
        output_function_encoder: Output encoder
        max_samples: Maximum number of samples to evaluate
        device: Device for computation

    Returns:
        per_model_mses: {model_name: [mse_per_sample]}
        sample_scores: [(sample_idx, median_mse_across_models)]
    """
    per_model_mses = {name: [] for name in models_dict.keys()}
    sample_scores = []

    n_eval = min(max_samples, len(test_dataset))

    for idx in range(n_eval):
        X, u_true, Y, s_observed = test_dataset[idx]
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)

        point = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )

        sample_mses = []
        for model_name, (model, evaluate_fn) in models_dict.items():
            model.eval()
            with torch.no_grad():
                u_pred, _ = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )
                mse = torch.mean((u_pred.squeeze(0) - u_true) ** 2).item()

            per_model_mses[model_name].append(mse)
            sample_mses.append(mse)

        if sample_mses:
            sample_scores.append((idx, float(np.median(sample_mses))))

    return per_model_mses, sample_scores


def select_models_and_sample(
    test_dataset,
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    max_models: int = 6,
    max_eval_samples: int = 32,
    sample_index: int | None = None,
    device: str = "cpu",
) -> Tuple[List[str], int]:
    """Evaluate models, rank by performance, and select a representative sample.

    This encapsulates the common workflow used by all publication plotting scripts.

    Args:
        test_dataset: Test dataset
        models_dict: Dictionary of loaded models
        input_function_encoder: Input encoder
        output_function_encoder: Output encoder
        max_models: Maximum number of top-performing models to return
        max_eval_samples: Number of samples to evaluate for ranking
        sample_index: Specific sample to use (if None, selects median sample)
        device: Device for computation

    Returns:
        models_to_plot: List of model names to plot (top performers)
        sample_index: Index of the representative sample
    """
    print("Evaluating models to select top performers...")
    per_model_mses, sample_scores = evaluate_models_on_subset(
        test_dataset,
        models_dict,
        input_function_encoder,
        output_function_encoder,
        max_samples=max_eval_samples,
        device=device,
    )

    # Rank models by mean MSE (lower is better)
    ranked_models = sorted(
        per_model_mses.keys(), key=lambda k: np.mean(per_model_mses[k])
    )
    models_to_plot = ranked_models[:max_models]
    print(f"  Selected models: {', '.join(models_to_plot)}")

    # Select representative sample
    if sample_index is None:
        if not sample_scores:
            raise ValueError("Unable to score samples - no sample scores available")
        median_idx = len(sample_scores) // 2
        sample_index = sample_scores[median_idx][0]
        print(f"  Selected representative sample: {sample_index}")
    else:
        print(f"  Using provided sample: {sample_index}")

    return models_to_plot, sample_index


def collect_predictions(
    sample,
    models_to_plot: Iterable[str],
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    output_transform=None,
    n_samples_per_model: int = 8,
    device: str = "cpu",
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Collect multiple prediction samples from each model for a single test sample.

    Args:
        sample: Single test sample (X, u_true, Y, s_observed)
        models_to_plot: Model names to collect predictions from
        models_dict: Dictionary of loaded models
        input_function_encoder: Input encoder
        output_function_encoder: Output encoder
        output_transform: Optional function to transform model outputs
        n_samples_per_model: Number of predictions to collect per model
        device: Device for computation

    Returns:
        predictions: {model_name: {"inputs": array, "outputs": array}}
        meta: {"x": array, "y": array, "u_true": array, "s_true": array}
    """
    X, u_true, Y, s_observed = sample
    X = X.to(device)
    u_true = u_true.to(device)
    Y = Y.to(device)
    s_observed = s_observed.to(device)

    point = (
        X.unsqueeze(0),
        u_true.unsqueeze(0),
        Y.unsqueeze(0),
        s_observed.unsqueeze(0),
    )

    predictions = {}
    meta = {
        "x": X.detach().cpu().numpy().flatten(),
        "y": Y.detach().cpu().numpy().flatten(),
        "u_true": u_true.detach().cpu().numpy().flatten(),
        "s_true": s_observed.detach().cpu().numpy().flatten(),
    }

    for model_name in models_to_plot:
        if model_name not in models_dict:
            continue

        model, evaluate_fn = models_dict[model_name]
        model.eval()

        input_samples = []
        output_samples = []

        with torch.no_grad():
            for _ in range(n_samples_per_model):
                eval_out = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )

                # Handle different evaluation output formats
                if isinstance(eval_out, (tuple, list)):
                    u_pred = eval_out[0]
                    second = eval_out[1] if len(eval_out) > 1 else None
                else:
                    u_pred = eval_out
                    second = None

                u_pred_np = u_pred.squeeze(0).detach().cpu().numpy().flatten()
                input_samples.append(u_pred_np)

                # Try to extract output samples if available
                if second is not None:
                    if output_transform is not None:
                        try:
                            s_resim = output_transform(second, Y)
                            s_resim_np = s_resim.squeeze(0).detach().cpu().numpy().flatten()
                            output_samples.append(s_resim_np)
                            continue
                        except Exception:
                            pass

                    try:
                        s_resim_np = second.squeeze(0).detach().cpu().numpy().flatten()
                        if s_resim_np.size == meta["s_true"].size:
                            output_samples.append(s_resim_np)
                    except Exception:
                        pass

        predictions[model_name] = {
            "inputs": np.stack(input_samples) if input_samples else np.empty((0,)),
            "outputs": np.stack(output_samples) if output_samples else np.empty((0,)),
        }

    return predictions, meta
