"""
Simple utilities for loading and evaluating inverse models in publication plotting scripts.
"""

import json
import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from models.load_model import load_models
from utils.imports import import_model_functions


def _load_cached_evaluation(
    cache_path: str, cache_key: str | None = None
) -> Dict | None:
    """Load cached evaluation summary from disk."""
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"  Warning: Failed to read cache at {cache_path}: {exc}")
        return None

    if cache_key and isinstance(data, dict):
        entries = data.get("entries")
        if isinstance(entries, dict):
            return entries.get(cache_key)
    return data


def _write_cached_evaluation(
    cache_path: str, payload: Dict, cache_key: str | None = None
) -> None:
    """Persist evaluation summary to cache."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if cache_key:
            existing_entries = {}
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as handle:
                        existing = json.load(handle)
                    if isinstance(existing, dict):
                        if isinstance(existing.get("entries"), dict):
                            existing_entries = dict(existing["entries"])
                except Exception as exc:
                    print(
                        f"  Warning: Unable to merge existing cache at {cache_path}: {exc}"
                    )
            existing_entries[cache_key] = payload
            final_payload = {"version": 1, "entries": existing_entries}
        else:
            final_payload = payload

        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=2)
        print(f"  Cached evaluation summary → {cache_path}")
    except Exception as exc:
        print(f"  Warning: Unable to write cache at {cache_path}: {exc}")


def load_all_models(
    log_dir: str,
    dataset_info,
    model_names: Iterable[str],
    seed: int = 1,
    device: str = "cpu",
) -> Tuple[Dict[str, Tuple[torch.nn.Module, callable]], object, object]:
    """Load inverse models and shared encoders from a results directory.

    Args:
        log_dir: Base directory or dataset log directory containing trained models.
        dataset_info: Dataset info or dataset name (used to locate models directory).
        model_names: Names of models to load
        seed: Random seed used during training
        device: Device to load models onto

    Returns:
        models_dict: {model_name: (model, evaluate_fn)}
        input_function_encoder: Shared input encoder
        output_function_encoder: Shared output encoder
    """

    def _infer_dataset_name(dataset_info_obj, default_name):
        if isinstance(dataset_info_obj, str):
            return dataset_info_obj
        if isinstance(dataset_info_obj, dict):
            for key in ("dataset", "dataset_name", "name"):
                if key in dataset_info_obj:
                    return dataset_info_obj[key]
        if hasattr(dataset_info_obj, "dataset"):
            return getattr(dataset_info_obj, "dataset")
        return default_name

    def _resolve_dataset_dir(base_path, dataset_name):
        normalized = os.path.abspath(base_path)
        candidates: List[str] = []

        if dataset_name:
            # Prefer explicit models/<dataset> layout which matches training outputs.
            candidates.append(os.path.join(normalized, "models", dataset_name))
            # Handle case where log_dir already points to models/
            if os.path.basename(normalized) == "models":
                candidates.append(os.path.join(normalized, dataset_name))
            # Handle case where log_dir already points to dataset dir.
            if os.path.basename(normalized) == dataset_name:
                parent = os.path.dirname(normalized)
                candidates.append(normalized)
                candidates.append(os.path.join(parent, "models", dataset_name))
            # Next, allow base/dataset style
            candidates.append(os.path.join(normalized, dataset_name))

        # Finally, fall back to the base path as last resort.
        candidates.append(normalized)

        # Deduplicate while preserving order.
        seen = set()
        deduped = []
        for path in candidates:
            if path not in seen:
                deduped.append(path)
                seen.add(path)

        for candidate in deduped:
            if os.path.isdir(candidate):
                return candidate

        # Fall back to default models/<dataset> or normalized path even if they don't exist.
        return os.path.join(normalized, "models", dataset_name) if dataset_name else normalized

    def _derive_base_dir(dataset_dir):
        parts = dataset_dir.rstrip(os.sep).split(os.sep)
        if "models" in parts:
            idx = parts.index("models")
            return os.sep.join(parts[:idx]) if idx > 0 else "."
        return os.path.dirname(dataset_dir)

    models_dict = {}
    input_function_encoder = None
    output_function_encoder = None

    normalized_log_dir = os.path.abspath(log_dir)
    default_dataset = os.path.basename(normalized_log_dir.rstrip(os.sep))
    dataset = _infer_dataset_name(dataset_info, default_dataset)
    dataset_dir = _resolve_dataset_dir(normalized_log_dir, dataset)
    base_dir = _derive_base_dir(dataset_dir)
    print(f"  Model lookup: dataset='{dataset}', dataset_dir='{dataset_dir}', base_dir='{base_dir}'")

    if not os.path.isdir(dataset_dir):
        print(f"  Warning: Dataset directory not found at {dataset_dir}")
        return models_dict, input_function_encoder, output_function_encoder

    for model_name in model_names:
        model_dir = os.path.join(dataset_dir, model_name, f"seed_{seed}")
        params_path = os.path.join(model_dir, "params.pth")

        if not os.path.exists(params_path):
            print(f"  Skipping {model_name} - not found (expected {params_path})")
            continue

        try:
            inp_enc, out_enc, model, evaluate_fn = load_models(
                base_dir=base_dir,
                dataset=dataset,
                model_name=model_name,
                seed=seed,
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


def evaluate_models(
    test_dataset,
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    forward_model,
    device: str = "cpu",
) -> Tuple[Dict[str, Dict[str, List[float]]], List[Tuple[int, float]]]:
    """Evaluate models on the entire test dataset using proper resimulation loss.

    Args:
        test_dataset: Test dataset to evaluate on
        models_dict: Dictionary of loaded models
        input_function_encoder: Input encoder
        output_function_encoder: Output encoder
        forward_model: Forward model for re-simulation
        device: Device for computation

    Returns:
        per_model_losses: {model_name: {"coeff_loss": [...], "pred_loss": [...]}}
        sample_scores: [(sample_idx, median_pred_loss_across_models)]
    """
    per_model_losses = {
        name: {"coeff_loss": [], "pred_loss": []} for name in models_dict.keys()
    }
    sample_scores = []

    # Import resimulation_loss functions for each model
    resim_loss_fns = {}
    for model_name in models_dict.keys():
        try:
            resim_loss_fns[model_name] = import_model_functions(
                model_name, "resimulation_loss"
            )
        except (ValueError, AttributeError) as e:
            print(
                f"  Warning: Could not import resimulation_loss for {model_name}: {e}"
            )
            continue

    print(f"  Evaluating on {len(test_dataset)} test samples...")

    for idx in range(len(test_dataset)):
        X, u_true, Y, s_observed = test_dataset[idx]
        X = X.to(device)
        u_true = u_true.to(device)
        Y = Y.to(device)
        s_observed = s_observed.to(device)

        batch = (
            X.unsqueeze(0),
            u_true.unsqueeze(0),
            Y.unsqueeze(0),
            s_observed.unsqueeze(0),
        )

        sample_pred_losses = []
        for model_name, (model, _) in models_dict.items():
            if model_name not in resim_loss_fns:
                continue

            model.eval()
            if forward_model is not None:
                forward_model.eval()

            with torch.no_grad():
                coeff_loss, pred_loss = resim_loss_fns[model_name](
                    model=model,
                    batch=batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    forward_model=forward_model,
                    n_samples=1,  # Deterministic evaluation
                )

            per_model_losses[model_name]["coeff_loss"].append(coeff_loss)
            per_model_losses[model_name]["pred_loss"].append(pred_loss)
            sample_pred_losses.append(pred_loss)

        if sample_pred_losses:
            sample_scores.append((idx, float(np.median(sample_pred_losses))))

    return per_model_losses, sample_scores


def select_models_and_sample(
    test_dataset,
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    forward_model,
    max_models: int = 6,
    sample_index: int | None = None,
    device: str = "cpu",
    return_metrics: bool = False,
    cache_path: str | None = None,
    cache_metadata: Dict | None = None,
    cache_key: str | None = None,
) -> Tuple[List[str], int] | Tuple[List[str], int, Dict[str, Dict[str, List[float]]]]:
    """Evaluate models on entire test set, rank by re-simulation performance, and select median sample.

    Args:
        test_dataset: Test dataset
        models_dict: Dictionary of loaded models
        input_function_encoder: Input encoder
        output_function_encoder: Output encoder
        forward_model: Forward model for re-simulation
        max_models: Maximum number of top-performing models to return
        sample_index: Specific sample to use (if None, selects median-performing sample)
        device: Device for computation
        return_metrics: If True, also return per-model loss history
        cache_path: Optional path for storing/loading cached evaluation summaries
        cache_metadata: Optional metadata dict to validate cached entries
        cache_key: Optional key when storing multiple cache entries in one file

    Returns:
        models_to_plot: List of model names to plot (top performers)
        sample_index: Index of the representative sample
        per_model_losses (optional): Re-simulation loss history per model
    """
    dataset_size = len(test_dataset)
    sorted_model_names = sorted(models_dict.keys())
    cache_metadata = cache_metadata or {}

    if cache_path and not return_metrics:
        cache_data = _load_cached_evaluation(cache_path, cache_key)
        cache_valid = bool(cache_data)
        if cache_valid:
            if cache_data.get("model_names_sorted") != sorted_model_names:
                cache_valid = False
            elif cache_data.get("dataset_size") != dataset_size:
                cache_valid = False
            else:
                cached_meta = cache_data.get("metadata", {})
                for key, value in cache_metadata.items():
                    if cached_meta.get(key) != value:
                        cache_valid = False
                        break

        if cache_valid:
            ranked_models = cache_data.get("ranked_models", [])
            if not ranked_models:
                cache_valid = False

        if cache_valid:
            models_to_plot = ranked_models[:max_models]
            if not models_to_plot:
                cache_valid = False

        if cache_valid:
            mean_pred_losses = cache_data.get("mean_pred_losses", {})
            print(
                f"  Loaded cached evaluation results ({len(ranked_models)} models ranked)"
            )
            print(f"  Top {len(models_to_plot)} models by cached re-simulation loss:")
            for idx, model_name in enumerate(models_to_plot, 1):
                value = mean_pred_losses.get(model_name)
                if isinstance(value, (float, int)):
                    print(f"    {idx}. {model_name}: {value:.6e}")
                else:
                    print(f"    {idx}. {model_name}: N/A")

            cache_sample_index = cache_data.get("median_sample_index")
            if sample_index is None:
                if cache_sample_index is None:
                    print("  Cached results missing median sample, recomputing...")
                else:
                    cached_loss = cache_data.get("median_pred_loss")
                    if isinstance(cached_loss, (float, int)):
                        print(
                            f"  Using cached median sample: {cache_sample_index} "
                            f"(median pred_loss: {cached_loss:.6e})"
                        )
                    else:
                        print(f"  Using cached median sample: {cache_sample_index}")
                    return models_to_plot, cache_sample_index
            else:
                print(f"  Using provided sample: {sample_index}")
                return models_to_plot, sample_index

    print("Evaluating models on full test set using re-simulation loss...")
    per_model_losses, sample_scores = evaluate_models(
        test_dataset,
        models_dict,
        input_function_encoder,
        output_function_encoder,
        forward_model,
        device=device,
    )

    mean_pred_losses = {
        name: (float(np.mean(losses["pred_loss"])) if losses["pred_loss"] else None)
        for name, losses in per_model_losses.items()
    }

    # Rank models by mean pred_loss (re-simulation error, lower is better)
    ranked_models = sorted(
        per_model_losses.keys(),
        key=lambda k: (
            mean_pred_losses[k] if mean_pred_losses[k] is not None else float("inf")
        ),
    )
    models_to_plot = ranked_models[:max_models]

    # Print model rankings with mean pred_loss
    print(f"  Top {max_models} models by mean re-simulation loss:")
    for i, model_name in enumerate(models_to_plot, 1):
        mean_pred_loss = mean_pred_losses.get(model_name)
        if mean_pred_loss is not None:
            print(f"    {i}. {model_name}: {mean_pred_loss:.6e}")
        else:
            print(f"    {i}. {model_name}: N/A")

    # Select representative sample (median-performing across all models by pred_loss)
    sample_scores_sorted = []
    median_sample_index = None
    median_pred_loss = None
    if sample_scores:
        sample_scores_sorted = sorted(sample_scores, key=lambda x: x[1])
        median_idx = len(sample_scores_sorted) // 2
        median_sample_index = sample_scores_sorted[median_idx][0]
        median_pred_loss = sample_scores_sorted[median_idx][1]

    if sample_index is None:
        if median_sample_index is None:
            raise ValueError("Unable to score samples - no sample scores available")
        if median_pred_loss is not None:
            print(
                f"  Selected median-performing sample: {median_sample_index} "
                f"(median pred_loss: {median_pred_loss:.6e})"
            )
        else:
            print(f"  Selected median-performing sample: {median_sample_index}")
        sample_index = median_sample_index
    else:
        print(f"  Using provided sample: {sample_index}")

    if cache_path and median_sample_index is not None:
        cache_payload = {
            "version": 1,
            "timestamp": time.time(),
            "model_names_sorted": sorted_model_names,
            "ranked_models": ranked_models,
            "mean_pred_losses": mean_pred_losses,
            "median_sample_index": int(median_sample_index),
            "median_pred_loss": (
                float(median_pred_loss) if median_pred_loss is not None else None
            ),
            "dataset_size": dataset_size,
            "metadata": cache_metadata,
        }
        _write_cached_evaluation(cache_path, cache_payload, cache_key)

    if return_metrics:
        return models_to_plot, sample_index, per_model_losses

    return models_to_plot, sample_index

    # # FAST MODE: Evaluate on small batch and use sample 0 (much faster for publication plots)
    # fast_mode_batch_size = 20
    # print(f"Fast mode: Evaluating models on {fast_mode_batch_size} samples...")

    # # Create a subset of the dataset for quick evaluation
    # subset_indices = list(range(min(fast_mode_batch_size, len(test_dataset))))
    # per_model_losses = {
    #     name: {"coeff_loss": [], "pred_loss": []} for name in models_dict.keys()
    # }

    # # Import resimulation_loss functions for each model
    # resim_loss_fns = {}
    # for model_name in models_dict.keys():
    #     try:
    #         resim_loss_fns[model_name] = import_model_functions(
    #             model_name, "resimulation_loss"
    #         )
    #     except (ValueError, AttributeError) as e:
    #         print(
    #             f"  Warning: Could not import resimulation_loss for {model_name}: {e}"
    #         )
    #         continue

    # # Quick evaluation on subset
    # for idx in subset_indices:
    #     X, u_true, Y, s_observed = test_dataset[idx]
    #     X = X.to(device)
    #     u_true = u_true.to(device)
    #     Y = Y.to(device)
    #     s_observed = s_observed.to(device)

    #     batch = (
    #         X.unsqueeze(0),
    #         u_true.unsqueeze(0),
    #         Y.unsqueeze(0),
    #         s_observed.unsqueeze(0),
    #     )

    #     for model_name, (model, _) in models_dict.items():
    #         if model_name not in resim_loss_fns:
    #             continue

    #         model.eval()
    #         if forward_model is not None:
    #             forward_model.eval()

    #         with torch.no_grad():
    #             coeff_loss, pred_loss = resim_loss_fns[model_name](
    #                 model=model,
    #                 batch=batch,
    #                 input_function_encoder=input_function_encoder,
    #                 output_function_encoder=output_function_encoder,
    #                 forward_model=forward_model,
    #                 n_samples=1,  # Deterministic evaluation
    #             )

    #         per_model_losses[model_name]["coeff_loss"].append(coeff_loss)
    #         per_model_losses[model_name]["pred_loss"].append(pred_loss)

    # # Rank models by mean pred_loss (re-simulation error, lower is better)
    # ranked_models = sorted(
    #     per_model_losses.keys(),
    #     key=lambda k: (
    #         np.mean(per_model_losses[k]["pred_loss"])
    #         if per_model_losses[k]["pred_loss"]
    #         else float("inf")
    #     ),
    # )
    # models_to_plot = ranked_models[:max_models]

    # # Print model rankings with mean pred_loss
    # print(f"  Top {max_models} models by mean re-simulation loss (on {fast_mode_batch_size} samples):")
    # for i, model_name in enumerate(models_to_plot, 1):
    #     pred_losses = per_model_losses[model_name]["pred_loss"]
    #     if pred_losses:
    #         mean_pred_loss = np.mean(pred_losses)
    #         print(f"    {i}. {model_name}: {mean_pred_loss:.6e}")
    #     else:
    #         print(f"    {i}. {model_name}: N/A")

    # # Use sample 0 by default in fast mode
    # if sample_index is None:
    #     sample_index = 0
    #     print(f"  Using sample: {sample_index} (fast mode default)")
    # else:
    #     print(f"  Using provided sample: {sample_index}")

    # if return_metrics:
    #     return models_to_plot, sample_index, per_model_losses

    # return models_to_plot, sample_index


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
                            s_resim_np = (
                                s_resim.squeeze(0).detach().cpu().numpy().flatten()
                            )
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


def evaluate_models_on_subset(
    dataset,
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    max_samples: int | None = None,
    device: str = "cpu",
):
    """Compute simple MSE scores for a subset of samples to compare inverse models."""
    per_model_mses = {name: [] for name in models_dict.keys()}
    per_sample_metrics = []

    if max_samples is None or max_samples <= 0:
        max_samples = len(dataset)

    n_samples = min(len(dataset), max_samples)

    for idx in range(n_samples):
        X, u_true, Y, s_observed = dataset[idx]
        batch = (
            X.unsqueeze(0).to(device),
            u_true.unsqueeze(0).to(device),
            Y.unsqueeze(0).to(device),
            s_observed.unsqueeze(0).to(device),
        )

        sample_metrics = {}
        for model_name, (model, evaluate_fn) in models_dict.items():
            model.eval()
            with torch.no_grad():
                eval_out = evaluate_fn(
                    model,
                    batch,
                    input_function_encoder,
                    output_function_encoder,
                )

                if isinstance(eval_out, (tuple, list)):
                    u_pred = eval_out[0]
                else:
                    u_pred = eval_out

                mse = torch.mean((u_pred - batch[1]) ** 2).item()
                per_model_mses[model_name].append(mse)
                sample_metrics[model_name] = mse

        per_sample_metrics.append((idx, sample_metrics))

    return per_model_mses, per_sample_metrics
