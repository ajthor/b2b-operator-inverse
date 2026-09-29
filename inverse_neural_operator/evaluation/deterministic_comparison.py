"""Evaluate and plot deterministic inverse baselines on one test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.runtime.paths import model_artifact_dir, models_root, results_root


MODEL_LABELS = {
    "b2b_nonlinear": "B2B nonlinear",
    "deeponet": "DeepONet",
    "fno": "FNO",
    "nystrom_gp": "Nyström GP mean",
    "rff_gp": "RFF GP mean",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--gp-config",
        default=None,
        help="Optional Nyström-GP config; adds its posterior mean to the comparison.",
    )
    parser.add_argument("--gp-artifact", default=None)
    parser.add_argument(
        "--gp-model",
        default="nystrom_gp",
        choices=["nystrom_gp", "rff_gp"],
    )
    return parser.parse_args()


def _parameter_count(model) -> int:
    return sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
    )


def _metrics(prediction, target):
    import torch

    difference = prediction - target
    batch = target.shape[0]
    relative = torch.linalg.vector_norm(difference.reshape(batch, -1), dim=1) / torch.clamp(
        torch.linalg.vector_norm(target.reshape(batch, -1), dim=1), min=1e-12
    )
    return {
        "mae": float(difference.abs().mean().item()),
        "mse": float(difference.square().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(relative.mean().item()),
    }


def _plot_samples(path: Path, x, target, predictions, sample_indices, dataset_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "b2b_nonlinear": "#4c78a8",
        "deeponet": "#f58518",
        "fno": "#54a24b",
        "nystrom_gp": "#b279a2",
        "rff_gp": "#e45756",
    }
    figure, axes = plt.subplots(
        len(sample_indices),
        2,
        figsize=(13, 3.4 * len(sample_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, sample_index in enumerate(sample_indices):
        coordinates = x[sample_index].reshape(-1).numpy()
        truth = target[sample_index].reshape(-1).numpy()
        axes[row, 0].plot(coordinates, truth, color="black", linewidth=2.2, label="Target")
        for name, values in predictions.items():
            prediction = values[sample_index].reshape(-1).numpy()
            axes[row, 0].plot(
                coordinates,
                prediction,
                color=colors[name],
                linewidth=1.35,
                label=MODEL_LABELS[name],
            )
            axes[row, 1].plot(
                coordinates,
                abs(prediction - truth),
                color=colors[name],
                linewidth=1.35,
                label=MODEL_LABELS[name],
            )
        axes[row, 0].set_title(f"Test sample {sample_index}: normalized input field")
        axes[row, 1].set_title(f"Test sample {sample_index}: absolute error")
        axes[row, 0].set_ylabel("normalized u")
        axes[row, 1].set_ylabel("absolute error")
        for axis in axes[row]:
            axis.set_xlabel("x")
            axis.grid(alpha=0.2)
        if row == 0:
            axes[row, 0].legend(ncol=2, fontsize=9)
            axes[row, 1].legend(fontsize=9)
    figure.suptitle(f"{dataset_name}: deterministic inverse comparison", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_metrics(path: Path, metrics, dataset_name, sample_count):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(metrics)
    color_map = {
        "b2b_nonlinear": "#4c78a8",
        "deeponet": "#f58518",
        "fno": "#54a24b",
        "nystrom_gp": "#b279a2",
        "rff_gp": "#e45756",
    }
    keys = ["mae", "mse", "relative_l2"]
    labels = ["MAE", "MSE", "Relative L2"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, key, label in zip(axes, keys, labels):
        values = [metrics[name][key] for name in names]
        bars = axis.bar(np.arange(len(names)), values, color=[color_map[name] for name in names])
        axis.set_xticks(np.arange(len(names)), [MODEL_LABELS[name] for name in names], rotation=20)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(f"{dataset_name}: full test split (n={sample_count})", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader

    from inverse_neural_operator.baselines.build import create_baseline_model
    from inverse_neural_operator.data.overhaul import load_overhaul_dataset
    from inverse_neural_operator.function_encoders.artifacts import load_function_encoder
    from inverse_neural_operator.inverse.build import create_inverse_model

    args = parse_args()
    config = load_experiment_config(args.config)
    device = torch.device(args.device)
    model_root = models_root(required=True, override=args.models_dir)
    output_root = results_root(args.results_dir) / "comparison" / config.dataset.name
    dataset = load_overhaul_dataset(config, "test")
    dataset_info = dataset.get_info()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    encoder_dir = model_artifact_dir(
        model_root,
        config.dataset.name,
        "function_encoders",
        config.inverse_models.function_encoder_artifact,
        args.seed,
    )
    input_encoder = load_function_encoder(
        encoder_dir, encoder_type="input", dataset_info=dataset_info, device=device
    )
    output_encoder = load_function_encoder(
        encoder_dir, encoder_type="output", dataset_info=dataset_info, device=device
    )
    n_basis = config.function_encoders.basis.n_basis
    b2b = create_inverse_model(
        "nonlinear",
        input_size=n_basis,
        output_size=n_basis,
        hidden_sizes=config.inverse_models.hidden_sizes,
    ).to(device)
    b2b_dir = model_artifact_dir(
        model_root,
        config.dataset.name,
        "inverse_models",
        config.inverse_models.artifact or "nonlinear",
        args.seed,
    )
    b2b.load_state_dict(load_file(str(b2b_dir / "model.safetensors"), device=str(device)))
    models = {"b2b_nonlinear": b2b.eval()}
    for name in ("deeponet", "fno"):
        model = create_baseline_model(name, dataset_info, config).to(device)
        model_dir = model_artifact_dir(
            model_root, config.dataset.name, "baselines", name, args.seed
        )
        model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=str(device)))
        models[name] = model.eval()
    if args.gp_config:
        gp_config = load_experiment_config(args.gp_config)
        if gp_config.dataset.name != config.dataset.name:
            raise ValueError("GP and comparison configs must use the same dataset.")
        gp = create_inverse_model(
            args.gp_model,
            input_size=n_basis,
            output_size=n_basis,
            hidden_sizes=[],
            nystrom_gp_config=gp_config.inverse_models.nystrom_gp,
            rff_gp_config=gp_config.inverse_models.rff_gp,
        ).to(device)
        gp_dir = model_artifact_dir(
            model_root,
            config.dataset.name,
            "inverse_models",
            args.gp_artifact or gp_config.inverse_models.artifact or args.gp_model,
            args.seed,
        )
        gp.load_state_dict(load_file(str(gp_dir / "model.safetensors"), device=str(device)))
        models[args.gp_model] = gp.eval()

    x_values = []
    targets = []
    prediction_batches = {name: [] for name in models}
    with torch.no_grad():
        for x, u, y, s in loader:
            x, u, y, s = (value.to(device) for value in (x, u, y, s))
            beta, _ = output_encoder.compute_coefficients(y, s)
            alpha_prediction = b2b(beta)
            prediction_batches["b2b_nonlinear"].append(input_encoder(x, alpha_prediction).cpu())
            prediction_batches["deeponet"].append(models["deeponet"].predict_inverse(x, y, s).cpu())
            prediction_batches["fno"].append(models["fno"].predict_inverse(x, y, s).cpu())
            if args.gp_model in models:
                gp_alpha = models[args.gp_model](beta)
                prediction_batches[args.gp_model].append(input_encoder(x, gp_alpha).cpu())
            x_values.append(x.cpu())
            targets.append(u.cpu())

    x_all = torch.cat(x_values)
    target_all = torch.cat(targets)
    predictions = {name: torch.cat(values) for name, values in prediction_batches.items()}
    metrics = {name: _metrics(prediction, target_all) for name, prediction in predictions.items()}
    for name, model in models.items():
        inverse_parameters = _parameter_count(model)
        representation_parameters = 0
        if name in {"b2b_nonlinear", "nystrom_gp", "rff_gp"}:
            representation_parameters = _parameter_count(input_encoder) + _parameter_count(
                output_encoder
            )
        metrics[name]["inverse_parameters"] = inverse_parameters
        metrics[name]["representation_parameters"] = representation_parameters
        metrics[name]["total_parameters"] = inverse_parameters + representation_parameters
        if name in {"nystrom_gp", "rff_gp"}:
            metrics[name]["inverse_state_values"] = sum(
                value.numel() for value in model.state_dict().values()
            )
    payload = {
        "dataset": config.dataset.name,
        "split": "test",
        "samples": len(dataset),
        "seed": args.seed,
        "training_steps": config.baselines.max_steps,
        "metrics": metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    _plot_samples(
        output_root / "predictions.png",
        x_all,
        target_all,
        predictions,
        args.sample_indices,
        config.dataset.name,
    )
    _plot_metrics(output_root / "metrics.png", metrics, config.dataset.name, len(dataset))
    print(json.dumps(payload, indent=2))
    print(f"Wrote comparison outputs to {output_root}")


if __name__ == "__main__":
    main()
