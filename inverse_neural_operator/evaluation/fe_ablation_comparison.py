"""Compare B2B FE architectures against deterministic DeepONet and FNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.runtime.paths import model_artifact_dir, models_root, results_root


VARIANTS = ("relu", "silu", "siren", "siren2")
LABELS = {
    "relu": "B2B + ReLU FE",
    "silu": "B2B + SiLU FE",
    "siren": "B2B + SIREN FE",
    "siren2": "B2B + SIREN² FE",
    "deeponet": "DeepONet",
    "fno": "FNO",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--b2b-config-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def _parameter_count(model):
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


def _plot_sample(path, x, target, predictions, sample_index, dataset_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    coordinates = x[sample_index].reshape(-1).numpy()
    truth = target[sample_index].reshape(-1).numpy()
    for axis, (name, values) in zip(axes.reshape(-1), predictions.items()):
        prediction = values[sample_index].reshape(-1).numpy()
        relative = ((prediction - truth) ** 2).sum() ** 0.5 / max(
            (truth**2).sum() ** 0.5, 1e-12
        )
        axis.plot(coordinates, truth, color="black", linewidth=2, label="Target")
        axis.plot(coordinates, prediction, linewidth=1.5, label=LABELS[name])
        axis.fill_between(coordinates, truth, prediction, alpha=0.12)
        axis.set_title(f"{LABELS[name]} · rel. L2={relative:.3f}")
        axis.set_xlabel("x")
        axis.set_ylabel("normalized u")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"{dataset_name} test sample {sample_index}", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_metrics(path, metrics, dataset_name, samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(metrics)
    keys = ("mae", "mse", "relative_l2")
    titles = ("MAE", "MSE", "Relative L2")
    colors = ("#4c78a8", "#72b7b2", "#f58518", "#e45756", "#b279a2", "#54a24b")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    for axis, key, title in zip(axes, keys, titles):
        values = [metrics[name][key] for name in names]
        bars = axis.bar(np.arange(len(names)), values, color=colors)
        axis.set_xticks(
            np.arange(len(names)),
            [LABELS[name] for name in names],
            rotation=28,
            ha="right",
        )
        axis.set_title(title)
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
    figure.suptitle(f"{dataset_name}: full test split (n={samples})", fontsize=15)
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
    base_config = load_experiment_config(args.base_config)
    dataset_name = base_config.dataset.name
    device = torch.device(args.device)
    model_root = models_root(required=True, override=args.models_dir)
    output_root = results_root(args.results_dir) / "comparison" / dataset_name
    dataset = load_overhaul_dataset(base_config, "test")
    dataset_info = dataset.get_info()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    b2b_models = {}
    b2b_encoders = {}
    for variant in VARIANTS:
        path = Path(args.b2b_config_dir) / f"{dataset_name}_b2b_fe_{variant}_full.yaml"
        config = load_experiment_config(path)
        encoder_dir = model_artifact_dir(
            model_root,
            dataset_name,
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
        model = create_inverse_model(
            "nonlinear",
            input_size=n_basis,
            output_size=n_basis,
            hidden_sizes=config.inverse_models.hidden_sizes,
        ).to(device)
        model_dir = model_artifact_dir(
            model_root,
            dataset_name,
            "inverse_models",
            config.inverse_models.artifact,
            args.seed,
        )
        model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=str(device)))
        b2b_models[variant] = model.eval()
        b2b_encoders[variant] = (input_encoder, output_encoder)

    direct_models = {}
    for name in ("deeponet", "fno"):
        model = create_baseline_model(name, dataset_info, base_config).to(device)
        model_dir = model_artifact_dir(model_root, dataset_name, "baselines", name, args.seed)
        model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=str(device)))
        direct_models[name] = model.eval()

    x_values = []
    targets = []
    batches = {name: [] for name in (*VARIANTS, "deeponet", "fno")}
    with torch.no_grad():
        for x, u, y, s in loader:
            x, u, y, s = (value.to(device) for value in (x, u, y, s))
            for variant in VARIANTS:
                input_encoder, output_encoder = b2b_encoders[variant]
                beta, _ = output_encoder.compute_coefficients(y, s)
                alpha = b2b_models[variant](beta)
                batches[variant].append(input_encoder(x, alpha).cpu())
            for name, model in direct_models.items():
                batches[name].append(model.predict_inverse(x, y, s).cpu())
            x_values.append(x.cpu())
            targets.append(u.cpu())

    x_all = torch.cat(x_values)
    target_all = torch.cat(targets)
    predictions = {name: torch.cat(values) for name, values in batches.items()}
    metrics = {name: _metrics(prediction, target_all) for name, prediction in predictions.items()}
    for variant in VARIANTS:
        input_encoder, output_encoder = b2b_encoders[variant]
        inverse_parameters = _parameter_count(b2b_models[variant])
        representation_parameters = _parameter_count(input_encoder) + _parameter_count(
            output_encoder
        )
        metrics[variant].update(
            {
                "inverse_parameters": inverse_parameters,
                "representation_parameters": representation_parameters,
                "total_parameters": inverse_parameters + representation_parameters,
            }
        )
    for name, model in direct_models.items():
        count = _parameter_count(model)
        metrics[name].update(
            {
                "inverse_parameters": count,
                "representation_parameters": 0,
                "total_parameters": count,
            }
        )

    payload = {
        "dataset": dataset_name,
        "split": "test",
        "samples": len(dataset),
        "training_samples": 10_000,
        "seed": args.seed,
        "training_steps": base_config.baselines.max_steps,
        "metrics": metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    _plot_metrics(output_root / "metrics.png", metrics, dataset_name, len(dataset))
    for sample_index in args.sample_indices:
        _plot_sample(
            output_root / f"sample_{sample_index:03d}_predictions.png",
            x_all,
            target_all,
            predictions,
            sample_index,
            dataset_name,
        )
    print(json.dumps(payload, indent=2))
    print(f"Wrote full comparison outputs to {output_root}")


if __name__ == "__main__":
    main()
