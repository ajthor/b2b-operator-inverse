"""Export Nyström-GP posterior uncertainty in coefficient and function space."""

from __future__ import annotations

import argparse
from pathlib import Path

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    results_root,
    run_artifact_dir,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model", default=None, choices=["nystrom_gp", "rff_gp"]
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _decode_diagonal_variance(encoder, coordinates, coefficient_variance):
    """Propagate independent coefficient marginals through a linear decoder."""
    import torch

    batch, _, _ = coordinates.shape
    n_basis = coefficient_variance.shape[-1]
    if batch != 1:
        raise ValueError("Variance export currently processes one function at a time.")
    zeros = torch.zeros(1, n_basis, device=coordinates.device, dtype=coordinates.dtype)
    residual = encoder(coordinates, zeros)
    eye = torch.eye(n_basis, device=coordinates.device, dtype=coordinates.dtype)
    expanded_coordinates = coordinates.expand(n_basis, -1, -1)
    basis_values = encoder(expanded_coordinates, eye) - residual
    return torch.einsum("knc,k->nc", basis_values.square(), coefficient_variance[0])


def _decode_covariance(encoder, coordinates, coefficient_covariance):
    """Propagate a full coefficient covariance through the linear decoder."""
    import torch

    batch, _, _ = coordinates.shape
    n_basis = coefficient_covariance.shape[-1]
    if batch != 1:
        raise ValueError("Variance export currently processes one function at a time.")
    zeros = torch.zeros(1, n_basis, device=coordinates.device, dtype=coordinates.dtype)
    residual = encoder(coordinates, zeros)
    eye = torch.eye(n_basis, device=coordinates.device, dtype=coordinates.dtype)
    basis_values = encoder(coordinates.expand(n_basis, -1, -1), eye) - residual
    return torch.einsum(
        "knc,kl,lnc->nc", basis_values, coefficient_covariance[0], basis_values
    )


def _plot(path: Path, target, mean, standard_deviation, spatial_dims, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.detach().float().cpu().numpy()
    mean = mean.detach().float().cpu().numpy()
    standard_deviation = standard_deviation.detach().float().cpu().numpy()
    error = np.abs(target - mean)
    if len(spatial_dims) == 2:
        shape = (*spatial_dims, -1)
        images = [item.reshape(shape)[..., 0] for item in (target, mean, standard_deviation, error)]
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
        for axis, image, label in zip(
            axes, images, ("target", "posterior mean", "posterior std", "absolute error")
        ):
            handle = axis.imshow(image, origin="lower", aspect="auto")
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    else:
        target, mean, standard_deviation, error = [item.reshape(-1) for item in (target, mean, standard_deviation, error)]
        axis_values = np.arange(target.size)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        axes[0].plot(axis_values, target, label="target")
        axes[0].plot(axis_values, mean, label="posterior mean")
        axes[0].fill_between(
            axis_values,
            mean - 1.96 * standard_deviation,
            mean + 1.96 * standard_deviation,
            alpha=0.25,
            label="95% interval",
        )
        axes[0].legend()
        axes[1].plot(axis_values, error, label="absolute error")
        axes[1].plot(axis_values, standard_deviation, label="posterior std")
        axes[1].legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    inverse_config = config.inverse_models
    model_name = args.model or inverse_config.models[0]
    if model_name not in {"nystrom_gp", "rff_gp"}:
        raise ValueError("Uncertainty export requires nystrom_gp or rff_gp.")
    artifact = inverse_config.artifact or model_name
    model_root = models_root(required=args.execute, override=args.models_dir)
    result_root = results_root(args.results_dir)
    model_dir = (
        model_artifact_dir(
            model_root, config.dataset.name, "inverse_models", artifact, args.seed
        )
        if model_root is not None
        else None
    )
    output_dir = (
        run_artifact_dir(
            result_root,
            config.dataset.name,
            "evaluation",
            "inverse_uncertainty",
            args.seed,
        )
        / artifact
        / args.split
    )
    if not args.execute:
        print(f"model_dir: {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"output_dir: {output_dir}")
        return

    import numpy as np
    import torch
    from safetensors.torch import load_file

    from inverse_neural_operator.data.overhaul import load_overhaul_dataset
    from inverse_neural_operator.function_encoders.artifacts import load_function_encoder
    from inverse_neural_operator.inverse.artifacts import require_inverse_model_artifact
    from inverse_neural_operator.inverse.build import create_inverse_model

    assert model_dir is not None
    require_inverse_model_artifact(model_dir)
    device = torch.device(args.device)
    dataset = load_overhaul_dataset(config, args.split)
    dataset_info = dataset.get_info()
    n_basis = config.function_encoders.basis.n_basis
    encoder_dir = model_artifact_dir(
        model_root,
        config.dataset.name,
        "function_encoders",
        inverse_config.function_encoder_artifact,
        args.seed,
    )
    input_encoder = load_function_encoder(
        encoder_dir, encoder_type="input", dataset_info=dataset_info, device=device
    )
    output_encoder = load_function_encoder(
        encoder_dir, encoder_type="output", dataset_info=dataset_info, device=device
    )
    model = create_inverse_model(
        model_name,
        input_size=n_basis,
        output_size=n_basis,
        hidden_sizes=inverse_config.hidden_sizes,
        nystrom_gp_config=inverse_config.nystrom_gp,
        rff_gp_config=inverse_config.rff_gp,
    ).to(device)
    model.load_state_dict(
        load_file(str(model_dir / "model.safetensors"), device=str(device))
    )
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(min(args.num_samples, len(dataset))):
        X, u, Y, s = [item.unsqueeze(0).to(device) for item in dataset[index]]
        with torch.no_grad():
            alpha, _ = input_encoder.compute_coefficients(X, u)
            beta, _ = output_encoder.compute_coefficients(Y, s)
            distribution = model.predict_distribution(beta)
            mean = input_encoder(X, distribution.mean)
            epistemic_variance = _decode_diagonal_variance(
                input_encoder, X, distribution.epistemic_variance
            )
            aleatoric_variance = _decode_diagonal_variance(
                input_encoder, X, distribution.aleatoric_variance
            )
            predictive_variance = _decode_covariance(
                input_encoder, X, model.predictive_covariance(beta)
            )
        standard_deviation = torch.sqrt(torch.clamp(predictive_variance, min=0))
        target = u[0]
        mean = mean[0]
        coverage = {}
        for level, multiplier in (("50", 0.67449), ("90", 1.64485), ("95", 1.95996)):
            coverage[level] = float(
                (torch.abs(target - mean) <= multiplier * standard_deviation)
                .float()
                .mean()
                .item()
            )
        np.savez_compressed(
            output_dir / f"sample_{index:03d}_uncertainty.npz",
            target=target.cpu().numpy(),
            mean=mean.cpu().numpy(),
            epistemic_standard_deviation=torch.sqrt(epistemic_variance).cpu().numpy(),
            aleatoric_standard_deviation=torch.sqrt(aleatoric_variance).cpu().numpy(),
            predictive_standard_deviation=standard_deviation.cpu().numpy(),
            coefficient_mean=distribution.mean[0].cpu().numpy(),
            coefficient_epistemic_variance=distribution.epistemic_variance[0].cpu().numpy(),
            coefficient_aleatoric_variance=distribution.aleatoric_variance[0].cpu().numpy(),
            coefficient_predictive_variance=distribution.predictive_variance[0].cpu().numpy(),
        )
        plot_path = output_dir / f"sample_{index:03d}_uncertainty.png"
        _plot(
            plot_path,
            target,
            mean,
            standard_deviation,
            dataset_info["input_spatial_dims"],
            f"{config.dataset.name} {model_name} uncertainty sample {index}",
        )
        records.append(
            {
                "sample": index,
                "field_mse": float(torch.mean((mean - target).square()).item()),
                "mean_predictive_std": float(standard_deviation.mean().item()),
                "coverage_50": coverage["50"],
                "coverage_90": coverage["90"],
                "coverage_95": coverage["95"],
                "data": str(output_dir / f"sample_{index:03d}_uncertainty.npz"),
                "plot": str(plot_path),
            }
        )
    summary = {
        key: sum(record[key] for record in records) / len(records)
        for key in ("field_mse", "mean_predictive_std", "coverage_50", "coverage_90", "coverage_95")
    }
    write_json(
        output_dir / "metrics.json",
        {"dataset": config.dataset.name, "artifact": artifact, "summary": summary, "samples": records},
    )
    print(f"Wrote uncertainty products to {output_dir}")


if __name__ == "__main__":
    main()
