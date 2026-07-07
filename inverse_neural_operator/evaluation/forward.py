"""Qualitative validation plots for saved forward model artifacts."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot forward model predictions.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=["b2b_linear", "b2b_nonlinear"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for plotting evaluation. Defaults to CPU to avoid occupying GPUs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write plots. Without this flag the command only prints paths.",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def _load_dataset(config, split: str):
    from inverse_neural_operator.data.overhaul import load_overhaul_dataset

    return load_overhaul_dataset(config, split)


def _plot_forward_prediction(
    path: Path,
    target,
    prediction,
    spatial_dims,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    target_flat = target.detach().float().cpu().reshape(-1).numpy()
    pred_flat = prediction.detach().float().cpu().reshape(-1).numpy()
    error_flat = np.abs(pred_flat - target_flat)

    if spatial_dims and len(spatial_dims) == 2:
        channels = max(1, target_flat.size // int(np.prod(spatial_dims)))
        target_plot = target_flat.reshape(*spatial_dims, channels)[..., 0]
        pred_plot = pred_flat.reshape(*spatial_dims, channels)[..., 0]
        error_plot = error_flat.reshape(*spatial_dims, channels)[..., 0]
        value_min = min(float(target_plot.min()), float(pred_plot.min()))
        value_max = max(float(target_plot.max()), float(pred_plot.max()))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for axis, image, label in zip(
            axes,
            [target_plot, pred_plot, error_plot],
            ["target output", "predicted output", "absolute error"],
        ):
            if label == "absolute error":
                im = axis.imshow(image, aspect="auto", origin="lower")
            else:
                im = axis.imshow(
                    image,
                    aspect="auto",
                    origin="lower",
                    vmin=value_min,
                    vmax=value_max,
                )
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axes[0].plot(target_flat, label="target output")
        axes[0].plot(pred_flat, label="predicted output")
        axes[0].legend()
        axes[0].set_title("values")
        axes[1].plot(error_flat)
        axes[1].set_title("absolute error")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    forward_config = config.forward_models

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)
    forward_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "forward_models",
            args.model,
            args.seed,
        )
        if model_root is not None
        else None
    )
    encoder_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "function_encoders",
            forward_config.function_encoder_artifact,
            args.seed,
        )
        if model_root is not None
        else None
    )
    plot_dir = (
        run_artifact_dir(
            result_root,
            config.dataset.name,
            "evaluation",
            "forward_models",
            args.seed,
        )
        / args.model
        / args.split
    )

    if not args.execute:
        print("Dry run: forward model validation plots will not execute.")
        print(f"forward_dir: {forward_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"encoder_dir: {encoder_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"plot_dir:    {plot_dir}")
        return
    assert forward_dir is not None
    assert encoder_dir is not None

    import torch

    from inverse_neural_operator.evaluation.metrics import mean_ssim, relative_l2
    from inverse_neural_operator.forward.artifacts import load_forward_model
    from inverse_neural_operator.function_encoders.artifacts import (
        load_function_encoder,
        require_function_encoder_artifact,
    )

    require_function_encoder_artifact(encoder_dir)
    dataset = _load_dataset(config, args.split)
    dataset_info = dataset.get_info()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(args.device)

    input_encoder = load_function_encoder(
        encoder_dir,
        encoder_type="input",
        dataset_info=dataset_info,
        device=device,
    )
    output_encoder = load_function_encoder(
        encoder_dir,
        encoder_type="output",
        dataset_info=dataset_info,
        device=device,
    )
    model = load_forward_model(
        forward_dir,
        model_name=args.model,
        input_size=config.function_encoders.basis.n_basis,
        output_size=config.function_encoders.basis.n_basis,
        hidden_sizes=forward_config.hidden_sizes,
        device=device,
    )

    records = []
    sample_count = min(args.num_samples, len(dataset))
    totals = {
        "coefficient_mse": 0.0,
        "output_mse": 0.0,
        "output_relative_l2": 0.0,
    }
    ssim_values = []
    spatial_dims = dataset_info.get("output_spatial_dims")
    with torch.no_grad():
        for sample_index in range(sample_count):
            X, u, Y, s = dataset[sample_index]
            X = X.unsqueeze(0).to(device)
            u = u.unsqueeze(0).to(device)
            Y = Y.unsqueeze(0).to(device)
            s = s.unsqueeze(0).to(device)

            alpha, _ = input_encoder.compute_coefficients(X, u)
            beta, _ = output_encoder.compute_coefficients(Y, s)
            beta_pred = model(alpha)
            s_pred = output_encoder(Y, beta_pred)

            coefficient_mse = torch.nn.functional.mse_loss(beta_pred, beta)
            output_mse = torch.nn.functional.mse_loss(s_pred, s)
            output_relative_l2 = relative_l2(s_pred, s)
            output_ssim = mean_ssim(s_pred, s, spatial_dims)

            path = plot_dir / f"sample_{sample_index:03d}_forward.png"
            _plot_forward_prediction(
                path,
                s[0],
                s_pred[0],
                spatial_dims,
                f"{config.dataset.name} {args.model} sample {sample_index}",
            )
            record = {
                "sample": sample_index,
                "coefficient_mse": float(coefficient_mse.item()),
                "output_mse": float(output_mse.item()),
                "output_relative_l2": float(output_relative_l2.item()),
                "output_ssim": output_ssim,
                "plot": str(path),
            }
            records.append(record)
            for key in totals:
                totals[key] += record[key]
            if output_ssim is not None:
                ssim_values.append(output_ssim)

    summary = {
        key: value / max(1, sample_count)
        for key, value in totals.items()
    }
    summary["output_ssim"] = (
        sum(ssim_values) / len(ssim_values) if ssim_values else None
    )
    write_json(
        plot_dir / "metrics.json",
        {
            "dataset": config.dataset.name,
            "model": args.model,
            "function_encoder_artifact": forward_config.function_encoder_artifact,
            "seed": args.seed,
            "split": args.split,
            "num_samples": sample_count,
            "summary": summary,
            "records": records,
        },
    )
    print(f"Wrote forward model validation plots to {plot_dir}")


if __name__ == "__main__":
    main()
