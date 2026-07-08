"""Qualitative validation plots for saved inverse model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.inverse.build import SUPPORTED_INVERSE_MODELS
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    results_root,
    run_artifact_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot inverse model predictions.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_INVERSE_MODELS,
        default=None,
        help="Models to plot. Defaults to completed models from inverse_models.models.",
    )
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


def _artifact_complete(path: Path) -> bool:
    from inverse_neural_operator.inverse.artifacts import missing_inverse_model_files

    return not missing_inverse_model_files(path)


def _plot_prediction(path: Path, target, prediction, spatial_dims, title: str) -> None:
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
            ["target", "prediction", "absolute error"],
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
        axes[0].plot(target_flat, label="target")
        axes[0].plot(pred_flat, label="prediction")
        axes[0].legend()
        axes[0].set_title("values")
        axes[1].plot(error_flat)
        axes[1].set_title("absolute error")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    inverse_config = config.inverse_models

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")

    candidate_models = args.models or inverse_config.models
    completed = []
    for model_name in candidate_models:
        model_dir = (
            model_artifact_dir(
                model_root,
                config.dataset.name,
                "inverse_models",
                model_name,
                args.seed,
            )
            if model_root is not None
            else None
        )
        if model_dir is not None and _artifact_complete(model_dir):
            completed.append((model_name, model_dir))

    encoder_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "function_encoders",
            inverse_config.function_encoder_artifact,
            args.seed,
        )
        if model_root is not None
        else None
    )
    forward_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "forward_models",
            inverse_config.forward_model,
            args.seed,
        )
        if model_root is not None and inverse_config.forward_model
        else None
    )

    if not args.execute:
        print("Dry run: inverse plots will not execute.")
        print(f"completed_models: {[name for name, _ in completed]}")
        print(f"encoder_dir:      {encoder_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"forward_dir:      {forward_dir or '<none>'}")
        for model_name, model_dir in completed:
            plot_dir = (
                run_artifact_dir(
                    result_root,
                    config.dataset.name,
                    "evaluation",
                    "inverse_models",
                    args.seed,
                )
                / model_name
                / args.split
                / "plots"
            )
            print(f"{model_name}: {model_dir} -> {plot_dir}")
        return
    assert encoder_dir is not None

    import torch
    from safetensors.torch import load_file

    from inverse_neural_operator.evaluation.metrics import mean_ssim, relative_l2
    from inverse_neural_operator.forward.artifacts import (
        load_forward_model,
        require_forward_model_artifact,
    )
    from inverse_neural_operator.function_encoders.artifacts import (
        load_function_encoder,
        require_function_encoder_artifact,
    )
    from inverse_neural_operator.inverse.build import create_inverse_model

    if not completed:
        raise SystemExit("No completed inverse artifacts found for the requested models.")

    require_function_encoder_artifact(encoder_dir)
    if forward_dir is not None:
        require_forward_model_artifact(forward_dir)

    dataset = _load_dataset(config, args.split)
    dataset_info = dataset.get_info()
    device = torch.device(args.device)
    n_basis = config.function_encoders.basis.n_basis

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
    forward_model = None
    if forward_dir is not None:
        forward_model = load_forward_model(
            forward_dir,
            model_name=inverse_config.forward_model,
            input_size=n_basis,
            output_size=n_basis,
            hidden_sizes=config.forward_models.hidden_sizes,
            device=device,
        )

    all_summaries = {}
    sample_count = min(args.num_samples, len(dataset))
    with torch.no_grad():
        samples = [dataset[index] for index in range(sample_count)]
        encoded_samples = []
        for X, u, Y, s in samples:
            X = X.unsqueeze(0).to(device)
            u = u.unsqueeze(0).to(device)
            Y = Y.unsqueeze(0).to(device)
            s = s.unsqueeze(0).to(device)
            alpha, _ = input_encoder.compute_coefficients(X, u)
            beta, _ = output_encoder.compute_coefficients(Y, s)
            encoded_samples.append((X, u, Y, s, alpha, beta))

        for model_name, model_dir in completed:
            model = create_inverse_model(
                model_name,
                input_size=n_basis,
                output_size=n_basis,
                hidden_sizes=inverse_config.hidden_sizes,
                latent_size=inverse_config.latent_size,
                n_coupling_layers=inverse_config.n_coupling_layers,
                n_components=inverse_config.n_components,
            ).to(device)
            model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=str(device)))
            model.eval()

            plot_dir = (
                run_artifact_dir(
                    result_root,
                    config.dataset.name,
                    "evaluation",
                    "inverse_models",
                    args.seed,
                )
                / model_name
                / args.split
                / "plots"
            )
            records = []
            metric_values = {
                "coefficient_mse": [],
                "input_mse": [],
                "input_relative_l2": [],
                "input_ssim": [],
                "resimulation_mse": [],
                "resimulation_relative_l2": [],
                "resimulation_ssim": [],
            }
            for sample_index, (X, u, Y, s, alpha, beta) in enumerate(encoded_samples):
                alpha_pred = model(beta)
                u_pred = input_encoder(X, alpha_pred)
                input_ssim = mean_ssim(
                    u_pred,
                    u,
                    dataset_info["input_spatial_dims"],
                )
                record = {
                    "sample": sample_index,
                    "coefficient_mse": float(torch.nn.functional.mse_loss(alpha_pred, alpha).item()),
                    "input_mse": float(torch.nn.functional.mse_loss(u_pred, u).item()),
                    "input_relative_l2": float(relative_l2(u_pred, u).item()),
                    "input_ssim": input_ssim,
                    "input_plot": str(plot_dir / f"sample_{sample_index:03d}_input.png"),
                }
                _plot_prediction(
                    plot_dir / f"sample_{sample_index:03d}_input.png",
                    u[0],
                    u_pred[0],
                    dataset_info["input_spatial_dims"],
                    f"{config.dataset.name} {model_name} input sample {sample_index}",
                )

                if forward_model is not None:
                    beta_resim = forward_model(alpha_pred)
                    s_resim = output_encoder(Y, beta_resim)
                    resim_ssim = mean_ssim(
                        s_resim,
                        s,
                        dataset_info["output_spatial_dims"],
                    )
                    record.update(
                        {
                            "resimulation_mse": float(torch.nn.functional.mse_loss(s_resim, s).item()),
                            "resimulation_relative_l2": float(relative_l2(s_resim, s).item()),
                            "resimulation_ssim": resim_ssim,
                            "resimulation_plot": str(
                                plot_dir / f"sample_{sample_index:03d}_resimulation.png"
                            ),
                        }
                    )
                    _plot_prediction(
                        plot_dir / f"sample_{sample_index:03d}_resimulation.png",
                        s[0],
                        s_resim[0],
                        dataset_info["output_spatial_dims"],
                        f"{config.dataset.name} {model_name} resimulation sample {sample_index}",
                    )
                records.append(record)
                for key, value in record.items():
                    if key in metric_values and value is not None:
                        metric_values[key].append(value)

            summary = {key: _mean(values) for key, values in metric_values.items()}
            all_summaries[model_name] = summary
            write_json(
                plot_dir / "plot_metrics.json",
                {
                    "dataset": config.dataset.name,
                    "model": model_name,
                    "function_encoder_artifact": inverse_config.function_encoder_artifact,
                    "forward_model": inverse_config.forward_model,
                    "seed": args.seed,
                    "split": args.split,
                    "num_samples": sample_count,
                    "summary": summary,
                    "records": records,
                },
            )
            print(f"Wrote inverse plots for {model_name} to {plot_dir}")

    summary_dir = run_artifact_dir(
        result_root,
        config.dataset.name,
        "evaluation",
        "inverse_models",
        args.seed,
    )
    write_json(
        summary_dir / f"{args.split}_plot_summary.json",
        {
            "dataset": config.dataset.name,
            "seed": args.seed,
            "split": args.split,
            "num_samples": sample_count,
            "models": all_summaries,
        },
    )


if __name__ == "__main__":
    main()
