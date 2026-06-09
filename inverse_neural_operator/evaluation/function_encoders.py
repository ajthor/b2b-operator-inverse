"""Qualitative validation plots for saved function encoder artifacts."""

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
    parser = argparse.ArgumentParser(description="Plot function encoder reconstructions.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=3)
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


def _relative_l2(prediction, target):
    from inverse_neural_operator.evaluation.metrics import relative_l2

    return relative_l2(prediction, target)


def _plot_reconstruction(path: Path, true_values, pred_values, spatial_dims, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    true_flat = true_values.detach().float().cpu().reshape(-1).numpy()
    pred_flat = pred_values.detach().float().cpu().reshape(-1).numpy()
    error_flat = np.abs(pred_flat - true_flat)

    if spatial_dims and len(spatial_dims) == 2:
        true_plot = true_flat.reshape(*spatial_dims)
        pred_plot = pred_flat.reshape(*spatial_dims)
        error_plot = error_flat.reshape(*spatial_dims)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for axis, image, label in zip(
            axes,
            [true_plot, pred_plot, error_plot],
            ["target", "reconstruction", "absolute error"],
        ):
            im = axis.imshow(image, aspect="auto", origin="lower")
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axes[0].plot(true_flat, label="target")
        axes[0].plot(pred_flat, label="reconstruction")
        axes[0].legend()
        axes[0].set_title("values")
        axes[1].plot(error_flat)
        axes[1].set_title("absolute error")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _encoder_payload(sample, encoder_type: str, dataset_info):
    X, u, Y, s = sample
    if encoder_type == "input":
        return X, u, dataset_info.get("input_spatial_dims")
    if encoder_type == "output":
        return Y, s, dataset_info.get("output_spatial_dims")
    raise ValueError(f"Unknown encoder_type: {encoder_type}")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    artifact = args.artifact or config.function_encoders.artifact

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)
    artifact_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "function_encoders",
            artifact,
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
            "function_encoders",
            args.seed,
        )
        / artifact
        / args.split
    )

    if not args.execute:
        print("Dry run: function encoder validation plots will not execute.")
        print(f"artifact_dir: {artifact_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"plot_dir:     {plot_dir}")
        return
    assert artifact_dir is not None

    import torch

    from inverse_neural_operator.function_encoders.artifacts import (
        load_function_encoder,
        require_function_encoder_artifact,
    )

    require_function_encoder_artifact(artifact_dir)
    dataset = _load_dataset(config, args.split)
    dataset_info = dataset.get_info()
    device = torch.device(
        "cuda"
        if config.runtime.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    encoders = {
        "input": load_function_encoder(
            artifact_dir,
            encoder_type="input",
            dataset_info=dataset_info,
            device=device,
        ),
        "output": load_function_encoder(
            artifact_dir,
            encoder_type="output",
            dataset_info=dataset_info,
            device=device,
        ),
    }

    records = []
    sample_count = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for sample_index in range(sample_count):
            sample = dataset[sample_index]
            for encoder_type, encoder in encoders.items():
                xs, ys, spatial_dims = _encoder_payload(sample, encoder_type, dataset_info)
                xs = xs.unsqueeze(0).to(device)
                ys = ys.unsqueeze(0).to(device)
                coefficients, _ = encoder.compute_coefficients(xs, ys)
                pred = encoder(xs, coefficients)
                mse = torch.nn.functional.mse_loss(pred, ys)
                rel_l2 = _relative_l2(pred, ys)
                path = plot_dir / f"sample_{sample_index:03d}_{encoder_type}.png"
                _plot_reconstruction(
                    path,
                    ys[0],
                    pred[0],
                    spatial_dims,
                    f"{config.dataset.name} {encoder_type} sample {sample_index}",
                )
                records.append(
                    {
                        "sample": sample_index,
                        "encoder_type": encoder_type,
                        "mse": float(mse.item()),
                        "relative_l2": float(rel_l2.item()),
                        "plot": str(path),
                    }
                )
    write_json(
        plot_dir / "metrics.json",
        {
            "dataset": config.dataset.name,
            "artifact": artifact,
            "seed": args.seed,
            "split": args.split,
            "num_samples": sample_count,
            "records": records,
        },
    )
    print(f"Wrote function encoder validation plots to {plot_dir}")


if __name__ == "__main__":
    main()
