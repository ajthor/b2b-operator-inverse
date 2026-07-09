"""Qualitative validation plots for saved reviewer baseline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from inverse_neural_operator.baselines.train import SUPPORTED_BASELINES
from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    results_root,
    run_artifact_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reviewer baseline predictions.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_BASELINES,
        default=None,
        help="Models to plot. Defaults to completed models from baselines.models.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for plotting evaluation.",
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


def _baseline_complete(path: Path) -> bool:
    return all(
        (path / filename).exists()
        for filename in ("model.safetensors", "config.yaml", "manifest.json", "metrics.json")
    )


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

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)

    candidate_models = args.models or config.baselines.models
    completed = []
    for model_name in candidate_models:
        model_dir = (
            model_artifact_dir(
                model_root,
                config.dataset.name,
                "baselines",
                model_name,
                args.seed,
            )
            if model_root is not None
            else None
        )
        if model_dir is not None and _baseline_complete(model_dir):
            completed.append((model_name, model_dir))

    if not args.execute:
        print("Dry run: baseline plots will not execute.")
        print(f"completed_models: {[name for name, _ in completed]}")
        for model_name, model_dir in completed:
            plot_dir = (
                run_artifact_dir(
                    result_root,
                    config.dataset.name,
                    "evaluation",
                    "baselines",
                    args.seed,
                )
                / model_name
                / args.split
                / "plots"
            )
            print(f"{model_name}: {model_dir} -> {plot_dir}")
        return
    if not completed:
        raise SystemExit("No completed baseline artifacts found for the requested models.")

    import torch
    from safetensors.torch import load_file

    from inverse_neural_operator.baselines.build import create_baseline_model
    from inverse_neural_operator.evaluation.metrics import mean_ssim, relative_l2

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(args.device)
    dataset = _load_dataset(config, args.split)
    dataset_info = dataset.get_info()
    sample_count = min(args.num_samples, len(dataset))
    samples = []
    for sample_index in range(sample_count):
        X, u, Y, s = dataset[sample_index]
        samples.append(
            (
                X.unsqueeze(0).to(device),
                u.unsqueeze(0).to(device),
                Y.unsqueeze(0).to(device),
                s.unsqueeze(0).to(device),
            )
        )

    all_summaries = {}
    with torch.no_grad():
        for model_name, model_dir in completed:
            model = create_baseline_model(model_name, dataset_info, config).to(device)
            model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=str(device)))
            model.eval()
            plot_dir = (
                run_artifact_dir(
                    result_root,
                    config.dataset.name,
                    "evaluation",
                    "baselines",
                    args.seed,
                )
                / model_name
                / args.split
                / "plots"
            )
            metric_values = {
                "input_mse": [],
                "input_relative_l2": [],
                "input_ssim": [],
                "forward_mse": [],
                "forward_relative_l2": [],
                "forward_ssim": [],
            }
            records = []
            for sample_index, (X, u, Y, s) in enumerate(samples):
                if model_name == "ifno":
                    raise ValueError("IFNO plotting is not implemented in this baseline plotter.")
                pred_u = model.predict_inverse(X, Y, s)
                input_ssim = mean_ssim(pred_u, u, dataset_info["input_spatial_dims"])
                record = {
                    "sample": sample_index,
                    "input_mse": float(torch.nn.functional.mse_loss(pred_u, u).item()),
                    "input_relative_l2": float(relative_l2(pred_u, u).item()),
                    "input_ssim": input_ssim,
                    "input_plot": str(plot_dir / f"sample_{sample_index:03d}_input.png"),
                }
                _plot_prediction(
                    plot_dir / f"sample_{sample_index:03d}_input.png",
                    u[0],
                    pred_u[0],
                    dataset_info["input_spatial_dims"],
                    f"{config.dataset.name} {model_name} input sample {sample_index}",
                )

                if hasattr(model, "forward_from_input"):
                    pred_s = model.forward_from_input(u, Y)
                    forward_ssim = mean_ssim(
                        pred_s,
                        s,
                        dataset_info["output_spatial_dims"],
                    )
                    record.update(
                        {
                            "forward_mse": float(torch.nn.functional.mse_loss(pred_s, s).item()),
                            "forward_relative_l2": float(relative_l2(pred_s, s).item()),
                            "forward_ssim": forward_ssim,
                            "forward_plot": str(
                                plot_dir / f"sample_{sample_index:03d}_forward.png"
                            ),
                        }
                    )
                    _plot_prediction(
                        plot_dir / f"sample_{sample_index:03d}_forward.png",
                        s[0],
                        pred_s[0],
                        dataset_info["output_spatial_dims"],
                        f"{config.dataset.name} {model_name} forward sample {sample_index}",
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
                    "seed": args.seed,
                    "split": args.split,
                    "num_samples": sample_count,
                    "summary": summary,
                    "records": records,
                },
            )
            print(f"Wrote baseline plots for {model_name} to {plot_dir}")

    summary_dir = run_artifact_dir(
        result_root,
        config.dataset.name,
        "evaluation",
        "baselines",
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
