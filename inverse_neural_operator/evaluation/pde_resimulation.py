"""PDE-solver-backed inverse model validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.inverse.build import SUPPORTED_INVERSE_MODELS
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    results_root,
    run_artifact_dir,
    write_json,
)


DEFAULT_MAX_SAMPLES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inverse predictions with an independent PDE solver."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_INVERSE_MODELS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Evaluate the full split instead of the bounded default slice.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write qualitative input/resimulation plots for evaluated samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing solver inputs/outputs and metrics.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually evaluate. Without this flag the command only prints paths.",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def _load_dataset(config, split: str):
    from inverse_neural_operator.data.overhaul import load_overhaul_dataset

    return load_overhaul_dataset(config, split)


def _normalization_stats(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    return dataset_info.get("normalization_stats") or {}


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _as_array(value: Any, default: float) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _denormalize(values: np.ndarray, stats: Dict[str, Any], prefix: str) -> np.ndarray:
    mean = _as_array(stats.get(f"{prefix}_mean"), 0.0)
    std = _as_array(stats.get(f"{prefix}_std"), 1.0)
    return values * (std + 1e-8) + mean


def _physical_batch(
    dataset_name: str,
    X,
    Y,
    u_true,
    u_pred,
    s_true,
    dataset_info: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    stats = _normalization_stats(dataset_info)
    X_np = X.detach().float().cpu().numpy()
    Y_np = Y.detach().float().cpu().numpy()
    u_true_np = u_true.detach().float().cpu().numpy()
    u_pred_np = u_pred.detach().float().cpu().numpy()
    s_true_np = s_true.detach().float().cpu().numpy()

    if dataset_name in {"burgers", "burgers_1d", "darcy", "darcy_1d", "chladni", "chladni_2d"}:
        u_true_np = _denormalize(u_true_np, stats, "u")
        u_pred_np = _denormalize(u_pred_np, stats, "u")
        s_true_np = _denormalize(s_true_np, stats, "s")
    if dataset_name in {"chladni", "chladni_2d"}:
        xy_mean = stats.get("xy_mean")
        xy_std = stats.get("xy_std")
        if xy_mean is not None and xy_std is not None:
            X_np = X_np * (_as_array(xy_std, 1.0) + 1e-8) + _as_array(xy_mean, 0.0)
            Y_np = Y_np * (_as_array(xy_std, 1.0) + 1e-8) + _as_array(xy_mean, 0.0)
    return {
        "X": X_np,
        "Y": Y_np,
        "u_true": u_true_np,
        "u_pred": u_pred_np,
        "s_true": s_true_np,
    }


def _metric_payload(prediction: np.ndarray, target: np.ndarray, spatial_dims) -> Dict[str, Optional[float]]:
    import torch

    from inverse_neural_operator.evaluation.metrics import mae, mean_ssim, mse, relative_l2, rmse

    pred = torch.as_tensor(prediction, dtype=torch.float32).unsqueeze(0)
    true = torch.as_tensor(target, dtype=torch.float32).unsqueeze(0)
    return {
        "mae": float(mae(pred, true).item()),
        "mse": float(mse(pred, true).item()),
        "rmse": float(rmse(pred, true).item()),
        "relative_l2": float(relative_l2(pred, true).item()),
        "ssim": mean_ssim(pred, true, spatial_dims),
    }


def _sample_filename(sample_id: int) -> str:
    return f"sample_{sample_id:06d}.npz"


def _write_solver_input(
    path: Path,
    *,
    sample_id: int,
    arrays: Dict[str, np.ndarray],
    normalization_stats: Dict[str, Any],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_id, dtype=np.int64),
        X=arrays["X"],
        Y=arrays["Y"],
        u_pred=arrays["u_pred"],
        u_true=arrays["u_true"],
        s_true=arrays["s_true"],
        normalization_stats=np.asarray(json.dumps(normalization_stats)),
    )


def _load_solver_output(path: Path, sample_id: int) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing solver output: {path}")
    data = np.load(path, allow_pickle=False)
    actual_id = int(np.asarray(data["sample_id"]).item())
    if actual_id != sample_id:
        raise ValueError(f"Solver output {path} has sample_id={actual_id}, expected {sample_id}.")
    return {"Y": np.asarray(data["Y"]), "s_resim": np.asarray(data["s_resim"])}


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def _command_hash(command: Optional[str]) -> str:
    value = command or ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _format_command(command: str, *, input_dir: Path, output_dir: Path, manifest: Path) -> str:
    return command.format(
        input_dir=shlex.quote(str(input_dir)),
        output_dir=shlex.quote(str(output_dir)),
        manifest=shlex.quote(str(manifest)),
    )


def _run_external_solver(
    command_template: Optional[str],
    *,
    input_dir: Path,
    output_dir: Path,
    manifest: Path,
) -> None:
    if not command_template:
        raise RuntimeError(
            "pde_validation.command is required for external PDE validation backends."
        )
    command = _format_command(
        command_template,
        input_dir=input_dir,
        output_dir=output_dir,
        manifest=manifest,
    )
    subprocess.run(command, shell=True, check=True)


def _chladni_original_path() -> Path:
    from inverse_neural_operator.runtime.paths import repo_root

    return repo_root() / "data" / "ChladniData_original.npz"


@lru_cache(maxsize=1)
def _chladni_basis_data():
    data = np.load(_chladni_original_path())
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    L = float(data["L"])
    M = float(data["M"])
    omega = float(data["omega"])
    t_fixed = float(data["t_fixed"])
    gamma = float(data["gamma"])
    v = float(data["v"])
    n_range = int(data["n_range"])
    m_range = int(data["m_range"])

    from scipy.integrate import IntegrationWarning, quad

    mu_vals = np.arange(1, n_range + 1) * np.pi / L
    lam_vals = np.arange(1, m_range + 1) * np.pi / M
    cosX = np.cos(mu_vals[:, None] * x[None, :])
    cosY = np.cos(lam_vals[:, None] * y[None, :])
    center = np.cos(mu_vals[:, None] * (L / 2)) * np.cos(lam_vals[None, :] * (M / 2))
    beta_nm = np.sqrt(mu_vals[:, None] ** 2 + lam_vals[None, :] ** 2 + 3 * v**2 - gamma**4)
    time_int = np.zeros((n_range, m_range))
    for n in range(n_range):
        for m in range(m_range):
            beta = beta_nm[n, m]

            def integrand(tau):
                return (
                    np.sin(omega * (tau - t_fixed))
                    * np.exp(-(gamma**2) + v**2 * tau)
                    * np.sin(beta * tau)
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)
                time_int[n, m], _ = quad(integrand, 0, t_fixed, limit=200)
    mode_factor = (v**2 / beta_nm) * time_int * (4 / (L * M)) * center
    columns = []
    for n in range(n_range):
        for m in range(m_range):
            columns.append((cosX[n, :, None] * cosY[m, None, :]).reshape(-1))
    return x, y, cosX, cosY, mode_factor, np.stack(columns, axis=1), n_range, m_range


def _chladni_resimulate(u_pred: np.ndarray) -> np.ndarray:
    x, y, cosX, cosY, mode_factor, design, n_range, m_range = _chladni_basis_data()
    force = np.asarray(u_pred).reshape(len(x), len(y))
    alpha_flat, *_ = np.linalg.lstsq(design, force.reshape(-1), rcond=None)
    alpha = alpha_flat.reshape(n_range, m_range)
    z = np.einsum("nm,ni,mj->ij", alpha * mode_factor, cosX, cosY)
    return z.reshape(-1, 1).astype(np.float32)


def _run_chladni_solver(input_paths: Iterable[Path], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in input_paths:
        data = np.load(input_path, allow_pickle=False)
        sample_id = int(np.asarray(data["sample_id"]).item())
        output_path = output_dir / _sample_filename(sample_id)
        if output_path.exists() and not overwrite:
            continue
        s_resim = _chladni_resimulate(np.asarray(data["u_pred"]))
        np.savez_compressed(
            output_path,
            sample_id=np.asarray(sample_id, dtype=np.int64),
            Y=np.asarray(data["Y"]),
            s_resim=s_resim,
        )


def _plot_sample(path: Path, target: np.ndarray, prediction: np.ndarray, spatial_dims, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    target_flat = target.reshape(-1)
    pred_flat = prediction.reshape(-1)
    err_flat = np.abs(pred_flat - target_flat)
    if spatial_dims and len(spatial_dims) == 2:
        target_plot = target_flat.reshape(*spatial_dims)
        pred_plot = pred_flat.reshape(*spatial_dims)
        err_plot = err_flat.reshape(*spatial_dims)
        vmin = min(float(target_plot.min()), float(pred_plot.min()))
        vmax = max(float(target_plot.max()), float(pred_plot.max()))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for axis, image, label in zip(
            axes,
            [target_plot, pred_plot, err_plot],
            ["target", "prediction", "absolute error"],
        ):
            kwargs = {} if label == "absolute error" else {"vmin": vmin, "vmax": vmax}
            im = axis.imshow(image, aspect="auto", origin="lower", **kwargs)
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
        axes[1].plot(err_flat)
        axes[1].set_title("absolute error")
    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _average_records(records: List[Dict[str, Any]], prefix: str, key: str) -> Optional[float]:
    values = [record[f"{prefix}_{key}"] for record in records if record.get(f"{prefix}_{key}") is not None]
    if not values:
        return None
    return float(np.mean(values))


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    inverse_config = config.inverse_models
    pde_config = config.pde_validation

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)

    inverse_dir = (
        model_artifact_dir(model_root, config.dataset.name, "inverse_models", args.model, args.seed)
        if model_root is not None
        else None
    )
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
    eval_dir = (
        run_artifact_dir(
            result_root,
            config.dataset.name,
            "evaluation",
            "pde_resimulation",
            args.seed,
        )
        / args.model
        / args.split
    )
    input_dir = eval_dir / "inputs"
    output_dir = eval_dir / "solver_outputs"
    plot_dir = eval_dir / "plots"
    manifest_path = eval_dir / "manifest.json"
    max_samples = (
        None
        if args.full_split
        else args.max_samples
        if args.max_samples is not None
        else pde_config.max_samples
        if pde_config.max_samples is not None
        else DEFAULT_MAX_SAMPLES
    )
    overwrite = args.overwrite or pde_config.overwrite
    backend = pde_config.backend
    if config.dataset.name in {"chladni", "chladni_2d"} and backend == "external":
        backend = "chladni_analytic"

    if not args.execute:
        print("Dry run: PDE solver validation will not execute.")
        print(f"inverse_dir: {inverse_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"encoder_dir: {encoder_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"eval_dir:    {eval_dir}")
        print(f"input_dir:   {input_dir}")
        print(f"output_dir:  {output_dir}")
        print(f"manifest:    {manifest_path}")
        print(f"backend:     {backend}")
        print(f"command:     {pde_config.command or '<none>'}")
        print(f"max_samples: {max_samples if max_samples is not None else '<full split>'}")
        return

    assert inverse_dir is not None
    assert encoder_dir is not None

    from safetensors.torch import load_file
    import torch

    from inverse_neural_operator.function_encoders.artifacts import (
        load_function_encoder,
        require_function_encoder_artifact,
    )
    from inverse_neural_operator.inverse.artifacts import require_inverse_model_artifact
    from inverse_neural_operator.inverse.build import create_inverse_model

    try:
        require_inverse_model_artifact(inverse_dir)
        require_function_encoder_artifact(encoder_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    device = torch.device(config.runtime.device if config.runtime.device == "cuda" and torch.cuda.is_available() else "cpu")
    start_time = time.time()
    dataset = _load_dataset(config, args.split)
    dataset_info = dataset.get_info()
    n_basis = config.function_encoders.basis.n_basis
    sample_count = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    model = create_inverse_model(
        args.model,
        input_size=n_basis,
        output_size=n_basis,
        hidden_sizes=inverse_config.hidden_sizes,
        latent_size=inverse_config.latent_size,
        n_coupling_layers=inverse_config.n_coupling_layers,
        n_components=inverse_config.n_components,
    ).to(device)
    model.load_state_dict(load_file(str(inverse_dir / "model.safetensors"), device=str(device)))
    model.eval()
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

    eval_dir.mkdir(parents=True, exist_ok=True)
    input_paths = []
    records: List[Dict[str, Any]] = []
    manifest_samples = []
    with torch.no_grad():
        for sample_id in range(sample_count):
            X, u, Y, s = dataset[sample_id]
            Xb = X.unsqueeze(0).to(device)
            ub = u.unsqueeze(0).to(device)
            Yb = Y.unsqueeze(0).to(device)
            sb = s.unsqueeze(0).to(device)
            beta, _ = output_encoder.compute_coefficients(Yb, sb)
            alpha_pred = model(beta)
            u_pred = input_encoder(Xb, alpha_pred)
            arrays = _physical_batch(
                config.dataset.name,
                X,
                Y,
                u,
                u_pred[0],
                s,
                dataset_info,
            )
            input_path = input_dir / _sample_filename(sample_id)
            output_path = output_dir / _sample_filename(sample_id)
            _write_solver_input(
                input_path,
                sample_id=sample_id,
                arrays=arrays,
                normalization_stats=_normalization_stats(dataset_info),
                overwrite=overwrite,
            )
            input_paths.append(input_path)
            manifest_samples.append(
                {
                    "sample_id": sample_id,
                    "input": str(input_path),
                    "output": str(output_path),
                }
            )

    manifest = {
        "dataset": config.dataset.name,
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "num_samples": sample_count,
        "backend": backend,
        "command": pde_config.command,
        "samples": manifest_samples,
    }
    _write_manifest(manifest_path, manifest)

    expected_outputs = [output_dir / _sample_filename(i) for i in range(sample_count)]
    outputs_missing = any(not path.exists() for path in expected_outputs)
    if backend == "chladni_analytic":
        _run_chladni_solver(input_paths, output_dir, overwrite=overwrite)
    elif outputs_missing or overwrite:
        _run_external_solver(
            pde_config.command,
            input_dir=input_dir,
            output_dir=output_dir,
            manifest=manifest_path,
        )

    for sample_id in range(sample_count):
        input_data = np.load(input_dir / _sample_filename(sample_id), allow_pickle=False)
        solver_data = _load_solver_output(output_dir / _sample_filename(sample_id), sample_id)
        Y_true = np.asarray(input_data["Y"])
        Y_resim = np.asarray(solver_data["Y"])
        s_true = np.asarray(input_data["s_true"])
        s_resim = solver_data["s_resim"]
        if Y_resim.shape != Y_true.shape:
            raise ValueError(
                f"Solver coordinate shape mismatch for sample {sample_id}: "
                f"{Y_resim.shape} != {Y_true.shape}"
            )
        if s_resim.shape != s_true.shape:
            raise ValueError(
                f"Solver output shape mismatch for sample {sample_id}: "
                f"{s_resim.shape} != {s_true.shape}"
            )
        input_metrics = _metric_payload(
            np.asarray(input_data["u_pred"]),
            np.asarray(input_data["u_true"]),
            dataset_info["input_spatial_dims"],
        )
        resim_metrics = _metric_payload(
            s_resim,
            s_true,
            dataset_info["output_spatial_dims"],
        )
        record = {"sample_id": sample_id}
        for key, value in input_metrics.items():
            record[f"input_{key}"] = value
        for key, value in resim_metrics.items():
            record[f"resimulation_{key}"] = value
        records.append(record)
        if args.plots:
            _plot_sample(
                plot_dir / f"sample_{sample_id:06d}_input.png",
                np.asarray(input_data["u_true"]),
                np.asarray(input_data["u_pred"]),
                dataset_info["input_spatial_dims"],
                f"{config.dataset.name} {args.model} input sample {sample_id}",
            )
            _plot_sample(
                plot_dir / f"sample_{sample_id:06d}_resimulation.png",
                s_true,
                s_resim,
                dataset_info["output_spatial_dims"],
                f"{config.dataset.name} {args.model} resimulation sample {sample_id}",
            )

    aggregate = {
        "num_samples": sample_count,
        "split": args.split,
        "dataset": config.dataset.name,
        "model": args.model,
        "solver_backend": backend,
        "solver_command_hash": _command_hash(pde_config.command),
        "elapsed_seconds": time.time() - start_time,
        "inverse_dir": str(inverse_dir),
        "function_encoder_artifact": inverse_config.function_encoder_artifact,
    }
    for prefix in ("input", "resimulation"):
        for key in ("mae", "mse", "rmse", "relative_l2", "ssim"):
            aggregate[f"{prefix}_{key}"] = _average_records(records, prefix, key)
    write_json(eval_dir / "metrics.json", aggregate)
    _write_jsonl(eval_dir / "records.jsonl", records)
    print(f"Wrote PDE resimulation metrics to {eval_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
