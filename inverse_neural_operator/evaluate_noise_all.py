"""
Evaluate inverse models under measurement noise for multiple datasets/models.

This script mirrors the structure of evaluate_models.py: it discovers trained
models in the results directory, loads the shared function encoders, evaluates
each model across a grid of additive Gaussian noise levels applied to the
observed measurements, and writes per-model JSON/CSV reports plus a
dataset-level CSV + summary JSON that plotting scripts can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from data.load_dataset import load_dataset
from models.load_model import load_models
from b2b.load_model import load_forward_model
from plots.utils.ifno_utils import load_ifno_model


DEFAULT_DATASETS = [
    "burgers_1d",
    "darcy_1d",
    "elastic_plate",
    "wave_scattering",
    "fwi",
    "chladni_2d",
]
DEFAULT_MODELS = [
    "linear_inverse",
    "linear",
    "nonlinear",
    "variational_autoencoder",
    "inn_additive",
    "cinn_additive",
    "cinn_additive_probabilistic",
    "inn_affine",
    "cinn_affine",
    "conditional_realnvp",
    "mixture_density_network",
    "ifno",
]
DEFAULT_SEEDS = [1]
DEFAULT_NOISE_LEVELS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


@dataclass
class NoiseStats:
    inverse_error_sq: float = 0.0
    inverse_target_sq: float = 0.0
    forward_error_sq: float = 0.0
    forward_target_sq: float = 0.0
    coeff_error_sq: float = 0.0
    coeff_target_sq: float = 0.0
    resim_coeff_error_sq: float = 0.0
    resim_coeff_target_sq: float = 0.0

    def _accumulate(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[float, float]:
        error_sq = torch.sum((pred - target) ** 2).item()
        target_sq = torch.sum(target**2).item()
        return error_sq, target_sq

    def update_inverse(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        err, tgt = self._accumulate(pred, target)
        self.inverse_error_sq += err
        self.inverse_target_sq += tgt

    def update_forward(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        err, tgt = self._accumulate(pred, target)
        self.forward_error_sq += err
        self.forward_target_sq += tgt

    def update_coeff(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        err, tgt = self._accumulate(pred, target)
        self.coeff_error_sq += err
        self.coeff_target_sq += tgt

    def update_resim_coeff(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        err, tgt = self._accumulate(pred, target)
        self.resim_coeff_error_sq += err
        self.resim_coeff_target_sq += tgt

    def to_metrics(self, noise_std: float) -> Dict[str, Optional[float]]:
        def rel(error_sq: float, target_sq: float) -> Optional[float]:
            if target_sq <= 0.0 or error_sq < 0.0:
                return None
            return float(np.sqrt(error_sq / target_sq))

        return {
            "noise_std": noise_std,
            "inverse_rel_l2": rel(self.inverse_error_sq, self.inverse_target_sq),
            "resim_pred_rel_l2": rel(
                self.forward_error_sq, self.forward_target_sq
            ),
            "coeff_rel_l2": rel(self.coeff_error_sq, self.coeff_target_sq),
            "resim_coeff_rel_l2": rel(
                self.resim_coeff_error_sq, self.resim_coeff_target_sq
            ),
        }


def _prepare_sample(sample: Iterable[torch.Tensor], device: torch.device):
    tensors = [tensor.to(device) for tensor in sample]
    batched = [tensor.unsqueeze(0) for tensor in tensors]
    return tensors, batched


def format_noise_value(std: float) -> str:
    trimmed = f"{std:.6f}".rstrip("0").rstrip(".")
    return trimmed or format(std, "g")


def evaluate_measurement_noise(
    model,
    evaluate_fn,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    dataset,
    noise_levels: List[float],
    device: torch.device,
    noise_draws: int,
) -> Dict[str, Dict[str, float]]:
    stats_per_noise = {level: NoiseStats() for level in noise_levels}

    model.eval()
    if forward_model is not None:
        forward_model.eval()

    with torch.no_grad():
        for sample in dataset:
            (X, u_true, Y, s_true), (X_b, u_b, Y_b, s_b) = _prepare_sample(
                sample, device
            )

            eval_out = evaluate_fn(
                model,
                (X_b, u_b, Y_b, s_b),
                input_function_encoder,
                output_function_encoder,
            )
            if isinstance(eval_out, (tuple, list)):
                u_pred = eval_out[0]
                alpha_pred_clean = eval_out[1] if len(eval_out) > 1 else None
            else:
                u_pred = eval_out
                alpha_pred_clean = None

            if alpha_pred_clean is not None:
                alpha_pred_clean = alpha_pred_clean.squeeze(0)
            alpha_true, _ = input_function_encoder.compute_coefficients(X_b, u_b)
            alpha_true = alpha_true.squeeze(0)
            if forward_model is not None:
                beta_true, _ = output_function_encoder.compute_coefficients(Y_b, s_b)
                beta_true = beta_true.squeeze(0)
            else:
                beta_true = None

            for noise_std, stats in stats_per_noise.items():
                draws = 1 if noise_std == 0.0 else noise_draws
                for _ in range(draws):
                    noise_tensor = (
                        torch.zeros_like(s_b)
                        if noise_std == 0.0
                        else noise_std * torch.randn_like(s_b)
                    )
                    noisy_batch = (X_b, u_b, Y_b, s_b + noise_tensor)
                    inv_eval = evaluate_fn(
                        model, noisy_batch, input_function_encoder, output_function_encoder
                    )
                    if isinstance(inv_eval, (tuple, list)):
                        u_pred_noise = inv_eval[0]
                        alpha_pred_noise = inv_eval[1] if len(inv_eval) > 1 else None
                    else:
                        u_pred_noise = inv_eval
                        alpha_pred_noise = None

                    stats.update_inverse(u_pred_noise.squeeze(0), u_true)
                    if alpha_pred_noise is not None:
                        stats.update_coeff(alpha_pred_noise.squeeze(0), alpha_true)

                    if (
                        forward_model is not None
                        and isinstance(alpha_pred_noise, torch.Tensor)
                    ):
                        beta_resim = forward_model(alpha_pred_noise)
                        if beta_true is not None:
                            stats.update_resim_coeff(
                                beta_resim.squeeze(0), beta_true
                            )
                        s_resim = output_function_encoder(Y_b, beta_resim)
                        stats.update_forward(s_resim.squeeze(0), s_true)

    return {
        (format_noise_value(level) if level > 0 else "0"): stats.to_metrics(level)
        for level, stats in stats_per_noise.items()
    }


def collect_dataset_rows(
    dataset_name: str,
    model_name: str,
    seed: int,
    metrics: Dict[str, Dict[str, float]],
) -> List[Dict[str, float]]:
    rows = []
    for label, data in metrics.items():
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "noise_label": label,
                "noise_std": data.get("noise_std"),
                "inverse_rel_l2": data.get("inverse_rel_l2"),
                "coeff_rel_l2": data.get("coeff_rel_l2"),
                "resim_coeff_rel_l2": data.get("resim_coeff_rel_l2"),
                "resim_pred_rel_l2": data.get("resim_pred_rel_l2"),
            }
        )
    return rows


METRIC_KEYS = [
    "inverse_rel_l2",
    "coeff_rel_l2",
    "resim_coeff_rel_l2",
    "resim_pred_rel_l2",
]


def aggregate_model_metrics(
    entries: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Compute mean/std/min/max across seeds for each noise label."""
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for entry in entries:
        label = entry["noise_label"]
        grouped.setdefault(label, {}).setdefault("noise_std", []).append(
            entry["noise_std"]
        )
        for key in METRIC_KEYS:
            grouped[label].setdefault(key, []).append(entry.get(key))

    summary: Dict[str, Dict[str, float]] = {}
    for label, metrics in grouped.items():
        summary[label] = {
            "noise_std": float(np.mean(metrics["noise_std"]))
            if metrics["noise_std"]
            else 0.0,
            "num_runs": len(metrics["noise_std"]),
        }
        for key in METRIC_KEYS:
            values = [v for v in metrics[key] if v is not None]
            if not values:
                continue
            summary[label][f"{key}_mean"] = float(np.mean(values))
            summary[label][f"{key}_std"] = float(np.std(values))
            summary[label][f"{key}_min"] = float(np.min(values))
            summary[label][f"{key}_max"] = float(np.max(values))
    return summary


def write_model_reports(
    model_dir: Path,
    metrics: Dict[str, Dict[str, float]],
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    json_path = model_dir / "measurement_noise_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def write_dataset_csv(dataset_dir: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    csv_path = dataset_dir / "measurement_noise.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "model",
            "seed",
            "noise_label",
            "noise_std",
            "inverse_rel_l2",
            "coeff_rel_l2",
            "resim_coeff_rel_l2",
            "resim_pred_rel_l2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Saved dataset CSV → {csv_path}")


def write_dataset_summary(
    dataset_dir: Path,
    dataset_name: str,
    model_entries: Dict[str, List[Dict[str, float]]],
) -> None:
    summary = {"dataset": dataset_name, "models": {}}
    for model_name, entries in model_entries.items():
        summary["models"][model_name] = aggregate_model_metrics(entries)

    summary_path = dataset_dir / "measurement_noise_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  ✓ Saved dataset summary → {summary_path}")

    write_dataset_text_summary(dataset_dir, summary)


def write_dataset_text_summary(dataset_dir: Path, summary: Dict) -> None:
    lines: List[str] = []
    dataset_name = summary.get("dataset", "unknown")
    lines.append(f"Measurement Noise Summary — {dataset_name}")
    lines.append("")
    models = summary.get("models", {})
    if not models:
        lines.append("No models evaluated.")
    else:
        for model_name in sorted(models.keys()):
            lines.append(model_name)
            entries = models[model_name]
            if not entries:
                lines.append("  (no data)")
                lines.append("")
                continue
            header = (
                "  noise_std  runs  resim_pred_rel_l2_mean  resim_coeff_rel_l2_mean"
                "  inverse_rel_l2_mean  coeff_rel_l2_mean"
            )
            lines.append(header)
            for label, data in sorted(
                entries.items(), key=lambda item: item[1].get("noise_std", 0.0)
            ):
                resim_pred = data.get("resim_pred_rel_l2_mean")
                resim_coeff = data.get("resim_coeff_rel_l2_mean")
                inverse_mean = data.get("inverse_rel_l2_mean")
                coeff_mean = data.get("coeff_rel_l2_mean")

                def fmt(val):
                    return f"{val:.6f}" if val is not None else "NA"

                line = (
                    f"  {data.get('noise_std', 0.0):>8.4f}  "
                    f"{int(data.get('num_runs', 0)):>4d}  "
                    f"{fmt(resim_pred):>24}  "
                    f"{fmt(resim_coeff):>25}  "
                    f"{fmt(inverse_mean):>20}  "
                    f"{fmt(coeff_mean):>18}"
                )
                lines.append(line)
            lines.append("")

    txt_path = dataset_dir / "measurement_noise_summary.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"  ✓ Saved dataset text summary → {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inverse models under measurement noise and store metrics."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.environ.get("B2B_RESULTS_DIR", "results"),
        help="Base directory containing models/ and runs/ (default: results)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate (defaults to all supported datasets)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to evaluate (default: 1)",
    )
    parser.add_argument(
        "--noise_levels",
        nargs="*",
        type=float,
        default=DEFAULT_NOISE_LEVELS,
        help="Noise std values applied to observed measurements",
    )
    parser.add_argument(
        "--noise_draws",
        type=int,
        default=10,
        help="Number of Gaussian draws per sample for nonzero noise levels (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use for evaluation (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    models_root = base_dir / "models"
    runs_root = base_dir / "runs"
    noise_levels = sorted(set(args.noise_levels))
    noise_draws = max(1, args.noise_draws)
    device = torch.device(args.device)

    if not models_root.exists():
        raise FileNotFoundError(f"Models directory not found at {models_root}")

    for dataset in args.datasets:
        dataset_rows: List[Dict[str, float]] = []
        model_entries: Dict[str, List[Dict[str, float]]] = {}
        dataset_runs_dir = runs_root / dataset
        os.makedirs(dataset_runs_dir, exist_ok=True)

        print(f"═══════════════════════════════════════════════════════════════")
        print(f"Dataset: {dataset}")
        print(f"Models: {', '.join(args.models)}")
        print(f"Seeds: {', '.join(str(s) for s in args.seeds)}")
        print(f"Noise levels: {noise_levels}")
        print(f"───────────────────────────────────────────────────────────────")

        for model_name in args.models:
            for seed in args.seeds:
                model_dir = models_root / dataset / model_name / f"seed_{seed}"
                if not model_dir.exists():
                    print(f"  ⚠ Skipping {dataset}/{model_name}/seed_{seed} (not found)")
                    continue

                params_path = model_dir / "params.pth"
                if not params_path.exists():
                    print(
                        f"  ⚠ Skipping {dataset}/{model_name}/seed_{seed}: missing params.pth"
                    )
                    continue

                params = torch.load(params_path, weights_only=False)
                test_dataset, dataset_info = load_dataset(
                    params.dataset,
                    params,
                    device,
                    split="test",
                    return_info=True,
                )

                if model_name == "ifno":
                    checkpoint_path = model_dir / "ifno_model.safetensors"
                    if not checkpoint_path.exists():
                        print(
                            f"  ⚠ Skipping {dataset}/{model_name}/seed_{seed}: missing {checkpoint_path.name}"
                        )
                        continue
                    try:
                        ifno_model = load_ifno_model(
                            dataset_info=dataset_info,
                            checkpoint_path=str(checkpoint_path),
                            device=str(device),
                        )
                    except Exception as exc:
                        print(
                            f"  ✗ Failed to load IFNO ({dataset}/{model_name}/seed_{seed}): {exc}"
                        )
                        continue
                    metrics = evaluate_measurement_noise_ifno(
                        ifno_model=ifno_model,
                        dataset=test_dataset,
                        dataset_info=dataset_info,
                        noise_levels=noise_levels,
                        device=device,
                        noise_draws=noise_draws,
                    )
                else:
                    try:
                        input_encoder, output_encoder, model, evaluate_fn = load_models(
                            base_dir=str(base_dir),
                            dataset=dataset,
                            model_name=model_name,
                            seed=seed,
                            device=str(device),
                        )
                    except Exception as exc:
                        print(
                            f"  ✗ Failed to load {dataset}/{model_name}/seed_{seed}: {exc}"
                        )
                        continue

                    shared_dir = models_root / dataset / "shared" / f"seed_{seed}"
                    forward_model = None
                    forward_model_name = getattr(
                        params, "forward_model", "b2b_nonlinear"
                    )
                    if shared_dir.exists():
                        try:
                            forward_model = load_forward_model(
                                str(shared_dir),
                                forward_model_name,
                                device=str(device),
                            )
                        except Exception as exc:
                            print(
                                f"  ⚠ Forward model {forward_model_name} unavailable for {dataset}/{model_name}/seed_{seed} ({exc})"
                            )
                    else:
                        print(
                            f"  ⚠ Shared encoders not found at {shared_dir}; skipping forward metrics."
                        )

                    metrics = evaluate_measurement_noise(
                        model=model,
                        evaluate_fn=evaluate_fn,
                        input_function_encoder=input_encoder,
                        output_function_encoder=output_encoder,
                        forward_model=forward_model,
                        dataset=test_dataset,
                        noise_levels=noise_levels,
                        device=device,
                        noise_draws=noise_draws,
                    )

                model_results_dir = (
                    dataset_runs_dir / model_name / f"seed_{seed}"
                )
                write_model_reports(model_results_dir, metrics)
                row_entries = collect_dataset_rows(dataset, model_name, seed, metrics)
                dataset_rows.extend(row_entries)
                model_entries.setdefault(model_name, []).extend(row_entries)

        write_dataset_csv(dataset_runs_dir, dataset_rows)
        write_dataset_summary(dataset_runs_dir, dataset, model_entries)


def evaluate_measurement_noise_ifno(
    ifno_model,
    dataset,
    dataset_info,
    noise_levels: List[float],
    device: torch.device,
    noise_draws: int,
) -> Dict[str, Dict[str, float]]:
    stats_per_noise = {level: NoiseStats() for level in noise_levels}

    ifno_model.eval()
    with torch.no_grad():
        for sample in dataset:
            (X, u_true, Y, s_true), (X_b, u_b, Y_b, s_b) = _prepare_sample(
                sample, device
            )
            for noise_std, stats in stats_per_noise.items():
                draws = 1 if noise_std == 0.0 else noise_draws
                for _ in range(draws):
                    noise_tensor = (
                        torch.zeros_like(s_b)
                        if noise_std == 0.0
                        else noise_std * torch.randn_like(s_b)
                    )
                    s_input = torch.cat([Y_b, s_b + noise_tensor], dim=-1)
                    result = ifno_model.inverse(s_input)
                    if isinstance(result, (tuple, list)):
                        u_pred = result[0]
                    else:
                        u_pred = result
                    if u_pred.shape[-1] > u_true.shape[-1]:
                        u_pred = u_pred[..., -u_true.shape[-1] :]
                    stats.update_inverse(u_pred.squeeze(0), u_true)

                    # Forward re-simulation
                    u_input = torch.cat([X_b, u_pred], dim=-1)
                    forward_result = ifno_model(u_input)
                    if isinstance(forward_result, (tuple, list)):
                        s_pred = forward_result[0]
                    else:
                        s_pred = forward_result
                    if s_pred.shape[-1] > s_true.shape[-1]:
                        s_pred = s_pred[..., -s_true.shape[-1] :]
                    stats.update_forward(s_pred.squeeze(0), s_true)

    return {
        (format_noise_value(level) if level > 0 else "0"): stats.to_metrics(level)
        for level, stats in stats_per_noise.items()
    }


if __name__ == "__main__":
    main()
