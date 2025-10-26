"""
Publication-quality plotting script for Burgers 1D inverse problem results.

Creates publication-ready figures with:
- 6.5 inch width for journal publications
- 5-7pt sans serif fonts
- Professional styling and layout
- High-resolution output (300 DPI)
- Single-row layout with paired input/output comparisons

To run: cd /workspaces/b2b-operator-inverse && python -m inverse_neural_operator.plots.plot_burgers_publication
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from b2b.load_model import load_forward_model
from data.load_dataset import load_dataset
from models.load_model import load_models

device = "cpu"

# Available inverse models to evaluate
INVERSE_MODELS: Sequence[str] = (
    "linear",
    "linear_inverse",
    "nonlinear",
    "inn_affine",
    "cinn_affine",
    "variational_autoencoder",
    "conditional_realnvp",
    "mixture_density_network",
)

MAX_MODELS_TO_DISPLAY = 6
N_SAMPLES_PER_MODEL = 8

_DISPLAY_NAMES = {
    "linear": "Linear",
    "linear_inverse": "Linear-Inv",
    "nonlinear": "Nonlinear",
    "inn_affine": "INN-Affine",
    "cinn_affine": "cINN-Affine",
    "variational_autoencoder": "cVAE",
    "conditional_realnvp": "RealNVP",
    "mixture_density_network": "MDN",
}

_COLORS = {
    "u_true": "#1f77b4",
    "u_pred": "#ff7f0e",
    "s_true": "#2E8B57",
    "s_pred": "#9467bd",
}

# Colormap to assign distinct colors to different models
MODEL_CMAP = mpl.cm.get_cmap("tab10")


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "figure.figsize": [6.5, 2.0],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": False,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )


def _display_name(model_name: str) -> str:
    return _DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def _evaluate_models_on_subset(
    test_dataset,
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    max_samples: int = 32,
) -> Tuple[Dict[str, List[float]], List[Tuple[int, float]]]:
    """Evaluate models on a subset to gather per-model MSEs and sample medians."""
    per_model_mses = {name: [] for name in models_dict.keys()}
    sample_scores: List[Tuple[int, float]] = []

    n_eval = min(max_samples, len(test_dataset))
    for idx in range(n_eval):
        X, u_true, Y, s_observed = test_dataset[idx]
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

        sample_mses: List[float] = []
        for model_name, (model, evaluate_fn) in models_dict.items():
            model.eval()
            with torch.no_grad():
                u_pred, _ = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )
                mse = torch.mean((u_pred.squeeze(0) - u_true) ** 2).item()
            per_model_mses[model_name].append(mse)
            sample_mses.append(mse)

        if sample_mses:
            sample_scores.append((idx, float(np.median(sample_mses))))

    return per_model_mses, sample_scores


def _make_burgers_output_transform(forward_model, output_function_encoder):
    if forward_model is None:
        return None

    def _transform(latent, Y):
        beta_pred = forward_model.forward(latent)
        return output_function_encoder(Y.unsqueeze(0), beta_pred)

    return _transform


def _collect_predictions(
    sample,
    models_to_plot: Iterable[str],
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]],
    input_function_encoder,
    output_function_encoder,
    output_transform,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Collect multiple samples of predictions and re-simulations for one test sample."""
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

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
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

        input_samples: List[np.ndarray] = []
        output_samples: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(N_SAMPLES_PER_MODEL):
                u_pred, latent = evaluate_fn(
                    model, point, input_function_encoder, output_function_encoder
                )
                u_pred_np = u_pred.squeeze(0).detach().cpu().numpy().flatten()
                input_samples.append(u_pred_np)

                if output_transform is None or latent is None:
                    continue

                try:
                    s_resim = output_transform(latent, Y)
                    s_resim_np = s_resim.squeeze(0).detach().cpu().numpy().flatten()
                    output_samples.append(s_resim_np)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  Warning: failed to create re-simulation for {model_name}: {exc}"
                    )
                    break

        predictions[model_name] = {
            "inputs": np.stack(input_samples) if input_samples else np.empty((0,)),
            "outputs": np.stack(output_samples) if output_samples else np.empty((0,)),
        }

    return predictions, meta


def _plot_single_row(
    sample_idx: int,
    models_to_plot: Sequence[str],
    predictions: Dict[str, Dict[str, np.ndarray]],
    meta: Dict[str, np.ndarray],
    save_dir: str | None,
) -> None:
    """Create the single-row publication figure."""
    if not models_to_plot:
        return

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(6.5, 2.0), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.5 / 72.0, h_pad=0.5 / 72.0, hspace=0.0, wspace=0.0
    )

    # 2 rows x 8 columns: 3 plots, spacer, 3 plots, spacer (matches test_gridspec)
    gs = fig.add_gridspec(
        2,
        8,
        width_ratios=[1, 1, 1, 0.05, 1, 1, 1, 0.05],
        hspace=0.0,
        wspace=0.0,
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    ax_left_parent = fig.add_subplot(gs[:, 0:3], frameon=False)
    ax_left_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_left_parent.set_xlabel(r"$x$", labelpad=-8)
    ax_left_parent.set_ylabel(r"$f(x)$", labelpad=-8)
    ax_left_parent.set_title("Input Reconstructions")

    ax_right_parent = fig.add_subplot(gs[:, 4:7], frameon=False)
    ax_right_parent.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    ax_right_parent.set_xlabel(r"$y$", labelpad=-8)
    ax_right_parent.set_ylabel(r"$h(y)$", labelpad=-8)
    ax_right_parent.set_title("Output Re-simulations")

    # 2x3 grids for left/right
    input_axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(3)]
    output_axes = [
        fig.add_subplot(gs[row, col + 4]) for row in range(2) for col in range(3)
    ]

    # spacer columns (for consistent spacing with image-style gridspec)
    spacer_left = fig.add_subplot(gs[:, 3])
    spacer_left.axis("off")
    spacer_right = fig.add_subplot(gs[:, 7])
    spacer_right.axis("off")

    x = meta["x"]
    y = meta["y"]
    u_true = meta["u_true"]
    s_true = meta["s_true"]

    def _make_axis(raw_axis: np.ndarray, length: int) -> np.ndarray:
        if length <= 0:
            return np.array([])
        if raw_axis.size == length:
            return raw_axis
        if raw_axis.size == 0:
            return np.linspace(0.0, 1.0, length)
        return np.linspace(raw_axis.min(), raw_axis.max(), length)

    def _resample(
        series: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray
    ) -> np.ndarray:
        if series.size == 0 or target_axis.size == 0:
            return series
        if source_axis.size != series.size:
            source_axis = _make_axis(source_axis, series.size)
        if source_axis.size == target_axis.size and np.allclose(
            source_axis, target_axis
        ):
            return series
        return np.interp(target_axis, source_axis, series)

    x_axis_true = _make_axis(x, u_true.size)
    y_axis_true = _make_axis(y, s_true.size)

    resampled_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    u_limit_arrays: List[np.ndarray] = [u_true]
    s_limit_arrays: List[np.ndarray] = [s_true]

    for model_name in models_to_plot:
        info = predictions.get(model_name, {})
        u_samples = info.get("inputs", np.empty((0,)))
        s_samples = info.get("outputs", np.empty((0,)))

        res_inputs: List[np.ndarray] = []
        if u_samples.size:
            for sample in np.atleast_2d(u_samples):
                sample_axis = _make_axis(x, sample.size)
                resampled = _resample(sample, sample_axis, x_axis_true)
                res_inputs.append(resampled)
        resampled_inputs = (
            np.stack(res_inputs) if res_inputs else np.empty((0, x_axis_true.size))
        )
        if res_inputs:
            u_limit_arrays.append(resampled_inputs)

        res_outputs: List[np.ndarray] = []
        if s_samples.size:
            for sample in np.atleast_2d(s_samples):
                sample_axis = _make_axis(y, sample.size)
                resampled = _resample(sample, sample_axis, y_axis_true)
                res_outputs.append(resampled)
        resampled_outputs = (
            np.stack(res_outputs) if res_outputs else np.empty((0, y_axis_true.size))
        )
        if res_outputs:
            s_limit_arrays.append(resampled_outputs)

        resampled_predictions[model_name] = {
            "inputs": resampled_inputs,
            "outputs": resampled_outputs,
        }

    def _axis_limits(values: List[np.ndarray]) -> Tuple[float, float]:
        flat = (
            np.concatenate([v.reshape(-1) for v in values])
            if values
            else np.array([0.0])
        )
        v_min, v_max = float(np.min(flat)), float(np.max(flat))
        if v_max == v_min:
            return v_min - 1.0, v_max + 1.0
        margin = 0.05 * (v_max - v_min)
        return v_min - margin, v_max + margin

    u_lim = _axis_limits(u_limit_arrays)
    s_lim = _axis_limits(s_limit_arrays)

    def _annotate(ax, text: str) -> None:
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=6,
            color="white",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    # Assign a unique color to each model to be plotted
    cmap = MODEL_CMAP
    model_colors: Dict[str, Tuple[float, float, float, float]] = {}
    for i, mn in enumerate(models_to_plot):
        model_colors[mn] = cmap(i % cmap.N)

    for idx, model_name in enumerate(models_to_plot):
        if idx >= len(input_axes):
            break

        resampled_info = resampled_predictions.get(model_name, {})
        res_inputs = resampled_info.get("inputs", np.empty((0, x_axis_true.size)))
        res_outputs = resampled_info.get("outputs", np.empty((0, y_axis_true.size)))

        ax_in = input_axes[idx]
        model_color = model_colors.get(model_name, _COLORS["u_pred"])
        if res_inputs.size:
            for sample in res_inputs:
                ax_in.plot(
                    x_axis_true,
                    sample,
                    color=model_color,
                    alpha=0.18,
                    linewidth=0.6,
                )
            ax_in.plot(
                x_axis_true,
                res_inputs.mean(axis=0),
                color=model_color,
                linewidth=0.9,
            )
        ax_in.plot(
            x_axis_true,
            u_true,
            color=_COLORS["u_true"],
            linewidth=1.0,
        )
        ax_in.set_ylim(u_lim)
        ax_in.set_xlim(x_axis_true.min(), x_axis_true.max())
        # keep tick positions (so grid lines can draw) but hide tick marks and labels
        ax_in.tick_params(labelsize=6, length=0, width=0.5, which="both")
        ax_in.set_ylabel("")
        ax_in.set_xlabel("")
        # hide tick labels while preserving tick positions for grid lines
        ax_in.tick_params(labelbottom=False, labelleft=False)
        _annotate(ax_in, _display_name(model_name))
        ax_in.set_axisbelow(True)
        ax_in.grid(True, linestyle="-", linewidth=0.4, alpha=0.6)

        ax_out = output_axes[idx]
        if res_outputs.size:
            for sample in res_outputs:
                ax_out.plot(
                    y_axis_true,
                    sample,
                    color=model_color,
                    alpha=0.18,
                    linewidth=0.6,
                )
            ax_out.plot(
                y_axis_true,
                res_outputs.mean(axis=0),
                color=model_color,
                linewidth=0.9,
            )
            ax_out.plot(
                y_axis_true,
                s_true,
                color=_COLORS["s_true"],
                linewidth=1.0,
            )
            ax_out.set_ylim(s_lim)
            ax_out.set_xlim(y_axis_true.min(), y_axis_true.max())
            ax_out.tick_params(labelsize=6, length=2, width=0.5)
            ax_out.set_ylabel("")
            ax_out.set_xlabel("")
            # hide tick marks and labels but keep tick positions for grid lines
            ax_out.tick_params(labelsize=6, length=0, width=0.5, which="both")
            ax_out.tick_params(labelbottom=False, labelleft=False)
            _annotate(ax_out, _display_name(model_name))
            ax_out.set_axisbelow(True)
            ax_out.grid(True, linestyle="-", linewidth=0.4, alpha=0.6)
        else:
            ax_out.axis("off")
            continue

    for ax in input_axes[len(models_to_plot) :]:
        ax.axis("off")
    for ax in output_axes[len(models_to_plot) :]:
        ax.axis("off")

    # Legend removed: model colors are annotated directly on panels and a legend
    # would clutter the tight publication layout.

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(
            save_dir, f"unified_comparison_sample_{sample_idx}_publication"
        )
        fig.savefig(
            f"{base}.pdf", format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.05
        )
        fig.savefig(
            f"{base}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.05
        )

    plt.close(fig)


def load_all_models(log_dir: str, dataset_info, seed: int = 1):
    """Load inverse models and shared encoders."""
    models_dict: Dict[str, Tuple[torch.nn.Module, callable]] = {}
    input_function_encoder = None
    output_function_encoder = None

    for model_name in INVERSE_MODELS:
        model_log_dir = os.path.join(log_dir, model_name, f"seed_{seed}")
        params_path = os.path.join(model_log_dir, "params.pth")
        if not os.path.exists(params_path):
            print(f"  Skipping {model_name} - not found")
            continue

        try:
            params = torch.load(params_path, weights_only=False)
            inp_enc, out_enc, model, evaluate_fn = load_models(
                log_dir=model_log_dir,
                dataset_info=dataset_info,
                params=params,
                device=device,
            )
            if input_function_encoder is None:
                input_function_encoder = inp_enc
                output_function_encoder = out_enc
            models_dict[model_name] = (model, evaluate_fn)
            print(f"  Loaded {model_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Error loading {model_name}: {exc}")

    return models_dict, input_function_encoder, output_function_encoder


def _load_forward_model(log_dir: str, seed: int):
    """Load the nonlinear forward model for Burgers re-simulations."""
    shared_log_dir = os.path.join(log_dir, "shared", f"seed_{seed}")
    forward_model_name = "b2b_nonlinear"
    try:
        forward_model = load_forward_model(
            log_dir=shared_log_dir,
            forward_model_name=forward_model_name,
            device=device,
        )
        forward_model.eval()
        print(f"  Loaded {forward_model_name}")
        return forward_model
    except Exception as exc:  # noqa: BLE001
        print(f"  Warning: could not load forward model {forward_model_name}: {exc}")
        return None


def plot_all_models_comparison(
    log_dir: str,
    results_dir: str,
    test_dataset,
    dataset_info,
    seed: int = 1,
    sample_index: int | None = None,
) -> bool:
    """Create the single-row publication figure."""
    print("Loading inverse models...")
    models_dict, input_function_encoder, output_function_encoder = load_all_models(
        log_dir, dataset_info, seed
    )
    if not models_dict:
        print("ERROR: no models available for plotting.")
        return False

    print("Evaluating models to select top performers...")
    per_model_mses, sample_scores = _evaluate_models_on_subset(
        test_dataset,
        models_dict,
        input_function_encoder,
        output_function_encoder,
    )

    ranked_models = sorted(
        per_model_mses.keys(), key=lambda k: np.mean(per_model_mses[k])
    )
    models_to_plot = ranked_models[:MAX_MODELS_TO_DISPLAY]
    print(f"  Displaying models: {', '.join(models_to_plot)}")

    if sample_index is None:
        if not sample_scores:
            print("ERROR: unable to score samples.")
            return False
        median_idx = len(sample_scores) // 2
        sample_index = sample_scores[median_idx][0]
        print(f"  Selected representative sample index: {sample_index}")
    else:
        print(f"  Using provided sample index: {sample_index}")

    sample = test_dataset[sample_index]
    forward_model = _load_forward_model(log_dir, seed)
    output_transform = _make_burgers_output_transform(
        forward_model, output_function_encoder
    )

    print("Collecting predictions...")
    predictions, meta = _collect_predictions(
        sample,
        models_to_plot,
        models_dict,
        input_function_encoder,
        output_function_encoder,
        output_transform,
    )

    print("Rendering publication figure...")
    _plot_single_row(
        sample_index, models_to_plot, predictions, meta, save_dir=results_dir
    )
    return True


def _find_any_params(log_dir: str, seed: int) -> torch.Tensor:
    """Locate any available params file to recover dataset settings."""
    for model_name in INVERSE_MODELS:
        params_path = os.path.join(log_dir, model_name, f"seed_{seed}", "params.pth")
        if os.path.exists(params_path):
            return torch.load(params_path, weights_only=False)
    raise FileNotFoundError(f"No trained models found under {log_dir}")


def main() -> None:
    setup_publication_style()

    parser = argparse.ArgumentParser(
        description="Create publication-quality Burgers plots."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/store/at46867/b2b_operator_inverse",
        help="Base log directory",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/burgers_1d",
        help="Results directory for saving publication plots",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Specific sample index to plot (overrides automatic selection)",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_name = "burgers_1d"
    log_dir = os.path.join(args.log_dir, dataset_name)
    results_dir = args.results_dir

    if not os.path.exists(log_dir):
        print(f"ERROR: log directory not found: {log_dir}")
        raise SystemExit(1)

    try:
        params = _find_any_params(log_dir, args.seed)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

    print("Loading dataset...")
    test_dataset, dataset_info = load_dataset(
        params.dataset, params, device, split="test", return_info=True
    )

    os.makedirs(results_dir, exist_ok=True)
    success = plot_all_models_comparison(
        log_dir=log_dir,
        results_dir=results_dir,
        test_dataset=test_dataset,
        dataset_info=dataset_info,
        seed=args.seed,
        sample_index=args.sample_index,
    )

    print("\n" + "=" * 50)
    if success:
        print(f"SUCCESS: Created publication figure → {results_dir}")
    else:
        print("FAILED: Could not create publication figure")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
