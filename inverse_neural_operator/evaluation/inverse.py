"""Evaluate migrated inverse model artifacts without retraining."""

from __future__ import annotations

import argparse
import time

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    results_root,
    run_artifact_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an inverse model artifact.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=["linear_inverse", "nonlinear"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Optional batch limit. Defaults to full split evaluation.",
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


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    inverse_config = config.inverse_models

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)

    inverse_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "inverse_models",
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
    eval_dir = (
        run_artifact_dir(
            result_root,
            config.dataset.name,
            "evaluation",
            "inverse_models",
            args.seed,
        )
        / args.model
        / args.split
    )

    if not args.execute:
        print("Dry run: inverse evaluation will not execute.")
        print(f"inverse_dir: {inverse_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"encoder_dir: {encoder_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"forward_dir: {forward_dir or '<none>'}")
        print(f"eval_dir:    {eval_dir}")
        return
    assert inverse_dir is not None
    assert encoder_dir is not None
    from inverse_neural_operator.forward.artifacts import require_forward_model_artifact
    from inverse_neural_operator.function_encoders.artifacts import (
        require_function_encoder_artifact,
    )
    from inverse_neural_operator.inverse.artifacts import require_inverse_model_artifact

    try:
        require_inverse_model_artifact(inverse_dir)
        require_function_encoder_artifact(encoder_dir)
        if forward_dir is not None:
            require_forward_model_artifact(forward_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from inverse_neural_operator.forward.artifacts import load_forward_model
    from inverse_neural_operator.function_encoders.artifacts import load_function_encoder
    from inverse_neural_operator.inverse.build import create_inverse_model
    from inverse_neural_operator.inverse.train import _evaluate
    from inverse_neural_operator.runtime.distributed import cleanup_distributed, setup_distributed

    context = setup_distributed(config.runtime.device)
    start_time = time.time()
    try:
        dataset = _load_dataset(config, args.split)
        dataset_info = dataset.get_info()
        n_basis = config.function_encoders.basis.n_basis

        weights_path = inverse_dir / "model.safetensors"
        model = create_inverse_model(
            args.model,
            input_size=n_basis,
            output_size=n_basis,
            hidden_sizes=inverse_config.hidden_sizes,
        ).to(context.device)
        model.load_state_dict(load_file(str(weights_path), device=str(context.device)))
        model.eval()

        input_encoder = load_function_encoder(
            encoder_dir,
            encoder_type="input",
            dataset_info=dataset_info,
            device=context.device,
        )
        output_encoder = load_function_encoder(
            encoder_dir,
            encoder_type="output",
            dataset_info=dataset_info,
            device=context.device,
        )

        forward_model = None
        if forward_dir is not None:
            forward_model = load_forward_model(
                forward_dir,
                model_name=inverse_config.forward_model,
                input_size=n_basis,
                output_size=n_basis,
                hidden_sizes=config.forward_models.hidden_sizes,
                device=context.device,
            )

        sampler = (
            DistributedSampler(dataset, shuffle=False)
            if context.is_distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=inverse_config.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )

        inverse_config.eval_batches = args.eval_batches
        metrics = _evaluate(
            model,
            loader,
            context,
            input_encoder,
            output_encoder,
            forward_model,
            dataset_info,
            inverse_config,
        )
        metrics.update(
            {
                "split": args.split,
                "elapsed_seconds": time.time() - start_time,
                "model_dir": str(inverse_dir),
                "function_encoder_artifact": inverse_config.function_encoder_artifact,
                "forward_model": inverse_config.forward_model,
            }
        )
        if context.is_rank_zero:
            write_json(eval_dir / "metrics.json", metrics)
            print(f"Wrote evaluation metrics to {eval_dir / 'metrics.json'}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
