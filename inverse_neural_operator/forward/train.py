"""DDP-ready forward model training entrypoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from inverse_neural_operator.config.schema import load_experiment_config
from inverse_neural_operator.runtime.paths import (
    model_artifact_dir,
    models_root,
    refuse_existing_artifact,
    results_root,
    run_artifact_dir,
    write_json,
    write_manifest,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a forward coefficient model.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=["b2b_linear", "b2b_nonlinear"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train. Without this flag the command only prints paths.",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional root for TensorBoard logs. Checkpoints still use --results-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing final forward-model artifact.",
    )
    return parser.parse_args()


def _load_dataset(config, split: str):
    from inverse_neural_operator.data.overhaul import load_overhaul_dataset

    return load_overhaul_dataset(config, split)


def _move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _coefficients(batch, input_encoder, output_encoder):
    import torch

    X, u, Y, s = batch
    with torch.no_grad():
        alpha, _ = input_encoder.compute_coefficients(X, u)
        beta, _ = output_encoder.compute_coefficients(Y, s)
    return alpha, beta, Y, s


def _loss(
    model,
    batch,
    input_encoder,
    output_encoder,
    coefficient_weight,
    reconstruction_weight,
):
    import torch

    alpha, beta, Y, s = _coefficients(batch, input_encoder, output_encoder)
    beta_pred = model(alpha)
    coefficient_loss = torch.nn.functional.mse_loss(beta_pred, beta)
    reconstruction_loss = torch.nn.functional.mse_loss(output_encoder(Y, beta_pred), s)
    total = coefficient_weight * coefficient_loss + reconstruction_weight * reconstruction_loss
    return total, coefficient_loss.detach(), reconstruction_loss.detach()


def _all_reduce_sum(tensor, context):
    import torch

    if context.is_distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _batch_metrics(
    model,
    batch,
    input_encoder,
    output_encoder,
    dataset_info,
    forward_config,
):
    import torch
    from inverse_neural_operator.evaluation.metrics import mean_ssim, relative_l2

    alpha, beta, Y, s = _coefficients(batch, input_encoder, output_encoder)
    beta_pred = model(alpha)
    s_pred = output_encoder(Y, beta_pred)
    coefficient_loss = torch.nn.functional.mse_loss(beta_pred, beta)
    reconstruction_mse = torch.nn.functional.mse_loss(s_pred, s)
    total = (
        forward_config.coefficient_loss_weight * coefficient_loss
        + forward_config.reconstruction_loss_weight * reconstruction_mse
    )
    return {
        "loss": total,
        "coefficient_mse": coefficient_loss.detach(),
        "output_mse": reconstruction_mse.detach(),
        "output_relative_l2": relative_l2(s_pred, s).detach(),
        "output_ssim": mean_ssim(s_pred, s, dataset_info["output_spatial_dims"]),
    }


def _evaluate(
    model,
    loader,
    context,
    input_encoder,
    output_encoder,
    dataset_info,
    forward_config,
):
    import torch

    model.eval()
    totals = {
        "loss": torch.zeros((), device=context.device),
        "coefficient_mse": torch.zeros((), device=context.device),
        "output_mse": torch.zeros((), device=context.device),
        "output_relative_l2": torch.zeros((), device=context.device),
    }
    ssim_total = 0.0
    ssim_count = 0
    count = torch.zeros((), device=context.device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if (
                forward_config.eval_batches is not None
                and batch_index >= forward_config.eval_batches
            ):
                break
            batch = _move_batch(batch, context.device)
            metrics = _batch_metrics(
                model,
                batch,
                input_encoder,
                output_encoder,
                dataset_info,
                forward_config,
            )
            for key in totals:
                totals[key] += metrics[key].detach()
            if metrics["output_ssim"] is not None:
                ssim_total += metrics["output_ssim"]
                ssim_count += 1
            count += 1
    for key in totals:
        _all_reduce_sum(totals[key], context)
    _all_reduce_sum(count, context)
    ssim_total_tensor = torch.tensor(ssim_total, device=context.device)
    ssim_count_tensor = torch.tensor(ssim_count, device=context.device)
    _all_reduce_sum(ssim_total_tensor, context)
    _all_reduce_sum(ssim_count_tensor, context)
    count = torch.clamp(count, min=1)
    payload = {key: float((value / count).item()) for key, value in totals.items()}
    payload["coefficient_loss"] = payload["coefficient_mse"]
    payload["reconstruction_loss"] = payload["output_mse"]
    payload["output_ssim"] = (
        float((ssim_total_tensor / ssim_count_tensor).item())
        if int(ssim_count_tensor.item())
        else None
    )
    payload["eval_batches"] = int(count.item())
    return payload


def _train_linear_closed_form(
    model,
    train_loader,
    test_loader,
    context,
    input_encoder,
    output_encoder,
    dataset_info,
    forward_config,
):
    import torch

    n = model.input_size
    m = model.output_size
    sxx = torch.zeros((n, n), device=context.device)
    sxy = torch.zeros((n, m), device=context.device)
    with torch.no_grad():
        for batch in train_loader:
            batch = _move_batch(batch, context.device)
            alpha, beta, _, _ = _coefficients(batch, input_encoder, output_encoder)
            sxx += torch.einsum("ij,ik->jk", alpha, alpha)
            sxy += torch.einsum("ij,ik->jk", alpha, beta)
    _all_reduce_sum(sxx, context)
    _all_reduce_sum(sxy, context)
    sxx += forward_config.linear_regularization * torch.eye(n, device=context.device)
    weights_t = torch.linalg.solve(sxx, sxy)
    model.linear.weight.copy_(weights_t.T)
    return _evaluate(
        model,
        test_loader,
        context,
        input_encoder,
        output_encoder,
        dataset_info,
        forward_config,
    )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    forward_config = config.forward_models

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)
    model_dir = (
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
    run_dir = run_artifact_dir(
        result_root,
        config.dataset.name,
        "forward_models",
        args.model,
        args.seed,
    )
    tensorboard_root = args.tensorboard_dir or config.runtime.tensorboard_dir
    tensorboard_dir = (
        run_artifact_dir(
            Path(tensorboard_root).expanduser().resolve(),
            config.dataset.name,
            "forward_models",
            args.model,
            args.seed,
        )
        if tensorboard_root
        else run_dir
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

    if not args.execute:
        print("Dry run: forward model training will not execute.")
        print(f"model_dir:   {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"encoder_dir: {encoder_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"run_dir:     {run_dir}")
        print(f"tensorboard_dir: {tensorboard_dir}")
        return

    if args.model not in forward_config.models:
        raise SystemExit(
            f"Model {args.model!r} is not listed in forward_models.models: "
            f"{forward_config.models}"
        )
    assert model_dir is not None
    refuse_existing_artifact(model_dir / "model.safetensors", overwrite=args.overwrite)
    assert encoder_dir is not None
    from inverse_neural_operator.function_encoders.artifacts import (
        require_function_encoder_artifact,
    )

    try:
        require_function_encoder_artifact(encoder_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    from inverse_neural_operator.forward.build import create_forward_model
    from inverse_neural_operator.function_encoders.artifacts import load_function_encoder
    from inverse_neural_operator.runtime.distributed import (
        barrier,
        cleanup_distributed,
        setup_distributed,
    )

    context = setup_distributed(config.runtime.device)
    try:
        torch.manual_seed(args.seed + context.rank)
        train_dataset = _load_dataset(config, "train")
        test_dataset = _load_dataset(config, "test")
        dataset_info = train_dataset.get_info()

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

        train_sampler = (
            DistributedSampler(train_dataset, shuffle=True)
            if context.is_distributed
            else None
        )
        test_sampler = (
            DistributedSampler(test_dataset, shuffle=False)
            if context.is_distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=forward_config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=forward_config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )

        model = create_forward_model(
            args.model,
            input_size=config.function_encoders.basis.n_basis,
            output_size=config.function_encoders.basis.n_basis,
            hidden_sizes=forward_config.hidden_sizes,
        ).to(context.device)
        ddp_model = model
        if args.model != "b2b_linear" and context.is_distributed:
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )
        writer = SummaryWriter(log_dir=str(tensorboard_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"
        start_time = time.time()

        if args.model == "b2b_linear":
            metrics = _train_linear_closed_form(
                model,
                train_loader,
                test_loader,
                context,
                input_encoder,
                output_encoder,
                dataset_info,
                forward_config,
            )
            if writer is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        writer.add_scalar(f"loss/test_{key}", value, 0)
        else:
            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=forward_config.learning_rate)
            best_test = None
            metrics = {}
            for epoch in range(forward_config.epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                ddp_model.train()
                total = torch.zeros((), device=context.device)
                count = torch.zeros((), device=context.device)
                for batch in train_loader:
                    batch = _move_batch(batch, context.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = _loss(
                        ddp_model,
                        batch,
                        input_encoder,
                        output_encoder,
                        forward_config.coefficient_loss_weight,
                        forward_config.reconstruction_loss_weight,
                    )
                    loss.backward()
                    optimizer.step()
                    total += loss.detach()
                    count += 1
                _all_reduce_sum(total, context)
                _all_reduce_sum(count, context)
                train_loss = float((total / torch.clamp(count, min=1)).item())
                metrics = _evaluate(
                    ddp_model,
                    test_loader,
                    context,
                    input_encoder,
                    output_encoder,
                    dataset_info,
                    forward_config,
                )
                best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
                if context.is_rank_zero:
                    assert writer is not None
                    writer.add_scalar("loss/train", train_loss, epoch)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and value is not None:
                            writer.add_scalar(f"loss/test_{key}", value, epoch)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": metrics["loss"],
                        },
                        checkpoint_path,
                    )
            metrics["best_test_loss"] = best_test

        barrier(context)
        if context.is_rank_zero:
            assert model_dir is not None
            model_dir.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), str(model_dir / "model.safetensors"))
            write_yaml(model_dir / "config.yaml", config.raw)
            metrics.update(
                {
                    "epochs": forward_config.epochs,
                    "elapsed_seconds": time.time() - start_time,
                    "function_encoder_artifact": forward_config.function_encoder_artifact,
                    "tensorboard_dir": str(tensorboard_dir),
                }
            )
            write_json(model_dir / "metrics.json", metrics)
            write_manifest(
                model_dir,
                artifact_type="forward_model",
                dataset=config.dataset.name,
                name=args.model,
                seed=args.seed,
                files={"model": "model.safetensors"},
                extra={"function_encoder_artifact": forward_config.function_encoder_artifact},
            )
            if writer is not None:
                writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
