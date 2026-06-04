"""DDP-ready inverse model training entrypoint."""

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
    write_manifest,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an inverse coefficient model.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=["linear_inverse", "nonlinear"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train. Without this flag the command only prints paths.",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def _load_dataset(config, split: str):
    from inverse_neural_operator.data.fwi_hf import load_fwi_dataset

    if config.dataset.name != "fwi":
        raise ValueError("The migrated inverse stage currently supports dataset=fwi only.")
    return load_fwi_dataset(
        split=split,
        source=config.dataset.source or "ajthor/fwi",
        sample_limit=config.dataset.sample_limit,
    )


def _move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _predict_alpha(model, beta):
    return model(beta)


def _coefficients(batch, input_encoder, output_encoder):
    import torch

    X, u, Y, s = batch
    with torch.no_grad():
        alpha, _ = input_encoder.compute_coefficients(X, u)
        beta, _ = output_encoder.compute_coefficients(Y, s)
    return alpha, beta, X, u, Y, s


def _relative_l2(prediction, target):
    import torch

    batch_size = prediction.shape[0]
    numerator = torch.norm(
        prediction.reshape(batch_size, -1) - target.reshape(batch_size, -1),
        p=2,
        dim=1,
    )
    denominator = torch.clamp(
        torch.norm(target.reshape(batch_size, -1), p=2, dim=1),
        min=1e-12,
    )
    return torch.mean(numerator / denominator)


def _mean_ssim(prediction, target, spatial_dims):
    if len(spatial_dims) != 2:
        return None
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except Exception:
        return None

    pred = prediction.detach().float().cpu().reshape(prediction.shape[0], *spatial_dims, -1)
    true = target.detach().float().cpu().reshape(target.shape[0], *spatial_dims, -1)
    values = []
    for pred_sample, true_sample in zip(pred, true):
        channel_values = []
        for channel in range(pred_sample.shape[-1]):
            pred_channel = pred_sample[..., channel].numpy()
            true_channel = true_sample[..., channel].numpy()
            data_range = float(np.max(true_channel) - np.min(true_channel))
            if data_range <= 0:
                data_range = 1.0
            min_dim = min(pred_channel.shape)
            if min_dim < 3:
                continue
            win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            channel_values.append(
                structural_similarity(
                    true_channel,
                    pred_channel,
                    data_range=data_range,
                    win_size=win_size,
                )
            )
        if channel_values:
            values.append(float(np.mean(channel_values)))
    if not values:
        return None
    return float(np.mean(values))


def _batch_metrics(
    model,
    batch,
    input_encoder,
    output_encoder,
    forward_model,
    dataset_info,
    coefficient_weight,
    prediction_weight,
):
    import torch

    alpha, beta, X, u, Y, s = _coefficients(batch, input_encoder, output_encoder)
    alpha_pred = _predict_alpha(model, beta)
    u_pred = input_encoder(X, alpha_pred)

    coefficient_mse = torch.nn.functional.mse_loss(alpha_pred, alpha)
    input_mse = torch.nn.functional.mse_loss(u_pred, u)
    input_relative_l2 = _relative_l2(u_pred, u)

    resimulation_mse = torch.zeros((), device=alpha.device)
    resimulation_relative_l2 = torch.zeros((), device=alpha.device)
    if forward_model is not None:
        beta_resim = forward_model(alpha_pred)
        s_resim = output_encoder(Y, beta_resim)
        resimulation_mse = torch.nn.functional.mse_loss(s_resim, s)
        resimulation_relative_l2 = _relative_l2(s_resim, s)
    else:
        s_resim = None

    total = prediction_weight * input_mse + coefficient_weight * coefficient_mse
    metrics = {
        "loss": total,
        "input_mse": input_mse.detach(),
        "input_relative_l2": input_relative_l2.detach(),
        "coefficient_mse": coefficient_mse.detach(),
        "resimulation_mse": resimulation_mse.detach(),
        "resimulation_relative_l2": resimulation_relative_l2.detach(),
        "input_ssim": _mean_ssim(u_pred, u, dataset_info["input_spatial_dims"]),
        "resimulation_ssim": (
            _mean_ssim(s_resim, s, dataset_info["output_spatial_dims"])
            if s_resim is not None
            else None
        ),
    }
    return metrics


def _all_reduce_sum(tensor, context):
    import torch

    if context.is_distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _evaluate(
    model,
    loader,
    context,
    input_encoder,
    output_encoder,
    forward_model,
    dataset_info,
    config,
):
    import torch

    model.eval()
    totals = {
        "loss": torch.zeros((), device=context.device),
        "input_mse": torch.zeros((), device=context.device),
        "input_relative_l2": torch.zeros((), device=context.device),
        "coefficient_mse": torch.zeros((), device=context.device),
        "resimulation_mse": torch.zeros((), device=context.device),
        "resimulation_relative_l2": torch.zeros((), device=context.device),
    }
    ssim_totals = {"input_ssim": 0.0, "resimulation_ssim": 0.0}
    ssim_counts = {"input_ssim": 0, "resimulation_ssim": 0}
    count = torch.zeros((), device=context.device)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, context.device)
            metrics = _batch_metrics(
                model,
                batch,
                input_encoder,
                output_encoder,
                forward_model,
                dataset_info,
                config.coefficient_loss_weight,
                config.prediction_loss_weight,
            )
            for key in totals:
                totals[key] += metrics[key].detach()
            for key in ssim_totals:
                if metrics[key] is not None:
                    ssim_totals[key] += metrics[key]
                    ssim_counts[key] += 1
            count += 1
    for key in totals:
        _all_reduce_sum(totals[key], context)
    _all_reduce_sum(count, context)
    for key in ssim_totals:
        value_tensor = torch.tensor(ssim_totals[key], device=context.device)
        count_tensor = torch.tensor(ssim_counts[key], device=context.device)
        _all_reduce_sum(value_tensor, context)
        _all_reduce_sum(count_tensor, context)
        ssim_totals[key] = float(value_tensor.item())
        ssim_counts[key] = int(count_tensor.item())
    count = torch.clamp(count, min=1)
    payload = {key: float((value / count).item()) for key, value in totals.items()}
    for key in ssim_totals:
        payload[key] = (
            ssim_totals[key] / ssim_counts[key] if ssim_counts[key] else None
        )
    return payload


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.config)
    inverse_config = experiment_config.inverse_models

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)
    model_dir = (
        model_artifact_dir(
            model_root,
            experiment_config.dataset.name,
            "inverse_models",
            args.model,
            args.seed,
        )
        if model_root is not None
        else None
    )
    run_dir = run_artifact_dir(
        result_root,
        experiment_config.dataset.name,
        "inverse_models",
        args.model,
        args.seed,
    )
    encoder_dir = (
        model_artifact_dir(
            model_root,
            experiment_config.dataset.name,
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
            experiment_config.dataset.name,
            "forward_models",
            inverse_config.forward_model,
            args.seed,
        )
        if model_root is not None and inverse_config.forward_model
        else None
    )

    if not args.execute:
        print("Dry run: inverse model training will not execute.")
        print(f"model_dir:   {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"encoder_dir: {encoder_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"forward_dir: {forward_dir or '<none>'}")
        print(f"run_dir:     {run_dir}")
        return

    if args.model not in inverse_config.models:
        raise SystemExit(
            f"Model {args.model!r} is not listed in inverse_models.models: "
            f"{inverse_config.models}"
        )
    assert encoder_dir is not None
    from inverse_neural_operator.forward.artifacts import require_forward_model_artifact
    from inverse_neural_operator.function_encoders.artifacts import (
        require_function_encoder_artifact,
    )

    try:
        require_function_encoder_artifact(encoder_dir)
        if forward_dir is not None:
            require_forward_model_artifact(forward_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    from inverse_neural_operator.forward.artifacts import load_forward_model
    from inverse_neural_operator.function_encoders.artifacts import load_function_encoder
    from inverse_neural_operator.inverse.build import create_inverse_model
    from inverse_neural_operator.runtime.distributed import (
        barrier,
        cleanup_distributed,
        setup_distributed,
    )

    context = setup_distributed(experiment_config.runtime.device)
    try:
        torch.manual_seed(args.seed + context.rank)
        train_dataset = _load_dataset(experiment_config, "train")
        test_dataset = _load_dataset(experiment_config, "test")
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
        n_basis = experiment_config.function_encoders.basis.n_basis

        forward_model = None
        if forward_dir is not None:
            forward_model = load_forward_model(
                forward_dir,
                model_name=inverse_config.forward_model,
                input_size=n_basis,
                output_size=n_basis,
                hidden_sizes=experiment_config.forward_models.hidden_sizes,
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
            batch_size=inverse_config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=experiment_config.runtime.num_workers,
            pin_memory=experiment_config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=experiment_config.runtime.num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=inverse_config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=experiment_config.runtime.num_workers,
            pin_memory=experiment_config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=experiment_config.runtime.num_workers > 0,
        )

        model = create_inverse_model(
            args.model,
            input_size=n_basis,
            output_size=n_basis,
            hidden_sizes=inverse_config.hidden_sizes,
        ).to(context.device)
        ddp_model = model
        if context.is_distributed:
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )

        writer = SummaryWriter(log_dir=str(run_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"
        start_time = time.time()

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=inverse_config.learning_rate)
        best_test = None
        metrics = {}
        for epoch in range(inverse_config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            ddp_model.train()
            total = torch.zeros((), device=context.device)
            count = torch.zeros((), device=context.device)
            for batch in train_loader:
                batch = _move_batch(batch, context.device)
                optimizer.zero_grad(set_to_none=True)
                metrics_for_batch = _batch_metrics(
                    ddp_model,
                    batch,
                    input_encoder,
                    output_encoder,
                    None,
                    dataset_info,
                    inverse_config.coefficient_loss_weight,
                    inverse_config.prediction_loss_weight,
                )
                loss = metrics_for_batch["loss"]
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
                forward_model,
                dataset_info,
                inverse_config,
            )
            best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
            if context.is_rank_zero:
                assert writer is not None
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/test", metrics["loss"], epoch)
                writer.add_scalar("loss/test_input_mse", metrics["input_mse"], epoch)
                writer.add_scalar("loss/test_resimulation_mse", metrics["resimulation_mse"], epoch)
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
            write_yaml(model_dir / "config.yaml", experiment_config.raw)
            metrics.update(
                {
                    "epochs": inverse_config.epochs,
                    "elapsed_seconds": time.time() - start_time,
                    "function_encoder_artifact": inverse_config.function_encoder_artifact,
                    "forward_model": inverse_config.forward_model,
                }
            )
            write_json(model_dir / "metrics.json", metrics)
            write_manifest(
                model_dir,
                artifact_type="inverse_model",
                dataset=experiment_config.dataset.name,
                name=args.model,
                seed=args.seed,
                files={"model": "model.safetensors"},
                extra={
                    "function_encoder_artifact": inverse_config.function_encoder_artifact,
                    "forward_model": inverse_config.forward_model,
                },
            )
            if writer is not None:
                writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
