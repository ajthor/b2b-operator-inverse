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


def _loss(
    model,
    batch,
    input_encoder,
    output_encoder,
    forward_model,
    coefficient_weight,
    resimulation_weight,
):
    import torch

    alpha, beta, X, u, Y, s = _coefficients(batch, input_encoder, output_encoder)
    alpha_pred = _predict_alpha(model, beta)
    coefficient_loss = torch.nn.functional.mse_loss(alpha_pred, alpha)
    resimulation_loss = torch.zeros((), device=alpha.device)
    reconstruction_loss = torch.nn.functional.mse_loss(input_encoder(X, alpha_pred), u)
    if forward_model is not None and resimulation_weight:
        beta_resim = forward_model(alpha_pred)
        s_resim = output_encoder(Y, beta_resim)
        resimulation_loss = torch.nn.functional.mse_loss(s_resim, s)
    total = coefficient_weight * coefficient_loss + resimulation_weight * resimulation_loss
    return (
        total,
        coefficient_loss.detach(),
        reconstruction_loss.detach(),
        resimulation_loss.detach(),
    )


def _all_reduce_sum(tensor, context):
    import torch

    if context.is_distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _evaluate(model, loader, context, input_encoder, output_encoder, forward_model, config):
    import torch

    model.eval()
    total = torch.zeros((), device=context.device)
    coefficient_total = torch.zeros((), device=context.device)
    reconstruction_total = torch.zeros((), device=context.device)
    resimulation_total = torch.zeros((), device=context.device)
    count = torch.zeros((), device=context.device)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, context.device)
            loss, coefficient_loss, reconstruction_loss, resimulation_loss = _loss(
                model,
                batch,
                input_encoder,
                output_encoder,
                forward_model,
                config.coefficient_loss_weight,
                config.resimulation_loss_weight,
            )
            total += loss.detach()
            coefficient_total += coefficient_loss
            reconstruction_total += reconstruction_loss
            resimulation_total += resimulation_loss
            count += 1
    _all_reduce_sum(total, context)
    _all_reduce_sum(coefficient_total, context)
    _all_reduce_sum(reconstruction_total, context)
    _all_reduce_sum(resimulation_total, context)
    _all_reduce_sum(count, context)
    count = torch.clamp(count, min=1)
    return {
        "loss": float((total / count).item()),
        "coefficient_loss": float((coefficient_total / count).item()),
        "input_reconstruction_loss": float((reconstruction_total / count).item()),
        "resimulation_loss": float((resimulation_total / count).item()),
    }


def _train_linear_closed_form(
    model,
    train_loader,
    test_loader,
    context,
    input_encoder,
    output_encoder,
    forward_model,
    config,
):
    import torch

    module = _unwrap(model)
    n = module.linear.in_features
    m = module.linear.out_features
    syy = torch.zeros((n, n), device=context.device)
    syx = torch.zeros((n, m), device=context.device)
    with torch.no_grad():
        for batch in train_loader:
            batch = _move_batch(batch, context.device)
            alpha, beta, *_ = _coefficients(batch, input_encoder, output_encoder)
            syy += torch.einsum("ij,ik->jk", beta, beta)
            syx += torch.einsum("ij,ik->jk", beta, alpha)
    _all_reduce_sum(syy, context)
    _all_reduce_sum(syx, context)
    syy += config.linear_regularization * torch.eye(n, device=context.device)
    weights_t = torch.linalg.solve(syy, syx)
    module.linear.weight.copy_(weights_t.T)
    return _evaluate(
        model,
        test_loader,
        context,
        input_encoder,
        output_encoder,
        forward_model,
        config,
    )


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

        assert encoder_dir is not None
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
        if args.model != "linear_inverse" and context.is_distributed:
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )

        writer = SummaryWriter(log_dir=str(run_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"
        start_time = time.time()

        if args.model == "linear_inverse":
            metrics = _train_linear_closed_form(
                model,
                train_loader,
                test_loader,
                context,
                input_encoder,
                output_encoder,
                forward_model,
                inverse_config,
            )
            if writer is not None:
                writer.add_scalar("loss/test", metrics["loss"], 0)
        else:
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
                    loss, _, _, _ = _loss(
                        ddp_model,
                        batch,
                        input_encoder,
                        output_encoder,
                        forward_model,
                        inverse_config.coefficient_loss_weight,
                        inverse_config.resimulation_loss_weight,
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
                    forward_model,
                    inverse_config,
                )
                best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
                if context.is_rank_zero:
                    assert writer is not None
                    writer.add_scalar("loss/train", train_loss, epoch)
                    writer.add_scalar("loss/test", metrics["loss"], epoch)
                    writer.add_scalar("loss/test_resimulation", metrics["resimulation_loss"], epoch)
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
