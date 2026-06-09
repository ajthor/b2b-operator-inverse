"""Baseline training entrypoint for migrated experiment configs."""

from __future__ import annotations

import argparse
import time

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
    parser = argparse.ArgumentParser(description="Train a baseline model.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=["ifno"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train. Without this flag the command only prints paths.",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing final baseline artifact.",
    )
    return parser.parse_args()


def _load_dataset(config, split: str):
    if config.dataset.name == "burgers_1d":
        from inverse_neural_operator.data.burgers_hf import load_burgers_dataset

        return load_burgers_dataset(
            split=split,
            source=config.dataset.source or "ajthor/burgers-fenics",
            sample_limit=config.dataset.sample_limit,
        )
    raise ValueError(
        "The migrated baseline path currently supports dataset=burgers_1d. "
        "FWI IFNO needs a separate shape fix before it should be ported here."
    )


def _move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _relative_l2_loss(x, y):
    import torch

    batch_size = x.shape[0]
    diff = torch.norm(x.reshape(batch_size, -1) - y.reshape(batch_size, -1), p=2, dim=1)
    denom = torch.clamp(torch.norm(y.reshape(batch_size, -1), p=2, dim=1), min=1e-12)
    return torch.mean(diff / denom)


def _forward_loss(model, batch):
    import torch

    module = _unwrap(model)
    x, u, _, s = batch
    result = model(torch.cat([x, u], dim=-1))
    if isinstance(result, tuple):
        pred_s, reconstruction_loss = result
    else:
        pred_s = result
        reconstruction_loss = torch.zeros((), device=s.device)
    if pred_s.shape[-1] > s.shape[-1]:
        pred_s = pred_s[..., -s.shape[-1] :]
    return _relative_l2_loss(pred_s, s) + reconstruction_loss


def _backward_loss(model, batch):
    import torch

    module = _unwrap(model)
    x, u, y, s = batch
    result = module.inverse(torch.cat([y, s], dim=-1))
    if isinstance(result, tuple):
        pred_u, reconstruction_loss = result
    else:
        pred_u = result
        reconstruction_loss = torch.zeros((), device=u.device)
    if pred_u.shape[-1] > u.shape[-1]:
        pred_u = pred_u[..., -u.shape[-1] :]
    return _relative_l2_loss(pred_u, u) + reconstruction_loss


def _vae_loss(model, batch):
    import torch

    module = _unwrap(model)
    _, u, _, _ = batch
    u_vae = module._get_vae_input_shape(u)
    vae_out, vae_input, mu, log_var = module.vae_net(u_vae)
    reconstruction_loss = _relative_l2_loss(
        vae_out.reshape(u.shape[0], -1),
        vae_input.reshape(u.shape[0], -1),
    )
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )
    return reconstruction_loss + 0.01 * kl_loss


def _all_reduce_sum(tensor, context):
    import torch

    if context.is_distributed:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _evaluate(model, loader, context):
    import torch

    model.eval()
    forward_total = torch.zeros((), device=context.device)
    backward_total = torch.zeros((), device=context.device)
    count = torch.zeros((), device=context.device)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, context.device)
            forward_total += _forward_loss(model, batch).detach()
            backward_total += _backward_loss(model, batch).detach()
            count += 1
    _all_reduce_sum(forward_total, context)
    _all_reduce_sum(backward_total, context)
    _all_reduce_sum(count, context)
    count = torch.clamp(count, min=1)
    forward_loss = float((forward_total / count).item())
    backward_loss = float((backward_total / count).item())
    return {
        "forward_loss": forward_loss,
        "backward_loss": backward_loss,
        "loss": forward_loss + backward_loss,
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    baseline_config = config.baselines

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)
    model_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "baselines",
            args.model,
            args.seed,
        )
        if model_root is not None
        else None
    )
    run_dir = run_artifact_dir(
        result_root,
        config.dataset.name,
        "baselines",
        args.model,
        args.seed,
    )

    if not args.execute:
        print("Dry run: baseline training will not execute.")
        print(f"model_dir: {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"run_dir:   {run_dir}")
        return

    if args.model not in baseline_config.models:
        raise SystemExit(
            f"Model {args.model!r} is not listed in baselines.models: "
            f"{baseline_config.models}"
        )
    assert model_dir is not None
    refuse_existing_artifact(model_dir / "model.safetensors", overwrite=args.overwrite)

    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    from inverse_neural_operator.baselines.build import create_baseline_model
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

        model = create_baseline_model(args.model, dataset_info, config).to(context.device)
        ddp_model = model
        if context.is_distributed:
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )
            ddp_model._set_static_graph()

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
            batch_size=baseline_config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=baseline_config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )

        writer = SummaryWriter(log_dir=str(run_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"
        start_time = time.time()

        ifno = baseline_config.ifno
        if ifno.epochs_vae and context.is_distributed:
            raise RuntimeError("Distributed IFNO VAE pretraining is not ported yet.")
        if ifno.epochs_ifno:
            raise RuntimeError(
                "IFNO pretraining is not ported in the migrated baseline stage yet. "
                "Set baselines.ifno.epochs_ifno to 0."
            )
        if ifno.epochs_vae:
            vae_optimizer = torch.optim.AdamW(model.vae_net.parameters(), lr=ifno.lr_vae)
            for epoch in range(ifno.epochs_vae):
                model.train()
                total = torch.zeros((), device=context.device)
                count = torch.zeros((), device=context.device)
                for batch in train_loader:
                    batch = _move_batch(batch, context.device)
                    vae_optimizer.zero_grad(set_to_none=True)
                    loss = _vae_loss(model, batch)
                    loss.backward()
                    vae_optimizer.step()
                    total += loss.detach()
                    count += 1
                if context.is_rank_zero and writer is not None:
                    writer.add_scalar(
                        "phase1_vae/train_loss",
                        float((total / torch.clamp(count, min=1)).item()),
                        epoch,
                    )

        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=baseline_config.learning_rate,
            weight_decay=1e-4,
        )
        best_test = None
        metrics = {}
        for epoch in range(baseline_config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            ddp_model.train()
            forward_total = torch.zeros((), device=context.device)
            backward_total = torch.zeros((), device=context.device)
            count = torch.zeros((), device=context.device)
            for batch in train_loader:
                batch = _move_batch(batch, context.device)
                optimizer.zero_grad(set_to_none=True)
                forward_loss = _forward_loss(ddp_model, batch)
                backward_loss = _backward_loss(ddp_model, batch)
                loss = forward_loss + backward_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=2.0)
                optimizer.step()
                forward_total += forward_loss.detach()
                backward_total += backward_loss.detach()
                count += 1
            _all_reduce_sum(forward_total, context)
            _all_reduce_sum(backward_total, context)
            _all_reduce_sum(count, context)
            count = torch.clamp(count, min=1)
            train_forward = float((forward_total / count).item())
            train_backward = float((backward_total / count).item())
            metrics = _evaluate(ddp_model, test_loader, context)
            best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
            if context.is_rank_zero:
                assert writer is not None
                writer.add_scalar("loss/train_forward", train_forward, epoch)
                writer.add_scalar("loss/train_backward", train_backward, epoch)
                writer.add_scalar("loss/test_forward", metrics["forward_loss"], epoch)
                writer.add_scalar("loss/test_backward", metrics["backward_loss"], epoch)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": metrics["loss"],
                    },
                    checkpoint_path,
                )

        barrier(context)
        if context.is_rank_zero:
            assert model_dir is not None
            model_dir.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), str(model_dir / "model.safetensors"))
            write_yaml(model_dir / "config.yaml", config.raw)
            metrics.update(
                {
                    "best_test_loss": best_test,
                    "epochs": baseline_config.epochs,
                    "elapsed_seconds": time.time() - start_time,
                    "dataset_info": dataset_info,
                }
            )
            write_json(model_dir / "metrics.json", metrics)
            write_manifest(
                model_dir,
                artifact_type="baseline",
                dataset=config.dataset.name,
                name=args.model,
                seed=args.seed,
                files={"model": "model.safetensors"},
            )
            if writer is not None:
                writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
