"""Baseline training entrypoint for migrated experiment configs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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


SUPPORTED_BASELINES = ["ifno", "invertible_deeponet", "nio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument("--model", required=True, choices=SUPPORTED_BASELINES)
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
        help="Allow overwriting an existing final baseline artifact.",
    )
    return parser.parse_args()


def _load_dataset(config, split: str):
    from inverse_neural_operator.data.overhaul import load_overhaul_dataset

    return load_overhaul_dataset(config, split)


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


def _loss_by_name(prediction, target, name: str):
    import torch

    if name == "mse":
        return torch.nn.functional.mse_loss(prediction, target)
    if name == "l1":
        return torch.nn.functional.l1_loss(prediction, target)
    if name == "relative_l2":
        return _relative_l2_loss(prediction, target)
    raise ValueError(f"Unsupported baseline loss {name!r}; use mse, l1, or relative_l2.")


def _ifno_predict_output(model, batch):
    import torch

    x, u, _, s = batch
    result = model(torch.cat([x, u], dim=-1))
    pred_s = result[0] if isinstance(result, tuple) else result
    if pred_s.shape[-1] > s.shape[-1]:
        pred_s = pred_s[..., -s.shape[-1] :]
    return pred_s


def _ifno_predict_input(model, batch):
    import torch

    module = _unwrap(model)
    x, u, y, s = batch
    result = module.inverse(torch.cat([y, s], dim=-1))
    pred_u = result[0] if isinstance(result, tuple) else result
    if pred_u.shape[-1] > u.shape[-1]:
        pred_u = pred_u[..., -u.shape[-1] :]
    return pred_u


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


def _numeric_metrics(prediction, target, prefix: str):
    import torch

    return {
        f"{prefix}_l1": torch.nn.functional.l1_loss(prediction, target),
        f"{prefix}_mse": torch.nn.functional.mse_loss(prediction, target),
        f"{prefix}_relative_l2": _relative_l2_loss(prediction, target),
    }


def _batch_metrics(model, batch, model_name: str, config):
    import torch

    module = _unwrap(model)
    x, u, y, s = batch
    metrics = {}
    forward_loss = torch.zeros((), device=u.device)

    if model_name == "ifno":
        pred_u = _ifno_predict_input(model, batch)
        pred_s = _ifno_predict_output(model, batch)
        input_loss = _relative_l2_loss(pred_u, u)
        forward_loss = _relative_l2_loss(pred_s, s)
        total = input_loss + forward_loss
        metrics.update(_numeric_metrics(pred_s, s, "forward"))
    elif model_name == "invertible_deeponet":
        baseline_config = config.baselines.invertible_deeponet
        pred_u = module.predict_inverse(x, y, s)
        pred_s = module.forward_from_input(u, y)
        input_loss = _loss_by_name(pred_u, u, baseline_config.loss)
        forward_loss = _loss_by_name(pred_s, s, baseline_config.loss)
        total = (
            baseline_config.inverse_loss_weight * input_loss
            + baseline_config.forward_loss_weight * forward_loss
        )
        metrics.update(_numeric_metrics(pred_s, s, "forward"))
    elif model_name == "nio":
        baseline_config = config.baselines.nio
        pred_u = module.predict_inverse(x, y, s)
        input_loss = _loss_by_name(pred_u, u, baseline_config.loss)
        total = input_loss
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    metrics.update(_numeric_metrics(pred_u, u, "input"))
    metrics["loss"] = total
    metrics["input_loss"] = input_loss.detach()
    metrics["forward_loss"] = forward_loss.detach()
    return metrics


def _evaluate(model, loader, context, model_name: str, config):
    import torch

    model.eval()
    totals = {}
    count = torch.zeros((), device=context.device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if config.baselines.eval_batches is not None and batch_index >= config.baselines.eval_batches:
                break
            batch = _move_batch(batch, context.device)
            metrics = _batch_metrics(model, batch, model_name, config)
            for key, value in metrics.items():
                totals.setdefault(key, torch.zeros((), device=context.device))
                totals[key] += value.detach()
            count += 1
    for value in totals.values():
        _all_reduce_sum(value, context)
    _all_reduce_sum(count, context)
    count = torch.clamp(count, min=1)
    payload = {key: float((value / count).item()) for key, value in totals.items()}
    payload["eval_batches"] = int(count.item())
    return payload


def _train_vae_if_requested(model, train_loader, context, writer, baseline_config):
    import torch

    if not baseline_config.ifno.epochs_vae:
        return
    if context.is_distributed:
        raise RuntimeError("Distributed IFNO VAE pretraining is not ported yet.")
    vae_optimizer = torch.optim.AdamW(
        _unwrap(model).vae_net.parameters(),
        lr=baseline_config.ifno.lr_vae,
    )
    for epoch in range(baseline_config.ifno.epochs_vae):
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


_IFNO_KL_WEIGHT = 0.01


def _ifno_forward(model, batch, out_function_channels):
    import torch

    x, u, _, _ = batch
    result = model(torch.cat([x, u], dim=-1))
    if isinstance(result, tuple):
        pred_s_full, recon = result
    else:
        pred_s_full, recon = result, torch.zeros((), device=x.device)
    return pred_s_full[..., -out_function_channels:], recon


def _ifno_inverse(model, batch):
    import torch

    _, _, y, s = batch
    result = model.inverse(torch.cat([y, s], dim=-1))
    if isinstance(result, tuple):
        pred_u_full, recon = result
    else:
        pred_u_full, recon = result, torch.zeros((), device=y.device)
    return pred_u_full, recon


def _evaluate_ifno(model, loader, context, config, in_function_channels, out_function_channels):
    """Faithful iFNO eval: forward error, and inverse error using the VAE-refined
    inverse (paper's inference path), with the raw (no-VAE) inverse also reported."""
    import torch

    model.eval()
    totals = {}
    count = torch.zeros((), device=context.device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if config.baselines.eval_batches is not None and batch_index >= config.baselines.eval_batches:
                break
            batch = _move_batch(batch, context.device)
            _, u, _, s = batch
            batch_size = u.shape[0]

            pred_s, _ = _ifno_forward(model, batch, out_function_channels)
            pred_u_full, _ = _ifno_inverse(model, batch)
            pred_u_raw = pred_u_full[..., -in_function_channels:]

            refined = model.refine_inverse(pred_u_full)
            u_vae = model._get_vae_input_shape(u)

            forward_loss = _relative_l2_loss(pred_s, s)
            input_loss = _relative_l2_loss(
                refined.reshape(batch_size, -1), u_vae.reshape(batch_size, -1)
            )
            input_loss_raw = _relative_l2_loss(pred_u_raw, u)

            metrics = {
                "loss": forward_loss + input_loss,
                "forward_loss": forward_loss,
                "input_loss": input_loss,
                "input_loss_raw": input_loss_raw,
                "forward_relative_l2": forward_loss,
                "input_relative_l2": input_loss,
            }
            for key, value in metrics.items():
                totals.setdefault(key, torch.zeros((), device=context.device))
                totals[key] += value.detach()
            count += 1
    for value in totals.values():
        _all_reduce_sum(value, context)
    _all_reduce_sum(count, context)
    count = torch.clamp(count, min=1)
    payload = {key: float((value / count).item()) for key, value in totals.items()}
    payload["eval_batches"] = int(count.item())
    return payload


def _run_ifno_training(model, train_loader, test_loader, context, config, writer):
    """Faithful three-phase iFNO training (arXiv:2402.11722), mirroring the
    reference implementation (BayesianAIGroup/iFNO):

      Phase 1 - pre-train invertible blocks with forward + backward data losses
                plus the P/Q and P'/Q' reconstruction losses.
      Phase 2 - pre-train the beta-VAE on input functions (recon + KL).
      Phase 3 - joint training: forward data loss (lr_forward) and a backward
                objective that passes the rough inverse through the VAE and
                reconstructs the true input (lr_backward).

    Eval uses the VAE-refined inverse, matching the paper's inverse inference.
    """
    import torch

    if context.is_distributed:
        raise RuntimeError(
            "Faithful iFNO baseline training is single-process only; run without DDP."
        )

    baseline_config = config.baselines
    ifno_cfg = baseline_config.ifno
    out_fc = model.output_function_channels
    in_fc = model.input_function_channels
    device = context.device

    def log_scalar(tag, value, step):
        if context.is_rank_zero and writer is not None:
            writer.add_scalar(tag, float(value), step)

    global_step = 0

    # Phase 1: pre-train invertible blocks (forward + backward + reconstruction).
    if ifno_cfg.epochs_ifno:
        optimizer = torch.optim.AdamW(model.parameters(), lr=ifno_cfg.lr_ifno)
        for epoch in range(ifno_cfg.epochs_ifno):
            model.train()
            for batch in train_loader:
                batch = _move_batch(batch, device)
                _, u, _, s = batch
                pred_s, recon_f = _ifno_forward(model, batch, out_fc)
                pred_u_full, recon_b = _ifno_inverse(model, batch)
                pred_u = pred_u_full[..., -in_fc:]
                loss = (
                    _relative_l2_loss(pred_s, s)
                    + recon_f
                    + _relative_l2_loss(pred_u, u)
                    + recon_b
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                global_step += 1
                if global_step % baseline_config.log_interval == 0:
                    log_scalar("train/ifno_pretrain_loss", loss.item(), global_step)

    # Phase 2: pre-train the beta-VAE on input functions.
    if ifno_cfg.epochs_vae:
        optimizer = torch.optim.AdamW(model.vae_net.parameters(), lr=ifno_cfg.lr_vae)
        for epoch in range(ifno_cfg.epochs_vae):
            model.train()
            for batch in train_loader:
                batch = _move_batch(batch, device)
                _, u, _, _ = batch
                vae_in = model._get_vae_input_shape(u)
                recon, vae_input, mu, log_var = model.vae_net(vae_in)
                recon_loss = _relative_l2_loss(
                    recon.reshape(u.shape[0], -1), vae_input.reshape(u.shape[0], -1)
                )
                kl_loss = torch.mean(
                    -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
                )
                loss = recon_loss + _IFNO_KL_WEIGHT * kl_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                global_step += 1
                if global_step % baseline_config.log_interval == 0:
                    log_scalar("train/ifno_vae_loss", loss.item(), global_step)

    # Phase 3: joint training (forward data loss + VAE-refined backward objective).
    lr_backward = ifno_cfg.lr_backward if ifno_cfg.lr_backward is not None else ifno_cfg.lr_forward
    optimizer_forward = torch.optim.AdamW(model.parameters(), lr=ifno_cfg.lr_forward)
    optimizer_backward = torch.optim.AdamW(model.parameters(), lr=lr_backward)
    metrics = {}
    best_test = None
    max_steps = getattr(baseline_config, "max_steps", None)
    eval_interval = getattr(baseline_config, "eval_interval", None)
    for epoch in range(baseline_config.epochs):
        model.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            _, u, _, s = batch

            pred_s, _ = _ifno_forward(model, batch, out_fc)
            loss_forward = _relative_l2_loss(pred_s, s)
            optimizer_forward.zero_grad(set_to_none=True)
            loss_forward.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer_forward.step()

            pred_u_full, _ = _ifno_inverse(model, batch)
            loss_backward = model.inverse_vae_loss(pred_u_full, u, _IFNO_KL_WEIGHT)
            optimizer_backward.zero_grad(set_to_none=True)
            loss_backward.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer_backward.step()

            global_step += 1
            if global_step % baseline_config.log_interval == 0:
                log_scalar("train/ifno_joint_forward", loss_forward.item(), global_step)
                log_scalar("train/ifno_joint_backward", loss_backward.item(), global_step)

            should_eval = max_steps is not None and global_step >= max_steps
            if eval_interval:
                should_eval = should_eval or global_step % eval_interval == 0
            if should_eval:
                metrics = _evaluate_ifno(model, test_loader, context, config, in_fc, out_fc)
                best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
                if context.is_rank_zero and writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"eval/{key}", value, global_step)
            if max_steps is not None and global_step >= max_steps:
                break
        if max_steps is None:
            metrics = _evaluate_ifno(model, test_loader, context, config, in_fc, out_fc)
            best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
            if context.is_rank_zero and writer is not None:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"eval/{key}", value, global_step)
        if max_steps is not None and global_step >= max_steps:
            break

    if not metrics:
        metrics = _evaluate_ifno(model, test_loader, context, config, in_fc, out_fc)
        best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])

    return metrics, best_test, global_step


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
    tensorboard_root = args.tensorboard_dir or config.runtime.tensorboard_dir
    tensorboard_dir = (
        run_artifact_dir(
            Path(tensorboard_root).expanduser().resolve(),
            config.dataset.name,
            "baselines",
            args.model,
            args.seed,
        )
        if tensorboard_root
        else run_dir
    )

    if not args.execute:
        print("Dry run: baseline training will not execute.")
        print(f"model_dir: {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"run_dir:   {run_dir}")
        print(f"tensorboard_dir: {tensorboard_dir}")
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
    from inverse_neural_operator.runtime.sampling import RandomFunctionSampler

    context = setup_distributed(config.runtime.device)
    writer = None
    try:
        torch.manual_seed(args.seed + context.rank)
        train_dataset = _load_dataset(config, "train")
        test_dataset = _load_dataset(config, "test")
        dataset_info = train_dataset.get_info()

        accumulation_steps = max(1, baseline_config.gradient_accumulation_steps)
        if baseline_config.max_steps is not None and baseline_config.sample_with_replacement:
            train_sample_count = (
                baseline_config.max_steps
                * accumulation_steps
                * baseline_config.batch_size
            )
            train_sampler = RandomFunctionSampler(
                len(train_dataset),
                num_samples=train_sample_count,
                seed=args.seed + context.rank * 1_000_003,
            )
        else:
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
            shuffle=(
                train_sampler is None
                and not (
                    baseline_config.max_steps is not None
                    and baseline_config.sample_with_replacement
                )
            ),
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

        model = create_baseline_model(args.model, dataset_info, config).to(context.device)
        ddp_model = model
        if context.is_distributed:
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            )

        writer = SummaryWriter(log_dir=str(tensorboard_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"
        start_time = time.time()

        best_test = None
        metrics = {}
        completed_steps = 0

        if args.model == "ifno":
            metrics, best_test, completed_steps = _run_ifno_training(
                model, train_loader, test_loader, context, config, writer
            )
        else:
            _train_vae_if_requested(ddp_model, train_loader, context, writer, baseline_config)

            optimizer = torch.optim.AdamW(
                ddp_model.parameters(),
                lr=baseline_config.learning_rate,
                weight_decay=1e-4,
            )
            data_epoch = 0
            train_iter = iter(train_loader)

            def run_train_step(step: int):
                nonlocal train_iter, data_epoch
                ddp_model.train()
                optimizer.zero_grad(set_to_none=True)
                step_totals = {}
                for micro_step in range(accumulation_steps):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        data_epoch += 1
                        if isinstance(train_sampler, DistributedSampler):
                            train_sampler.set_epoch(data_epoch)
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    batch = _move_batch(batch, context.device)
                    should_sync = micro_step == accumulation_steps - 1
                    sync_context = (
                        nullcontext()
                        if should_sync or not hasattr(ddp_model, "no_sync")
                        else ddp_model.no_sync()
                    )
                    with sync_context:
                        batch_metrics = _batch_metrics(ddp_model, batch, args.model, config)
                        (batch_metrics["loss"] / accumulation_steps).backward()
                    for key, value in batch_metrics.items():
                        step_totals.setdefault(key, torch.zeros((), device=context.device))
                        step_totals[key] += value.detach() / accumulation_steps
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=2.0)
                optimizer.step()
                for value in step_totals.values():
                    _all_reduce_sum(value, context)
                    value /= context.world_size
                if context.is_rank_zero and writer is not None and step % baseline_config.log_interval == 0:
                    for key, value in step_totals.items():
                        writer.add_scalar(f"train/{key}", float(value.item()), step)
                return float(step_totals["loss"].item())

            if baseline_config.max_steps is not None:
                for step in range(1, baseline_config.max_steps + 1):
                    completed_steps = step
                    train_loss = run_train_step(step)
                    if step % baseline_config.eval_interval == 0 or step == baseline_config.max_steps:
                        metrics = _evaluate(ddp_model, test_loader, context, args.model, config)
                        best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
                        if context.is_rank_zero and writer is not None:
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    writer.add_scalar(f"eval/{key}", value, step)
                    if step % baseline_config.checkpoint_interval == 0 and context.is_rank_zero:
                        torch.save(
                            {
                                "step": step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": train_loss,
                                "best_test_loss": best_test,
                            },
                            checkpoint_path,
                        )
            else:
                step = 0
                for epoch in range(baseline_config.epochs):
                    if isinstance(train_sampler, DistributedSampler):
                        train_sampler.set_epoch(epoch)
                    for _ in range(len(train_loader)):
                        step += 1
                        completed_steps = step
                        train_loss = run_train_step(step)
                    metrics = _evaluate(ddp_model, test_loader, context, args.model, config)
                    best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])
                    if context.is_rank_zero and writer is not None:
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(f"eval/{key}", value, epoch)
                    if context.is_rank_zero:
                        torch.save(
                            {
                                "epoch": epoch,
                                "step": step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": train_loss,
                                "best_test_loss": best_test,
                            },
                            checkpoint_path,
                        )

            if not metrics:
                metrics = _evaluate(ddp_model, test_loader, context, args.model, config)
                best_test = metrics["loss"] if best_test is None else min(best_test, metrics["loss"])

        barrier(context)
        if context.is_rank_zero:
            model_dir.mkdir(parents=True, exist_ok=True)
            save_file(model.state_dict(), str(model_dir / "model.safetensors"))
            write_yaml(model_dir / "config.yaml", config.raw)
            metrics.update(
                {
                    "best_test_loss": best_test,
                    "steps": completed_steps,
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
                files={
                    "model": "model.safetensors",
                    "config": "config.yaml",
                    "manifest": "manifest.json",
                    "metrics": "metrics.json",
                },
                extra={"baseline_model": args.model},
            )
            print(f"Wrote baseline artifact to {model_dir}")
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
