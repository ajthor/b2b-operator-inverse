"""DDP-ready function encoder training entrypoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path


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
    parser = argparse.ArgumentParser(description="Train a function encoder.")
    parser.add_argument("--config", required=True, help="Experiment YAML.")
    parser.add_argument(
        "--encoder-type", choices=["input", "output"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually train. Without this flag the command only prints paths.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest recovery checkpoint in the run directory.",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Optional root for TensorBoard logs. Checkpoints still use --results-dir.",
    )
    return parser.parse_args()


def _move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _loss(model, batch, *, coefficient_grad: bool):
    import torch

    example_xs, example_ys, xs, ys = batch
    encoder = model.module if hasattr(model, "module") else model
    if coefficient_grad:
        coefficients, gram = encoder.compute_coefficients(example_xs, example_ys)
    else:
        with torch.no_grad():
            coefficients, gram = encoder.compute_coefficients(example_xs, example_ys)
    pred = model(xs, coefficients)
    pred_loss = torch.nn.functional.mse_loss(pred, ys)
    try:
        from function_encoder.losses import basis_orthonormality_loss

        norm_loss = basis_orthonormality_loss(gram, device=ys.device)
    except Exception:
        norm_loss = torch.zeros((), device=ys.device)
    return pred_loss + norm_loss, pred_loss.detach(), norm_loss.detach()


def _load_dataset(config, split: str):
    from inverse_neural_operator.data.overhaul import load_overhaul_dataset

    return load_overhaul_dataset(config, split)


class RandomFunctionSampler:
    """Uniform random function sampler for step-budget training."""

    def __init__(self, dataset_size: int, num_samples: int, seed: int):
        self.dataset_size = dataset_size
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for _ in range(self.num_samples):
            yield int(
                torch.randint(
                    low=0,
                    high=self.dataset_size,
                    size=(1,),
                    generator=generator,
                ).item()
            )

    def __len__(self):
        return self.num_samples


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    fe_config = config.function_encoders

    try:
        model_root = models_root(required=args.execute, override=args.models_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result_root = results_root(args.results_dir)
    model_dir = (
        model_artifact_dir(
            model_root,
            config.dataset.name,
            "function_encoders",
            fe_config.artifact,
            args.seed,
        )
        if model_root is not None
        else None
    )
    run_dir = (
        run_artifact_dir(
            result_root,
            config.dataset.name,
            "function_encoders",
            fe_config.artifact,
            args.seed,
        )
        / args.encoder_type
    )
    tensorboard_root = args.tensorboard_dir or config.runtime.tensorboard_dir
    tensorboard_dir = (
        run_artifact_dir(
            Path(tensorboard_root).expanduser().resolve(),
            config.dataset.name,
            "function_encoders",
            fe_config.artifact,
            args.seed,
        )
        / args.encoder_type
        if tensorboard_root
        else run_dir
    )

    if not args.execute:
        print("Dry run: function encoder training will not execute.")
        print(f"model_dir: {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"run_dir:   {run_dir}")
        print(f"tensorboard_dir: {tensorboard_dir}")
        return

    import torch
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    from inverse_neural_operator.data.process_data import (
        InputFunctionEncoderDataset,
        OutputFunctionEncoderDataset,
    )
    from inverse_neural_operator.function_encoders.build import (
        create_function_encoder,
        memory_efficient_inner_product,
    )
    from inverse_neural_operator.runtime.distributed import (
        barrier,
        cleanup_distributed,
        reduce_mean,
        setup_distributed,
    )

    context = setup_distributed(config.runtime.device)
    try:
        torch.manual_seed(args.seed + context.rank)
        train_base = _load_dataset(config, "train")
        test_base = _load_dataset(config, "test")
        if hasattr(train_base, "get_info"):
            dataset_info = train_base.get_info()
        elif hasattr(train_base, "dataset") and hasattr(train_base.dataset, "get_info"):
            dataset_info = train_base.dataset.get_info()
        else:
            raise AttributeError("FWI dataset wrapper does not expose get_info().")

        if args.encoder_type == "input":
            train_dataset = InputFunctionEncoderDataset(train_base, device="cpu")
            test_dataset = InputFunctionEncoderDataset(test_base, device="cpu")
            input_size = dataset_info["X_size"]
            output_size = dataset_info["u_size"]
            weights_name = "input_encoder.safetensors"
        else:
            train_dataset = OutputFunctionEncoderDataset(train_base, device="cpu")
            test_dataset = OutputFunctionEncoderDataset(test_base, device="cpu")
            input_size = dataset_info["Y_size"]
            output_size = dataset_info["s_size"]
            weights_name = "output_encoder.safetensors"

        accumulation_steps = max(1, fe_config.gradient_accumulation_steps)
        if fe_config.max_steps is not None and fe_config.sample_with_replacement:
            train_sample_count = (
                fe_config.max_steps
                * accumulation_steps
                * fe_config.batch_size
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
            batch_size=fe_config.batch_size,
            shuffle=(
                train_sampler is None
                and not (
                    fe_config.max_steps is not None
                    and fe_config.sample_with_replacement
                )
            ),
            sampler=train_sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=fe_config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory and context.device.type == "cuda",
            persistent_workers=config.runtime.num_workers > 0,
        )

        model = create_function_encoder(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=fe_config.basis.hidden_sizes,
            n_basis=fe_config.basis.n_basis,
            basis_kind=fe_config.basis.kind,
            activation=fe_config.basis.activation,
            omega_0=fe_config.basis.omega_0,
            regularization=fe_config.regularization,
            basis_chunk_size=fe_config.basis_chunk_size,
            inner_product=(
                memory_efficient_inner_product
                if config.dataset.name == "fwi"
                else None
            ),
        ).to(context.device)
        if context.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[context.local_rank] if context.device.type == "cuda" else None
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=fe_config.learning_rate)
        writer = SummaryWriter(log_dir=str(tensorboard_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"

        best_test = None
        start_epoch = 0
        start_step = 0
        if args.resume:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume because checkpoint is missing: {checkpoint_path}"
                )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=context.device,
                weights_only=False,
            )
            target_model = model.module if hasattr(model, "module") else model
            target_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            start_step = int(checkpoint.get("step", -1)) + 1
            best_test = checkpoint.get("best_test_loss")

        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        epoch_seconds = []
        completed_steps = start_step
        step_seconds = []
        start_time = time.time()

        def save_checkpoint(epoch, step, loss_value):
            if not context.is_rank_zero:
                return
            torch.save(
                {
                    "epoch": epoch,
                    "step": step,
                    "model_state_dict": (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_value,
                    "best_test_loss": best_test,
                },
                checkpoint_path,
            )

        def evaluate(step_or_epoch):
            model.eval()
            test_total = torch.zeros((), device=context.device)
            test_count = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(test_loader):
                    if (
                        fe_config.eval_batches is not None
                        and batch_index >= fe_config.eval_batches
                    ):
                        break
                    batch = _move_batch(batch, context.device)
                    loss, _, _ = _loss(
                        model,
                        batch,
                        coefficient_grad=fe_config.coefficient_grad,
                    )
                    test_total += loss.detach()
                    test_count += 1
            mean_test = test_total / max(test_count, 1)
            mean_test = reduce_mean(mean_test, context)
            return mean_test, test_count

        if fe_config.max_steps is not None:
            data_epoch = 0
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            for step in range(start_step, fe_config.max_steps):
                step_start = time.time()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                step_loss_total = torch.zeros((), device=context.device)
                step_pred_total = torch.zeros((), device=context.device)
                step_norm_total = torch.zeros((), device=context.device)
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
                        if should_sync or not hasattr(model, "no_sync")
                        else model.no_sync()
                    )
                    with sync_context:
                        loss, pred_loss, norm_loss = _loss(
                            model,
                            batch,
                            coefficient_grad=fe_config.coefficient_grad,
                        )
                        (loss / accumulation_steps).backward()
                    step_loss_total += loss.detach()
                    step_pred_total += pred_loss.detach()
                    step_norm_total += norm_loss.detach()
                optimizer.step()
                completed_steps = step + 1
                step_seconds.append(time.time() - step_start)
                mean_step_loss = reduce_mean(
                    step_loss_total / accumulation_steps,
                    context,
                )
                mean_step_pred = reduce_mean(
                    step_pred_total / accumulation_steps,
                    context,
                )
                mean_step_norm = reduce_mean(
                    step_norm_total / accumulation_steps,
                    context,
                )

                if (
                    context.is_rank_zero
                    and fe_config.log_interval > 0
                    and (completed_steps == 1 or completed_steps % fe_config.log_interval == 0)
                ):
                    assert writer is not None
                    writer.add_scalar(
                        f"loss_train/{args.encoder_type}",
                        mean_step_loss.item(),
                        completed_steps,
                    )
                    writer.add_scalar(
                        f"loss_train_pred/{args.encoder_type}",
                        mean_step_pred.item(),
                        completed_steps,
                    )
                    writer.add_scalar(
                        f"loss_train_norm/{args.encoder_type}",
                        mean_step_norm.item(),
                        completed_steps,
                    )

                should_evaluate = (
                    fe_config.eval_interval > 0
                    and (
                        completed_steps % fe_config.eval_interval == 0
                        or completed_steps == fe_config.max_steps
                    )
                )
                should_checkpoint = (
                    fe_config.checkpoint_interval > 0
                    and completed_steps % fe_config.checkpoint_interval == 0
                )
                if should_evaluate:
                    mean_test, test_count = evaluate(completed_steps)
                    best_test = (
                        float(mean_test.item())
                        if best_test is None
                        else min(best_test, float(mean_test.item()))
                    )
                    if context.is_rank_zero:
                        assert writer is not None
                        writer.add_scalar(
                            f"loss_test/{args.encoder_type}",
                            mean_test.item(),
                            completed_steps,
                        )
                        writer.add_scalar(
                            f"eval_batches/{args.encoder_type}",
                            test_count,
                            completed_steps,
                        )
                    save_checkpoint(data_epoch, completed_steps, mean_test.item())
                elif should_checkpoint:
                    save_checkpoint(data_epoch, completed_steps, mean_step_loss.item())
        else:
            for epoch in range(start_epoch, fe_config.epochs):
                epoch_start = time.time()
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                model.train()
                train_total = torch.zeros((), device=context.device)
                train_count = 0
                for batch in train_loader:
                    batch = _move_batch(batch, context.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, pred_loss, norm_loss = _loss(
                        model,
                        batch,
                        coefficient_grad=fe_config.coefficient_grad,
                    )
                    loss.backward()
                    optimizer.step()
                    train_total += loss.detach()
                    train_count += 1
                    completed_steps += 1

                mean_train = train_total / max(train_count, 1)
                mean_train = reduce_mean(mean_train, context)

                mean_test, test_count = evaluate(epoch)
                best_test = (
                    float(mean_test.item())
                    if best_test is None
                    else min(best_test, float(mean_test.item()))
                )
                epoch_seconds.append(time.time() - epoch_start)

                if context.is_rank_zero:
                    assert writer is not None
                    writer.add_scalar(
                        f"loss_train/{args.encoder_type}",
                        mean_train.item(),
                        epoch,
                    )
                    writer.add_scalar(f"loss_test/{args.encoder_type}", mean_test.item(), epoch)
                    writer.add_scalar(f"eval_batches/{args.encoder_type}", test_count, epoch)
                save_checkpoint(epoch, completed_steps, mean_test.item())

        barrier(context)
        if context.is_rank_zero:
            assert model_dir is not None
            model_dir.mkdir(parents=True, exist_ok=True)
            state_dict = (
                model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            )
            save_file(state_dict, str(model_dir / weights_name))
            write_yaml(model_dir / "config.yaml", config.raw)

            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics_payload = json.load(handle)
            else:
                metrics_payload = {}
            metrics_payload[args.encoder_type] = {
                "best_test_loss": best_test,
                "training_mode": "steps" if fe_config.max_steps is not None else "epochs",
                "epochs": fe_config.epochs,
                "max_steps": fe_config.max_steps,
                "completed_steps": completed_steps,
                "start_epoch": start_epoch,
                "start_step": start_step,
                "resumed": args.resume,
                "elapsed_seconds": time.time() - start_time,
                "epoch_seconds": epoch_seconds,
                "average_step_seconds": (
                    sum(step_seconds) / len(step_seconds)
                    if step_seconds
                    else None
                ),
                "eval_interval": fe_config.eval_interval,
                "eval_batches": fe_config.eval_batches,
                "log_interval": fe_config.log_interval,
                "tensorboard_dir": str(tensorboard_dir),
                "world_size": context.world_size,
                "batch_size_per_rank": fe_config.batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_functions_per_step": (
                    fe_config.batch_size * accumulation_steps * context.world_size
                ),
                "sample_with_replacement": fe_config.sample_with_replacement,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "n_basis": fe_config.basis.n_basis,
                "basis_kind": fe_config.basis.kind,
                "basis_chunk_size": fe_config.basis_chunk_size,
                "coefficient_grad": fe_config.coefficient_grad,
                "rank0_device": str(context.device),
                "rank0_device_name": (
                    torch.cuda.get_device_name(context.device)
                    if context.device.type == "cuda"
                    else "cpu"
                ),
                "rank0_peak_cuda_memory_bytes": (
                    torch.cuda.max_memory_allocated(context.device)
                    if context.device.type == "cuda"
                    else None
                ),
            }
            write_json(
                metrics_path,
                metrics_payload,
            )

            manifest_path = model_dir / "manifest.json"
            files = {}
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as handle:
                    existing_manifest = json.load(handle)
                files.update(existing_manifest.get("files", {}))
            files[args.encoder_type] = weights_name
            write_manifest(
                model_dir,
                artifact_type="function_encoders",
                dataset=config.dataset.name,
                name=fe_config.artifact,
                seed=args.seed,
                files=files,
                extra={"encoder_types": sorted(files.keys())},
            )
            if writer is not None:
                writer.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
