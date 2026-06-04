"""DDP-ready function encoder training entrypoint."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def _move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _loss(model, batch):
    import torch

    example_xs, example_ys, xs, ys = batch
    encoder = model.module if hasattr(model, "module") else model
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
    from inverse_neural_operator.data.fwi_hf import load_fwi_dataset

    if config.dataset.name != "fwi":
        raise ValueError(
            "The new DDP function encoder path currently supports dataset=fwi only."
        )
    return load_fwi_dataset(
        split=split,
        source=config.dataset.source or "ajthor/fwi",
        sample_limit=config.dataset.sample_limit,
    )


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

    if not args.execute:
        print("Dry run: function encoder training will not execute.")
        print(f"model_dir: {model_dir or '<B2B_MODELS_DIR unset>'}")
        print(f"run_dir:   {run_dir}")
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
            shuffle=train_sampler is None,
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
        writer = SummaryWriter(log_dir=str(run_dir)) if context.is_rank_zero else None
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "latest_checkpoint.pt"

        best_test = None
        start_time = time.time()
        for epoch in range(fe_config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            train_total = torch.zeros((), device=context.device)
            train_count = 0
            for batch in train_loader:
                batch = _move_batch(batch, context.device)
                optimizer.zero_grad(set_to_none=True)
                loss, pred_loss, norm_loss = _loss(model, batch)
                loss.backward()
                optimizer.step()
                train_total += loss.detach()
                train_count += 1

            mean_train = train_total / max(train_count, 1)
            mean_train = reduce_mean(mean_train, context)

            model.eval()
            test_total = torch.zeros((), device=context.device)
            test_count = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = _move_batch(batch, context.device)
                    loss, _, _ = _loss(model, batch)
                    test_total += loss.detach()
                    test_count += 1
            mean_test = test_total / max(test_count, 1)
            mean_test = reduce_mean(mean_test, context)
            best_test = (
                float(mean_test.item())
                if best_test is None
                else min(best_test, float(mean_test.item()))
            )

            if context.is_rank_zero:
                assert writer is not None
                writer.add_scalar(f"loss_train/{args.encoder_type}", mean_train.item(), epoch)
                writer.add_scalar(f"loss_test/{args.encoder_type}", mean_test.item(), epoch)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            model.module.state_dict()
                            if hasattr(model, "module")
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": mean_test.item(),
                    },
                    checkpoint_path,
                )

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
                "epochs": fe_config.epochs,
                "elapsed_seconds": time.time() - start_time,
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
