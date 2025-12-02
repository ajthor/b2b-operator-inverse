#!/usr/bin/env python3
"""
Standalone IFNO training script for selected datasets (Darcy 1D, Burgers 1D, Parametric Heat 2D, Chladni 2D, Wave Scattering, FWI).
No function encoders required - IFNO works directly with raw data.
"""

import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import os
import json
import sys

# Import IFNO model (unconditional)
from models.ifno import (
    create_model,
    train as train_model,
    save as save_model,
    count_model_params,
)
from config.paths import get_model_dir, get_runs_dir
from utils.params import save_params
from utils.device import set_seed, dataset_on_cpu
from utils.distributed import (
    init_distributed_mode,
    cleanup_distributed,
    is_main_process,
    NullSummaryWriter,
)


def main():
    parser = argparse.ArgumentParser(description="Train IFNO on selected dataset")

    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "darcy_1d",
            "burgers_1d",
            "parametric_heat",
            "chladni_2d",
            "wave_scattering",
            "fwi",
            "elastic_plate",
        ],
        help="Which dataset to use",
    )

    # Model args
    parser.add_argument("--model", type=str)

    # Training args
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total number of joint-training epochs (phase 3).",
    )
    parser.add_argument("--learning_rate", type=float)

    # IFNO-specific training parameters (following paper recommendations)
    parser.add_argument(
        "--epochs_vae",
        type=int,
        help="VAE pretraining epochs (paper: {50,100,200})",
    )
    parser.add_argument(
        "--epochs_ifno",
        type=int,
        help="IFNO pretraining epochs (paper: {100,200,500})",
    )
    parser.add_argument("--lr_vae", type=float, help="VAE learning rate")
    parser.add_argument(
        "--lr_ifno",
        type=float,
        help="IFNO pretraining learning rate (paper: 5e-3)",
    )
    parser.add_argument(
        "--lr_forward", type=float, help="Joint training learning rate"
    )
    parser.add_argument(
        "--lr_backward",
        type=float,
        default=None,
        help="Joint training backward learning rate (defaults to 0.5 * lr_forward)",
    )

    # IFNO architecture parameters (following paper recommendations)
    parser.add_argument(
        "--n_layers",
        type=int,
        help="Number of invertible Fourier blocks (paper: {1,2,3,4})",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Channel lifting dimension (paper: {32,64,128})",
    )
    parser.add_argument(
        "--modes",
        type=int,
        help="Number of Fourier modes (paper: {8,12,16,32})",
    )
    parser.add_argument(
        "--vae_latent_dim", type=int, help="VAE latent dimension"
    )
    parser.add_argument(
        "--beta", type=float, help="Beta parameter for softplus activation"
    )

    # I/O args
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory for models/results/logs (overrides B2B_RESULTS_DIR / ./results fallback)",
    )
    parser.add_argument("--checkpoint_interval", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Device args
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Batches to prefetch per worker (requires num_workers>0)",
    )
    parser.add_argument(
        "--dataset_device",
        type=str,
        default="cpu",
        help='Location for dataset tensors ("cpu", "cuda", or "same" to match training device)',
    )

    # Load defaults from YAML
    defaults_path = os.path.join(
        os.path.dirname(__file__), "train_ifno_defaults.yaml"
    )
    defaults = load_defaults_from_yaml(defaults_path)
    parser.set_defaults(**defaults)

    args = parser.parse_args()

    dist_config = init_distributed_mode(args.device)
    device = dist_config.device
    use_cuda = isinstance(device, str) and device.startswith("cuda")
    if is_main_process():
        print(
            f"Using device: {device} "
            f"(distributed={dist_config.is_distributed}, world_size={dist_config.world_size})"
        )
    set_seed(args.seed + dist_config.rank)

    dataset_device = args.dataset_device or "cpu"
    if dataset_device in ("same", "match"):
        dataset_device = device

    # Construct paths (CLI base_dir overrides env → YAML defaults)
    log_dir = str(
        get_runs_dir(args.dataset, "ifno", args.seed, base_dir_override=args.base_dir)
    )
    model_dir = str(
        get_model_dir(args.dataset, "ifno", args.seed, base_dir_override=args.base_dir)
    )

    # Create directories
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    # Checkpoints saved in runs (log_dir) alongside TensorBoard events
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_main_process():
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs -> {log_dir}")
        print(f"Model directory -> {model_dir}")
        print(f"Checkpoints -> {checkpoint_dir}")
    else:
        writer = NullSummaryWriter()

    # Save args to model directory
    if is_main_process():
        save_params(args, model_dir)

    # Select dataset loader
    if args.dataset == "darcy_1d":
        from data.darcy_1d import load_data as load_data_fn
    elif args.dataset == "burgers_1d":
        from data.burgers_1d import load_data as load_data_fn
    elif args.dataset == "parametric_heat":
        from data.parametric_heat import (
            load_data as load_data_fn,
        )
    elif args.dataset == "chladni_2d":
        from data.chladni_2d import load_data as load_data_fn
    elif args.dataset == "wave_scattering":
        from data.wave_scattering import (
            load_data as load_data_fn,
        )
    elif args.dataset == "fwi":
        from data.fwi_data import load_data as load_data_fn
    elif args.dataset == "elastic_plate":
        from data.elastic_plate import load_data as load_data_fn
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    train_dataset = load_data_fn(args, device=dataset_device, split="train")
    test_dataset = load_data_fn(args, device=dataset_device, split="test")
    dataset_info = train_dataset.get_info()

    train_pin_memory = use_cuda and dataset_on_cpu(train_dataset)
    test_pin_memory = use_cuda and dataset_on_cpu(test_dataset)

    effective_num_workers = args.num_workers
    if not train_pin_memory and effective_num_workers > 0:
        if is_main_process():
            print(
                "Dataset tensors already on accelerator; forcing num_workers=0 to avoid CUDA access in worker processes."
            )
        effective_num_workers = 0
    persistent_workers = effective_num_workers > 0

    print(f"Dataset info: {dataset_info}")
    if is_main_process():
        writer.add_text("setup/dataset", args.dataset)
        writer.add_text("setup/dataset_info", json.dumps(dataset_info, indent=2))
        writer.add_text("setup/hyperparameters", json.dumps(vars(args), indent=2))

    # Create IFNO model with paper-recommended hyperparameters
    print("Creating IFNO model...")
    model = create_model(
        input_size=None,  # Not used by IFNO
        hidden_sizes=[256, 256, 256],  # Not used by IFNO
        n_coupling_layers=2,  # Not used by IFNO
        modes1=args.modes,  # Use paper-recommended values
        modes2=args.modes,  # Use paper-recommended values
        width=args.width,  # Use paper-recommended values
        beta=args.beta,
        n_layers=args.n_layers,  # Use paper-recommended values
        padding=20,
        vae_latent_dim=args.vae_latent_dim,
        intermediate_dim=args.width // 2,  # Scale with width
        # IFNO-specific parameters from dataset info
        input_spatial_dims=dataset_info["input_spatial_dims"],
        output_spatial_dims=dataset_info["output_spatial_dims"],
        input_function_channels=dataset_info["input_function_channels"],
        output_function_channels=dataset_info["output_function_channels"],
        coordinate_dim=dataset_info["coordinate_dim"],
    ).to(device)

    # Use complex-aware parameter counting (complex params counted as 2)
    print(f"Model created with {count_model_params(model)} parameters")

    # Create optimizer
    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if dist_config.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_config.local_rank] if use_cuda else None,
            output_device=dist_config.local_rank if use_cuda else None,
            broadcast_buffers=False,
        )

    # Create data loaders
    print("Creating dataloaders...")
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=dist_config.world_size,
            rank=dist_config.rank,
            shuffle=True,
        )
        if dist_config.is_distributed
        else None
    )
    test_sampler = (
        DistributedSampler(
            test_dataset,
            num_replicas=dist_config.world_size,
            rank=dist_config.rank,
            shuffle=False,
        )
        if dist_config.is_distributed
        else None
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=effective_num_workers,
        pin_memory=train_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=effective_num_workers,
        pin_memory=test_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    print(
        f"Train batches: {len(train_dataloader)}, Test batches: {len(test_dataloader)}"
    )

    # Train model (function encoders are unused placeholders)
    print("Starting IFNO training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        input_function_encoder=None,  # Unused placeholder
        output_function_encoder=None,  # Unused placeholder
        n_epochs=args.epochs,
        summary_writer=writer,
        model_name=args.model,
        params=args,
        forward_model=None,  # Unused placeholder for IFNO
        resume_from_checkpoint=args.resume,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        device=device,
        epochs_vae=args.epochs_vae,
        epochs_ifno=args.epochs_ifno,
        lr_vae=args.lr_vae,
        lr_ifno=args.lr_ifno,
        lr_forward=args.lr_forward,
        lr_backward=args.lr_backward,
    )

    # Save model
    if is_main_process():
        model_path = os.path.join(model_dir, "ifno_model.safetensors")
        model_to_save = model.module if dist_config.is_distributed else model
        save_model(model=model_to_save, path=model_path)
        print(f"Model saved to: {model_path}")
        writer.close()
        print("Training completed!")
    else:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
