#!/usr/bin/env python3
"""
Standalone IFNO training script for selected datasets (Darcy 1D, Burgers 1D, Parametric Heat 2D, Chladni 2D, Wave Scattering, FWI).
No function encoders required - IFNO works directly with raw data.
"""

import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# Import IFNO model
from inverse_neural_operator.models.ifno import (
    create_model,
    train as train_model,
    save as save_model,
    count_model_params,
)

def main():
    parser = argparse.ArgumentParser(description='Train IFNO on selected dataset')
    
    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="darcy_1d",
        choices=[
            "darcy_1d",
            "burgers_1d",
            "parametric_heat",
            "chladni_2d",
            "wave_scattering",
            "fwi",
        ],
        help="Which dataset to use",
    )
    
    # Model args  
    parser.add_argument("--model", type=str, default="ifno")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    
    # IFNO-specific training parameters
    parser.add_argument("--epochs_vae", type=int, default=50, help="VAE pretraining epochs")
    parser.add_argument("--epochs_ifno", type=int, default=100, help="IFNO pretraining epochs") 
    parser.add_argument("--lr_vae", type=float, default=1e-4, help="VAE learning rate")
    parser.add_argument("--lr_ifno", type=float, default=1e-3, help="IFNO pretraining learning rate")
    parser.add_argument("--lr_forward", type=float, default=1e-4, help="Joint training learning rate")
    
    # I/O args
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Log directory (defaults to ./logs/{dataset}_ifno_standalone/)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    # Device args
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda:1"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    
    # Resolve log directory
    if args.log_dir is None:
        args.log_dir = f"./logs/{args.dataset}_ifno_standalone/"
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Set checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Select dataset loader
    if args.dataset == "darcy_1d":
        from inverse_neural_operator.data.darcy_1d import load_data as load_data_fn
    elif args.dataset == "burgers_1d":
        from inverse_neural_operator.data.burgers_1d import load_data as load_data_fn
    elif args.dataset == "parametric_heat":
        from inverse_neural_operator.data.parametric_heat import load_data as load_data_fn
    elif args.dataset == "chladni_2d":
        from inverse_neural_operator.data.chladni_2d import load_data as load_data_fn
    elif args.dataset == "wave_scattering":
        from inverse_neural_operator.data.wave_scattering import load_data as load_data_fn
    elif args.dataset == "fwi":
        from inverse_neural_operator.data.fwi_data import load_data as load_data_fn
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    train_dataset = load_data_fn(args, device=device, split="train")
    test_dataset = load_data_fn(args, device=device, split="test")
    dataset_info = train_dataset.get_info()
    
    print(f"Dataset info: {dataset_info}")
    
    # Create IFNO model
    print("Creating IFNO model...")
    model = create_model(
        input_size=None,  # Not used by IFNO
        hidden_sizes=[256, 256, 256],  # Not used by IFNO
        n_coupling_layers=2,
        modes1=8,
        modes2=8, 
        width=8,
        beta=2.0,
        n_layers=2,
        padding=20,
        vae_latent_dim=8,
        intermediate_dim=32,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    print(f"Train batches: {len(train_dataloader)}, Test batches: {len(test_dataloader)}")
    
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
        resume_from_checkpoint=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        device=device,
        epochs_vae=args.epochs_vae,
        epochs_ifno=args.epochs_ifno,
        lr_vae=args.lr_vae,
        lr_ifno=args.lr_ifno,
        lr_forward=args.lr_forward,
    )
    
    # Save model
    model_path = os.path.join(args.log_dir, "ifno_model.pth")
    save_model(model=model, path=model_path)
    print(f"Model saved to: {model_path}")
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()