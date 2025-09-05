import argparse

import torch
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os

from models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

torch.set_float32_matmul_precision("high")

# Parse command line args

parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument("--dataset", type=str, default="burgers_1d")

# Model args
parser.add_argument("--model", type=str, default="b2b_nonlinear")
parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])

# DeepONet specific args
parser.add_argument("--branch_hidden_sizes", type=int, nargs="+", default=[256, 256])
parser.add_argument("--trunk_hidden_sizes", type=int, nargs="+", default=[256, 256])
parser.add_argument("--trunk_input_size", type=int, default=1)
parser.add_argument("--output_channels", type=int, default=1)


# Training args
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--lambda_u", type=float, default=0.0)

# SummaryWriter args
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse/burgers_1d/b2b_nonlinear/seed_1/",
)
parser.add_argument("--comment", type=str, default="")

# Device args
parser.add_argument("--device", type=str, default=None)

# Seed args
parser.add_argument("--seed", type=int, default=42)

# Checkpoint args
parser.add_argument("--checkpoint_interval", type=int, default=100)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--resume", type=bool, default=False)

params = parser.parse_args()


if params.device is None:
    if torch.cuda.is_available():
        device = "cuda:5"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = params.device

print(f"Using device: {device}")
torch.manual_seed(params.seed)

# Create SummaryWriter
writer = SummaryWriter(log_dir=params.log_dir, comment=params.comment)
log_dir = writer.log_dir

# Save args
with open(f"{log_dir}/params.txt", "w") as f:
    f.write(str(params))

torch.save(params, f"{log_dir}/params.pth")

# Create checkpoint directories
if params.checkpoint_dir is None:
    params.checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(params.checkpoint_dir, exist_ok=True)

# Load dataset using utility
from data.load_dataset import load_dataset

train_dataset = load_dataset(params.dataset, params, device, split="train")
test_dataset = load_dataset(params.dataset, params, device, split="test")
dataset_info = train_dataset.get_info()

# Create model using utility
from models.create_model import create_model

# Get the appropriate train/save functions for the model
match params.model:
    case "b2b_linear":
        from models.b2b_operator_linear import (
            train as train_model,
            save as save_model,
        )
    case "b2b_nonlinear":
        from models.b2b_operator_nonlinear import (
            train as train_model,
            save as save_model,
        )
    case "b2b_nonlinear_fwd":
        from models.b2b_operator_nonlinear_fwd import (
            train as train_model,
            save as save_model,
        )
    case "deeponet":
        from models.deeponet import (
            train as train_model,
            save as save_model,
        )
    case "variational_autoencoder":
        from models.variational_autoencoder import (
            train as train_model,
            save as save_model,
        )
    case "invertible_network":
        from models.invertible_network import (
            train as train_model,
            save as save_model,
        )
    case "realnvp":
        from models.realnvp import (
            train as train_model,
            save as save_model,
        )
    case "ifno":
        from models.ifno import (
            train as train_model,
            save as save_model,
        )
    case _:
        raise ValueError(f"Unknown model: {params.model}")

# Load function encoders and parameters for models that need them
if params.model.startswith("b2b") or params.model in ["variational_autoencoder", "invertible_network", "realnvp", "ifno"]:
    from models.load_model import load_function_encoders, load_function_encoder_params
    
    # Load function encoder parameters to get sizes for model creation
    input_encoder_params, output_encoder_params = load_function_encoder_params(log_dir)
    
    # Create model with correct sizes
    model, optimizer = create_model(
        params.model, 
        params, 
        dataset_info, 
        device, 
        input_encoder_params.n_basis,  # input size (alpha coefficients)
        output_encoder_params.n_basis   # output size (beta coefficients)
    )
    
    # Load function encoders
    input_function_encoder, output_function_encoder = load_function_encoders(
        log_dir, dataset_info, params, device
    )
else:
    # For models that don't need function encoders (like deeponet)
    model, optimizer = create_model(params.model, params, dataset_info, device)
    input_function_encoder = None
    output_function_encoder = None

# Train model

train_dataloader = DataLoader(
    train_dataset,
    batch_size=params.batch_size,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=params.batch_size,
    shuffle=True,
)

train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=params.epochs,
    summary_writer=writer,
    model_name=params.model,
    params=params,
    resume_from_checkpoint=params.resume,
    checkpoint_dir=params.checkpoint_dir,
    checkpoint_interval=params.checkpoint_interval,
    device=device,
)

# Save model

save_model(model=model, path=os.path.join(log_dir, "model.pth"))
