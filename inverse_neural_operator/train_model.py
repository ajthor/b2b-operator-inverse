import argparse

import torch
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os

from b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)
from b2b.load_model import load_forward_model
from b2b.load_model import (
    load_function_encoders,
    load_function_encoder_params,
)

from data.load_dataset import load_dataset
from models.create_model import create_model

from utils.device import get_device, set_seed
from utils.params import save_params
from utils.checkpoints import setup_checkpoint_dir
from utils.args import load_defaults_from_yaml
from utils.imports import import_model_functions
from config.paths import get_runs_dir, get_model_dir, get_shared_dir

torch.set_float32_matmul_precision("high")

# Parse command line args
parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument("--dataset", type=str)

# Model args
parser.add_argument("--model", type=str)
parser.add_argument("--hidden_sizes", type=int, nargs="+")
parser.add_argument(
    "--forward_model",
    type=str,
    help="Forward model name for re-simulation loss (e.g., b2b_nonlinear, b2b_linear)",
)

# Training args
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--lambda_u", type=float)

# Path args
parser.add_argument(
    "--base_dir",
    type=str,
    default=None,
    help="Base directory for models/results/logs (overrides B2B_RESULTS_DIR / ./results fallback)",
)

# Device args
parser.add_argument("--device", type=str)

# Seed args
parser.add_argument("--seed", type=int)

# Checkpoint args
parser.add_argument("--checkpoint_interval", type=int)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--resume", type=bool)

# Load defaults from YAML
defaults_path = os.path.join(os.path.dirname(__file__), "train_model_defaults.yaml")
defaults = load_defaults_from_yaml(defaults_path)
parser.set_defaults(**defaults)

params = parser.parse_args()

device = get_device(params.device)
print(f"Using device: {device}")
set_seed(params.seed)

log_dir = str(
    get_runs_dir(
        params.dataset, params.model, params.seed, base_dir_override=params.base_dir
    )
)
model_dir = str(
    get_model_dir(
        params.dataset, params.model, params.seed, base_dir_override=params.base_dir
    )
)
shared_dir = str(
    get_shared_dir(params.dataset, params.seed, base_dir_override=params.base_dir)
)

# Create SummaryWriter
writer = SummaryWriter(log_dir=log_dir)
log_dir = writer.log_dir

# Save args to model directory
os.makedirs(model_dir, exist_ok=True)
save_params(params, model_dir)

# Create checkpoint directories
params.checkpoint_dir = setup_checkpoint_dir(params.checkpoint_dir, log_dir)

# Load dataset using utility
train_dataset = load_dataset(params.dataset, params, device, split="train")
test_dataset = load_dataset(params.dataset, params, device, split="test")
dataset_info = train_dataset.get_info()

# Get the appropriate train/save functions for the model
train_model, save_model = import_model_functions(params.model, "train", "save")

# Load function encoder parameters to get sizes for model creation
input_encoder_params, output_encoder_params = load_function_encoder_params(shared_dir)

# Create model with correct sizes (pass sizes for all models, some will ignore them)
model, optimizer = create_model(
    params.model,
    params,
    dataset_info,
    device,
    input_encoder_params.n_basis,  # input size (alpha coefficients)
    output_encoder_params.n_basis,  # output size (beta coefficients)
)

# Load function encoders (all models will receive them, some may not use them)
input_function_encoder, output_function_encoder = load_function_encoders(
    shared_dir, dataset_info, params, device
)

# Load forward model (all models will receive it, some may not use it)
try:
    forward_model = load_forward_model(
        shared_dir, device=device, forward_model_name=params.forward_model
    )
    print(f"Loaded forward model '{params.forward_model}' for re-simulation loss")
except Exception as e:
    print(f"Warning: Could not load forward model for re-simulation loss: {e}")
    forward_model = None

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

# Single consistent training function call for all models
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
    forward_model=forward_model,
)

# Save model
os.makedirs(model_dir, exist_ok=True)
save_model(model=model, path=os.path.join(model_dir, "model.safetensors"))
