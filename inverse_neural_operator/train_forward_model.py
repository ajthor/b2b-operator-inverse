import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os

from inverse_neural_operator.utils.device import get_device, set_seed
from inverse_neural_operator.utils.params import save_params
from inverse_neural_operator.utils.checkpoints import setup_checkpoint_dir
from inverse_neural_operator.utils.args import load_defaults_from_yaml
from inverse_neural_operator.utils.imports import import_model_functions
from inverse_neural_operator.data.load_dataset import load_dataset
from inverse_neural_operator.b2b.load_model import load_function_encoder_params, load_function_encoders
from inverse_neural_operator.b2b.create_model import create_forward_model

torch.set_float32_matmul_precision("high")

# Parse command line args
parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument("--dataset", type=str)

# Model args
parser.add_argument("--model", type=str)
parser.add_argument("--hidden_sizes", type=int, nargs="+")

# Training args
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--lambda_u", type=float)

# SummaryWriter args
parser.add_argument("--log_dir", type=str)

# Device args
parser.add_argument("--device", type=str)

# Seed args
parser.add_argument("--seed", type=int)

# Checkpoint args
parser.add_argument("--checkpoint_interval", type=int)
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--resume", type=bool)

# Load defaults from YAML
defaults_path = os.path.join(
    os.path.dirname(__file__), "train_forward_model_defaults.yaml"
)
defaults = load_defaults_from_yaml(defaults_path)
parser.set_defaults(**defaults)

params = parser.parse_args()

device = get_device(params.device)
print(f"Using device: {device}")
set_seed(params.seed)

# Create SummaryWriter
writer = SummaryWriter(log_dir=params.log_dir)
log_dir = writer.log_dir

# Save args
save_params(params, log_dir)

# Create checkpoint directories
params.checkpoint_dir = setup_checkpoint_dir(params.checkpoint_dir, log_dir)

# Load dataset using utility
train_dataset = load_dataset(params.dataset, params, device, split="train")
test_dataset = load_dataset(params.dataset, params, device, split="test")
dataset_info = train_dataset.get_info()
# Load function encoder parameters to get sizes for model creation
input_encoder_params, output_encoder_params = load_function_encoder_params(log_dir)

# Get the appropriate train/save functions based on model type
train_model, save_model = import_model_functions(params.model, "train", "save")

# Create forward model and optimizer
model, optimizer = create_forward_model(
    params.model,
    params,
    input_encoder_params.n_basis,  # input size (alpha coefficients)
    output_encoder_params.n_basis,  # output size (beta coefficients)
    device,
)

# Load function encoders (forward models need them for training)
input_function_encoder, output_function_encoder = load_function_encoders(
    log_dir, dataset_info, params, device
)

# Train forward model

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

# Save forward model with model-specific name in shared directory

save_model(model=model, path=os.path.join(log_dir, f"forward_{params.model}.pth"))
