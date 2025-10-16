import argparse
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)

from b2b.function_encoder import (
    create_model as create_function_encoder,
    train as train_function_encoder,
    save as save_function_encoder,
    memory_efficient_inner_product,
)
from utils.device import get_device, set_seed
from utils.params import save_params
from utils.checkpoints import setup_checkpoint_dir
from utils.args import load_defaults_from_yaml
from data.load_dataset import load_dataset

torch.set_float32_matmul_precision("high")

# Parse command line args
parser = argparse.ArgumentParser()

# Encoder type arg
parser.add_argument("--encoder_type", type=str, choices=["input", "output"])

# Dataset args
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=str)

# Function encoder args
parser.add_argument("--n_basis", type=int)
parser.add_argument("--hidden_sizes", type=int, nargs="+")
parser.add_argument("--regularization", type=float)

# Training args
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--learning_rate", type=float)

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
    os.path.dirname(__file__), "train_function_encoder_defaults.yaml"
)
defaults = load_defaults_from_yaml(defaults_path)
parser.set_defaults(**defaults)

params = parser.parse_args()

if params.encoder_type not in ["input", "output"]:
    raise ValueError(f"Unknown encoder type: {params.encoder_type}")

device = get_device(params.device)
print(f"Using device: {device}")
set_seed(params.seed)

match params.encoder_type:
    case "input":
        model_name = "input_function_encoder"
    case "output":
        model_name = "output_function_encoder"
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")

# Create SummaryWriter
writer = SummaryWriter(log_dir=params.log_dir)
log_dir = writer.log_dir

# Save args
save_params(params, log_dir, filename_prefix=f"{model_name}_params")

# Create checkpoint directories
params.checkpoint_dir = setup_checkpoint_dir(params.checkpoint_dir, log_dir)

# Load dataset using utility
model_train_dataset = load_dataset(params.dataset, params, device, split="train")
model_test_dataset = load_dataset(params.dataset, params, device, split="test")

dataset_info = model_train_dataset.get_info()

match params.encoder_type:
    case "input":
        train_dataset = InputFunctionEncoderDataset(model_train_dataset, device=device)
        test_dataset = InputFunctionEncoderDataset(model_test_dataset, device=device)
    case "output":
        train_dataset = OutputFunctionEncoderDataset(model_train_dataset, device=device)
        test_dataset = OutputFunctionEncoderDataset(model_test_dataset, device=device)
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")

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

match params.encoder_type:
    case "input":
        input_size = dataset_info["X_size"]
        output_size = dataset_info["u_size"]
    case "output":
        input_size = dataset_info["Y_size"]
        output_size = dataset_info["s_size"]
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")


function_encoder = create_function_encoder(
    input_size=input_size,
    hidden_sizes=params.hidden_sizes,
    output_size=output_size,
    n_basis=params.n_basis,
    inner_product=(
        memory_efficient_inner_product if params.dataset in ["fwi"] else None
    ),
    regularization=params.regularization,
)
# function_encoder = torch.compile(function_encoder)
function_encoder.to(device)
optimizer = torch.optim.Adam(function_encoder.parameters(), lr=params.learning_rate)

# Train the function encoder
train_function_encoder(
    model=function_encoder,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    n_epochs=params.epochs,
    summary_writer=writer,
    model_name=model_name,
    resume_from_checkpoint=params.resume,
    checkpoint_dir=params.checkpoint_dir,
    checkpoint_interval=params.checkpoint_interval,
    device=device,
)

# Save the function encoder
save_function_encoder(
    model=function_encoder,
    path=os.path.join(log_dir, f"{model_name}.pth"),
)
