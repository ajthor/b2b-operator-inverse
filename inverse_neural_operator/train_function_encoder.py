import argparse
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data.process_data import (
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)

from models.function_encoder import (
    create_model as create_function_encoder,
    train as train_function_encoder,
    save as save_function_encoder,
    memory_efficient_inner_product,
)

torch.set_float32_matmul_precision("high")

# Parse command line args
parser = argparse.ArgumentParser()

# Encoder type arg
parser.add_argument(
    "--encoder_type", type=str, choices=["input", "output"], default="input"
)

# Dataset args
parser.add_argument("--dataset", type=str, default="fwi_flat")

parser.add_argument("--model", type=str, default="b2b_nonlinear")

# Function encoder args
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256])

# Training args
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--learning_rate", type=float, default=1e-4)

# SummaryWriter args
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default="")

# Device args
parser.add_argument("--device", type=str, default=None)

# Seed args
parser.add_argument("--seed", type=int, default=42)

# Checkpoint args
parser.add_argument("--checkpoint_interval", type=int, default=100)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--resume", type=bool, default=True)

params = parser.parse_args()

if params.encoder_type not in ["input", "output"]:
    raise ValueError(f"Unknown encoder type: {params.encoder_type}")

# If the dataset is wave_scattering, limit the batch size to 5.
if params.dataset in ["wave_scattering", "fwi_flat", "fwi_curve"]:
    if params.batch_size > 5:
        print(
            f"Batch size {params.batch_size} is too large for the dataset. "
            "Setting batch size to 5."
        )
        params.batch_size = 5

# Set device
if params.device is None:
    if torch.cuda.is_available():
        device = "cuda:1"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    device = params.device

print(f"Using device: {device}")
torch.manual_seed(params.seed)

match params.encoder_type:
    case "input":
        model_name = "input_function_encoder"
    case "output":
        model_name = "output_function_encoder"
    case _:
        raise ValueError(f"Unknown encoder type: {params.encoder_type}")

# Create SummaryWriter
writer = SummaryWriter(log_dir=params.log_dir, comment=params.comment)
log_dir = writer.log_dir

# Save args
with open(f"{log_dir}/{model_name}_params.txt", "w") as f:
    f.write(str(params))

torch.save(params, f"{log_dir}/{model_name}_params.pth")

# Create checkpoint directories
if params.checkpoint_dir is None:
    params.checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(params.checkpoint_dir, exist_ok=True)

# Load dataset based on encoder type
match params.dataset:
    case "burgers_1d":
        from data.burgers_1d import load_data
    case "darcy_1d":
        from data.darcy_1d import load_data
    case "parametric_heat":
        from data.parametric_heat import load_data
    case "wave_scattering":
        from data.wave_scattering import load_data
    case "chladni_2d":
        from data.chladni_2d import load_data

    case "fwi_flat":
        from data.fwi_data import load_data
    case "fwi_curve":
        from data.fwi_data import load_data

    case _:
        raise ValueError(f"Unknown dataset: {params.dataset}")

# Load data

model_train_dataset = load_data(params, device=device, split="train")
model_test_dataset = load_data(params, device=device, split="test")

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
    # shuffle=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=params.batch_size,
    # shuffle=True,
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
        memory_efficient_inner_product
        if params.dataset in ["fwi_flat", "fwi_curve"]
        else None
    ),
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
