import argparse

import torch
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os

from models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
)


# Parse command line args

parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument("--dataset", type=str, default="wave_scattering")

# Model args
parser.add_argument("--model", type=str, default="b2b_nonlinear")
parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256, 256])

# DeepONet specific args
parser.add_argument("--branch_hidden_sizes", type=int, nargs="+", default=[256, 256])
parser.add_argument("--trunk_hidden_sizes", type=int, nargs="+", default=[256, 256])
parser.add_argument("--trunk_input_size", type=int, default=1)
parser.add_argument("--output_channels", type=int, default=1)


# Training args
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=1e-3)

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
parser.add_argument("--resume", type=bool, default=False)

params = parser.parse_args()

# Set default values for function encoder training if none are provided
if params.input_fe_epochs is None:
    params.input_fe_epochs = params.epochs
if params.output_fe_epochs is None:
    params.output_fe_epochs = params.epochs

if params.input_fe_learning_rate is None:
    params.input_fe_learning_rate = params.learning_rate
if params.output_fe_learning_rate is None:
    params.output_fe_learning_rate = params.learning_rate

# If the dataset is wave_scattering, limit the batch size to 5.
if params.dataset == "wave_scattering":
    if params.batch_size > 5:
        print(
            f"Batch size {params.batch_size} is too large for the wave_scattering dataset. "
            "Setting batch size to 5."
        )
        params.batch_size = 5

if params.device is None:
    if torch.cuda.is_available():
        device = "cuda"
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

# Load dataset

match params.dataset:

    # case "derivative_polynomial":
    # train_ds = load_dataset("ajthor/derivative_polynomial", split="train")
    # test_ds = load_dataset("ajthor/derivative_polynomial", split="test")

    case "burgers_1d":
        from data.burgers_1d import load_data

    case "darcy_1d":
        from data.darcy_1d import load_data

    case "parametric_heat":
        from data.parametric_heat import load_data

    case "wave_scattering":
        from data.wave_scattering import load_data

    case _:
        raise ValueError(f"Unknown dataset: {params.dataset}")

# Load data

model_train_dataset = load_data(params, device=device, split="train")
model_test_dataset = load_data(params, device=device, split="test")

# Load the function encoders

dataset_info = model_train_dataset.get_info()


input_function_encoder = create_function_encoder(
    input_size=dataset_info["X_size"],
    hidden_sizes=params.input_fe_hidden_sizes,
    output_size=dataset_info["u_size"],
    n_basis=params.input_fe_n_basis,
)
input_function_encoder = load_function_encoder(
    input_function_encoder,
    os.path.join(log_dir, "input_function_encoder.pth"),
    device=device,
)
input_function_encoder = torch.compile(input_function_encoder)
input_function_encoder.to(device)


output_function_encoder = create_function_encoder(
    input_size=dataset_info["Y_size"],
    hidden_sizes=params.output_fe_hidden_sizes,
    output_size=dataset_info["s_size"],
    n_basis=params.output_fe_n_basis,
)
output_function_encoder = load_function_encoder(
    output_function_encoder,
    os.path.join(log_dir, "output_function_encoder.pth"),
    device=device,
)
output_function_encoder = torch.compile(output_function_encoder)
output_function_encoder.to(device)


# Define model

match params.model:

    case "b2b_linear":
        from models.b2b_operator_linear import (
            create_model,
            train as train_model,
        )

        model = create_model(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
        ).to(device)
        model_optimizer = None

    case "b2b_nonlinear":
        from models.b2b_operator_nonlinear import (
            create_model,
            train as train_model,
        )

        model = create_model(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        model = torch.compile(model)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "deeponet":
        from models.deeponet import (
            create_model,
            train as train_model,
        )

        model = create_model(
            branch_input_size=dataset_info["Y_size"] * dataset_info["Y_len"],
            trunk_input_size=dataset_info["X_size"],
            output_size=dataset_info["u_size"],
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        model = torch.compile(model)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "variational_autoencoder":
        from models.variational_autoencoder import (
            create_model,
            train as train_model,
        )

        model = create_model(
            alpha_size=params.input_fe_n_basis,
            beta_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            latent_size=params.output_fe_n_basis,
        ).to(device)
        model = torch.compile(model)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "invertible_network":
        from models.invertible_network import (
            create_model,
            train as train_model,
        )

        model = create_model(
            input_size=params.input_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            n_coupling_layers=2,
        ).to(device)
        model = torch.compile(model)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case _:
        raise ValueError(f"Unknown model: {params.model}")

# Train model

model_train_dataloader = DataLoader(
    model_train_dataset,
    batch_size=params.batch_size,
    shuffle=True,
)
model_test_dataloader = DataLoader(
    model_test_dataset,
    batch_size=params.batch_size,
    shuffle=True,
)

train_model(
    model=model,
    train_dataloader=model_train_dataloader,
    test_dataloader=model_test_dataloader,
    optimizer=model_optimizer,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=params.epochs,
    summary_writer=writer,
    model_name=params.model,
    params=params,
    resume_from_checkpoint=params.model_resume,
    checkpoint_dir=params.checkpoint_dir,
    checkpoint_interval=params.checkpoint_interval,
    device=device,
)

# Save model

torch.save(
    model.state_dict(),
    f"{log_dir}/model.pth",
)
