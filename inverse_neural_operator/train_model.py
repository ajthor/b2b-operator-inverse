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
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--epochs", type=int, default=5000)
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

# Limit batch sizes for memory-intensive datasets
if params.dataset in ["wave_scattering", "fwi_flat", "fwi_curve"]:
    if params.batch_size > 5:
        print(
            f"Batch size {params.batch_size} is too large for the dataset. "
            "Setting batch size to 5."
        )
        params.batch_size = 5
elif params.dataset == "parametric_heat":
    if params.batch_size > 1:
        print(
            f"Batch size {params.batch_size} is too large for the parametric_heat dataset. "
            "Setting batch size to 1."
        )
        params.batch_size = 1

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
    case "chladni_2d":
        from data.chladni_2d import load_data

    case "fwi_flat":
        from data.fwi_data import load_data
    case "fwi_curve":
        from data.fwi_data import load_data

    case _:
        raise ValueError(f"Unknown dataset: {params.dataset}")

# Load data

train_dataset = load_data(params, device=device, split="train")
test_dataset = load_data(params, device=device, split="test")
dataset_info = train_dataset.get_info()

# Load the input function encoder

input_function_encoder_params = torch.load(
    os.path.join(log_dir, "input_function_encoder_params.pth"), weights_only=False
)
input_function_encoder = create_function_encoder(
    input_size=dataset_info["X_size"],
    hidden_sizes=input_function_encoder_params.hidden_sizes,
    output_size=dataset_info["u_size"],
    n_basis=input_function_encoder_params.n_basis,
    regularization=input_function_encoder_params.regularization,
    inner_product=(
        memory_efficient_inner_product
        if params.dataset in ["fwi_flat", "fwi_curve"]
        else None
    ),
)
# input_function_encoder = torch.compile(input_function_encoder)
input_function_encoder.to(device)
input_function_encoder = load_function_encoder(
    input_function_encoder,
    os.path.join(log_dir, "input_function_encoder.pth"),
    device=device,
)

# Load the output function encoder

output_function_encoder_params = torch.load(
    os.path.join(log_dir, "output_function_encoder_params.pth"), weights_only=False
)
output_function_encoder = create_function_encoder(
    input_size=dataset_info["Y_size"],
    hidden_sizes=output_function_encoder_params.hidden_sizes,
    output_size=dataset_info["s_size"],
    n_basis=output_function_encoder_params.n_basis,
    regularization=output_function_encoder_params.regularization,
    inner_product=(
        memory_efficient_inner_product
        if params.dataset in ["fwi_flat", "fwi_curve"]
        else None
    ),
)
# output_function_encoder = torch.compile(output_function_encoder)
output_function_encoder.to(device)
output_function_encoder = load_function_encoder(
    output_function_encoder,
    os.path.join(log_dir, "output_function_encoder.pth"),
    device=device,
)


# Define model

match params.model:

    case "b2b_linear":
        from models.b2b_operator_linear import (
            create_model,
            train as train_model,
            save as save_model,
        )

        model = create_model(
            input_size=input_function_encoder_params.n_basis,
            output_size=output_function_encoder_params.n_basis,
        ).to(device)
        optimizer = None

    case "b2b_nonlinear":
        from models.b2b_operator_nonlinear import (
            create_model,
            train as train_model,
            save as save_model,
        )

        model = create_model(
            input_size=input_function_encoder_params.n_basis,
            output_size=output_function_encoder_params.n_basis,
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        # model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "deeponet":
        from models.deeponet import (
            create_model,
            train as train_model,
            save as save_model,
        )

        model = create_model(
            branch_input_size=dataset_info["Y_size"] * dataset_info["Y_len"],
            trunk_input_size=dataset_info["X_size"],
            output_size=dataset_info["u_size"],
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        # model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "variational_autoencoder":
        from models.variational_autoencoder import (
            create_model,
            train as train_model,
            save as save_model,
        )

        model = create_model(
            alpha_size=input_function_encoder_params.n_basis,
            beta_size=output_function_encoder_params.n_basis,
            hidden_sizes=params.hidden_sizes,
            latent_size=output_function_encoder_params.n_basis,
        ).to(device)
        # model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "invertible_network":
        from models.invertible_network import (
            create_model,
            train as train_model,
            save as save_model,
        )

        model = create_model(
            input_size=input_function_encoder_params.n_basis,
            hidden_sizes=params.hidden_sizes,
            n_coupling_layers=2,
        ).to(device)
        # model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case _:
        raise ValueError(f"Unknown model: {params.model}")

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
