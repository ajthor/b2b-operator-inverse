import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, concatenate_datasets

from model.function_encoder import (
    FunctionEncoderFactory,
    train as train_function_encoder,
)

import tqdm


# Parse command line args

parser = argparse.ArgumentParser(
    description="Train a model to predict the inverse of a b2b operator"
)
# Dataset args
parser.add_argument("--dataset", type=str, default="derivative_polynomial")
# Model args
parser.add_argument("--model", type=str, default="invertible_network")
# Function encoder args
parser.add_argument("--n_basis", type=int, default=8)
parser.add_argument("--input_hidden_sizes", type=int, nargs="+", default=[64])
parser.add_argument("--output_hidden_sizes", type=int, nargs="+", default=[64])
# Training args
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
# SummaryWriter args
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default="")
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)


# Load dataset

match args.dataset:
    case "derivative_polynomial":
        ds = load_dataset("ajthor/derivative_polynomial", split="train")

    case "burgers_1d":
        ds = load_dataset("ajthor/burgers_1d", split="train")

    case _:
        raise ValueError(f"Unknown dataset: {args.dataset}")


ds = ds.with_format("torch", device=device)


def process_input_ds(point):
    pt_len = len(point["X"])
    indices = torch.randperm(pt_len)
    split_idx = pt_len // 2
    example_indices = indices[:split_idx]
    remaining_indices = indices[split_idx:]
    point["example_xs"] = point["X"][example_indices].unsqueeze(-1)
    point["example_ys"] = point["f"][example_indices].unsqueeze(-1)
    point["xs"] = point["X"][remaining_indices].unsqueeze(-1)
    point["ys"] = point["f"][remaining_indices].unsqueeze(-1)
    return point


input_ds = ds.map(process_input_ds).remove_columns(ds.column_names)


def process_output_ds(point):
    pt_len = len(point["Y"])
    indices = torch.randperm(pt_len)
    split_idx = pt_len // 2
    example_indices = indices[:split_idx]
    remaining_indices = indices[split_idx:]
    point["example_xs"] = point["Y"][example_indices].unsqueeze(-1)
    point["example_ys"] = point["Tf"][example_indices].unsqueeze(-1)
    point["xs"] = point["Y"][remaining_indices].unsqueeze(-1)
    point["ys"] = point["Tf"][remaining_indices].unsqueeze(-1)
    return point


output_ds = ds.map(process_output_ds).remove_columns(ds.column_names)


def process_ds(point):
    if point["X"].dim() == 1:
        point["X"] = point["X"].unsqueeze(-1)
    if point["f"].dim() == 1:
        point["f"] = point["f"].unsqueeze(-1)
    if point["Y"].dim() == 1:
        point["Y"] = point["Y"].unsqueeze(-1)
    if point["Tf"].dim() == 1:
        point["Tf"] = point["Tf"].unsqueeze(-1)
    return point


ds = ds.map(process_ds)
dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

input_ds = input_ds.with_format("torch", device=device)
input_dataloader = DataLoader(input_ds, batch_size=args.batch_size, shuffle=True)

output_ds = input_ds.with_format("torch", device=device)
output_dataloader = DataLoader(output_ds, batch_size=args.batch_size, shuffle=True)


# Define model

n_basis = args.n_basis

input_input_size = input_ds[0]["xs"].shape[-1]
input_output_size = input_ds[0]["ys"].shape[-1]

output_input_size = output_ds[0]["xs"].shape[-1]
output_output_size = output_ds[0]["ys"].shape[-1]


match args.model:

    case "variational_autoencoder":
        from model.variational_autoencoder import (
            VariationalAutoencoderFactory,
            train as train_model,
        )

        model = VariationalAutoencoderFactory.create(
            alpha_size=n_basis,
            beta_size=n_basis,
            hidden_sizes=[64],
            latent_size=n_basis,
        )

    case "invertible_network":
        from model.invertible_network import (
            InvertibleNetworkFactory,
            train as train_model,
        )

        model = InvertibleNetworkFactory.create(
            input_size=n_basis,
            condition_size=n_basis,
            hidden_sizes=[64],
            n_coupling_layers=1,
        )

    case _:
        raise ValueError(f"Unknown model: {args.model}")


input_function_encoder = FunctionEncoderFactory.create(
    input_size=input_input_size,
    hidden_sizes=args.input_hidden_sizes,
    output_size=input_output_size,
    n_basis=n_basis,
)

output_function_encoder = FunctionEncoderFactory.create(
    input_size=output_input_size,
    hidden_sizes=args.output_hidden_sizes,
    output_size=output_output_size,
    n_basis=n_basis,
)

# Train model

# Train the input function encoder
train_function_encoder(
    model=input_function_encoder,
    dataloader=input_dataloader,
    optimizer=torch.optim.Adam(
        input_function_encoder.parameters(), lr=args.learning_rate
    ),
    n_epochs=args.epochs,
)

# Train the output function encoder
train_function_encoder(
    model=output_function_encoder,
    dataloader=output_dataloader,
    optimizer=torch.optim.Adam(
        output_function_encoder.parameters(), lr=args.learning_rate
    ),
    n_epochs=args.epochs,
)

# Train the model
train_model(
    model=model,
    dataloader=dataloader,
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=args.epochs,
    summary_writer=writer,
)
