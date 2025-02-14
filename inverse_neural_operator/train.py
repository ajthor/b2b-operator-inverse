import argparse

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, concatenate_datasets

from model.function_encoder import (
    FunctionEncoderFactory,
    train as train_function_encoder,
)

import tqdm


# Parse command line args

parser = argparse.ArgumentParser()
# Dataset args
parser.add_argument("--dataset", type=str, default="derivative_polynomial")
# Model args
parser.add_argument("--model", type=str, default="variational_autoencoder")
# Function encoder args
parser.add_argument("--n_basis", type=int, default=8)
parser.add_argument("--input_hidden_sizes", type=int, nargs="+", default=[64])
parser.add_argument("--output_hidden_sizes", type=int, nargs="+", default=[64])
# Training args
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-3)
# SummaryWriter args
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default="")
# Seed args
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(args.seed)


# Load dataset

match args.dataset:
    case "derivative_polynomial":
        train_ds = load_dataset("ajthor/derivative_polynomial", split="train")
        test_ds = load_dataset("ajthor/derivative_polynomial", split="test")

    case "burgers_1d":
        train_ds = load_dataset("ajthor/burgers_1d", split="train")
        test_ds = load_dataset("ajthor/burgers_1d", split="test")

    case _:
        raise ValueError(f"Unknown dataset: {args.dataset}")


train_ds = train_ds.with_format("torch", device=device)
test_ds = test_ds.with_format("torch", device=device)


def process_input_ds(point):
    pt_len = len(point["X"])
    indices = torch.randperm(pt_len)
    split_idx = pt_len // 2
    example_indices = indices[:split_idx]
    remaining_indices = indices[split_idx:]
    point["example_xs"] = point["X"][example_indices]
    point["example_ys"] = point["f"][example_indices]
    point["xs"] = point["X"][remaining_indices]
    point["ys"] = point["f"][remaining_indices]

    if point["example_xs"].dim() == 1:
        point["example_xs"] = point["example_xs"].unsqueeze(-1)
    if point["example_ys"].dim() == 1:
        point["example_ys"] = point["example_ys"].unsqueeze(-1)
    if point["xs"].dim() == 1:
        point["xs"] = point["xs"].unsqueeze(-1)
    if point["ys"].dim() == 1:
        point["ys"] = point["ys"].unsqueeze(-1)

    return point


train_input_ds = train_ds.map(process_input_ds).remove_columns(train_ds.column_names)
test_input_ds = test_ds.map(process_input_ds).remove_columns(test_ds.column_names)


def process_output_ds(point):
    pt_len = len(point["Y"])
    indices = torch.randperm(pt_len)
    split_idx = pt_len // 2
    example_indices = indices[:split_idx]
    remaining_indices = indices[split_idx:]
    point["example_xs"] = point["Y"][example_indices]
    point["example_ys"] = point["Tf"][example_indices]
    point["xs"] = point["Y"][remaining_indices]
    point["ys"] = point["Tf"][remaining_indices]

    if point["example_xs"].dim() == 1:
        point["example_xs"] = point["example_xs"].unsqueeze(-1)
    if point["example_ys"].dim() == 1:
        point["example_ys"] = point["example_ys"].unsqueeze(-1)
    if point["xs"].dim() == 1:
        point["xs"] = point["xs"].unsqueeze(-1)
    if point["ys"].dim() == 1:
        point["ys"] = point["ys"].unsqueeze(-1)

    return point


train_output_ds = train_ds.map(process_output_ds).remove_columns(train_ds.column_names)
test_output_ds = test_ds.map(process_output_ds).remove_columns(test_ds.column_names)


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


# ds = ds.map(process_ds)
train_ds = train_ds.map(process_ds)
test_ds = test_ds.map(process_ds)
train_dataloader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_ds,
    batch_size=args.batch_size,
    shuffle=True,
)

train_input_ds = train_input_ds.with_format("torch", device=device)
test_input_ds = test_input_ds.with_format("torch", device=device)
train_input_dataloader = DataLoader(
    train_input_ds,
    batch_size=args.batch_size,
    shuffle=True,
)
test_input_dataloader = DataLoader(
    test_input_ds,
    batch_size=args.batch_size,
    shuffle=True,
)

train_output_ds = train_output_ds.with_format("torch", device=device)
test_output_ds = test_output_ds.with_format("torch", device=device)
train_output_dataloader = DataLoader(
    train_output_ds,
    batch_size=args.batch_size,
    shuffle=True,
)
test_output_dataloader = DataLoader(
    test_output_ds,
    batch_size=args.batch_size,
    shuffle=True,
)


# Define model

n_basis = args.n_basis

input_input_size = train_input_ds[0]["xs"].shape[-1]
input_output_size = train_input_ds[0]["ys"].shape[-1]

output_input_size = train_output_ds[0]["xs"].shape[-1]
output_output_size = train_output_ds[0]["ys"].shape[-1]


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

writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)
log_dir = writer.log_dir

# Train the input function encoder
train_function_encoder(
    model=input_function_encoder,
    train_dataloader=train_input_dataloader,
    test_dataloader=test_input_dataloader,
    optimizer=torch.optim.Adam(
        input_function_encoder.parameters(), lr=args.learning_rate
    ),
    n_epochs=args.epochs,
    summary_writer=writer,
    model_name="input_function_encoder",
)

# Train the output function encoder
train_function_encoder(
    model=output_function_encoder,
    train_dataloader=train_output_dataloader,
    test_dataloader=test_output_dataloader,
    optimizer=torch.optim.Adam(
        output_function_encoder.parameters(), lr=args.learning_rate
    ),
    n_epochs=args.epochs,
    summary_writer=writer,
    model_name="output_function_encoder",
)

# Train the model
train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=1000,
    summary_writer=writer,
    model_name=args.model,
)

# Save args

with open(f"{log_dir}/args.txt", "w") as f:
    f.write(str(args))

torch.save(args, f"{log_dir}/args.pth")

# Save model

torch.save(input_function_encoder.state_dict(), f"{log_dir}/input_function_encoder.pth")
torch.save(
    output_function_encoder.state_dict(), f"{log_dir}/output_function_encoder.pth"
)

torch.save(model.state_dict(), f"{log_dir}/model.pth")
