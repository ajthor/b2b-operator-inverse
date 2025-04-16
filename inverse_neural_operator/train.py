import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset

from models.function_encoder import (
    FunctionEncoderFactory,
    train as train_function_encoder,
)


# Parse command line args

parser = argparse.ArgumentParser()
# Dataset args
parser.add_argument("--dataset", type=str, default="derivative_polynomial")
# Model args
parser.add_argument("--model", type=str, default="variational_autoencoder")
# Function encoder args
parser.add_argument("--n_basis", type=int, default=8)
parser.add_argument("--input_fe_hidden_sizes", type=int, nargs="+", default=[64])
parser.add_argument("--output_fe_hidden_sizes", type=int, nargs="+", default=[64])
# Training args
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=1e-3)

parser.add_argument("--input_fe_epochs", type=int, default=None)
parser.add_argument("--input_fe_learning_rate", type=float, default=None)

parser.add_argument("--output_fe_epochs", type=int, default=None)
parser.add_argument("--output_fe_learning_rate", type=float, default=None)
# SummaryWriter args
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default="")
# Seed args
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Set default values for function encoder training if none are provided
if args.input_fe_epochs is None:
    args.input_fe_epochs = args.epochs
if args.output_fe_epochs is None:
    args.output_fe_epochs = args.epochs

if args.input_fe_learning_rate is None:
    args.input_fe_learning_rate = args.learning_rate
if args.output_fe_learning_rate is None:
    args.output_fe_learning_rate = args.learning_rate


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
        from datasets.burgers_1d import load_dataset

    case "darcy_1d":
        from datasets.darcy_1d import load_dataset

    case "parametric_heat":
        from datasets.parametric_heat import load_dataset

    case "wave_scattering":
        from datasets.wave_scattering import load_dataset

    case _:
        raise ValueError(f"Unknown dataset: {args.dataset}")


(
    model_train_dataloader,
    model_test_dataloader,
    input_fe_train_dataloader,
    input_fe_test_dataloader,
    output_fe_train_dataloader,
    output_fe_test_dataloader,
    input_info,
    output_info,
    model_info,
) = load_dataset(args, device=device)

input_fe_input_size = input_info["input_size"]
input_fe_output_size = input_info["output_size"]

output_fe_input_size = output_info["input_size"]
output_fe_output_size = output_info["output_size"]


# Define model

match args.model:

    case "autoencoder":
        from models.autoencoder import (
            ConditionalAutoencoderFactory,
            train as train_model,
        )

        model = ConditionalAutoencoderFactory.create(
            alpha_size=args.n_basis,
            beta_size=args.n_basis,
            hidden_sizes=[64],
            latent_size=args.n_basis,
        )

    case "variational_autoencoder":
        from models.variational_autoencoder import (
            ConditionalVariationalAutoencoderFactory,
            train as train_model,
        )

        model = ConditionalVariationalAutoencoderFactory.create(
            alpha_size=args.n_basis,
            beta_size=args.n_basis,
            hidden_sizes=[64],
            latent_size=args.n_basis,
        )

    case "invertible_network":
        from models.invertible_network import (
            ConditionalInvertibleNetworkFactory,
            train as train_model,
        )

        model = ConditionalInvertibleNetworkFactory.create(
            input_size=args.n_basis,
            condition_size=args.n_basis,
            hidden_sizes=[64],
            n_coupling_layers=1,
        )

    case _:
        raise ValueError(f"Unknown model: {args.model}")


input_function_encoder = FunctionEncoderFactory.create(
    input_size=input_fe_input_size,
    hidden_sizes=args.input_fe_hidden_sizes,
    output_size=input_fe_output_size,
    n_basis=args.n_basis,
)

output_function_encoder = FunctionEncoderFactory.create(
    input_size=output_fe_input_size,
    hidden_sizes=args.output_fe_hidden_sizes,
    output_size=output_fe_output_size,
    n_basis=args.n_basis,
)

# Train model

writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)
log_dir = writer.log_dir

# Save args

with open(f"{log_dir}/args.txt", "w") as f:
    f.write(str(args))

torch.save(args, f"{log_dir}/args.pth")

# Train the input function encoder
train_function_encoder(
    model=input_function_encoder,
    train_dataloader=input_fe_train_dataloader,
    test_dataloader=input_fe_test_dataloader,
    optimizer=torch.optim.Adam(
        input_function_encoder.parameters(), lr=args.input_fe_learning_rate
    ),
    n_epochs=args.input_fe_epochs,
    summary_writer=writer,
    model_name="input_function_encoder",
)

# Train the output function encoder
train_function_encoder(
    model=output_function_encoder,
    train_dataloader=output_fe_train_dataloader,
    test_dataloader=output_fe_test_dataloader,
    optimizer=torch.optim.Adam(
        output_function_encoder.parameters(), lr=args.output_fe_learning_rate
    ),
    n_epochs=args.output_fe_epochs,
    summary_writer=writer,
    model_name="output_function_encoder",
)

# Train the model
train_model(
    model=model,
    train_dataloader=model_train_dataloader,
    test_dataloader=model_test_dataloader,
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=args.epochs,
    summary_writer=writer,
    model_name=args.model,
)

# Save model

torch.save(
    input_function_encoder.state_dict(),
    f"{log_dir}/input_function_encoder.pth",
)

torch.save(
    output_function_encoder.state_dict(),
    f"{log_dir}/output_function_encoder.pth",
)

torch.save(
    model.state_dict(),
    f"{log_dir}/model.pth",
)
