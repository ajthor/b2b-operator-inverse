import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset

from inverse_neural_operator.model.function_encoder import (
    FunctionEncoderFactory,
    train as train_function_encoder,
)

import tqdm


# Parse command line args

parser = argparse.ArgumentParser(
    description="Train a model to predict the inverse of a b2b operator"
)
# Dataset args
parser.add_argument(
    "--dataset", type=str, help="Dataset to train on", default="derivative_polynomial"
)
# Model args
parser.add_argument(
    "--model", type=str, help="Model to train", default="variational_autoencoder"
)
# Function encoder args
parser.add_argument("--n_basis", type=int, help="Number of basis functions", default=8)
parser.add_argument(
    "--input_basis_function_layer_sizes",
    type=int,
    nargs="+",
    help="Layer sizes of input basis functions",
    default=[1, 64, 1],
)
parser.add_argument(
    "--output_basis_function_layer_sizes",
    type=int,
    nargs="+",
    help="Layer sizes of output basis functions",
    default=[1, 64, 1],
)
# Training args
parser.add_argument("--batch_size", type=int, help="Batch size", default=5)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1000)
parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-3)
# SummaryWriter args
parser.add_argument(
    "--log_dir",
    type=str,
    help="Directory to save tensorboard logs",
    default=None,
)
parser.add_argument(
    "--comment",
    type=str,
    help="Comment to append to tensorboard log directory",
    default="",
)
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


# Define model

n_basis = args.n_basis

# input_basis_functions = MultiHeadedMLP(layer_sizes=[1, 64, 1], num_heads=n_basis)
# input_function_encoder = FunctionEncoder(input_basis_functions)

# output_basis_functions = MultiHeadedMLP(layer_sizes=[1, 64, 1], num_heads=n_basis)
# output_function_encoder = FunctionEncoder(output_basis_functions)


match args.model:
    case "autoencoder":
        from inverse_neural_operator.model.autoencoder import (
            Autoencoder,
            loss_function,
        )

        model = Autoencoder(
            input_size=n_basis,
            hidden_sizes=[64],
            latent_size=n_basis,
        )

    case "variational_autoencoder":
        from inverse_neural_operator.model.variational_autoencoder import (
            VariationalAutoencoder,
            loss_function,
        )

        model = VariationalAutoencoder(
            alpha_size=n_basis,
            beta_size=n_basis,
            hidden_sizes=[64],
            latent_size=n_basis,
        )

    case "invertible_network":
        from inverse_neural_operator.model.invertible_network import (
            InvertibleNetwork,
            loss_function,
        )

        model = InvertibleNetwork(
            input_size=n_basis,
            condition_size=n_basis,
            hidden_sizes=[64],
        )

    case _:
        raise ValueError(f"Unknown model: {args.model}")


# Train model
