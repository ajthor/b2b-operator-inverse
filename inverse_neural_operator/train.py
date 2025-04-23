import argparse

import torch
from torch.utils.tensorboard import SummaryWriter


from models.function_encoder import (
    FunctionEncoderFactory,
    train as train_function_encoder,
)


# Parse command line args

parser = argparse.ArgumentParser()

# Dataset args
parser.add_argument("--dataset", type=str, default="darcy_1d")

# Model args
parser.add_argument("--model", type=str, default="variational_autoencoder")
parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128, 128])

# Function encoder args
parser.add_argument("--input_fe_n_basis", type=int, default=100)
parser.add_argument("--input_fe_hidden_sizes", type=int, nargs="+", default=[128, 128])

parser.add_argument("--output_fe_n_basis", type=int, default=100)
parser.add_argument("--output_fe_hidden_sizes", type=int, nargs="+", default=[128, 128])

# Training args
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)

parser.add_argument("--input_fe_epochs", type=int, default=None)
parser.add_argument("--input_fe_learning_rate", type=float, default=None)

parser.add_argument("--output_fe_epochs", type=int, default=None)
parser.add_argument("--output_fe_learning_rate", type=float, default=None)

# SummaryWriter args
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--comment", type=str, default="")

# Device args
parser.add_argument("--device", type=str, default=None)

# Seed args
parser.add_argument("--seed", type=int, default=42)
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
) = load_data(params, device=device)

input_fe_input_size = input_info["input_size"]
input_fe_output_size = input_info["output_size"]

output_fe_input_size = output_info["input_size"]
output_fe_output_size = output_info["output_size"]


# Define model

match params.model:

    case "b2b_linear":
        from models.b2b_operator_linear import (
            LinearB2BOperatorFactory,
            train as train_model,
        )

        model = LinearB2BOperatorFactory.create(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
        ).to(device)
        optimizer = None

    case "b2b_nonlinear":
        from models.b2b_operator_nonlinear import (
            NonlinearB2BOperatorFactory,
            train as train_model,
        )

        model = NonlinearB2BOperatorFactory.create(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "variational_autoencoder":
        from models.variational_autoencoder import (
            ConditionalVariationalAutoencoderFactory,
            train as train_model,
        )

        model = ConditionalVariationalAutoencoderFactory.create(
            alpha_size=params.input_fe_n_basis,
            beta_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            latent_size=params.output_fe_n_basis,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case "invertible_network":
        from models.invertible_network import (
            ConditionalInvertibleNetworkFactory,
            train as train_model,
        )

        model = ConditionalInvertibleNetworkFactory.create(
            input_size=params.input_fe_n_basis,
            condition_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            n_coupling_layers=2,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    case _:
        raise ValueError(f"Unknown model: {params.model}")


input_function_encoder = FunctionEncoderFactory.create(
    input_size=input_fe_input_size,
    hidden_sizes=params.input_fe_hidden_sizes,
    output_size=input_fe_output_size,
    n_basis=params.input_fe_n_basis,
).to(device)

output_function_encoder = FunctionEncoderFactory.create(
    input_size=output_fe_input_size,
    hidden_sizes=params.output_fe_hidden_sizes,
    output_size=output_fe_output_size,
    n_basis=params.output_fe_n_basis,
).to(device)

# Train model

writer = SummaryWriter(log_dir=params.log_dir, comment=params.comment)
log_dir = writer.log_dir

# Save args

with open(f"{log_dir}/params.txt", "w") as f:
    f.write(str(params))

torch.save(params, f"{log_dir}/params.pth")

# Train the input function encoder
train_function_encoder(
    model=input_function_encoder,
    train_dataloader=input_fe_train_dataloader,
    test_dataloader=input_fe_test_dataloader,
    optimizer=torch.optim.Adam(
        input_function_encoder.parameters(), lr=params.input_fe_learning_rate
    ),
    n_epochs=params.input_fe_epochs,
    summary_writer=writer,
    model_name="input_function_encoder",
    params=params,
    device=device,
)

# Train the output function encoder
train_function_encoder(
    model=output_function_encoder,
    train_dataloader=output_fe_train_dataloader,
    test_dataloader=output_fe_test_dataloader,
    optimizer=torch.optim.Adam(
        output_function_encoder.parameters(), lr=params.output_fe_learning_rate
    ),
    n_epochs=params.output_fe_epochs,
    summary_writer=writer,
    model_name="output_function_encoder",
    params=params,
    device=device,
)

# Train the model
train_model(
    model=model,
    train_dataloader=model_train_dataloader,
    test_dataloader=model_test_dataloader,
    optimizer=optimizer,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
    n_epochs=params.epochs,
    summary_writer=writer,
    model_name=params.model,
    params=params,
    device=device,
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
