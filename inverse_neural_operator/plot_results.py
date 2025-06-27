import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch

from models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
)

from models.model_evaluation import evaluate_random, find_best_case, find_worst_case

device = "cpu"

torch.manual_seed(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results.")
parser.add_argument("--dataset", type=str, default="burgers_1d")
parser.add_argument("--model", type=str, default="variational_autoencoder")
parser.add_argument(
    "--log_dir", type=str, default="/store/at46867/b2b_operator_inverse"
)
parser.add_argument(
    "--results_dir", type=str, default="results/burgers_1d/variational_autoencoder"
)

args = parser.parse_args()

log_dir = args.log_dir
model_path = args.model
dataset_path = args.dataset
results_dir = args.results_dir

log_dir = os.path.join(log_dir, dataset_path, model_path, "seed_1")

# load params
params = torch.load(f"{log_dir}/params.pth", weights_only=False)

# Load dataset

match params.dataset:
    case "burgers_1d":
        from data.burgers_1d import (
            load_data,
            plot_input,
            plot_output,
        )

    case "darcy_1d":
        from data.darcy_1d import (
            load_data,
            plot_input,
            plot_output,
        )

    case "parametric_heat":
        from data.parametric_heat import (
            load_data,
            plot_input,
            plot_output,
        )

    case "wave_scattering":
        from data.wave_scattering import (
            load_data,
            plot_input,
            plot_output,
        )

    case "chladni_2d":
        from data.chladni_2d import (
            load_data,
            plot_input,
            plot_output,
        )

    case _:
        raise ValueError(f"Unknown dataset: {params.dataset}")

# Load data

test_dataset = load_data(params, device=device, split="test")
dataset_info = test_dataset.get_info()

# Load model

match params.model:
    case "b2b_linear":
        from models.b2b_operator_linear import (
            create_model,
            load,
            evaluate,
        )

        model = create_model(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
        ).to(device)
        model = load(
            model=model, path=os.path.join(log_dir, "model.pth"), device=device
        )

    case "b2b_nonlinear":
        from models.b2b_operator_nonlinear import (
            create_model,
            load,
            evaluate,
        )

        model = create_model(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        model = load(
            model=model, path=os.path.join(log_dir, "model.pth"), device=device
        )

    case "deeponet":
        from models.deeponet import (
            create_model,
            load,
            evaluate,
        )

        model = create_model(
            branch_input_size=dataset_info["Y_size"] * dataset_info["Y_len"],
            trunk_input_size=dataset_info["X_size"],
            output_size=dataset_info["u_size"],
            hidden_sizes=params.hidden_sizes,
        ).to(device)
        model = load(
            model=model, path=os.path.join(log_dir, "model.pth"), device=device
        )

    case "variational_autoencoder":
        from models.variational_autoencoder import (
            create_model,
            load,
            evaluate,
        )

        model = create_model(
            alpha_size=params.input_fe_n_basis,
            beta_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            latent_size=params.output_fe_n_basis,
        ).to(device)
        model = load(
            model=model, path=os.path.join(log_dir, "model.pth"), device=device
        )

    case "invertible_network":
        from models.invertible_network import (
            create_model,
            load,
            evaluate,
        )

        model = create_model(
            input_size=params.input_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            n_coupling_layers=2,
        ).to(device)
        model = load(
            model=model, path=os.path.join(log_dir, "model.pth"), device=device
        )

    case _:
        raise ValueError(f"Unknown model: {params.model}")


# Load the input function encoder

input_function_encoder_params = torch.load(
    os.path.join(log_dir, "input_function_encoder_params.pth"), weights_only=False
)
input_function_encoder = create_function_encoder(
    input_size=dataset_info["X_size"],
    hidden_sizes=input_function_encoder_params.hidden_sizes,
    output_size=dataset_info["u_size"],
    n_basis=input_function_encoder_params.n_basis,
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
)
# output_function_encoder = torch.compile(output_function_encoder)
output_function_encoder.to(device)
output_function_encoder = load_function_encoder(
    output_function_encoder,
    os.path.join(log_dir, "output_function_encoder.pth"),
    device=device,
)

# Load model


# Helper function for function encoder plots
def plot_fe_evaluations(model, dataset, file_name):
    """Plot function encoder evaluations."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Evaluate and plot for a random instance
    idx = np.random.randint(0, len(dataset))
    X, u = dataset[idx]

    # Get prediction
    pred = model(X)

    X = X.cpu().numpy()
    u = u.cpu().numpy()
    pred = pred.cpu().numpy()

    # Plot
    axs[0].plot(X, u)
    axs[1].plot(X, pred)

    # Add legend at the top center
    fig.legend(
        ["Ground Truth", "Reconstruction"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


# Create plots for function encoders
plot_fe_evaluations(
    model=input_function_encoder,
    dataset=input_fe_test_dataset,
    file_name=os.path.join(results_dir, "input_function_encoder_evaluation.png"),
)

plot_fe_evaluations(
    model=output_function_encoder,
    dataset=output_fe_test_dataset,
    file_name=os.path.join(results_dir, "output_function_encoder_evaluation.png"),
)

# Plot random cases for model evaluation
for i in range(5):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Get a random case
    point, pred, alpha_pred, loss = evaluate_random(
        model=model,
        dataset=model_test_dataset,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
    )

    # Use dataset-specific plotting for consistent visualization
    idx = np.random.randint(
        0, len(model_test_dataset)
    )  # Use random index for evaluation
    plot_dataset_evaluation(pred, model_test_dataset, idx, axs)

    plt.savefig(os.path.join(results_dir, f"model_evaluation_{i}.png"))
    plt.close()

# Plot best case
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Find the best case
best_case, best_pred, best_alpha_pred, best_loss = find_best_case(
    model=model,
    dataset=model_test_dataset,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
)

# Handle variational autoencoder's multiple samples
if params.model == "variational_autoencoder":
    # Get mean prediction for plotting
    best_pred = torch.mean(best_pred, dim=0)

# Find the index in the dataset that corresponds to best_case
best_idx = 0  # Default to 0 if we can't find it
for i in range(len(model_test_dataset)):
    if torch.all(model_test_dataset[i][0] == best_case[0][0]):
        best_idx = i
        break

# Use dataset-specific plotting for consistent visualization
plot_dataset_evaluation(best_pred, model_test_dataset, best_idx, axs)

plt.savefig(os.path.join(results_dir, "model_best_case_evaluation.png"))
plt.close()

# Plot worst case
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Find the worst case
worst_case, worst_pred, worst_alpha_pred, worst_loss = find_worst_case(
    model=model,
    dataset=model_test_dataset,
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
)

# Handle variational autoencoder's multiple samples
if params.model == "variational_autoencoder":
    # Get mean prediction for plotting
    worst_pred = torch.mean(worst_pred, dim=0)

# Find the index in the dataset that corresponds to worst_case
worst_idx = 0  # Default to 0 if we can't find it
for i in range(len(model_test_dataset)):
    if torch.all(model_test_dataset[i][0] == worst_case[0][0]):
        worst_idx = i
        break

# Use dataset-specific plotting for consistent visualization
plot_dataset_evaluation(worst_pred, model_test_dataset, worst_idx, axs)

plt.savefig(os.path.join(results_dir, "model_worst_case_evaluation.png"))
plt.close()
