import os
import argparse

import torch

from models.function_encoder import (
    FunctionEncoderFactory,
    evaluate_instance as evaluate_instance_function_encoder,
    plot_evaluations as plot_fe_evaluations,
)

device = "cpu"

torch.manual_seed(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results.")
parser.add_argument("--dataset", type=str, default="burgers_1d")
parser.add_argument("--model", type=str, default="variational_autoencoder")
parser.add_argument("--log_dir", type=str, default="/store/at46867")
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


match params.dataset:

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
    model_test_dataset,
    input_fe_test_dataset,
    output_fe_test_dataset,
    input_info,
    output_info,
    model_info,
) = load_data(params, device=device, split="test")

input_fe_input_size = input_info["input_size"]
input_fe_output_size = input_info["output_size"]

output_fe_input_size = output_info["input_size"]
output_fe_output_size = output_info["output_size"]


match params.model:

    case "b2b_linear":
        from models.b2b_operator_linear import (
            LinearB2BOperatorFactory,
            plot_evaluation,
            plot_worst_case_evaluation,
        )

        model = LinearB2BOperatorFactory.create(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
        )

    case "b2b_nonlinear":
        from models.b2b_operator_nonlinear import (
            NonlinearB2BOperatorFactory,
            plot_evaluation,
            plot_worst_case_evaluation,
        )

        model = NonlinearB2BOperatorFactory.create(
            input_size=params.input_fe_n_basis,
            output_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
        )

    case "variational_autoencoder":
        from models.variational_autoencoder import (
            ConditionalVariationalAutoencoderFactory,
            plot_evaluation,
            plot_worst_case_evaluation,
        )

        model = ConditionalVariationalAutoencoderFactory.create(
            alpha_size=params.input_fe_n_basis,
            beta_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            latent_size=params.output_fe_n_basis,
        )

    case "invertible_network":
        from models.invertible_network import (
            ConditionalInvertibleNetworkFactory,
            plot_evaluation,
            plot_worst_case_evaluation,
        )

        model = ConditionalInvertibleNetworkFactory.create(
            input_size=params.input_fe_n_basis,
            condition_size=params.output_fe_n_basis,
            hidden_sizes=params.hidden_sizes,
            n_coupling_layers=2,
        )

    case _:
        raise ValueError(f"Unknown model: {params.model}")


input_function_encoder = FunctionEncoderFactory.create(
    input_size=input_fe_input_size,
    hidden_sizes=params.input_fe_hidden_sizes,
    output_size=input_fe_output_size,
    n_basis=params.input_fe_n_basis,
)

output_function_encoder = FunctionEncoderFactory.create(
    input_size=output_fe_input_size,
    hidden_sizes=params.output_fe_hidden_sizes,
    output_size=output_fe_output_size,
    n_basis=params.output_fe_n_basis,
)

# Load models
input_function_encoder.load_state_dict(
    torch.load(f"{log_dir}/input_function_encoder.pth")
)
input_function_encoder.eval()

output_function_encoder.load_state_dict(
    torch.load(f"{log_dir}/output_function_encoder.pth")
)
output_function_encoder.eval()

model.load_state_dict(torch.load(f"{log_dir}/model.pth"))
model.eval()


# Create plots
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

for i in range(5):
    plot_evaluation(
        model=model,
        dataset=model_test_dataset,
        file_name=os.path.join(results_dir, f"model_evaluation_{i}.png"),
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
    )

plot_worst_case_evaluation(
    model=model,
    dataset=model_test_dataset,
    file_name=os.path.join(results_dir, "model_worst_case_evaluation.png"),
    input_function_encoder=input_function_encoder,
    output_function_encoder=output_function_encoder,
)
