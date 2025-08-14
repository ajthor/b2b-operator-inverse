import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch

from inverse_neural_operator.models.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from plots.load_dataset import load_dataset
from plots.load_model import load_models

from inverse_neural_operator.models.model_evaluation import evaluate_random, find_best_case, find_worst_case

device = "cpu"

torch.manual_seed(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results.")
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
test_dataset, dataset_info = load_dataset(params, device)

# Load models
input_function_encoder, output_function_encoder, model = load_models(
    params,
    log_dir,
    dataset_info,
    device=device,
)


# Plot
