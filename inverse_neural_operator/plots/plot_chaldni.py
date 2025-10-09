import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch

from inverse_neural_operator.b2b.function_encoder import (
    create_model as create_function_encoder,
    load as load_function_encoder,
    memory_efficient_inner_product,
)

from data.load_dataset import load_dataset
from models.load_model import load_models


device = "cpu"

torch.manual_seed(1)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot results.")
parser.add_argument("--model", type=str, required=True, help="Model name to plot results for")
parser.add_argument(
    "--log_dir",
    type=str,
    default="/store/at46867/b2b_operator_inverse",
    help="Complete path to model directory (e.g., /path/to/logs/dataset/model/seed_1)"
)
parser.add_argument(
    "--results_dir", type=str, default="results/chaldni/variational_autoencoder"
)

args = parser.parse_args()

# log_dir is now the complete path to the model directory
log_dir = args.log_dir
model_name = args.model
results_dir = args.results_dir

# load params
params = torch.load(f"{log_dir}/params.pth", weights_only=False)


# Load dataset
test_dataset, dataset_info = load_dataset(
    params.dataset, params, device, split="test", return_info=True
)

# Load models
input_function_encoder, output_function_encoder, model = load_models(
    params,
    log_dir,
    dataset_info,
    device=device,
)


# Plot
