import torch
import matplotlib.pyplot as plt

from datasets import load_dataset

from data.process_data import (
    ModelDataset,
    InputFunctionEncoderDataset,
    OutputFunctionEncoderDataset,
)


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params: Parameters for processing
        device: The device to use
        split: The dataset split to load (default: "train")

    Returns:
        A tuple containing the datasets and info for the specified split
    """

    ds = load_dataset("ajthor/darcy_1d", split=split)

    model_dataset = ModelDataset(ds, device=device)
    input_fe_dataset = InputFunctionEncoderDataset(model_dataset, device=device)
    output_fe_dataset = OutputFunctionEncoderDataset(model_dataset, device=device)

    return (model_dataset, input_fe_dataset, output_fe_dataset)


def plot_instance(dataset, idx, axs=None):
    """Plot a single instance from the Darcy 1D dataset."""
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    X, u, Y, s = dataset[idx]
    X = X.cpu().numpy()
    u = u.cpu().numpy()
    Y = Y.cpu().numpy()
    s = s.cpu().numpy()

    axs[0].plot(X, u)
    axs[1].plot(Y, s)

    # Add a single legend at the top center
    fig = plt.gcf()
    fig.legend(
        ["Input", "Output"], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2
    )

    plt.tight_layout()

    return axs


def plot_evaluation(result, dataset, idx, axs=None):
    """Plot the evaluation results against the ground truth for Darcy 1D."""
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    result = result.cpu().numpy()
    X, u, Y, s = dataset[idx]
    X = X.cpu().numpy()
    u = u.cpu().numpy()
    Y = Y.cpu().numpy()
    s = s.cpu().numpy()

    axs[0].plot(X, u, color="blue")
    axs[0].plot(X, result, color="red", linestyle="--")
    axs[1].plot(Y, s, color="green")

    # Add a single legend at the top center
    fig = plt.gcf()
    fig.legend(
        ["Ground Truth", "Prediction", "Output"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )

    plt.tight_layout()

    return axs
