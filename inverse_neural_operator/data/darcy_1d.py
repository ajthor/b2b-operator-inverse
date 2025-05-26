import torch
import matplotlib.pyplot as plt

from datasets import load_dataset

from data.process_data import ModelDataset


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

    return model_dataset

def plot_input(ax, x, y):
    """
    Plot the input data.

    Args:
        ax: The axis to plot on
        x: The x-coordinates of the data
        y: The y-coordinates of the data
    """

    ax.plot(x, y, label="Input", color="blue")

def plot_output(ax, x, y):
    """
    Plot the output data.

    Args:
        ax: The axis to plot on
        x: The x-coordinates of the data
        y: The y-coordinates of the data
    """

    ax.plot(x, y, label="Output", color="red")