import torch

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

    # Create datasets
    model_dataset = ModelDataset(ds, device=device)
    model_info = model_dataset.get_info()

    input_fe_dataset = InputFunctionEncoderDataset(model_dataset, device=device)
    input_info = input_fe_dataset.get_info()

    output_fe_dataset = OutputFunctionEncoderDataset(model_dataset, device=device)
    output_info = output_fe_dataset.get_info()

    return (
        model_dataset,
        input_fe_dataset,
        output_fe_dataset,
        input_info,
        output_info,
        model_info,
    )
