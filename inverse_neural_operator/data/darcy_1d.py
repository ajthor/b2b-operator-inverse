import torch

from datasets import load_dataset

from data.process_data import process_dataset


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params: Parameters for processing
        device: The device to use
        split: The dataset split to load (default: "train")

    Returns:
        A tuple containing the processed dataset info for the specified split
    """
    ds = load_dataset("ajthor/darcy_1d", split=split)

    # Process the dataset
    (
        model_dataset,
        input_fe_dataset,
        output_fe_dataset,
        input_info,
        output_info,
        model_info,
    ) = process_dataset(
        ds,
        params,
        device=device,
    )

    return (
        model_dataset,
        input_fe_dataset,
        output_fe_dataset,
        input_info,
        output_info,
        model_info,
    )
