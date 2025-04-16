import torch

from datasets import load_dataset

from data.process_data import process_dataset


def load_data(params, device):

    train_ds = load_dataset("ajthor/darcy_1d", split="train")
    test_ds = load_dataset("ajthor/darcy_1d", split="test")

    datasets_and_info = process_dataset(
        train_ds,
        test_ds,
        params,
        device=device,
    )

    return datasets_and_info
