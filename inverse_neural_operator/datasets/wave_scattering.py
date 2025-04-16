import torch

from datasets import load_dataset

from inverse_neural_operator.datasets.process_data import process_dataset


def load_dataset(args, device):

    train_ds = load_dataset("ajthor/wave_scattering", split="train")
    test_ds = load_dataset("ajthor/wave_scattering", split="test")

    datasets_and_info = process_dataset(
        train_ds,
        test_ds,
        args,
        device=device,
    )

    return datasets_and_info
