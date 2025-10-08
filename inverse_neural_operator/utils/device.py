"""Device selection and seed utilities."""

import torch


def get_device(device_arg=None):
    """
    Get the appropriate device for computation.

    Args:
        device_arg: Optional device string (e.g., "cuda", "cpu", "mps").
                   If None, will auto-select based on availability.

    Returns:
        str: Device string to use for torch tensors and models
    """
    if device_arg is not None:
        return device_arg

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
