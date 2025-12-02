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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_device_string(device_obj):
    """Return a normalized string representation of a torch device-like object."""
    if isinstance(device_obj, torch.device):
        if device_obj.index is not None:
            return f"{device_obj.type}:{device_obj.index}"
        return device_obj.type
    return str(device_obj)


def get_dataset_device(dataset):
    """
    Best-effort lookup for the device backing a dataset.

    Many datasets store a ``device`` attribute; when absent we walk through
    common wrappers (e.g., torch.utils.data.Subset) via their ``dataset`` attr.
    Returns "cpu" as a safe default when the device cannot be determined.
    """
    visited = set()
    current = dataset
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        device_attr = getattr(current, "device", None)
        if device_attr is not None:
            return _normalize_device_string(device_attr)
        current = getattr(current, "dataset", None)
    return "cpu"


def dataset_on_cpu(dataset):
    """
    Return True if the dataset's tensors live on CPU memory.

    Used to decide whether DataLoader pin_memory should be enabled.
    """
    device_str = get_dataset_device(dataset)
    return device_str.startswith("cpu")
