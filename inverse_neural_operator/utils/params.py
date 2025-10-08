"""Utilities for saving parameters."""

import torch


def save_params(params, log_dir, filename_prefix="params"):
    """
    Save training parameters to both text and torch formats.

    Args:
        params: ArgumentParser namespace containing parameters
        log_dir: Directory to save parameters to
        filename_prefix: Prefix for parameter files (default: "params")
    """
    with open(f"{log_dir}/{filename_prefix}.txt", "w") as f:
        f.write(str(params))

    torch.save(params, f"{log_dir}/{filename_prefix}.pth")
