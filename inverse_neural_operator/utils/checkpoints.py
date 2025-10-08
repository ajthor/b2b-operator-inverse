"""Utilities for checkpoint directory management."""

import os


def setup_checkpoint_dir(checkpoint_dir, log_dir):
    """
    Create checkpoint directory if it doesn't exist.

    Args:
        checkpoint_dir: Desired checkpoint directory path. If None, creates
                       a "checkpoints" subdirectory in log_dir
        log_dir: Log directory to use as parent if checkpoint_dir is None

    Returns:
        str: Path to the checkpoint directory
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(log_dir, "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
