"""
Path configuration module for B2B Operator Inverse project.

This module provides centralized path management for models and runs.

Directory structure:
- models/{dataset}/{model}/seed_{seed}/  - Model weights and params
- runs/{dataset}/{model}/seed_{seed}/    - Training logs, checkpoints, evaluation, plots
"""

import os
from pathlib import Path


def get_repo_root():
    """Get the repository root directory."""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_base_dir():
    """
    Get the base directory for all outputs.

    Priority:
    1. Command line override (passed into get_* functions)
    2. Environment variable B2B_RESULTS_DIR
    3. Default: ./results relative to the repository root

    Returns:
        Path: Absolute path to the base directory
    """
    env_base_dir = os.environ.get('B2B_RESULTS_DIR')
    if env_base_dir:
        return Path(env_base_dir).expanduser().resolve()

    repo_root = get_repo_root()
    return (repo_root / "results").resolve()


def resolve_base_dir(base_dir_override=None):
    """
    Resolve the effective base directory using the override/CLI param if provided.

    Priority:
    1. Explicit override passed to helper
    2. Environment variable / default handled by get_base_dir()

    Args:
        base_dir_override: Optional override string from CLI scripts.

    Returns:
        Path: Absolute path to the base directory
    """
    if base_dir_override:
        base_dir = Path(base_dir_override).expanduser()
        if not base_dir.is_absolute():
            repo_root = get_repo_root()
            base_dir = repo_root / base_dir
        return base_dir.resolve()

    return get_base_dir()


def get_model_dir(dataset, model_name, seed, base_dir_override=None):
    """
    Get the directory for saving/loading model checkpoints.

    Args:
        dataset: Dataset name (e.g., 'burgers_1d', 'darcy_1d')
        model_name: Model name (e.g., 'ifno', 'cinn_additive')
        seed: Random seed (e.g., 0, 1, 2)

    Returns:
        Path: models/{dataset}/{model_name}/seed_{seed}/
    """
    base_dir = resolve_base_dir(base_dir_override)
    model_dir = base_dir / "models" / dataset / model_name / f"seed_{seed}"
    return model_dir


def get_runs_dir(dataset, model_name, seed, base_dir_override=None):
    """
    Get the directory for training runs (logs, checkpoints, evaluation, plots).

    Args:
        dataset: Dataset name (e.g., 'burgers_1d', 'darcy_1d')
        model_name: Model name (e.g., 'ifno', 'cinn_additive')
        seed: Random seed (e.g., 0, 1, 2)

    Returns:
        Path: runs/{dataset}/{model_name}/seed_{seed}/
    """
    base_dir = resolve_base_dir(base_dir_override)
    runs_dir = base_dir / "runs" / dataset / model_name / f"seed_{seed}"
    return runs_dir


def get_results_dir(dataset, model_name, seed, base_dir_override=None):
    """
    Get the directory for evaluation results and figures.

    Note: Results now go in runs/ directory (same as get_runs_dir).
    This function is kept for compatibility but returns the runs directory.

    Args:
        dataset: Dataset name (e.g., 'burgers_1d', 'darcy_1d')
        model_name: Model name (e.g., 'ifno', 'cinn_additive')
        seed: Random seed (e.g., 0, 1, 2)

    Returns:
        Path: runs/{dataset}/{model_name}/seed_{seed}/
    """
    # Results now go in runs/ directory
    return get_runs_dir(dataset, model_name, seed, base_dir_override=base_dir_override)


def get_shared_dir(dataset, seed, base_dir_override=None):
    """
    Get the shared directory for function encoders and forward models.

    Args:
        dataset: Dataset name (e.g., 'burgers_1d', 'darcy_1d')
        seed: Random seed (e.g., 0, 1, 2)

    Returns:
        Path: models/{dataset}/shared/seed_{seed}/
    """
    base_dir = resolve_base_dir(base_dir_override)
    shared_dir = base_dir / "models" / dataset / "shared" / f"seed_{seed}"
    return shared_dir


def ensure_dir_exists(path):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path object or string
    """
    Path(path).mkdir(parents=True, exist_ok=True)


# Legacy path compatibility functions
def get_old_model_path(log_base_dir, dataset, model_name, seed):
    """
    Get the old model path for backward compatibility.

    Args:
        log_base_dir: Old base directory (e.g., '/store/at46867/b2b_operator_inverse')
        dataset: Dataset name
        model_name: Model name
        seed: Random seed

    Returns:
        Path: Old path format
    """
    return Path(log_base_dir) / dataset / model_name / f"seed_{seed}"


def try_load_with_fallback(new_path, old_path_func, *old_path_args):
    """
    Try to load from new path, fall back to old path if it doesn't exist.

    Args:
        new_path: New path to try first
        old_path_func: Function to generate old path
        *old_path_args: Arguments to pass to old_path_func

    Returns:
        Path: The path that exists, or new_path if neither exists
    """
    new_path = Path(new_path)
    if new_path.exists():
        return new_path

    old_path = old_path_func(*old_path_args)
    if old_path.exists():
        print(f"Warning: Using old path {old_path} (new path not found: {new_path})")
        return old_path

    # Return new path anyway (for saving new files)
    return new_path
