"""Utilities for dynamic model function imports."""

import importlib


def get_model_module_path(model_name):
    """
    Convert model name to its module path.

    Args:
        model_name: Name of the model (e.g., "nonlinear", "deeponet")

    Returns:
        str: Module path (e.g., "models.nonlinear", "b2b.deeponet")
    """
    # Handle special cases for b2b package models
    if model_name in ["b2b_nonlinear", "b2b_linear", "deeponet"]:
        return f"b2b.{model_name}"
    else:
        # All other models follow models.{model_name} pattern
        return f"models.{model_name}"


def import_model_functions(model_name, *function_names):
    """
    Dynamically import functions from a model module.

    Args:
        model_name: Name of the model (e.g., "b2b_nonlinear")
        *function_names: Names of functions to import (e.g., "train", "save")

    Returns:
        If one function name: returns the function
        If multiple function names: returns tuple of functions in order requested

    Raises:
        ValueError: If model module cannot be imported
        AttributeError: If a requested function doesn't exist in the module

    Examples:
        train_fn, save_fn = import_model_functions("nonlinear", "train", "save")
        load_fn = import_model_functions("linear", "load")
    """
    module_path = get_model_module_path(model_name)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Cannot import model '{model_name}' from {module_path}: {e}"
        ) from e

    functions = []
    for func_name in function_names:
        if not hasattr(module, func_name):
            raise AttributeError(
                f"Function '{func_name}' not found in {module_path} for model '{model_name}'"
            )
        functions.append(getattr(module, func_name))

    # Return single function if only one requested, otherwise return tuple
    if len(functions) == 1:
        return functions[0]
    return tuple(functions)
