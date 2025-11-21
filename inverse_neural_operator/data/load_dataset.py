"""
Dataset loading utility for centralized dataset management.

This module provides a unified interface for loading all supported datasets.
"""


def load_dataset(dataset_name, params, device, split="train", return_info=False):
    """
    Load a dataset based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset to load
        params: Parameters object containing dataset configuration
        device (str): Device to load data on
        split (str): Dataset split to load ("train" or "test")
        return_info (bool): If True, return (dataset, dataset_info) tuple for plotting compatibility

    Returns:
        Dataset object with standardized interface, or (dataset, dataset_info) tuple if return_info=True

    Raises:
        ValueError: If dataset_name is not supported
    """
    # Apply dataset-specific batch size adjustments

    if dataset_name == "fwi":
        if hasattr(params, "batch_size") and params.batch_size > 20:
            print(
                f"Batch size {params.batch_size} is too large for the fwi dataset. "
                "Setting batch size to 20."
            )
            params.batch_size = 20

    if dataset_name == "wave_scattering":
        if hasattr(params, "batch_size") and params.batch_size > 4:
            print(
                f"Batch size {params.batch_size} is too large for the wave_scattering dataset. "
                "Setting batch size to 4."
            )
            params.batch_size = 4

    if dataset_name == "chladni_2d":
        if hasattr(params, "batch_size") and params.batch_size > 20:
            print(
                f"Batch size {params.batch_size} is too large for the chladni_2d dataset. "
                "Setting batch size to 20."
            )
            params.batch_size = 20

    # Load dataset based on name
    match dataset_name:
        case "burgers_1d":
            from inverse_neural_operator.data.burgers_1d import load_data
        case "darcy_1d":
            from inverse_neural_operator.data.darcy_1d import load_data
        case "parametric_heat":
            from inverse_neural_operator.data.parametric_heat import load_data
        case "wave_scattering":
            from inverse_neural_operator.data.wave_scattering import load_data
        case "chladni_2d":
            from inverse_neural_operator.data.chladni_2d import load_data
        case "fwi":
            from inverse_neural_operator.data.fwi_data import load_data
        case "elastic_plate":
            from inverse_neural_operator.data.elastic_plate import load_data
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load and return the dataset
    dataset = load_data(params, device=device, split=split)

    if return_info:
        dataset_info = dataset.get_info()
        return dataset, dataset_info
    else:
        return dataset
