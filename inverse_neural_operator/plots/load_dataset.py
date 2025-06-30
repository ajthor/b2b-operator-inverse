def load_dataset(params, device):
    # Load dataset

    match params.dataset:
        case "burgers_1d":
            from data.burgers_1d import (
                load_data,
            )

        case "darcy_1d":
            from data.darcy_1d import (
                load_data,
            )

        case "parametric_heat":
            from data.parametric_heat import (
                load_data,
            )

        case "wave_scattering":
            from data.wave_scattering import (
                load_data,
            )

        case "fwi_flat":
            from data.fwi_data import (
                load_data,
            )

        case "fwi_curve":
            from data.fwi_data import (
                load_data,
            )

        case _:
            raise ValueError(f"Unknown dataset: {params.dataset}")

    # Load data

    test_dataset = load_data(params, device=device, split="test")
    dataset_info = test_dataset.get_info()

    return test_dataset, dataset_info
