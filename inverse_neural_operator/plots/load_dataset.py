def load_dataset(params, device):
    # Load dataset

    match params.dataset:
        case "burgers_1d":
            from inverse_neural_operator.data.burgers_1d import load_data

        case "darcy_1d":
            from inverse_neural_operator.data.darcy_1d import load_data

        case "parametric_heat":
            from inverse_neural_operator.data.parametric_heat import load_data

        case "wave_scattering":
            from inverse_neural_operator.data.wave_scattering import load_data

        case "fwi":
            from inverse_neural_operator.data.fwi_data import load_data

        case "chladni_2d":
            from inverse_neural_operator.data.chladni_2d import load_data

        case _:
            raise ValueError(f"Unknown dataset: {params.dataset}")

    # Load data

    test_dataset = load_data(params, device=device, split="test")
    dataset_info = test_dataset.get_info()

    return test_dataset, dataset_info
