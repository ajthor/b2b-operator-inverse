import torch

from torch.utils.data import DataLoader, TensorDataset


def process_input_function_encoder_dataset(ds, params, device):

    def process_ds(point):
        pt_len = len(point["X"])
        indices = torch.randperm(pt_len, device=device)
        split_idx = pt_len // 2
        example_indices = indices[:split_idx]
        remaining_indices = indices[split_idx:]
        point["example_xs"] = point["X"][example_indices]
        point["example_ys"] = point["u"][example_indices]
        point["xs"] = point["X"][remaining_indices]
        point["ys"] = point["u"][remaining_indices]

        if point["example_xs"].dim() == 1:
            point["example_xs"] = point["example_xs"].unsqueeze(-1)
        if point["example_ys"].dim() == 1:
            point["example_ys"] = point["example_ys"].unsqueeze(-1)
        if point["xs"].dim() == 1:
            point["xs"] = point["xs"].unsqueeze(-1)
        if point["ys"].dim() == 1:
            point["ys"] = point["ys"].unsqueeze(-1)

        return point

    input_fe_ds = ds.map(process_ds).remove_columns(ds.column_names)
    input_fe_ds = input_fe_ds.with_format("numpy")

    input_size = input_fe_ds[0]["xs"].shape[-1]
    output_size = input_fe_ds[0]["ys"].shape[-1]

    input_len = input_fe_ds[0]["xs"].shape[0]
    output_len = input_fe_ds[0]["ys"].shape[0]

    input_fe_dataset = TensorDataset(
        torch.from_numpy(input_fe_ds["example_xs"]).to(device),
        torch.from_numpy(input_fe_ds["example_ys"]).to(device),
        torch.from_numpy(input_fe_ds["xs"]).to(device),
        torch.from_numpy(input_fe_ds["ys"]).to(device),
    )

    input_info = {
        "input_size": input_size,
        "output_size": output_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    return input_fe_dataset, input_info


def process_output_function_encoder_dataset(ds, params, device):

    def process_ds(point):
        pt_len = len(point["Y"])
        indices = torch.randperm(pt_len, device=device)
        split_idx = pt_len // 2
        example_indices = indices[:split_idx]
        remaining_indices = indices[split_idx:]
        point["example_xs"] = point["Y"][example_indices]
        point["example_ys"] = point["s"][example_indices]
        point["xs"] = point["Y"][remaining_indices]
        point["ys"] = point["s"][remaining_indices]

        if point["example_xs"].dim() == 1:
            point["example_xs"] = point["example_xs"].unsqueeze(-1)
        if point["example_ys"].dim() == 1:
            point["example_ys"] = point["example_ys"].unsqueeze(-1)
        if point["xs"].dim() == 1:
            point["xs"] = point["xs"].unsqueeze(-1)
        if point["ys"].dim() == 1:
            point["ys"] = point["ys"].unsqueeze(-1)

        return point

    output_fe_ds = ds.map(process_ds).remove_columns(ds.column_names)
    output_fe_ds = output_fe_ds.with_format("numpy")

    input_size = output_fe_ds[0]["xs"].shape[-1]
    output_size = output_fe_ds[0]["ys"].shape[-1]

    input_len = output_fe_ds[0]["xs"].shape[0]
    output_len = output_fe_ds[0]["ys"].shape[0]

    output_fe_dataset = TensorDataset(
        torch.from_numpy(output_fe_ds["example_xs"]).to(device),
        torch.from_numpy(output_fe_ds["example_ys"]).to(device),
        torch.from_numpy(output_fe_ds["xs"]).to(device),
        torch.from_numpy(output_fe_ds["ys"]).to(device),
    )

    output_info = {
        "input_size": input_size,
        "output_size": output_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    return output_fe_dataset, output_info


def process_model_dataset(ds, params, device):

    def process_ds(point):
        if point["X"].dim() == 1:
            point["X"] = point["X"].unsqueeze(-1)
        if point["u"].dim() == 1:
            point["u"] = point["u"].unsqueeze(-1)
        if point["Y"].dim() == 1:
            point["Y"] = point["Y"].unsqueeze(-1)
        if point["s"].dim() == 1:
            point["s"] = point["s"].unsqueeze(-1)
        return point

    model_ds = ds.map(process_ds)
    model_ds = model_ds.with_format("numpy")

    input_size = model_ds[0]["X"].shape[-1]
    output_size = model_ds[0]["Y"].shape[-1]

    input_len = model_ds[0]["X"].shape[0]
    output_len = model_ds[0]["Y"].shape[0]

    model_dataset = TensorDataset(
        torch.from_numpy(model_ds["X"]).to(device),
        torch.from_numpy(model_ds["u"]).to(device),
        torch.from_numpy(model_ds["Y"]).to(device),
        torch.from_numpy(model_ds["s"]).to(device),
    )

    model_info = {
        "input_size": input_size,
        "output_size": output_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    return model_dataset, model_info


def process_dataset(ds, params, device):
    """
    Process a single dataset for all three model types.

    Args:
        ds: The dataset to process
        params: Parameters for processing
        device: The device to use

    Returns:
        A tuple of TensorDatasets and information for the models
    """
    ds = ds.with_format("torch", device=device)

    # Process the dataset for input function encoder
    input_fe_dataset, input_info = process_input_function_encoder_dataset(
        ds,
        params,
        device=device,
    )

    # Process the dataset for output function encoder
    output_fe_dataset, output_info = process_output_function_encoder_dataset(
        ds,
        params,
        device=device,
    )

    # Process the dataset for the model
    model_dataset, model_info = process_model_dataset(
        ds,
        params,
        device=device,
    )

    return (
        model_dataset,
        input_fe_dataset,
        output_fe_dataset,
        input_info,
        output_info,
        model_info,
    )
