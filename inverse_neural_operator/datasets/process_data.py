import torch

from torch.utils.data import DataLoader


def process_input_function_encoder_dataset(train_ds, test_ds, args, device):

    def process_ds(point):
        pt_len = len(point["X"])
        indices = torch.randperm(pt_len, device=device)
        split_idx = pt_len // 2
        example_indices = indices[:split_idx]
        remaining_indices = indices[split_idx:]
        point["example_xs"] = point["X"][example_indices]
        point["example_ys"] = point["f"][example_indices]
        point["xs"] = point["X"][remaining_indices]
        point["ys"] = point["f"][remaining_indices]

        if point["example_xs"].dim() == 1:
            point["example_xs"] = point["example_xs"].unsqueeze(-1)
        if point["example_ys"].dim() == 1:
            point["example_ys"] = point["example_ys"].unsqueeze(-1)
        if point["xs"].dim() == 1:
            point["xs"] = point["xs"].unsqueeze(-1)
        if point["ys"].dim() == 1:
            point["ys"] = point["ys"].unsqueeze(-1)

        return point

    input_fe_train_ds = train_ds.map(process_ds).remove_columns(train_ds.column_names)
    input_fe_test_ds = test_ds.map(process_ds).remove_columns(test_ds.column_names)

    input_fe_train_ds = input_fe_train_ds.with_format("torch", device=device)
    input_fe_test_ds = input_fe_test_ds.with_format("torch", device=device)
    input_fe_train_dataloader = DataLoader(
        input_fe_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    input_fe_test_dataloader = DataLoader(
        input_fe_test_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    input_info = {
        "input_size": input_fe_train_ds[0]["xs"].shape[-1],
        "output_size": input_fe_train_ds[0]["ys"].shape[-1],
    }

    return (
        input_fe_train_dataloader,
        input_fe_test_dataloader,
        input_info,
    )


def process_output_function_encoder_dataset(train_ds, test_ds, args, device):

    def process_ds(point):
        pt_len = len(point["Y"])
        indices = torch.randperm(pt_len, device=device)
        split_idx = pt_len // 2
        example_indices = indices[:split_idx]
        remaining_indices = indices[split_idx:]
        point["example_xs"] = point["Y"][example_indices]
        point["example_ys"] = point["Tf"][example_indices]
        point["xs"] = point["Y"][remaining_indices]
        point["ys"] = point["Tf"][remaining_indices]

        if point["example_xs"].dim() == 1:
            point["example_xs"] = point["example_xs"].unsqueeze(-1)
        if point["example_ys"].dim() == 1:
            point["example_ys"] = point["example_ys"].unsqueeze(-1)
        if point["xs"].dim() == 1:
            point["xs"] = point["xs"].unsqueeze(-1)
        if point["ys"].dim() == 1:
            point["ys"] = point["ys"].unsqueeze(-1)

        return point

    output_fe_train_ds = train_ds.map(process_ds).remove_columns(train_ds.column_names)
    output_fe_test_ds = test_ds.map(process_ds).remove_columns(test_ds.column_names)

    output_fe_train_ds = output_fe_train_ds.with_format("torch", device=device)
    output_fe_test_ds = output_fe_test_ds.with_format("torch", device=device)
    output_fe_train_dataloader = DataLoader(
        output_fe_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    output_fe_test_dataloader = DataLoader(
        output_fe_test_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    output_info = {
        "input_size": output_fe_train_ds[0]["xs"].shape[-1],
        "output_size": output_fe_train_ds[0]["ys"].shape[-1],
    }

    return output_fe_train_dataloader, output_fe_test_dataloader, output_info


def process_model_dataset(train_ds, test_ds, args, device):

    def process_ds(point):
        if point["X"].dim() == 1:
            point["X"] = point["X"].unsqueeze(-1)
        if point["f"].dim() == 1:
            point["f"] = point["f"].unsqueeze(-1)
        if point["Y"].dim() == 1:
            point["Y"] = point["Y"].unsqueeze(-1)
        if point["Tf"].dim() == 1:
            point["Tf"] = point["Tf"].unsqueeze(-1)
        return point

    # ds = ds.map(process_ds)
    model_train_ds = train_ds.map(process_ds)
    model_test_ds = test_ds.map(process_ds)

    model_train_ds = model_train_ds.with_format("torch", device=device)
    model_test_ds = model_test_ds.with_format("torch", device=device)
    model_train_dataloader = DataLoader(
        model_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model_test_dataloader = DataLoader(
        model_test_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model_info = {
        "input_size": model_train_ds[0]["X"].shape[-1],
        "output_size": model_train_ds[0]["Y"].shape[-1],
    }

    return model_train_dataloader, model_test_dataloader, model_info


def process_dataset(train_ds, test_ds, args, device):

    train_ds = train_ds.with_format("torch", device=device)
    test_ds = test_ds.with_format("torch", device=device)

    # Process the datasets to train the input function encoder
    input_fe_train_dataloader, input_fe_test_dataloader, input_info = (
        process_input_function_encoder_dataset(train_ds, test_ds, args, device=device)
    )

    # Process the datasets to train the output function encoder
    output_fe_train_dataloader, output_fe_test_dataloader, output_info = (
        process_output_function_encoder_dataset(train_ds, test_ds, args, device=device)
    )

    # Process the datasets to train the model
    model_train_dataloader, model_test_dataloader, model_info = process_model_dataset(
        train_ds, test_ds, args, device=device
    )

    return (
        model_train_dataloader,
        model_test_dataloader,
        input_fe_train_dataloader,
        input_fe_test_dataloader,
        output_fe_train_dataloader,
        output_fe_test_dataloader,
        input_info,
        output_info,
        model_info,
    )
