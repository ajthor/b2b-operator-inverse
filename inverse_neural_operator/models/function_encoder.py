import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from functools import partial

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.coefficients import least_squares

import tqdm
import os
from torch.utils.tensorboard import SummaryWriter


def create_model(
    input_size,
    hidden_sizes,
    output_size,
    n_basis,
    activation=torch.nn.ReLU(),
    regularization=1e-3,
):
    """
    Create a function encoder model.

    Args:
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the MLP
        output_size: Size of the output features
        n_basis: Number of basis functions
        activation: Activation function to use in the MLP
        regularization: Regularization parameter for least squares computation

    Returns:
        FunctionEncoder instance
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    basis_functions = MultiHeadedMLP(
        layer_sizes=layer_sizes,
        num_heads=n_basis,
        activation=activation,
    )

    # Create a custom coefficients method with the specified regularization
    coefficients_method = partial(least_squares, regularization=regularization)

    return FunctionEncoder(
        basis_functions=basis_functions,
        coefficients_method=coefficients_method,
    )


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    path,
    optimizer=None,
    device=None,
):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()
    return model, optimizer, checkpoint["epoch"], checkpoint["loss"]


def loss_function(model, batch):
    example_xs, example_ys, xs, ys = batch

    coefficients = model.compute_coefficients(example_xs, example_ys)
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)
    # norm_loss = basis_normalization_loss(model.basis_functions(xs))
    G = model.basis_functions(xs)
    K = model.inner_product(G, G)
    K = K + torch.eye(K.shape[1], device=K.device)
    norm_loss = ((torch.diagonal(K, dim1=-2, dim2=-1) - 1) ** 2).mean()

    return pred_loss + norm_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    n_epochs,
    summary_writer,
    model_name,
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
):
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")
    if resume_from_checkpoint:
        if os.path.exists(checkpoint_path):
            model, optimizer, start_epoch, loss = load_checkpoint(
                model=model,
                path=checkpoint_path,
                optimizer=optimizer,
                device=device,
            )
            print(f"Resuming training from epoch {start_epoch}...")

    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
        model.train()
        batch = next(iter(train_dataloader))
        optimizer.zero_grad()
        loss = loss_function(model=model, batch=batch)
        loss.backward()
        optimizer.step()

        summary_writer.add_scalars("loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = test_model(model=model, test_dataloader=test_dataloader)
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            loss = loss_function(model=model, batch=batch)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate(model, point):
    model.eval()
    with torch.no_grad():
        example_xs, example_ys, xs, ys = point

        coefficients = model.compute_coefficients(example_xs, example_ys)
        pred = model(xs, coefficients)

        return pred


def plot_evaluations(
    model,
    dataset,
    file_name="results/function_encoder_evaluation.png",
):
    model.eval()
    with torch.no_grad():

        indices = np.random.choice(len(dataset), 9, replace=False)
        subset = Subset(dataset, indices)

        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
        )

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))

        for i, point in enumerate(dataloader):
            pred = evaluate(model, point)
            pred = pred.squeeze(0).cpu().numpy()

            example_xs, example_ys, xs, ys = point
            example_xs = example_xs.squeeze(0).cpu().numpy()
            example_ys = example_ys.squeeze(0).cpu().numpy()
            xs = xs.squeeze(0).cpu().numpy()
            ys = ys.squeeze(0).cpu().numpy()

            # Sort by the xs
            sort_indices = np.argsort(xs, axis=0)
            xs = xs[sort_indices.squeeze()]
            ys = ys[sort_indices.squeeze()]
            pred = pred[sort_indices.squeeze()]

            # Plot the input data
            axs[i // 3, i % 3].scatter(
                example_xs, example_ys, label="Example Data", color="gray", alpha=0.5
            )
            axs[i // 3, i % 3].plot(xs, ys, label="True")
            axs[i // 3, i % 3].plot(xs, pred, label="Prediction")

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
