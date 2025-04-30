import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class NonlinearB2BOperator(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes: list[int] = [128, 128],
        output_size: int = 128,
    ):
        super(NonlinearB2BOperator, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = torch.nn.ModuleList()

        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1]),
            )

        self.activation = torch.nn.ReLU()

    def forward(self, alpha):
        return None

    def inverse(self, beta):
        for layer in self.layers[:-1]:
            beta = self.activation(layer(beta))

        beta = self.layers[-1](beta)

        return beta


def create_model(input_size, hidden_sizes, output_size):
    """
    Create a nonlinear B2B operator model.

    Args:
        input_size: Size of the input coefficients (alpha)
        hidden_sizes: List of hidden layer sizes
        output_size: Size of the output coefficients (beta)

    Returns:
        NonlinearB2BOperator instance
    """
    return NonlinearB2BOperator(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
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


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha = input_function_encoder.compute_coefficients(X, u)
    beta = output_function_encoder.compute_coefficients(Y, s)

    alpha_pred = model.inverse(beta)

    u_pred = input_function_encoder(X, alpha_pred)

    # pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    return pred_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    input_function_encoder,
    output_function_encoder,
    n_epochs,
    summary_writer,
    params,
    model_name,
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
):
    tqdm_bar = tqdm.tqdm(range(n_epochs))
    for epoch in range(n_epochs):

        model.train()
        batch = next(iter(train_dataloader))
        optimizer.zero_grad()
        loss = loss_function(
            model=model,
            batch=batch,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        loss.backward()
        optimizer.step()

        summary_writer.add_scalars("loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            loss = loss_function(
                model=model,
                batch=batch,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
            )
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        beta = output_function_encoder.compute_coefficients(Y, s)
        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)

        return pred
