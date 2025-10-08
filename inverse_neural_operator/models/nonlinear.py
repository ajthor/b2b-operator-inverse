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
        for layer in self.layers[:-1]:
            alpha = self.activation(layer(alpha))

        beta = self.layers[-1](alpha)

        return beta

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


def loss_function(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model=None,
    lambda_forward=0.0,
):
    X, u, Y, s = batch

    alpha, _ = input_function_encoder.compute_coefficients(X, u)
    beta, _ = output_function_encoder.compute_coefficients(Y, s)

    alpha_pred = model.inverse(beta)
    pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")

    # u_pred = input_function_encoder(X, alpha_pred)
    # pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

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
    forward_model,
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

        # Compute and log re-simulation loss on single batch
        if forward_model is not None:
            with torch.no_grad():
                test_batch = next(iter(test_dataloader))
                resim_coeff_loss, resim_pred_loss = resimulation_loss(
                    model=model,
                    batch=test_batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    forward_model=forward_model,
                    n_samples=1,
                )
            summary_writer.add_scalars(
                "loss/resimulation_coeff", {model_name: resim_coeff_loss}, epoch
            )
            summary_writer.add_scalars(
                "loss/resimulation_pred", {model_name: resim_pred_loss}, epoch
            )

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
):
    model.eval()
    # total_test_loss = 0.0
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        loss = loss_function(
            model=model,
            batch=batch,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        # for batch in test_dataloader:
        #     loss = loss_function(
        #         model=model,
        #         batch=batch,
        #         input_function_encoder=input_function_encoder,
        #         output_function_encoder=output_function_encoder,
        #     )
        #     total_test_loss += loss.item()

    # avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return loss.item()


def resimulation_loss(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples=5,
):
    """
    Compute re-simulation loss for B2B nonlinear operator model.

    For B2B nonlinear: Apply inverse operator to get alpha, then forward operator to get beta, measure MSE.

    Returns:
        resim_coeff_loss: MSE between re-simulated and target coefficients
        resim_pred_loss: MSE between predictions from re-simulated coefficients and ground truth
    """

    X, u, Y, s = batch

    # Get target beta coefficients
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        # Apply inverse operator to get alpha
        alpha_pred = model.inverse(beta_target)

        # Apply forward operator to get predicted beta
        beta_resim = forward_model(alpha_pred)

        # Coefficient error: re-simulated beta vs target beta
        resim_coeff_loss = torch.nn.functional.mse_loss(beta_resim, beta_target)

        # Prediction error: function predictions using re-simulated beta
        s_pred = output_function_encoder(Y, beta_resim)
        resim_pred_loss = torch.nn.functional.mse_loss(s_pred, s)

    model.train()
    return resim_coeff_loss.item(), resim_pred_loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        beta, _ = output_function_encoder.compute_coefficients(Y, s)

        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)

        return pred, alpha_pred
