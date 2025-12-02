import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from safetensors.torch import save_file, load_file

from utils.distributed import is_main_process

import tqdm
import os


class LinearB2BOperatorDeterministic(torch.nn.Module):
    """
    Deterministic linear operator for the inverse problem of parameter estimation.
    Directly maps from output coefficients (beta) to input coefficients (alpha) using a linear transformation.
    Unlike the standard b2b_linear model which learns alpha->beta and inverts, this model directly learns beta->alpha.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size: Size of the input coefficients (alpha)
            output_size: Size of the output coefficients (beta)
        """
        super(LinearB2BOperatorDeterministic, self).__init__()
        # Note: linear layer maps from beta (output_size) to alpha (input_size)
        self.linear = torch.nn.Linear(output_size, input_size, bias=False)
        self.linear.weight.requires_grad = False

    def forward(self, beta):
        """
        Forward pass: alpha = W * beta
        Directly maps from output coefficients to input coefficients.
        """
        return self.linear(beta)


def create_model(input_size, output_size):
    """
    Create a deterministic linear B2B operator model.

    Args:
        input_size: Size of the input coefficients (alpha)
        output_size: Size of the output coefficients (beta)

    Returns:
        LinearB2BOperatorDeterministic instance
    """
    return LinearB2BOperatorDeterministic(
        input_size=input_size,
        output_size=output_size,
    )


def save(model, path):
    """Save model weights in safetensors format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(model.state_dict(), path)


def load(model, path, device=None):
    """Load model weights from safetensors format."""
    state_dict = load_file(path, device=str(device) if device else 'cpu')
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    if not is_main_process():
        return
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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
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

    # Direct prediction: beta -> alpha
    alpha_pred = model(beta)
    pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")

    # # Reconstruct u from predicted alpha
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
    model_name,
    params,
    forward_model,
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
):

    n = 100  # alpha size
    m = 100  # beta size

    SYY = torch.zeros((m, m), device=device)
    SYX = torch.zeros((m, n), device=device)

    with torch.no_grad():
        tqdm_bar = tqdm.tqdm(len(train_dataloader))
        for batch in train_dataloader:

            # Compute the alpha and beta coefficients
            X, u, Y, s = batch

            alpha, _ = input_function_encoder.compute_coefficients(X, u)
            beta, _ = output_function_encoder.compute_coefficients(Y, s)

            # Compute the normal equations: we want to solve beta @ W = alpha
            # This gives us W^T = (beta^T @ beta)^(-1) @ (beta^T @ alpha)
            # Or equivalently: (beta^T @ beta) @ W^T = (beta^T @ alpha)
            SYY += torch.einsum("ij,ik->jk", beta, beta)
            SYX += torch.einsum("ij,ik->jk", beta, alpha)

            tqdm_bar.update(1)

        # Add small regularization term to SYY
        SYY += 1e-6 * torch.eye(m, device=device)

        # Solve for W^T: SYY @ W^T = SYX
        W_T = torch.linalg.solve(SYY, SYX)

        # Set the linear layer weight (which expects W, not W^T)
        model.linear.weight.copy_(W_T.T)

        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss})


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        loss = loss_function(
            model=model,
            batch=batch,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )

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
    Compute re-simulation loss for deterministic linear operator model.

    For deterministic linear: Apply forward operator (beta->alpha) to get alpha,
    then forward model (alpha->beta) to get predicted beta, measure MSE.

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
        # Apply direct operator to get alpha
        alpha_pred = model(beta_target)

        # Apply forward model to get predicted beta
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

        # Direct prediction: beta -> alpha
        alpha_pred = model(beta)
        pred = input_function_encoder(X, alpha_pred)

        return pred, alpha_pred
