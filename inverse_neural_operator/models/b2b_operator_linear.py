import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class LinearB2BOperator(torch.nn.Module):
    """
    Linear operator for the inverse problem of parameter estimation.
    Maps from input coefficients (alpha) to output coefficients (beta) using a linear transformation.
    """

    def __init__(self, input_size, output_size):
        super(LinearB2BOperator, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.requires_grad = False

    def forward(self, alpha):
        """
        Forward pass: beta = W * alpha
        Maps from input coefficients to output coefficients.
        """
        return self.linear(alpha)

    def inverse(self, beta):
        """
        Inverse pass: alpha = W^(-1) * beta
        Maps from output coefficients back to input coefficients using a least-squares solve.
        """
        # For batched inputs, we solve for each sample in the batch
        weight = self.linear.weight

        # Check if beta is a batch or single sample
        if beta.dim() == 1:
            # Single sample case
            return torch.linalg.lstsq(weight, beta.unsqueeze(1)).solution.squeeze(1)
        else:
            # Batched case
            solutions = []
            for b in beta:
                sol = torch.linalg.lstsq(weight, b.unsqueeze(1)).solution.squeeze(1)
                solutions.append(sol)
            return torch.stack(solutions)

        # weight = self.linear.weight.unsqueeze(0).expand(beta.shape[0], -1, -1)
        # solutions = torch.linalg.lstsq(weight, beta).solution

        # return solutions


def create_model(input_size, output_size):
    """
    Create a linear B2B operator model.

    Args:
        input_size: Size of the input coefficients (alpha)
        output_size: Size of the output coefficients (beta)

    Returns:
        LinearB2BOperator instance
    """
    return LinearB2BOperator(
        input_size=input_size,
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


def loss_function(model, batch, input_function_encoder, output_function_encoder, forward_model=None, lambda_forward=0.0):
    X, u, Y, s = batch

    alpha_result = input_function_encoder.compute_coefficients(X, u)
    alpha = alpha_result[0] if isinstance(alpha_result, tuple) else alpha_result

    beta_result = output_function_encoder.compute_coefficients(Y, s)
    beta = beta_result[0] if isinstance(beta_result, tuple) else beta_result

    alpha_pred = model.inverse(beta)

    u_pred = input_function_encoder(X, alpha_pred)

    # pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")
    
    # Add forward model consistency loss if available
    if forward_model is not None and lambda_forward > 0.0:
        with torch.no_grad():
            forward_model.eval()
        # Forward consistency: alpha_pred -> beta_pred should match beta
        beta_pred = forward_model.forward(alpha_pred)
        forward_loss = torch.nn.functional.mse_loss(beta_pred, beta, reduction="mean")
        pred_loss = pred_loss + lambda_forward * forward_loss

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
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
):

    n = 100
    m = 100

    SXX = torch.zeros((n, n), device=device)
    SXY = torch.zeros((n, m), device=device)

    with torch.no_grad():
        tqdm_bar = tqdm.tqdm(len(train_dataloader))
        for batch in train_dataloader:

            # Compute the alpha and beta coefficients
            X, u, Y, s = batch

            alpha, _ = input_function_encoder.compute_coefficients(X, u)
            beta, _ = output_function_encoder.compute_coefficients(Y, s)

            # Compute the normal equations in chunks
            SXX += torch.einsum("ij,ik->jk", alpha, alpha)
            SXY += torch.einsum("ij,ik->jk", alpha, beta)

            tqdm_bar.update(1)

        # Add small regularization term to SXX
        SXX += 1e-6 * torch.eye(n, device=device)

        # Compute the linear operator
        W = torch.linalg.solve(SXX, SXY)

        model.linear.weight.copy_(W.T)

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

        beta_result = output_function_encoder.compute_coefficients(Y, s)
        beta = beta_result[0] if isinstance(beta_result, tuple) else beta_result

        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)

        return pred
