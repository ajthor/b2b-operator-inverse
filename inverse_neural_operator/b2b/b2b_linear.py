import torch
from safetensors.torch import save_file, load_file
import tqdm
import os


class LinearB2BOperatorFwd(torch.nn.Module):
    """
    Deterministic linear forward operator for mapping input coefficients to output coefficients.
    Directly maps from input coefficients (alpha) to output coefficients (beta) using a linear transformation.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size: Size of the input coefficients (alpha)
            output_size: Size of the output coefficients (beta)
        """
        super(LinearB2BOperatorFwd, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Linear layer maps from alpha (input_size) to beta (output_size)
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.requires_grad = False

    def forward(self, alpha):
        """
        Forward pass: beta = W * alpha
        Directly maps from input coefficients to output coefficients.
        """
        return self.linear(alpha)


def create_model(input_size, output_size):
    """
    Create a linear B2B operator model.

    Args:
        input_size: Size of the input coefficients (alpha)
        output_size: Size of the output coefficients (beta)

    Returns:
        LinearB2BOperatorFwd instance
    """
    return LinearB2BOperatorFwd(
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
):
    X, u, Y, s = batch

    alpha, _ = input_function_encoder.compute_coefficients(X, u)
    beta, _ = output_function_encoder.compute_coefficients(Y, s)

    # Direct prediction: alpha -> beta
    beta_pred = model(alpha)
    pred_loss = torch.nn.functional.mse_loss(beta_pred, beta, reduction="mean")

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
    """
    Train the linear forward model using closed-form least squares solution.
    Solves: alpha @ W = beta for W using normal equations.
    """
    n = model.input_size  # alpha size
    m = model.output_size  # beta size

    SXX = torch.zeros((n, n), device=device)
    SXY = torch.zeros((n, m), device=device)

    with torch.no_grad():
        tqdm_bar = tqdm.tqdm(train_dataloader, desc="Computing closed-form solution")
        for batch in train_dataloader:
            # Compute the alpha and beta coefficients
            X, u, Y, s = batch

            alpha, _ = input_function_encoder.compute_coefficients(X, u)
            beta, _ = output_function_encoder.compute_coefficients(Y, s)

            # Compute the normal equations: we want to solve alpha @ W = beta
            # This gives us W^T = (alpha^T @ alpha)^(-1) @ (alpha^T @ beta)
            # Or equivalently: (alpha^T @ alpha) @ W^T = (alpha^T @ beta)
            SXX += torch.einsum("ij,ik->jk", alpha, alpha)
            SXY += torch.einsum("ij,ik->jk", alpha, beta)

            tqdm_bar.update(1)

        # Add small regularization term to SXX
        SXX += 1e-6 * torch.eye(n, device=device)

        # Solve for W^T: SXX @ W^T = SXY
        W_T = torch.linalg.solve(SXX, SXY)

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


def evaluate(model, point, input_function_encoder, output_function_encoder):
    """
    Evaluate forward model: given input u, predict output s
    """
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        # Compute alpha from input u
        alpha, _ = input_function_encoder.compute_coefficients(X, u)

        # Forward pass: alpha -> beta
        beta_pred = model.forward(alpha)

        # Reconstruct output s from predicted beta
        s_pred = output_function_encoder(Y, beta_pred)

        return s_pred
