import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class AffineCoupling(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[128, 128],
        split_dim=None,
        activation=torch.nn.ReLU(),
    ):
        super(AffineCoupling, self).__init__()

        self.input_size = input_size
        if split_dim is None:
            self.split_dim = input_size // 2
        else:
            self.split_dim = split_dim

        # Neural network to compute both scale and translation
        self.net = torch.nn.Sequential()

        # Create layers with the specified hidden sizes
        # Input: x2 (the part being transformed), Output: 2 * x1_dim (for scale and translation)
        x1_dim = self.split_dim
        x2_dim = input_size - self.split_dim
        layer_sizes = [x2_dim] + hidden_sizes + [2 * x1_dim]

        for i in range(len(layer_sizes) - 1):
            self.net.add_module(
                f"linear_{i}", torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.net.add_module(f"activation_{i}", activation)

    def forward(self, x):
        """
        Forward transformation: x -> y
        Implements the affine coupling layer: y1 = x1 * exp(s(x2)) + t(x2), y2 = x2
        Returns: (y, log_det_jacobian)
        """
        x1, x2 = torch.split(x, [self.split_dim, x.size(-1) - self.split_dim], dim=-1)

        # Compute scale and translation from x2
        net_output = self.net(x2)
        s, t = torch.chunk(net_output, 2, dim=-1)

        # Bound scale parameters for numerical stability
        # Use clipping instead of tanh to avoid scaling issues
        s = torch.clamp(s, min=-10.0, max=10.0)

        # Apply affine transformation to x1
        y1 = x1 * torch.exp(s) + t
        y2 = x2

        # Log determinant of Jacobian
        log_det_J = torch.sum(s, dim=-1)

        return torch.cat([y1, y2], dim=-1), log_det_J

    def inverse(self, y):
        """
        Inverse transformation: y -> x
        Implements the inverse of affine coupling: x1 = (y1 - t(y2)) / exp(s(y2)), x2 = y2
        """
        y1, y2 = torch.split(y, [self.split_dim, y.size(-1) - self.split_dim], dim=-1)

        # Compute scale and translation from y2
        net_output = self.net(y2)
        s, t = torch.chunk(net_output, 2, dim=-1)

        # Bound scale parameters for numerical stability (consistent with forward)
        s = torch.clamp(s, min=-10.0, max=10.0)

        # Apply inverse affine transformation to y1
        x1 = (y1 - t) * torch.exp(-s)
        x2 = y2

        return torch.cat([x1, x2], dim=-1)


class InnAffine(torch.nn.Module):
    def __init__(self, coupling_layers, output_size):
        super(InnAffine, self).__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)
        self.output_size = output_size

    def forward(self, alpha):
        """
        Forward transformation: alpha -> (beta, z)
        Splits alpha into beta (output coefficients) and z (latent variables)
        Returns: (beta, z, log_det_jacobian)
        """
        x = alpha
        log_det_J_total = 0.0

        for layer in self.coupling_layers:
            x, log_det_J = layer.forward(x)
            log_det_J_total += log_det_J

        # Split output into beta and z
        # beta should have the same size as output coefficients
        beta = x[..., : self.output_size]
        z = x[..., self.output_size :]

        return beta, z, log_det_J_total

    def inverse(self, beta=None, z=None):
        """
        Inverse transformation: (beta, z) -> alpha
        If only beta provided, z is sampled from standard normal
        """
        if beta is None:
            raise ValueError("beta must be provided")

        if z is None:
            # Sample z from standard normal distribution
            # z should have the remaining dimensions after beta
            input_size = self.coupling_layers[0].input_size
            z_size = input_size - self.output_size
            z = torch.randn(beta.shape[0], z_size, device=beta.device, dtype=beta.dtype)

        # Concatenate beta and z
        y = torch.cat([beta, z], dim=-1)

        # Apply inverse coupling layers
        for layer in reversed(self.coupling_layers):
            y = layer.inverse(y)
        return y

    def sample_posterior(self, beta, n_samples):
        """
        Sample from the posterior distribution given observed beta.

        Args:
            beta: Observed output coefficients [batch_size, beta_dim]
            n_samples: Number of samples to generate

        Returns:
            samples: Generated alpha samples [n_samples, batch_size, alpha_dim]
        """
        samples = []
        batch_size, beta_dim = beta.shape

        for _ in range(n_samples):
            # Sample z from standard normal distribution
            # z should have the remaining dimensions after beta
            input_size = self.coupling_layers[0].input_size
            z_size = input_size - self.output_size
            z = torch.randn(batch_size, z_size, device=beta.device, dtype=beta.dtype)

            # Generate alpha sample
            alpha_sample = self.inverse(beta=beta, z=z)
            samples.append(alpha_sample)

        return torch.stack(samples, dim=0)


def create_model(
    input_size, output_size=None, hidden_sizes=[128, 128], n_coupling_layers=2
):
    """
    Create an InnAffine model.

    Args:
        input_size: Size of the input features (alpha coefficients)
        output_size: Size of the output features (beta coefficients)
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use

    Returns:
        InnAffine instance
    """
    if output_size is None:
        output_size = input_size // 2  # Default fallback

    coupling_layers = []

    for i in range(n_coupling_layers):
        # Alternate between splitting at output_size and remaining dimensions
        # This ensures compatibility with the final beta/z split
        if i % 2 == 0:
            split_dim = output_size
        else:
            split_dim = input_size - output_size

        layer = AffineCoupling(
            input_size=input_size, hidden_sizes=hidden_sizes, split_dim=split_dim
        )
        coupling_layers.append(layer)

    return InnAffine(coupling_layers=coupling_layers, output_size=output_size)


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

    # Encode ground-truth coefficients
    alpha_gt, _ = input_function_encoder.compute_coefficients(X, u)
    beta_gt, _ = output_function_encoder.compute_coefficients(Y, s)

    # Inverse loss: beta -> alpha -> u_pred vs u_gt
    # Use deterministic inverse by setting z = 0 for stability
    batch_size = beta_gt.shape[0]
    input_size = model.coupling_layers[0].input_size
    z_size = input_size - model.output_size
    z_zero = torch.zeros(batch_size, z_size, device=beta_gt.device, dtype=beta_gt.dtype)

    alpha_pred = model.inverse(beta=beta_gt, z=z_zero)
    u_pred = input_function_encoder(X, alpha_pred)
    inverse_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    # Forward prediction loss: alpha_gt -> beta_pred vs beta_gt
    beta_pred, _, _ = model.forward(alpha_gt)  # Extract tensor, ignore log_det_J
    s_pred = output_function_encoder(Y, beta_pred)
    # forward_loss = torch.nn.functional.mse_loss(beta_pred, beta_gt, reduction="mean")
    forward_loss = torch.nn.functional.mse_loss(s_pred, s, reduction="mean")

    return inverse_loss + forward_loss


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

        # Compute and log re-simulation loss (average over batches)
        total_resim_loss = 0.0
        n_resim_batches = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch_resim_loss = resimulation_loss(
                    model=model,
                    batch=batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    forward_model=forward_model,
                    n_samples=1,
                )
                total_resim_loss += batch_resim_loss
                n_resim_batches += 1
        avg_resim_loss = total_resim_loss / max(n_resim_batches, 1)
        summary_writer.add_scalars(
            "loss/resimulation", {model_name: avg_resim_loss}, epoch
        )

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
):
    model.eval()
    # total_test_loss = 0.0
    # n_batches = 0
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
        #     n_batches += 1

    # avg_test_loss = total_test_loss / max(n_batches, 1)
    return loss.item()


def resimulation_loss(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples=1,
):
    """
    Compute re-simulation loss for affine INN model.

    For affine INN: Use deterministic inverse mapping (z=0) beta* -> alpha,
    apply forward operator alpha -> beta_resim, measure MSE(beta_resim, beta*).
    Since we use z=0 for deterministic evaluation, n_samples parameter is ignored.

    Re-simulation flow: beta_measured -> alpha_pred -> beta_resim -> loss(beta_resim, beta_measured)
    """
    X, u, Y, s = batch

    # Get target beta coefficients from observed output
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        # Deterministic inverse: beta -> alpha (using z=0 for deterministic behavior)
        batch_size = beta_target.shape[0]
        input_size = model.coupling_layers[0].input_size
        z_size = input_size - model.output_size
        z_zero = torch.zeros(
            batch_size, z_size, device=beta_target.device, dtype=beta_target.dtype
        )

        alpha_pred = model.inverse(
            beta=beta_target, z=z_zero
        )  # [batch_size, alpha_dim]

        # Forward re-simulation: alpha -> beta
        beta_resim = forward_model(alpha_pred)  # [batch_size, beta_dim]

        # Compare re-simulated beta with measured beta
        resim_loss = torch.nn.functional.mse_loss(beta_resim, beta_target)

    model.train()
    return resim_loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        beta, _ = output_function_encoder.compute_coefficients(Y, s)

        # Deterministic inverse with z = 0 for evaluation
        batch_size = beta.shape[0]
        input_size = model.coupling_layers[0].input_size
        z_size = input_size - model.output_size
        z_zero = torch.zeros(batch_size, z_size, device=beta.device, dtype=beta.dtype)
        alpha_pred = model.inverse(beta=beta, z=z_zero)
        pred = input_function_encoder(X, alpha_pred)

        return pred
