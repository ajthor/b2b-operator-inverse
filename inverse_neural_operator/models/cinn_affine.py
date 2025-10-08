import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class ConditionalAffineCoupling(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
        split_dim=None,
        activation=torch.nn.ReLU(),
    ):
        super(ConditionalAffineCoupling, self).__init__()

        self.input_size = input_size
        self.condition_size = condition_size
        if split_dim is None:
            self.split_dim = input_size // 2
        else:
            self.split_dim = split_dim

        # Neural network to compute scale and translation conditioned on [x2, condition]
        # Input: [x2, condition] where x2 has size (input_size - split_dim) and condition has size condition_size
        # Output: [scale, translation] for x1 which has size 2 * split_dim
        self.net = torch.nn.Sequential()

        # Create layers with the specified hidden sizes
        x1_dim = self.split_dim
        x2_dim = input_size - self.split_dim
        layer_sizes = [x2_dim + condition_size] + hidden_sizes + [2 * x1_dim]
        for i in range(len(layer_sizes) - 1):
            self.net.add_module(
                f"linear_{i}", torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.net.add_module(f"activation_{i}", activation)

    def forward(self, x, condition):
        """
        Forward transformation: x -> z conditioned on condition
        Implements conditional affine coupling: z1 = x1 * exp(s(x2, condition)) + t(x2, condition), z2 = x2
        Returns: (z, log_det_jacobian)
        """
        x1, x2 = torch.split(x, [self.split_dim, x.size(-1) - self.split_dim], dim=-1)

        # Concatenate x2 and condition for the neural network input
        net_input = torch.cat([x2, condition], dim=-1)
        net_output = self.net(net_input)
        s, t = torch.chunk(net_output, 2, dim=-1)

        # Bound scale parameters for numerical stability
        # Use clipping instead of tanh to avoid scaling issues
        s = torch.clamp(s, min=-10.0, max=10.0)

        # Apply affine transformation to x1
        z1 = x1 * torch.exp(s) + t
        z2 = x2

        # Log determinant of Jacobian
        log_det_J = torch.sum(s, dim=-1)

        return torch.cat([z1, z2], dim=-1), log_det_J

    def inverse(self, z, condition):
        """
        Inverse transformation: z -> x conditioned on condition
        Implements inverse conditional affine coupling: x1 = (z1 - t(z2, condition)) / exp(s(z2, condition)), x2 = z2
        """
        z1, z2 = torch.split(z, [self.split_dim, z.size(-1) - self.split_dim], dim=-1)

        # Concatenate z2 and condition for the neural network input
        net_input = torch.cat([z2, condition], dim=-1)
        net_output = self.net(net_input)
        s, t = torch.chunk(net_output, 2, dim=-1)

        # Bound scale parameters for numerical stability
        # Use clipping instead of tanh to avoid scaling issues
        s = torch.clamp(s, min=-10.0, max=10.0)

        # Apply inverse affine transformation to z1
        x1 = (z1 - t) * torch.exp(-s)
        x2 = z2

        return torch.cat([x1, x2], dim=-1)


class CinnAffine(torch.nn.Module):
    def __init__(self, coupling_layers):
        super(CinnAffine, self).__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)

    def forward(self, alpha, beta):
        """
        Forward transformation: transform alpha to latent z conditioned on beta
        This is the key difference from standard INN - we transform x to z given y
        Returns: (z, log_det_jacobian)
        """
        x = alpha
        log_det_J_total = 0.0

        for layer in self.coupling_layers:
            x, log_det_J = layer.forward(x, beta)
            log_det_J_total += log_det_J

        return x, log_det_J_total

    def inverse(self, z, beta):
        """
        Inverse transformation: transform latent z back to alpha conditioned on beta
        """
        y = z
        for layer in reversed(self.coupling_layers):
            y = layer.inverse(y, beta)
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
        batch_size = beta.shape[0]

        # Determine alpha dimension based on model architecture
        # This assumes input_size was used to create the model
        # For cINN, alpha_dim typically equals beta_dim (or can be inferred from coupling layers)
        alpha_dim = self.coupling_layers[0].input_size

        for _ in range(n_samples):
            # Sample z from standard normal distribution
            z = torch.randn(batch_size, alpha_dim, device=beta.device, dtype=beta.dtype)

            # Generate alpha sample
            alpha_sample = self.inverse(z, beta)
            samples.append(alpha_sample)

        return torch.stack(samples, dim=0)


def create_model(
    input_size, condition_size, hidden_sizes=[128, 128], n_coupling_layers=2
):
    """
    Create a conditional affine invertible neural network model.

    Args:
        input_size: Size of the input features (alpha coefficients)
        condition_size: Size of the conditioning features (beta coefficients)
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use

    Returns:
        CinnAffine instance
    """
    coupling_layers = []

    for i in range(n_coupling_layers):
        # Alternate between splitting at different positions for better flow
        if i % 2 == 0:
            split_dim = input_size // 2
        else:
            split_dim = input_size - input_size // 2

        layer = ConditionalAffineCoupling(
            input_size=input_size,
            condition_size=condition_size,
            hidden_sizes=hidden_sizes,
            split_dim=split_dim,
        )
        coupling_layers.append(layer)

    return CinnAffine(coupling_layers=coupling_layers)


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

    # Encode observed coefficients
    alpha_gt, _ = input_function_encoder.compute_coefficients(X, u)
    beta_gt, _ = output_function_encoder.compute_coefficients(Y, s)

    # Inverse loss: beta -> alpha -> u_pred vs u_gt
    # Train inverse: deterministic z = 0
    batch_size = beta_gt.shape[0]
    alpha_dim = model.coupling_layers[0].input_size
    z_zero = torch.zeros(
        batch_size, alpha_dim, device=beta_gt.device, dtype=beta_gt.dtype
    )

    alpha_pred = model.inverse(z_zero, beta_gt)
    inverse_loss = torch.nn.functional.mse_loss(alpha_pred, alpha_gt, reduction="mean")
    # u_pred = input_function_encoder(X, alpha_pred)
    # inverse_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    # Weak NLL-style regularization on forward pass
    z_fwd, log_det_J = model.forward(alpha_gt, beta_gt)
    latent_term = 0.5 * torch.mean(z_fwd**2)
    change_of_vars = -torch.mean(log_det_J)

    # # Forward prediction loss: alpha_gt -> z -> alpha_reconstructed vs alpha_gt
    # # Since cINN maps alpha to latent z conditioned on beta, we test round-trip consistency
    # z_pred, _ = model.forward(alpha_gt, beta_gt)
    # alpha_reconstructed = model.inverse(z_pred, beta_gt)
    # forward_loss = torch.nn.functional.mse_loss(
    #     alpha_reconstructed, alpha_gt, reduction="mean"
    # )

    return inverse_loss + (latent_term + change_of_vars)


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
    n_samples=1,
):
    """
    Compute re-simulation loss for affine cINN model.

    For cINN: Use deterministic inverse mapping (z=0) beta* -> alpha,
    apply forward operator alpha -> beta_resim, measure MSE(beta_resim, beta*).
    Since we use z=0 for deterministic evaluation, n_samples parameter is ignored.

    Re-simulation flow: beta_measured -> alpha_pred -> beta_resim -> loss(beta_resim, beta_measured)

    Returns:
        resim_coeff_loss: MSE between re-simulated and target coefficients
        resim_pred_loss: MSE between predictions from re-simulated coefficients and ground truth
    """
    X, u, Y, s = batch

    # Get target beta coefficients from observed output
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        # Deterministic inverse: beta -> alpha (using z=0 for deterministic behavior)
        batch_size = beta_target.shape[0]
        alpha_dim = model.coupling_layers[0].input_size
        z_zero = torch.zeros(
            batch_size, alpha_dim, device=beta_target.device, dtype=beta_target.dtype
        )

        alpha_pred = model.inverse(z_zero, beta_target)  # [batch_size, alpha_dim]

        # Forward re-simulation: alpha -> beta
        beta_resim = forward_model(alpha_pred)  # [batch_size, beta_dim]

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

        # Deterministic inverse with z = 0 for evaluation
        batch_size = beta.shape[0]
        alpha_dim = model.coupling_layers[0].input_size
        z_zero = torch.zeros(
            batch_size, alpha_dim, device=beta.device, dtype=beta.dtype
        )
        alpha_pred = model.inverse(z_zero, beta)

        pred = input_function_encoder(X, alpha_pred)

        return pred, alpha_pred
