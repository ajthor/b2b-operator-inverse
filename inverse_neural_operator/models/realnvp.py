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
        # Output dimension is 2 * remaining_dim (for both scale and translation)
        remaining_dim = input_size - self.split_dim
        layer_sizes = [self.split_dim] + hidden_sizes + [2 * remaining_dim]

        for i in range(len(layer_sizes) - 1):
            self.net.add_module(
                f"linear_{i}", torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.net.add_module(f"activation_{i}", activation)

    def forward(self, x):
        """
        Forward transformation: x -> y
        Implements the affine coupling layer: y1 = x1, y2 = x2 * exp(s(x1)) + t(x1)
        Returns: (y, log_det_jacobian)
        """
        x1, x2 = torch.split(x, [self.split_dim, x.size(-1) - self.split_dim], dim=-1)

        # Compute scale and translation
        net_output = self.net(x1)
        s, t = torch.chunk(net_output, 2, dim=-1)

        # s = torch.tanh(s)  # Ensure scale is bounded

        # Apply affine transformation
        y1 = x1
        y2 = x2 * torch.exp(s) + t

        # Log determinant of Jacobian
        log_det_J = torch.sum(s, dim=-1)

        return torch.cat([y1, y2], dim=-1), log_det_J

    def inverse(self, y):
        """
        Inverse transformation: y -> x
        Implements the inverse of affine coupling: x1 = y1, x2 = (y2 - t(y1)) / exp(s(y1))
        """
        y1, y2 = torch.split(y, [self.split_dim, y.size(-1) - self.split_dim], dim=-1)

        # Compute scale and translation
        net_output = self.net(y1)
        s, t = torch.chunk(net_output, 2, dim=-1)

        # s = torch.tanh(s)  # Ensure scale is bounded

        # Apply inverse affine transformation
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=-1)


class RealNVP(torch.nn.Module):
    def __init__(self, coupling_layers):
        super(RealNVP, self).__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)

    def forward(self, alpha):
        """
        Forward transformation, applies all coupling layers in sequence.
        Returns: (beta, log_det_jacobian)
        """
        x = alpha
        log_det_J_total = 0.0

        for layer in self.coupling_layers:
            x, log_det_J = layer.forward(x)
            log_det_J_total += log_det_J

        return x, log_det_J_total

    def inverse(self, beta):
        """
        Inverse transformation, applies all coupling layers in reverse order.
        """
        y = beta
        for layer in reversed(self.coupling_layers):
            y = layer.inverse(y)
        return y

    def log_prob(self, alpha, prior_log_prob_fn=None):
        """
        Compute log probability of alpha under the learned distribution.
        """
        if prior_log_prob_fn is None:
            # Default to standard Gaussian prior
            def prior_log_prob_fn(z):
                D = z.size(-1)
                log2pi = torch.log(
                    torch.tensor(2.0 * np.pi, device=z.device, dtype=z.dtype)
                )
                return -0.5 * torch.sum(z**2, dim=-1) - 0.5 * D * log2pi

        beta, log_det_J = self.forward(alpha)
        log_prob_prior = prior_log_prob_fn(beta)

        return log_prob_prior + log_det_J


def create_model(input_size, hidden_sizes=[128, 128], n_coupling_layers=2):
    """
    Create a RealNVP model.

    Args:
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use

    Returns:
        RealNVP instance
    """
    coupling_layers = []

    for i in range(n_coupling_layers):
        # Alternate between splitting at different positions for better flow
        if i % 2 == 0:
            split_dim = input_size // 2
        else:
            split_dim = (input_size + 1) // 2

        layer = AffineCoupling(
            input_size=input_size, hidden_sizes=hidden_sizes, split_dim=split_dim
        )
        coupling_layers.append(layer)

    return RealNVP(coupling_layers=coupling_layers)


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

    alpha, _ = input_function_encoder.compute_coefficients(X, u)
    beta, _ = output_function_encoder.compute_coefficients(Y, s)

    # Compute negative log-likelihood
    log_prob = model.log_prob(alpha)
    nll_loss = -log_prob.mean()

    # Reconstruction loss in function space
    alpha_pred = model.inverse(beta)
    u_pred = input_function_encoder(X, alpha_pred)
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    return nll_loss + pred_loss


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

    # train_dataloader_iter = iter(train_dataloader)
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
