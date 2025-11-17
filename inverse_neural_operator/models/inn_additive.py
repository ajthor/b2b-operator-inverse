import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class AdditiveCoupling(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[128, 128],
        split_dim=None,
        activation=torch.nn.ReLU(),
    ):
        super(AdditiveCoupling, self).__init__()

        self.input_size = input_size
        if split_dim is None:
            self.split_dim = input_size // 2
        else:
            self.split_dim = split_dim

        # Neural network to transform the second part conditioned on the first
        self.net = torch.nn.Sequential()

        # Create layers with the specified hidden sizes
        # We keep x1 unchanged and transform x2 using f(x1):
        #   y1 = x1
        #   y2 = x2 + f(x1)
        # Therefore, net input dim = split_dim (x1), output dim = input_size - split_dim (x2)
        layer_sizes = [self.split_dim] + hidden_sizes + [input_size - self.split_dim]
        for i in range(len(layer_sizes) - 1):
            self.net.add_module(
                f"linear_{i}", torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.net.add_module(f"activation_{i}", activation)

    def forward(self, x):
        """
        Forward transformation: x -> y
        Implements the additive coupling layer: y1 = x1, y2 = x2 + f(x1)
        """
        x1, x2 = torch.split(x, [self.split_dim, x.size(-1) - self.split_dim], dim=-1)
        y1 = x1
        y2 = x2 + self.net(x1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        """
        Inverse transformation: y -> x
        Implements the inverse of additive coupling: x1 = y1, x2 = y2 - f(y1)
        """
        y1, y2 = torch.split(y, [self.split_dim, y.size(-1) - self.split_dim], dim=-1)
        x1 = y1
        x2 = y2 - self.net(y1)
        return torch.cat([x1, x2], dim=-1)


class InnAdditive(torch.nn.Module):
    def __init__(self, coupling_layers):
        super(InnAdditive, self).__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)

    def forward(self, alpha):
        """
        Forward transformation, applies all coupling layers in sequence.
        """
        x = alpha
        for layer in self.coupling_layers:
            x = layer.forward(x)
        return x

    def inverse(self, beta):
        """
        Inverse transformation, applies all coupling layers in reverse order.
        """
        y = beta
        for layer in reversed(self.coupling_layers):
            y = layer.inverse(y)
        return y

    def sample_posterior(self, beta, n_samples):
        """
        Sample from the posterior distribution given observed beta.
        For additive INN, this is just the deterministic inverse mapping.

        Args:
            beta: Observed output coefficients [batch_size, beta_dim]
            n_samples: Number of samples to generate (ignored for deterministic mapping)

        Returns:
            samples: Generated alpha samples [n_samples, batch_size, alpha_dim]
        """
        alpha_sample = self.inverse(beta)
        # For deterministic mapping, return the same sample n_samples times
        return alpha_sample.unsqueeze(0).repeat(n_samples, 1, 1)


def create_model(
    input_size, output_size=None, hidden_sizes=[128, 128], n_coupling_layers=2
):
    """
    Create an invertible neural network model.

    Args:
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use

    Returns:
        InvertibleNeuralNetwork instance
    """
    # For additive INN, input_size must equal output_size
    if output_size is not None and input_size != output_size:
        raise ValueError(
            f"For additive INN, input_size ({input_size}) must equal output_size ({output_size})"
        )

    coupling_layers = []

    for i in range(n_coupling_layers):
        # Alternate between splitting at different positions for better flow
        if i % 2 == 0:
            split_dim = input_size // 2
        else:
            split_dim = input_size - input_size // 2

        layer = AdditiveCoupling(
            input_size=input_size, hidden_sizes=hidden_sizes, split_dim=split_dim
        )
        coupling_layers.append(layer)

    return InnAdditive(coupling_layers=coupling_layers)


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

    # Inverse loss: reconstruct boundary forces directly
    alpha_pred = model.inverse(beta)
    u_pred = input_function_encoder(X, alpha_pred)
    inverse_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    # Forward prediction loss: alpha_gt -> beta_pred vs beta_gt
    beta_pred = model.forward(alpha)
    forward_loss = torch.nn.functional.mse_loss(beta_pred, beta, reduction="mean")
    # s_pred = output_function_encoder(Y, beta_pred)
    # forward_loss = torch.nn.functional.mse_loss(s_pred, s, reduction="mean")

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

        # Compute and log re-simulation loss on single batch
        with torch.no_grad():
            test_batch = next(iter(test_dataloader))
            resim_coeff_loss, resim_pred_loss = resimulation_loss(
                model=model,
                batch=test_batch,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                forward_model=forward_model,
                n_samples=1,  # Use deterministic evaluation
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
    Compute re-simulation loss for additive INN model.

    For additive INN: Use deterministic inverse mapping beta* -> alpha,
    apply forward operator alpha -> beta_resim, measure MSE(beta_resim, beta*).
    Since additive INN is deterministic, n_samples parameter is ignored.

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
        # Deterministic inverse: beta -> alpha (no sampling needed)
        alpha_raw = model.inverse(beta_target)  # [batch_size, alpha_dim]
        # Reconstruct boundary forces and re-encode coefficients for consistency
        u_pred = input_function_encoder(X, alpha_raw)
        alpha_pred, _ = input_function_encoder.compute_coefficients(X, u_pred)

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

        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)

        alpha_encoded, _ = input_function_encoder.compute_coefficients(X, pred)

        return pred, alpha_encoded
