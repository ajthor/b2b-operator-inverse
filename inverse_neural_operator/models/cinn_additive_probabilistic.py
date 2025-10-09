import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class ConditionalAdditiveCoupling(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
        split_dim=None,
        activation=torch.nn.ReLU(),
    ):
        super(ConditionalAdditiveCoupling, self).__init__()

        self.input_size = input_size
        self.condition_size = condition_size
        if split_dim is None:
            self.split_dim = input_size // 2
        else:
            self.split_dim = split_dim

        # Neural network to transform the second part conditioned on [x1, y]
        # We keep x1 unchanged and transform x2 using f(x1, y):
        #   y1 = x1
        #   y2 = x2 + f(x1, y)
        # Therefore, net input dim = split_dim (x1) + condition_size, output dim = input_size - split_dim (x2)
        self.net = torch.nn.Sequential()

        # Create layers with the specified hidden sizes
        layer_sizes = (
            [self.split_dim + condition_size]
            + hidden_sizes
            + [input_size - self.split_dim]
        )
        for i in range(len(layer_sizes) - 1):
            self.net.add_module(
                f"linear_{i}", torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.net.add_module(f"activation_{i}", activation)

    def forward(self, x, condition):
        """
        Forward transformation: x -> y conditioned on condition
        Implements the conditional additive coupling layer: y1 = x1, y2 = x2 + f(x1, condition)
        """
        x1, x2 = torch.split(x, [self.split_dim, x.size(-1) - self.split_dim], dim=-1)
        y1 = x1
        # Concatenate x1 and condition for the neural network input
        net_input = torch.cat([x1, condition], dim=-1)
        y2 = x2 + self.net(net_input)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y, condition):
        """
        Inverse transformation: y -> x conditioned on condition
        Implements the inverse of conditional additive coupling: x1 = y1, x2 = y2 - f(y1, condition)
        """
        y1, y2 = torch.split(y, [self.split_dim, y.size(-1) - self.split_dim], dim=-1)
        x1 = y1
        # Concatenate y1 and condition for the neural network input
        net_input = torch.cat([y1, condition], dim=-1)
        x2 = y2 - self.net(net_input)
        return torch.cat([x1, x2], dim=-1)


class ConditionalInvertibleNeuralNetworkProbabilistic(torch.nn.Module):
    def __init__(self, coupling_layers):
        super(ConditionalInvertibleNeuralNetworkProbabilistic, self).__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)

    def forward(self, alpha, beta):
        """
        Forward transformation: transform alpha to latent z conditioned on beta
        This is the key difference from standard INN - we transform x to z given y
        """
        x = alpha
        for layer in self.coupling_layers:
            x = layer.forward(x, beta)
        return x

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
    Create a conditional invertible neural network model.

    Args:
        input_size: Size of the input features (alpha coefficients)
        condition_size: Size of the conditioning features (beta coefficients)
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use

    Returns:
        ConditionalInvertibleNeuralNetworkProbabilistic instance
    """
    coupling_layers = []

    for i in range(n_coupling_layers):
        # Alternate between splitting at different positions for better flow
        if i % 2 == 0:
            split_dim = input_size // 2
        else:
            split_dim = input_size - input_size // 2

        layer = ConditionalAdditiveCoupling(
            input_size=input_size,
            condition_size=condition_size,
            hidden_sizes=hidden_sizes,
            split_dim=split_dim,
        )
        coupling_layers.append(layer)

    return ConditionalInvertibleNeuralNetworkProbabilistic(coupling_layers=coupling_layers)


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
    """
    Canonical cINN training via maximum likelihood (NLL loss).

    Train the model to maximize p(alpha | beta) by minimizing the negative log-likelihood:
    -log p(alpha | beta) = -log p(z) - log |det J|
                         = 0.5 * ||z||^2 + 0.5 * log(2π) * dim(z) - 0

    For additive coupling, log|det J| = 0, so we only need the Gaussian prior term.
    """
    X, u, Y, s = batch

    # Encode coefficients
    alpha_gt, _ = input_function_encoder.compute_coefficients(X, u)
    beta_gt, _ = output_function_encoder.compute_coefficients(Y, s)

    # Forward pass: alpha -> z conditioned on beta
    z = model.forward(alpha_gt, beta_gt)

    # Negative log-likelihood loss
    # log p(z) for standard normal: -0.5 * ||z||^2 - 0.5 * dim * log(2π)
    # We minimize -log p(z) = 0.5 * ||z||^2 + constant
    # For additive coupling, log|det J| = 0, so no change-of-variables term
    # Sum over latent dimensions so the prior keeps its intended scale
    nll_loss = 0.5 * torch.mean(torch.sum(z ** 2, dim=-1))

    return nll_loss


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
                n_samples=10,
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
    n_samples=10,
):
    """
    Compute re-simulation loss for probabilistic additive cINN model.

    For probabilistic cINN: Sample z ~ N(0, I), map beta* -> alpha via inverse,
    apply forward operator alpha -> beta_resim, measure MSE(beta_resim, beta*).
    Average over multiple samples to get expected performance.

    Re-simulation flow: beta_measured -> sample z -> alpha_pred -> beta_resim -> loss(beta_resim, beta_measured)

    Returns:
        resim_coeff_loss: MSE between re-simulated and target coefficients (averaged over samples)
        resim_pred_loss: MSE between predictions from re-simulated coefficients and ground truth (averaged over samples)
    """
    X, u, Y, s = batch

    # Get target beta coefficients from observed output
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        batch_size = beta_target.shape[0]
        alpha_dim = model.coupling_layers[0].input_size

        resim_coeff_losses = []
        resim_pred_losses = []

        for _ in range(n_samples):
            # Sample z from standard normal distribution
            z = torch.randn(batch_size, alpha_dim, device=beta_target.device, dtype=beta_target.dtype)

            # Stochastic inverse: beta -> alpha using sampled z
            alpha_pred = model.inverse(z, beta_target)  # [batch_size, alpha_dim]

            # Forward re-simulation: alpha -> beta
            beta_resim = forward_model(alpha_pred)  # [batch_size, beta_dim]

            # Coefficient error: re-simulated beta vs target beta
            resim_coeff_loss = torch.nn.functional.mse_loss(beta_resim, beta_target)
            resim_coeff_losses.append(resim_coeff_loss.item())

            # Prediction error: function predictions using re-simulated beta
            s_pred = output_function_encoder(Y, beta_resim)
            resim_pred_loss = torch.nn.functional.mse_loss(s_pred, s)
            resim_pred_losses.append(resim_pred_loss.item())

    model.train()

    # Return average over all samples
    return np.mean(resim_coeff_losses), np.mean(resim_pred_losses)


def evaluate(model, point, input_function_encoder, output_function_encoder):
    """
    Evaluate conditional invertible network using probabilistic sampling.

    Given output observation (Y, s), predict input (u) by:
    1. Encode output to beta coefficients
    2. Sample z ~ N(0, I) (one sample per call)
    3. Transform z to alpha conditioned on beta
    4. Reconstruct input function from alpha
    5. Return prediction and alpha for this sample

    Note: For probabilistic behavior, call this function multiple times to get different samples.
    """
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        # Encode the output observation to beta coefficients
        beta, _ = output_function_encoder.compute_coefficients(Y, s)

        batch_size = beta.shape[0]
        alpha_dim = model.coupling_layers[0].input_size

        # Sample latent z from standard normal (one sample per call)
        z = torch.randn(batch_size, alpha_dim, device=beta.device, dtype=beta.dtype)

        # Transform latent z to input coefficients alpha conditioned on beta
        alpha_pred = model.inverse(z, beta)

        # Reconstruct input function from predicted coefficients
        pred = input_function_encoder(X, alpha_pred)

        return pred, alpha_pred
