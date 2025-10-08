import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class DeepONet(torch.nn.Module):
    """
    Deep Operator Network (DeepONet) for mapping between function spaces.

    This implementation takes output function values (s) as input to the branch network
    and evaluation points (X) as input to the trunk network to predict the input
    function values (u).
    """

    def __init__(
        self,
        branch_net,
        trunk_net,
        output_channels=1,
    ):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.output_channels = output_channels

        # Initialize bias term
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))

    def forward(self, s, X):
        """
        Forward is not used in this inverse-operator variant. Use `.inverse(s, X)` instead.
        """
        return None

    def inverse(self, s, X):
        """
        Maps from output function values (s) and evaluation points (X) to
        input function values (u).

        Shapes:
            - s: (B, S)  arbitrary branch input per sample
            - X: (B, d) for a single evaluation point per sample, or (B, N, d)
                 for N evaluation points per sample.

        Returns:
            - u_pred: (B, C) if X is (B, d), or (B, N, C) if X is (B, N, d),
              where C == self.output_channels.
        """
        # Process through branch network (processes output function s)
        branch_output = self.branch_net(s)  # (B, C*D)

        # Process through trunk network (processes evaluation points X)
        trunk_output = self.trunk_net(X)  # (B, D) or (B, N, D)

        # Determine dimensions
        B = s.shape[0]
        C = self.output_channels

        # Reshape branch to (B, C, D)
        try:
            branch_output = branch_output.view(B, C, -1)
        except RuntimeError as e:
            raise RuntimeError(
                f"Branch output shape {tuple(branch_output.shape)} is not compatible with output_channels={C}. "
                f"Expected last dim to be a multiple of C."
            ) from e

        # Ensure trunk has an explicit point dimension N: (B, N, D)
        if trunk_output.dim() == 2:
            trunk_output = trunk_output.unsqueeze(1)  # (B, 1, D)
        elif trunk_output.dim() != 3:
            raise RuntimeError(
                f"Trunk output must be rank-2 or rank-3. Got shape {tuple(trunk_output.shape)}"
            )

        # Validate feature dimension match (D)
        if branch_output.shape[-1] != trunk_output.shape[-1]:
            raise RuntimeError(
                f"Dot-product feature mismatch: branch D={branch_output.shape[-1]} vs trunk D={trunk_output.shape[-1]}"
            )

        # Compute inner product along D to get (B, C, N)
        # Using einsum for clarity and to support broadcasting over N
        output = torch.einsum("bcd,bnd->bcn", branch_output, trunk_output)

        # Add bias (C,) across N evaluation points
        output = output + self.bias.view(1, C, 1)

        # Return as (B, N, C) for consistency with typical datasets
        output = output.permute(0, 2, 1).contiguous()

        # If there was only a single point (N==1), optionally squeeze the N dim
        if output.shape[1] == 1:
            output = output.squeeze(1)  # (B, C)

        return output


def create_mlp(input_size, hidden_sizes, output_size, activation=torch.nn.ReLU()):
    """Helper function to create a simple MLP."""
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(layer_sizes) - 1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:  # No activation after the last layer
            layers.append(activation)

    return torch.nn.Sequential(*layers)


def create_model(
    branch_input_size,  # Size of input for the branch network (s values)
    trunk_input_size,  # Dimension of spatial coordinates for trunk network (X)
    output_size,  # Size of the output (u values)
    hidden_sizes=[128, 128, 128],
):
    """
    Create a DeepONet model.

    Args:
        branch_input_size: Size of input for the branch network (s values)
        trunk_input_size: Dimension of spatial coordinates for trunk network (X)
        output_size: Size of the output (u values)
        hidden_sizes: List of hidden layer sizes

    Returns:
        DeepONet instance
    """
    # Width of the last hidden layer, used for branch-trunk dot product
    dot_product_dim = hidden_sizes[-1]

    # Create branch network to process output function values (s)
    branch_net = create_mlp(
        input_size=branch_input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size * dot_product_dim,
        activation=torch.nn.ReLU(),
    )

    # Create trunk network to process spatial coordinates (X)
    trunk_net = create_mlp(
        input_size=trunk_input_size,
        hidden_sizes=hidden_sizes,
        output_size=dot_product_dim,
        activation=torch.nn.ReLU(),
    )

    # Create DeepONet
    return DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net,
        output_channels=output_size,
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


def loss_function(model, batch):
    """Loss function that works directly with the raw data without function encoders."""
    X, u, Y, s = batch

    # Predict input function values from output function values and evaluation points
    u_pred = model.inverse(s, X)

    # Compute the MSE loss
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    return pred_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    input_function_encoder,  # Not used but kept for API compatibility
    output_function_encoder,  # Not used but kept for API compatibility
    n_epochs,
    summary_writer,
    model_name,
    params,
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

    train_dataloader_iter = iter(train_dataloader)
    test_dataloader_iter = iter(test_dataloader)
    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
        model.train()
        batch = next(train_dataloader_iter)
        optimizer.zero_grad()
        loss = loss_function(
            model=model,
            batch=batch,
        )
        loss.backward()
        optimizer.step()

        if summary_writer:
            summary_writer.add_scalars("loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = test_model(
            model=model,
            test_dataloader_iter=test_dataloader_iter,
        )

        if summary_writer:
            summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader_iter,
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
):
    model.eval()
    with torch.no_grad():
        batch = next(test_dataloader_iter)
        loss = loss_function(
            model=model,
            batch=batch,
        )

    return loss.item()


def resimulation_loss(model, batch, input_function_encoder, output_function_encoder, forward_model, n_samples=5):
    """
    Compute re-simulation loss for DeepONet model.
    
    For DeepONet: The model predicts u from s directly, so this is not applicable.
    Returns 0.0 as placeholder.
    """
    return 0.0


def evaluate(model, point, input_function_encoder=None, output_function_encoder=None):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        pred = model.inverse(s, X)

        return pred
