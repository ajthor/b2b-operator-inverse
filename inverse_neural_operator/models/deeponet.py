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
        Not used for the inverse problem.
        """
        return None

    def inverse(self, s, X):
        """
        Maps from output function values (s) and evaluation points (X) to
        input function values (u).

        Args:
            s: Output function values
            X: Spatial coordinates for evaluation

        Returns:
            Predicted input function values (u) at points X
        """
        # Process through branch network (processes output function s)
        branch_output = self.branch_net(s)

        # Process through trunk network (processes evaluation points X)
        trunk_output = self.trunk_net(X)

        # Reshape for dot product
        batch_size = s.shape[0]
        branch_output = branch_output.view(batch_size, self.output_channels, -1)
        trunk_output = trunk_output.view(batch_size, -1, 1)

        # Compute the output with bias
        output = torch.bmm(branch_output, trunk_output).squeeze(-1) + self.bias

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

    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
        model.train()
        batch = next(iter(train_dataloader))
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
            test_dataloader=test_dataloader,
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
    test_dataloader,
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            loss = loss_function(
                model=model,
                batch=batch,
            )
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate(model, point, input_function_encoder=None, output_function_encoder=None):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        pred = model.inverse(s, X)

        return pred
