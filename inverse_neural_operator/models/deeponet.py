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
    """
    Save a DeepONet model to a file.
    
    Args:
        model: The model to save
        path: Path where the model will be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path, branch_input_size, trunk_input_size, output_size, hidden_sizes=[128, 128, 128], device=None):
    """
    Load a DeepONet model from a file.
    
    Args:
        path: Path to the saved model
        branch_input_size: Size of input for the branch network (s values)
        trunk_input_size: Dimension of spatial coordinates for trunk network (X)
        output_size: Size of the output (u values)
        hidden_sizes: List of hidden layer sizes
        device: Device to load the model to ('cpu', 'cuda', etc.)
        
    Returns:
        Loaded DeepONet instance
    """
    model = create_model(branch_input_size, trunk_input_size, output_size, hidden_sizes)
    model.load_state_dict(torch.load(path, map_location=device))
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save a DeepONet checkpoint including training state.
    
    Args:
        model: The model to save
        optimizer: The optimizer used for training
        epoch: Current epoch number
        loss: Current loss value
        path: Path where the checkpoint will be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, branch_input_size, trunk_input_size, output_size, 
                   hidden_sizes=[128, 128, 128], optimizer=None, device=None):
    """
    Load a DeepONet checkpoint including training state.
    
    Args:
        path: Path to the saved checkpoint
        branch_input_size: Size of input for the branch network (s values)
        trunk_input_size: Dimension of spatial coordinates for trunk network (X)
        output_size: Size of the output (u values)
        hidden_sizes: List of hidden layer sizes
        optimizer: Optimizer to load state into (optional)
        device: Device to load the model to ('cpu', 'cuda', etc.)
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    model = create_model(branch_input_size, trunk_input_size, output_size, hidden_sizes)
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


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
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
    n_epochs=1000,
    summary_writer=None,
    model_name="deeponet",
    params=None,
    device="cpu",
):
    tqdm_bar = tqdm.tqdm(range(n_epochs))
    for epoch in range(n_epochs):
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


def evaluate(model, point):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        pred = model.inverse(s, X)

        return pred
