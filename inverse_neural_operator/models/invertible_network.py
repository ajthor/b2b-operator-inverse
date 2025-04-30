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

        # Neural network to transform the first part
        self.net = torch.nn.Sequential()

        # Create layers with the specified hidden sizes
        layer_sizes = [input_size - self.split_dim] + hidden_sizes + [self.split_dim]
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


class InvertibleNeuralNetwork(torch.nn.Module):
    def __init__(self, coupling_layers):
        super(InvertibleNeuralNetwork, self).__init__()
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


def create_model(input_size, hidden_sizes=[128, 128], n_coupling_layers=2):
    """
    Create an invertible neural network model.

    Args:
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use

    Returns:
        InvertibleNeuralNetwork instance
    """
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

    return InvertibleNeuralNetwork(coupling_layers=coupling_layers)


def save(model, path):
    """
    Save an invertible network model to a file.
    
    Args:
        model: The model to save
        path: Path where the model will be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path, input_size, hidden_sizes=[128, 128], n_coupling_layers=2, device=None):
    """
    Load an invertible network model from a file.
    
    Args:
        path: Path to the saved model
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use
        device: Device to load the model to ('cpu', 'cuda', etc.)
        
    Returns:
        Loaded InvertibleNeuralNetwork instance
    """
    model = create_model(input_size, hidden_sizes, n_coupling_layers)
    model.load_state_dict(torch.load(path, map_location=device))
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save an invertible network checkpoint including training state.
    
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


def load_checkpoint(path, input_size, hidden_sizes=[128, 128], n_coupling_layers=2, 
                   optimizer=None, device=None):
    """
    Load an invertible network checkpoint including training state.
    
    Args:
        path: Path to the saved checkpoint
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the coupling layers
        n_coupling_layers: Number of coupling layers to use
        optimizer: Optimizer to load state into (optional)
        device: Device to load the model to ('cpu', 'cuda', etc.)
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    model = create_model(input_size, hidden_sizes, n_coupling_layers)
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha = input_function_encoder.compute_coefficients(X, u)
    beta = output_function_encoder.compute_coefficients(Y, s)

    alpha_pred = model.inverse(beta)

    u_pred = input_function_encoder(X, alpha_pred)

    # pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

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
    device,
):

    tqdm_bar = tqdm.tqdm(range(n_epochs))
    for epoch in range(n_epochs):
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

        beta = output_function_encoder.compute_coefficients(Y, s)
        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)

        return pred
