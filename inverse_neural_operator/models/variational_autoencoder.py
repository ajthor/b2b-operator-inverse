import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [128, 128],
        latent_size: int = 128,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size

        self.activation = activation

        self.layers = torch.nn.ModuleList()

        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias),
            )

        self.mu = torch.nn.Linear(hidden_sizes[-1], latent_size, bias=bias)
        self.logvar = torch.nn.Linear(hidden_sizes[-1], latent_size, bias=bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar


class Decoder(torch.nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_sizes: list[int] = [128, 128],
        latent_size: int = 128,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.activation = activation

        self.layers = torch.nn.ModuleList()

        sizes = [latent_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias),
            )

        self.output_activation = torch.nn.Identity()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        x = self.output_activation(x)

        return x


class ConditionalVariationalAutoencoder(torch.nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_size: int = 128,
    ):
        super(ConditionalVariationalAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.latent_size = latent_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)

        return mu + eps * std

    def sample_prior(self, batch_size, device=None):
        z = torch.randn(batch_size, self.latent_size, device=device)
        return z

    def forward(self, alpha, beta):
        mu, logvar = self.encoder(torch.cat([alpha, beta], dim=-1))
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def inverse(self, beta, z):
        return self.decoder(torch.cat([z, beta], dim=-1))


def create_model(
    alpha_size: int,
    beta_size: int,
    hidden_sizes: list[int] = [128, 128],
    latent_size: int = 128,
):
    encoder = Encoder(
        input_size=alpha_size + beta_size,
        hidden_sizes=hidden_sizes,
        latent_size=latent_size,
    )

    decoder = Decoder(
        output_size=alpha_size,
        hidden_sizes=hidden_sizes[::-1],
        latent_size=latent_size + beta_size,
    )

    return ConditionalVariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_size=latent_size,
    )


def save(model, path):
    """
    Save a conditional variational autoencoder model to a file.
    
    Args:
        model: The model to save
        path: Path where the model will be saved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path, alpha_size, beta_size, hidden_sizes=[128, 128], latent_size=128, device=None):
    """
    Load a conditional variational autoencoder model from a file.
    
    Args:
        path: Path to the saved model
        alpha_size: Size of the input coefficients
        beta_size: Size of the output coefficients
        hidden_sizes: List of hidden layer sizes
        latent_size: Size of the latent space
        device: Device to load the model to ('cpu', 'cuda', etc.)
        
    Returns:
        Loaded ConditionalVariationalAutoencoder instance
    """
    model = create_model(alpha_size, beta_size, hidden_sizes, latent_size)
    model.load_state_dict(torch.load(path, map_location=device))
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save a conditional variational autoencoder checkpoint including training state.
    
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


def load_checkpoint(path, alpha_size, beta_size, hidden_sizes=[128, 128], latent_size=128, 
                   optimizer=None, device=None):
    """
    Load a conditional variational autoencoder checkpoint including training state.
    
    Args:
        path: Path to the saved checkpoint
        alpha_size: Size of the input coefficients
        beta_size: Size of the output coefficients
        hidden_sizes: List of hidden layer sizes
        latent_size: Size of the latent space
        optimizer: Optimizer to load state into (optional)
        device: Device to load the model to ('cpu', 'cuda', etc.)
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    model = create_model(alpha_size, beta_size, hidden_sizes, latent_size)
    
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

    z, mu, logvar = model(alpha, beta)
    alpha_pred = model.inverse(beta, z)

    u_pred = input_function_encoder(X, alpha_pred)

    # pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss.mean()

    # # consistency loss
    # consistency_loss = torch.nn.functional.mse_loss(
    #     beta,
    #     torch.einsum(
    #         "kl,bk->bl", operator, alpha_pred
    #     ),  # torch.matmul(alpha_pred, operator.T)
    # )

    return pred_loss + kl_loss  # + consistency_loss


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
        z = model.sample_prior(1, device=X.device)
        alpha_pred = model.inverse(beta, z)
        pred = input_function_encoder(X, alpha_pred)

        return pred
