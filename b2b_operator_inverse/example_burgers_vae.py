import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

from variational_autoencoder import VariationalAutoencoder

import tqdm

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Load dataset

ds = load_dataset("ajthor/burgers_1d", split="train")
ds = ds.with_format("torch", device=device)

dataloader = DataLoader(ds, batch_size=5)


# Define model

n_basis = 8

basis_functions = MultiHeadedMLP(layer_sizes=[1, 128, 128, 1], num_heads=n_basis)
function_encoder = FunctionEncoder(basis_functions)

vae = VariationalAutoencoder(
    input_size=n_basis, hidden_sizes=[128], latent_size=n_basis
)


# Train model


# Train function encoder
def function_encoder_loss(model, batch):
    x, u = batch["x"], batch["u"]
    x = x.unsqueeze(-1)
    u = u.unsqueeze(-1)

    x_size = x.size(1)

    # Split this into two sets of data. Randomly select a subset of the x, u pairs
    perm = torch.randperm(x_size, device=device)
    example_xs = x[:, perm[: x_size // 2]]
    example_us = u[:, perm[: x_size // 2]]
    xs = x[:, perm[x_size // 2 :]]
    us = u[:, perm[x_size // 2 :]]

    coefficients = model.compute_coefficients(example_xs, example_us)
    u_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(u_pred, us)

    return pred_loss


function_encoder = fit(
    model=function_encoder,
    ds=dataloader,
    loss_function=function_encoder_loss,
    epochs=1000,
)


# Train VAE
def vae_loss(model, batch):
    a, x, u = batch["a"], batch["x"], batch["u"]
    a = a.unsqueeze(-1)
    x = x.unsqueeze(-1)
    u = u.unsqueeze(-1)

    y = function_encoder.compute_coefficients(x, a)
    z = function_encoder.compute_coefficients(x, u)

    mu, logvar = model.encoder(y)
    z_pred = model.reparameterize(mu, logvar)
    y_pred = model.decoder(z_pred)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.mean()

    latent_loss = torch.nn.functional.mse_loss(z_pred, z)

    return pred_loss + kl_loss + latent_loss


epochs = 1000
learning_rate = 1e-3
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
vae.train()

with tqdm.tqdm(range(epochs)) as tqdm_bar:
    for i, epoch in enumerate(tqdm_bar):

        for batch in dataloader:

            optimizer.zero_grad()

            loss = vae_loss(vae, batch)
            loss.backward()

            optimizer.step()

            break

        if i % 10 == 0:
            tqdm_bar.set_postfix_str(f"Loss: {loss.item():.2f}")


# Plot results
