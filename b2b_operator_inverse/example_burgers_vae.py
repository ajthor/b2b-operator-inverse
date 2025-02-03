import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

from variational_autoencoder import (
    VariationalAutoencoder,
    VariationalEncoder,
    VariationalDecoder,
)

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

# Remove all rows with nans.
ds = ds.filter(
    lambda x: not torch.isnan(x["a"]).any() and not torch.isnan(x["u"]).any()
)

dataloader = DataLoader(ds, batch_size=50, shuffle=True)


# Define model

n_basis = 100

input_basis_functions = MultiHeadedMLP(
    layer_sizes=[1, 128, 128, 128, 1], num_heads=n_basis
)
input_function_encoder = FunctionEncoder(input_basis_functions)

output_basis_functions = MultiHeadedMLP(
    layer_sizes=[1, 128, 128, 128, 1], num_heads=n_basis
)
output_function_encoder = FunctionEncoder(output_basis_functions)

# autoencoder = VariationalAutoencoder(
#     input_size=n_basis, hidden_sizes=[128, 128, 128], latent_size=n_basis
# )

encoder = VariationalEncoder(
    input_size=n_basis, hidden_sizes=[128, 128, 128], latent_size=n_basis
)

decoder = VariationalDecoder(
    latent_size=n_basis, hidden_sizes=[128, 128, 128], output_size=n_basis
)


# Train model


epochs = 1000
learning_rate = 1e-3


# Train the input function encoder
def input_loss_function(model, batch):
    x, a = batch["x"], batch["a"]
    x = x.unsqueeze(-1)
    a = a.unsqueeze(-1)

    x_size = x.size(1)

    # Split this into two sets of data. Randomly select a subset of the x, a pairs
    perm = torch.randperm(x_size, device=device)
    example_xs = x[:, perm[: x_size // 2]]
    example_ys = a[:, perm[: x_size // 2]]
    xs = x[:, perm[x_size // 2 :]]
    ys = a[:, perm[x_size // 2 :]]

    coefficients = model.compute_coefficients(example_xs, example_ys)
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)
    norm_loss = basis_normalization_loss(model.basis_functions(xs))

    return pred_loss + norm_loss


input_function_encoder = fit(
    model=input_function_encoder,
    ds=dataloader,
    loss_function=input_loss_function,
    epochs=epochs,
    learning_rate=learning_rate,
)


# Train the output function encoder
def output_loss_function(model, batch):
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
    norm_loss = basis_normalization_loss(model.basis_functions(xs))

    return pred_loss + norm_loss


output_function_encoder = fit(
    model=output_function_encoder,
    ds=dataloader,
    loss_function=output_loss_function,
    epochs=epochs,
    learning_rate=learning_rate,
)


# # Train AE
# def autoencoder_loss(model, batch):
#     a, x, u = batch["a"], batch["x"], batch["u"]
#     a = a.unsqueeze(-1)
#     x = x.unsqueeze(-1)
#     u = u.unsqueeze(-1)

#     c = input_function_encoder.compute_coefficients(x, a)
#     z = output_function_encoder.compute_coefficients(x, u)

#     # z_pred = model.encoder(c)
#     # c_pred = model.decoder(z_pred)

#     mu, logvar = model.encoder(c)
#     z_pred = model.reparameterize(mu, logvar)
#     c_pred = model.decoder(z_pred)

#     pred_loss = torch.nn.functional.mse_loss(c_pred, c)
#     latent_loss = torch.nn.functional.mse_loss(z_pred, z)

#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     kl_loss = kl_loss.mean()

#     return pred_loss + latent_loss + kl_loss


# def train_autoencoder(model, dataloader, epochs, learning_rate):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()

#     with tqdm.tqdm(range(epochs)) as tqdm_bar:
#         for epoch in tqdm_bar:
#             for batch in dataloader:
#                 optimizer.zero_grad()
#                 loss = autoencoder_loss(model, batch)
#                 loss.backward()
#                 optimizer.step()
#                 break

#             if epoch % 10 == 0:
#                 tqdm_bar.set_postfix_str(f"Loss {loss.item()}")


# train_autoencoder(autoencoder, dataloader, epochs=2000, learning_rate=learning_rate)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std, device=device)
    return mu + eps * std


# Train encoder
def encoder_loss(model, batch):
    a, x, u = batch["a"], batch["x"], batch["u"]
    a = a.unsqueeze(-1)
    x = x.unsqueeze(-1)
    u = u.unsqueeze(-1)

    c = input_function_encoder.compute_coefficients(x, a)
    z = output_function_encoder.compute_coefficients(x, u)

    mu, logvar = model(c)
    z_pred = reparameterize(mu, logvar)

    pred_loss = torch.nn.functional.mse_loss(z_pred, z)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss.mean()

    return pred_loss + kl_loss


def train_encoder(model, dataloader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    with tqdm.tqdm(range(epochs)) as tqdm_bar:
        for epoch in tqdm_bar:
            for batch in dataloader:
                optimizer.zero_grad()
                loss = encoder_loss(model, batch)
                loss.backward()
                optimizer.step()
                break

            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss {loss.item()}")


train_encoder(encoder, dataloader, epochs=2000, learning_rate=learning_rate)


# Train decoder
def decoder_loss(model, batch):
    a, x, u = batch["a"], batch["x"], batch["u"]
    a = a.unsqueeze(-1)
    x = x.unsqueeze(-1)
    u = u.unsqueeze(-1)

    c = input_function_encoder.compute_coefficients(x, a)
    z = output_function_encoder.compute_coefficients(x, u)

    mu, logvar = encoder(c)
    z_pred = reparameterize(mu, logvar)

    c_pred = model(z_pred)

    pred_loss = torch.nn.functional.mse_loss(c_pred, c)

    return pred_loss


def train_decoder(model, dataloader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    with tqdm.tqdm(range(epochs)) as tqdm_bar:
        for epoch in tqdm_bar:
            for batch in dataloader:
                optimizer.zero_grad()
                loss = decoder_loss(model, batch)
                loss.backward()
                optimizer.step()
                break

            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss {loss.item()}")


train_decoder(decoder, dataloader, epochs=2000, learning_rate=learning_rate)


# Save models

# torch.save(function_encoder.state_dict(), "function_encoder.pt")
# torch.save(autoencoder.state_dict(), "autoencoder.pt")


# Plot

input_function_encoder.eval()
output_function_encoder.eval()
# autoencoder.eval()
encoder.eval()
decoder.eval()

point = ds.take(1)[0]

x = point["x"]
a = point["a"]
u = point["u"]

idx = torch.argsort(x, dim=0).squeeze()

x = x[idx]
a = a[idx]
u = u[idx]

x = x.unsqueeze(-1).unsqueeze(0)
a = a.unsqueeze(-1).unsqueeze(0)
u = u.unsqueeze(-1).unsqueeze(0)

c = input_function_encoder.compute_coefficients(x, a)
z = output_function_encoder.compute_coefficients(x, u)

a_est = input_function_encoder(x, c)
u_est = output_function_encoder(x, z)

mu, logvar = encoder(c)
z_pred = reparameterize(mu, logvar)
c_pred = decoder(z_pred)

# mu, logvar = autoencoder.encoder(c)
# z_pred = autoencoder.reparameterize(mu, logvar)
# c_pred = autoencoder.decoder(z_pred)

a_pred = input_function_encoder(x, c_pred)
u_pred = output_function_encoder(x, z_pred)

x = x.squeeze().cpu().detach().numpy()
a = a.squeeze().cpu().detach().numpy()
u = u.squeeze().cpu().detach().numpy()
a_est = a_est.squeeze().cpu().detach().numpy()
u_est = u_est.squeeze().cpu().detach().numpy()
a_pred = a_pred.squeeze().cpu().detach().numpy()
u_pred = u_pred.squeeze().cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x, a, label="a")
ax[0].plot(x, a_est, label="a_est")
ax[0].plot(x, a_pred, label="a_pred")

ax[1].plot(x, u, label="u")
ax[1].plot(x, u_est, label="u_est")
ax[1].plot(x, u_pred, label="u_pred")

plt.show()
