# Interface of a variational autoencoder. Torch module with an init and forward.

import torch
import sys


class VariationalEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        latent_size,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(VariationalEncoder, self).__init__()

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


class VariationalDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size,
        hidden_sizes,
        output_size,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(VariationalDecoder, self).__init__()

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


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = VariationalEncoder(input_size, hidden_sizes, latent_size)
        self.decoder = VariationalDecoder(latent_size, hidden_sizes[::-1], input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class CustomVariationalAutoencoder(torch.nn.Module):
    def __init__(self, alpha_size, beta_size, hidden_sizes, latent_size):
        super(CustomVariationalAutoencoder, self).__init__()

        self.encoder = VariationalEncoder(
            alpha_size + beta_size, hidden_sizes, latent_size
        )
        self.decoder = VariationalDecoder(
            latent_size + beta_size, hidden_sizes[::-1], alpha_size
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, alpha, beta):
        mu, logvar = self.encoder(torch.cat([alpha, beta], dim=-1))
        z = self.reparameterize(mu, logvar)

        if torch.isnan(z).any():
            print("NaN in z")

        if torch.isnan(mu).any():
            print("mu is nan")

        if torch.isnan(logvar).any():
            print("logvar is nan")

        return self.decoder(torch.cat([z, beta], dim=-1)), mu, logvar
