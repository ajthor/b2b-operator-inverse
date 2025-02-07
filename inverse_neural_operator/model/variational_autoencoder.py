import torch

import tqdm


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


class VariationalAutoencoder(torch.nn.Module):
    def __init__(
        self,
        alpha_size,
        beta_size,
        hidden_sizes: list[int] = [128, 128],
        latent_size: int = 128,
    ):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = Encoder(alpha_size + beta_size, hidden_sizes, latent_size)
        self.decoder = Decoder(alpha_size, hidden_sizes[::-1], latent_size + beta_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, alpha, beta):
        mu, logvar = self.encoder(torch.cat([alpha, beta], dim=-1))
        z = self.reparameterize(mu, logvar)

        return self.decoder(torch.cat([z, beta], dim=-1)), mu, logvar


def loss_function(model, batch, input_function_encoder, output_function_encoder):

    alpha = input_function_encoder.compute_coefficients(X, f)
    beta = output_function_encoder.compute_coefficients(Y, Tf)

    alpha_pred, mu, logvar = model(alpha, beta)

    pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")
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
    dataloader,
    optimizer,
    input_function_encoder,
    output_function_encoder,
    n_epochs=100,
):
    model.train()

    with tqdm.tqdm(range(n_epochs)) as tqdm_bar:
        for epoch in tqdm_bar:
            for batch in dataloader:
                optimizer.zero_grad()

                loss = loss_function(
                    model,
                    batch,
                    input_function_encoder,
                    output_function_encoder,
                )
                loss.backward()

                optimizer.step()

                break

            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"loss {loss.item():.4e}")
