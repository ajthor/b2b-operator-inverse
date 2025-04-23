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


class ConditionalVariationalAutoencoderFactory:
    @staticmethod
    def create(
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


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha = input_function_encoder.compute_coefficients(X, u)
    beta = output_function_encoder.compute_coefficients(Y, s)

    z, mu, logvar = model(alpha, beta)
    alpha_pred = model.inverse(beta, z)

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

        avg_test_loss = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def evaluate_model(
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


def evaluate_instance(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        beta = output_function_encoder.compute_coefficients(Y, s)
        alpha_pred = model.inverse(beta)

        pred = input_function_encoder(X, alpha_pred)

        return pred, alpha_pred
