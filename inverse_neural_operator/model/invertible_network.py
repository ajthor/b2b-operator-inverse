import torch

import tqdm


class Scale(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
    ):
        super(Scale, self).__init__()

        self.input_size = input_size

        self.scale = torch.nn.ModuleList()

        sizes = [input_size // 2 + condition_size] + hidden_sizes + [input_size // 2]

        for i in range(len(sizes) - 1):
            self.scale.append(
                torch.nn.Linear(sizes[i], sizes[i + 1]),
            )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)

        for layer in self.scale[:-1]:
            x = torch.nn.ReLU(layer(x))

        x = self.scale[-1](x)

        return torch.tanh(x)


class Translate(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
    ):
        super(Translate, self).__init__()

        self.input_size = input_size

        self.translate = torch.nn.ModuleList()

        sizes = [input_size // 2 + condition_size] + hidden_sizes + [input_size // 2]
        for i in range(len(sizes) - 1):
            self.translate.append(
                torch.nn.Linear(sizes[i], sizes[i + 1]),
            )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)

        for layer in self.translate[:-1]:
            x = torch.nn.ReLU(layer(x))

        x = self.translate[-1](x)

        return x


class AffineCoupling(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
    ):
        super(AffineCoupling, self).__init__()

        self.input_size = input_size

        self.scale = Scale(input_size, condition_size, hidden_sizes)
        self.translate = Translate(input_size, condition_size, hidden_sizes)

    def forward(self, x, condition):
        x1, x2 = x.chunk(2, dim=-1)

        s = self.scale(x1, condition)
        t = self.translate(x1, condition)

        x2 = x2 * torch.exp(s) + t

        return torch.cat([x1, x2], dim=-1), (s, t)

    def inverse(self, z, condition):
        z1, z2 = z.chunk(2, dim=-1)

        s = self.scale(z1, condition)
        t = self.translate(z1, condition)

        z2 = (z2 - t) * torch.exp(-s)

        return torch.cat([z1, z2], dim=-1)


class InvertibleNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        n_coupling_layers=3,
    ):
        super(InvertibleNetwork, self).__init__()

        self.input_size = input_size

        self.layers = torch.nn.ModuleList(
            [
                AffineCoupling(input_size, condition_size)
                for _ in range(n_coupling_layers)
            ]
        )

    def forward(self, x, condition):
        log_det = 0

        for layer in self.layers:
            x, (s, _) = layer(x, condition)
            log_det += torch.sum(torch.exp(s), dim=1)

        return x, log_det

    def inverse(self, z, condition):

        for layer in reversed(self.layers):
            z = layer.inverse(z, condition)

        return z


class InvertibleNetworkFactory:
    def __init__(self):
        pass


def loss_function(model, batch, input_function_encoder, output_function_encoder):

    alpha = input_function_encoder.compute_coefficients(X, f)
    beta = output_function_encoder.compute_coefficients(Y, Tf)

    z, log_det = model(alpha, beta)
    alpha_pred = model.inverse(z, beta)

    # reconstruction loss
    pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")

    # regularization loss
    regularization_loss = torch.mean(log_det)

    return pred_loss + regularization_loss


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
