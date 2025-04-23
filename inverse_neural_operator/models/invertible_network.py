import torch

import tqdm


class ScaleNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
        activation=torch.nn.ReLU(),
    ):
        super(ScaleNetwork, self).__init__()

        self.input_size = input_size

        self.scale = torch.nn.ModuleList()

        sizes = [input_size // 2 + condition_size] + \
            hidden_sizes + [input_size // 2]

        for i in range(len(sizes) - 1):
            self.scale.append(
                torch.nn.Linear(sizes[i], sizes[i + 1]),
            )

        self.activation = activation

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)

        for layer in self.scale[:-1]:
            x = self.activation(layer(x))

        x = self.scale[-1](x)

        return torch.tanh(x)


class TranslateNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
        condition_size,
        hidden_sizes=[128, 128],
        activation=torch.nn.ReLU(),
    ):
        super(TranslateNetwork, self).__init__()

        self.input_size = input_size

        self.translate = torch.nn.ModuleList()

        sizes = [input_size // 2 + condition_size] + \
            hidden_sizes + [input_size // 2]
        for i in range(len(sizes) - 1):
            self.translate.append(
                torch.nn.Linear(sizes[i], sizes[i + 1]),
            )

        self.activation = activation

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)

        for layer in self.translate[:-1]:
            x = self.activation(layer(x))

        x = self.translate[-1](x)

        return x


class AffineCoupling(torch.nn.Module):
    def __init__(
        self,
        scale_network,
        translate_network,
    ):
        super(AffineCoupling, self).__init__()

        self.scale_network = scale_network
        self.translate_network = translate_network

    def forward(self, x, condition):
        x1, x2 = x.chunk(2, dim=-1)

        s = self.scale_network(x1, condition)
        t = self.translate_network(x1, condition)

        x2 = x2 * torch.exp(s) + t

        return torch.cat([x1, x2], dim=-1), (s, t)

    def inverse(self, z, condition):
        z1, z2 = z.chunk(2, dim=-1)

        s = self.scale_network(z1, condition)
        t = self.translate_network(z1, condition)

        z2 = (z2 - t) * torch.exp(-s)

        return torch.cat([z1, z2], dim=-1)


class ConditionalInvertibleNetwork(torch.nn.Module):
    def __init__(
        self,
        coupling_layers,
    ):
        super(ConditionalInvertibleNetwork, self).__init__()

        self.layers = coupling_layers

    def sample_prior(self, batch_size, device=None):
        z = torch.randn(
            batch_size, self.layers[0].scale_network.input_size, device=device
        )
        return z

    def forward(self, alpha, beta):
        log_det = 0

        for layer in self.layers:
            alpha, (s, _) = layer(alpha, beta)
            log_det += torch.sum(torch.exp(s), dim=1)

        return alpha, log_det

    def inverse(self, beta, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z, beta)

        return z


class ConditionalInvertibleNetworkFactory:
    @staticmethod
    def create(
        input_size,
        condition_size,
        hidden_sizes,
        n_coupling_layers,
        activation=torch.nn.ReLU(),
    ):
        coupling_layers = torch.nn.ModuleList(
            [
                AffineCoupling(
                    ScaleNetwork(
                        input_size=input_size,
                        condition_size=condition_size,
                        hidden_sizes=hidden_sizes,
                        activation=activation,
                    ),
                    TranslateNetwork(
                        input_size=input_size,
                        condition_size=condition_size,
                        hidden_sizes=hidden_sizes,
                        activation=activation,
                    ),
                )
                for _ in range(n_coupling_layers)
            ]
        )

        return ConditionalInvertibleNetwork(coupling_layers=coupling_layers)


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha = input_function_encoder.compute_coefficients(X, u)
    beta = output_function_encoder.compute_coefficients(Y, s)

    z, log_det = model(alpha, beta)
    alpha_pred = model.inverse(beta, z)

    # reconstruction loss
    pred_loss = torch.nn.functional.mse_loss(
        alpha_pred, alpha, reduction="mean")

    # regularization loss
    regularization_loss = torch.mean(log_det)

    return pred_loss + regularization_loss


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

        summary_writer.add_scalars(
            "loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        summary_writer.add_scalars(
            "loss/test", {model_name: avg_test_loss}, epoch)

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
