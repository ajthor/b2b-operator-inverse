import torch


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        latent_size,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size

        self.activation = activation

        self.layers = torch.nn.ModuleList()

        sizes = [input_size] + hidden_sizes + [latent_size]
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias),
            )

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))

        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
        latent_size,
        hidden_sizes,
        output_size,
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

        self.output_activation = torch.nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        x = self.output_activation(x)

        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_size, hidden_sizes, latent_size)
        self.decoder = Decoder(latent_size, hidden_sizes[::-1], input_size)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
