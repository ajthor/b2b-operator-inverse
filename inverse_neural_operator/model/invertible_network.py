import torch

# Invertible neural network


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
    ):
        super(AffineCoupling, self).__init__()

        self.input_size = input_size

        self.scale = Scale(input_size)
        self.translate = Translate(input_size)

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
