import torch

# Invertible neural network


class CouplingLayer(torch.nn.Module):
    def __init__(
        self,
        input_size,
    ):
        super(CouplingLayer, self).__init__()

        self.input_size = input_size

        # self.scale = torch.nn.Linear(input_size // 2, input_size // 2)
        # self.translate = torch.nn.Linear(input_size // 2, input_size // 2)

        # self.scale = torch.nn.Sequential(
        #     torch.nn.Linear(input_size // 2, 128),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(128, input_size // 2),
        # )

        # self.translate = torch.nn.Sequential(
        #     torch.nn.Linear(input_size // 2, 128),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(128, 128),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(128, input_size // 2),
        # )

        self.scale_x = torch.nn.Sequential(
            torch.nn.Linear(input_size // 2, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, input_size // 2),
        )

        self.translate_x = torch.nn.Sequential(
            torch.nn.Linear(input_size // 2, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, input_size // 2),
        )

        self.scale_y = torch.nn.Sequential(
            torch.nn.Linear(input_size // 2, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, input_size // 2),
        )

        self.translate_y = torch.nn.Sequential(
            torch.nn.Linear(input_size // 2, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, input_size // 2),
        )

    # def forward(self, x):
    #     x1, x2 = x.chunk(2, dim=-1)

    #     y1 = x1
    #     y2 = x2 * torch.exp(torch.tanh(self.scale(x1))) + self.translate(x1)

    #     return torch.cat([y1, y2], dim=-1)

    # def inverse(self, y):
    #     y1, y2 = y.chunk(2, dim=-1)

    #     x1 = y1
    #     x2 = (y2 - self.translate(y1)) * torch.exp(-torch.tanh(self.scale(y1)))

    #     return torch.cat([x1, x2], dim=-1)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)

        s_x = torch.tanh(self.scale_x(x2))
        t_x = self.translate_x(x2)

        y1 = x1 * torch.exp(s_x) + t_x

        s_y = torch.tanh(self.scale_y(y1))
        t_y = self.translate_y(y1)

        y2 = x2 * torch.exp(s_y) + t_y

        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)

        s_y = torch.tanh(self.scale_y(y1))
        t_y = self.translate_y(y1)

        x2 = (y2 - t_y) * torch.exp(-s_y)

        s_x = torch.tanh(self.scale_x(x2))
        t_x = self.translate_x(x2)

        x1 = y1 * torch.exp(-s_x) - t_x

        return torch.cat([x1, x2], dim=-1)


class InvertibleNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size,
    ):
        super(InvertibleNetwork, self).__init__()

        self.input_size = input_size

        self.layers = torch.nn.ModuleList([CouplingLayer(input_size) for _ in range(3)])

        # self.coupling_layer = CouplingLayer(input_size)

    def forward(self, x):
        # y = self.coupling_layer(x)

        # return y

        for layer in self.layers:
            x = layer(x)

        return x

    def inverse(self, y):
        # x = self.coupling_layer.inverse(y)

        # return x

        for layer in reversed(self.layers):
            y = layer.inverse(y)

        return y
