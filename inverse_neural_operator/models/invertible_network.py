import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

import tqdm


class AdditiveCoupling(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=[128, 128],
        split_dim=None,
        activation=torch.nn.ReLU(),
    ):
        super(AdditiveCoupling, self).__init__()

        self.input_size = input_size
        if split_dim is None:
            self.split_dim = input_size // 2
        else:
            self.split_dim = split_dim

        # Neural network to transform the first part
        self.net = torch.nn.Sequential()

        # Create layers with the specified hidden sizes
        layer_sizes = [input_size - self.split_dim] + hidden_sizes + [self.split_dim]
        for i in range(len(layer_sizes) - 1):
            self.net.add_module(
                f"linear_{i}", torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.net.add_module(f"activation_{i}", activation)

    def forward(self, x):
        """
        Forward transformation: x -> y
        Implements the additive coupling layer: y1 = x1, y2 = x2 + f(x1)
        """
        x1, x2 = torch.split(x, [self.split_dim, x.size(-1) - self.split_dim], dim=-1)
        y1 = x1
        y2 = x2 + self.net(x1)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y):
        """
        Inverse transformation: y -> x
        Implements the inverse of additive coupling: x1 = y1, x2 = y2 - f(y1)
        """
        y1, y2 = torch.split(y, [self.split_dim, y.size(-1) - self.split_dim], dim=-1)
        x1 = y1
        x2 = y2 - self.net(y1)
        return torch.cat([x1, x2], dim=-1)


class InvertibleNeuralNetwork(torch.nn.Module):
    def __init__(self, coupling_layers):
        super(InvertibleNeuralNetwork, self).__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)

    def forward(self, alpha):
        """
        Forward transformation, applies all coupling layers in sequence.
        """
        x = alpha
        for layer in self.coupling_layers:
            x = layer.forward(x)
        return x

    def inverse(self, beta):
        """
        Inverse transformation, applies all coupling layers in reverse order.
        """
        y = beta
        for layer in reversed(self.coupling_layers):
            y = layer.inverse(y)
        return y


class InvertibleNetworkFactory:
    @staticmethod
    def create(input_size, hidden_sizes=[128, 128], n_coupling_layers=2):
        coupling_layers = []

        for i in range(n_coupling_layers):
            # Alternate between splitting at different positions for better flow
            if i % 2 == 0:
                split_dim = input_size // 2
            else:
                split_dim = input_size - input_size // 2

            layer = AdditiveCoupling(
                input_size=input_size, hidden_sizes=hidden_sizes, split_dim=split_dim
            )
            coupling_layers.append(layer)

        return InvertibleNeuralNetwork(coupling_layers=coupling_layers)


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha = input_function_encoder.compute_coefficients(X, u)
    beta = output_function_encoder.compute_coefficients(Y, s)

    alpha_pred = model.inverse(beta)

    # reconstruction loss
    pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")

    return pred_loss


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


def _plot_case(
    model,
    point,
    file_name,
    input_function_encoder,
    output_function_encoder,
):
    """Helper function to plot a specific case evaluation."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    pred, alpha_pred = evaluate_instance(
        model,
        point,
        input_function_encoder=input_function_encoder,
        output_function_encoder=output_function_encoder,
    )
    pred = pred.squeeze(0).cpu().numpy()

    X, u, Y, s = point
    X = X.squeeze(0).cpu().numpy()
    u = u.squeeze(0).cpu().numpy()
    Y = Y.squeeze(0).cpu().numpy()
    s = s.squeeze(0).cpu().numpy()

    # Plot the input data
    ax[0].plot(X, u, label="Input Function", color="gray", alpha=0.5)
    ax[0].plot(X, pred, label="Prediction")

    # Plot the output data
    ax[1].plot(Y, s, label="Output Function", color="gray", alpha=0.5)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_best_case_evaluation(
    model,
    dataset,
    file_name="results/model_best_case_evaluation.png",
    input_function_encoder=None,
    output_function_encoder=None,
):
    """Find and plot the best case (lowest loss) from the dataset."""
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        best_case = None
        best_case_loss = float("inf")
        best_case_index = -1

        for i, point in enumerate(dataloader):
            loss = loss_function(
                model=model,
                batch=point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
            )
            if loss < best_case_loss:
                best_case_loss = loss
                best_case = point
                best_case_index = i

        _plot_case(
            model=model,
            point=best_case,
            file_name=file_name,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )


def plot_worst_case_evaluation(
    model,
    dataset,
    file_name="results/model_worst_case_evaluation.png",
    input_function_encoder=None,
    output_function_encoder=None,
):
    """Find and plot the worst case (highest loss) from the dataset."""
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        worst_case = None
        worst_case_loss = float("-inf")
        worst_case_index = -1

        for i, point in enumerate(dataloader):
            loss = loss_function(
                model=model,
                batch=point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
            )
            if loss > worst_case_loss:
                worst_case_loss = loss
                worst_case = point
                worst_case_index = i

        _plot_case(
            model=model,
            point=worst_case,
            file_name=file_name,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )


def plot_evaluation(
    model,
    dataset,
    file_name="results/b2b_operator_evaluation.png",
    input_function_encoder=None,
    output_function_encoder=None,
):
    model.eval()
    with torch.no_grad():

        index = np.random.choice(len(dataset), 1, replace=False)[0]
        subset = Subset(dataset, [index])

        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
        )

        for point in dataloader:
            _plot_case(
                model=model,
                point=point,
                file_name=file_name,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
            )
