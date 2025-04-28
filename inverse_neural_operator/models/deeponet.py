import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

import tqdm


class DeepONet(torch.nn.Module):
    """
    Deep Operator Network (DeepONet) for mapping between function spaces.

    This implementation takes output function values (s) as input to the branch network
    and evaluation points (X) as input to the trunk network to predict the input
    function values (u).
    """

    def __init__(
        self,
        branch_net,
        trunk_net,
        output_channels=1,
    ):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.output_channels = output_channels

        # Initialize bias term
        self.bias = torch.nn.Parameter(torch.zeros(output_channels))

    def forward(self, s, X):
        """
        Not used for the inverse problem.
        """
        return None

    def inverse(self, s, X):
        """
        Maps from output function values (s) and evaluation points (X) to
        input function values (u).

        Args:
            s: Output function values
            X: Spatial coordinates for evaluation

        Returns:
            Predicted input function values (u) at points X
        """
        # Process through branch network (processes output function s)
        branch_output = self.branch_net(s)

        # Process through trunk network (processes evaluation points X)
        trunk_output = self.trunk_net(X)

        # Reshape for dot product
        batch_size = s.shape[0]
        branch_output = branch_output.view(batch_size, self.output_channels, -1)
        trunk_output = trunk_output.view(batch_size, -1, 1)

        # Compute the output with bias
        output = torch.bmm(branch_output, trunk_output).squeeze(-1) + self.bias

        return output


def create_mlp(input_size, hidden_sizes, output_size, activation=torch.nn.ReLU()):
    """Helper function to create a simple MLP."""
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    for i in range(len(layer_sizes) - 1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:  # No activation after the last layer
            layers.append(activation)

    return torch.nn.Sequential(*layers)


class DeepONetFactory:
    @staticmethod
    def create(
        branch_input_size,  # Size of input for the branch network (s values)
        trunk_input_size,  # Dimension of spatial coordinates for trunk network (X)
        output_size,  # Size of the output (u values)
        hidden_sizes=[128, 128, 128],
    ):
        # Width of the last hidden layer, used for branch-trunk dot product
        dot_product_dim = hidden_sizes[-1]

        # Create branch network to process output function values (s)
        branch_net = create_mlp(
            input_size=branch_input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size * dot_product_dim,
            activation=torch.nn.ReLU(),
        )

        # Create trunk network to process spatial coordinates (X)
        trunk_net = create_mlp(
            input_size=trunk_input_size,
            hidden_sizes=hidden_sizes,
            output_size=dot_product_dim,
            activation=torch.nn.ReLU(),
        )

        # Create DeepONet
        return DeepONet(
            branch_net=branch_net,
            trunk_net=trunk_net,
            output_channels=output_size,
        )


def loss_function(model, batch):
    """Loss function that works directly with the raw data without function encoders."""
    X, u, Y, s = batch

    # Predict input function values from output function values and evaluation points
    u_pred = model.inverse(s, X)

    # Compute the MSE loss
    pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    return pred_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
    n_epochs=1000,
    summary_writer=None,
    model_name="deeponet",
    params=None,
    device="cpu",
):
    tqdm_bar = tqdm.tqdm(range(n_epochs))
    for epoch in range(n_epochs):
        model.train()
        batch = next(iter(train_dataloader))

        optimizer.zero_grad()
        loss = loss_function(
            model=model,
            batch=batch,
        )
        loss.backward()
        optimizer.step()

        if summary_writer:
            summary_writer.add_scalars("loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
        )

        if summary_writer:
            summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def evaluate_model(
    model,
    test_dataloader,
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            loss = loss_function(
                model=model,
                batch=batch,
            )
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate_instance(model, point):
    """Evaluate the model on a single data point."""
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        u_pred = model.inverse(s, X)

        return u_pred


def plot_evaluation(
    model,
    dataset,
    file_name="results/deeponet_evaluation.png",
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
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
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            u_pred = evaluate_instance(model, point)
            u_pred = u_pred.squeeze(0).cpu().numpy()

            X, u, Y, s = point
            X = X.squeeze(0).cpu().numpy()
            u = u.squeeze(0).cpu().numpy()
            Y = Y.squeeze(0).cpu().numpy()
            s = s.squeeze(0).cpu().numpy()

            # Plot the input and predicted data
            ax[0].plot(X, u, label="True Input Function", color="gray", alpha=0.5)
            ax[0].plot(X, u_pred, label="Predicted Input Function")
            ax[0].legend()

            # Plot the output data
            ax[1].plot(Y, s, label="Output Function", color="gray", alpha=0.5)
            ax[1].legend()

            plt.tight_layout()
            plt.savefig(file_name)
            plt.close()


def _plot_case(
    model,
    point,
    file_name,
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
):
    """Helper function to plot a specific case evaluation."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    u_pred = evaluate_instance(model, point)
    u_pred = u_pred.squeeze(0).cpu().numpy()

    X, u, Y, s = point
    X = X.squeeze(0).cpu().numpy()
    u = u.squeeze(0).cpu().numpy()
    Y = Y.squeeze(0).cpu().numpy()
    s = s.squeeze(0).cpu().numpy()

    # Plot the input and predicted data
    ax[0].plot(X, u, label="True Input Function", color="gray", alpha=0.5)
    ax[0].plot(X, u_pred, label="Predicted Input Function")
    ax[0].legend()

    # Plot the output data
    ax[1].plot(Y, s, label="Output Function", color="gray", alpha=0.5)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_best_case_evaluation(
    model,
    dataset,
    file_name="results/model_best_case_evaluation.png",
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
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
            )
            if loss < best_case_loss:
                best_case_loss = loss
                best_case = point
                best_case_index = i

        _plot_case(
            model=model,
            point=best_case,
            file_name=file_name,
        )


def plot_worst_case_evaluation(
    model,
    dataset,
    file_name="results/model_worst_case_evaluation.png",
    input_function_encoder=None,  # Not used but kept for API compatibility
    output_function_encoder=None,  # Not used but kept for API compatibility
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
            )
            if loss > worst_case_loss:
                worst_case_loss = loss
                worst_case = point
                worst_case_index = i

        _plot_case(
            model=model,
            point=worst_case,
            file_name=file_name,
        )
