import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

import tqdm


class NonlinearB2BOperator(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes: list[int] = [128, 128],
        output_size: int = 128,
    ):
        super(NonlinearB2BOperator, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.layers = torch.nn.ModuleList()

        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1]),
            )

        self.activation = torch.nn.ReLU()

    def forward(self, alpha):
        return None

    def inverse(self, beta):
        for layer in self.layers[:-1]:
            beta = self.activation(layer(beta))

        beta = self.layers[-1](beta)

        return beta


class NonlinearB2BOperatorFactory:
    @staticmethod
    def create(input_size, hidden_sizes, output_size):
        return NonlinearB2BOperator(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha = input_function_encoder.compute_coefficients(X, u)
    beta = output_function_encoder.compute_coefficients(Y, s)

    alpha_pred = model.inverse(beta)

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


def plot_worst_case_evaluation(
    model,
    dataset,
    file_name="results/b2b_operator_evaluation.png",
    input_function_encoder=None,
    output_function_encoder=None,
):
    model.eval()
    with torch.no_grad():

        # Find worst case
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        worst_case = None
        worst_case_loss = float("inf")
        worst_case_index = -1

        for i, point in enumerate(dataloader):
            loss = loss_function(
                model=model,
                batch=point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
            )
            if loss < worst_case_loss:
                worst_case_loss = loss
                worst_case = point
                worst_case_index = i

        # Plot worst case
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        pred, alpha_pred = evaluate_instance(
            model,
            worst_case,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        pred = pred.squeeze(0).cpu().numpy()

        X, u, Y, s = worst_case
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
