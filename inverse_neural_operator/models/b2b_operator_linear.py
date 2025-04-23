import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

import tqdm


class LinearB2BOperator(torch.nn.Module):
    """
    Linear operator for the inverse problem of parameter estimation.
    """

    def __init__(self, input_size, output_size):
        super(LinearB2BOperator, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)
        self.linear.weight.requires_grad = False

    def forward(self, alpha):
        return self.linear(alpha)

    def inverse(self, beta):
        """Compute the inverse of the linear operator."""
        return torch.linalg.solve(
            self.linear.weight.unsqueeze(0).expand(beta.size(0), -1, -1), beta
        )


class LinearB2BOperatorFactory:
    @staticmethod
    def create(input_size, output_size):
        return LinearB2BOperator(
            input_size=input_size,
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

    n = params.input_fe_n_basis
    m = params.output_fe_n_basis

    SXX = torch.zeros((n, n), device=device)
    SXY = torch.zeros((n, m), device=device)

    with torch.no_grad():
        tqdm_bar = tqdm.tqdm(len(train_dataloader))
        for batch in train_dataloader:

            # Compute the alpha and beta coefficients
            X, u, Y, s = batch
            alpha = input_function_encoder.compute_coefficients(X, u)
            beta = output_function_encoder.compute_coefficients(Y, s)

            # Compute the normal equations in chunks
            SXX += torch.einsum("ij,ik->jk", alpha, alpha)
            SXY += torch.einsum("ij,ik->jk", alpha, beta)

            tqdm_bar.update(1)

        # Add small regularization term to SXX
        SXX += 1e-6 * torch.eye(n, device=device)

        # Compute the linear operator
        W = torch.linalg.solve(SXX, SXY)

        model.linear.weight.copy_(W.T)

        avg_test_loss = evaluate_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss})


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
    file_name="results/b2b_operator_worst_case_evaluation.png",
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
