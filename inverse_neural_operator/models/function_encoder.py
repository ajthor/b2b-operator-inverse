import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss

import tqdm
from torch.utils.tensorboard import SummaryWriter


class FunctionEncoderFactory:
    @staticmethod
    def create(
        input_size,
        hidden_sizes,
        output_size,
        n_basis,
        activation=torch.nn.ReLU(),
    ):
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        basis_functions = MultiHeadedMLP(
            layer_sizes=layer_sizes,
            num_heads=n_basis,
            activation=activation,
        )

        return FunctionEncoder(basis_functions=basis_functions)


def loss_function(model, batch):
    example_xs, example_ys, xs, ys = batch

    coefficients = model.compute_coefficients(example_xs, example_ys)
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)
    norm_loss = basis_normalization_loss(model.basis_functions(xs))

    return pred_loss + norm_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
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
        loss = loss_function(model=model, batch=batch)
        loss.backward()
        optimizer.step()

        summary_writer.add_scalars("loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = evaluate_model(model=model, test_dataloader=test_dataloader)
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def evaluate_model(
    model,
    test_dataloader,
):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            loss = loss_function(model=model, batch=batch)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate_instance(model, point):
    model.eval()
    with torch.no_grad():
        example_xs, example_ys, xs, ys = point

        coefficients = model.compute_coefficients(example_xs, example_ys)
        pred = model(xs, coefficients)

        return pred


def plot_evaluations(
    model,
    dataset,
    file_name="results/function_encoder_evaluation.png",
):
    model.eval()
    with torch.no_grad():

        indices = np.random.choice(len(dataset), 9, replace=False)
        subset = Subset(dataset, indices)

        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
        )

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))

        for i, point in enumerate(dataloader):
            pred = evaluate_instance(model, point)
            pred = pred.squeeze(0).cpu().numpy()

            example_xs, example_ys, xs, ys = point
            example_xs = example_xs.squeeze(0).cpu().numpy()
            example_ys = example_ys.squeeze(0).cpu().numpy()
            xs = xs.squeeze(0).cpu().numpy()
            ys = ys.squeeze(0).cpu().numpy()

            # Sort by the xs
            sort_indices = np.argsort(xs, axis=0)
            xs = xs[sort_indices.squeeze()]
            ys = ys[sort_indices.squeeze()]
            pred = pred[sort_indices.squeeze()]

            # Plot the input data
            axs[i // 3, i % 3].scatter(
                example_xs, example_ys, label="Example Data", color="gray", alpha=0.5
            )
            axs[i // 3, i % 3].plot(xs, ys, label="True")
            axs[i // 3, i % 3].plot(xs, pred, label="Prediction")

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
