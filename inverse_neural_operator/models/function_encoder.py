import torch

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

        summary_writer.add_scalars(
            "loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = evaluate_model(
            model=model, test_dataloader=test_dataloader)
        summary_writer.add_scalars(
            "loss/test", {model_name: avg_test_loss}, epoch)

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
