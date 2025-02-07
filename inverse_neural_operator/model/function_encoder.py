import torch

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss

import tqdm


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
    example_xs = batch["example_xs"]
    example_ys = batch["example_ys"]
    xs = batch["xs"]
    ys = batch["ys"]

    coefficients = model.compute_coefficients(example_xs, example_ys)
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)
    norm_loss = basis_normalization_loss(model.basis_functions(xs))

    return pred_loss + norm_loss


def train(
    model,
    dataloader,
    optimizer,
    n_epochs=100,
):
    model.train()

    with tqdm.tqdm(range(n_epochs)) as tqdm_bar:
        for epoch in tqdm_bar:
            for batch in dataloader:
                optimizer.zero_grad()

                loss = loss_function(
                    model=model,
                    batch=batch,
                )
                loss.backward()

                optimizer.step()

                break

            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"loss {loss.item():.4e}")
