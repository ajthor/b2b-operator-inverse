import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file, load_file

from function_encoder.model.mlp import MultiHeadedMLP, MLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss, residual_loss
import functools

from utils.distributed import is_main_process

import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


def memory_efficient_inner_product(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    # f: (b, m, d, k), g: (b, m, d, l)
    # Reshape to (b, m*d, k) and (b, m*d, l)
    b, m, d, k = f.shape
    l = g.shape[-1]
    f_flat = f.reshape(b, m * d, k)
    g_flat = g.reshape(b, m * d, l)
    # Compute (b, k, l): (b, k, m*d) @ (b, m*d, l)
    result = torch.matmul(f_flat.transpose(1, 2), g_flat) / m

    return result


def unwrap_model(model):
    """Return the underlying module when wrapped by DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def cholesky_least_squares(
    f: torch.Tensor,
    g: torch.Tensor,
    inner_product,
    regularization: float = 1e-3,
):
    """
    Solve (G + lambda I) alpha = F using batched Cholesky for speed/stability.

    Args:
        f: tensor of shape (batch, n_points, features)
        g: tensor of shape (batch, n_points, features, n_basis)
    """
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)

    eye = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)
    G_reg = G + regularization * eye

    chol, info = torch.linalg.cholesky_ex(G_reg)
    if torch.any(info > 0):
        # Fall back to general solve if decomposition failed for any batch
        coefficients = torch.linalg.solve(G_reg, F)
    else:
        coefficients = torch.cholesky_solve(F.unsqueeze(-1), chol).squeeze(-1)

    return coefficients, G


def create_model(
    input_size,
    hidden_sizes,
    output_size,
    n_basis,
    activation=torch.nn.ReLU(),
    inner_product=None,
    regularization=1e-3,
):
    """
    Create a function encoder model.

    Args:
        input_size: Size of the input features
        hidden_sizes: List of hidden layer sizes for the MLP
        output_size: Size of the output features
        n_basis: Number of basis functions
        activation: Activation function to use in the MLP
        inner_product: Custom inner product function (optional)
        regularization: Regularization parameter for least squares (default: 1e-4)

    Returns:
        FunctionEncoder instance
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    basis_functions = MultiHeadedMLP(
        layer_sizes=layer_sizes,
        num_heads=n_basis,
        activation=activation,
    )

    # Create a custom coefficients method with the desired regularization
    coefficients_method = functools.partial(
        cholesky_least_squares, regularization=regularization
    )

    kwargs = {}
    if inner_product is not None:
        kwargs["inner_product"] = inner_product
    kwargs["coefficients_method"] = coefficients_method
    return FunctionEncoder(basis_functions=basis_functions, **kwargs)


def save(model, path):
    """Save model weights in safetensors format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(model.state_dict(), path)


def load(model, path, device=None):
    """Load model weights from safetensors format."""
    state_dict = load_file(path, device=str(device) if device else 'cpu')
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    if not is_main_process():
        return
    model_to_save = unwrap_model(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    path,
    optimizer=None,
    device=None,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_to_load = unwrap_model(model)
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()
    return model, optimizer, checkpoint["epoch"], checkpoint["loss"]


def loss_function(model, batch):
    model = unwrap_model(model)
    example_xs, example_ys, xs, ys = batch
    # Inner least-squares solve is numerically sensitive; run it in full precision
    with autocast(enabled=False):
        coefficients, G = model.compute_coefficients(
            example_xs.float(), example_ys.float()
        )
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)

    norm_loss = basis_normalization_loss(G)
    # G = model.basis_functions(xs)
    # K = model.inner_product(G, G)
    # K = K + torch.eye(K.shape[1], device=K.device)
    # norm_loss = ((torch.diagonal(K, dim1=-2, dim2=-1) - 1) ** 2).mean()

    return pred_loss + norm_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    n_epochs,
    summary_writer,
    model_name,
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
    grad_accumulation_steps=1,
):
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir or ".", f"{model_name}_checkpoint.pt")
    if resume_from_checkpoint:
        if os.path.exists(checkpoint_path):
            model, optimizer, start_epoch, loss = load_checkpoint(
                model=model,
                path=checkpoint_path,
                optimizer=optimizer,
                device=device,
            )
            print(f"Resuming training from epoch {start_epoch}...")

    enable_amp = device is not None and str(device).startswith("cuda")
    scaler = GradScaler(enabled=enable_amp)
    accumulation_steps = max(grad_accumulation_steps, 1)
    total_steps = n_epochs
    current_step = start_epoch

    train_sampler = getattr(train_dataloader, "sampler", None)
    next_sampler_epoch = start_epoch

    def make_iterator(epoch_seed):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch_seed)
        return iter(train_dataloader)

    train_iter = make_iterator(next_sampler_epoch)

    tqdm_bar = tqdm.tqdm(range(start_epoch, total_steps))
    while current_step < total_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                next_sampler_epoch += 1
                train_iter = make_iterator(next_sampler_epoch)
                batch = next(train_iter)

            with autocast(enabled=enable_amp):
                loss = loss_function(model=model, batch=batch)
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        summary_writer.add_scalars(
            "loss/train", {model_name: running_loss / accumulation_steps}, current_step
        )

        avg_test_loss = test_model(
            model=model, test_dataloader=test_dataloader, epoch=current_step
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, current_step)

        if checkpoint_interval > 0 and (current_step + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, current_step + 1, avg_test_loss, checkpoint_path
            )

        current_step += 1
        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    epoch=None,
):
    model = unwrap_model(model)
    model.eval()
    with torch.no_grad():
        test_sampler = getattr(test_dataloader, "sampler", None)
        if epoch is not None and isinstance(test_sampler, DistributedSampler):
            test_sampler.set_epoch(epoch)
        batch = next(iter(test_dataloader))
        loss = loss_function(model=model, batch=batch)

    return loss.item()


def evaluate(model, point):
    model.eval()
    with torch.no_grad():
        example_xs, example_ys, xs, ys = point

        coefficients, _ = model.compute_coefficients(example_xs, example_ys)
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
        subset = Subset(dataset, indices.tolist())

        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
        )

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))

        for i, point in enumerate(dataloader):
            pred = evaluate(model, point)
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
