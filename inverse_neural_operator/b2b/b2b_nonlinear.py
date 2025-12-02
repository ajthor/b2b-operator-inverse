import torch
from safetensors.torch import save_file, load_file
import tqdm
import os

from utils.distributed import is_main_process
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler


class NonlinearB2BOperatorFwd(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes: list[int] = [128, 128],
        output_size: int = 128,
    ):
        super(NonlinearB2BOperatorFwd, self).__init__()
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
        for layer in self.layers[:-1]:
            alpha = self.activation(layer(alpha))

        beta = self.layers[-1](alpha)

        return beta


def create_model(input_size, hidden_sizes, output_size):
    """
    Create a nonlinear B2B operator model.

    Args:
        input_size: Size of the input coefficients (alpha)
        hidden_sizes: List of hidden layer sizes
        output_size: Size of the output coefficients (beta)

    Returns:
        NonlinearB2BOperatorFwd instance
    """
    return NonlinearB2BOperatorFwd(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
    )


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
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
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()
    return model, optimizer, checkpoint["epoch"], checkpoint["loss"]


def loss_function(model, batch, input_function_encoder, output_function_encoder):
    X, u, Y, s = batch

    alpha, _ = input_function_encoder.compute_coefficients(X, u)
    beta, _ = output_function_encoder.compute_coefficients(Y, s)

    beta_pred = model.forward(alpha)

    # Use reconstruction loss for more direct supervision
    s_pred = output_function_encoder(Y, beta_pred)
    pred_loss = torch.nn.functional.mse_loss(s_pred, s, reduction="mean")

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
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
):
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")
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
    accumulation_steps = max(getattr(params, "grad_accumulation_steps", 1), 1)
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
                loss = loss_function(
                    model=model,
                    batch=batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                )
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            running_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        summary_writer.add_scalars(
            "loss/train", {model_name: running_loss / accumulation_steps}, current_step
        )

        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            use_amp=enable_amp,
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
    input_function_encoder,
    output_function_encoder,
    use_amp=False,
):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        with autocast(enabled=use_amp):
            loss = loss_function(
                model=model,
                batch=batch,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
            )

    return loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    """
    Evaluate forward model: given input u, predict output s
    """
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        # Compute alpha from input u
        alpha, _ = input_function_encoder.compute_coefficients(X, u)

        # Forward pass: alpha -> beta
        beta_pred = model.forward(alpha)

        # Reconstruct output s from predicted beta
        s_pred = output_function_encoder(Y, beta_pred)

        return s_pred
