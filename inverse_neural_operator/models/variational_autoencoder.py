import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import tqdm
import os
from safetensors.torch import save_file, load_file

from utils.distributed import is_main_process
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [128, 128],
        latent_size: int = 128,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size

        self.activation = activation

        self.layers = torch.nn.ModuleList()

        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias),
            )

        self.mu = torch.nn.Linear(hidden_sizes[-1], latent_size, bias=bias)
        self.logvar = torch.nn.Linear(hidden_sizes[-1], latent_size, bias=bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar


class Decoder(torch.nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_sizes: list[int] = [128, 128],
        latent_size: int = 128,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.activation = activation

        self.layers = torch.nn.ModuleList()

        sizes = [latent_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias),
            )

        self.output_activation = torch.nn.Identity()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        x = self.output_activation(x)

        return x


class ConditionalVariationalAutoencoder(torch.nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        latent_size: int = 128,
    ):
        super(ConditionalVariationalAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.latent_size = latent_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)

        return mu + eps * std

    def sample_prior(self, batch_size, device=None):
        z = torch.randn(batch_size, self.latent_size, device=device)
        return z

    def forward(self, alpha, beta):
        mu, logvar = self.encoder(torch.cat([alpha, beta], dim=-1))
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def inverse(self, beta, z):
        return self.decoder(torch.cat([z, beta], dim=-1))


def create_model(
    alpha_size: int,
    beta_size: int,
    hidden_sizes: list[int] = [128, 128],
    latent_size: int = 128,
):
    encoder = Encoder(
        input_size=alpha_size + beta_size,
        hidden_sizes=hidden_sizes,
        latent_size=latent_size,
    )

    decoder = Decoder(
        output_size=alpha_size,
        hidden_sizes=hidden_sizes[::-1],
        latent_size=latent_size + beta_size,
    )

    return ConditionalVariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_size=latent_size,
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


def loss_function(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model=None,
    lambda_forward=0.0,
    lambda_u: float = 0.0,
):
    X, u, Y, s = batch

    alpha, _ = input_function_encoder.compute_coefficients(X, u)
    beta, _ = output_function_encoder.compute_coefficients(Y, s)

    z, mu, logvar = model(alpha, beta)
    alpha_pred = model.inverse(beta, z)

    pred_loss = torch.nn.functional.mse_loss(alpha_pred, alpha, reduction="mean")

    # # Reconstruction loss: negative log probability assuming unit variance Gaussian
    # reconstruction_loss = 0.5 * torch.sum((alpha_pred - alpha) ** 2, dim=-1).mean()

    # # Forward consistency loss: encode alpha_pred with beta and compare z values
    # z_reconstructed, *_ = model(alpha_pred, beta)
    # consistency_loss = torch.nn.functional.mse_loss(
    #     z_reconstructed, z, reduction="mean"
    # )

    # Function space loss (u-loss)
    # u_pred = input_function_encoder(X, alpha_pred)
    # pred_loss = torch.nn.functional.mse_loss(u_pred, u, reduction="mean")

    # # KL divergence loss
    # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) -
    #                            logvar.exp(), dim=-1).mean()

    #    # --- KL divergence KL(q(z|x,y) || p(z|y)) for diagonal Gaussians ---
    mu_p = torch.zeros_like(mu)
    logvar_p = torch.zeros_like(logvar)  # log(1)

    var_q = logvar.exp()
    var_p = logvar_p.exp()
    kl_per = 0.5 * (logvar_p - logvar + (var_q + (mu - mu_p) ** 2) / var_p - 1.0).sum(
        dim=-1
    )
    kl_loss = kl_per.mean()

    total_loss = pred_loss + kl_loss

    # Add forward model consistency loss if available
    if forward_model is not None:
        with torch.no_grad():
            forward_model.eval()
        # Forward consistency: alpha_pred -> beta_pred should match beta
        beta_pred = forward_model.forward(alpha_pred)
        forward_loss = torch.nn.functional.mse_loss(beta_pred, beta, reduction="mean")
        # s_pred = output_function_encoder(Y, beta_pred)
        # forward_loss = torch.nn.functional.mse_loss(s_pred, s, reduction="mean")
        total_loss = total_loss + forward_loss

    return total_loss


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
    forward_model,
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
                    lambda_u=params.lambda_u if hasattr(params, "lambda_u") else 0.0,
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
            lambda_u=params.lambda_u if hasattr(params, "lambda_u") else 0.0,
            use_amp=enable_amp,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, current_step)

        with torch.no_grad():
            test_batch = next(iter(test_dataloader))
            with autocast(enabled=enable_amp):
                resim_coeff_loss, resim_pred_loss = resimulation_loss(
                    model=model,
                    batch=test_batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    forward_model=forward_model,
                    n_samples=1,
                )
        summary_writer.add_scalars(
            "loss/resimulation_coeff", {model_name: resim_coeff_loss}, current_step
        )
        summary_writer.add_scalars(
            "loss/resimulation_pred", {model_name: resim_pred_loss}, current_step
        )

        if (current_step + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, current_step + 1, avg_test_loss, checkpoint_path)

        current_step += 1
        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
    lambda_u: float = 0.0,
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
                lambda_u=lambda_u,
            )

    return loss.item()


def resimulation_loss(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples=1,
):
    """
    Compute re-simulation loss for VAE model.

    For VAE: Use deterministic inverse mapping (z=0) beta* -> alpha,
    apply forward operator alpha -> beta_resim, measure MSE(beta_resim, beta*).
    Since we use z=0 for deterministic evaluation, n_samples parameter is ignored.

    Re-simulation flow: beta_measured -> alpha_pred -> beta_resim -> loss(beta_resim, beta_measured)

    Returns:
        resim_coeff_loss: MSE between re-simulated and target coefficients
        resim_pred_loss: MSE between predictions from re-simulated coefficients and ground truth
    """
    X, u, Y, s = batch

    # Get target beta coefficients from observed output
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        # Deterministic inverse: beta -> alpha (using z=0 for deterministic behavior)
        batch_size = beta_target.shape[0]
        z_zero = (
            model.sample_prior(batch_size, device=beta_target.device) * 0
        )  # Zero out the prior sample

        alpha_pred = model.inverse(beta_target, z_zero)  # [batch_size, alpha_dim]

        # Forward re-simulation: alpha -> beta
        beta_resim = forward_model(alpha_pred)  # [batch_size, beta_dim]

        # Coefficient error: re-simulated beta vs target beta
        resim_coeff_loss = torch.nn.functional.mse_loss(beta_resim, beta_target)

        # Prediction error: function predictions using re-simulated beta
        s_pred = output_function_encoder(Y, beta_resim)
        resim_pred_loss = torch.nn.functional.mse_loss(s_pred, s)

    model.train()
    return resim_coeff_loss.item(), resim_pred_loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        beta, _ = output_function_encoder.compute_coefficients(Y, s)

        z = model.sample_prior(1, device=X.device)
        alpha_pred = model.inverse(beta, z)
        pred = input_function_encoder(X, alpha_pred)

        return pred, alpha_pred
