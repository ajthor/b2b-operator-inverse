import math
import os
import warnings
from typing import Iterable

import torch
from torch.utils.data import DataLoader

import tqdm

LOG_2PI = math.log(2 * math.pi)


class ConditionalRealNVPCoupling(torch.nn.Module):
    """
    Conditional affine coupling layer used in RealNVP.

    The transformation keeps a subset of the input dimensions fixed (defined by the
    binary mask) and applies an affine transformation to the complementary subset.
    The scale and translation parameters are predicted by an MLP that conditions on
    the masked input and the external conditioning variables (beta).
    """

    def __init__(
        self,
        input_size: int,
        condition_size: int,
        hidden_sizes: Iterable[int] = (128, 128),
        mask: torch.Tensor | None = None,
        activation: torch.nn.Module = torch.nn.ReLU(),
        scale_clip: float = 5.0,
    ):
        super().__init__()

        if mask is None:
            raise ValueError("A binary mask must be provided for RealNVP coupling.")

        if mask.dim() != 1 or mask.shape[0] != input_size:
            raise ValueError(
                "Mask must be a 1D tensor with length equal to input_size."
            )

        self.input_size = input_size
        self.condition_size = condition_size
        self.hidden_sizes = list(hidden_sizes)
        self.scale_clip = float(scale_clip)

        # Store mask as non-trainable buffer to ensure it is moved with the module.
        mask = mask.to(dtype=torch.float32)
        self.register_buffer("mask", mask)
        self.register_buffer("inv_mask", 1.0 - mask)

        layer_sizes = (
            [input_size + condition_size] + list(hidden_sizes) + [2 * input_size]
        )
        mlp_layers: list[torch.nn.Module] = []
        for idx in range(len(layer_sizes) - 1):
            in_features = layer_sizes[idx]
            out_features = layer_sizes[idx + 1]
            mlp_layers.append(torch.nn.Linear(in_features, out_features))
            if idx < len(layer_sizes) - 2:
                mlp_layers.append(activation)
        self.net = torch.nn.Sequential(*mlp_layers)

    def _scale_transform(self, s: torch.Tensor) -> torch.Tensor:
        # Bound the scale to keep the flow numerically stable.
        if self.scale_clip > 0:
            return torch.tanh(s) * self.scale_clip
        return s

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Maps x -> z and accumulates Jacobian log-determinant.

        Args:
            x: Input tensor of shape (batch_size, input_size).
            condition: Conditioning tensor (beta) of shape (batch_size, condition_size).

        Returns:
            z: Transformed tensor.
            log_det: Log determinant of the Jacobian.
        """
        mask = self.mask
        inv_mask = self.inv_mask
        if mask.dtype != x.dtype:
            mask = mask.to(dtype=x.dtype)
            inv_mask = inv_mask.to(dtype=x.dtype)

        x_masked = x * mask
        net_in = torch.cat([x_masked, condition], dim=-1)
        s_t = self.net(net_in)
        s, t = torch.chunk(s_t, 2, dim=-1)

        s = self._scale_transform(s)

        exp_s = torch.exp(s * inv_mask)
        y = x_masked + inv_mask * (x * exp_s + t)

        log_det = (s * inv_mask).sum(dim=-1)
        return y, log_det

    def inverse(self, z: torch.Tensor, condition: torch.Tensor):
        """
        Maps z -> x using the inverse affine transformation.
        """
        mask = self.mask
        inv_mask = self.inv_mask
        if mask.dtype != z.dtype:
            mask = mask.to(dtype=z.dtype)
            inv_mask = inv_mask.to(dtype=z.dtype)

        z_masked = z * mask
        net_in = torch.cat([z_masked, condition], dim=-1)
        s_t = self.net(net_in)
        s, t = torch.chunk(s_t, 2, dim=-1)

        s = self._scale_transform(s)

        exp_neg_s = torch.exp(-s * inv_mask)
        x = z_masked + inv_mask * ((z - t) * exp_neg_s)
        return x


class ConditionalRealNVP(torch.nn.Module):
    def __init__(self, coupling_layers: Iterable[ConditionalRealNVPCoupling]):
        super().__init__()
        self.coupling_layers = torch.nn.ModuleList(coupling_layers)

        if len(self.coupling_layers) == 0:
            raise ValueError("At least one coupling layer is required.")

        self.latent_size = self.coupling_layers[0].input_size

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor):
        """
        Transform alpha to latent z conditioned on beta.

        Returns:
            z: Latent representation.
            log_det: Accumulated log determinant of the Jacobian.
        """
        z = alpha
        log_det_total = torch.zeros(
            alpha.shape[0], dtype=alpha.dtype, device=alpha.device
        )
        for layer in self.coupling_layers:
            z, log_det = layer(z, beta)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor, beta: torch.Tensor):
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, beta)
        return x

    def sample_prior(self, batch_size: int, device=None):
        return torch.randn(batch_size, self.latent_size, device=device)

    def sample_posterior(self, beta: torch.Tensor, n_samples: int):
        samples = []
        batch_size = beta.shape[0]
        for _ in range(n_samples):
            z = torch.randn(
                batch_size, self.latent_size, device=beta.device, dtype=beta.dtype
            )
            samples.append(self.inverse(z, beta))
        return torch.stack(samples, dim=0)


def create_model(
    alpha_size: int,
    beta_size: int,
    hidden_sizes: Iterable[int] = (128, 128),
    latent_size: int = 128,
    n_coupling_layers: int = 6,
    activation: torch.nn.Module = torch.nn.ReLU(),
    scale_clip: float = 5.0,
):
    """
    Factory to construct a conditional RealNVP model with alternating binary masks.
    """
    if latent_size != alpha_size:
        warnings.warn(
            "Ignoring latent_size argument: RealNVP latent dimension is fixed to alpha_size.",
            stacklevel=2,
        )

    if n_coupling_layers <= 0:
        raise ValueError("n_coupling_layers must be positive.")

    coupling_layers = []
    base_mask = (torch.arange(alpha_size) % 2).float()

    for layer_idx in range(n_coupling_layers):
        if layer_idx % 2 == 0:
            mask = base_mask.clone()
        else:
            mask = 1.0 - base_mask

        coupling_layers.append(
            ConditionalRealNVPCoupling(
                input_size=alpha_size,
                condition_size=beta_size,
                hidden_sizes=hidden_sizes,
                mask=mask,
                activation=activation,
                scale_clip=scale_clip,
            )
        )

    return ConditionalRealNVP(coupling_layers=coupling_layers)


def save(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(model, path: str, device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
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
    path: str,
    optimizer=None,
    device=None,
):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()
    return model, optimizer, checkpoint["epoch"], checkpoint["loss"]


def _gaussian_nll(z: torch.Tensor, log_det: torch.Tensor):
    dim = z.shape[-1]
    quadratic = 0.5 * z.pow(2).sum(dim=-1)
    normalization = 0.5 * dim * LOG_2PI
    return quadratic + normalization - log_det


def loss_function(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
):
    X, u, Y, s = batch

    alpha_gt, _ = input_function_encoder.compute_coefficients(X, u)
    beta_gt, _ = output_function_encoder.compute_coefficients(Y, s)

    z, log_det = model(alpha_gt, beta_gt)
    return _gaussian_nll(z, log_det).mean()

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

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")

    if resume_from_checkpoint and checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            model, optimizer, start_epoch, _ = load_checkpoint(
                model=model,
                path=checkpoint_path,
                optimizer=optimizer,
                device=device,
            )
            print(f"Resuming training from epoch {start_epoch}...")

    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
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

        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        with torch.no_grad():
            test_batch = next(iter(test_dataloader))
            resim_coeff_loss, resim_pred_loss = resimulation_loss(
                model=model,
                batch=test_batch,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                forward_model=forward_model,
                n_samples=1,
            )
        summary_writer.add_scalars(
            "loss/resimulation_coeff", {model_name: resim_coeff_loss}, epoch
        )
        summary_writer.add_scalars(
            "loss/resimulation_pred", {model_name: resim_pred_loss}, epoch
        )

        if (
            checkpoint_path is not None
            and checkpoint_interval > 0
            and (epoch + 1) % checkpoint_interval == 0
        ):
            save_checkpoint(model, optimizer, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        loss = loss_function(
            model=model,
            batch=batch,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )

    return loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point
        beta, _ = output_function_encoder.compute_coefficients(Y, s)
        z = model.sample_prior(beta.shape[0], device=beta.device)
        alpha_pred = model.inverse(z, beta)
        pred = input_function_encoder(X, alpha_pred)
        return pred, alpha_pred


def resimulation_loss(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples=1,
):
    """
    Compute deterministic re-simulation metrics using a zero latent code.

    Re-simulation flow: beta -> alpha_pred -> beta_resim.
    """
    X, u, Y, s = batch

    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        batch_size = beta_target.shape[0]
        z_zero = torch.zeros(
            batch_size,
            model.latent_size,
            device=beta_target.device,
            dtype=beta_target.dtype,
        )
        alpha_pred = model.inverse(z_zero, beta_target)
        beta_resim = forward_model(alpha_pred)

        resim_coeff_loss = torch.nn.functional.mse_loss(beta_resim, beta_target)

        s_pred = output_function_encoder(Y, beta_resim)
        resim_pred_loss = torch.nn.functional.mse_loss(s_pred, s)

    model.train()
    return resim_coeff_loss.item(), resim_pred_loss.item()
