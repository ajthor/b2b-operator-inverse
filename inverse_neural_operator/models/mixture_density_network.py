import math
import os
from typing import Iterable

import torch
import torch.nn.functional as F

import tqdm
from safetensors.torch import save_file, load_file

LOG_2PI = math.log(2 * math.pi)
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 5.0


class MixtureDensityNetwork(torch.nn.Module):
    """Canonical Mixture Density Network (MDN).

    The network predicts a diagonal Gaussian mixture p(alpha | beta) with K components.
    Given conditioning coefficients ``beta`` it outputs mixture weights, component means,
    and log standard deviations in line with Bishop (1994).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Iterable[int] = (128, 128),
        n_components: int = 5,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
    ) -> None:
        super().__init__()

        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive.")
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        hidden_sizes = list(hidden_sizes)
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer.")

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.n_components = int(n_components)
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        layer_sizes = [self.input_size] + hidden_sizes
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(torch.nn.Linear(in_features, out_features, bias=bias))

        final_hidden = layer_sizes[-1]
        self.pi_head = torch.nn.Linear(final_hidden, self.n_components, bias=bias)
        self.mu_head = torch.nn.Linear(final_hidden, self.n_components * self.output_size, bias=bias)
        self.log_sigma_head = torch.nn.Linear(final_hidden, self.n_components * self.output_size, bias=bias)

    def _shared_forward(self, beta: torch.Tensor) -> torch.Tensor:
        x = beta
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    def forward(self, beta: torch.Tensor):
        """Return mixture parameters conditioned on beta.

        Args:
            beta: (batch, input_size)

        Returns:
            pi: (batch, K) mixture probabilities
            log_pi: (batch, K) log mixture probabilities
            mu: (batch, K, D) component means
            log_sigma: (batch, K, D) component log standard deviations
        """
        if beta.dim() != 2 or beta.shape[1] != self.input_size:
            raise ValueError("beta must have shape (batch, input_size).")

        hidden = self._shared_forward(beta)
        pi_logits = self.pi_head(hidden)
        log_pi = torch.log_softmax(pi_logits, dim=-1)
        pi = torch.exp(log_pi)

        batch = beta.shape[0]
        mu = self.mu_head(hidden).view(batch, self.n_components, self.output_size)
        log_sigma = self.log_sigma_head(hidden).view(batch, self.n_components, self.output_size)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)

        return pi, log_pi, mu, log_sigma

    def log_prob(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Log probability log p(alpha | beta)."""
        if alpha.dim() != 2 or alpha.shape[1] != self.output_size:
            raise ValueError("alpha must have shape (batch, output_size).")
        if alpha.shape[0] != beta.shape[0]:
            raise ValueError("alpha and beta must share the same batch dimension.")

        pi, log_pi, mu, log_sigma = self.forward(beta)
        diff = alpha.unsqueeze(1) - mu  # (batch, K, D)
        precision = torch.exp(-2.0 * log_sigma)
        quadratic = (diff.pow(2) * precision).sum(dim=-1)
        log_det = 2.0 * log_sigma.sum(dim=-1)
        component_log_prob = -0.5 * (quadratic + log_det + self.output_size * LOG_2PI)
        return torch.logsumexp(log_pi + component_log_prob, dim=-1)

    def inverse(self, beta: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample alpha ~ p(alpha | beta)."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        with torch.no_grad():
            pi, _, mu, log_sigma = self.forward(beta)
            sigma = torch.exp(log_sigma)
            batch = beta.shape[0]
            categorical = torch.distributions.Categorical(probs=pi)

            if n_samples == 1:
                component_idx = categorical.sample()
                batch_idx = torch.arange(batch, device=beta.device)
                mu_sel = mu[batch_idx, component_idx]
                sigma_sel = sigma[batch_idx, component_idx]
                eps = torch.randn_like(mu_sel)
                return mu_sel + sigma_sel * eps

            component_idx = categorical.sample((n_samples,))  # (n_samples, batch)
            mu_expanded = mu.unsqueeze(0).expand(n_samples, -1, -1, -1)
            sigma_expanded = sigma.unsqueeze(0).expand(n_samples, -1, -1, -1)
            gather_idx = component_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.output_size)
            mu_sel = mu_expanded.gather(2, gather_idx).squeeze(2)
            sigma_sel = sigma_expanded.gather(2, gather_idx).squeeze(2)
            eps = torch.randn_like(mu_sel)
            return mu_sel + sigma_sel * eps

    def rsample(self, beta: torch.Tensor, num_samples: int = 1, gumbel_temp: float = 0.5) -> torch.Tensor:
        """Differentiable reparameterized sampling via Gumbel-Softmax."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        pi, log_pi, mu, log_sigma = self.forward(beta)
        sigma = torch.exp(log_sigma)

        pi = pi.unsqueeze(0).expand(num_samples, -1, -1)
        log_pi = log_pi.unsqueeze(0).expand(num_samples, -1, -1)
        mu = mu.unsqueeze(0).expand(num_samples, -1, -1, -1)
        sigma = sigma.unsqueeze(0).expand(num_samples, -1, -1, -1)

        gumbel = -torch.log(-torch.log(torch.rand_like(pi).clamp_min(1e-8)))
        weights_soft = F.softmax((log_pi + gumbel) / gumbel_temp, dim=-1)
        weights_hard = torch.zeros_like(weights_soft)
        weights_hard.scatter_(-1, weights_soft.argmax(-1, keepdim=True), 1.0)
        weights = (weights_hard - weights_soft).detach() + weights_soft

        eps = torch.randn_like(mu)
        samples = mu + sigma * eps
        return (weights.unsqueeze(-1) * samples).sum(dim=2)

    def _get_hidden_features(self, beta: torch.Tensor) -> torch.Tensor:
        return self._shared_forward(beta)


def create_model(
    input_size: int,
    output_size: int,
    hidden_sizes: Iterable[int] = (128, 128),
    n_components: int = 5,
):
    return MixtureDensityNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        n_components=n_components,
    )


def save(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(model, path: str, device=None):
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, path: str, optimizer=None, device=None):
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
):
    """Negative log-likelihood loss for the MDN."""
    X, u, Y, s = batch

    alpha_gt, _ = input_function_encoder.compute_coefficients(X, u)
    beta_gt, _ = output_function_encoder.compute_coefficients(Y, s)

    log_likelihood = model.log_prob(alpha_gt, beta_gt)
    loss = -log_likelihood.mean()

    return loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    input_function_encoder,
    output_function_encoder,
    n_epochs,
    summary_writer,
    params,
    model_name,
    forward_model,
    resume_from_checkpoint: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 100,
    device=None,
):
    start_epoch = 0

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")

    if resume_from_checkpoint and checkpoint_path is not None and os.path.exists(checkpoint_path):
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
            forward_model=forward_model,
        )
        loss.backward()
        optimizer.step()

        summary_writer.add_scalars("loss/train", {model_name: loss.item()}, epoch)

        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        if forward_model is not None:
            with torch.no_grad():
                test_batch = next(iter(test_dataloader))
                resim_coeff_loss, resim_pred_loss = resimulation_loss(
                    model=model,
                    batch=test_batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    forward_model=forward_model,
                    n_samples=5,
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
    forward_model=None,
):
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        loss = loss_function(
            model=model,
            batch=batch,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
        )
    return loss.item()


def resimulation_loss(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples: int = 10,
):
    """Resimulation diagnostic for the MDN."""
    if forward_model is None:
        raise ValueError("forward_model is required for resimulation metrics.")

    X, u, Y, s = batch
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    forward_model.eval()
    with torch.no_grad():
        alpha_samples = model.inverse(beta_target, n_samples=n_samples)
        if alpha_samples.dim() == 2:
            alpha_samples = alpha_samples.unsqueeze(0)

        effective_samples = alpha_samples.shape[0]
        alpha_dim = alpha_samples.shape[-1]
        batch_size = beta_target.shape[0]

        alpha_samples_flat = alpha_samples.reshape(-1, alpha_dim)
        beta_resim_flat = forward_model(alpha_samples_flat)
        beta_resim = beta_resim_flat.view(effective_samples, batch_size, -1)

        beta_target_expanded = beta_target.unsqueeze(0).expand(effective_samples, -1, -1)
        mse_per_sample = torch.mean((beta_resim - beta_target_expanded) ** 2, dim=(1, 2))
        resim_coeff_loss = torch.mean(mse_per_sample)

        pred_errors = []
        for i in range(effective_samples):
            s_pred = output_function_encoder(Y, beta_resim[i])
            pred_errors.append(torch.nn.functional.mse_loss(s_pred, s))
        resim_pred_loss = torch.mean(torch.stack(pred_errors))

    model.train()
    return resim_coeff_loss.item(), resim_pred_loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point
        beta, _ = output_function_encoder.compute_coefficients(Y, s)
        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)
        return pred, alpha_pred
