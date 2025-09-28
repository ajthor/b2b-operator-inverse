import torch
import torch.nn.functional as F

import tqdm
import os


class MixtureDensityNetwork(torch.nn.Module):
    """
    Mixture Density Network that takes observation y as input and predicts
    a Gaussian mixture p(x|y) with full precision matrices Σ_x^{-1}.

    The network predicts mixture weights, means, and precision matrices.
    Used for going from beta coefficients to alpha coefficients.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [128, 128],
        n_components: int = 5,
        activation=torch.nn.ReLU(),
        bias=True,
    ):
        super(MixtureDensityNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.n_components = n_components
        self.activation = activation

        # Shared hidden layers
        self.layers = torch.nn.ModuleList()
        sizes = [input_size] + hidden_sizes
        for i in range(len(sizes) - 1):
            self.layers.append(
                torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias),
            )

        # Output heads for mixture components
        final_hidden_size = hidden_sizes[-1]

        # Mixture weights (π)
        self.pi_head = torch.nn.Linear(final_hidden_size, n_components, bias=bias)

        # Component means (μ)
        self.mu_head = torch.nn.Linear(
            final_hidden_size, n_components * output_size, bias=bias
        )

        # Component precision matrices (Σ^{-1}) - we'll predict the lower triangular part
        # For full precision matrix, we need output_size * (output_size + 1) / 2 parameters per component
        n_precision_params = output_size * (output_size + 1) // 2
        self.precision_head = torch.nn.Linear(
            final_hidden_size, n_components * n_precision_params, bias=bias
        )

    def forward(self, y):
        """
        Forward pass through the network.

        Args:
            y: Observation tensor of shape (batch_size, input_size)

        Returns:
            tuple: (pi, mu, precision_matrices)
                - pi: mixture weights (batch_size, n_components)
                - mu: component means (batch_size, n_components, output_size)
                - precision_matrices: precision matrices (batch_size, n_components, output_size, output_size)
        """
        batch_size = y.shape[0]

        # Forward through shared layers
        x = y
        for layer in self.layers:
            x = self.activation(layer(x))

        # Compute mixture weights (apply softmax to ensure they sum to 1)
        pi_logits = self.pi_head(x)
        pi = F.softmax(pi_logits, dim=-1)  # (batch_size, n_components)

        # Compute component means
        mu_flat = self.mu_head(x)  # (batch_size, n_components * output_size)
        mu = mu_flat.view(batch_size, self.n_components, self.output_size)

        # Compute precision matrices
        precision_flat = self.precision_head(
            x
        )  # (batch_size, n_components * n_precision_params)
        n_precision_params = self.output_size * (self.output_size + 1) // 2
        precision_params = precision_flat.view(
            batch_size, self.n_components, n_precision_params
        )

        # Build precision matrices from lower triangular parameters
        precision_matrices = self._build_precision_matrices(precision_params)

        return pi, mu, precision_matrices

    def _build_precision_matrices(self, precision_params):
        """
        Build full precision matrices from lower triangular parameters.

        Args:
            precision_params: (batch_size, n_components, n_precision_params)

        Returns:
            precision_matrices: (batch_size, n_components, output_size, output_size)
        """
        batch_size, n_components, _ = precision_params.shape
        precision_matrices = torch.zeros(
            batch_size,
            n_components,
            self.output_size,
            self.output_size,
            device=precision_params.device,
            dtype=precision_params.dtype,
        )

        # Fill lower triangular part
        tril_indices = torch.tril_indices(self.output_size, self.output_size, offset=0)

        for k in range(n_components):
            # For each component, build the lower triangular matrix
            L = torch.zeros(
                batch_size,
                self.output_size,
                self.output_size,
                device=precision_params.device,
                dtype=precision_params.dtype,
            )

            # Fill lower triangular part
            L[:, tril_indices[0], tril_indices[1]] = precision_params[:, k, :]

            # Ensure positive diagonal elements with better numerical stability
            diag_indices = torch.arange(self.output_size)
            # Use softplus with larger minimum value for better conditioning
            L[:, diag_indices, diag_indices] = (
                F.softplus(L[:, diag_indices, diag_indices]) + 1e-3
            )

            # Precision matrix is L @ L^T (ensures positive definiteness)
            precision_matrices[:, k, :, :] = torch.bmm(L, L.transpose(-2, -1))

        return precision_matrices

    def inverse(self, beta):
        """
        Sample from the mixture distribution p(x|y) given observation y=beta.

        Args:
            beta: Observation tensor of shape (batch_size, input_size)

        Returns:
            alpha: Sampled alpha coefficients of shape (batch_size, output_size)
        """
        with torch.no_grad():
            pi, mu, precision_matrices = self.forward(beta)
            batch_size = beta.shape[0]

            # Sample component indices based on mixture weights
            component_dist = torch.distributions.Categorical(pi)
            component_indices = component_dist.sample()  # (batch_size,)

            # For each sample, use the selected component's parameters
            selected_mu = mu[
                torch.arange(batch_size), component_indices
            ]  # (batch_size, output_size)
            selected_precision = precision_matrices[
                torch.arange(batch_size), component_indices
            ]  # (batch_size, output_size, output_size)

            # Improve numerical stability: symmetrize and add jitter
            eye = torch.eye(
                self.output_size,
                device=selected_precision.device,
                dtype=selected_precision.dtype,
            ).unsqueeze(0)
            selected_precision = (
                0.5 * (selected_precision + selected_precision.transpose(-1, -2))
                + 1e-6 * eye
            )

            # Sample from multivariate normal using precision directly (avoids inversion)
            dist = torch.distributions.MultivariateNormal(
                selected_mu, precision_matrix=selected_precision
            )
            alpha = dist.sample()

            return alpha

    def log_prob(self, alpha, beta):
        """
        Compute log probability of alpha given beta under the mixture model.

        Args:
            alpha: Target values (batch_size, output_size)
            beta: Observations (batch_size, input_size)

        Returns:
            log_prob: Log probabilities (batch_size,)
        """
        pi, mu, precision_matrices = self.forward(beta)
        batch_size = alpha.shape[0]

        # Use mixture weights from forward pass for consistency
        # Add small epsilon to avoid log(0)
        log_pi = torch.log(pi + 1e-12)  # (batch_size, n_components)

        # Compute log probabilities for each component
        log_probs_components = torch.zeros(
            batch_size, self.n_components, device=alpha.device
        )

        for k in range(self.n_components):
            # Extract component parameters
            mu_k = mu[:, k, :]  # (batch_size, output_size)
            precision_k = precision_matrices[
                :, k, :, :
            ]  # (batch_size, output_size, output_size)

            # Compute quadratic term: (x - μ)^T Σ^{-1} (x - μ)
            diff = alpha - mu_k  # (batch_size, output_size)
            quadratic = torch.sum(
                diff.unsqueeze(-2) @ precision_k @ diff.unsqueeze(-1), dim=(-2, -1)
            )
            quadratic = quadratic.squeeze(-1)  # (batch_size,)

            # Symmetrize and add small jitter for numerical stability
            precision_k = 0.5 * (precision_k + precision_k.transpose(-1, -2))
            precision_k = precision_k + 1e-6 * torch.eye(
                self.output_size, device=precision_k.device, dtype=precision_k.dtype
            ).unsqueeze(0)

            # Compute log-determinant of precision matrix robustly
            sign, logabsdet = torch.linalg.slogdet(precision_k)
            # For SPD matrices, sign should be +1; clamp otherwise to avoid NaNs
            log_det_precision = torch.where(
                sign > 0, logabsdet, torch.full_like(logabsdet, -50.0)
            )

            # Clamp log determinant to prevent extreme values
            log_det_precision = torch.clamp(log_det_precision, min=-50.0, max=50.0)

            # Log probability for this component (including normalization constant)
            # Normalization constant: -0.5 * output_size * log(2π)
            normalization_constant = (
                -0.5
                * self.output_size
                * torch.log(
                    torch.tensor(2 * torch.pi, device=alpha.device, dtype=alpha.dtype)
                )
            )
            log_prob_k = (
                -0.5 * quadratic + 0.5 * log_det_precision + normalization_constant
            )
            log_probs_components[:, k] = log_prob_k

        # Add mixture weights and compute log-sum-exp with better numerical stability
        log_probs_weighted = log_probs_components + log_pi
        log_prob_mixture = torch.logsumexp(log_probs_weighted, dim=-1)

        return log_prob_mixture

    def _get_hidden_features(self, y):
        """Helper method to get hidden features."""
        x = y
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


def create_model(
    input_size: int,
    output_size: int,
    hidden_sizes: list[int] = [128, 128],
    n_components: int = 5,
):
    """
    Create a Mixture Density Network model.

    Args:
        input_size: Size of the input (beta coefficients)
        output_size: Size of the output (alpha coefficients)
        hidden_sizes: List of hidden layer sizes
        n_components: Number of mixture components

    Returns:
        MixtureDensityNetwork instance
    """
    return MixtureDensityNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        n_components=n_components,
    )


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
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
    checkpoint = torch.load(path, map_location=device)
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
):
    """
    Loss function for Mixture Density Network.
    Uses the direct loss formula from the docs: L(μx,Σ⁻¹x) = ½·(x-μ)ᵀ·Σ⁻¹·(x-μ) - log|Σ⁻¹|^½
    """
    X, u, Y, s = batch

    # Get alpha and beta coefficients
    alpha, _ = input_function_encoder.compute_coefficients(X, u)
    beta, _ = output_function_encoder.compute_coefficients(Y, s)

    # Get mixture parameters from model
    pi, mu, precision_matrices = model.forward(beta)
    batch_size = alpha.shape[0]
    n_components = model.n_components
    output_size = model.output_size

    # Compute loss for each component
    component_losses = torch.zeros(batch_size, n_components, device=alpha.device)

    for k in range(n_components):
        # Extract component parameters
        mu_k = mu[:, k, :]  # (batch_size, output_size)
        precision_k = precision_matrices[
            :, k, :, :
        ]  # (batch_size, output_size, output_size)

        # Symmetrize and add small jitter for numerical stability
        precision_k = 0.5 * (precision_k + precision_k.transpose(-1, -2))
        precision_k = precision_k + 1e-6 * torch.eye(
            output_size, device=precision_k.device, dtype=precision_k.dtype
        ).unsqueeze(0)

        # Compute quadratic term: ½·(x-μ)ᵀ·Σ⁻¹·(x-μ)
        diff = alpha * mu_k  # (batch_size, output_size)
        quadratic = torch.sum(
            diff.unsqueeze(-2) @ precision_k @ diff.unsqueeze(-1), dim=(-2, -1)
        )
        quadratic = quadratic.squeeze(-1)  # (batch_size,)
        quadratic_term = 0.5 * quadratic

        # Compute log-determinant term: log|Σ⁻¹|^½ = 0.5 * log|Σ⁻¹|
        sign, logabsdet = torch.linalg.slogdet(precision_k)
        log_det_precision = torch.where(
            sign > 0, logabsdet, torch.full_like(logabsdet, -50.0)
        )
        log_det_precision = torch.clamp(log_det_precision, min=-50.0, max=50.0)
        log_det_term = 0.5 * log_det_precision

        # Direct loss for this component: L = ½·(x-μ)ᵀ·Σ⁻¹·(x-μ) - log|Σ⁻¹|^½
        component_loss = quadratic_term - log_det_term
        component_losses[:, k] = component_loss

    # Weight by mixture probabilities and sum across components
    # Add small epsilon to avoid numerical issues
    pi_stable = pi + 1e-12
    weighted_losses = component_losses * pi_stable
    total_loss_per_sample = torch.sum(weighted_losses, dim=-1)  # (batch_size,)

    # Return mean loss across batch
    total_loss = total_loss_per_sample.mean()

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
    params,
    model_name,
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
            lambda_forward=getattr(params, "lambda_forward", 0.0),
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
            lambda_forward=getattr(params, "lambda_forward", 0.0),
        )
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        # Compute and log re-simulation loss (average over batches)
        total_resim_loss = 0.0
        n_resim_batches = 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch_resim_loss = resimulation_loss(
                    model=model,
                    batch=batch,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    forward_model=forward_model,
                    n_samples=5,  # MDN uses sampling, keep n_samples=5
                )
                total_resim_loss += batch_resim_loss
                n_resim_batches += 1
        avg_resim_loss = total_resim_loss / max(n_resim_batches, 1)
        summary_writer.add_scalars(
            "loss/resimulation", {model_name: avg_resim_loss}, epoch
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.set_postfix_str(f"loss {avg_test_loss:.4e}")
        tqdm_bar.update(1)


def test_model(
    model,
    test_dataloader,
    input_function_encoder,
    output_function_encoder,
    forward_model=None,
    lambda_forward=0.0,
):
    model.eval()
    # total_test_loss = 0.0
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        loss = loss_function(
            model=model,
            batch=batch,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
            forward_model=forward_model,
            lambda_forward=lambda_forward,
        )
        # for batch in test_dataloader:
        #     loss = loss_function(
        #         model=model,
        #         batch=batch,
        #         input_function_encoder=input_function_encoder,
        #         output_function_encoder=output_function_encoder,
        #         forward_model=forward_model,
        #         lambda_forward=lambda_forward,
        #     )
        #     total_test_loss += loss.item()

    # avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return loss.item()


def resimulation_loss(
    model,
    batch,
    input_function_encoder,
    output_function_encoder,
    forward_model,
    n_samples=5,
):
    """
    Compute re-simulation loss for Mixture Density Network.

    For MDN: Sample from the mixture given beta*, apply forward operator, measure MSE to beta*
    """

    X, u, Y, s = batch

    # Get target beta coefficients
    beta_target, _ = output_function_encoder.compute_coefficients(Y, s)

    model.eval()
    if forward_model is None:
        return 0.0
    forward_model.eval()
    with torch.no_grad():
        # Generate samples from the mixture
        samples = []
        batch_size = beta_target.shape[0]

        for _ in range(n_samples):
            # Sample from the mixture distribution
            alpha_sample = model.inverse(beta_target)
            samples.append(alpha_sample)

        alpha_samples = torch.stack(
            samples, dim=0
        )  # [n_samples, batch_size, alpha_dim]

        # Reshape for forward pass: [n_samples * batch_size, alpha_dim]
        alpha_dim = alpha_samples.shape[2]
        alpha_samples_flat = alpha_samples.view(-1, alpha_dim)

        # Apply forward operator to generated samples
        beta_predicted_flat = forward_model(
            alpha_samples_flat
        )  # [n_samples * batch_size, beta_dim]

        # Reshape back: [n_samples, batch_size, beta_dim]
        beta_predicted = beta_predicted_flat.view(n_samples, batch_size, -1)

        # Compute MSE between predicted and target beta for each sample, then average
        beta_target_expanded = beta_target.unsqueeze(0).expand(
            n_samples, -1, -1
        )  # [n_samples, batch_size, beta_dim]
        mse_per_sample = torch.mean(
            (beta_predicted - beta_target_expanded) ** 2, dim=(1, 2)
        )  # [n_samples]
        resim_loss = torch.mean(mse_per_sample)

    model.train()
    return resim_loss.item()


def evaluate(model, point, input_function_encoder, output_function_encoder):
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        beta, _ = output_function_encoder.compute_coefficients(Y, s)

        alpha_pred = model.inverse(beta)
        pred = input_function_encoder(X, alpha_pred)

        return pred
