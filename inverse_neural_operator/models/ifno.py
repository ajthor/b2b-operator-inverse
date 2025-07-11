import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import os
from neuralop.layers.fno_block import FNOBlocks


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.GELU(),
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def forward2(self, input):
        mu, log_var = self.encode(input)
        return [self.decode(mu), input, mu, log_var]


class IFNO(nn.Module):
    def __init__(
        self,
        modes1: int = 16,
        modes2: int = 16,
        width: int = 64,
        beta: float = 2.0,
        n_layers: int = 4,
        padding: int = 20,
        vae_latent_dim: int = 24,
        resolution: int = 64,
        mm: int = 64,  # Additional dimension parameter
        input_channels: int = 3,  # Input channels for projection layers
        output_channels: int = 3,  # Output channels
        intermediate_dim: int = 64,  # Intermediate dimension for p1, p2
        vae_hidden_dims: list = None,  # VAE hidden dimensions
    ):
        super(IFNO, self).__init__()
        self.beta = beta
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.n_layers = n_layers
        self.resolution = resolution
        self.mm = mm
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.intermediate_dim = intermediate_dim

        # Input/output projection layers (configurable)
        self.p0 = nn.Linear(input_channels, 1)
        self.p1 = nn.Linear(intermediate_dim, self.width)
        self.p2 = nn.Linear(intermediate_dim, self.width)
        self.p4 = nn.Linear(input_channels, 1)

        # Output projection layers (configurable)
        self.q1 = MLP(self.width, output_channels, self.width * 4)
        self.q2 = MLP(self.width, intermediate_dim, self.width * 4)
        self.q3 = MLP(self.mm, output_channels, self.width * 4)

        # Half the width for the coupling layers
        self.half_width = int(self.width / 2)

        # FNO blocks and MLP layers
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.ws = nn.ModuleList()

        for _ in range(2 * self.n_layers):
            self.convs.append(
                FNOBlocks(self.half_width, self.half_width,
                          (self.modes1, self.modes2))
            )
            self.mlps.append(
                MLP(self.half_width, self.half_width, self.half_width))
            self.ws.append(nn.Conv2d(self.half_width, self.half_width, 1))

        # VAE for reconstruction (configurable)
        self.vae_net = VanillaVAE(
            in_channels=1, 
            latent_dim=vae_latent_dim,
            hidden_dims=vae_hidden_dims
        )

    def _softplus(self, x):
        """Softplus activation with beta parameter"""
        return torch.nn.Softplus(beta=self.beta)(x)

    def forward(self, x):
        """Forward pass from input to output"""
        s = self.resolution
        mm = self.mm
        awidth = self.half_width
        
        # Input processing
        x = self.p0(x)
        x = x.reshape(x.shape[0], s, s, 1).repeat(1, 1, 1, mm)
        x = x.permute(0, 3, 2, 1)
        x = self.p1(x)
        x = x.permute(0, 3, 1, 2)
        
        # Split for coupling layers
        u1 = x[:, :awidth, :, :]
        u2 = x[:, awidth:, :, :]

        # Coupling layers
        for i in range(self.n_layers):
            u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            s2 = s2[..., : (s2.size(-2) - self.padding),
                    : (s2.size(-1) - self.padding)]

            v1 = u1 * self._softplus(s2)
            
            v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            s1 = s1[..., : (s1.size(-2) - self.padding),
                    : (s1.size(-1) - self.padding)]
            v2 = u2 * self._softplus(s1)

            u1 = v1
            u2 = v2

        # Output processing
        x = torch.cat((u1, u2), axis=1)
        y_pred = self.q1(x)
        y_pred = y_pred.permute(0, 2, 3, 1)

        return y_pred

    def inverse(self, y):
        """Backward pass from output to input (inverse operation)"""
        s = self.resolution
        mm = self.mm
        awidth = self.half_width
        
        # Input processing for inverse
        y = self.p4(y)
        y = y.reshape(y.shape[0], mm, s, 1).repeat(1, 1, 1, self.intermediate_dim)
        v = self.p2(y)
        v = v.permute(0, 3, 1, 2)
        
        # Split for inverse coupling layers
        v1 = v[:, :awidth, :, :]
        v2 = v[:, awidth:, :, :]

        # Inverse coupling layers (reverse order)
        for i in range(self.n_layers - 1, -1, -1):
            v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            s1 = s1[..., : (s1.size(-2) - self.padding),
                    : (s1.size(-1) - self.padding)]
            u2 = v2 * self._softplus(s1) ** (-1)
            
            u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            s2 = s2[..., : (s2.size(-2) - self.padding),
                    : (s2.size(-1) - self.padding)]
            u1 = v1 * self._softplus(s2) ** (-1)
            
            v1 = u1
            v2 = u2

        # Output processing for inverse
        x = torch.cat((v1, v2), axis=1)
        x = self.q2(x)
        x = x.permute(0, 2, 1, 3)
        x_preds = self.q3(x)
        x_preds = x_preds.permute(0, 2, 3, 1)

        return x_preds


def create_model(
    modes1=16,
    modes2=16,
    width=64,
    beta=2.0,
    n_layers=4,
    padding=20,
    vae_latent_dim=24,
    resolution=64,
    mm=64,
    input_channels=3,
    output_channels=3,
    intermediate_dim=64,
    vae_hidden_dims=None,
):
    """
    Create an IFNO model with configurable parameters.

    Args:
        modes1: Number of modes in first dimension
        modes2: Number of modes in second dimension
        width: Hidden dimension width
        beta: Beta parameter for softplus activation
        n_layers: Number of coupling layers
        padding: Padding for convolutions
        vae_latent_dim: Latent dimension for VAE
        resolution: Spatial resolution
        mm: Additional dimension parameter
        input_channels: Number of input channels
        output_channels: Number of output channels
        intermediate_dim: Intermediate dimension for projections
        vae_hidden_dims: VAE hidden dimensions list

    Returns:
        IFNO instance
    """
    return IFNO(
        modes1=modes1,
        modes2=modes2,
        width=width,
        beta=beta,
        n_layers=n_layers,
        padding=padding,
        vae_latent_dim=vae_latent_dim,
        resolution=resolution,
        mm=mm,
        input_channels=input_channels,
        output_channels=output_channels,
        intermediate_dim=intermediate_dim,
        vae_hidden_dims=vae_hidden_dims,
    )


def compute_vae_loss(model, pred_x, x_true, x_normalizer, kl_weight=0.01):
    """Compute VAE training loss separate from model"""
    batch_size = pred_x.shape[0]
    recon_img, _, mu, log_var = model.vae_net.forward(pred_x[:, :1, :, :])

    # KL divergence loss
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var -
                         mu**2 - log_var.exp(), dim=1), dim=0)

    # Reconstruction loss
    myloss = LpLoss(size_average=False)
    unnorm_img = x_normalizer.decode(x_true.permute(0, 2, 3, 1).clone())
    unnorm_recon_img = x_normalizer.decode(
        torch.cat(
            (
                recon_img.permute(0, 2, 3, 1).clone(),
                x_true.permute(0, 2, 3, 1).clone()[:, :, :, 1:],
            ),
            axis=-1,
        )
    )

    mse_loss = myloss(unnorm_recon_img.reshape(
        batch_size, -1), unnorm_img.reshape(batch_size, -1))
    loss = kl_weight * kl_loss + mse_loss

    return loss, recon_img


def compute_reconstruction_loss(pred, target):
    """Compute reconstruction loss for consistency"""
    return ((pred - target) ** 2).mean()


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


def load_checkpoint(model, path, optimizer=None, device=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()
    return model, optimizer, checkpoint["epoch"], checkpoint["loss"]


class LpLoss:
    """Lp loss for measuring relative error"""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) -
            y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def loss_function(model, batch, x_normalizer, y_normalizer, kl_weight=0.01):
    """
    Compute loss for IFNO model with clean separation of concerns
    
    Args:
        model: IFNO model
        batch: Batch of data (x, y)
        x_normalizer: Normalizer for input data
        y_normalizer: Normalizer for output data
        kl_weight: Weight for KL divergence in VAE loss
    
    Returns:
        Total loss
    """
    x, y = batch
    batch_size = x.shape[0]
    resolution = x.shape[1]  # Assuming square resolution
    
    # Forward pass
    pred_y = model(x)
    pred_y = pred_y.reshape(batch_size, resolution, resolution, -1)
    
    # Denormalize for loss computation
    pred_y_denorm = y_normalizer.decode(pred_y.clone())
    y_denorm = y_normalizer.decode(y.clone())
    
    # Forward loss
    myloss = LpLoss(size_average=False)
    forward_loss = (
        myloss(pred_y_denorm[:, :, :, 0], y_denorm[:, :, :, 0])
        + myloss(pred_y_denorm[:, :, :, 1:], y_denorm[:, :, :, 1:]) / (100 * batch_size ** 2)
    )
    
    # Backward pass
    pred_x = model.inverse(y.reshape(batch_size, resolution, resolution, -1))
    
    # VAE loss for backward pass
    vae_loss, _ = compute_vae_loss(
        model, pred_x.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2), 
        x_normalizer, kl_weight
    )
    
    # Grid loss for backward pass
    x_denorm = x_normalizer.decode(x.reshape(batch_size, resolution, resolution, -1).clone())
    grid_loss = myloss(pred_x[:, :, :, 1:], x_denorm[:, :, :, 1:]) / (100 * batch_size ** 2)
    
    # Reconstruction losses for consistency
    # Forward reconstruction: check if input can be reconstructed from intermediate representations
    x_recon = model.inverse(pred_y)
    x_recon_denorm = x_normalizer.decode(x_recon.clone())
    forward_recon_loss = compute_reconstruction_loss(x_recon_denorm, x_denorm)
    
    # Backward reconstruction: check if output can be reconstructed from intermediate representations  
    y_recon = model(pred_x)
    y_recon_denorm = y_normalizer.decode(y_recon.clone())
    backward_recon_loss = compute_reconstruction_loss(y_recon_denorm, y_denorm)
    
    backward_loss = vae_loss + grid_loss + backward_recon_loss
    total_loss = forward_loss + backward_loss + forward_recon_loss
    
    return total_loss


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer_forward,
    optimizer_backward,
    x_normalizer,
    y_normalizer,
    n_epochs,
    summary_writer,
    params,
    model_name,
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
    kl_weight=0.01,
):
    """
    Train IFNO model with clean separation of concerns

    Args:
        model: IFNO model
        train_dataloader: Training data loader
        test_dataloader: Test data loader
        optimizer_forward: Optimizer for forward pass
        optimizer_backward: Optimizer for backward pass
        x_normalizer: Input data normalizer
        y_normalizer: Output data normalizer
        n_epochs: Number of training epochs
        summary_writer: TensorBoard writer
        params: Training parameters
        model_name: Name for saving
        resume_from_checkpoint: Whether to resume from checkpoint
        checkpoint_dir: Directory for checkpoints
        checkpoint_interval: Interval for saving checkpoints
        device: Device to train on
        kl_weight: Weight for KL divergence loss
    """
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")
    if resume_from_checkpoint:
        if os.path.exists(checkpoint_path):
            model, _, start_epoch, loss = load_checkpoint(
                model=model, path=checkpoint_path, device=device
            )
            print(f"Resuming training from epoch {start_epoch}...")

    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
        model.train()

        for batch in train_dataloader:
            x, y = batch
            if device is not None:
                x, y = x.to(device), y.to(device)

            batch_size = x.shape[0]
            resolution = x.shape[1]

            # Forward pass training
            optimizer_forward.zero_grad()
            pred_y = model(x)
            pred_y = pred_y.reshape(batch_size, resolution, resolution, -1)
            pred_y_denorm = y_normalizer.decode(pred_y.clone())
            y_denorm = y_normalizer.decode(y.clone())

            myloss = LpLoss(size_average=False)
            forward_loss = (
                myloss(pred_y_denorm[:, :, :, 0], y_denorm[:, :, :, 0])
                + myloss(pred_y_denorm[:, :, :, 1:], y_denorm[:, :, :, 1:]) / (100 * batch_size ** 2)
            )

            forward_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer_forward.step()

            # Backward pass training
            optimizer_backward.zero_grad()
            pred_x = model.inverse(y.reshape(batch_size, resolution, resolution, -1))

            # VAE loss
            vae_loss, _ = compute_vae_loss(
                model, pred_x.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2),
                x_normalizer, kl_weight
            )

            # Grid loss
            x_denorm = x_normalizer.decode(x.reshape(batch_size, resolution, resolution, -1).clone())
            grid_loss = myloss(pred_x[:, :, :, 1:], x_denorm[:, :, :, 1:]) / (100 * batch_size ** 2)

            backward_loss = vae_loss + grid_loss
            backward_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer_backward.step()

        # Test evaluation
        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            kl_weight=kl_weight,
        )

        # Logging
        summary_writer.add_scalars("loss/forward", {model_name: forward_loss.item()}, epoch)
        summary_writer.add_scalars("loss/backward", {model_name: backward_loss.item()}, epoch)
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer_forward, epoch + 1, avg_test_loss, checkpoint_path)

        tqdm_bar.set_postfix_str(
            f"forward_loss {forward_loss.item():.4e} backward_loss {backward_loss.item():.4e} test_loss {avg_test_loss:.4e}"
        )
        tqdm_bar.update(1)


def test_model(model, test_dataloader, x_normalizer, y_normalizer, kl_weight=0.01):
    """Test the IFNO model with clean separation"""
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            batch_size = x.shape[0]
            resolution = x.shape[1]

            # Forward test
            pred_y = model(x)
            pred_y = pred_y.reshape(batch_size, resolution, resolution, -1)
            pred_y_denorm = y_normalizer.decode(pred_y.clone())
            y_denorm = y_normalizer.decode(y.clone())

            myloss = LpLoss(size_average=False)
            forward_loss = myloss(pred_y_denorm[:, :, :, 0], y_denorm[:, :, :, 0])

            # Backward test
            pred_x = model.inverse(y.reshape(batch_size, resolution, resolution, -1))
            pred_x_vae, _, _, _ = model.vae_net.forward2(
                pred_x[:, :, :, 0].reshape(batch_size, 1, resolution, resolution))

            pred_x_final = x_normalizer.decode(
                torch.cat(
                    (pred_x_vae.reshape(batch_size, resolution, resolution, 1).clone(), x[:, :, :, 1:]),
                    axis=-1,
                )
            )
            x_denorm = x_normalizer.decode(x.reshape(batch_size, resolution, resolution, -1).clone())

            backward_loss = myloss(pred_x_final[:, :, :, 0], x_denorm[:, :, :, 0])

            total_test_loss += (forward_loss.item() + backward_loss.item()) / 2

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate(model, point, x_normalizer, y_normalizer):
    """Evaluate model on a single data point with clean separation"""
    model.eval()
    with torch.no_grad():
        x, y = point
        batch_size = x.shape[0]
        resolution = x.shape[1]

        # Forward evaluation
        pred_y = model(x)
        pred_y = pred_y.reshape(batch_size, resolution, resolution, -1)
        pred_y_denorm = y_normalizer.decode(pred_y.clone())

        # Backward evaluation  
        pred_x = model.inverse(y.reshape(batch_size, resolution, resolution, -1))
        pred_x_vae, _, _, _ = model.vae_net.forward2(
            pred_x[:, :, :, 0].reshape(batch_size, 1, resolution, resolution))

        pred_x_final = x_normalizer.decode(
            torch.cat(
                (pred_x_vae.reshape(batch_size, resolution, resolution, 1).clone(), x[:, :, :, 1:]),
                axis=-1,
            )
        )

        return pred_y_denorm, pred_x_final
