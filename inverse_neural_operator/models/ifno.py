import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import os
import warnings
from neuralop.layers.fno_block import FNOBlocks


# Utility classes for faithful iFNO implementation
def relative_l2_loss(x, y):
    """Simple relative L2 loss helper function"""
    num_examples = x.size()[0]
    diff_norms = torch.norm(
        x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=2, dim=1
    )
    y_norms = torch.norm(y.reshape(num_examples, -1), p=2, dim=1)
    return torch.mean(diff_norms / y_norms)


def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )


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
            nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
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
        mm: int = 58,  # Set to original value from provided code
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
                FNOBlocks(self.half_width, self.half_width, (self.modes1, self.modes2))
            )
            self.mlps.append(MLP(self.half_width, self.half_width, self.half_width))
            self.ws.append(nn.Conv2d(self.half_width, self.half_width, 1))

        # VAE for reconstruction (configurable)
        self.vae_net = VanillaVAE(
            in_channels=1, latent_dim=vae_latent_dim, hidden_dims=vae_hidden_dims
        )

    def _softplus(self, x):
        """Softplus activation with beta parameter"""
        return torch.nn.Softplus(beta=self.beta)(x)

    def forward(self, x):
        """Forward pass from input to output"""
        s = self.resolution
        mm = self.mm
        awidth = self.half_width

        # Input processing - handle variable input shapes
        batch_size = x.shape[0]
        
        # If x is not already the right shape, reshape it
        if len(x.shape) == 2:
            # Assume x is (batch_size, input_channels) and reshape to spatial format
            x = x.view(batch_size, -1, 1)  # Flatten to (batch, features, 1)
        
        x = self.p0(x)
        
        # Ensure we have the right spatial dimensions
        if x.shape[1] != s * s:
            # Pad or crop to match expected resolution
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), s * s).transpose(1, 2)
        
        x = x.reshape(batch_size, s, s, 1).repeat(1, 1, 1, mm)
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
            s2 = s2[..., : (s2.size(-2) - self.padding), : (s2.size(-1) - self.padding)]

            v1 = u1 * self._softplus(s2)

            v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            s1 = s1[..., : (s1.size(-2) - self.padding), : (s1.size(-1) - self.padding)]
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

        # Input processing for inverse - handle variable input shapes
        batch_size = y.shape[0]
        
        # If y is not already the right shape, reshape it
        if len(y.shape) == 2:
            # Assume y is (batch_size, output_channels) and reshape to spatial format
            y = y.view(batch_size, -1, 1)  # Flatten to (batch, features, 1)
        
        y = self.p4(y)
        
        # Ensure we have the right spatial dimensions
        if y.shape[1] != mm * s:
            # Pad or crop to match expected resolution
            y = F.adaptive_avg_pool1d(y.transpose(1, 2), mm * s).transpose(1, 2)
        
        y = y.reshape(batch_size, mm, s, 1).repeat(1, 1, 1, self.intermediate_dim)
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
            s1 = s1[..., : (s1.size(-2) - self.padding), : (s1.size(-1) - self.padding)]
            u2 = v2 * self._softplus(s1) ** (-1)

            u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            s2 = s2[..., : (s2.size(-2) - self.padding), : (s2.size(-1) - self.padding)]
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
    input_size,
    hidden_sizes=[128, 128],
    n_coupling_layers=2,
    modes1=16,
    modes2=16,
    width=64,
    beta=2.0,
    n_layers=4,
    padding=20,
    vae_latent_dim=24,
    resolution=64,
    mm=58,
    input_channels=3,
    output_channels=3,
    intermediate_dim=64,
    vae_hidden_dims=None,
):
    """
    Create an IFNO model with configurable parameters.

    Compatible with existing training framework - input_size parameter is for compatibility
    but the actual input/output sizes are determined by the function encoders.

    Args:
        input_size: Input size for compatibility (not used directly)
        hidden_sizes: Hidden sizes for compatibility (not used directly)
        n_coupling_layers: Number of coupling layers for compatibility (not used directly)
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


# Loss functions for different training phases - work directly with batch
def ifno_vae_loss(model, batch):
    """VAE pretraining loss using the VAE component"""
    X, u, Y, s = batch
    # Use VAE for reconstruction loss
    # Input u should be reshaped to match VAE input format (batch, channels, height, width)
    batch_size = u.shape[0]
    u_reshaped = u.reshape(batch_size, 1, 64, 64)  # Assuming 64x64 resolution
    
    # VAE forward pass
    vae_out, vae_input, mu, log_var = model.vae_net(u_reshaped)
    
    # Reconstruction loss
    reconstruction_loss = relative_l2_loss(vae_out, vae_input)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss.mean()
    
    return reconstruction_loss + 0.01 * kl_loss


def ifno_forward_loss(model, batch):
    """Forward pass loss for IFNO pretraining"""
    X, u, Y, s = batch
    # Forward pass: input u -> model -> pred_s, compare with s
    # Reshape u to match model input format
    u_input = torch.cat([X, u], dim=-1)  # Combine spatial coordinates with function values
    
    pred_s = model(u_input)
    return relative_l2_loss(pred_s, s)


def ifno_backward_loss(model, batch):
    """Backward pass loss for IFNO pretraining"""
    X, u, Y, s = batch
    # Backward pass: input s -> model.inverse -> pred_u, compare with u
    # Reshape s to match model input format
    s_input = torch.cat([Y, s], dim=-1)  # Combine spatial coordinates with function values
    
    pred_u = model.inverse(s_input)
    return relative_l2_loss(pred_u, u)


def ifno_joint_loss(model, batch):
    """Joint training loss - both forward and backward"""
    X, u, Y, s = batch

    # Forward loss: input u -> model -> pred_s, compare with s
    u_input = torch.cat([X, u], dim=-1)
    pred_s = model(u_input)
    forward_loss = relative_l2_loss(pred_s, s)

    # Backward loss: input s -> model.inverse -> pred_u, compare with u
    s_input = torch.cat([Y, s], dim=-1)
    pred_u = model.inverse(s_input)
    backward_loss = relative_l2_loss(pred_u, u)

    return forward_loss, backward_loss


def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_checkpoint(model, optimizer, epoch, loss, path, training_stage="joint"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "loss": loss,
        "training_stage": training_stage,
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
    return (
        model,
        optimizer,
        checkpoint["epoch"],
        checkpoint["loss"],
        checkpoint["training_stage"],
    )


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
    resume_from_checkpoint=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
    device=None,
    epochs_vae=2,
    epochs_ifno=2,
    lr_vae=0.0001,
    lr_ifno=0.005,
    lr_forward=0.0001,
):
    """
    Train IFNO model with three-phase training following original implementation

    Phases:
    1. VAE pretraining
    2. IFNO pretraining (forward + backward)
    3. Joint training

    Maintains compatibility with existing training framework
    """
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")
    if resume_from_checkpoint:
        if os.path.exists(checkpoint_path):
            model, optimizer, start_epoch, _, training_stage = load_checkpoint(
                model=model, path=checkpoint_path, optimizer=optimizer, device=device
            )
            print(
                f"Resuming training from epoch {start_epoch}, stage: {training_stage}..."
            )
    else:
        # Phase 1: VAE Pretraining
        print("Phase 1: VAE Pretraining")
        vae_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_vae)
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, factor=0.9, patience=10, verbose=False
        )

        for epoch in range(epochs_vae):
            model.train()
            train_loss = 0.0

            for batch in train_dataloader:
                vae_optimizer.zero_grad()
                loss = ifno_vae_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                vae_optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_dataloader)
            vae_scheduler.step(avg_loss)
            print(f"VAE Epoch {epoch+1}/{epochs_vae}, Loss: {avg_loss:.6f}")

        # Phase 2: IFNO Pretraining
        print("Phase 2: IFNO Pretraining")
        ifno_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_ifno)
        ifno_scheduler = torch.optim.lr_scheduler.StepLR(
            ifno_optimizer, step_size=100, gamma=0.5
        )

        for epoch in range(epochs_ifno):
            model.train()
            train_forward_loss = 0.0
            train_backward_loss = 0.0

            for batch in train_dataloader:
                # Forward pass training
                ifno_optimizer.zero_grad()
                forward_loss = ifno_forward_loss(model, batch)
                forward_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                ifno_optimizer.step()
                train_forward_loss += forward_loss.item()

                # Backward pass training
                ifno_optimizer.zero_grad()
                backward_loss = ifno_backward_loss(model, batch)
                backward_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                ifno_optimizer.step()
                train_backward_loss += backward_loss.item()

            ifno_scheduler.step()
            avg_forward_loss = train_forward_loss / len(train_dataloader)
            avg_backward_loss = train_backward_loss / len(train_dataloader)
            print(
                f"IFNO Epoch {epoch+1}/{epochs_ifno}, Forward: {avg_forward_loss:.6f}, Backward: {avg_backward_loss:.6f}"
            )

    # Phase 3: Joint Training
    print("Phase 3: Joint Training")
    joint_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_forward)

    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            joint_optimizer.zero_grad()
            forward_loss, backward_loss = ifno_joint_loss(model, batch)
            total_loss = forward_loss + backward_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            joint_optimizer.step()
            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        # Test evaluation
        avg_test_loss = test_model(
            model=model,
            test_dataloader=test_dataloader,
            input_function_encoder=input_function_encoder,
            output_function_encoder=output_function_encoder,
        )

        # Logging
        summary_writer.add_scalars("loss/train", {model_name: avg_epoch_loss}, epoch)
        summary_writer.add_scalars("loss/test", {model_name: avg_test_loss}, epoch)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model,
                joint_optimizer,
                epoch + 1,
                avg_test_loss,
                checkpoint_path,
                "joint",
            )

        tqdm_bar.set_postfix_str(
            f"train_loss {avg_epoch_loss:.4e} test_loss {avg_test_loss:.4e}"
        )
        tqdm_bar.update(1)


def test_model(model, test_dataloader, input_function_encoder, output_function_encoder):
    """Test the IFNO model with compatibility for main training framework"""
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in test_dataloader:
            forward_loss, backward_loss = ifno_joint_loss(model, batch)
            total_loss = forward_loss + backward_loss
            total_test_loss += total_loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader.dataset)
    return avg_test_loss


def evaluate(model, point, input_function_encoder, output_function_encoder):
    """Evaluate model on a single data point

    Note: input_function_encoder and output_function_encoder are unused placeholders
    for compatibility - iFNO works directly with raw data X, u, Y, s
    """
    model.eval()
    with torch.no_grad():
        X, u, Y, s = point

        # Forward pass: predict s from u
        u_input = torch.cat([X, u], dim=-1)
        pred_s = model(u_input)
        
        return pred_s
