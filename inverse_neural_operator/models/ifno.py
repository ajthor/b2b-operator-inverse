import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import os
import warnings
from neuralop.layers.fno_block import FNOBlocks


# Simple 1D Fourier Layer for IFNO compatibility
class SimpleFourierLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SimpleFourierLayer1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Complex weights for Fourier modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))
        
    def forward(self, x):
        # x: (batch, channels, length)
        batch_size = x.shape[0]
        
        # Fourier transform
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Extract relevant modes
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1), 
                           dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant modes  
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        
        # Inverse Fourier transform
        x = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
        
        return x


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
        [p.numel() * 2 if p.is_complex() else p.numel()
         for p in model.parameters()]
    )


class MLP1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP1D, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class MLP2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2D, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class MLP3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP3D, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class VAE1D(nn.Module):
    def __init__(self, in_channels, latent_dim, input_length, hidden_dims=None):
        super(VAE1D, self).__init__()
        self.latent_dim = latent_dim
        self.input_length = input_length
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(
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
        
        # Calculate actual flattened size after convolutions using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_output = self.encoder(dummy_input)
            self.encoded_size = dummy_output.numel()
            self.reduced_length = dummy_output.shape[-1]
        
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_var = nn.Linear(self.encoded_size, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.encoded_size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
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
            nn.ConvTranspose1d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.reduced_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        # Adjust output size to match input length
        if result.shape[-1] != self.input_length:
            if result.shape[-1] > self.input_length:
                # Crop if output is too long
                result = result[:, :, :self.input_length]
            else:
                # Pad if output is too short  
                pad_size = self.input_length - result.shape[-1]
                result = F.pad(result, (0, pad_size))
        
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


class VAE2D(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size, hidden_dims=None):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size  # (H, W) tuple
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
        
        # Calculate actual flattened size after convolutions using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1])
            dummy_output = self.encoder(dummy_input)
            self.encoded_size = dummy_output.numel()
            self.decoder_h = dummy_output.shape[-2]
            self.decoder_w = dummy_output.shape[-1]
        
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_var = nn.Linear(self.encoded_size, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.encoded_size)
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
        result = result.view(-1, 512, self.decoder_h, self.decoder_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        # Adjust output size to match input dimensions
        target_h, target_w = self.input_size
        current_h, current_w = result.shape[-2:]
        
        if current_h != target_h or current_w != target_w:
            if current_h > target_h:
                result = result[:, :, :target_h, :]
            elif current_h < target_h:
                pad_h = target_h - current_h
                result = F.pad(result, (0, 0, 0, pad_h))
                
            if current_w > target_w:
                result = result[:, :, :, :target_w]
            elif current_w < target_w:
                pad_w = target_w - current_w
                result = F.pad(result, (0, pad_w, 0, 0))
        
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


class VAE3D(nn.Module):
    def __init__(self, in_channels, latent_dim, input_size, hidden_dims=None):
        super(VAE3D, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size  # (D, H, W) tuple
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(
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
        
        # Calculate actual flattened size after convolutions using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1], input_size[2])
            dummy_output = self.encoder(dummy_input)
            self.encoded_size = dummy_output.numel()
            self.decoder_d = dummy_output.shape[-3]
            self.decoder_h = dummy_output.shape[-2]
            self.decoder_w = dummy_output.shape[-1]
        
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_var = nn.Linear(self.encoded_size, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.encoded_size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
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
            nn.ConvTranspose3d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.Conv3d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.decoder_d, self.decoder_h, self.decoder_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        # Adjust output size to match input dimensions
        target_d, target_h, target_w = self.input_size
        current_d, current_h, current_w = result.shape[-3:]
        
        if current_d != target_d or current_h != target_h or current_w != target_w:
            if current_d > target_d:
                result = result[:, :, :target_d, :, :]
            elif current_d < target_d:
                pad_d = target_d - current_d
                result = F.pad(result, (0, 0, 0, 0, 0, pad_d))
                
            if current_h > target_h:
                result = result[:, :, :, :target_h, :]
            elif current_h < target_h:
                pad_h = target_h - current_h
                result = F.pad(result, (0, 0, 0, pad_h, 0, 0))
                
            if current_w > target_w:
                result = result[:, :, :, :, :target_w]
            elif current_w < target_w:
                pad_w = target_w - current_w
                result = F.pad(result, (0, pad_w, 0, 0, 0, 0))
        
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
        # Dataset-adaptive parameters
        input_spatial_dims: tuple = (64, 64),  # Input spatial dimensions
        output_spatial_dims: tuple = (64, 58),  # Output spatial dimensions
        input_function_channels: int = 1,  # Function value channels in input
        output_function_channels: int = 1,  # Function value channels in output
        coordinate_dim: int = 2,  # Spatial coordinate dimension (1D, 2D, 3D)
        # Legacy parameters for backward compatibility
        resolution: int = None,  # Will be derived from spatial_dims if None
        mm: int = None,  # Will be derived from output_spatial_dims if None
        input_channels: int = None,  # Will be derived from function_channels + coord_dim
        output_channels: int = None,  # Will be derived from function_channels + coord_dim
        intermediate_dim: int = 64,  # Intermediate dimension for projections
        vae_hidden_dims: list = None,  # VAE hidden dimensions
    ):
        super(IFNO, self).__init__()
        self.beta = beta
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding
        self.n_layers = n_layers

        # Dataset-adaptive parameters
        self.input_spatial_dims = input_spatial_dims
        self.output_spatial_dims = output_spatial_dims
        self.input_function_channels = input_function_channels
        self.output_function_channels = output_function_channels
        self.coordinate_dim = coordinate_dim
        self.intermediate_dim = intermediate_dim

        # Derive legacy parameters for compatibility
        self.resolution = resolution or self._get_primary_resolution()
        self.mm = mm or self._get_output_size()
        self.input_channels = input_channels or (
            input_function_channels + coordinate_dim)
        self.output_channels = output_channels or (
            output_function_channels + coordinate_dim)

        # Detect if this is a symmetric vs asymmetric problem
        self.is_symmetric = (input_spatial_dims == output_spatial_dims)

        print(f"iFNO Configuration:")
        print(f"  Input spatial dims: {self.input_spatial_dims}")
        print(f"  Output spatial dims: {self.output_spatial_dims}")
        print(f"  Symmetric: {self.is_symmetric}")
        print(f"  Resolution: {self.resolution}, Output size: {self.mm}")

        # Choose appropriate MLP class based on coordinate dimensions
        if coordinate_dim == 1:
            self.MLP = MLP1D
        elif coordinate_dim == 2:
            self.MLP = MLP2D
        elif coordinate_dim == 3:
            self.MLP = MLP3D
        else:
            raise ValueError(f"Unsupported coordinate dimension: {coordinate_dim}")

        # Adaptive projection layers based on problem symmetry
        if self.is_symmetric:
            # Symmetric problem (like Darcy, Chladni, Heat)
            self.p1 = nn.Linear(self.input_channels, self.width)
            self.p2 = nn.Linear(self.output_channels, self.width)
            self.q1 = self.MLP(self.width, self.output_channels, self.width * 4)
            self.q2 = self.MLP(self.width, self.input_channels, self.width * 4)
            # Optional reconstruction layers for symmetric problems
            self.reconstruction_loss_weight = 1.0
        else:
            # Asymmetric problem (like Wave-Equation, Wave Scattering, FWI)
            self.p0 = nn.Linear(self.input_channels, 1)
            self.p1 = nn.Linear(self.intermediate_dim, self.width)
            self.p2 = nn.Linear(self.intermediate_dim, self.width)
            self.p4 = nn.Linear(self.output_channels, 1)
            self.q1 = self.MLP(self.width, self.output_channels, self.width * 4)
            self.q2 = self.MLP(self.width, self.intermediate_dim, self.width * 4)
            self.q3 = self.MLP(self.mm, self.input_channels, self.width * 4)
            self.reconstruction_loss_weight = 0.0  # No reconstruction loss

        # Half the width for the coupling layers
        self.half_width = int(self.width / 2)

        # FNO blocks and MLP layers
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.ws = nn.ModuleList()

        for _ in range(2 * self.n_layers):
            # Use dimension-adaptive FNO blocks
            if coordinate_dim == 1:
                self.convs.append(
                    SimpleFourierLayer1D(self.half_width, self.half_width, self.modes1)
                )
            elif coordinate_dim == 2:
                self.convs.append(
                    FNOBlocks(self.half_width, self.half_width,
                              (self.modes1, self.modes2))
                )
            elif coordinate_dim == 3:
                # For 3D, we might need a different approach, but let's use 2D for now
                self.convs.append(
                    FNOBlocks(self.half_width, self.half_width,
                              (self.modes1, self.modes2))
                )
            else:
                raise ValueError(f"Unsupported coordinate dimension: {coordinate_dim}")
            self.mlps.append(
                self.MLP(self.half_width, self.half_width, self.half_width))
            # Use appropriate conv layer for dimension
            if coordinate_dim == 1:
                self.ws.append(nn.Conv1d(self.half_width, self.half_width, 1))
            elif coordinate_dim == 2:
                self.ws.append(nn.Conv2d(self.half_width, self.half_width, 1))
            elif coordinate_dim == 3:
                self.ws.append(nn.Conv3d(self.half_width, self.half_width, 1))

        # VAE for reconstruction (dimension-adaptive)
        if coordinate_dim == 1:
            # For 1D data, use primary spatial dimension as length
            vae_input_size = self.input_spatial_dims[0]
            self.vae_net = VAE1D(
                in_channels=1, latent_dim=vae_latent_dim, 
                input_length=vae_input_size, hidden_dims=vae_hidden_dims
            )
        elif coordinate_dim == 2:
            # For 2D data, use spatial dimensions as (H, W)
            vae_input_size = self.input_spatial_dims
            self.vae_net = VAE2D(
                in_channels=1, latent_dim=vae_latent_dim,
                input_size=vae_input_size, hidden_dims=vae_hidden_dims
            )
        elif coordinate_dim == 3:
            # For 3D data, use spatial dimensions as (D, H, W)
            vae_input_size = self.input_spatial_dims
            self.vae_net = VAE3D(
                in_channels=1, latent_dim=vae_latent_dim,
                input_size=vae_input_size, hidden_dims=vae_hidden_dims
            )
        else:
            raise ValueError(f"Unsupported coordinate dimension: {coordinate_dim}")

    def _get_primary_resolution(self):
        """Get primary spatial resolution for FNO operations"""
        if len(self.input_spatial_dims) >= 2:
            return self.input_spatial_dims[0]  # Use first dimension
        else:
            return self.input_spatial_dims[0]  # 1D case

    def _get_output_size(self):
        """Get output size parameter (like mm in original)"""
        if len(self.output_spatial_dims) >= 2:
            return self.output_spatial_dims[1]  # Second dimension (like mm=58)
        else:
            return self.output_spatial_dims[0]  # 1D case

    def _reshape_for_fno(self, x, spatial_dims):
        """Reshape data to appropriate format for dimension-adaptive FNO operations"""
        batch_size = x.shape[0]

        if self.coordinate_dim == 1:
            # 1D data -> keep as 1D but reshape for processing
            size = spatial_dims[0]
            return x.view(batch_size, size, -1)
        elif self.coordinate_dim == 2:
            # 2D data -> direct reshape
            h, w = spatial_dims
            return x.view(batch_size, h, w, -1)
        elif self.coordinate_dim == 3:
            # 3D data -> direct reshape
            d, h, w = spatial_dims
            return x.view(batch_size, d, h, w, -1)
        else:
            raise ValueError(f"Unsupported coordinate dimension: {self.coordinate_dim}")
    
    def _get_vae_input_shape(self, data):
        """Get proper shape for VAE input based on coordinate dimension"""
        batch_size = data.shape[0]
        
        if self.coordinate_dim == 1:
            # Shape: (batch, channels, length)
            return data.reshape(batch_size, 1, self.input_spatial_dims[0])
        elif self.coordinate_dim == 2:
            # Shape: (batch, channels, height, width)
            h, w = self.input_spatial_dims
            return data.reshape(batch_size, 1, h, w)
        elif self.coordinate_dim == 3:
            # Shape: (batch, channels, depth, height, width)
            d, h, w = self.input_spatial_dims
            return data.reshape(batch_size, 1, d, h, w)
        else:
            raise ValueError(f"Unsupported coordinate dimension: {self.coordinate_dim}")

    def _softplus(self, x):
        """Softplus activation with beta parameter"""
        return torch.nn.Softplus(beta=self.beta)(x)

    def forward(self, x):
        """Forward pass from input to output"""
        batch_size = x.shape[0]
        awidth = self.half_width

        # Reshape input to FNO-compatible format
        x = self._reshape_for_fno(x, self.input_spatial_dims)

        if self.is_symmetric:
            # Symmetric problem (like Darcy, Chladni)
            input_x = x  # Store for reconstruction loss
            x = self.p1(x)
            
            # Dimension-adaptive permutation
            if self.coordinate_dim == 1:
                x = x.permute(0, 2, 1)  # (batch, length, channels) -> (batch, channels, length)
            elif self.coordinate_dim == 2:
                x = x.permute(0, 3, 1, 2)  # (batch, h, w, channels) -> (batch, channels, h, w)
            elif self.coordinate_dim == 3:
                x = x.permute(0, 4, 1, 2, 3)  # (batch, d, h, w, channels) -> (batch, channels, d, h, w)

            # Compute reconstruction loss
            if self.coordinate_dim == 1:
                x_recon = self.q2(x).permute(0, 2, 1)
            elif self.coordinate_dim == 2:
                x_recon = self.q2(x).permute(0, 2, 3, 1)
            elif self.coordinate_dim == 3:
                x_recon = self.q2(x).permute(0, 2, 3, 4, 1)
            
            reconstruction_loss = ((x_recon - input_x) **
                                   2).mean() * self.reconstruction_loss_weight
        else:
            # Asymmetric problem (like Wave-Equation)
            x = self.p0(x)

            # Dimension-adaptive reshape for asymmetric case 
            if self.coordinate_dim == 1:
                s = self.resolution
                mm = self.mm
                x = x.reshape(batch_size, s, 1).repeat(1, 1, mm)
                x = x.permute(0, 2, 1)
                x = self.p1(x)
                x = x.permute(0, 2, 1)
            elif self.coordinate_dim == 2:
                s = self.resolution
                mm = self.mm
                x = x.reshape(batch_size, s, s, 1).repeat(1, 1, 1, mm)
                x = x.permute(0, 3, 2, 1)
                x = self.p1(x)
                x = x.permute(0, 3, 1, 2)
            elif self.coordinate_dim == 3:
                # For 3D asymmetric (like FWI), handle appropriately
                d, h, w = self.input_spatial_dims
                mm = self.mm
                x = x.reshape(batch_size, d, h, w, 1).repeat(1, 1, 1, 1, mm)
                x = x.permute(0, 4, 1, 2, 3)
                x = self.p1(x)
                x = x.permute(0, 4, 1, 2, 3)
            
            reconstruction_loss = 0.0

        # Split for coupling layers (dimension-adaptive)
        if self.coordinate_dim == 1:
            u1 = x[:, :awidth, :]
            u2 = x[:, awidth:, :]
        elif self.coordinate_dim == 2:
            u1 = x[:, :awidth, :, :]
            u2 = x[:, awidth:, :, :]
        elif self.coordinate_dim == 3:
            u1 = x[:, :awidth, :, :, :]
            u2 = x[:, awidth:, :, :, :]

        # Coupling layers (dimension-adaptive)
        for i in range(self.n_layers):
            # Dimension-adaptive padding
            if self.coordinate_dim == 1:
                u2_pad = F.pad(u2, [0, self.padding])
            elif self.coordinate_dim == 2:
                u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            elif self.coordinate_dim == 3:
                u2_pad = F.pad(u2, [0, self.padding, 0, self.padding, 0, self.padding])
                
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            
            # Dimension-adaptive cropping
            if self.coordinate_dim == 1:
                s2 = s2[..., : (s2.size(-1) - self.padding)]
            elif self.coordinate_dim == 2:
                s2 = s2[..., : (s2.size(-2) - self.padding),
                        : (s2.size(-1) - self.padding)]
            elif self.coordinate_dim == 3:
                s2 = s2[..., : (s2.size(-3) - self.padding),
                        : (s2.size(-2) - self.padding),
                        : (s2.size(-1) - self.padding)]

            v1 = u1 * self._softplus(s2)

            # Dimension-adaptive padding for v1
            if self.coordinate_dim == 1:
                v1_pad = F.pad(v1, [0, self.padding])
            elif self.coordinate_dim == 2:
                v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            elif self.coordinate_dim == 3:
                v1_pad = F.pad(v1, [0, self.padding, 0, self.padding, 0, self.padding])
                
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            
            # Dimension-adaptive cropping for s1
            if self.coordinate_dim == 1:
                s1 = s1[..., : (s1.size(-1) - self.padding)]
            elif self.coordinate_dim == 2:
                s1 = s1[..., : (s1.size(-2) - self.padding),
                        : (s1.size(-1) - self.padding)]
            elif self.coordinate_dim == 3:
                s1 = s1[..., : (s1.size(-3) - self.padding),
                        : (s1.size(-2) - self.padding),
                        : (s1.size(-1) - self.padding)]
                        
            v2 = u2 * self._softplus(s1)

            u1 = v1
            u2 = v2

        # Output processing
        x = torch.cat((u1, u2), axis=1)
        y_pred = self.q1(x)
        
        # Dimension-adaptive permutation back
        if self.coordinate_dim == 1:
            y_pred = y_pred.permute(0, 2, 1)
        elif self.coordinate_dim == 2:
            y_pred = y_pred.permute(0, 2, 3, 1)
        elif self.coordinate_dim == 3:
            y_pred = y_pred.permute(0, 2, 3, 4, 1)

        # Return appropriate format based on problem type
        if self.is_symmetric:
            return y_pred, reconstruction_loss
        else:
            return y_pred

    def inverse(self, y):
        """Backward pass from output to input (inverse operation)"""
        batch_size = y.shape[0]
        awidth = self.half_width

        # Reshape output to FNO-compatible format
        y = self._reshape_for_fno(y, self.output_spatial_dims)

        if self.is_symmetric:
            # Symmetric problem (like Darcy, Chladni)
            input_y = y  # Store for reconstruction loss
            v = self.p2(y)
            
            # Dimension-adaptive permutation
            if self.coordinate_dim == 1:
                v = v.permute(0, 2, 1)
            elif self.coordinate_dim == 2:
                v = v.permute(0, 3, 1, 2)
            elif self.coordinate_dim == 3:
                v = v.permute(0, 4, 1, 2, 3)

            # Compute reconstruction loss
            if self.coordinate_dim == 1:
                y_recon = self.q1(v).permute(0, 2, 1)
            elif self.coordinate_dim == 2:
                y_recon = self.q1(v).permute(0, 2, 3, 1)
            elif self.coordinate_dim == 3:
                y_recon = self.q1(v).permute(0, 2, 3, 4, 1)
                
            reconstruction_loss = ((y_recon - input_y) **
                                   2).mean() * self.reconstruction_loss_weight
        else:
            # Asymmetric problem (like Wave-Equation)
            y = self.p4(y)

            # Dimension-adaptive reshape for asymmetric case
            if self.coordinate_dim == 1:
                s = self.resolution
                mm = self.mm
                y = y.reshape(batch_size, mm, s, 1).repeat(1, 1, 1, self.intermediate_dim)
                v = self.p2(y)
                v = v.permute(0, 3, 1, 2)
            elif self.coordinate_dim == 2:
                s = self.resolution
                mm = self.mm
                y = y.reshape(batch_size, mm, s, 1).repeat(1, 1, 1, self.intermediate_dim)
                v = self.p2(y)
                v = v.permute(0, 3, 1, 2)
            elif self.coordinate_dim == 3:
                # For 3D asymmetric (like FWI)
                h, w = self.output_spatial_dims
                y = y.reshape(batch_size, h, w, 1).repeat(1, 1, 1, self.intermediate_dim)
                v = self.p2(y)
                v = v.permute(0, 3, 1, 2)
                
            reconstruction_loss = 0.0

        # Split for inverse coupling layers (dimension-adaptive)
        if self.coordinate_dim == 1:
            v1 = v[:, :awidth, :]
            v2 = v[:, awidth:, :]
        elif self.coordinate_dim == 2:
            v1 = v[:, :awidth, :, :]
            v2 = v[:, awidth:, :, :]
        elif self.coordinate_dim == 3:
            v1 = v[:, :awidth, :, :, :]
            v2 = v[:, awidth:, :, :, :]

        # Inverse coupling layers (reverse order, dimension-adaptive)
        for i in range(self.n_layers - 1, -1, -1):
            # Dimension-adaptive padding for v1
            if self.coordinate_dim == 1:
                v1_pad = F.pad(v1, [0, self.padding])
            elif self.coordinate_dim == 2:
                v1_pad = F.pad(v1, [0, self.padding, 0, self.padding])
            elif self.coordinate_dim == 3:
                v1_pad = F.pad(v1, [0, self.padding, 0, self.padding, 0, self.padding])
                
            x1 = self.mlps[2 * i + 1](self.convs[2 * i + 1](v1_pad))
            x2 = self.ws[2 * i + 1](v1_pad)
            s1 = F.gelu(x1 + x2)
            
            # Dimension-adaptive cropping for s1
            if self.coordinate_dim == 1:
                s1 = s1[..., : (s1.size(-1) - self.padding)]
            elif self.coordinate_dim == 2:
                s1 = s1[..., : (s1.size(-2) - self.padding),
                        : (s1.size(-1) - self.padding)]
            elif self.coordinate_dim == 3:
                s1 = s1[..., : (s1.size(-3) - self.padding),
                        : (s1.size(-2) - self.padding),
                        : (s1.size(-1) - self.padding)]
                        
            u2 = v2 * self._softplus(s1) ** (-1)

            # Dimension-adaptive padding for u2
            if self.coordinate_dim == 1:
                u2_pad = F.pad(u2, [0, self.padding])
            elif self.coordinate_dim == 2:
                u2_pad = F.pad(u2, [0, self.padding, 0, self.padding])
            elif self.coordinate_dim == 3:
                u2_pad = F.pad(u2, [0, self.padding, 0, self.padding, 0, self.padding])
                
            x1 = self.mlps[2 * i](self.convs[2 * i](u2_pad))
            x2 = self.ws[2 * i](u2_pad)
            s2 = F.gelu(x1 + x2)
            
            # Dimension-adaptive cropping for s2
            if self.coordinate_dim == 1:
                s2 = s2[..., : (s2.size(-1) - self.padding)]
            elif self.coordinate_dim == 2:
                s2 = s2[..., : (s2.size(-2) - self.padding),
                        : (s2.size(-1) - self.padding)]
            elif self.coordinate_dim == 3:
                s2 = s2[..., : (s2.size(-3) - self.padding),
                        : (s2.size(-2) - self.padding),
                        : (s2.size(-1) - self.padding)]
                        
            u1 = v1 * self._softplus(s2) ** (-1)

            v1 = u1
            v2 = u2

        # Output processing for inverse
        x = torch.cat((v1, v2), axis=1)

        if self.is_symmetric:
            # Symmetric case - direct output
            x_preds = self.q2(x)
            
            # Dimension-adaptive permutation back
            if self.coordinate_dim == 1:
                x_preds = x_preds.permute(0, 2, 1)
            elif self.coordinate_dim == 2:
                x_preds = x_preds.permute(0, 2, 3, 1)
            elif self.coordinate_dim == 3:
                x_preds = x_preds.permute(0, 2, 3, 4, 1)
                
            return x_preds, reconstruction_loss
        else:
            # Asymmetric case - use q3 projection
            x = self.q2(x)
            
            if self.coordinate_dim == 1:
                x = x.permute(0, 2, 1)
                x_preds = self.q3(x)
                x_preds = x_preds.permute(0, 2, 1)
            elif self.coordinate_dim == 2:
                x = x.permute(0, 2, 1, 3)
                x_preds = self.q3(x)
                x_preds = x_preds.permute(0, 2, 3, 1)
            elif self.coordinate_dim == 3:
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
    # Required spatial parameters for iFNO
    input_spatial_dims=(64, 64),     # Input spatial grid shape
    output_spatial_dims=(64, 58),    # Output spatial grid shape
    input_function_channels=1,       # Function value channels in input
    output_function_channels=1,      # Function value channels in output
    # Spatial coordinate dimension (1D, 2D, 3D)
    coordinate_dim=2,
    # Optional parameters
    intermediate_dim=64,
    vae_hidden_dims=None,
):
    """
    Create an IFNO model with explicit spatial configuration.

    Args:
        input_size: Input size for compatibility (not used directly by iFNO)
        hidden_sizes: Hidden sizes for compatibility (not used directly)
        n_coupling_layers: Number of coupling layers for compatibility (not used directly)
        modes1: Number of modes in first dimension
        modes2: Number of modes in second dimension
        width: Hidden dimension width
        beta: Beta parameter for softplus activation
        n_layers: Number of coupling layers
        padding: Padding for convolutions
        vae_latent_dim: Latent dimension for VAE
        input_spatial_dims: Input spatial grid shape (e.g., (25, 25), (5, 1000, 70))
        output_spatial_dims: Output spatial grid shape (e.g., (25, 25), (70, 70))
        input_function_channels: Function value channels in input
        output_function_channels: Function value channels in output
        coordinate_dim: Spatial coordinate dimension (1D, 2D, 3D)
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
        input_spatial_dims=input_spatial_dims,
        output_spatial_dims=output_spatial_dims,
        input_function_channels=input_function_channels,
        output_function_channels=output_function_channels,
        coordinate_dim=coordinate_dim,
        intermediate_dim=intermediate_dim,
        vae_hidden_dims=vae_hidden_dims,
    )


# All dataset-specific configuration is now handled via dataset get_info() functions


# Loss functions for different training phases - work directly with batch
def ifno_vae_loss(model, batch, kl_weight=0.01):
    """VAE pretraining loss using the VAE component (dimension-adaptive)"""
    X, u, Y, s = batch
    batch_size = u.shape[0]

    # Extract function values (take first channel if multi-channel)
    if len(u.shape) == 4 and u.shape[-1] > 1:
        u_for_vae = u[:, :, :, 0:1]  # Take function values, ignore coordinates
    else:
        u_for_vae = u

    # Use dimension-adaptive reshaping for VAE input
    u_reshaped = model._get_vae_input_shape(u_for_vae)

    # VAE forward pass
    vae_out, vae_input, mu, log_var = model.vae_net(u_reshaped)

    # Reconstruction loss (dimension-adaptive)
    reconstruction_loss = relative_l2_loss(vae_out.reshape(batch_size, -1),
                                           vae_input.reshape(batch_size, -1))

    # KL divergence loss
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var -
                         mu**2 - log_var.exp(), dim=1), dim=0)

    return reconstruction_loss + kl_weight * kl_loss


def ifno_forward_loss(model, batch):
    """Forward pass loss for IFNO pretraining"""
    X, u, Y, s = batch
    # Forward pass: input u -> model -> pred_s, compare with s
    # Reshape u to match model input format
    # Combine spatial coordinates with function values
    u_input = torch.cat([X, u], dim=-1)

    result = model(u_input)
    # Handle both symmetric (returns tuple) and asymmetric (returns tensor) cases
    if isinstance(result, tuple):
        pred_s, reconstruction_loss = result
    else:
        pred_s = result
        reconstruction_loss = 0.0
    
    # Extract function values only (last channel) to match target s
    if pred_s.shape[-1] > s.shape[-1]:
        pred_s_func = pred_s[..., -s.shape[-1]:]  # Take last channels (function values)
    else:
        pred_s_func = pred_s
        
    return relative_l2_loss(pred_s_func, s) + reconstruction_loss


def ifno_backward_loss(model, batch):
    """Backward pass loss for IFNO pretraining"""
    X, u, Y, s = batch
    # Backward pass: input s -> model.inverse -> pred_u, compare with u
    # Reshape s to match model input format
    # Combine spatial coordinates with function values
    s_input = torch.cat([Y, s], dim=-1)

    result = model.inverse(s_input)
    # Handle both symmetric (returns tuple) and asymmetric (returns tensor) cases
    if isinstance(result, tuple):
        pred_u, reconstruction_loss = result
    else:
        pred_u = result
        reconstruction_loss = 0.0
    
    # Extract function values only (last channel) to match target u
    if pred_u.shape[-1] > u.shape[-1]:
        pred_u_func = pred_u[..., -u.shape[-1]:]  # Take last channels (function values)
    else:
        pred_u_func = pred_u
        
    return relative_l2_loss(pred_u_func, u) + reconstruction_loss


def ifno_joint_loss(model, batch, grid_loss_weight=0.01):
    """Joint training loss - adaptive for symmetric vs asymmetric problems"""
    X, u, Y, s = batch
    batch_size = u.shape[0]

    # Forward loss: input u -> model -> pred_s, compare with s
    u_input = torch.cat([X, u], dim=-1)
    if model.is_symmetric:
        pred_s, forward_reconstruction_loss = model(u_input)
        forward_loss = relative_l2_loss(
            pred_s, s) + forward_reconstruction_loss
    else:
        pred_s = model(u_input)
        forward_loss = relative_l2_loss(pred_s, s)

    # Backward loss with VAE integration (like provided implementation)
    s_input = torch.cat([Y, s], dim=-1)
    if model.is_symmetric:
        pred_u, backward_reconstruction_loss = model.inverse(s_input)
        backward_loss = relative_l2_loss(
            pred_u, u) + backward_reconstruction_loss
    else:
        pred_u = model.inverse(s_input)

        # VAE reconstruction loss on predicted u (for asymmetric problems)
        # Reshape for VAE input format (batch, channels, height, width)
        pred_u_reshaped = pred_u[:, :, :, 0:1].permute(
            0, 3, 1, 2)  # Take first channel
        u_reshaped = u[:, :, :, 0:1].permute(0, 3, 1, 2)

        # VAE forward pass for reconstruction
        vae_out, _, mu, log_var = model.vae_net(pred_u_reshaped)

        # VAE reconstruction loss
        vae_reconstruction_loss = relative_l2_loss(vae_out, u_reshaped)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss.mean()

        # Grid coordinate loss (separate loss for spatial coordinates)
        if pred_u.shape[-1] > 1:  # If we have coordinate channels
            grid_loss = relative_l2_loss(pred_u[:, :, :, 1:], u[:, :, :, 1:])
            grid_loss = grid_loss * grid_loss_weight
        else:
            grid_loss = 0.0

        # Combined backward loss (like provided implementation)
        backward_loss = vae_reconstruction_loss + 0.01 * kl_loss + grid_loss

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
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{model_name}_checkpoint.pt")
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
            vae_optimizer, factor=0.9, patience=10
        )

        for epoch in range(epochs_vae):
            model.train()
            train_loss = 0.0

            for batch in train_dataloader:
                vae_optimizer.zero_grad()
                loss = ifno_vae_loss(model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0)
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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                ifno_optimizer.step()
                train_forward_loss += forward_loss.item()

                # Backward pass training
                ifno_optimizer.zero_grad()
                backward_loss = ifno_backward_loss(model, batch)
                backward_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                ifno_optimizer.step()
                train_backward_loss += backward_loss.item()

            ifno_scheduler.step()
            avg_forward_loss = train_forward_loss / len(train_dataloader)
            avg_backward_loss = train_backward_loss / len(train_dataloader)
            print(
                f"IFNO Epoch {epoch+1}/{epochs_ifno}, Forward: {avg_forward_loss:.6f}, Backward: {avg_backward_loss:.6f}"
            )

    # Phase 3: Joint Training (Enhanced like provided implementation)
    print("Phase 3: Joint Training")
    forward_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_forward)
    backward_optimizer = torch.optim.AdamW(model.parameters(), lr=lr_forward)

    min_err_forward = 1.0
    min_err_backward = 1.0

    tqdm_bar = tqdm.tqdm(range(start_epoch, n_epochs))
    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_forward_loss = 0.0
        train_backward_loss = 0.0

        for batch in train_dataloader:
            X, u, Y, s = batch
            batch_size = u.shape[0]

            # Forward pass training
            forward_optimizer.zero_grad()
            u_input = torch.cat([X, u], dim=-1)
            result = model(u_input)
            
            # Handle tuple return for symmetric models
            if isinstance(result, tuple):
                pred_s, reconstruction_loss = result
                # Extract function values only to match target s
                if pred_s.shape[-1] > s.shape[-1]:
                    pred_s_func = pred_s[..., -s.shape[-1]:]
                else:
                    pred_s_func = pred_s
                forward_loss = relative_l2_loss(pred_s_func, s) + reconstruction_loss
            else:
                pred_s = result
                # Extract function values only to match target s
                if pred_s.shape[-1] > s.shape[-1]:
                    pred_s_func = pred_s[..., -s.shape[-1]:]
                else:
                    pred_s_func = pred_s
                forward_loss = relative_l2_loss(pred_s_func, s)
            forward_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            forward_optimizer.step()
            train_forward_loss += forward_loss.item()

            # Backward pass training with VAE integration (like provided implementation)
            backward_optimizer.zero_grad()
            s_input = torch.cat([Y, s], dim=-1)
            result = model.inverse(s_input)
            
            # Handle tuple return for symmetric models
            if isinstance(result, tuple):
                pred_u, _ = result
            else:
                pred_u = result

            # VAE training loss computation (dimension-adaptive)
            # Extract function values and reshape for VAE
            if pred_u.shape[-1] > u.shape[-1]:
                # pred_u contains both coordinates and function values, extract only function values
                pred_u_values = pred_u[..., -u.shape[-1]:]  # Take last channels (function values)
            else:
                pred_u_values = pred_u
                
            # u should already contain only function values
            u_values = u
                
            # Use dimension-adaptive VAE input reshaping
            pred_u_for_vae = model._get_vae_input_shape(pred_u_values)
            u_for_vae = model._get_vae_input_shape(u_values)

            # VAE forward pass
            recon_img, _, mu, log_var = model.vae_net(pred_u_for_vae)

            # KL loss (like provided implementation)
            kl_loss = torch.mean(-0.5 * torch.sum(1 +
                                 log_var - mu**2 - log_var.exp(), dim=1), dim=0)

            # Reconstruction loss (MSE-based like provided implementation)
            vae_reconstruction_loss = relative_l2_loss(
                recon_img.reshape(batch_size, -1),
                u_for_vae.reshape(batch_size, -1)
            )

            # Grid coordinate loss (dimension-adaptive)
            if pred_u.shape[-1] > u.shape[-1]:
                # Only compute grid loss if pred_u has more channels (i.e., includes coordinates)
                if len(pred_u.shape) == 3:  # 1D case
                    grid_loss = relative_l2_loss(
                        pred_u[:, :, :-u.shape[-1]], X) / (100 * batch_size**2)
                elif len(pred_u.shape) == 4:  # 2D case  
                    grid_loss = relative_l2_loss(
                        pred_u[:, :, :, :-u.shape[-1]], torch.cat([X, Y], dim=-1)) / (100 * batch_size**2)
                else:
                    grid_loss = 0.0
            else:
                grid_loss = 0.0

            # Combined backward loss (like provided implementation)
            backward_loss = vae_reconstruction_loss + 0.01 * kl_loss + grid_loss
            backward_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            backward_optimizer.step()
            train_backward_loss += backward_loss.item()

        avg_forward_loss = train_forward_loss / len(train_dataloader)
        avg_backward_loss = train_backward_loss / len(train_dataloader)
        avg_epoch_loss = avg_forward_loss + avg_backward_loss

        # Enhanced test evaluation (like provided implementation)
        model.eval()
        test_forward_loss = 0.0
        test_backward_loss = 0.0

        with torch.no_grad():
            for batch in test_dataloader:
                X, u, Y, s = batch
                batch_size = u.shape[0]

                # Test forward pass
                u_input = torch.cat([X, u], dim=-1)
                result = model(u_input)
                
                # Handle tuple return for symmetric models  
                if isinstance(result, tuple):
                    pred_s, _ = result
                else:
                    pred_s = result
                
                # Extract function values only to match target s
                if pred_s.shape[-1] > s.shape[-1]:
                    pred_s_func = pred_s[..., -s.shape[-1]:]
                else:
                    pred_s_func = pred_s
                    
                test_forward_loss += relative_l2_loss(pred_s_func, s).item()

                # Test backward pass with VAE
                s_input = torch.cat([Y, s], dim=-1)
                result = model.inverse(s_input)
                
                # Handle tuple return for symmetric models  
                if isinstance(result, tuple):
                    pred_u, _ = result
                else:
                    pred_u = result

                # VAE reconstruction on predicted input (dimension-adaptive)
                if pred_u.shape[-1] > u.shape[-1]:
                    # pred_u contains both coordinates and function values, extract only function values
                    pred_u_values = pred_u[..., -u.shape[-1]:]  # Take last channels (function values)
                else:
                    pred_u_values = pred_u
                    
                # u should already contain only function values
                u_values = u
                    
                pred_u_for_vae = model._get_vae_input_shape(pred_u_values)
                u_for_vae = model._get_vae_input_shape(u_values)
                vae_out, _, _, _ = model.vae_net.forward2(
                    pred_u_for_vae)  # Use deterministic forward

                test_backward_loss += relative_l2_loss(
                    vae_out, u_for_vae).item()

        avg_test_forward = test_forward_loss / len(test_dataloader)
        avg_test_backward = test_backward_loss / len(test_dataloader)
        avg_test_loss = avg_test_forward + avg_test_backward

        # Track minimum errors
        if avg_test_forward < min_err_forward:
            min_err_forward = avg_test_forward
        if avg_test_backward < min_err_backward:
            min_err_backward = avg_test_backward

        # Enhanced logging
        summary_writer.add_scalars(
            "loss/train_forward", {model_name: avg_forward_loss}, epoch)
        summary_writer.add_scalars(
            "loss/train_backward", {model_name: avg_backward_loss}, epoch)
        summary_writer.add_scalars(
            "loss/test_forward", {model_name: avg_test_forward}, epoch)
        summary_writer.add_scalars(
            "loss/test_backward", {model_name: avg_test_backward}, epoch)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model,
                forward_optimizer,  # Save forward optimizer
                epoch + 1,
                avg_test_loss,
                checkpoint_path,
                "joint",
            )

        tqdm_bar.set_postfix_str(
            f"train_fwd {avg_forward_loss:.4e} train_bwd {avg_backward_loss:.4e} "
            f"test_fwd {avg_test_forward:.4e} test_bwd {avg_test_backward:.4e} "
            f"min_fwd {min_err_forward:.4e} min_bwd {min_err_backward:.4e}"
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
