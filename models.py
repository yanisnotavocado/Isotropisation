import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, VarAutoEncoder
from monai.networks.nets import Discriminator as Dics
from typing import Sequence, Union, Tuple
UNet = UNet

class VAE(nn.Module):
    def __init__(
        self,
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int, 
        latent_channels: int, 
        channels: Sequence[int], 
        strides: Sequence[int], 
        num_res_units: int = 0, 
        act: Union[tuple, str, None] = "PReLU", 
        norm: Union[tuple, str] = "Instance", 
        bias: bool = True,
    ):
        super(VAE, self).__init__()
        vae = VarAutoEncoder(
            spatial_dims = spatial_dims,
            in_shape = [in_channels] + [1] * spatial_dims,
            out_channels = out_channels,
            latent_size = latent_channels,
            channels = channels,
            strides = strides,
            num_res_units = num_res_units,
            num_inter_units = 0,
            act = act,
            norm = norm,
            bias = bias,
            use_sigmoid = False,
        )
        
        self.encoder = vae.encode
        self.decoder = vae.decode
        self.mu = getattr(nn, f"Conv{spatial_dims}d")(channels[-1], latent_channels, 1)
        self.logvar = getattr(nn, f"Conv{spatial_dims}d")(channels[-1], latent_channels, 1)
        self.decodeL = getattr(nn, f"Conv{spatial_dims}d")(latent_channels, channels[-1], 1)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.decodeL(z))
        x = self.decoder(x)
        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        if self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)
        return std.add_(mu)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
        

class Discriminator(nn.Module):
    def __init__(
        self,
        spatial_dims: int, 
        in_channels: int,
        out_channels: int,
        channels: Sequence[int], 
        strides: Sequence[int],
        num_res_units: int = 0, 
        act: Union[tuple, str, None] = "PReLU", 
        norm: Union[tuple, str] = "Instance", 
        bias: bool = True,
    ) :
        super(Discriminator, self).__init__()
        vae = VarAutoEncoder(
            spatial_dims = spatial_dims,
            in_shape = [in_channels] + [1] * spatial_dims,
            out_channels = 1,
            latent_size = 1,
            channels = channels,
            strides = strides,
            num_res_units = num_res_units,
            num_inter_units = 0,
            act = act,
            norm = norm,
            bias = bias,
            use_sigmoid = False,
        )
        self.net = vae.encode
        self.features = dict()
        def forward_hook(mod, inputs, outputs):
            self.features[mod] = outputs
        def add_hooks(mod):
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                mod.register_forward_hook(forward_hook)
        self.net.apply(add_hooks)
        self.last = getattr(nn, f"Conv{spatial_dims}d")(channels[-1], out_channels, 1)
        
    def forward(self, x):
        out = self.net(x)
        out = self.last(out)
        return out, list(self.features.values())