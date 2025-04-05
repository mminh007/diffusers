import torch
import torch.nn as nn
from ultis import Encoder, EncoderV2, Decoder, DecoderV2, CNNBlock, DownBlock, MidBlock, UpBlock


class ResNetVAE(nn.Module):
    """
    """
    def __init__(self, in_chans, out_chans, latent_dim, activation="swish"):
        super().__init__()
        self.encode = Encoder(in_chans, latent_dim, activation)
        self.decode = Decoder(latent_dim, out_chans, activation)


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        
        return mu + eps * std
    

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
    

class ResNetVAEV2(nn.Module):
    """
    Args:
        num_chans (int): Number of input channels.
        in_chans (int): Number of initial convolution channels.
        z_dim (int): is the number of channels in the embedding space.
        embed_channels (int): is the number of dimensions in the quantized embedding space.
        channel_multipliers (list): Multiplicative factors for the number of channels.
        blocks (int): Number of residual blocks per layer.       
    """

    def __init__(self, in_chans=64,
                 num_chans=3,
                 out_chans=3,
                 z_dim=4,
                 embed_dim=4,
                 activation="swish",
                 blocks=2,
                 channel_multipliers=[1,2,4,4]):
        super().__init__()
        self.encode = EncoderV2(num_chans, in_chans, z_dim, activation, channel_multipliers, blocks)
        self.decode = DecoderV2(in_chans, embed_dim, activation, channel_multipliers, blocks, out_chans)

        self.quant_conv = nn.Conv2d(z_dim * 2, embed_dim * 2, kernel_size=1, stride=1, padding=0)

        self.post_quant_conv = nn.Conv2d(embed_dim, z_dim, kernel_size=1, padding=0)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        
        return mu + eps * std
    

    def forward(self, x):
        # encode
        x = self.encode(x)

        moments = self.quant_conv(x)
        mu, logvar = torch.chunk(moments, 2, dim=1)

        # parameter trick
        z = self._reparameterize(mu, logvar)

        # decode
        z = self.post_quant_conv(z)
        img = self.decode(z)

        return img, mu, logvar


class Unet(nn.Module):
    """
    U-Net architecture with attention, time conditioning, and residual connections.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        hidden_dim (int): Base number of channels for the U-Net.
        time_dim (int): Dimension of the positional/time embedding.
        activation (str): Activation function to use in CNN blocks.
        dropout (float): Dropout rate in CNN blocks.
        n_heads (int): Number of heads in multi-head attention.
        qkv_bias (bool): Whether to add bias to QKV linear layers.
        qk_scale (float): Optional scale for QK attention.
        is_attn (List[bool]): List of booleans indicating attention at each down/up level.
        down_scale (int): Number of downsampling (and upsampling) steps.
        residual (bool): Whether to include residual connections.

    Inputs:
        x (Tensor): Input image tensor of shape (B, in_chans, H, W)
        t (Tensor): Time tensor of shape (B,), used for time-based conditioning

    Outputs:
        output (Tensor): Output image tensor of shape (B, out_chans, H, W)
    """
    def __init__(self,
                 in_chans=3,
                 out_chans=3,
                 hidden_dim=64,
                 time_dim=256,
                 activation="swish",
                 dropout=0.3,
                 n_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 is_attn=[True, True, True, True],
                 down_scale=3,
                 residual=True):
        super().__init__()
        
        self.dim = hidden_dim
        self.time_dim = time_dim
        self.conv1 = CNNBlock(in_chans, hidden_dim, activation=activation, kernel_size=3, padding=1, stride=1)

        self.downscales = nn.ModuleList()
        
        for i in range(down_scale):
            self.downscales.append(DownBlock(in_chans=self.dim, out_chans=self.dim * 2, time_dim=time_dim,
                                             n_heads=n_heads, activation=activation, qk_scale=qk_scale,
                                             qkv_bias=qkv_bias, has_attn=is_attn[i], dropout=dropout, residual=True))
            self.dim = self.dim * 2

        self.middle_1 = MidBlock(in_chans=self.dim, out_chans=self.dim * 2, time_dim=256,
                               activation="swish", dropout=dropout, n_heads=n_heads, qk_scale=qkv_bias,
                               qkv_bias=qkv_bias, has_attn=is_attn[-1], residual=residual)  
        
        self.middle_2 = UpBlock(in_chans=self.dim * 2, out_chans= self.dim, time_dim=256,
                                        n_heads=n_heads, activation=activation, qk_scale=qk_scale,
                                        qkv_bias=qkv_bias, has_attn=is_attn[-1], dropout=dropout, residual=True)
        self.dim = self.dim // 2
        
        self.upscale = nn.ModuleList()

        for i in reversed(range(down_scale)):
            self.upscale.append(UpBlock(in_chans=self.dim * 2, out_chans= self.dim, time_dim=256,
                                        n_heads=n_heads, activation=activation, qk_scale=qk_scale,
                                        qkv_bias=qkv_bias, has_attn=is_attn[i], dropout=dropout, residual=True))
            self.dim = self.dim // 2
        

        self.final = nn.Conv2d(self.dim, out_chans, kernel_size=1)

    def _pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self._pos_encoding(t, self.time_dim)

        x = self.conv1(x)

        h = [x]

        for layer in self.downscales:
            x = layer(x, t)
            h.append(x)
        
        x = self.middle_1(x, t)

        x = self.middle_2(x, h.pop(), t)

        for layer in self.upscale:
            x = layer(x, h.pop(), t)
        output = self.final(x)

        return output
