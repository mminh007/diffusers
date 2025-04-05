import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

def get_act(act_type = "lrelu"):
    """
    Returns the activation function based on the given activation type.
    
    Args:
        act_type (str): Type of activation function. Options are "relu", "lrelu", or "silu".
                        Defaults to "lrelu".
    
    Returns:
        nn.Module: The corresponding activation function.
    """

    if act_type == "relu":
        return nn.ReLU()
    
    elif act_type == "lrelu":
        return nn.LeakyReLU(0.2)

    elif act_type == "silu":
        return nn.SiLU()
    
    elif act_type == "swish":
        return Swish() 
    
    else:
        return nn.LeakyReLU(0.2)
    

class CNNBlock(nn.Module):
    """
    A convolutional block with the following structure:
    Conv2D -> BatchNorm -> Activation -> (Optional) Dropout.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        activation (str): Type of activation function (e.g., "relu", "lrelu", "silu").
        dropout (float or None): Dropout rate. If None, dropout is not applied.
        **kwargs: Additional arguments for the convolutional layer.
    """
    
    def __init__(self, in_chans, out_chans, activation="lrelu", dropout=0.3, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, **kwargs)
        self.norm = nn.BatchNorm2d(num_features=out_chans)
        self.act = get_act(act_type=activation)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
    
    def forward(self, x):
        """
        Forward pass of CNNBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_chans, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_chans, height, width).
        """
        x = self.conv(x)
        x = self.norm(x)

        out = self.act(x)
        if self.dropout is not None:
            return self.dropout(out)
        
        return out


class ResBlock(nn.Module):
    """
    Residual Block with expansion.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        stride (int): Stride for convolution.
        padding (int): Padding for convolution.
        downsample (nn.Module or None): Downsampling layer for the skip connection.
        dropout (float): Dropout rate.
    """

    expansion = 4
    def __init__(self, in_chans, out_chans, stride=1, padding=1, activation="swish", downsample=None, dropout=0.3):
        super().__init__()
        self.conv1 = CNNBlock(in_chans, out_chans, kernel_size=1, stride=1, padding=0, activation=activation, dropout=0.3)
        self.conv2 = CNNBlock(out_chans, out_chans,  kernel_size=3, stride=stride, padding=padding, activation=activation, dropout=0.3)
        self.conv3 = nn.Conv2d(out_chans, out_chans*self.expansion, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(num_features=out_chans*self.expansion)
        self.act3 = get_act(get_act=activation)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.i_downsample = downsample

    def forward(self, x):
        """
        Forward pass of ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        identity = x.clone()

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + identity
        out = self.act3(x)

        if self.dropout is not None:
            return self.dropout(out)
        
        return out
    

class Encoder(nn.Module):
    """
    """

    def __init__(self, in_chans, latent_dim=128, activation="swish"):
        super().__init__()
        self.dim = 32
        self.conv1 = CNNBlock(in_chans, 32, kernel_size=3, activation=activation, stride=2, padding=1)
        self.layer = self._make_layer(ResBlock, planes=32, stride=2)

        self.flatten_dim = 128*8*8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)


    def _make_layer(self, ResBlock, planes, stride=1):
        ii_downsample = None

        if stride != 1 or self.dim != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.dim, planes*ResBlock.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(num_features=planes*ResBlock.expansion)
            )

        return ResBlock(self.dim, planes, downsample=ii_downsample, stride=stride)
    

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.layer(x)

        x = x.view(-1, self.flatten_dim) # (-1, 128*8*8)
        mu = self.fc_mu(x) # (-1, 128)
        logvar = self.fc_logvar(x) # (-1, 128)

        return mu, logvar
    

class EncoderV2(nn.Module):
    """
    Encoder module for feature extraction.
    
    Args:
        num_chans (int): Number of input channels.
        in_chans (int): Number of initial convolution channels.
        z_dim (int): Dimension of the latent space.
        channel_multipliers (list): Multiplicative factors for the number of channels.
        blocks (int): Number of residual blocks per layer.
    """

    def __init__(self, num_chans, in_chans=64, z_dim=4, activation="swish", channel_multipliers=[1,2,4,4], blocks=2):
        super().__init__()
        self.dim = in_chans   
        self.conv1 = CNNBlock(num_chans, in_chans, kernel_size=3, activation=activation, stride=1, padding=1)


        self.layer = nn.ModuleList()
        for factor in channel_multipliers:
            self.layer.append(self._make_layer(ResBlock, planes= factor * in_chans, stride=2, block=blocks))

        self.conv2 = nn.Conv2d(self.dim, z_dim * 2, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, ResBlock, planes, block, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.dim != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.dim, planes*ResBlock.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(num_features=planes*ResBlock.expansion)
            )
        layers.append(ResBlock(self.dim, planes, downsample=ii_downsample, stride=stride))
        self.dim = planes * ResBlock.expansion

        for i in range(block - 1):
          layers.append(ResBlock(self.dim, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layer:
            x = layer(x)

        output = self.conv2(x)
        return output
    

class Decoder(nn.Module):
    """
    """
    def __init__(self, latent_dim=128, out_chans=3, activation="swish"):
        super().__init__()
        self.fc_decode = nn.Linear(latent_dim, 128*8*8)
        self.conv1 = CNNBlock(latent_dim, latent_dim // 2, kernel_size=3, stride=1, padding=1, activation=activation)
        self.conv2 = CNNBlock(latent_dim // 2, latent_dim, kernel_size=3, stride=1, padding=1, activation=activation)

        self.deconv1 = nn.ConvTranspose2d(latent_dim, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc_decode(x).view(-1, 128, 8, 8)

        indentity = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + indentity

        x = self.deconv1(x)
        x = self.deconv2(x)
        out = self.sigmoid(x)

        return out
    

    def forward(self, x):       
        x = self.conv1(x)
        for layer in self.layer:
            x = layer(x)

        output = self.conv2(x)
        return output


class DecoderV2(nn.Module):
    """
    """
    def __init__(self, in_chans=64, embed_dim=4, activation="swish", channel_multipliers=[1,2,4,4], blocks=2, out_channels=3):
        super().__init__()
        self.dim = in_chans
        self.conv1 = CNNBlock(embed_dim, in_chans, kernel_size=1, activation=activation)  # (4, H, W) -> (265, H, W)

        self.layers = nn.ModuleList()
        for factor in reversed(channel_multipliers):
            self.layers.append(self._make_layer(ResBlock, planes= factor * in_chans, block=blocks)) # 256 > 

        self.conv2 = CNNBlock(self.dim, out_channels, kernel_size=3, activation=activation, stride=1, padding=1)
        

    def _make_layer(self, ResBlock, planes, block, stride=1):
        layers = []
        ii_upsample = None
        if self.dim != planes * ResBlock.expansion:
            ii_upsample = nn.Sequential(
                nn.Conv2d(self.dim, planes*ResBlock.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(num_features=planes*ResBlock.expansion))
            
        layers.append(ResBlock(self.dim, planes, stride=stride, padding=1, downsample=ii_upsample))
        self.dim = planes * ResBlock.expansion
          
        for i in range(block - 1):
          layers.append(ResBlock(self.dim, planes, stride=1, padding=1))
              
        layers.append(nn.Conv2d(self.dim, self.dim * 4, kernel_size=3, padding=1))
        layers.append(nn.PixelShuffle(2))
              
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        for layer in self.layers:
            x = layer(x)

        out = self.conv2(x)
        return out


class AttnBlock(nn.Module):
    """
    Multi-head Self-Attention Block for 2D feature maps.

    Args:
        in_chans (int): Number of input channels.
        n_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to add a learnable bias to Q, K, V projections.
        qk_scale (float): Optional scaling factor for QK^T. Defaults to sqrt(d_k).

    Shape:
        Input: (B, C, H, W)
        Output: (B, C, H, W)
    """

    def __init__(self, in_chans, n_heads, qkv_bias, qk_scale):
        super().__init__()
        self.n_heads = n_heads
        
        head_dim = in_chans // n_heads
        self.qk_scale = head_dim ** 0.5 if qk_scale == None else qk_scale

        self.qkv = nn.Linear(in_chans, in_chans * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_chans, in_chans)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x):
        B, C, H, W = x.shape

        x = x.view(B, C, H*W).permute(0,2,1)
        qkv = self.qkv(x)

        q,k,v = tuple(rearrange(qkv, "b l (d f k) -> k (b f) l d", k=3, f=self.n_heads))

        qk_dot_product = torch.einsum("b i d, b j d-> b i j", q, k) * self.qk_scale

        attn = qk_dot_product.softmax(dim=-1)

        x = (attn @ v).view(B, self.n_heads, H, W, -1).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)

        x = self.proj(x)

        return x


class Down(nn.Module):
    """
    Downsampling block with optional residual connection and time embedding.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        time_dim (int): Dimensionality of time embedding.
        activation (str): Activation function.
        dropout (float): Dropout rate.
        residual (bool): Whether to apply residual shortcut.

    Shape:
        Input: (B, C, H, W), (B, time_dim)
        Output: (B, out_chans, H/2, W/2)
    """
    def __init__(self, in_chans=64,
                 out_chans=64,
                 time_dim=256,
                 activation="swish",
                 dropout=0.3,
                 residual=False):
        
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(2),
            CNNBlock(in_chans,
                     out_chans,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     activation=activation, dropout=dropout))
        
        self.conv2 = CNNBlock(out_chans,
                     out_chans,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     activation=activation, dropout=dropout)

        self.time_emb = nn.Sequential(
            get_act(act_type=activation),
            nn.Linear(time_dim, out_chans)
        )

        if residual:
            self.shortcut = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        h = self.conv1(x)
        
        h = F.gelu(h + self.shortcut(x))

        h = self.conv2(h)

        emb = self.time_emb(t)[:, :, None, None].repeat(1,1, h.shape[-2], h.shape[-1])

        return h + emb
    
    
class DownBlock(nn.Module):
    """
    Combines Down block and optional attention.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation function.
        dropout (float): Dropout rate.
        n_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in QKV projection.
        qk_scale (float): Scaling for QK attention.
        has_attn (bool): Whether to apply attention.
        residual (bool): Whether to apply residual shortcut.
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 time_dim,
                 activation,
                 dropout,
                 n_heads,
                 qkv_bias,
                 qk_scale,
                 has_attn: bool,
                 residual):
        super().__init__()

        self.down = Down(in_chans=in_chans, out_chans=out_chans, time_dim=time_dim,
                         activation=activation, dropout=dropout, residual=residual)

        if has_attn:
            self.attn = AttnBlock(in_chans=out_chans, n_heads=n_heads,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale)
        else:
            self.attn = nn.Identity()
        
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.down(x, t)
        x = self.attn(x)
        return x


class MidBlock(nn.Module):
    """
    Middle block of U-Net containing down path and optional attention.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation function.
        dropout (float): Dropout rate.
        n_heads (int): Number of attention heads.
        qkv_bias (bool): Use bias in QKV.
        qk_scale (float): Scale factor for attention.
        has_attn (bool): Whether to apply attention.
        residual (bool): Apply residual connection or not.
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 time_dim,
                 activation,
                 dropout,
                 n_heads,
                 qkv_bias,
                 qk_scale,
                 has_attn: bool,
                 residual):
        
        super().__init__()

        self.down = Down(in_chans, out_chans, time_dim, activation, dropout, residual)

        if has_attn:
            self.attn = AttnBlock(in_chans, n_heads, qkv_bias, qk_scale)
        else:
            self.attn = nn.Identity()
        
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.down(x, t)
        x = self.attn(x)
        return x


class Up(nn.Module):
    """
    Upsampling block with time embedding and optional residual shortcut.

    Args:
        in_chans (int): Input channels.
        out_chans (int): Output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation function.
        dropout (float): Dropout rate.
        residual (bool): Use residual shortcut.
    
    Shape:
        Inputs:
            x (B, in_chans, H/2, W/2)
            x_skip (B, out_chans, H, W)
            t (B, time_dim)
        Output: (B, out_chans, H, W)
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 time_dim,
                 activation="swish",
                 dropout=0.3,
                 residual=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1)

        self.conv1 = CNNBlock(in_chans=in_chans, out_chans=out_chans, kernel_size=3, stride=1, padding=1,
                     activation=activation, dropout=dropout)
        self. conv2 = CNNBlock(in_chans=out_chans, out_chans=out_chans, kernel_size=3, stride=1, padding=1,
                              activation=activation, dropout=dropout)
        
        self.time_emb = nn.Sequential(
            get_act(act_type=activation),
            nn.Linear(time_dim, out_chans)
        )

        if residual:
            self.shortcut = nn.Conv2d(out_chans, out_chans, kernel_size=1, padding=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, x_skip, t):
        x = self.up(x)
        h = torch.cat([x, x_skip], dim=1)

        h = self.conv1(h)

        h = F.gelu(h + self.shortcut(x))

        h = self.conv2(h)

        embed = self.time_emb(t)[:,:, None, None].repeat(1,1, h.shape[-2], h.shape[-1])

        return h + embed


class UpBlock(nn.Module):
    """
    Combines Up block and optional attention.

    Args:
        in_chans (int): Input channels.
        out_chans (int): Output channels.
        time_dim (int): Time embedding dimension.
        activation (str): Activation function.
        dropout (float): Dropout rate.
        n_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use QKV bias.
        qk_scale (float): Attention scale factor.
        has_attn (bool): Whether to use attention.
        residual (bool): Whether to use residual shortcut.
    """
    
    def __init__(self, in_chans,
                 out_chans,
                 time_dim,
                 activation,
                 dropout,
                 n_heads,
                 qkv_bias,
                 qk_scale,
                 has_attn: bool,
                 residual):
        super().__init__()

        self.up = Up(in_chans=in_chans, out_chans=out_chans, time_dim=time_dim, activation=activation,
                     dropout=dropout, residual=residual)
        
        if has_attn:
            self.attn = AttnBlock(in_chans=in_chans, n_heads=n_heads,
                                  qk_scale=qk_scale, qkv_bias=qkv_bias)
        
        else:
            self.attn = nn.Identity()

    def forward(self, x, x_skip, t):
        x = self.up(x, x_skip, t)
        x = self.attn(x)
        return x


    
