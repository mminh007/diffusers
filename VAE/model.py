import torch
import torch.nn as nn


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
    def __init__(self, in_chans, out_chans, stride=1, padding=1, downsample=None, dropout=0.3):
        super().__init__()
        self.conv1 = CNNBlock(in_chans, out_chans, kernel_size=1, stride=1, padding=0, activation="lrelu", dropout=0.3)
        self.conv2 = CNNBlock(out_chans, out_chans,  kernel_size=3, stride=stride, padding=padding, activation="lrelu", dropout=0.3)
        self.conv3 = nn.Conv2d(out_chans, out_chans*self.expansion, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(num_features=out_chans*self.expansion)
        self.act3 = nn.LeakyReLU()
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

    def __init__(self, in_chans, latent_dim=128):
        super().__init__()
        self.dim = 32
        self.conv1 = CNNBlock(in_chans, 32, kernel_size=3, activation='lrelu', stride=2, padding=1)
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

    def __init__(self, num_chans, in_chans=64, z_dim=4, channel_multipliers=[1,2,4,4], blocks=2):
        super().__init__()
        self.dim = in_chans   
        self.conv1 = CNNBlock(num_chans, in_chans, kernel_size=3, activation='lrelu', stride=1, padding=1)


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
    def __init__(self, latent_dim=128, out_chans=3):
        super().__init__()
        self.fc_decode = nn.Linear(latent_dim, 128*8*8)
        self.conv1 = CNNBlock(latent_dim, latent_dim // 2, kernel_size=3, stride=1, padding=1, activation="lrelu")
        self.conv2 = CNNBlock(latent_dim // 2, latent_dim, kernel_size=3, stride=1, padding=1, activation="lrelu")

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
    def __init__(self, in_chans=64, embed_dim=4, channel_multipliers=[1,2,4,4], blocks=2, out_channels=3):
        super().__init__()
        self.dim = in_chans
        self.conv1 = CNNBlock(embed_dim, in_chans, kernel_size=1, activation='lrelu')  # (4, H, W) -> (265, H, W)

        self.layers = nn.ModuleList()
        for factor in reversed(channel_multipliers):
            self.layers.append(self._make_layer(ResBlock, planes= in_chans, block=blocks)) # 256 > 

        self.conv2 = CNNBlock(self.dim, out_channels, kernel_size=3, activation='lrelu', stride=1, padding=1)
        

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

class ResNetVAE(nn.Module):
    """
    """
    def __init__(self, in_chans, out_chans, latent_dim):
        super().__init__()
        self.encode = Encoder(in_chans, latent_dim)
        self.decode = Decoder(latent_dim, out_chans)


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
                 blocks=2,
                 channel_multipliers=[1,2,4,4]):
        super().__init__()
        self.encode = EncoderV2(num_chans, in_chans, z_dim, channel_multipliers, blocks)
        self.decode = DecoderV2(in_chans, embed_dim, channel_multipliers, blocks, out_chans)

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