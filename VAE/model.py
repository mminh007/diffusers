import torch
import torch.nn as nn


def get_act(act_type = "relu"):
    if act_type == "relu":
        return nn.ReLU()
    
    elif act_type == "lrelu":
        return nn.LeakyReLU()

    elif act_type == "silu":
        return nn.SiLU()
    
    else:
        return nn.ReLU()
    

class CNNBlock(nn.Module):
    """
    A convolutional block with the following structure:
    Conv2D -> BatchNorm -> Activation.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        activation (str): Type of activation function (e.g., "relu", "lrelu", "silu").
        **kwargs: Additional arguments for the convolutional layer.
    """
    
    def __init__(self, in_chans, out_chans, activation="relu", **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, **kwargs)
        self.norm = nn.BatchNorm2d(num_features=out_chans)
        self.act = get_act(act_type=activation)

    
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
        return out


class ResBlock(nn.Module):
    """
    """
    expansion = 4
    def __init__(self, in_chans, out_chans, stride, downsample=None):
        super().__init__()
        self.conv1 = CNNBlock(in_chans, out_chans, kernel_size=1, stride=1, padding=0, activation="relu")
        self.conv2 = CNNBlock(out_chans, out_chans,  kernel_size=3, stride=stride, padding=1, activation="relu")
        self.conv3 = nn.Conv2d(out_chans, out_chans*self.expansion, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(num_features=out_chans*self.expansion)
        self.act3 = nn.ReLU()

        self.i_downsample = downsample

    def forward(self, x):
        indentity = x.clone()

        if self.i_downsample is not None:
            indentity = self.i_downsample(indentity)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + indentity
        out = self.act3(x)
        
        return out
    

class Encode(nn.Module):
    """
    """

    def __init__(self, in_chans, latent_dim=128):
        super().__init__()
        self.dim = 32
        self.conv1 = CNNBlock(in_chans, 32, kernel_size=3, activation='relu', stride=2, padding=1)
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
    

class Decode(nn.Module):
    """
    """
    def __init__(self, latent_dim=128, out_chans=3):
        super().__init__()
        self.fc_decode = nn.Linear(latent_dim, 128*8*8)
        self.conv1 = CNNBlock(latent_dim, latent_dim // 2, kernel_size=3, stride=1, padding=1, activation="relu")
        self.conv2 = CNNBlock(latent_dim // 2, latent_dim, kernel_size=3, stride=1, padding=1, activation="relu")

        self.deconv1 = nn.ConvTranspose2d(latent_dim, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc_decode(x).view(-1, 128, 8, 8)

        indentity = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        x = + indentity

        x = self.deconv1(x)
        x = self.deconv2(x)
        out = self.sigmoid(x)

        return out


class ResNetVAE(nn.Module):
    """
    """
    def __init__(self, in_chans, out_chans, latent_dim):
        super().__init__()
        self.encoder = Encode(in_chans, latent_dim)
        self.decoder = Decode(latent_dim, out_chans)


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        
        return mu + eps * std
    

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar
    

