import torch
import torch.nn.functional as F
import torch.nn as nn


class VAE_loss(nn.Module):
    def __init__(self, reduction="mean", B=1000):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.B = B

    def forward(self, x_hat, x, mu, logvar):
        
        recons_loss = self.mse(x_hat, x) / x.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) / x.shape[0]

        total_loss = self.B * recons_loss + torch.mean(kl_loss)

        return total_loss, recons_loss, torch.mean(kl_loss)
