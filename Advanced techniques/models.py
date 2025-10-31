import torch
import torch.nn as nn


class Autoencoder_linear(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
            )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,16),
            nn.ReLU(),
            nn.Linear(16, 30)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z



