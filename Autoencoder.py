import torch
import torch.nn as nn
import torch.nn.functional as F


# Autoencoder Definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Use MSE as reconstruction error
            error = F.mse_loss(reconstructed, x, reduction='none').mean(dim=1)
            return error
