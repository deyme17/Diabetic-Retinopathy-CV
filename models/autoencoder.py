import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder used for: anomaly detection, denoising, dim.reduction."""
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # (batch_size x 256 x 256 x 3)
            # -> 256x256
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # -> 16x16
            # (batch_size x 16 x 16 x 256)
        )
        self.encoder = nn.Sequential(
            # (batch_size x 16 x 16 x 256)
            # -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)
            # -> 256x256
            # (batch_size x 256 x 256 x 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoencoder forward pass: Encode -> Decode"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return embedding vector for image:
            (batch_size x 256 x 256 x 3) -> (batch_size x 256)
        Used for visualization latent space (t-SNE).
        """
        with torch.no_grad():
            encoded = self.encoder(x) # (b, 256, 16, 16)
            embedding = F.adaptive_avg_pool2d(encoded, (1, 1))
            return torch.flatten(embedding, 1)