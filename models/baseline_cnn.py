import torch
import torch.nn as nn

class BaseLineCNN(nn.Module):
    """Base line CNN for diabetic rethinopathy recognition"""
    def __init__(self, n_classes: int = 5, dropout_rate: float = 0):
        super().__init__()
        # conv layers
        self.featuresExtraction = nn.Sequential(
            # 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # classification
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass.
        - Feature extraction:
            Conv2d -> BatchNorm -> ReLU (n times)
            -> (opt. MaxPool2d)
        - Classification:
            Linear -> ReLU -> Linear
        """
        X = self.featuresExtraction(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        return X