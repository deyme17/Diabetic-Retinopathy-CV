import torch
import torch.nn as nn
from typing import List

class EnsembleModel(nn.Module):
    """An ensemble model that averages predictions from multiple models."""
    def __init__(self, models: List[nn.Module], device: str = "cuda"):
        super().__init__()
        self.models = models
        self.device = device
        for model in self.models:
            model.to(self.device)
            model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        Returns:
            Average probabilities across all models
        """
        x = x.to(self.device)
        
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs