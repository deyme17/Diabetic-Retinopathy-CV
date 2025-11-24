from torch import nn

class EarlyStopping:
    """Class responsible for model training early stopping"""
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self._best_state = None
        self.counter = 0

    def step(self, model: nn.Module, val_loss: float) -> bool:
        """
        Should be called on each model training step.
        Args:
            model (nn.Module): model that currently trains
            val_loss (float): Validation loss on current training epoch
        Returns:
            bool (keep the model taining?)
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            return self.counter < self.patience
        
    @property
    def best_state(self) -> dict|None:
        return self._best_state