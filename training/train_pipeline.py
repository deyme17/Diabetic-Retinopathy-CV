from sklearn.metrics import precision_score, recall_score, f1_score
from .early_stopping import EarlyStopping
from typing import Optional, Any, Tuple, Dict
from torch.utils.data import DataLoader
from torch import nn
import torch


class TrainPipeline:
    """Main model training pipeline"""
    def __init__(self,
                 model: nn.Module,
                 criterion: Any,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 epochs: int = 50,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 early_stopping: Optional[EarlyStopping] = None) -> None:
        """
        Args:
            model (Module): Model to train
            criterion (SomeCriterion): Loss function
            optimizer (Optimizer): model optimizer
            train_loader (DataLoader): train dataset
            val_loader (DataLoader): validation dataset
            test_loader (DataLoader): test dataset
            epochs (int, default=50): number of epochs
            scheduler (_LRScheduler): optional learning rate scheduler
            early_stopping (EarlyStopping): optional class responsible for early stopping
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }

    def _train_epoch(self) -> float:
        """
        Runs one training epoch.
        Returns:
            float: average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate(self) -> Tuple[float, float, Dict[str, float]]:
        """Runs validation step and returns loss, accuracy, and other metrics"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                running_loss += loss.item()

                _, cls = pred.max(1)
                all_preds.extend(cls.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_acc = sum([p==t for p,t in zip(all_preds, all_labels)]) / len(all_labels)
        val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        metrics = {
            "accuracy": val_acc,
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1
        }

        return val_loss, val_acc, metrics

    def _test(self) -> tuple[float, float]:
        """
        Runs final evaluation on test dataset.
        Returns:
            tuple[float, float]: (test loss, test accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)
                running_loss += loss.item()

                _, cls = pred.max(1)
                correct += (cls == y).sum().item()
                total += y.size(0)

        return running_loss / len(self.test_loader), correct / total

    def train(self) -> Optional[Tuple[nn.Module, Dict[str, float]]]:
        """
        Runs full training loop with validation, scheduler stepping and early stopping.
        Returns:
            (model, metrics): best model (restored using EarlyStopping) and dictionary with metrics
        """
        if self.device == "cpu":
            print("CUDA is NOT available!!")
            answer = input("Type 'yes' if you have tons of time and want to continue: ")
            if not answer.startswith("y"):
                return (None, None)

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_acc, metrics = self._validate()

            # save metrics
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(metrics["accuracy"])
            self.metrics["val_precision"].append(metrics["precision"])
            self.metrics["val_recall"].append(metrics["recall"])
            self.metrics["val_f1"].append(metrics["f1"])

            print(f"[{epoch+1}/{self.epochs}] train loss={train_loss:.4f} | val loss={val_loss:.4f} | val acc={val_acc:.4f}")

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if self.early_stopping is not None:
                if not self.early_stopping.step(self.model, val_loss):
                    print(f"[{epoch+1}/{self.epochs}] Early stopping triggered.\n",
                          f"\ttrain loss={train_loss:.4f} | val loss={val_loss:.4f} | val acc={val_acc:.4f}")
                    break

        # best model state
        if self.early_stopping is not None:
            self.model.load_state_dict(self.early_stopping.best_state)

        # test
        test_loss, test_acc = self._test()
        print(f"\n[TEST]: test loss={test_loss:.4f}  test acc={test_acc:.4f}")

        return self.model, self.metrics