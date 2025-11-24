
import torch
import torch.nn as nn
from torchvision import models
from typing import Literal, List, Dict
from abc import ABC, abstractmethod


class BaseTransferModel(nn.Module, ABC):
    """
    Abstract base class for transfer learning models.
    """
    def __init__(self, num_classes: int, mode: Literal['feature_extraction', 'fine_tuning'], dropout_rate: float = 0):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.dropout_rate = dropout_rate
        self._backbone = None
        self._backbone_layers = []
    
    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """Build and return the backbone model with modified classifier."""
        pass

    @abstractmethod
    def _collect_backbone_layers(self):
        """Fill self._backbone_layers as [layer0, layer1, ...]"""
        pass
    
    @abstractmethod
    def _get_classifier_params(self) -> List:
        """Return parameters of the classifier layer(s)."""
        pass
    
    @abstractmethod
    def _get_backbone_params(self) -> List:
        """Return parameters of the backbone (excluding classifier)."""
        pass
    
    def _apply_freezing(self, freeze_until_layer: int = -1):
        """
        Apply freezing strategy based on mode.
        Args:
            freeze_until_layer: For fine-tuning, freeze layers until this index (-1 = unfreeze all)
        """
        for p in self._get_classifier_params():
            p.requires_grad = True

        if self.mode == "feature_extraction":
            for p in self._get_backbone_params():
                p.requires_grad = False
            return

        if freeze_until_layer == -1:
            for p in self._get_backbone_params():
                p.requires_grad = True
            return

        for i, layer in enumerate(self._backbone_layers):
            for p in layer.parameters():
                p.requires_grad = (i >= freeze_until_layer)
    
    def forward(self, x):
        return self._backbone(x)
    
    def get_trainable_params(self):
        """Get only parameters that require gradients."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def get_param_groups(self, lr_backbone: float = 1e-4, lr_classifier: float = 1e-3) -> List[Dict]:
        """
        Get parameter groups with different learning rates.
        Args:
            lr_backbone: Learning rate for backbone
            lr_classifier: Learning rate for classifier
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self._get_backbone_params(), 'lr': lr_backbone},
            {'params': self._get_classifier_params(), 'lr': lr_classifier}
        ]
    
    def print_model_info(self):
        """Print information about the model architecture and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model: {self.__class__.__name__}")
        print(f"Mode: {self.mode}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params/total_params:.2f}%)")


class TransferResNet50(BaseTransferModel):
    """
    Tranfer learning model with ResNet50 backbone.
    """
    def __init__(self, num_classes: int, mode: Literal['feature_extraction', 'fine_tuning'] = "feature_extraction",
                freeze_until_layer: int = -1, dropout_rate: float = 0) -> None:
        super().__init__(num_classes, mode, dropout_rate)
        self._backbone = self._build_backbone()
        self._collect_backbone_layers()
        self._apply_freezing(freeze_until_layer)
    
    def _build_backbone(self) -> models.ResNet:
        """Build and return the backbone model with modified classifier."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features, self.num_classes)
            )
        return model
    
    def _get_classifier_params(self) -> List:
        """Return parameters of the classifier layer(s)."""
        return self._backbone.fc.parameters()
    
    def _get_backbone_params(self) -> List:
        """Return parameters of the backbone (excluding classifier)."""
        return [param for name, param in self._backbone.named_parameters() if "fc" not in name]
    
    def _collect_backbone_layers(self) -> None:
        """Return parameters of the backbone (excluding classifier)."""
        self._backbone_layers = [
            self._backbone.conv1,
            self._backbone.bn1,
            self._backbone.layer1,
            self._backbone.layer2,
            self._backbone.layer3,
            self._backbone.layer4,
        ]


class TransferEfficientNetb0(BaseTransferModel):
    """
    Tranfer learning model with EfficientNetb0 backbone.
    """
    def __init__(self, num_classes: int, mode: Literal['feature_extraction', 'fine_tuning'] = "feature_extraction",
                freeze_until_layer: int = -1, dropout_rate: float = 0) -> None:
        super().__init__(num_classes, mode, dropout_rate)
        self._backbone = self._build_backbone()
        self._collect_backbone_layers()
        self._apply_freezing(freeze_until_layer)
    
    def _build_backbone(self) -> models.EfficientNet:
        """Build and return the backbone model with modified classifier."""
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features, self.num_classes)
            )
        return model
    
    def _get_classifier_params(self) -> List:
        """Return parameters of the classifier layer(s)."""
        return self._backbone.classifier.parameters()
    
    def _get_backbone_params(self) -> List:
        """Return parameters of the backbone (excluding classifier)."""
        return [param for name, param in self._backbone.named_parameters() if "classifier" not in name]
    
    def _collect_backbone_layers(self) -> None:
        """Return parameters of the backbone (excluding classifier)."""
        self._backbone_layers = list(self._backbone.features.children())
        self._backbone_layers.append(self._backbone.avgpool)