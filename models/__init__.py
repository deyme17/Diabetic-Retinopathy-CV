from .baseline_cnn import BaseLineCNN
from .ensemble import EnsembleModel
from .transfer_models import BaseTransferModel, TransferResNet50, TransferEfficientNetb0

transfer_models: dict[str, type[BaseTransferModel]] = {
    "resnet50": TransferResNet50,
    "efficientnet_b0": TransferEfficientNetb0,
}