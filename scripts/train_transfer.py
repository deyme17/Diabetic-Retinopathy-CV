from multiprocessing import freeze_support
from datetime import datetime
import os
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import transfer_models
from training.data_processor import DataProcessor
from training.early_stopping import EarlyStopping
from training.train_pipeline import TrainPipeline
from training.config import (
    BATCH_SIZE, NUM_EPOCHS, ES_PATIANCE, 
    LRS_PATIANCE, LRS_PLATO_FACTOR, DROPOUT_RATE,
    MANUAL_SEED, TRAIN_VAL_TEST_SPLIT, IMAGE_SIZE,
    TR_LR_BACKBONE, TR_LR_CLASSIFIER, 
    WEIGHT_DECAY, LABEL_SMOOTHING
)


def train_transfer_model(
    model_name: str = 'resnet50',
    mode: str = 'feature_extraction',
    lr_backbone: float = TR_LR_BACKBONE,
    lr_classifier: float = TR_LR_CLASSIFIER,
    freeze_until_layer: int = -1,
    augmentation_level: int = 0,
    batch_size: int = BATCH_SIZE,
    dropout: float = DROPOUT_RATE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
    num_epochs: int = NUM_EPOCHS,
    save_name: str = None
):
    """
    Train a transfer learning model.
    Args:
        model_name: 'resnet50', 'efficientnet_b0'
        mode: 'feature_extraction' or 'fine_tuning'
        lr_backbone: Learning rate for backbone (used in fine-tuning)
        lr_classifier: Learning rate for classifier
        freeze_until_layer: Freeze layers until this index (-1 = unfreeze all)
        augmentation_level: Augmentation level (0-No, 1-Baseline, 2-Advanced)
        batch_size: Batch size for training
        dropout: Dropout rate for regularization
        num_epochs: Number of training epochs
        weight_decay: L2 regularization parameter
        label_smoothing: label smoothing parameter
        save_name: Model name to save
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load and preprocess data
    processor = DataProcessor(
        data_path="data",
        image_size=IMAGE_SIZE,
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        manual_seed=MANUAL_SEED
    )
    train_ds, val_ds, test_ds = processor.process(
        batch_size=batch_size,
        augmentation_level=augmentation_level,
    )
    # dataLoaders
    print("Loading data...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=3)
    
    # model
    model_cls = transfer_models.get(model_name, None)
    if model_cls is None:
        raise Exception(f"No transfer model with name `{model_name}`.")
    model = model_cls(
        num_classes=processor.num_classes,
        mode=mode,
        freeze_until_layer=freeze_until_layer,
        dropout_rate=dropout
    )
    # info
    model.print_model_info()
    print(f"Backbone learning rate: {lr_backbone}")
    print(f"Classifier learning rate: {lr_classifier}")
    print(f"Dropout rate: {dropout}")
    print(f"Weight decay: {weight_decay}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Augmentation level: {augmentation_level}")
    print(f"Device: {device}")
    
    # criterion, optimizer, scheduler
    class_weights = processor.compute_class_weights(dataset=train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    if mode == 'feature_extraction':
        optimizer = torch.optim.Adam(model.get_trainable_params(), lr=lr_classifier, weight_decay=weight_decay)
    else:
        param_groups = model.get_param_groups(lr_backbone=lr_backbone, lr_classifier=lr_classifier)
        optimizer = torch.optim.Adam(param_groups)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=LRS_PLATO_FACTOR,
        patience=LRS_PATIANCE
    )
    early_stopping = EarlyStopping(patience=ES_PATIANCE)
    
    # training pipeline
    pipeline = TrainPipeline(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=num_epochs,
        scheduler=scheduler,
        early_stopping=early_stopping
    )
    # train
    print("Training starts...")
    best_model, metrics = pipeline.train()
    
    # save model & metrics
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    now = datetime.now()
    postfix = now.strftime('%d-%m_%H-%M')
    filename = save_name or f"{model_name}_{mode}"
    
    if best_model is not None:
        checkpoint = {
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': model_name,
            'mode': mode,
            'num_classes': processor.num_classes,
            'config': {
                'lr_backbone': lr_backbone,
                'lr_classifier': lr_classifier,
                'freeze_until_layer': freeze_until_layer,
                'augmentation_level': augmentation_level,
                'batch_size': batch_size,
                'dropout': dropout,
                "Weight decay": weight_decay,
                "Label smoothing": label_smoothing
            }
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_{filename}({postfix}).pth")
    if metrics is not None:
        with open(f"results/metrics_{filename}({postfix}).json", "w") as f:
            json.dump(metrics, f, indent=4)
    
    return best_model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train transfer learning model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b0'],
                       help='Pretrained model to use')
    parser.add_argument('--mode', type=str, default='feature_extraction',
                       choices=['feature_extraction', 'fine_tuning'],
                       help='Training mode')
    parser.add_argument('--lr-backbone', type=float, default=TR_LR_BACKBONE,
                       help='Learning rate for backbone')
    parser.add_argument('--lr-classifier', type=float, default=TR_LR_CLASSIFIER,
                       help='Learning rate for classifier')
    parser.add_argument('--freeze-until', type=int, default=-1,
                       help='Freeze layers until this index (-1 = unfreeze all)')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2], default=0,
                       help='Augmentation level (0-No, 1-Baseline, 2-Advanced)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE,
                       help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                       help='Weight decay L2 regularization parameter')
    parser.add_argument('--label-smoothing', type=float, default=LABEL_SMOOTHING,
                       help='Label smoothing regularization parameter')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--save-name', type=str, default="baseline",
                       help='Model name to save')
    args = parser.parse_args()
    
    train_transfer_model(
        model_name=args.model,
        mode=args.mode,
        lr_backbone=args.lr_backbone,
        lr_classifier=args.lr_classifier,
        freeze_until_layer=args.freeze_until,
        augmentation_level=args.augmentation,
        batch_size=args.batch_size,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        save_name=args.save_name
    )

# main section
if __name__ == '__main__':
    freeze_support()
    main()