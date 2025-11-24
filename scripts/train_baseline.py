from multiprocessing import freeze_support
from datetime import datetime
import argparse
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from models import BaseLineCNN

from training.data_processor import DataProcessor
from training.early_stopping import EarlyStopping
from training.train_pipeline import TrainPipeline
from training.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DROPOUT_RATE,
    ES_PATIANCE, LRS_PATIANCE, LRS_PLATO_FACTOR,
    MANUAL_SEED, TRAIN_VAL_TEST_SPLIT, IMAGE_SIZE,
    WEIGHT_DECAY, LABEL_SMOOTHING
)

def start_training(
    lr: float = LEARNING_RATE,
    augmentation_level: int = 0,
    batch_size: int = BATCH_SIZE,
    dropout: float = DROPOUT_RATE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
    num_epochs: int = NUM_EPOCHS,
    save_name: str = "baseline"
):
    """
    Train a model.
    Args:
        lr: Learning rate
        augmentation_level: Augmentation level (0-No, 1-Baseline, 2-Advanced)
        batch_size: Batch size for training
        dropout: Dropout rate for regularization
        weight_decay: L2 regularization parameter
        label_smoothing: label smoothing parameter
        num_epochs: Number of training epochs
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
    # dataloaders
    print("Load data...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=3)

    # model
    model = BaseLineCNN(processor.num_classes, dropout_rate=dropout)

    # criterion, optimizer, scheduler
    class_weights = processor.compute_class_weights(dataset=train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        factor=LRS_PLATO_FACTOR,
                                                        patience=LRS_PATIANCE)
    early_stopping = EarlyStopping(patience=ES_PATIANCE)

    print(f"Train name: {save_name}")
    print(f"Learning rate: {lr}")
    print(f"Dropout rate: {dropout}")
    print(f"Weight decay: {weight_decay}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Augmentation level: {augmentation_level}")
    print(f"Device: {device}")

    # main pipeline
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

    if best_model is not None:
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"checkpoints/checkpoint_{save_name}({postfix}).pth")
    if metrics is not None:
        with open(f"results/metrics_{save_name}({postfix}).json", "w") as f:
            json.dump(metrics, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2],
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
    
    start_training(
        lr=args.lr,
        augmentation_level=args.augmentation,
        batch_size=args.batch_size,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        save_name=args.save_name
    )

# main section
if __name__=='__main__':
    freeze_support()
    main()