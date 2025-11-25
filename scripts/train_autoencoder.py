from multiprocessing import freeze_support
from datetime import datetime
import os
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from models import ConvAutoencoder
from training.data_processor import DataProcessor
from training.early_stopping import EarlyStopping
from training.config import (
    BATCH_SIZE, NUM_EPOCHS, ES_PATIANCE,
    LRS_PATIANCE, LRS_PLATO_FACTOR, LEARNING_RATE,
    MANUAL_SEED, IMAGE_SIZE, TRAIN_VAL_TEST_SPLIT
)

def train_autoencoder(
    lr: float = 1e-3,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    augmentation_level: int = 0, 
    save_name: str = None,
    include_classes: list[int] = None
):
    """
    Train a Convolutional Autoencoder.
    Args:
        lr: Learning rate
        batch_size: Batch size
        num_epochs: Max epochs
        augmentation_level: 0=None, 1=Base, 2=Adv
        save_name: Custom name for checkpoint
        include_classes: list of class labels to learn autoencoder
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # load data
    processor = DataProcessor(
        data_path="data",
        image_size=IMAGE_SIZE,
        train_val_test_split=(0.8, 0.1, 0.1),
        manual_seed=MANUAL_SEED
    )
    train_ds, val_ds, _ = processor.process(
        batch_size=batch_size,
        augmentation_level=augmentation_level
    )
    if include_classes:
            print(f"Filtering dataset. Keeping classes: {include_classes}")
            def filter_dataset(dataset, allowed_classes):
                all_targets = dataset.dataset.targets
                filtered_indices = []
                for i, real_idx in enumerate(dataset.indices):
                    if all_targets[real_idx] in allowed_classes:
                        filtered_indices.append(i)
                return Subset(dataset, filtered_indices)

            train_ds = filter_dataset(train_ds, include_classes)
            val_ds = filter_dataset(val_ds, include_classes)

    print("Loading data...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # model
    model = ConvAutoencoder()
    model.to(device)
    
    # criterion, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=LRS_PLATO_FACTOR,
        patience=LRS_PATIANCE,
    )
    early_stopping = EarlyStopping(patience=ES_PATIANCE)

    # train
    print("Training starts...")
    metrics = {"train_loss": [], "val_loss": []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # save loss
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        
        print(f"[{epoch + 1}/{num_epochs}] train_loss: {avg_train_loss:.5f} | val_loss: {avg_val_loss:.5f}")
        
        scheduler.step(avg_val_loss)
        if not early_stopping.step(model, avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # save results
    model.load_state_dict(early_stopping.best_state)
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    now = datetime.now()
    postfix = now.strftime('%d-%m_%H-%M')
    name = save_name or "autoencoder"
    
    # save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'lr': lr,
            'batch_size': batch_size,
            'augmentation': augmentation_level
        }
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_{name}({postfix}).pth")
    
    # save metrics
    with open(f"results/metrics_{name}({postfix}).json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Done! Model saved to checkpoints/checkpoint_{name}({postfix}).pth")

def main():
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--augmentation', type=int, default=0, choices=[0, 1, 2], 
                        help='Augmentation level (0-No, 1-Base, 2-Adv)')
    parser.add_argument('--save-name', type=str, default="autoencoder", help='Checkpoint name')
    parser.add_argument('--include-classes', type=int, nargs='+', 
                        help='class labels to include in autoencoder learning')
    
    args = parser.parse_args()
    
    train_autoencoder(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        augmentation_level=args.augmentation,
        save_name=args.save_name,
        include_classes=args.include_classes
    )

# main section
if __name__ == '__main__':
    freeze_support()
    main()