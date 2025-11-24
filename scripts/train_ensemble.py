from multiprocessing import freeze_support
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import json
import argparse
from typing import List

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import transfer_models
from training.data_processor import DataProcessor
from models.ensemble import EnsembleModel
from training.config import (
    BATCH_SIZE, MANUAL_SEED, TRAIN_VAL_TEST_SPLIT, IMAGE_SIZE
)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load model from checkpoint file.
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, strict=False)
    
    model_name = checkpoint['model_name']
    mode = checkpoint['mode']
    num_classes = checkpoint['num_classes']
    config = checkpoint['config']
    
    model_cls = transfer_models.get(model_name, None)
    if model_cls is None:
        raise Exception(f"No transfer model with name `{model_name}`.")
    
    model = model_cls(
        num_classes=num_classes,
        mode=mode,
        freeze_until_layer=config.get('freeze_until_layer', -1),
        dropout_rate=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_ensemble(
    ensemble: EnsembleModel,
    test_loader: DataLoader,
    device: str = "cuda"
):
    """
    Evaluate ensemble model on test set.
    Args:
        ensemble: Ensemble model
        test_loader: Test data loader
        device: Device to run evaluation on
    Returns:
        Dictionary with metrics
    """
    ensemble.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            
            probs = ensemble(x)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # calculate metrics
    accuracy = sum([p == t for p, t in zip(all_preds, all_labels)]) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }
    return metrics


def train_ensemble(
    checkpoint_paths: List[str],
    batch_size: int = BATCH_SIZE,
    save_name: str = None
):
    """
    Create and evaluate ensemble model.
    Args:
        checkpoint_paths: List of paths to model checkpoints
        batch_size: Batch size for evaluation
        save_name: Name to save ensemble results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # load test data
    processor = DataProcessor(
        data_path="data",
        image_size=IMAGE_SIZE,
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        manual_seed=MANUAL_SEED
    )
    _, _, test_ds = processor.process(
        batch_size=batch_size,
        augmentation_level=0
    )
    
    print("Loading test data...")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=3)
    
    # load models
    print(f"Loading {len(checkpoint_paths)} models...")
    models = []
    for i, path in enumerate(checkpoint_paths):
        print(f"  [{i+1}/{len(checkpoint_paths)}] Loading {path}")
        model = load_model_from_checkpoint(path, device)
        models.append(model)
    
    # create ensemble
    print("Creating ensemble...")
    ensemble = EnsembleModel(models, device=device)
    print(f"Ensemble size: {len(models)} models")
    print(f"Device: {device}")
    
    # evaluate
    print("Evaluating ensemble on test set...")
    metrics = evaluate_ensemble(ensemble, test_loader, device)
    
    print(f"\n[ENSEMBLE TEST RESULTS]:")
    print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Test Recall:    {metrics['test_recall']:.4f}")
    print(f"  Test F1-Score:  {metrics['test_f1']:.4f}")
    
    # save results
    os.makedirs("results", exist_ok=True)
    
    now = datetime.now()
    postfix = now.strftime('%d-%m_%H-%M')
    filename = save_name or "ensemble"
    
    # save metrics (without large arrays)
    save_metrics = {
        "test_accuracy": metrics['test_accuracy'],
        "test_precision": metrics['test_precision'],
        "test_recall": metrics['test_recall'],
        "test_f1": metrics['test_f1'],
        "confusion_matrix": metrics['confusion_matrix'],
        "ensemble_size": len(models),
        "checkpoint_paths": checkpoint_paths
    }
    
    with open(f"results/metrics_{filename}({postfix}).json", "w") as f:
        json.dump(save_metrics, f, indent=4)
    
    # save full predictions
    np.savez(
        f"results/predictions_{filename}({postfix}).npz",
        predictions=metrics['predictions'],
        labels=metrics['labels'],
        probabilities=metrics['probabilities']
    )
    print(f"\nResults saved to results/metrics_{filename}({postfix}).json")
    
    return ensemble, metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble model')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--save-name', type=str, default="ensemble",
                       help='Name to save results')
    args = parser.parse_args()
    
    train_ensemble(
        checkpoint_paths=args.checkpoints,
        batch_size=args.batch_size,
        save_name=args.save_name
    )

# main section
if __name__ == '__main__':
    freeze_support()
    main()