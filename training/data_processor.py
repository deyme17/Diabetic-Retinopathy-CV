import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset

from typing import Tuple, List, Optional
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class DataProcessor:
    """
    Handles dataset loading, splitting, normalization, and augmentation.
    """
    def __init__(self, data_path: str, image_size: Tuple[int, int],
                 train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 manual_seed: int = 42) -> None:
        """
        Initialize DataProcessor.
        Args:
            data_path: Path to dataset directory
            image_size: Target image size (height, width)
            train_val_test_split: Ratios for train/val/test split
            manual_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.image_size = image_size
        self.split_ratios = train_val_test_split
        self.manual_seed = manual_seed
        self.mean = None
        self.std = None

        self._full_dataset = None
        self._num_classes = None

    @property
    def full_dataset(self) -> Optional[ImageFolder]:
        if not self._full_dataset:
            raise Exception("Dataset is not processed yet")
        return self._full_dataset

    @property
    def num_classes(self) -> Optional[int]:
        if not self._num_classes:
            raise Exception("Dataset is not processed yet")
        return self._num_classes
    
    def process(self, batch_size: int = 32, augmentation_level: int = 0) -> Tuple[Subset, Subset, Subset]:
        """
        Full pipeline: load, split, normalize, and prepare datasets.
        Args:
            batch_size: Batch size for normalization calculation
            augmentation_level: Level of data augmentation:
                0: No augmentation
                1: Baseline augmentation
                2: Advanced augmentation
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        # load and split
        self._full_dataset = self.load_dataset()
        train_ds, val_ds, test_ds = self.split_dataset(self._full_dataset)
        # normalize
        self.calculate_normalization(train_ds, batch_size)
        # transform
        self.apply_transforms(train_ds, val_ds, test_ds, augmentation_level)
        return train_ds, val_ds, test_ds
        
    def load_dataset(self) -> ImageFolder:
        """Load the full dataset with basic transforms."""
        base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(self.data_path, transform=base_transform)
        self._num_classes = len(dataset.classes)
        return dataset
    
    def split_dataset(self, dataset: ImageFolder) -> Tuple[Subset, Subset, Subset]:
        """
        Split dataset into train, validation, and test sets.
        Args:
            dataset: Full dataset to split
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        train_size = int(self.split_ratios[0] * len(dataset))
        val_size = int(self.split_ratios[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.manual_seed)
        )
        return train_ds, val_ds, test_ds
    
    def calculate_normalization(self, train_ds: Subset,batch_size: int = 32) -> Tuple[List[float], List[float]]:
        """
        Calculate mean and std from training set for normalization.
        Args:
            train_ds: Training dataset
            batch_size: Batch size for calculation
        Returns:
            Tuple of (mean, std)
        """
        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        
        mean = 0.0
        std = 0.0
        total_samples = 0
        
        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_samples
        
        mean /= total_samples
        std /= total_samples
        
        self.mean = mean.tolist()
        self.std = std.tolist()
        
        return self.mean, self.std
    
    def get_baseline_augmentation(self) -> transforms.Compose:
        """
        Get baseline augmentation transforms for training data.
        Includes: HorizontalFlip, Rotation, ColorJitter
        Returns:
            Composed augmentation transforms
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def get_advanced_augmentation(self) -> transforms.Compose:
        """
        Get advanced augmentation transforms for training data.
        Includes: HorizontalFlip, Rotation, ColorJitter, 
                  RandomAffine, GaussianBlur
        Returns:
            Composed augmentation transforms
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1), shear=2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
    
    def get_base_transform(self) -> transforms.Compose:
        """
        Get base transforms without augmentation (for val/test).
        Returns:
            Composed base transforms
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def apply_transforms(self, train_ds: Subset, val_ds: Subset, test_ds: Subset, 
                         augmentation_level: int = 0) -> None:
        """
        Apply transforms to all datasets.
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            test_ds: Test dataset
            augmentation_level: Level of data augmentation:
                0: No augmentation
                1: Baseline augmantation
                2: Advanced augmentation
        """
        if augmentation_level == 1:
            train_ds.dataset.transform = self.get_baseline_augmentation()
        elif augmentation_level == 2:
            train_ds.dataset.transform = self.get_advanced_augmentation()
        else:
            train_ds.dataset.transform = self.get_base_transform()
            
        val_ds.dataset.transform = self.get_base_transform()
        test_ds.dataset.transform = self.get_base_transform()
    
    @staticmethod
    def compute_class_weights(dataset: Subset, device: str = "cpu") -> torch.Tensor:
        """
        Compute class weights using only the training subset.
        """
        full_dataset = dataset.dataset
        indices = dataset.indices
        targets = [full_dataset.samples[i][1] for i in indices]

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(targets),
            y=targets
        )
        return torch.tensor(class_weights, dtype=torch.float).to(device)
