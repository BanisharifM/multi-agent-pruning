#!/usr/bin/env python3
"""
Dataset Factory for Multi-Agent LLM Pruning Framework

This module provides a centralized factory for creating and managing datasets
for training, validation, and testing during the pruning process.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np

logger = logging.getLogger(__name__)

class DatasetFactory:
    """
    Factory class for creating and managing datasets for pruning experiments.
    Supports common datasets and custom data loading.
    """
    
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            'imagenet': {
                'num_classes': 1000,
                'input_size': (224, 224),
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'dataset_class': ImageFolder
            },
            'cifar10': {
                'num_classes': 10,
                'input_size': (32, 32),
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2023, 0.1994, 0.2010],
                'dataset_class': torchvision.datasets.CIFAR10
            },
            'cifar100': {
                'num_classes': 100,
                'input_size': (32, 32),
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761],
                'dataset_class': torchvision.datasets.CIFAR100
            },
            'mnist': {
                'num_classes': 10,
                'input_size': (28, 28),
                'mean': [0.1307],
                'std': [0.3081],
                'dataset_class': torchvision.datasets.MNIST
            }
        }
        
        logger.info(f"📁 Dataset Factory initialized with data root: {self.data_root}")
    
    def create_dataloaders(self, dataset_name: str, batch_size: int = 32,
                          num_workers: int = 4, pin_memory: bool = True,
                          subset_ratio: Optional[float] = None,
                          custom_transforms: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.
        
        Args:
            dataset_name: Name of the dataset ('imagenet', 'cifar10', etc.)
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            subset_ratio: Ratio of dataset to use (for quick experiments)
            custom_transforms: Custom transforms to override defaults
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        
        logger.info(f"📁 Creating dataloaders for {dataset_name}")
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        
        # Create transforms
        train_transform, val_transform = self._create_transforms(
            dataset_name, config, custom_transforms
        )
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self._create_datasets(
            dataset_name, config, train_transform, val_transform, **kwargs
        )
        
        # Apply subset if requested
        if subset_ratio is not None:
            train_dataset = self._create_subset(train_dataset, subset_ratio)
            val_dataset = self._create_subset(val_dataset, subset_ratio)
            test_dataset = self._create_subset(test_dataset, subset_ratio)
            logger.info(f"📊 Using {subset_ratio:.1%} subset of dataset")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"✅ Created dataloaders: Train={len(train_dataset)}, "
                   f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_single_dataloader(self, dataset_name: str, split: str = 'train',
                                batch_size: int = 32, **kwargs) -> DataLoader:
        """
        Create a single dataloader for a specific split.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size
            **kwargs: Additional arguments
            
        Returns:
            DataLoader for the specified split
        """
        
        train_loader, val_loader, test_loader = self.create_dataloaders(
            dataset_name, batch_size, **kwargs
        )
        
        if split == 'train':
            return train_loader
        elif split == 'val':
            return val_loader
        elif split == 'test':
            return test_loader
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name].copy()
        
        # Add additional info
        config.update({
            'supported': True,
            'data_path': self.data_root / dataset_name,
            'download_available': dataset_name in ['cifar10', 'cifar100', 'mnist']
        })
        
        return config
    
    def list_supported_datasets(self) -> list:
        """List all supported datasets."""
        return list(self.dataset_configs.keys())
    
    def _create_transforms(self, dataset_name: str, config: Dict[str, Any],
                          custom_transforms: Optional[Dict[str, Any]] = None) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create train and validation transforms."""
        
        input_size = config['input_size']
        mean = config['mean']
        std = config['std']
        
        # Default transforms
        if dataset_name == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        elif dataset_name in ['cifar10', 'cifar100']:
            train_transform = transforms.Compose([
                transforms.RandomCrop(input_size[0], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        elif dataset_name == 'mnist':
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
        else:
            # Generic transforms
            train_transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        # Apply custom transforms if provided
        if custom_transforms:
            if 'train' in custom_transforms:
                train_transform = custom_transforms['train']
            if 'val' in custom_transforms:
                val_transform = custom_transforms['val']
        
        return train_transform, val_transform
    
    def _create_datasets(self, dataset_name: str, config: Dict[str, Any],
                        train_transform: transforms.Compose,
                        val_transform: transforms.Compose,
                        **kwargs) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train, validation, and test datasets."""
        
        dataset_class = config['dataset_class']
        dataset_path = self.data_root / dataset_name
        
        if dataset_name == 'imagenet':
            # ImageNet structure: train/, val/
            train_path = dataset_path / 'train'
            val_path = dataset_path / 'val'
            
            if not train_path.exists() or not val_path.exists():
                raise FileNotFoundError(f"ImageNet data not found at {dataset_path}")
            
            train_dataset = dataset_class(train_path, transform=train_transform)
            val_dataset = dataset_class(val_path, transform=val_transform)
            test_dataset = dataset_class(val_path, transform=val_transform)  # Use val as test
            
        elif dataset_name in ['cifar10', 'cifar100', 'mnist']:
            # Torchvision datasets with automatic download
            train_dataset = dataset_class(
                root=dataset_path,
                train=True,
                download=True,
                transform=train_transform
            )
            
            # Create validation split from training data
            val_size = int(0.1 * len(train_dataset))  # 10% for validation
            train_size = len(train_dataset) - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            # Test dataset
            test_dataset = dataset_class(
                root=dataset_path,
                train=False,
                download=True,
                transform=val_transform
            )
            
        else:
            raise ValueError(f"Dataset creation not implemented for {dataset_name}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, dataset: Dataset, ratio: float) -> Subset:
        """Create a subset of the dataset."""
        
        dataset_size = len(dataset)
        subset_size = int(dataset_size * ratio)
        
        # Random indices
        indices = torch.randperm(dataset_size)[:subset_size].tolist()
        
        return Subset(dataset, indices)
    
    def create_custom_dataset(self, data_path: str, transform: Optional[transforms.Compose] = None,
                            num_classes: Optional[int] = None) -> Dataset:
        """
        Create a custom dataset from a directory.
        
        Args:
            data_path: Path to the data directory
            transform: Transform to apply to the data
            num_classes: Number of classes (if known)
            
        Returns:
            Custom dataset
        """
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Assume ImageFolder structure
        dataset = ImageFolder(data_path, transform=transform)
        
        logger.info(f"📁 Created custom dataset: {len(dataset)} samples, {len(dataset.classes)} classes")
        
        return dataset
    
    def get_sample_batch(self, dataset_name: str, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch from the dataset for testing.
        
        Args:
            dataset_name: Name of the dataset
            batch_size: Size of the sample batch
            
        Returns:
            Tuple of (inputs, targets)
        """
        
        dataloader = self.create_single_dataloader(
            dataset_name, split='train', batch_size=batch_size
        )
        
        # Get first batch
        inputs, targets = next(iter(dataloader))
        
        logger.info(f"📊 Sample batch: {inputs.shape}, targets: {targets.shape}")
        
        return inputs, targets
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate that a dataset is properly configured and accessible.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Validation results
        """
        
        validation_results = {
            'dataset_name': dataset_name,
            'is_valid': False,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check if dataset is supported
            if dataset_name not in self.dataset_configs:
                validation_results['issues'].append(f"Dataset {dataset_name} not supported")
                return validation_results
            
            # Try to create dataloaders
            train_loader, val_loader, test_loader = self.create_dataloaders(
                dataset_name, batch_size=2, subset_ratio=0.01  # Small subset for validation
            )
            
            # Test data loading
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            test_batch = next(iter(test_loader))
            
            # Collect statistics
            validation_results['statistics'] = {
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
                'test_samples': len(test_loader.dataset),
                'input_shape': list(train_batch[0].shape),
                'num_classes': len(torch.unique(train_batch[1])),
                'batch_size': train_batch[0].size(0)
            }
            
            validation_results['is_valid'] = True
            
        except FileNotFoundError as e:
            validation_results['issues'].append(f"Data files not found: {str(e)}")
        except Exception as e:
            validation_results['issues'].append(f"Validation failed: {str(e)}")
        
        return validation_results
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download a dataset if automatic download is supported.
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            True if download was successful
        """
        
        if dataset_name not in self.dataset_configs:
            logger.error(f"Dataset {dataset_name} not supported")
            return False
        
        if dataset_name not in ['cifar10', 'cifar100', 'mnist']:
            logger.warning(f"Automatic download not available for {dataset_name}")
            return False
        
        try:
            logger.info(f"📥 Downloading {dataset_name}...")
            
            # Create a dummy dataloader to trigger download
            _ = self.create_single_dataloader(dataset_name, batch_size=1)
            
            logger.info(f"✅ {dataset_name} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {str(e)}")
            return False
    
    def get_class_names(self, dataset_name: str) -> list:
        """Get class names for a dataset."""
        
        if dataset_name == 'cifar10':
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset_name == 'cifar100':
            # CIFAR-100 has 100 classes - return first few as example
            return [f'class_{i}' for i in range(100)]
        elif dataset_name == 'mnist':
            return [str(i) for i in range(10)]
        elif dataset_name == 'imagenet':
            # Would need to load from file in practice
            return [f'class_{i}' for i in range(1000)]
        else:
            return []
    
    def create_balanced_subset(self, dataset: Dataset, samples_per_class: int) -> Subset:
        """Create a balanced subset with equal samples per class."""
        
        # This is a simplified implementation
        # In practice, you'd need to analyze the dataset to get class distributions
        
        total_samples = samples_per_class * self.get_dataset_info('cifar10')['num_classes']  # Example
        subset_size = min(total_samples, len(dataset))
        
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        
        return Subset(dataset, indices)

