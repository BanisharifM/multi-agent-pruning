#!/usr/bin/env python3
"""
Multi-Agent LLM Pruning Framework - Data Package

This package contains data loading, preprocessing, and dataset management
utilities for the pruning framework.
"""

from .dataset_factory import DatasetFactory

__all__ = [
    'DatasetFactory'
]

# Supported datasets
SUPPORTED_DATASETS = [
    'imagenet',
    'cifar10',
    'cifar100',
    'mnist',
    'custom'
]

# Dataset configurations
DATASET_CONFIGS = {
    'imagenet': {
        'num_classes': 1000,
        'input_size': (224, 224),
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'cifar10': {
        'num_classes': 10,
        'input_size': (32, 32),
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'cifar100': {
        'num_classes': 100,
        'input_size': (32, 32),
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    }
}

