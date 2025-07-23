#!/usr/bin/env python3
"""
Multi-Agent LLM Pruning Framework - Models Package

This package contains model factory, model utilities, and model-specific
implementations for different architectures (ResNet, ViT, DeiT, etc.).
"""

from .model_factory import ModelFactory
from .model_utils import ModelUtils
from .architecture_registry import ArchitectureRegistry

__all__ = [
    'ModelFactory',
    'ModelUtils', 
    'ArchitectureRegistry'
]

# Supported model architectures
SUPPORTED_ARCHITECTURES = [
    'resnet50',
    'resnet101', 
    'vit_base_patch16_224',
    'vit_small_patch16_224',
    'deit_base_patch16_224',
    'deit_small_patch16_224',
    'deit_tiny_patch16_224'
]

# Architecture categories
ARCHITECTURE_CATEGORIES = {
    'cnn': ['resnet50', 'resnet101'],
    'transformer': ['vit_base_patch16_224', 'vit_small_patch16_224'],
    'hybrid': ['deit_base_patch16_224', 'deit_small_patch16_224', 'deit_tiny_patch16_224']
}

