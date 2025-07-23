#!/usr/bin/env python3
"""
Model Factory for Multi-Agent LLM Pruning Framework

This module provides a centralized factory for creating and loading different
model architectures with proper initialization and configuration.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm not available - some models may not be accessible")

from .model_utils import ModelUtils
from .architecture_registry import ArchitectureRegistry

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating and managing different model architectures.
    Supports both torchvision and timm models with automatic configuration.
    """
    
    def __init__(self):
        self.model_utils = ModelUtils()
        self.architecture_registry = ArchitectureRegistry()
        
        # Model cache for efficiency
        self._model_cache = {}
        
        logger.info("🏭 Model Factory initialized")
    
    def create_model(self, model_name: str, num_classes: int = 1000, 
                    pretrained: bool = True, **kwargs) -> nn.Module:
        """
        Create a model instance with the specified configuration.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
            **kwargs: Additional model-specific arguments
            
        Returns:
            Initialized model instance
        """
        
        logger.info(f"🏭 Creating model: {model_name}")
        
        # Check if model is supported
        if not self.is_supported(model_name):
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get model configuration
        model_config = self.architecture_registry.get_config(model_name)
        
        # Create model based on source
        if model_config['source'] == 'torchvision':
            model = self._create_torchvision_model(model_name, num_classes, pretrained, **kwargs)
        elif model_config['source'] == 'timm':
            model = self._create_timm_model(model_name, num_classes, pretrained, **kwargs)
        else:
            raise ValueError(f"Unknown model source: {model_config['source']}")
        
        # Apply model-specific configurations
        model = self._apply_model_config(model, model_name, model_config)
        
        # Validate model
        self._validate_model(model, model_name)
        
        logger.info(f"✅ Model {model_name} created successfully")
        return model
    
    def load_model(self, model_name: str, checkpoint_path: str, 
                  num_classes: int = 1000, **kwargs) -> nn.Module:
        """
        Load a model from checkpoint.
        
        Args:
            model_name: Name of the model architecture
            checkpoint_path: Path to the checkpoint file
            num_classes: Number of output classes
            **kwargs: Additional model-specific arguments
            
        Returns:
            Model loaded with checkpoint weights
        """
        
        logger.info(f"📂 Loading model {model_name} from {checkpoint_path}")
        
        # Create base model
        model = self.create_model(model_name, num_classes, pretrained=False, **kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            # Try non-strict loading
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info(f"✅ Model {model_name} loaded successfully from checkpoint")
        return model
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model architecture.
        
        Args:
            model_name: Name of the model architecture
            
        Returns:
            Dictionary with model information
        """
        
        if not self.is_supported(model_name):
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get base configuration
        config = self.architecture_registry.get_config(model_name)
        
        # Create temporary model to get detailed info
        try:
            temp_model = self.create_model(model_name, pretrained=False)
            model_info = self.model_utils.analyze_model(temp_model)
            
            # Combine configuration and analysis
            full_info = {
                **config,
                'analysis': model_info,
                'supported_operations': self._get_supported_operations(model_name),
                'pruning_compatibility': self._assess_pruning_compatibility(model_name)
            }
            
            return full_info
            
        except Exception as e:
            logger.error(f"Failed to analyze model {model_name}: {e}")
            return config
    
    def is_supported(self, model_name: str) -> bool:
        """Check if a model architecture is supported."""
        return self.architecture_registry.is_registered(model_name)
    
    def list_supported_models(self) -> Dict[str, list]:
        """List all supported model architectures by category."""
        return self.architecture_registry.list_models()
    
    def download_model(self, model_name: str, cache_dir: Optional[str] = None) -> str:
        """
        Download pretrained model weights.
        
        Args:
            model_name: Name of the model architecture
            cache_dir: Directory to cache downloaded models
            
        Returns:
            Path to the downloaded model file
        """
        
        logger.info(f"📥 Downloading model: {model_name}")
        
        if not self.is_supported(model_name):
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Create model to trigger download
        model = self.create_model(model_name, pretrained=True)
        
        # For torchvision models, weights are automatically cached
        # For timm models, weights are also automatically cached
        
        # Return cache location (implementation depends on the library)
        if cache_dir:
            cache_path = Path(cache_dir) / f"{model_name}.pth"
            torch.save(model.state_dict(), cache_path)
            logger.info(f"✅ Model cached at: {cache_path}")
            return str(cache_path)
        else:
            logger.info(f"✅ Model {model_name} downloaded and cached by library")
            return "cached_by_library"
    
    def _create_torchvision_model(self, model_name: str, num_classes: int, 
                                 pretrained: bool, **kwargs) -> nn.Module:
        """Create model using torchvision."""
        
        # Map model names to torchvision functions
        torchvision_mapping = {
            'resnet50': torchvision_models.resnet50,
            'resnet101': torchvision_models.resnet101,
            'resnet152': torchvision_models.resnet152,
        }
        
        if model_name not in torchvision_mapping:
            raise ValueError(f"Model {model_name} not available in torchvision")
        
        model_fn = torchvision_mapping[model_name]
        
        # Handle different torchvision versions
        try:
            # New torchvision API (v0.13+)
            if pretrained:
                weights = 'DEFAULT'  # Use default pretrained weights
            else:
                weights = None
            model = model_fn(weights=weights, **kwargs)
        except TypeError:
            # Old torchvision API
            model = model_fn(pretrained=pretrained, **kwargs)
        
        # Adjust final layer for different number of classes
        if num_classes != 1000:
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Linear):
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                else:
                    # Handle more complex classifiers
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
        return model
    
    def _create_timm_model(self, model_name: str, num_classes: int, 
                          pretrained: bool, **kwargs) -> nn.Module:
        """Create model using timm."""
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for this model but not installed")
        
        # Create model using timm
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
        
        return model
    
    def _apply_model_config(self, model: nn.Module, model_name: str, 
                          config: Dict[str, Any]) -> nn.Module:
        """Apply model-specific configurations."""
        
        # Set model to evaluation mode by default
        model.eval()
        
        # Apply any model-specific modifications
        if 'modifications' in config:
            for modification in config['modifications']:
                model = self._apply_modification(model, modification)
        
        return model
    
    def _apply_modification(self, model: nn.Module, modification: Dict[str, Any]) -> nn.Module:
        """Apply a specific modification to the model."""
        
        mod_type = modification.get('type')
        
        if mod_type == 'dropout_adjustment':
            # Adjust dropout rates
            target_rate = modification.get('rate', 0.1)
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = target_rate
        
        elif mod_type == 'batch_norm_momentum':
            # Adjust batch norm momentum
            target_momentum = modification.get('momentum', 0.1)
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.momentum = target_momentum
        
        return model
    
    def _validate_model(self, model: nn.Module, model_name: str):
        """Validate that the model was created correctly."""
        
        # Basic validation
        if model is None:
            raise ValueError(f"Failed to create model {model_name}")
        
        # Check if model has parameters
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            raise ValueError(f"Model {model_name} has no parameters")
        
        # Test forward pass with dummy input
        try:
            model.eval()
            with torch.no_grad():
                # Get expected input shape from config
                config = self.architecture_registry.get_config(model_name)
                input_shape = config.get('input_shape', (1, 3, 224, 224))
                
                dummy_input = torch.randn(*input_shape)
                output = model(dummy_input)
                
                # Check output shape
                if output.dim() != 2:
                    logger.warning(f"Unexpected output shape for {model_name}: {output.shape}")
                
        except Exception as e:
            logger.warning(f"Forward pass validation failed for {model_name}: {e}")
    
    def _get_supported_operations(self, model_name: str) -> Dict[str, bool]:
        """Get supported operations for a model."""
        
        config = self.architecture_registry.get_config(model_name)
        architecture_type = config.get('type', 'unknown')
        
        # Define supported operations based on architecture type
        if architecture_type == 'cnn':
            return {
                'structured_pruning': True,
                'unstructured_pruning': True,
                'channel_pruning': True,
                'layer_pruning': True,
                'quantization': True,
                'knowledge_distillation': True
            }
        elif architecture_type == 'transformer':
            return {
                'structured_pruning': True,
                'unstructured_pruning': True,
                'attention_head_pruning': True,
                'layer_pruning': True,
                'quantization': True,
                'knowledge_distillation': True
            }
        else:
            return {
                'structured_pruning': True,
                'unstructured_pruning': True,
                'quantization': False,
                'knowledge_distillation': True
            }
    
    def _assess_pruning_compatibility(self, model_name: str) -> Dict[str, Any]:
        """Assess pruning compatibility for a model."""
        
        config = self.architecture_registry.get_config(model_name)
        architecture_type = config.get('type', 'unknown')
        
        compatibility = {
            'overall_compatibility': 'high',
            'recommended_methods': [],
            'constraints': [],
            'special_considerations': []
        }
        
        if architecture_type == 'cnn':
            compatibility.update({
                'recommended_methods': ['magnitude_pruning', 'taylor_pruning', 'channel_pruning'],
                'constraints': ['maintain_spatial_dimensions'],
                'special_considerations': ['batch_norm_adjustment', 'residual_connection_handling']
            })
        elif architecture_type == 'transformer':
            compatibility.update({
                'recommended_methods': ['attention_head_pruning', 'structured_pruning', 'layer_pruning'],
                'constraints': ['maintain_attention_dimensions', 'preserve_positional_encoding'],
                'special_considerations': ['layer_norm_adjustment', 'attention_pattern_preservation']
            })
        
        return compatibility
    
    def create_model_ensemble(self, model_configs: list) -> nn.ModuleList:
        """
        Create an ensemble of models.
        
        Args:
            model_configs: List of model configuration dictionaries
            
        Returns:
            ModuleList containing the ensemble models
        """
        
        logger.info(f"🏭 Creating model ensemble with {len(model_configs)} models")
        
        models = []
        for i, config in enumerate(model_configs):
            try:
                model = self.create_model(**config)
                models.append(model)
                logger.info(f"✅ Ensemble model {i+1}/{len(model_configs)} created")
            except Exception as e:
                logger.error(f"❌ Failed to create ensemble model {i+1}: {e}")
                raise
        
        ensemble = nn.ModuleList(models)
        logger.info(f"✅ Model ensemble created with {len(models)} models")
        
        return ensemble
    
    def compare_models(self, model_names: list) -> Dict[str, Any]:
        """
        Compare multiple model architectures.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Comparison results
        """
        
        logger.info(f"📊 Comparing {len(model_names)} models")
        
        comparison = {
            'models': {},
            'summary': {},
            'recommendations': []
        }
        
        for model_name in model_names:
            try:
                model_info = self.get_model_info(model_name)
                comparison['models'][model_name] = model_info
            except Exception as e:
                logger.error(f"Failed to get info for {model_name}: {e}")
                comparison['models'][model_name] = {'error': str(e)}
        
        # Generate comparison summary
        comparison['summary'] = self._generate_comparison_summary(comparison['models'])
        comparison['recommendations'] = self._generate_model_recommendations(comparison['models'])
        
        logger.info("📊 Model comparison completed")
        return comparison
    
    def _generate_comparison_summary(self, models_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of model comparison."""
        
        summary = {
            'total_models': len(models_info),
            'architecture_types': {},
            'parameter_range': {'min': float('inf'), 'max': 0},
            'complexity_distribution': {}
        }
        
        for model_name, info in models_info.items():
            if 'error' in info:
                continue
            
            # Count architecture types
            arch_type = info.get('type', 'unknown')
            summary['architecture_types'][arch_type] = summary['architecture_types'].get(arch_type, 0) + 1
            
            # Track parameter range
            if 'analysis' in info:
                params = info['analysis'].get('total_parameters', 0)
                summary['parameter_range']['min'] = min(summary['parameter_range']['min'], params)
                summary['parameter_range']['max'] = max(summary['parameter_range']['max'], params)
        
        return summary
    
    def _generate_model_recommendations(self, models_info: Dict[str, Any]) -> list:
        """Generate recommendations based on model comparison."""
        
        recommendations = []
        
        # Find models with best pruning compatibility
        high_compatibility_models = [
            name for name, info in models_info.items()
            if info.get('pruning_compatibility', {}).get('overall_compatibility') == 'high'
        ]
        
        if high_compatibility_models:
            recommendations.append(f"Models with high pruning compatibility: {', '.join(high_compatibility_models)}")
        
        # Find smallest and largest models
        param_counts = {}
        for name, info in models_info.items():
            if 'analysis' in info:
                param_counts[name] = info['analysis'].get('total_parameters', 0)
        
        if param_counts:
            smallest = min(param_counts, key=param_counts.get)
            largest = max(param_counts, key=param_counts.get)
            
            recommendations.append(f"Smallest model: {smallest} ({param_counts[smallest]:,} parameters)")
            recommendations.append(f"Largest model: {largest} ({param_counts[largest]:,} parameters)")
        
        return recommendations

