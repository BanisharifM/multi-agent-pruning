#!/usr/bin/env python3
"""
Architecture Registry for Multi-Agent LLM Pruning Framework

This module maintains a registry of supported model architectures with their
configurations, capabilities, and metadata.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a model architecture."""
    name: str
    source: str  # 'torchvision', 'timm', 'custom'
    type: str   # 'cnn', 'transformer', 'hybrid', 'mlp'
    input_shape: tuple = (1, 3, 224, 224)
    num_classes: int = 1000
    pretrained_available: bool = True
    description: str = ""
    paper_url: str = ""
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    pruning_notes: str = ""
    
class ArchitectureRegistry:
    """
    Registry for managing supported model architectures and their configurations.
    """
    
    def __init__(self):
        self._registry = {}
        self._initialize_default_architectures()
        logger.info("🏛️ Architecture Registry initialized")
    
    def _initialize_default_architectures(self):
        """Initialize registry with default supported architectures."""
        
        # ResNet architectures
        self.register_architecture(ModelConfig(
            name='resnet50',
            source='torchvision',
            type='cnn',
            input_shape=(1, 3, 224, 224),
            description='ResNet-50 with 50 layers, widely used CNN architecture',
            paper_url='https://arxiv.org/abs/1512.03385',
            pruning_notes='Excellent for channel pruning, handles structured pruning well'
        ))
        
        self.register_architecture(ModelConfig(
            name='resnet101',
            source='torchvision',
            type='cnn',
            input_shape=(1, 3, 224, 224),
            description='ResNet-101 with 101 layers, deeper version of ResNet',
            paper_url='https://arxiv.org/abs/1512.03385',
            pruning_notes='Good candidate for layer pruning due to depth'
        ))
        
        self.register_architecture(ModelConfig(
            name='resnet152',
            source='torchvision',
            type='cnn',
            input_shape=(1, 3, 224, 224),
            description='ResNet-152 with 152 layers, deepest standard ResNet',
            paper_url='https://arxiv.org/abs/1512.03385',
            pruning_notes='Excellent for layer pruning, can handle aggressive pruning'
        ))
        
        # Vision Transformer architectures (timm)
        if self._check_timm_availability():
            self.register_architecture(ModelConfig(
                name='vit_base_patch16_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Vision Transformer Base with 16x16 patches',
                paper_url='https://arxiv.org/abs/2010.11929',
                pruning_notes='Attention head pruning works well, structured pruning requires care'
            ))
            
            self.register_architecture(ModelConfig(
                name='vit_small_patch16_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Vision Transformer Small with 16x16 patches',
                paper_url='https://arxiv.org/abs/2010.11929',
                pruning_notes='More sensitive to pruning than base model'
            ))
            
            self.register_architecture(ModelConfig(
                name='vit_large_patch16_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Vision Transformer Large with 16x16 patches',
                paper_url='https://arxiv.org/abs/2010.11929',
                pruning_notes='Robust to pruning due to size, good for aggressive pruning'
            ))
            
            # DeiT architectures
            self.register_architecture(ModelConfig(
                name='deit_base_patch16_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Data-efficient Image Transformer Base',
                paper_url='https://arxiv.org/abs/2012.12877',
                pruning_notes='Knowledge distillation friendly, good for structured pruning'
            ))
            
            self.register_architecture(ModelConfig(
                name='deit_small_patch16_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Data-efficient Image Transformer Small',
                paper_url='https://arxiv.org/abs/2012.12877',
                pruning_notes='Compact model, moderate pruning recommended'
            ))
            
            self.register_architecture(ModelConfig(
                name='deit_tiny_patch16_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Data-efficient Image Transformer Tiny',
                paper_url='https://arxiv.org/abs/2012.12877',
                pruning_notes='Very small model, conservative pruning recommended'
            ))
            
            # Swin Transformer
            self.register_architecture(ModelConfig(
                name='swin_base_patch4_window7_224',
                source='timm',
                type='transformer',
                input_shape=(1, 3, 224, 224),
                description='Swin Transformer Base with hierarchical structure',
                paper_url='https://arxiv.org/abs/2103.14030',
                pruning_notes='Hierarchical structure allows for sophisticated pruning strategies'
            ))
            
            # EfficientNet
            self.register_architecture(ModelConfig(
                name='efficientnet_b0',
                source='timm',
                type='cnn',
                input_shape=(1, 3, 224, 224),
                description='EfficientNet B0 - compound scaling baseline',
                paper_url='https://arxiv.org/abs/1905.11946',
                pruning_notes='Already optimized for efficiency, careful pruning needed'
            ))
            
            self.register_architecture(ModelConfig(
                name='efficientnet_b4',
                source='timm',
                type='cnn',
                input_shape=(1, 3, 380, 380),
                description='EfficientNet B4 - larger input resolution',
                paper_url='https://arxiv.org/abs/1905.11946',
                pruning_notes='Good balance of size and performance for pruning'
            ))
        
        logger.info(f"✅ Initialized {len(self._registry)} default architectures")
    
    def register_architecture(self, config: ModelConfig):
        """Register a new architecture configuration."""
        
        self._registry[config.name] = config
        logger.debug(f"Registered architecture: {config.name}")
    
    def get_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a model architecture."""
        
        if model_name not in self._registry:
            raise ValueError(f"Architecture '{model_name}' not registered")
        
        config = self._registry[model_name]
        return {
            'name': config.name,
            'source': config.source,
            'type': config.type,
            'input_shape': config.input_shape,
            'num_classes': config.num_classes,
            'pretrained_available': config.pretrained_available,
            'description': config.description,
            'paper_url': config.paper_url,
            'modifications': config.modifications,
            'pruning_notes': config.pruning_notes
        }
    
    def is_registered(self, model_name: str) -> bool:
        """Check if a model architecture is registered."""
        return model_name in self._registry
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models by category."""
        
        categories = {
            'cnn': [],
            'transformer': [],
            'hybrid': [],
            'mlp': [],
            'other': []
        }
        
        for name, config in self._registry.items():
            category = config.type if config.type in categories else 'other'
            categories[category].append(name)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def get_models_by_source(self, source: str) -> List[str]:
        """Get all models from a specific source."""
        
        return [name for name, config in self._registry.items() 
                if config.source == source]
    
    def get_models_by_type(self, model_type: str) -> List[str]:
        """Get all models of a specific type."""
        
        return [name for name, config in self._registry.items() 
                if config.type == model_type]
    
    def search_models(self, query: str) -> List[str]:
        """Search for models by name or description."""
        
        query = query.lower()
        matches = []
        
        for name, config in self._registry.items():
            if (query in name.lower() or 
                query in config.description.lower()):
                matches.append(name)
        
        return matches
    
    def get_pruning_recommendations(self, model_name: str) -> Dict[str, Any]:
        """Get pruning recommendations for a specific model."""
        
        if model_name not in self._registry:
            raise ValueError(f"Architecture '{model_name}' not registered")
        
        config = self._registry[model_name]
        
        # Base recommendations by type
        type_recommendations = {
            'cnn': {
                'recommended_methods': ['channel_pruning', 'structured_pruning', 'magnitude_pruning'],
                'safe_pruning_ratio': 0.3,
                'aggressive_pruning_ratio': 0.5,
                'considerations': ['Maintain spatial dimensions', 'Handle batch normalization']
            },
            'transformer': {
                'recommended_methods': ['attention_head_pruning', 'structured_pruning', 'layer_pruning'],
                'safe_pruning_ratio': 0.25,
                'aggressive_pruning_ratio': 0.4,
                'considerations': ['Preserve attention patterns', 'Maintain sequence length compatibility']
            },
            'hybrid': {
                'recommended_methods': ['mixed_strategy', 'structured_pruning'],
                'safe_pruning_ratio': 0.2,
                'aggressive_pruning_ratio': 0.35,
                'considerations': ['Different strategies for different components']
            },
            'mlp': {
                'recommended_methods': ['structured_pruning', 'magnitude_pruning'],
                'safe_pruning_ratio': 0.4,
                'aggressive_pruning_ratio': 0.6,
                'considerations': ['Simple architecture allows aggressive pruning']
            }
        }
        
        base_rec = type_recommendations.get(config.type, type_recommendations['cnn'])
        
        # Model-specific adjustments
        model_specific = self._get_model_specific_recommendations(model_name, config)
        
        # Combine recommendations
        recommendations = {
            **base_rec,
            'model_specific_notes': config.pruning_notes,
            'model_specific_adjustments': model_specific
        }
        
        return recommendations
    
    def compare_architectures(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple architectures."""
        
        if not all(self.is_registered(name) for name in model_names):
            missing = [name for name in model_names if not self.is_registered(name)]
            raise ValueError(f"Unregistered architectures: {missing}")
        
        comparison = {
            'models': {},
            'summary': {
                'types': set(),
                'sources': set(),
                'input_shapes': set(),
                'pruning_difficulty': {}
            }
        }
        
        for name in model_names:
            config = self._registry[name]
            comparison['models'][name] = {
                'type': config.type,
                'source': config.source,
                'input_shape': config.input_shape,
                'description': config.description,
                'pruning_recommendations': self.get_pruning_recommendations(name)
            }
            
            # Update summary
            comparison['summary']['types'].add(config.type)
            comparison['summary']['sources'].add(config.source)
            comparison['summary']['input_shapes'].add(config.input_shape)
        
        # Convert sets to lists for JSON serialization
        comparison['summary']['types'] = list(comparison['summary']['types'])
        comparison['summary']['sources'] = list(comparison['summary']['sources'])
        comparison['summary']['input_shapes'] = list(comparison['summary']['input_shapes'])
        
        return comparison
    
    def get_paper_reproduction_models(self) -> Dict[str, List[str]]:
        """Get models commonly used in pruning papers for reproduction."""
        
        paper_models = {
            'isomorphic_pruning_paper': [
                'resnet50',
                'deit_small_patch16_224',
                'deit_base_patch16_224'
            ],
            'vision_transformer_pruning': [
                'vit_base_patch16_224',
                'vit_small_patch16_224',
                'deit_base_patch16_224'
            ],
            'cnn_pruning_benchmarks': [
                'resnet50',
                'resnet101',
                'efficientnet_b0'
            ]
        }
        
        # Filter to only include registered models
        filtered_models = {}
        for paper, models in paper_models.items():
            filtered_models[paper] = [m for m in models if self.is_registered(m)]
        
        return filtered_models
    
    def export_registry(self) -> Dict[str, Any]:
        """Export the entire registry for backup or sharing."""
        
        export_data = {}
        for name, config in self._registry.items():
            export_data[name] = {
                'name': config.name,
                'source': config.source,
                'type': config.type,
                'input_shape': config.input_shape,
                'num_classes': config.num_classes,
                'pretrained_available': config.pretrained_available,
                'description': config.description,
                'paper_url': config.paper_url,
                'modifications': config.modifications,
                'pruning_notes': config.pruning_notes
            }
        
        return export_data
    
    def import_registry(self, registry_data: Dict[str, Any]):
        """Import registry data from external source."""
        
        imported_count = 0
        for name, config_dict in registry_data.items():
            try:
                config = ModelConfig(**config_dict)
                self.register_architecture(config)
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import architecture {name}: {e}")
        
        logger.info(f"Imported {imported_count} architectures")
    
    def _check_timm_availability(self) -> bool:
        """Check if timm library is available."""
        try:
            import timm
            return True
        except ImportError:
            logger.warning("timm not available - some models will not be registered")
            return False
    
    def _get_model_specific_recommendations(self, model_name: str, config: ModelConfig) -> Dict[str, Any]:
        """Get model-specific pruning recommendations."""
        
        adjustments = {}
        
        # Size-based adjustments
        if 'tiny' in model_name.lower():
            adjustments['pruning_ratio_multiplier'] = 0.7  # More conservative
            adjustments['note'] = 'Tiny model - use conservative pruning'
        elif 'large' in model_name.lower():
            adjustments['pruning_ratio_multiplier'] = 1.3  # More aggressive
            adjustments['note'] = 'Large model - can handle aggressive pruning'
        
        # Architecture-specific adjustments
        if 'efficientnet' in model_name.lower():
            adjustments['special_consideration'] = 'Already optimized for efficiency'
            adjustments['recommended_approach'] = 'knowledge_distillation'
        
        if 'swin' in model_name.lower():
            adjustments['special_consideration'] = 'Hierarchical structure'
            adjustments['recommended_approach'] = 'stage_aware_pruning'
        
        return adjustments
    
    def get_model_families(self) -> Dict[str, List[str]]:
        """Group models by family (ResNet, ViT, etc.)."""
        
        families = {}
        
        for name in self._registry.keys():
            # Extract family name
            if name.startswith('resnet'):
                family = 'ResNet'
            elif name.startswith('vit'):
                family = 'Vision Transformer'
            elif name.startswith('deit'):
                family = 'DeiT'
            elif name.startswith('swin'):
                family = 'Swin Transformer'
            elif name.startswith('efficientnet'):
                family = 'EfficientNet'
            else:
                family = 'Other'
            
            if family not in families:
                families[family] = []
            families[family].append(name)
        
        return families
    
    def validate_registry(self) -> Dict[str, Any]:
        """Validate the registry for consistency and completeness."""
        
        validation_results = {
            'total_models': len(self._registry),
            'issues': [],
            'warnings': [],
            'statistics': {
                'by_type': {},
                'by_source': {},
                'with_pretrained': 0,
                'with_pruning_notes': 0
            }
        }
        
        # Collect statistics
        for name, config in self._registry.items():
            # Type statistics
            validation_results['statistics']['by_type'][config.type] = \
                validation_results['statistics']['by_type'].get(config.type, 0) + 1
            
            # Source statistics
            validation_results['statistics']['by_source'][config.source] = \
                validation_results['statistics']['by_source'].get(config.source, 0) + 1
            
            # Feature statistics
            if config.pretrained_available:
                validation_results['statistics']['with_pretrained'] += 1
            
            if config.pruning_notes:
                validation_results['statistics']['with_pruning_notes'] += 1
            
            # Validation checks
            if not config.description:
                validation_results['warnings'].append(f"{name}: Missing description")
            
            if not config.paper_url:
                validation_results['warnings'].append(f"{name}: Missing paper URL")
            
            if not config.pruning_notes:
                validation_results['warnings'].append(f"{name}: Missing pruning notes")
            
            if config.input_shape[0] != 1:
                validation_results['issues'].append(f"{name}: Batch size in input_shape should be 1")
        
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        
        return validation_results

