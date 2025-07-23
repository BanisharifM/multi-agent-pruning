#!/usr/bin/env python3
"""
Model Utilities for Multi-Agent LLM Pruning Framework

This module provides utility functions for model analysis, manipulation,
and common operations across different architectures.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class ModelUtils:
    """
    Utility class for common model operations and analysis.
    """
    
    def __init__(self):
        logger.info("🔧 Model Utils initialized")
    
    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a model.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary with detailed model analysis
        """
        
        logger.info("🔍 Analyzing model structure")
        
        analysis = {
            'total_parameters': self.count_parameters(model),
            'trainable_parameters': self.count_trainable_parameters(model),
            'model_size_mb': self.calculate_model_size(model),
            'layer_analysis': self.analyze_layers(model),
            'architecture_type': self.identify_architecture_type(model),
            'memory_requirements': self.estimate_memory_requirements(model),
            'computational_complexity': self.estimate_computational_complexity(model),
            'pruning_potential': self.assess_pruning_potential(model)
        }
        
        logger.info(f"✅ Model analysis completed: {analysis['total_parameters']:,} parameters")
        return analysis
    
    def count_parameters(self, model: nn.Module, only_trainable: bool = False) -> int:
        """Count total parameters in model."""
        
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    def count_trainable_parameters(self, model: nn.Module) -> int:
        """Count only trainable parameters."""
        return self.count_parameters(model, only_trainable=True)
    
    def calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb
    
    def analyze_layers(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze layer composition of the model."""
        
        layer_counts = defaultdict(int)
        layer_details = []
        total_layers = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_type = type(module).__name__
                layer_counts[layer_type] += 1
                total_layers += 1
                
                # Get layer details
                layer_info = {
                    'name': name,
                    'type': layer_type,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                # Add type-specific information
                if isinstance(module, nn.Linear):
                    layer_info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features,
                        'bias': module.bias is not None
                    })
                elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    layer_info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding
                    })
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    layer_info.update({
                        'num_features': module.num_features,
                        'eps': module.eps,
                        'momentum': module.momentum
                    })
                
                layer_details.append(layer_info)
        
        return {
            'total_layers': total_layers,
            'layer_counts': dict(layer_counts),
            'layer_details': layer_details,
            'dominant_layer_type': max(layer_counts, key=layer_counts.get) if layer_counts else 'unknown'
        }
    
    def identify_architecture_type(self, model: nn.Module) -> str:
        """Identify the general architecture type of the model."""
        
        layer_analysis = self.analyze_layers(model)
        layer_counts = layer_analysis['layer_counts']
        
        # Check for transformer characteristics
        if 'MultiheadAttention' in layer_counts or 'Attention' in str(type(model)):
            return 'transformer'
        
        # Check for CNN characteristics
        conv_layers = sum(count for layer_type, count in layer_counts.items() 
                         if 'Conv' in layer_type)
        linear_layers = layer_counts.get('Linear', 0)
        
        if conv_layers > linear_layers:
            return 'cnn'
        elif linear_layers > conv_layers and conv_layers == 0:
            return 'mlp'
        elif conv_layers > 0 and linear_layers > 0:
            return 'hybrid'
        else:
            return 'unknown'
    
    def estimate_memory_requirements(self, model: nn.Module, 
                                   input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, float]:
        """Estimate memory requirements for the model."""
        
        # Model parameters memory
        param_memory = self.calculate_model_size(model)
        
        # Estimate activation memory (rough approximation)
        try:
            model.eval()
            dummy_input = torch.randn(*input_shape)
            
            # Hook to capture intermediate activations
            activation_sizes = []
            
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    size_mb = output.numel() * output.element_size() / (1024 * 1024)
                    activation_sizes.append(size_mb)
            
            # Register hooks
            hooks = []
            for module in model.modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    hooks.append(module.register_forward_hook(hook_fn))
            
            # Forward pass to capture activations
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            activation_memory = sum(activation_sizes)
            
        except Exception as e:
            logger.warning(f"Failed to estimate activation memory: {e}")
            activation_memory = param_memory * 2  # Rough estimate
        
        return {
            'parameter_memory_mb': param_memory,
            'activation_memory_mb': activation_memory,
            'total_memory_mb': param_memory + activation_memory,
            'gradient_memory_mb': param_memory,  # Gradients same size as parameters
            'optimizer_memory_mb': param_memory * 2,  # Adam optimizer states
            'peak_training_memory_mb': param_memory * 4 + activation_memory
        }
    
    def estimate_computational_complexity(self, model: nn.Module,
                                        input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        """Estimate computational complexity (FLOPs/MACs)."""
        
        try:
            from ..utils.metrics import compute_macs
            macs = compute_macs(model, input_shape)
            
            return {
                'macs': macs,
                'gmacs': macs / 1e9,
                'estimated_inference_time_ms': macs / 1e6,  # Very rough estimate
                'complexity_category': self._categorize_complexity(macs)
            }
            
        except Exception as e:
            logger.warning(f"Failed to compute MACs: {e}")
            
            # Fallback: estimate based on parameters
            total_params = self.count_parameters(model)
            estimated_macs = total_params * 2  # Very rough estimate
            
            return {
                'macs': estimated_macs,
                'gmacs': estimated_macs / 1e9,
                'estimated_inference_time_ms': estimated_macs / 1e6,
                'complexity_category': self._categorize_complexity(estimated_macs),
                'note': 'Estimated based on parameters (MACs computation failed)'
            }
    
    def assess_pruning_potential(self, model: nn.Module) -> Dict[str, Any]:
        """Assess the pruning potential of the model."""
        
        layer_analysis = self.analyze_layers(model)
        layer_counts = layer_counts = layer_analysis['layer_counts']
        
        # Calculate pruning potential based on layer types
        prunable_layers = ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']
        total_prunable = sum(layer_counts.get(layer_type, 0) for layer_type in prunable_layers)
        total_layers = layer_analysis['total_layers']
        
        pruning_ratio = total_prunable / total_layers if total_layers > 0 else 0.0
        
        # Assess different pruning strategies
        strategies = {
            'unstructured_pruning': {
                'feasible': True,
                'potential_reduction': 0.5,  # Can typically remove 50% of weights
                'complexity': 'low'
            },
            'structured_pruning': {
                'feasible': total_prunable > 0,
                'potential_reduction': 0.3,  # More conservative for structured
                'complexity': 'medium'
            },
            'channel_pruning': {
                'feasible': any('Conv' in layer_type for layer_type in layer_counts),
                'potential_reduction': 0.4,
                'complexity': 'high'
            },
            'layer_pruning': {
                'feasible': total_layers > 10,  # Need sufficient layers
                'potential_reduction': 0.2,
                'complexity': 'high'
            }
        }
        
        return {
            'overall_pruning_potential': pruning_ratio,
            'prunable_layers': total_prunable,
            'total_layers': total_layers,
            'strategies': strategies,
            'recommended_strategy': self._recommend_pruning_strategy(layer_counts, total_layers),
            'estimated_speedup_range': (1.2, 2.5),  # Conservative estimate
            'risk_assessment': self._assess_pruning_risk(model)
        }
    
    def get_layer_hierarchy(self, model: nn.Module) -> Dict[str, Any]:
        """Get hierarchical structure of the model."""
        
        hierarchy = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                parts = name.split('.')
                current = hierarchy
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = {
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
        
        return hierarchy
    
    def find_similar_layers(self, model: nn.Module, similarity_threshold: float = 0.9) -> List[List[str]]:
        """Find groups of similar layers in the model."""
        
        layer_groups = []
        layer_info = {}
        
        # Collect layer information
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    layer_info[name] = {
                        'type': type(module).__name__,
                        'input_size': getattr(module, 'in_features', getattr(module, 'in_channels', None)),
                        'output_size': getattr(module, 'out_features', getattr(module, 'out_channels', None)),
                        'parameters': sum(p.numel() for p in module.parameters())
                    }
        
        # Group similar layers
        processed = set()
        for name1, info1 in layer_info.items():
            if name1 in processed:
                continue
            
            group = [name1]
            processed.add(name1)
            
            for name2, info2 in layer_info.items():
                if name2 in processed:
                    continue
                
                # Check similarity
                if (info1['type'] == info2['type'] and
                    info1['input_size'] == info2['input_size'] and
                    info1['output_size'] == info2['output_size']):
                    
                    # Check parameter count similarity
                    param_ratio = min(info1['parameters'], info2['parameters']) / max(info1['parameters'], info2['parameters'])
                    
                    if param_ratio >= similarity_threshold:
                        group.append(name2)
                        processed.add(name2)
            
            if len(group) > 1:
                layer_groups.append(group)
        
        return layer_groups
    
    def get_model_summary(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> str:
        """Generate a human-readable model summary."""
        
        analysis = self.analyze_model(model)
        
        summary = f"""
Model Summary:
==============
Architecture Type: {analysis['architecture_type']}
Total Parameters: {analysis['total_parameters']:,}
Trainable Parameters: {analysis['trainable_parameters']:,}
Model Size: {analysis['model_size_mb']:.2f} MB
Total Layers: {analysis['layer_analysis']['total_layers']}

Layer Composition:
"""
        
        for layer_type, count in analysis['layer_analysis']['layer_counts'].items():
            summary += f"  {layer_type}: {count}\n"
        
        summary += f"""
Memory Requirements:
  Parameters: {analysis['memory_requirements']['parameter_memory_mb']:.2f} MB
  Activations: {analysis['memory_requirements']['activation_memory_mb']:.2f} MB
  Peak Training: {analysis['memory_requirements']['peak_training_memory_mb']:.2f} MB

Computational Complexity:
  MACs: {analysis['computational_complexity']['macs']:,}
  GMACs: {analysis['computational_complexity']['gmacs']:.2f}
  Category: {analysis['computational_complexity']['complexity_category']}

Pruning Potential:
  Overall Potential: {analysis['pruning_potential']['overall_pruning_potential']:.1%}
  Recommended Strategy: {analysis['pruning_potential']['recommended_strategy']}
  Estimated Speedup: {analysis['pruning_potential']['estimated_speedup_range'][0]:.1f}x - {analysis['pruning_potential']['estimated_speedup_range'][1]:.1f}x
"""
        
        return summary
    
    def compare_models(self, model1: nn.Module, model2: nn.Module, 
                      model1_name: str = "Model 1", model2_name: str = "Model 2") -> Dict[str, Any]:
        """Compare two models."""
        
        analysis1 = self.analyze_model(model1)
        analysis2 = self.analyze_model(model2)
        
        comparison = {
            'models': {
                model1_name: analysis1,
                model2_name: analysis2
            },
            'differences': {
                'parameter_ratio': analysis1['total_parameters'] / analysis2['total_parameters'],
                'size_ratio': analysis1['model_size_mb'] / analysis2['model_size_mb'],
                'complexity_ratio': analysis1['computational_complexity']['macs'] / analysis2['computational_complexity']['macs']
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if comparison['differences']['parameter_ratio'] > 2:
            comparison['recommendations'].append(f"{model1_name} has significantly more parameters than {model2_name}")
        elif comparison['differences']['parameter_ratio'] < 0.5:
            comparison['recommendations'].append(f"{model2_name} has significantly more parameters than {model1_name}")
        
        if comparison['differences']['complexity_ratio'] > 2:
            comparison['recommendations'].append(f"{model1_name} is computationally more expensive than {model2_name}")
        elif comparison['differences']['complexity_ratio'] < 0.5:
            comparison['recommendations'].append(f"{model2_name} is computationally more expensive than {model1_name}")
        
        return comparison
    
    def extract_feature_extractor(self, model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
        """Extract feature extractor from a model."""
        
        if layer_name is None:
            # Try to find the last feature layer automatically
            layer_name = self._find_feature_layer(model)
        
        if layer_name is None:
            raise ValueError("Could not automatically determine feature layer. Please specify layer_name.")
        
        # Create feature extractor
        feature_extractor = nn.Sequential()
        
        found_target = False
        for name, module in model.named_children():
            feature_extractor.add_module(name, module)
            if name == layer_name:
                found_target = True
                break
        
        if not found_target:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        return feature_extractor
    
    def freeze_layers(self, model: nn.Module, layer_names: List[str]) -> nn.Module:
        """Freeze specified layers in the model."""
        
        frozen_count = 0
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        
        logger.info(f"Frozen {frozen_count} parameters in {len(layer_names)} layers")
        return model
    
    def unfreeze_layers(self, model: nn.Module, layer_names: List[str]) -> nn.Module:
        """Unfreeze specified layers in the model."""
        
        unfrozen_count = 0
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break
        
        logger.info(f"Unfrozen {unfrozen_count} parameters in {len(layer_names)} layers")
        return model
    
    # Helper methods
    def _categorize_complexity(self, macs: int) -> str:
        """Categorize computational complexity."""
        
        if macs < 1e8:  # < 100M MACs
            return 'low'
        elif macs < 1e9:  # < 1G MACs
            return 'medium'
        elif macs < 10e9:  # < 10G MACs
            return 'high'
        else:
            return 'very_high'
    
    def _recommend_pruning_strategy(self, layer_counts: Dict[str, int], total_layers: int) -> str:
        """Recommend pruning strategy based on model characteristics."""
        
        conv_layers = sum(count for layer_type, count in layer_counts.items() 
                         if 'Conv' in layer_type)
        linear_layers = layer_counts.get('Linear', 0)
        
        if conv_layers > linear_layers:
            return 'channel_pruning'
        elif linear_layers > conv_layers:
            return 'structured_pruning'
        elif total_layers > 20:
            return 'layer_pruning'
        else:
            return 'unstructured_pruning'
    
    def _assess_pruning_risk(self, model: nn.Module) -> str:
        """Assess risk level for pruning."""
        
        total_params = self.count_parameters(model)
        
        if total_params < 1e6:  # < 1M parameters
            return 'high'  # Small models are more sensitive
        elif total_params < 10e6:  # < 10M parameters
            return 'medium'
        else:
            return 'low'  # Large models are more robust to pruning
    
    def _find_feature_layer(self, model: nn.Module) -> Optional[str]:
        """Find the last feature extraction layer."""
        
        # Common patterns for feature layers
        feature_patterns = ['features', 'backbone', 'encoder', 'feature_extractor']
        classifier_patterns = ['classifier', 'fc', 'head', 'linear']
        
        # Get all module names
        module_names = [name for name, _ in model.named_children()]
        
        # Look for feature layer patterns
        for pattern in feature_patterns:
            for name in module_names:
                if pattern in name.lower():
                    return name
        
        # If no feature pattern found, return the layer before classifier
        for pattern in classifier_patterns:
            for i, name in enumerate(module_names):
                if pattern in name.lower() and i > 0:
                    return module_names[i-1]
        
        # Fallback: return second-to-last layer
        if len(module_names) > 1:
            return module_names[-2]
        
        return None

