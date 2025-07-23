#!/usr/bin/env python3
"""
Isomorphic Analyzer for Multi-Agent LLM Pruning Framework

This module provides isomorphic grouping analysis for neural networks,
identifying layers with similar structures that can be pruned together
while maintaining architectural consistency.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from .dependency_analyzer import DependencyAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class IsomorphicGroup:
    """Represents an isomorphic group of layers with similar structure."""
    name: str
    layers: List[nn.Module]
    layer_names: List[str]
    dimensions: List[int]
    pruning_ratio: float
    constraints: List[str]
    group_type: str  # 'mlp', 'attention', 'conv', 'residual'
    
    def __post_init__(self):
        """Validate group consistency after initialization."""
        if len(self.layers) != len(self.layer_names):
            raise ValueError("Number of layers must match number of layer names")
    
    def get_total_parameters(self) -> int:
        """Calculate total parameters in this isomorphic group."""
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'weight') and layer.weight is not None:
                total += layer.weight.numel()
            if hasattr(layer, 'bias') and layer.bias is not None:
                total += layer.bias.numel()
        return total
    
    def validate_isomorphism(self) -> bool:
        """Validate that all layers in the group are truly isomorphic."""
        if not self.layers:
            return True
        
        reference_layer = self.layers[0]
        reference_type = type(reference_layer)
        
        for layer in self.layers[1:]:
            if type(layer) != reference_type:
                return False
            
            # Check dimension compatibility
            if isinstance(layer, nn.Linear):
                if (layer.in_features != reference_layer.in_features or 
                    layer.out_features != reference_layer.out_features):
                    return False
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                if (layer.in_channels != reference_layer.in_channels or 
                    layer.out_channels != reference_layer.out_channels or
                    layer.kernel_size != reference_layer.kernel_size):
                    return False
        
        return True

@dataclass
class MLPCouple:
    """Represents a coupled MLP fc1+fc2 pair that must be pruned together."""
    fc1: nn.Linear
    fc2: nn.Linear
    fc1_name: str
    fc2_name: str
    
    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the MLP."""
        return self.fc1.out_features
    
    def validate_coupling(self) -> bool:
        """Validate that fc1 output matches fc2 input."""
        return self.fc1.out_features == self.fc2.in_features
    
    def get_total_parameters(self) -> int:
        """Get total parameters in the MLP couple."""
        total = self.fc1.weight.numel() + self.fc2.weight.numel()
        if self.fc1.bias is not None:
            total += self.fc1.bias.numel()
        if self.fc2.bias is not None:
            total += self.fc2.bias.numel()
        return total

@dataclass  
class AttentionCouple:
    """Represents a coupled QKV+Proj pair that must be pruned together."""
    qkv: nn.Linear
    proj: nn.Linear
    qkv_name: str
    proj_name: str
    
    def get_embed_dim(self) -> int:
        """Get the embedding dimension."""
        return self.qkv.out_features // 3  # QKV is 3x embed_dim
    
    def validate_coupling(self) -> bool:
        """Validate that QKV output is compatible with projection input."""
        return self.qkv.out_features // 3 == self.proj.in_features
    
    def get_total_parameters(self) -> int:
        """Get total parameters in the attention couple."""
        total = self.qkv.weight.numel() + self.proj.weight.numel()
        if self.qkv.bias is not None:
            total += self.qkv.bias.numel()
        if self.proj.bias is not None:
            total += self.proj.bias.numel()
        return total

class IsomorphicAnalyzer:
    """
    Enhanced analyzer for creating isomorphic groups with dependency awareness.
    
    This analyzer identifies groups of layers that have similar structures and
    can be pruned together while respecting dependency constraints.
    """
    
    def __init__(self, model: nn.Module, model_name: Optional[str] = None):
        self.model = model
        self.model_name = model_name or "unknown_model"
        self.dependency_analyzer = DependencyAnalyzer(model, model_name)
        
        # Architecture detection
        self.architecture_type = self._detect_architecture_type()
        
        logger.info(f"🔍 Isomorphic analysis initialized for {self.model_name}")
        logger.info(f"   Detected architecture: {self.architecture_type}")
    
    def _detect_architecture_type(self) -> str:
        """Detect the type of neural network architecture."""
        
        layer_types = []
        layer_names = []
        
        for name, module in self.model.named_modules():
            layer_types.append(type(module).__name__)
            layer_names.append(name.lower())
        
        # Vision Transformer detection
        vit_indicators = ['blocks.', 'attn', 'mlp', 'qkv', 'proj']
        if any(indicator in ' '.join(layer_names) for indicator in vit_indicators):
            return 'vision_transformer'
        
        # CNN detection
        if 'Conv2d' in layer_types or 'Conv1d' in layer_types:
            return 'cnn'
        
        # MLP detection
        if 'Linear' in layer_types and 'Conv2d' not in layer_types:
            return 'mlp'
        
        return 'unknown'
    
    def create_isomorphic_groups(self, target_ratio: float, 
                               group_ratios: Optional[Dict[str, float]] = None) -> Dict[str, IsomorphicGroup]:
        """Create dependency-aware isomorphic groups."""
        
        if self.architecture_type == 'vision_transformer':
            return self._create_vit_isomorphic_groups(target_ratio, group_ratios)
        elif self.architecture_type == 'cnn':
            return self._create_cnn_isomorphic_groups(target_ratio, group_ratios)
        else:
            return self._create_generic_isomorphic_groups(target_ratio, group_ratios)
    
    def _create_vit_isomorphic_groups(self, target_ratio: float, 
                                    group_ratios: Optional[Dict[str, float]] = None) -> Dict[str, IsomorphicGroup]:
        """Create isomorphic groups for Vision Transformer architectures."""
        
        # Default ratios for ViT components
        default_ratios = {
            'qkv_multiplier': 0.4,      # Conservative for attention
            'mlp_multiplier': 1.0,      # Full ratio for MLP
            'proj_multiplier': 0.0,     # Don't prune projections
            'head_multiplier': 0.0      # Don't prune classification head
        }
        
        ratios = group_ratios if group_ratios else default_ratios
        
        # Find transformer blocks
        transformer_blocks = self._find_transformer_blocks()
        
        groups = {
            'mlp_blocks': IsomorphicGroup(
                name='MLP Blocks (Coupled)',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("mlp_multiplier", 1.0),
                constraints=['fc1.out_features == fc2.in_features'],
                group_type='mlp'
            ),
            'attention_blocks': IsomorphicGroup(
                name='Attention Blocks (Coupled)', 
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("qkv_multiplier", 0.4),
                constraints=['qkv.out_features == proj.in_features * 3'],
                group_type='attention'
            ),
            'output_projections': IsomorphicGroup(
                name='Output Projections',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("proj_multiplier", 0.0),
                constraints=['Preserve residual connections'],
                group_type='projection'
            ),
            'classification_head': IsomorphicGroup(
                name='Classification Head',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("head_multiplier", 0.0),
                constraints=['Preserve output dimensions'],
                group_type='classification'
            )
        }
        
        # Populate groups with coupled layers
        for block_idx, block_layers in transformer_blocks.items():
            
            # Add MLP couples (fc1 + fc2 together)
            if 'fc1' in block_layers and 'fc2' in block_layers:
                mlp_couple = MLPCouple(
                    fc1=block_layers['fc1']['layer'],
                    fc2=block_layers['fc2']['layer'],
                    fc1_name=block_layers['fc1']['name'],
                    fc2_name=block_layers['fc2']['name']
                )
                
                if mlp_couple.validate_coupling():
                    groups['mlp_blocks'].layers.append(mlp_couple)
                    groups['mlp_blocks'].layer_names.append(f"block_{block_idx}_mlp")
                    groups['mlp_blocks'].dimensions.append(mlp_couple.get_hidden_dim())
                else:
                    logger.warning(f"MLP coupling validation failed for block {block_idx}")
            
            # Add Attention couples (qkv + proj together)
            if 'qkv' in block_layers and 'proj' in block_layers:
                attn_couple = AttentionCouple(
                    qkv=block_layers['qkv']['layer'],
                    proj=block_layers['proj']['layer'], 
                    qkv_name=block_layers['qkv']['name'],
                    proj_name=block_layers['proj']['name']
                )
                
                if attn_couple.validate_coupling():
                    groups['attention_blocks'].layers.append(attn_couple)
                    groups['attention_blocks'].layer_names.append(f"block_{block_idx}_attn")
                    groups['attention_blocks'].dimensions.append(attn_couple.get_embed_dim())
                else:
                    logger.warning(f"Attention coupling validation failed for block {block_idx}")
        
        # Add classification head
        head_layers = self._find_classification_layers()
        for layer_name, layer in head_layers.items():
            groups['classification_head'].layers.append(layer)
            groups['classification_head'].layer_names.append(layer_name)
            if isinstance(layer, nn.Linear):
                groups['classification_head'].dimensions.append(layer.in_features)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v.layers}
        
        # Validate all groups
        for group_name, group in groups.items():
            if not group.validate_isomorphism():
                logger.warning(f"Isomorphism validation failed for group: {group_name}")
        
        logger.info(f"✅ Created {len(groups)} isomorphic groups for ViT")
        for group_name, group in groups.items():
            logger.info(f"   {group_name}: {len(group.layers)} components, "
                       f"ratio={group.pruning_ratio:.3f}")
        
        return groups
    
    def _create_cnn_isomorphic_groups(self, target_ratio: float, 
                                    group_ratios: Optional[Dict[str, float]] = None) -> Dict[str, IsomorphicGroup]:
        """Create isomorphic groups for CNN architectures."""
        
        default_ratios = {
            'conv_multiplier': 0.8,     # Moderate pruning for conv layers
            'fc_multiplier': 1.0,       # Full pruning for FC layers
            'head_multiplier': 0.0      # Don't prune classification head
        }
        
        ratios = group_ratios if group_ratios else default_ratios
        
        groups = {
            'conv_layers': IsomorphicGroup(
                name='Convolutional Layers',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("conv_multiplier", 0.8),
                constraints=['Channel dimension compatibility'],
                group_type='conv'
            ),
            'fc_layers': IsomorphicGroup(
                name='Fully Connected Layers',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("fc_multiplier", 1.0),
                constraints=['Feature dimension compatibility'],
                group_type='fc'
            ),
            'classification_head': IsomorphicGroup(
                name='Classification Head',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * ratios.get("head_multiplier", 0.0),
                constraints=['Preserve output dimensions'],
                group_type='classification'
            )
        }
        
        # Classify layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                groups['conv_layers'].layers.append(module)
                groups['conv_layers'].layer_names.append(name)
                groups['conv_layers'].dimensions.append(module.out_channels)
            elif isinstance(module, nn.Linear):
                if 'head' in name.lower() or 'classifier' in name.lower():
                    groups['classification_head'].layers.append(module)
                    groups['classification_head'].layer_names.append(name)
                    groups['classification_head'].dimensions.append(module.in_features)
                else:
                    groups['fc_layers'].layers.append(module)
                    groups['fc_layers'].layer_names.append(name)
                    groups['fc_layers'].dimensions.append(module.in_features)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v.layers}
        
        logger.info(f"✅ Created {len(groups)} isomorphic groups for CNN")
        for group_name, group in groups.items():
            logger.info(f"   {group_name}: {len(group.layers)} layers, "
                       f"ratio={group.pruning_ratio:.3f}")
        
        return groups
    
    def _create_generic_isomorphic_groups(self, target_ratio: float, 
                                        group_ratios: Optional[Dict[str, float]] = None) -> Dict[str, IsomorphicGroup]:
        """Create generic isomorphic groups for unknown architectures."""
        
        groups = {
            'linear_layers': IsomorphicGroup(
                name='Linear Layers',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio,
                constraints=['Dimension compatibility'],
                group_type='linear'
            ),
            'conv_layers': IsomorphicGroup(
                name='Convolutional Layers',
                layers=[],
                layer_names=[],
                dimensions=[],
                pruning_ratio=target_ratio * 0.8,  # Conservative for conv
                constraints=['Channel compatibility'],
                group_type='conv'
            )
        }
        
        # Classify all layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                groups['linear_layers'].layers.append(module)
                groups['linear_layers'].layer_names.append(name)
                groups['linear_layers'].dimensions.append(module.in_features)
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                groups['conv_layers'].layers.append(module)
                groups['conv_layers'].layer_names.append(name)
                groups['conv_layers'].dimensions.append(module.out_channels)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v.layers}
        
        logger.info(f"✅ Created {len(groups)} generic isomorphic groups")
        
        return groups
    
    def _find_transformer_blocks(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Find and organize transformer blocks."""
        
        blocks = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'blocks.' in name:
                # Parse: "blocks.0.mlp.fc1" -> block=0, component=mlp, layer=fc1
                try:
                    parts = name.split('.')
                    block_idx = int(parts[1])
                    component = parts[2]  # 'mlp' or 'attn'
                    layer_type = parts[3]  # 'fc1', 'fc2', 'qkv', 'proj'
                    
                    if block_idx not in blocks:
                        blocks[block_idx] = {}
                    
                    blocks[block_idx][layer_type] = {
                        'layer': module,
                        'name': name,
                        'component': component
                    }
                except (IndexError, ValueError) as e:
                    logger.debug(f"Failed to parse transformer block name: {name}, error: {e}")
                    continue
        
        return blocks
    
    def _find_classification_layers(self) -> Dict[str, nn.Module]:
        """Find classification/head layers."""
        
        head_layers = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(keyword in name.lower() for keyword in ['head', 'classifier', 'fc']):
                    # Check if it's likely a classification layer (output to num_classes)
                    if module.out_features <= 10000:  # Reasonable number of classes
                        head_layers[name] = module
        
        return head_layers
    
    def get_group_statistics(self, groups: Dict[str, IsomorphicGroup]) -> Dict[str, Any]:
        """Get comprehensive statistics about the isomorphic groups."""
        
        stats = {
            'total_groups': len(groups),
            'total_layers': 0,
            'total_parameters': 0,
            'group_details': {},
            'pruning_summary': {}
        }
        
        for group_name, group in groups.items():
            group_params = group.get_total_parameters()
            stats['total_layers'] += len(group.layers)
            stats['total_parameters'] += group_params
            
            stats['group_details'][group_name] = {
                'num_layers': len(group.layers),
                'parameters': group_params,
                'pruning_ratio': group.pruning_ratio,
                'group_type': group.group_type,
                'constraints': group.constraints
            }
            
            # Calculate parameters to be pruned
            pruned_params = int(group_params * group.pruning_ratio)
            stats['pruning_summary'][group_name] = {
                'original_params': group_params,
                'pruned_params': pruned_params,
                'remaining_params': group_params - pruned_params,
                'reduction_ratio': group.pruning_ratio
            }
        
        return stats
    
    def validate_groups(self, groups: Dict[str, IsomorphicGroup]) -> Dict[str, Any]:
        """Validate that all isomorphic groups are consistent and safe to prune."""
        
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        for group_name, group in groups.items():
            # Check isomorphism
            if not group.validate_isomorphism():
                validation_result['issues'].append(
                    f"Group {group_name} failed isomorphism validation"
                )
                validation_result['valid'] = False
            
            # Check for empty groups
            if not group.layers:
                validation_result['warnings'].append(
                    f"Group {group_name} is empty"
                )
            
            # Check for extreme pruning ratios
            if group.pruning_ratio > 0.9:
                validation_result['warnings'].append(
                    f"Group {group_name} has very high pruning ratio: {group.pruning_ratio:.1%}"
                )
            
            # Validate coupled layers (for ViT)
            if group.group_type in ['mlp', 'attention']:
                for layer in group.layers:
                    if isinstance(layer, (MLPCouple, AttentionCouple)):
                        if not layer.validate_coupling():
                            validation_result['issues'].append(
                                f"Coupling validation failed in group {group_name}"
                            )
                            validation_result['valid'] = False
        
        # Check dependency constraints
        dependency_validation = self.dependency_analyzer.validate_pruning_compatibility(
            {name: group.pruning_ratio for name, group in groups.items()}
        )
        
        if not dependency_validation['valid']:
            validation_result['issues'].extend(dependency_validation['violations'])
            validation_result['recommendations'].extend(dependency_validation['recommendations'])
            validation_result['valid'] = False
        
        return validation_result
    
    def print_groups_summary(self, groups: Dict[str, IsomorphicGroup]):
        """Print a comprehensive summary of the isomorphic groups."""
        
        print(f"\n🔍 Isomorphic Groups Summary for {self.model_name}")
        print("=" * 60)
        
        stats = self.get_group_statistics(groups)
        
        print(f"📊 Overall Statistics:")
        print(f"   Architecture type: {self.architecture_type}")
        print(f"   Total groups: {stats['total_groups']}")
        print(f"   Total layers: {stats['total_layers']}")
        print(f"   Total parameters: {stats['total_parameters']:,}")
        
        print(f"\n📋 Group Details:")
        for group_name, group in groups.items():
            details = stats['group_details'][group_name]
            pruning = stats['pruning_summary'][group_name]
            
            print(f"   {group_name}:")
            print(f"      Type: {details['group_type']}")
            print(f"      Layers: {details['num_layers']}")
            print(f"      Parameters: {details['parameters']:,}")
            print(f"      Pruning ratio: {details['pruning_ratio']:.1%}")
            print(f"      Will remove: {pruning['pruned_params']:,} params")
            print(f"      Constraints: {', '.join(details['constraints'])}")
        
        # Validation
        validation = self.validate_groups(groups)
        print(f"\n✅ Validation:")
        print(f"   Status: {'PASSED' if validation['valid'] else 'FAILED'}")
        
        if validation['issues']:
            print(f"   Issues: {len(validation['issues'])}")
            for issue in validation['issues']:
                print(f"      - {issue}")
        
        if validation['warnings']:
            print(f"   Warnings: {len(validation['warnings'])}")
            for warning in validation['warnings']:
                print(f"      - {warning}")
        
        print("=" * 60)

