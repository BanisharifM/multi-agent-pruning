#!/usr/bin/env python3
"""
Dependency Analyzer for Multi-Agent LLM Pruning Framework

This module provides comprehensive dependency analysis for neural network layers,
detecting coupling constraints and building dependency graphs to ensure safe pruning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LayerDependency:
    """Represents a dependency relationship between layers."""
    source_layer: str
    target_layer: str
    dependency_type: str  # 'dimension_match', 'residual_connection', 'attention_coupling'
    constraint: str
    critical: bool = True

@dataclass
class CouplingConstraint:
    """Represents a coupling constraint between layers that must be pruned together."""
    layer_group: List[str]
    constraint_type: str  # 'mlp_coupling', 'attention_coupling', 'residual_coupling'
    description: str
    dimensions_affected: List[str]  # ['in_features', 'out_features', 'hidden_dim']

class DependencyAnalyzer:
    """
    Detects layer dependencies and coupling constraints to prevent dimension mismatches
    during pruning operations.
    """
    
    def __init__(self, model: nn.Module, model_name: Optional[str] = None):
        self.model = model
        self.model_name = model_name or "unknown_model"
        self.dependency_graph: Dict[str, Dict[str, Any]] = {}
        self.coupling_constraints: List[CouplingConstraint] = []
        self.layer_dependencies: List[LayerDependency] = []
        
        # Build dependency analysis
        self._build_dependency_graph()
        self._detect_coupling_constraints()
        
        logger.info(f"🔗 Dependency analysis completed for {self.model_name}")
        logger.info(f"   Found {len(self.dependency_graph)} layers with dependencies")
        logger.info(f"   Detected {len(self.coupling_constraints)} coupling constraints")
    
    def _build_dependency_graph(self):
        """Build comprehensive dependency graph between layers."""
        
        layers = list(self.model.named_modules())
        
        for i, (name1, layer1) in enumerate(layers):
            if not self._is_prunable_layer(layer1):
                continue
                
            self.dependency_graph[name1] = {
                'layer': layer1,
                'layer_type': type(layer1).__name__,
                'depends_on': [],
                'dependents': [],
                'dimensions': self._get_layer_dimensions(layer1),
                'block_info': self._parse_layer_location(name1)
            }
            
            # Find layers this one depends on
            for j, (name2, layer2) in enumerate(layers):
                if i != j and self._are_connected(name1, name2, layer1, layer2):
                    dependency = self._analyze_connection(name1, name2, layer1, layer2)
                    
                    self.dependency_graph[name1]['depends_on'].append({
                        'layer_name': name2,
                        'dependency_type': dependency['type'],
                        'constraint': dependency['constraint']
                    })
                    
                    if name2 in self.dependency_graph:
                        self.dependency_graph[name2]['dependents'].append({
                            'layer_name': name1,
                            'dependency_type': dependency['type'],
                            'constraint': dependency['constraint']
                        })
                    
                    # Store dependency relationship
                    self.layer_dependencies.append(LayerDependency(
                        source_layer=name2,
                        target_layer=name1,
                        dependency_type=dependency['type'],
                        constraint=dependency['constraint'],
                        critical=dependency['critical']
                    ))
    
    def _is_prunable_layer(self, layer: nn.Module) -> bool:
        """Check if a layer is prunable."""
        return isinstance(layer, (nn.Linear, nn.Conv2d, nn.Conv1d))
    
    def _get_layer_dimensions(self, layer: nn.Module) -> Dict[str, int]:
        """Extract dimension information from a layer."""
        
        if isinstance(layer, nn.Linear):
            return {
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'weight_shape': list(layer.weight.shape),
                'bias_shape': list(layer.bias.shape) if layer.bias is not None else None
            }
        elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            return {
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'weight_shape': list(layer.weight.shape),
                'bias_shape': list(layer.bias.shape) if layer.bias is not None else None
            }
        else:
            return {}
    
    def _parse_layer_location(self, layer_name: str) -> Dict[str, Any]:
        """Parse layer location information (block, component, etc.)."""
        
        info = {
            'is_transformer_block': False,
            'block_index': None,
            'component': None,
            'layer_type': None
        }
        
        # Parse transformer block structure: "blocks.0.mlp.fc1"
        if 'blocks.' in layer_name:
            parts = layer_name.split('.')
            try:
                info['is_transformer_block'] = True
                info['block_index'] = int(parts[1])
                info['component'] = parts[2] if len(parts) > 2 else None  # 'mlp', 'attn'
                info['layer_type'] = parts[3] if len(parts) > 3 else None  # 'fc1', 'fc2', 'qkv', 'proj'
            except (IndexError, ValueError):
                pass
        
        return info
    
    def _are_connected(self, name1: str, name2: str, layer1: nn.Module, layer2: nn.Module) -> bool:
        """Check if two layers are directly connected."""
        
        # Vision Transformer connections
        if self._in_same_transformer_block(name1, name2):
            # MLP coupling: fc1 -> fc2
            if 'fc1' in name2 and 'fc2' in name1:
                return True
            # Attention coupling: qkv -> proj
            if 'qkv' in name2 and 'proj' in name1:
                return True
        
        # CNN connections (basic sequential detection)
        if self._are_sequential_cnn_layers(name1, name2, layer1, layer2):
            return True
        
        # Residual connections
        if self._has_residual_connection(name1, name2):
            return True
        
        return False
    
    def _in_same_transformer_block(self, name1: str, name2: str) -> bool:
        """Check if layers are in the same transformer block."""
        try:
            if 'blocks.' in name1 and 'blocks.' in name2:
                block1 = name1.split('.')[1]
                block2 = name2.split('.')[1]
                return block1 == block2
        except (IndexError, ValueError):
            pass
        return False
    
    def _are_sequential_cnn_layers(self, name1: str, name2: str, 
                                 layer1: nn.Module, layer2: nn.Module) -> bool:
        """Check if CNN layers are sequentially connected."""
        
        # Simple heuristic: if one layer's output channels match another's input channels
        if isinstance(layer1, (nn.Conv2d, nn.Conv1d)) and isinstance(layer2, (nn.Conv2d, nn.Conv1d)):
            return layer1.out_channels == layer2.in_channels
        
        # Conv to Linear connection
        if isinstance(layer1, (nn.Conv2d, nn.Conv1d)) and isinstance(layer2, nn.Linear):
            # This would require more sophisticated analysis of feature map sizes
            return False
        
        return False
    
    def _has_residual_connection(self, name1: str, name2: str) -> bool:
        """Check if layers are connected via residual connections."""
        
        # Heuristic: layers in the same block with similar names might have residual connections
        if self._in_same_transformer_block(name1, name2):
            # Skip connections in transformer blocks
            if ('attn' in name1 and 'mlp' in name2) or ('mlp' in name1 and 'attn' in name2):
                return True
        
        return False
    
    def _analyze_connection(self, name1: str, name2: str, 
                          layer1: nn.Module, layer2: nn.Module) -> Dict[str, Any]:
        """Analyze the type and constraints of a connection between layers."""
        
        # MLP coupling in transformer blocks
        if 'fc1' in name2 and 'fc2' in name1:
            return {
                'type': 'mlp_coupling',
                'constraint': 'fc1.out_features == fc2.in_features',
                'critical': True
            }
        
        # Attention coupling in transformer blocks
        if 'qkv' in name2 and 'proj' in name1:
            return {
                'type': 'attention_coupling',
                'constraint': 'qkv.out_features // 3 == proj.in_features',
                'critical': True
            }
        
        # Sequential CNN layers
        if isinstance(layer1, (nn.Conv2d, nn.Conv1d)) and isinstance(layer2, (nn.Conv2d, nn.Conv1d)):
            return {
                'type': 'sequential_conv',
                'constraint': 'layer1.out_channels == layer2.in_channels',
                'critical': True
            }
        
        # Residual connections
        if self._has_residual_connection(name1, name2):
            return {
                'type': 'residual_connection',
                'constraint': 'dimension_preservation_required',
                'critical': False  # Can often be handled with projection layers
            }
        
        return {
            'type': 'unknown',
            'constraint': 'unknown_constraint',
            'critical': False
        }
    
    def _detect_coupling_constraints(self):
        """Detect coupling constraints that require coordinated pruning."""
        
        # Group layers by transformer blocks
        transformer_blocks = self._group_transformer_blocks()
        
        for block_idx, block_layers in transformer_blocks.items():
            # MLP coupling constraints
            if 'fc1' in block_layers and 'fc2' in block_layers:
                self.coupling_constraints.append(CouplingConstraint(
                    layer_group=[block_layers['fc1'], block_layers['fc2']],
                    constraint_type='mlp_coupling',
                    description=f'MLP layers in block {block_idx} must maintain dimension compatibility',
                    dimensions_affected=['hidden_dim']
                ))
            
            # Attention coupling constraints
            if 'qkv' in block_layers and 'proj' in block_layers:
                self.coupling_constraints.append(CouplingConstraint(
                    layer_group=[block_layers['qkv'], block_layers['proj']],
                    constraint_type='attention_coupling',
                    description=f'Attention layers in block {block_idx} must maintain head dimension compatibility',
                    dimensions_affected=['embed_dim', 'num_heads']
                ))
        
        # Add global constraints
        self._detect_global_constraints()
    
    def _group_transformer_blocks(self) -> Dict[int, Dict[str, str]]:
        """Group transformer layers by block index."""
        
        blocks = {}
        
        for layer_name in self.dependency_graph:
            block_info = self.dependency_graph[layer_name]['block_info']
            
            if block_info['is_transformer_block']:
                block_idx = block_info['block_index']
                layer_type = block_info['layer_type']
                
                if block_idx not in blocks:
                    blocks[block_idx] = {}
                
                if layer_type:
                    blocks[block_idx][layer_type] = layer_name
        
        return blocks
    
    def _detect_global_constraints(self):
        """Detect global constraints that affect the entire model."""
        
        # Find embedding and classification layers
        embedding_layers = []
        classification_layers = []
        
        for layer_name, layer_info in self.dependency_graph.items():
            if 'embed' in layer_name.lower() or 'patch_embed' in layer_name.lower():
                embedding_layers.append(layer_name)
            elif 'head' in layer_name.lower() or 'classifier' in layer_name.lower():
                classification_layers.append(layer_name)
        
        # Add constraints for critical layers
        if embedding_layers:
            self.coupling_constraints.append(CouplingConstraint(
                layer_group=embedding_layers,
                constraint_type='embedding_preservation',
                description='Embedding layers should be preserved or pruned very conservatively',
                dimensions_affected=['embed_dim']
            ))
        
        if classification_layers:
            self.coupling_constraints.append(CouplingConstraint(
                layer_group=classification_layers,
                constraint_type='classification_preservation',
                description='Classification layers should be preserved to maintain output structure',
                dimensions_affected=['num_classes']
            ))
    
    def get_coupled_layers(self, layer_name: str) -> List[str]:
        """Get all layers that must be pruned together with the specified layer."""
        
        if layer_name not in self.dependency_graph:
            return [layer_name]
        
        coupled = set([layer_name])
        
        # Add directly dependent layers
        for dep in self.dependency_graph[layer_name]['dependents']:
            if dep['dependency_type'] in ['mlp_coupling', 'attention_coupling']:
                coupled.add(dep['layer_name'])
        
        # Add layers this depends on
        for dep in self.dependency_graph[layer_name]['depends_on']:
            if dep['dependency_type'] in ['mlp_coupling', 'attention_coupling']:
                coupled.add(dep['layer_name'])
        
        # Add layers from coupling constraints
        for constraint in self.coupling_constraints:
            if layer_name in constraint.layer_group:
                coupled.update(constraint.layer_group)
        
        return list(coupled)
    
    def get_dependency_graph(self) -> Dict[str, Dict[str, Any]]:
        """Return the complete dependency graph."""
        return self.dependency_graph.copy()
    
    def get_coupling_constraints(self) -> List[CouplingConstraint]:
        """Return all detected coupling constraints."""
        return self.coupling_constraints.copy()
    
    def validate_pruning_compatibility(self, pruning_plan: Dict[str, float]) -> Dict[str, Any]:
        """Validate that a pruning plan respects all dependency constraints."""
        
        validation_result = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check coupling constraints
        for constraint in self.coupling_constraints:
            violation = self._check_constraint_violation(constraint, pruning_plan)
            if violation:
                validation_result['violations'].append(violation)
                validation_result['valid'] = False
        
        # Check dimension compatibility
        for dependency in self.layer_dependencies:
            if dependency.critical:
                compatibility_issue = self._check_dimension_compatibility(dependency, pruning_plan)
                if compatibility_issue:
                    validation_result['violations'].append(compatibility_issue)
                    validation_result['valid'] = False
        
        # Generate recommendations
        if not validation_result['valid']:
            validation_result['recommendations'] = self._generate_fix_recommendations(
                validation_result['violations']
            )
        
        return validation_result
    
    def _check_constraint_violation(self, constraint: CouplingConstraint, 
                                  pruning_plan: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Check if a coupling constraint is violated by the pruning plan."""
        
        # Get pruning ratios for layers in the constraint group
        ratios = []
        for layer_name in constraint.layer_group:
            if layer_name in pruning_plan:
                ratios.append(pruning_plan[layer_name])
        
        if not ratios:
            return None
        
        # For coupling constraints, all layers should have similar pruning ratios
        if constraint.constraint_type in ['mlp_coupling', 'attention_coupling']:
            ratio_variance = max(ratios) - min(ratios)
            if ratio_variance > 0.1:  # 10% tolerance
                return {
                    'type': 'coupling_violation',
                    'constraint': constraint,
                    'issue': f'Pruning ratios vary too much: {ratios}',
                    'severity': 'high'
                }
        
        return None
    
    def _check_dimension_compatibility(self, dependency: LayerDependency, 
                                     pruning_plan: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Check if dimension compatibility is maintained."""
        
        source_ratio = pruning_plan.get(dependency.source_layer, 0.0)
        target_ratio = pruning_plan.get(dependency.target_layer, 0.0)
        
        # For dimension matching dependencies, ratios should be identical
        if dependency.dependency_type in ['mlp_coupling', 'attention_coupling']:
            if abs(source_ratio - target_ratio) > 0.05:  # 5% tolerance
                return {
                    'type': 'dimension_mismatch',
                    'dependency': dependency,
                    'issue': f'Incompatible pruning ratios: {source_ratio} vs {target_ratio}',
                    'severity': 'high'
                }
        
        return None
    
    def _generate_fix_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations to fix constraint violations."""
        
        recommendations = []
        
        for violation in violations:
            if violation['type'] == 'coupling_violation':
                constraint = violation['constraint']
                recommendations.append(
                    f"Use identical pruning ratios for coupled layers: {constraint.layer_group}"
                )
            elif violation['type'] == 'dimension_mismatch':
                dependency = violation['dependency']
                recommendations.append(
                    f"Ensure {dependency.source_layer} and {dependency.target_layer} "
                    f"have compatible pruning ratios for {dependency.constraint}"
                )
        
        return recommendations
    
    def get_safe_pruning_groups(self) -> List[List[str]]:
        """Get groups of layers that can be safely pruned together."""
        
        safe_groups = []
        processed_layers = set()
        
        for layer_name in self.dependency_graph:
            if layer_name in processed_layers:
                continue
            
            # Get all coupled layers
            coupled_layers = self.get_coupled_layers(layer_name)
            
            if len(coupled_layers) > 1:
                safe_groups.append(coupled_layers)
                processed_layers.update(coupled_layers)
            else:
                # Individual layer can be pruned independently
                safe_groups.append([layer_name])
                processed_layers.add(layer_name)
        
        return safe_groups
    
    def get_critical_layers(self) -> List[str]:
        """Get layers that are critical and should be pruned conservatively."""
        
        critical_layers = []
        
        for layer_name, layer_info in self.dependency_graph.items():
            # Embedding and classification layers are critical
            if any(keyword in layer_name.lower() for keyword in ['embed', 'head', 'classifier']):
                critical_layers.append(layer_name)
            
            # Layers with many dependencies are critical
            total_deps = len(layer_info['depends_on']) + len(layer_info['dependents'])
            if total_deps > 2:
                critical_layers.append(layer_name)
        
        return critical_layers
    
    def print_dependency_summary(self):
        """Print a comprehensive summary of the dependency analysis."""
        
        print(f"\n🔗 Dependency Analysis Summary for {self.model_name}")
        print("=" * 60)
        
        print(f"📊 Statistics:")
        print(f"   Total prunable layers: {len(self.dependency_graph)}")
        print(f"   Layer dependencies: {len(self.layer_dependencies)}")
        print(f"   Coupling constraints: {len(self.coupling_constraints)}")
        
        print(f"\n🔒 Coupling Constraints:")
        for i, constraint in enumerate(self.coupling_constraints, 1):
            print(f"   {i}. {constraint.constraint_type}: {constraint.description}")
            print(f"      Layers: {constraint.layer_group}")
        
        print(f"\n⚠️ Critical Layers:")
        critical_layers = self.get_critical_layers()
        for layer in critical_layers:
            print(f"   - {layer}")
        
        print(f"\n👥 Safe Pruning Groups:")
        safe_groups = self.get_safe_pruning_groups()
        for i, group in enumerate(safe_groups, 1):
            if len(group) > 1:
                print(f"   Group {i}: {group}")
        
        print("=" * 60)

