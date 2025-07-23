#!/usr/bin/env python3
"""
Architecture Detector for Multi-Agent LLM Pruning Framework

This module provides automatic detection and analysis of neural network
architectures to identify model types, components, and pruning opportunities.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional, Set
import torch
import torch.nn as nn
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

class ArchitectureDetector:
    """
    Detector for automatically analyzing and classifying neural network architectures.
    Identifies model types, key components, and provides pruning-relevant insights.
    """
    
    def __init__(self):
        # Component patterns for different architectures
        self.component_patterns = {
            'attention': ['attention', 'attn', 'multihead', 'self_attn'],
            'convolution': ['conv1d', 'conv2d', 'conv3d', 'convolution'],
            'linear': ['linear', 'dense', 'fc', 'fully_connected'],
            'normalization': ['batchnorm', 'layernorm', 'groupnorm', 'instancenorm'],
            'activation': ['relu', 'gelu', 'swish', 'sigmoid', 'tanh'],
            'pooling': ['maxpool', 'avgpool', 'adaptiveavgpool', 'globalavgpool'],
            'dropout': ['dropout'],
            'embedding': ['embedding', 'embed']
        }
        
        # Architecture signatures
        self.architecture_signatures = {
            'resnet': ['basicblock', 'bottleneck', 'downsample'],
            'vit': ['patch_embed', 'pos_embed', 'transformer', 'attention'],
            'deit': ['distillation', 'cls_token', 'dist_token'],
            'swin': ['window_attention', 'shifted_window', 'patch_merging'],
            'efficientnet': ['mbconv', 'inverted_residual', 'squeeze_excite'],
            'bert': ['encoder', 'decoder', 'attention', 'feed_forward'],
            'gpt': ['causal_attention', 'decoder_only', 'autoregressive']
        }
        
        logger.info("🔍 Architecture Detector initialized")
    
    def detect_architecture(self, model: nn.Module, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect and analyze the architecture of a given model.
        
        Args:
            model: PyTorch model to analyze
            model_name: Optional model name for additional context
            
        Returns:
            Comprehensive architecture analysis
        """
        
        logger.info(f"🔍 Detecting architecture for model: {model_name or 'unnamed'}")
        
        # Basic model analysis
        basic_info = self._analyze_basic_structure(model)
        
        # Component analysis
        component_analysis = self._analyze_components(model)
        
        # Architecture type detection
        architecture_type = self._detect_architecture_type(model, component_analysis)
        
        # Specific architecture detection
        specific_architecture = self._detect_specific_architecture(model, model_name)
        
        # Pruning analysis
        pruning_analysis = self._analyze_pruning_opportunities(model, component_analysis)
        
        # Dependency analysis
        dependency_analysis = self._analyze_dependencies(model)
        
        # Combine all analyses
        detection_result = {
            'model_name': model_name,
            'basic_info': basic_info,
            'component_analysis': component_analysis,
            'architecture_type': architecture_type,
            'specific_architecture': specific_architecture,
            'pruning_analysis': pruning_analysis,
            'dependency_analysis': dependency_analysis,
            'detection_confidence': self._calculate_confidence(architecture_type, specific_architecture),
            'recommendations': self._generate_recommendations(architecture_type, pruning_analysis)
        }
        
        logger.info(f"✅ Architecture detection completed: {architecture_type['primary_type']}")
        return detection_result
    
    def _analyze_basic_structure(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze basic structure of the model."""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count modules by type
        module_counts = defaultdict(int)
        total_modules = 0
        
        for module in model.modules():
            if len(list(module.children())) == 0:  # Leaf modules
                module_type = type(module).__name__
                module_counts[module_type] += 1
                total_modules += 1
        
        # Calculate model depth
        depth = self._calculate_model_depth(model)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_modules': total_modules,
            'module_counts': dict(module_counts),
            'model_depth': depth,
            'parameter_efficiency': trainable_params / total_params if total_params > 0 else 0.0
        }
    
    def _analyze_components(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze components present in the model."""
        
        component_counts = defaultdict(int)
        component_details = defaultdict(list)
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                module_type = type(module).__name__.lower()
                
                # Classify component
                for component_type, patterns in self.component_patterns.items():
                    if any(pattern in module_type for pattern in patterns):
                        component_counts[component_type] += 1
                        component_details[component_type].append({
                            'name': name,
                            'type': type(module).__name__,
                            'parameters': sum(p.numel() for p in module.parameters())
                        })
                        break
                else:
                    # Unclassified component
                    component_counts['other'] += 1
                    component_details['other'].append({
                        'name': name,
                        'type': type(module).__name__,
                        'parameters': sum(p.numel() for p in module.parameters())
                    })
        
        # Calculate component ratios
        total_components = sum(component_counts.values())
        component_ratios = {
            comp_type: count / total_components if total_components > 0 else 0.0
            for comp_type, count in component_counts.items()
        }
        
        return {
            'component_counts': dict(component_counts),
            'component_ratios': component_ratios,
            'component_details': dict(component_details),
            'dominant_component': max(component_counts, key=component_counts.get) if component_counts else 'unknown'
        }
    
    def _detect_architecture_type(self, model: nn.Module, component_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect the primary architecture type."""
        
        component_counts = component_analysis['component_counts']
        
        # Decision logic for architecture type
        has_attention = component_counts.get('attention', 0) > 0
        has_conv = component_counts.get('convolution', 0) > 0
        has_linear = component_counts.get('linear', 0) > 0
        has_embedding = component_counts.get('embedding', 0) > 0
        
        # Calculate ratios
        total_computational = component_counts.get('convolution', 0) + component_counts.get('linear', 0) + component_counts.get('attention', 0)
        
        if total_computational == 0:
            primary_type = 'unknown'
            confidence = 0.0
        elif has_attention and component_counts['attention'] / total_computational > 0.3:
            primary_type = 'transformer'
            confidence = min(0.9, component_counts['attention'] / total_computational + 0.5)
        elif has_conv and component_counts['convolution'] / total_computational > 0.5:
            primary_type = 'cnn'
            confidence = min(0.9, component_counts['convolution'] / total_computational + 0.3)
        elif has_linear and component_counts['linear'] / total_computational > 0.7:
            primary_type = 'mlp'
            confidence = min(0.9, component_counts['linear'] / total_computational + 0.2)
        elif has_conv and has_attention:
            primary_type = 'hybrid'
            confidence = 0.7
        else:
            primary_type = 'mixed'
            confidence = 0.5
        
        # Additional characteristics
        characteristics = []
        if has_attention:
            characteristics.append('attention_based')
        if has_conv:
            characteristics.append('convolutional')
        if has_linear:
            characteristics.append('fully_connected')
        if has_embedding:
            characteristics.append('embedding_based')
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'characteristics': characteristics,
            'component_distribution': {
                'attention_ratio': component_counts.get('attention', 0) / total_computational if total_computational > 0 else 0.0,
                'convolution_ratio': component_counts.get('convolution', 0) / total_computational if total_computational > 0 else 0.0,
                'linear_ratio': component_counts.get('linear', 0) / total_computational if total_computational > 0 else 0.0
            }
        }
    
    def _detect_specific_architecture(self, model: nn.Module, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Detect specific architecture family (ResNet, ViT, etc.)."""
        
        # Get all module names and types
        module_names = [name.lower() for name, _ in model.named_modules()]
        module_types = [type(module).__name__.lower() for module in model.modules()]
        
        all_text = ' '.join(module_names + module_types)
        if model_name:
            all_text += ' ' + model_name.lower()
        
        # Score each architecture
        architecture_scores = {}
        
        for arch_name, signatures in self.architecture_signatures.items():
            score = 0
            matches = []
            
            for signature in signatures:
                if signature in all_text:
                    score += 1
                    matches.append(signature)
            
            if score > 0:
                architecture_scores[arch_name] = {
                    'score': score,
                    'max_score': len(signatures),
                    'confidence': score / len(signatures),
                    'matches': matches
                }
        
        # Find best match
        if architecture_scores:
            best_match = max(architecture_scores, key=lambda k: architecture_scores[k]['confidence'])
            best_score = architecture_scores[best_match]
            
            return {
                'detected_architecture': best_match,
                'confidence': best_score['confidence'],
                'matches': best_score['matches'],
                'all_scores': architecture_scores
            }
        else:
            return {
                'detected_architecture': 'unknown',
                'confidence': 0.0,
                'matches': [],
                'all_scores': {}
            }
    
    def _analyze_pruning_opportunities(self, model: nn.Module, component_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pruning opportunities in the model."""
        
        component_counts = component_analysis['component_counts']
        component_details = component_analysis['component_details']
        
        # Pruning opportunities by component type
        pruning_opportunities = {}
        
        # Linear layers
        if 'linear' in component_counts:
            linear_details = component_details['linear']
            total_linear_params = sum(detail['parameters'] for detail in linear_details)
            
            pruning_opportunities['linear_pruning'] = {
                'layer_count': component_counts['linear'],
                'total_parameters': total_linear_params,
                'pruning_potential': 'high',
                'recommended_methods': ['magnitude_pruning', 'structured_pruning'],
                'estimated_reduction': 0.4
            }
        
        # Convolutional layers
        if 'convolution' in component_counts:
            conv_details = component_details['convolution']
            total_conv_params = sum(detail['parameters'] for detail in conv_details)
            
            pruning_opportunities['channel_pruning'] = {
                'layer_count': component_counts['convolution'],
                'total_parameters': total_conv_params,
                'pruning_potential': 'high',
                'recommended_methods': ['channel_pruning', 'filter_pruning'],
                'estimated_reduction': 0.35
            }
        
        # Attention layers
        if 'attention' in component_counts:
            attn_details = component_details['attention']
            total_attn_params = sum(detail['parameters'] for detail in attn_details)
            
            pruning_opportunities['attention_pruning'] = {
                'layer_count': component_counts['attention'],
                'total_parameters': total_attn_params,
                'pruning_potential': 'medium',
                'recommended_methods': ['head_pruning', 'attention_pruning'],
                'estimated_reduction': 0.25
            }
        
        # Overall assessment
        total_prunable_params = sum(
            opp.get('total_parameters', 0) for opp in pruning_opportunities.values()
        )
        total_params = sum(p.numel() for p in model.parameters())
        
        overall_potential = total_prunable_params / total_params if total_params > 0 else 0.0
        
        return {
            'opportunities': pruning_opportunities,
            'overall_potential': overall_potential,
            'prunable_parameter_ratio': overall_potential,
            'recommended_strategy': self._recommend_pruning_strategy(pruning_opportunities),
            'complexity_assessment': self._assess_pruning_complexity(model, component_analysis)
        }
    
    def _analyze_dependencies(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze dependencies between model components."""
        
        # Find skip connections and residual blocks
        skip_connections = self._find_skip_connections(model)
        
        # Find coupled layers (like MLP blocks, attention blocks)
        coupled_layers = self._find_coupled_layers(model)
        
        # Find normalization dependencies
        norm_dependencies = self._find_normalization_dependencies(model)
        
        return {
            'skip_connections': skip_connections,
            'coupled_layers': coupled_layers,
            'normalization_dependencies': norm_dependencies,
            'dependency_complexity': self._calculate_dependency_complexity(skip_connections, coupled_layers),
            'pruning_constraints': self._identify_pruning_constraints(skip_connections, coupled_layers, norm_dependencies)
        }
    
    def _calculate_model_depth(self, model: nn.Module) -> int:
        """Calculate the depth of the model."""
        
        max_depth = 0
        
        def calculate_depth(module, current_depth=0):
            nonlocal max_depth
            children = list(module.children())
            
            if not children:  # Leaf node
                max_depth = max(max_depth, current_depth)
            else:
                for child in children:
                    calculate_depth(child, current_depth + 1)
        
        calculate_depth(model)
        return max_depth
    
    def _find_skip_connections(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Find skip connections in the model."""
        
        skip_connections = []
        
        # Look for common skip connection patterns
        for name, module in model.named_modules():
            module_name_lower = name.lower()
            
            # ResNet-style skip connections
            if 'downsample' in module_name_lower or 'shortcut' in module_name_lower:
                skip_connections.append({
                    'name': name,
                    'type': 'residual',
                    'module_type': type(module).__name__
                })
            
            # DenseNet-style connections
            elif 'dense' in module_name_lower and 'block' in module_name_lower:
                skip_connections.append({
                    'name': name,
                    'type': 'dense',
                    'module_type': type(module).__name__
                })
        
        return skip_connections
    
    def _find_coupled_layers(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Find coupled layers that should be pruned together."""
        
        coupled_layers = []
        
        # Look for MLP blocks (fc1 -> fc2 pattern)
        mlp_patterns = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'fc1' in name or 'mlp.0' in name or 'feed_forward.0' in name:
                    # Look for corresponding fc2
                    fc2_name = name.replace('fc1', 'fc2').replace('mlp.0', 'mlp.2').replace('feed_forward.0', 'feed_forward.2')
                    for name2, module2 in model.named_modules():
                        if name2 == fc2_name and isinstance(module2, nn.Linear):
                            coupled_layers.append({
                                'type': 'mlp_block',
                                'layers': [name, name2],
                                'coupling_strength': 'strong'
                            })
                            break
        
        # Look for attention blocks (qkv -> proj pattern)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(pattern in name for pattern in ['qkv', 'query', 'key', 'value']):
                    # Look for corresponding projection
                    base_name = name.split('.')[:-1]  # Remove last part
                    proj_patterns = ['proj', 'out_proj', 'projection']
                    
                    for proj_pattern in proj_patterns:
                        proj_name = '.'.join(base_name + [proj_pattern])
                        for name2, module2 in model.named_modules():
                            if name2 == proj_name and isinstance(module2, nn.Linear):
                                coupled_layers.append({
                                    'type': 'attention_block',
                                    'layers': [name, name2],
                                    'coupling_strength': 'strong'
                                })
                                break
        
        return coupled_layers
    
    def _find_normalization_dependencies(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Find normalization layer dependencies."""
        
        norm_dependencies = []
        
        # Find BatchNorm dependencies on Conv layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Look for preceding conv layer
                name_parts = name.split('.')
                for i in range(len(name_parts)):
                    potential_conv_name = '.'.join(name_parts[:i+1])
                    for conv_name, conv_module in model.named_modules():
                        if (conv_name.startswith(potential_conv_name) and 
                            isinstance(conv_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))):
                            norm_dependencies.append({
                                'norm_layer': name,
                                'dependent_layer': conv_name,
                                'dependency_type': 'channel_size'
                            })
                            break
        
        return norm_dependencies
    
    def _calculate_dependency_complexity(self, skip_connections: List, coupled_layers: List) -> str:
        """Calculate overall dependency complexity."""
        
        total_dependencies = len(skip_connections) + len(coupled_layers)
        
        if total_dependencies == 0:
            return 'low'
        elif total_dependencies < 5:
            return 'medium'
        else:
            return 'high'
    
    def _identify_pruning_constraints(self, skip_connections: List, coupled_layers: List, 
                                   norm_dependencies: List) -> List[str]:
        """Identify constraints for pruning based on dependencies."""
        
        constraints = []
        
        if skip_connections:
            constraints.append('maintain_skip_connection_dimensions')
        
        if coupled_layers:
            constraints.append('coordinate_coupled_layer_pruning')
        
        if norm_dependencies:
            constraints.append('adjust_normalization_layers')
        
        # Add general constraints
        constraints.extend([
            'preserve_model_functionality',
            'maintain_gradient_flow'
        ])
        
        return constraints
    
    def _recommend_pruning_strategy(self, pruning_opportunities: Dict[str, Any]) -> str:
        """Recommend overall pruning strategy based on opportunities."""
        
        if not pruning_opportunities:
            return 'conservative_unstructured'
        
        # Count different types of opportunities
        has_linear = 'linear_pruning' in pruning_opportunities
        has_conv = 'channel_pruning' in pruning_opportunities
        has_attention = 'attention_pruning' in pruning_opportunities
        
        if has_attention and has_linear:
            return 'mixed_transformer_strategy'
        elif has_conv and has_linear:
            return 'mixed_cnn_strategy'
        elif has_conv:
            return 'channel_focused_strategy'
        elif has_linear:
            return 'structured_linear_strategy'
        elif has_attention:
            return 'attention_focused_strategy'
        else:
            return 'conservative_unstructured'
    
    def _assess_pruning_complexity(self, model: nn.Module, component_analysis: Dict[str, Any]) -> str:
        """Assess the complexity of pruning this model."""
        
        total_params = sum(p.numel() for p in model.parameters())
        total_modules = component_analysis['component_counts']
        
        # Factors that increase complexity
        complexity_factors = 0
        
        # Large number of parameters
        if total_params > 100e6:  # > 100M parameters
            complexity_factors += 1
        
        # Many different component types
        if len(total_modules) > 8:
            complexity_factors += 1
        
        # Presence of attention (more complex to prune)
        if 'attention' in total_modules:
            complexity_factors += 1
        
        # Mixed architecture types
        component_ratios = [
            total_modules.get('convolution', 0),
            total_modules.get('linear', 0),
            total_modules.get('attention', 0)
        ]
        non_zero_components = sum(1 for ratio in component_ratios if ratio > 0)
        if non_zero_components > 2:
            complexity_factors += 1
        
        # Determine complexity level
        if complexity_factors == 0:
            return 'low'
        elif complexity_factors <= 2:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_confidence(self, architecture_type: Dict[str, Any], 
                            specific_architecture: Dict[str, Any]) -> float:
        """Calculate overall detection confidence."""
        
        type_confidence = architecture_type.get('confidence', 0.0)
        specific_confidence = specific_architecture.get('confidence', 0.0)
        
        # Weighted average with more weight on type detection
        overall_confidence = 0.7 * type_confidence + 0.3 * specific_confidence
        
        return min(1.0, overall_confidence)
    
    def _generate_recommendations(self, architecture_type: Dict[str, Any], 
                                pruning_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detection results."""
        
        recommendations = []
        
        # Architecture-specific recommendations
        primary_type = architecture_type.get('primary_type', 'unknown')
        
        if primary_type == 'transformer':
            recommendations.extend([
                'Consider attention head pruning for efficiency',
                'Use structured pruning for feed-forward layers',
                'Be careful with positional embeddings'
            ])
        elif primary_type == 'cnn':
            recommendations.extend([
                'Channel pruning is highly effective for CNNs',
                'Consider filter pruning for convolutional layers',
                'Maintain spatial dimension compatibility'
            ])
        elif primary_type == 'mlp':
            recommendations.extend([
                'Structured pruning works well for MLPs',
                'Consider magnitude-based pruning',
                'Layer pruning may be effective for deep MLPs'
            ])
        elif primary_type == 'hybrid':
            recommendations.extend([
                'Use different strategies for different components',
                'Coordinate pruning across component boundaries',
                'Consider progressive pruning approach'
            ])
        
        # Pruning potential recommendations
        overall_potential = pruning_analysis.get('overall_potential', 0.0)
        
        if overall_potential > 0.7:
            recommendations.append('High pruning potential - aggressive pruning possible')
        elif overall_potential > 0.4:
            recommendations.append('Moderate pruning potential - balanced approach recommended')
        else:
            recommendations.append('Limited pruning potential - conservative approach advised')
        
        return recommendations
    
    def compare_architectures(self, models: List[Tuple[nn.Module, str]]) -> Dict[str, Any]:
        """Compare multiple architectures."""
        
        logger.info(f"🔍 Comparing {len(models)} architectures")
        
        comparisons = {}
        
        for model, name in models:
            try:
                detection = self.detect_architecture(model, name)
                comparisons[name] = detection
            except Exception as e:
                logger.error(f"Failed to analyze {name}: {e}")
                comparisons[name] = {'error': str(e)}
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(comparisons)
        
        return {
            'individual_analyses': comparisons,
            'comparison_summary': summary
        }
    
    def _generate_comparison_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of architecture comparisons."""
        
        # Count architecture types
        type_counts = Counter()
        specific_counts = Counter()
        
        for name, analysis in comparisons.items():
            if 'error' not in analysis:
                arch_type = analysis.get('architecture_type', {}).get('primary_type', 'unknown')
                specific_arch = analysis.get('specific_architecture', {}).get('detected_architecture', 'unknown')
                
                type_counts[arch_type] += 1
                specific_counts[specific_arch] += 1
        
        return {
            'total_models': len(comparisons),
            'architecture_type_distribution': dict(type_counts),
            'specific_architecture_distribution': dict(specific_counts),
            'most_common_type': type_counts.most_common(1)[0] if type_counts else ('unknown', 0),
            'most_common_specific': specific_counts.most_common(1)[0] if specific_counts else ('unknown', 0)
        }
