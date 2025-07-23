#!/usr/bin/env python3
"""
Pruning Engine for Multi-Agent LLM Pruning Framework

This module provides the core pruning execution engine that applies
pruning decisions while respecting dependency constraints and safety limits.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import numpy as np
from .dependency_analyzer import DependencyAnalyzer
from .isomorphic_analyzer import IsomorphicAnalyzer, IsomorphicGroup, MLPCouple, AttentionCouple

logger = logging.getLogger(__name__)

@dataclass
class PruningResult:
    """Results from a pruning operation."""
    success: bool
    original_params: int
    final_params: int
    params_reduction: float
    layers_pruned: List[str]
    pruning_ratios: Dict[str, float]
    execution_time: float
    warnings: List[str]
    errors: List[str]

class PruningEngine:
    """
    Core pruning execution engine that applies pruning decisions while
    respecting dependency constraints and architectural requirements.
    """
    
    def __init__(self, model: nn.Module, model_name: Optional[str] = None):
        self.model = model
        self.model_name = model_name or "unknown_model"
        
        # Initialize analyzers
        self.dependency_analyzer = DependencyAnalyzer(model, model_name)
        self.isomorphic_analyzer = IsomorphicAnalyzer(model, model_name)
        
        # Track original model state
        self.original_params = self._count_parameters()
        self.original_state = None
        
        # Pruning state
        self.is_pruned = False
        self.pruning_history = []
        
        logger.info(f"🔧 Pruning engine initialized for {self.model_name}")
        logger.info(f"   Original parameters: {self.original_params:,}")
    
    def _count_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
    
    def save_model_state(self):
        """Save the current model state for potential restoration."""
        self.original_state = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        logger.debug("💾 Model state saved")
    
    def restore_model_state(self):
        """Restore the model to its saved state."""
        if self.original_state is None:
            logger.warning("⚠️ No saved state to restore")
            return False
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state:
                    param.copy_(self.original_state[name])
        
        self.is_pruned = False
        logger.info("🔄 Model state restored")
        return True
    
    def apply_isomorphic_pruning(self, isomorphic_groups: Dict[str, IsomorphicGroup],
                               importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> PruningResult:
        """
        Apply pruning based on isomorphic groups with dependency awareness.
        
        Args:
            isomorphic_groups: Dictionary of isomorphic groups to prune
            importance_scores: Optional importance scores for guided pruning
            
        Returns:
            PruningResult with detailed information about the pruning operation
        """
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Save state before pruning
        self.save_model_state()
        
        result = PruningResult(
            success=False,
            original_params=self.original_params,
            final_params=0,
            params_reduction=0.0,
            layers_pruned=[],
            pruning_ratios={},
            execution_time=0.0,
            warnings=[],
            errors=[]
        )
        
        try:
            # Validate groups before pruning
            validation = self.isomorphic_analyzer.validate_groups(isomorphic_groups)
            if not validation['valid']:
                result.errors.extend(validation['issues'])
                result.warnings.extend(validation['warnings'])
                logger.error("❌ Group validation failed, aborting pruning")
                return result
            
            # Apply pruning to each group
            for group_name, group in isomorphic_groups.items():
                if group.pruning_ratio <= 0:
                    logger.debug(f"⏭️ Skipping group {group_name} (zero pruning ratio)")
                    continue
                
                logger.info(f"✂️ Pruning group: {group_name} (ratio: {group.pruning_ratio:.1%})")
                
                group_result = self._prune_isomorphic_group(group, importance_scores)
                
                if group_result['success']:
                    result.layers_pruned.extend(group_result['layers_pruned'])
                    result.pruning_ratios.update(group_result['pruning_ratios'])
                    logger.info(f"✅ Successfully pruned {len(group_result['layers_pruned'])} layers in {group_name}")
                else:
                    result.warnings.extend(group_result['warnings'])
                    result.errors.extend(group_result['errors'])
                    logger.warning(f"⚠️ Partial failure in group {group_name}")
            
            # Calculate final statistics
            result.final_params = self._count_parameters()
            result.params_reduction = 1.0 - (result.final_params / result.original_params)
            result.success = len(result.errors) == 0
            
            if result.success:
                self.is_pruned = True
                self.pruning_history.append({
                    'groups': list(isomorphic_groups.keys()),
                    'params_reduction': result.params_reduction,
                    'layers_pruned': len(result.layers_pruned)
                })
                logger.info(f"🎉 Pruning completed successfully!")
                logger.info(f"   Parameters: {result.original_params:,} → {result.final_params:,}")
                logger.info(f"   Reduction: {result.params_reduction:.1%}")
            else:
                logger.error("❌ Pruning failed, restoring original state")
                self.restore_model_state()
                result.final_params = self.original_params
                result.params_reduction = 0.0
        
        except Exception as e:
            logger.error(f"💥 Pruning engine error: {str(e)}")
            result.errors.append(f"Pruning engine error: {str(e)}")
            self.restore_model_state()
            result.final_params = self.original_params
        
        finally:
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                result.execution_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        return result
    
    def _prune_isomorphic_group(self, group: IsomorphicGroup, 
                              importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Prune a single isomorphic group."""
        
        group_result = {
            'success': True,
            'layers_pruned': [],
            'pruning_ratios': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            if group.group_type == 'mlp':
                return self._prune_mlp_group(group, importance_scores)
            elif group.group_type == 'attention':
                return self._prune_attention_group(group, importance_scores)
            elif group.group_type in ['conv', 'fc', 'linear']:
                return self._prune_standard_group(group, importance_scores)
            else:
                group_result['warnings'].append(f"Unknown group type: {group.group_type}")
                return group_result
                
        except Exception as e:
            group_result['success'] = False
            group_result['errors'].append(f"Group pruning error: {str(e)}")
            logger.error(f"Error pruning group {group.name}: {str(e)}")
            return group_result
    
    def _prune_mlp_group(self, group: IsomorphicGroup, 
                        importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Prune MLP groups with coupled fc1+fc2 layers."""
        
        result = {
            'success': True,
            'layers_pruned': [],
            'pruning_ratios': {},
            'warnings': [],
            'errors': []
        }
        
        for i, mlp_couple in enumerate(group.layers):
            if not isinstance(mlp_couple, MLPCouple):
                result['errors'].append(f"Expected MLPCouple, got {type(mlp_couple)}")
                continue
            
            try:
                # Validate coupling before pruning
                if not mlp_couple.validate_coupling():
                    result['errors'].append(f"MLP coupling validation failed for {mlp_couple.fc1_name}")
                    continue
                
                # Determine neurons to prune
                hidden_dim = mlp_couple.get_hidden_dim()
                num_to_prune = int(hidden_dim * group.pruning_ratio)
                
                if num_to_prune == 0:
                    result['warnings'].append(f"No neurons to prune in {mlp_couple.fc1_name}")
                    continue
                
                # Get importance scores for this MLP
                fc1_importance = importance_scores.get(mlp_couple.fc1_name) if importance_scores else None
                
                # Select neurons to prune
                if fc1_importance is not None:
                    # Use importance-based selection
                    neurons_to_prune = self._select_neurons_by_importance(
                        fc1_importance, num_to_prune
                    )
                else:
                    # Use magnitude-based selection
                    neurons_to_prune = self._select_neurons_by_magnitude(
                        mlp_couple.fc1, num_to_prune
                    )
                
                # Apply coordinated pruning to both fc1 and fc2
                self._prune_mlp_couple(mlp_couple, neurons_to_prune)
                
                result['layers_pruned'].extend([mlp_couple.fc1_name, mlp_couple.fc2_name])
                result['pruning_ratios'][mlp_couple.fc1_name] = group.pruning_ratio
                result['pruning_ratios'][mlp_couple.fc2_name] = group.pruning_ratio
                
                logger.debug(f"✂️ Pruned {len(neurons_to_prune)} neurons from MLP couple {i}")
                
            except Exception as e:
                result['errors'].append(f"Error pruning MLP couple {i}: {str(e)}")
                result['success'] = False
        
        return result
    
    def _prune_attention_group(self, group: IsomorphicGroup, 
                             importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Prune attention groups with coupled qkv+proj layers."""
        
        result = {
            'success': True,
            'layers_pruned': [],
            'pruning_ratios': {},
            'warnings': [],
            'errors': []
        }
        
        for i, attn_couple in enumerate(group.layers):
            if not isinstance(attn_couple, AttentionCouple):
                result['errors'].append(f"Expected AttentionCouple, got {type(attn_couple)}")
                continue
            
            try:
                # Validate coupling before pruning
                if not attn_couple.validate_coupling():
                    result['errors'].append(f"Attention coupling validation failed for {attn_couple.qkv_name}")
                    continue
                
                # Determine attention heads to prune
                embed_dim = attn_couple.get_embed_dim()
                num_heads = embed_dim // 64  # Assume 64-dim heads (common in ViTs)
                heads_to_prune = max(1, int(num_heads * group.pruning_ratio))
                
                if heads_to_prune >= num_heads:
                    result['warnings'].append(f"Cannot prune all heads in {attn_couple.qkv_name}")
                    heads_to_prune = num_heads - 1
                
                # Get importance scores for attention
                qkv_importance = importance_scores.get(attn_couple.qkv_name) if importance_scores else None
                
                # Select heads to prune
                if qkv_importance is not None:
                    heads_to_prune_indices = self._select_attention_heads_by_importance(
                        qkv_importance, heads_to_prune, num_heads
                    )
                else:
                    heads_to_prune_indices = self._select_attention_heads_by_magnitude(
                        attn_couple.qkv, heads_to_prune, num_heads
                    )
                
                # Apply coordinated pruning to both qkv and proj
                self._prune_attention_couple(attn_couple, heads_to_prune_indices, num_heads)
                
                result['layers_pruned'].extend([attn_couple.qkv_name, attn_couple.proj_name])
                actual_ratio = len(heads_to_prune_indices) / num_heads
                result['pruning_ratios'][attn_couple.qkv_name] = actual_ratio
                result['pruning_ratios'][attn_couple.proj_name] = actual_ratio
                
                logger.debug(f"✂️ Pruned {len(heads_to_prune_indices)} heads from attention couple {i}")
                
            except Exception as e:
                result['errors'].append(f"Error pruning attention couple {i}: {str(e)}")
                result['success'] = False
        
        return result
    
    def _prune_standard_group(self, group: IsomorphicGroup, 
                            importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Prune standard groups (conv, fc, linear layers)."""
        
        result = {
            'success': True,
            'layers_pruned': [],
            'pruning_ratios': {},
            'warnings': [],
            'errors': []
        }
        
        for layer, layer_name in zip(group.layers, group.layer_names):
            try:
                if isinstance(layer, nn.Linear):
                    success = self._prune_linear_layer(layer, layer_name, group.pruning_ratio, importance_scores)
                elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                    success = self._prune_conv_layer(layer, layer_name, group.pruning_ratio, importance_scores)
                else:
                    result['warnings'].append(f"Unsupported layer type: {type(layer)}")
                    continue
                
                if success:
                    result['layers_pruned'].append(layer_name)
                    result['pruning_ratios'][layer_name] = group.pruning_ratio
                else:
                    result['warnings'].append(f"Failed to prune layer: {layer_name}")
                
            except Exception as e:
                result['errors'].append(f"Error pruning layer {layer_name}: {str(e)}")
                result['success'] = False
        
        return result
    
    def _select_neurons_by_importance(self, importance_scores: torch.Tensor, 
                                    num_to_prune: int) -> List[int]:
        """Select neurons to prune based on importance scores."""
        
        # Lower importance = higher priority for pruning
        _, indices = torch.sort(importance_scores)
        return indices[:num_to_prune].tolist()
    
    def _select_neurons_by_magnitude(self, layer: nn.Linear, num_to_prune: int) -> List[int]:
        """Select neurons to prune based on weight magnitude."""
        
        # Calculate L2 norm of each neuron's weights
        weight_norms = torch.norm(layer.weight, dim=1)
        
        # Select neurons with smallest norms
        _, indices = torch.sort(weight_norms)
        return indices[:num_to_prune].tolist()
    
    def _select_attention_heads_by_importance(self, importance_scores: torch.Tensor,
                                            heads_to_prune: int, num_heads: int) -> List[int]:
        """Select attention heads to prune based on importance scores."""
        
        # Reshape importance scores by heads (assuming head_dim = embed_dim // num_heads)
        head_dim = importance_scores.shape[0] // (3 * num_heads)  # QKV
        
        # Average importance across Q, K, V for each head
        head_importance = []
        for head_idx in range(num_heads):
            start_q = head_idx * head_dim
            end_q = start_q + head_dim
            start_k = num_heads * head_dim + head_idx * head_dim
            end_k = start_k + head_dim
            start_v = 2 * num_heads * head_dim + head_idx * head_dim
            end_v = start_v + head_dim
            
            q_importance = importance_scores[start_q:end_q].mean()
            k_importance = importance_scores[start_k:end_k].mean()
            v_importance = importance_scores[start_v:end_v].mean()
            
            head_importance.append((q_importance + k_importance + v_importance) / 3)
        
        head_importance = torch.tensor(head_importance)
        _, indices = torch.sort(head_importance)
        return indices[:heads_to_prune].tolist()
    
    def _select_attention_heads_by_magnitude(self, qkv_layer: nn.Linear,
                                           heads_to_prune: int, num_heads: int) -> List[int]:
        """Select attention heads to prune based on weight magnitude."""
        
        head_dim = qkv_layer.out_features // (3 * num_heads)
        
        # Calculate magnitude for each head
        head_magnitudes = []
        for head_idx in range(num_heads):
            start_q = head_idx * head_dim
            end_q = start_q + head_dim
            start_k = num_heads * head_dim + head_idx * head_dim
            end_k = start_k + head_dim
            start_v = 2 * num_heads * head_dim + head_idx * head_dim
            end_v = start_v + head_dim
            
            q_magnitude = torch.norm(qkv_layer.weight[start_q:end_q])
            k_magnitude = torch.norm(qkv_layer.weight[start_k:end_k])
            v_magnitude = torch.norm(qkv_layer.weight[start_v:end_v])
            
            head_magnitudes.append(q_magnitude + k_magnitude + v_magnitude)
        
        head_magnitudes = torch.tensor(head_magnitudes)
        _, indices = torch.sort(head_magnitudes)
        return indices[:heads_to_prune].tolist()
    
    def _prune_mlp_couple(self, mlp_couple: MLPCouple, neurons_to_prune: List[int]):
        """Apply coordinated pruning to MLP couple."""
        
        with torch.no_grad():
            # Create mask for neurons to keep
            hidden_dim = mlp_couple.get_hidden_dim()
            keep_mask = torch.ones(hidden_dim, dtype=torch.bool)
            keep_mask[neurons_to_prune] = False
            
            # Prune fc1 output dimensions
            mlp_couple.fc1.weight.data = mlp_couple.fc1.weight.data[keep_mask]
            if mlp_couple.fc1.bias is not None:
                mlp_couple.fc1.bias.data = mlp_couple.fc1.bias.data[keep_mask]
            
            # Update fc1 output features
            mlp_couple.fc1.out_features = keep_mask.sum().item()
            
            # Prune fc2 input dimensions
            mlp_couple.fc2.weight.data = mlp_couple.fc2.weight.data[:, keep_mask]
            mlp_couple.fc2.in_features = keep_mask.sum().item()
    
    def _prune_attention_couple(self, attn_couple: AttentionCouple, 
                              heads_to_prune: List[int], num_heads: int):
        """Apply coordinated pruning to attention couple."""
        
        with torch.no_grad():
            head_dim = attn_couple.get_embed_dim() // num_heads
            
            # Create mask for heads to keep
            heads_to_keep = [i for i in range(num_heads) if i not in heads_to_prune]
            
            # Build dimension masks for Q, K, V
            qkv_keep_mask = []
            
            # Q dimensions
            for head_idx in heads_to_keep:
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim
                qkv_keep_mask.extend(range(start_idx, end_idx))
            
            # K dimensions
            for head_idx in heads_to_keep:
                start_idx = num_heads * head_dim + head_idx * head_dim
                end_idx = start_idx + head_dim
                qkv_keep_mask.extend(range(start_idx, end_idx))
            
            # V dimensions
            for head_idx in heads_to_keep:
                start_idx = 2 * num_heads * head_dim + head_idx * head_dim
                end_idx = start_idx + head_dim
                qkv_keep_mask.extend(range(start_idx, end_idx))
            
            qkv_keep_mask = torch.tensor(qkv_keep_mask, dtype=torch.long)
            
            # Prune QKV layer
            attn_couple.qkv.weight.data = attn_couple.qkv.weight.data[qkv_keep_mask]
            if attn_couple.qkv.bias is not None:
                attn_couple.qkv.bias.data = attn_couple.qkv.bias.data[qkv_keep_mask]
            attn_couple.qkv.out_features = len(qkv_keep_mask)
            
            # Create mask for projection layer input (only one copy of each head)
            proj_keep_mask = []
            for head_idx in heads_to_keep:
                start_idx = head_idx * head_dim
                end_idx = start_idx + head_dim
                proj_keep_mask.extend(range(start_idx, end_idx))
            
            proj_keep_mask = torch.tensor(proj_keep_mask, dtype=torch.long)
            
            # Prune projection layer
            attn_couple.proj.weight.data = attn_couple.proj.weight.data[:, proj_keep_mask]
            attn_couple.proj.in_features = len(proj_keep_mask)
    
    def _prune_linear_layer(self, layer: nn.Linear, layer_name: str, pruning_ratio: float,
                          importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        """Prune a standard linear layer."""
        
        try:
            num_neurons = layer.out_features
            num_to_prune = int(num_neurons * pruning_ratio)
            
            if num_to_prune == 0:
                return True
            
            # Select neurons to prune
            if importance_scores and layer_name in importance_scores:
                neurons_to_prune = self._select_neurons_by_importance(
                    importance_scores[layer_name], num_to_prune
                )
            else:
                neurons_to_prune = self._select_neurons_by_magnitude(layer, num_to_prune)
            
            # Apply pruning
            with torch.no_grad():
                keep_mask = torch.ones(num_neurons, dtype=torch.bool)
                keep_mask[neurons_to_prune] = False
                
                layer.weight.data = layer.weight.data[keep_mask]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[keep_mask]
                layer.out_features = keep_mask.sum().item()
            
            return True
            
        except Exception as e:
            logger.error(f"Error pruning linear layer {layer_name}: {str(e)}")
            return False
    
    def _prune_conv_layer(self, layer: Union[nn.Conv2d, nn.Conv1d], layer_name: str, 
                         pruning_ratio: float, importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        """Prune a convolutional layer."""
        
        try:
            num_filters = layer.out_channels
            num_to_prune = int(num_filters * pruning_ratio)
            
            if num_to_prune == 0:
                return True
            
            # Select filters to prune based on magnitude
            with torch.no_grad():
                # Calculate L2 norm of each filter
                filter_norms = torch.norm(layer.weight.view(num_filters, -1), dim=1)
                _, indices = torch.sort(filter_norms)
                filters_to_prune = indices[:num_to_prune].tolist()
                
                # Create keep mask
                keep_mask = torch.ones(num_filters, dtype=torch.bool)
                keep_mask[filters_to_prune] = False
                
                # Apply pruning
                layer.weight.data = layer.weight.data[keep_mask]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[keep_mask]
                layer.out_channels = keep_mask.sum().item()
            
            return True
            
        except Exception as e:
            logger.error(f"Error pruning conv layer {layer_name}: {str(e)}")
            return False
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current pruning state."""
        
        current_params = self._count_parameters()
        
        stats = {
            'is_pruned': self.is_pruned,
            'original_params': self.original_params,
            'current_params': current_params,
            'params_reduction': 1.0 - (current_params / self.original_params) if self.original_params > 0 else 0.0,
            'pruning_history': self.pruning_history.copy(),
            'model_name': self.model_name
        }
        
        return stats
    
    def print_pruning_summary(self):
        """Print a comprehensive summary of the pruning state."""
        
        stats = self.get_pruning_statistics()
        
        print(f"\n🔧 Pruning Engine Summary for {self.model_name}")
        print("=" * 60)
        
        print(f"📊 Current State:")
        print(f"   Pruned: {'Yes' if stats['is_pruned'] else 'No'}")
        print(f"   Original parameters: {stats['original_params']:,}")
        print(f"   Current parameters: {stats['current_params']:,}")
        print(f"   Parameters reduction: {stats['params_reduction']:.1%}")
        
        if stats['pruning_history']:
            print(f"\n📋 Pruning History:")
            for i, entry in enumerate(stats['pruning_history'], 1):
                print(f"   Operation {i}:")
                print(f"      Groups: {', '.join(entry['groups'])}")
                print(f"      Layers pruned: {entry['layers_pruned']}")
                print(f"      Reduction: {entry['params_reduction']:.1%}")
        
        print("=" * 60)

