#!/usr/bin/env python3
"""
Importance Criteria for Multi-Agent LLM Pruning Framework

This module provides various importance criteria for determining which
parameters to prune, including magnitude-based, gradient-based, and
advanced importance measures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable, Union
from abc import ABC, abstractmethod
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ImportanceScore:
    """Container for importance scores with metadata."""
    scores: torch.Tensor
    criterion: str
    layer_name: str
    computation_time: float
    metadata: Dict[str, Any]

class ImportanceCriterion(ABC):
    """Abstract base class for importance criteria."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module, dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> ImportanceScore:
        """Compute importance scores for a layer."""
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

class MagnitudeL1Criterion(ImportanceCriterion):
    """L1 magnitude-based importance criterion."""
    
    def __init__(self):
        super().__init__("magnitude_l1")
    
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module, dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> ImportanceScore:
        """Compute L1 magnitude importance scores."""
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        with torch.no_grad():
            if isinstance(layer, nn.Linear):
                # For linear layers, compute L1 norm of each output neuron
                scores = torch.norm(layer.weight, p=1, dim=1)
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                # For conv layers, compute L1 norm of each filter
                scores = torch.norm(layer.weight.view(layer.weight.size(0), -1), p=1, dim=1)
            else:
                # Fallback: flatten and compute element-wise L1
                scores = torch.abs(layer.weight.flatten())
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            computation_time = 0.0
        
        return ImportanceScore(
            scores=scores,
            criterion=self.name,
            layer_name=layer_name,
            computation_time=computation_time,
            metadata={'norm_type': 'L1', 'layer_type': type(layer).__name__}
        )

class MagnitudeL2Criterion(ImportanceCriterion):
    """L2 magnitude-based importance criterion."""
    
    def __init__(self):
        super().__init__("magnitude_l2")
    
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module, dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> ImportanceScore:
        """Compute L2 magnitude importance scores."""
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        with torch.no_grad():
            if isinstance(layer, nn.Linear):
                # For linear layers, compute L2 norm of each output neuron
                scores = torch.norm(layer.weight, p=2, dim=1)
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                # For conv layers, compute L2 norm of each filter
                scores = torch.norm(layer.weight.view(layer.weight.size(0), -1), p=2, dim=1)
            else:
                # Fallback: flatten and compute element-wise L2
                scores = torch.norm(layer.weight.flatten(), p=2)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            computation_time = 0.0
        
        return ImportanceScore(
            scores=scores,
            criterion=self.name,
            layer_name=layer_name,
            computation_time=computation_time,
            metadata={'norm_type': 'L2', 'layer_type': type(layer).__name__}
        )

class TaylorCriterion(ImportanceCriterion):
    """Taylor expansion-based importance criterion."""
    
    def __init__(self, num_samples: int = 1000):
        super().__init__("taylor")
        self.num_samples = num_samples
    
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module, dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> ImportanceScore:
        """Compute Taylor expansion importance scores."""
        
        if dataloader is None:
            logger.warning(f"No dataloader provided for Taylor criterion, falling back to L2 magnitude")
            fallback = MagnitudeL2Criterion()
            return fallback.compute_importance(layer, layer_name, model, **kwargs)
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # FIXED: Determine device from model parameters
        model_device = next(model.parameters()).device
        logger.debug(f"🎯 Model device for {layer_name}: {model_device}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Accumulate gradients
        accumulated_gradients = None
        samples_processed = 0
        
        try:
            for batch_idx, (data, target) in enumerate(dataloader):
                if samples_processed >= self.num_samples:
                    break
                
                # FIXED: Move data to the same device as the model
                data = data.to(model_device)
                target = target.to(model_device)
                
                # Forward pass
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                model.zero_grad()
                loss.backward()
                
                # Accumulate gradients for this layer
                if layer.weight.grad is not None:
                    current_grad = layer.weight.grad.clone()
                    
                    if accumulated_gradients is None:
                        accumulated_gradients = current_grad
                    else:
                        accumulated_gradients += current_grad
                
                samples_processed += data.size(0)
                
                # Clear gradients for next iteration
                model.zero_grad()
            
            # Compute Taylor importance: |weight * gradient|
            if accumulated_gradients is not None:
                # Normalize by number of batches
                accumulated_gradients /= (batch_idx + 1)
                
                # Compute Taylor scores
                taylor_scores = torch.abs(layer.weight * accumulated_gradients)
                
                if isinstance(layer, nn.Linear):
                    # Sum across input dimensions for each output neuron
                    scores = torch.sum(taylor_scores, dim=1)
                elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                    # Sum across all dimensions except output channels
                    scores = torch.sum(taylor_scores.view(taylor_scores.size(0), -1), dim=1)
                else:
                    scores = taylor_scores.flatten()
            else:
                logger.warning(f"No gradients available for layer {layer_name}, using magnitude fallback")
                fallback = MagnitudeL2Criterion()
                return fallback.compute_importance(layer, layer_name, model, **kwargs)
        
        except Exception as e:
            logger.error(f"Error computing Taylor importance for {layer_name}: {str(e)}")
            fallback = MagnitudeL2Criterion()
            return fallback.compute_importance(layer, layer_name, model, **kwargs)
        
        finally:
            model.zero_grad()  # Clean up gradients
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            computation_time = 0.0
        
        return ImportanceScore(
            scores=scores,
            criterion=self.name,
            layer_name=layer_name,
            computation_time=computation_time,
            metadata={
                'num_samples': samples_processed,
                'layer_type': type(layer).__name__,
                'method': 'first_order_taylor',
                'device': str(model_device)
            }
        )

class GradientCriterion(ImportanceCriterion):
    """Gradient-based importance criterion."""
    
    def __init__(self, num_samples: int = 500):
        super().__init__("gradient")
        self.num_samples = num_samples
    
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module, dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> ImportanceScore:
        """Compute gradient-based importance scores."""
        
        if dataloader is None:
            logger.warning(f"No dataloader provided for Gradient criterion, falling back to L2 magnitude")
            fallback = MagnitudeL2Criterion()
            return fallback.compute_importance(layer, layer_name, model, **kwargs)
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # FIXED: Determine device from model parameters
        model_device = next(model.parameters()).device
        
        model.train()  # Set to training mode for gradient computation
        
        accumulated_gradients = None
        samples_processed = 0
        
        try:
            for batch_idx, (data, target) in enumerate(dataloader):
                if samples_processed >= self.num_samples:
                    break
                
                # FIXED: Move data to the same device as the model
                data = data.to(model_device)
                target = target.to(model_device)
                
                # Forward pass
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                model.zero_grad()
                loss.backward()
                
                # Accumulate gradients
                if layer.weight.grad is not None:
                    current_grad = torch.abs(layer.weight.grad.clone())
                    
                    if accumulated_gradients is None:
                        accumulated_gradients = current_grad
                    else:
                        accumulated_gradients += current_grad
                
                samples_processed += data.size(0)
                model.zero_grad()
            
            # Compute importance scores from accumulated gradients
            if accumulated_gradients is not None:
                # Normalize by number of batches
                accumulated_gradients /= (batch_idx + 1)
                
                if isinstance(layer, nn.Linear):
                    # Sum across input dimensions for each output neuron
                    scores = torch.sum(accumulated_gradients, dim=1)
                elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                    # Sum across all dimensions except output channels
                    scores = torch.sum(accumulated_gradients.view(accumulated_gradients.size(0), -1), dim=1)
                else:
                    scores = accumulated_gradients.flatten()
            else:
                logger.warning(f"No gradients available for layer {layer_name}, using magnitude fallback")
                fallback = MagnitudeL2Criterion()
                return fallback.compute_importance(layer, layer_name, model, **kwargs)
        
        except Exception as e:
            logger.error(f"Error computing Gradient importance for {layer_name}: {str(e)}")
            fallback = MagnitudeL2Criterion()
            return fallback.compute_importance(layer, layer_name, model, **kwargs)
        
        finally:
            model.zero_grad()  # Clean up gradients
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            computation_time = 0.0
        
        return ImportanceScore(
            scores=scores,
            criterion=self.name,
            layer_name=layer_name,
            computation_time=computation_time,
            metadata={
                'num_samples': samples_processed,
                'layer_type': type(layer).__name__,
                'method': 'gradient_accumulation',
                'device': str(model_device)
            }
        )

class RandomCriterion(ImportanceCriterion):
    """Random importance criterion (baseline for comparison)."""
    
    def __init__(self, seed: int = 42):
        super().__init__("random")
        self.seed = seed
    
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module, dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> ImportanceScore:
        """Compute random importance scores."""
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        
        with torch.no_grad():
            if isinstance(layer, nn.Linear):
                scores = torch.rand(layer.out_features)
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                scores = torch.rand(layer.out_channels)
            else:
                scores = torch.rand(layer.weight.numel())
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            computation_time = 0.0
        
        return ImportanceScore(
            scores=scores,
            criterion=self.name,
            layer_name=layer_name,
            computation_time=computation_time,
            metadata={'seed': self.seed, 'layer_type': type(layer).__name__}
        )

class ImportanceCriteria:
    """
    Manager class for different importance criteria.
    
    This class provides a unified interface for computing importance scores
    using various criteria and supports caching for efficiency.
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, ImportanceScore] = {}
        
        # Initialize available criteria
        self.criteria = {
            'magnitude_l1': MagnitudeL1Criterion(),
            'magnitude_l2': MagnitudeL2Criterion(),
            'taylor': TaylorCriterion(),
            'gradient': GradientCriterion(),
            'random': RandomCriterion()
        }
        
        logger.info(f"🎯 ImportanceCriteria initialized with {len(self.criteria)} criteria")
        logger.info(f"   Available criteria: {list(self.criteria.keys())}")
        logger.info(f"   Caching: {'enabled' if cache_enabled else 'disabled'}")
    
    def compute_importance(self, layer: nn.Module, layer_name: str, 
                          criterion: str, model: nn.Module,
                          dataloader: Optional[torch.utils.data.DataLoader] = None,
                          force_recompute: bool = False,
                          **kwargs) -> ImportanceScore:
        """
        Compute importance scores for a layer using the specified criterion.
        
        Args:
            layer: The neural network layer
            layer_name: Name/identifier of the layer
            criterion: Name of the importance criterion to use
            model: The complete model (needed for gradient-based criteria)
            dataloader: DataLoader for gradient-based criteria
            force_recompute: Whether to bypass cache and recompute
            **kwargs: Additional arguments for the criterion
            
        Returns:
            ImportanceScore object containing scores and metadata
        """
        
        # Check cache first
        cache_key = f"{layer_name}_{criterion}"
        if self.cache_enabled and not force_recompute and cache_key in self.cache:
            logger.debug(f"📋 Using cached importance scores for {layer_name} ({criterion})")
            return self.cache[cache_key]
        
        # Validate criterion
        if criterion not in self.criteria:
            logger.error(f"Unknown importance criterion: {criterion}")
            logger.info(f"Available criteria: {list(self.criteria.keys())}")
            raise ValueError(f"Unknown importance criterion: {criterion}")
        
        # Compute importance scores
        logger.debug(f"🎯 Computing {criterion} importance for layer: {layer_name}")
        
        try:
            importance_score = self.criteria[criterion].compute_importance(
                layer=layer,
                layer_name=layer_name,
                model=model,
                dataloader=dataloader,
                **kwargs
            )
            
            # Cache the result
            if self.cache_enabled:
                self.cache[cache_key] = importance_score
                logger.debug(f"💾 Cached importance scores for {layer_name} ({criterion})")
            
            logger.debug(f"✅ Computed {criterion} importance for {layer_name} "
                        f"in {importance_score.computation_time:.3f}s")
            
            return importance_score
            
        except Exception as e:
            logger.error(f"❌ Failed to compute {criterion} importance for {layer_name}: {str(e)}")
            raise
    
    def compute_all_criteria(self, layer: nn.Module, layer_name: str, 
                           model: nn.Module,
                           dataloader: Optional[torch.utils.data.DataLoader] = None,
                           criteria_subset: Optional[List[str]] = None,
                           **kwargs) -> Dict[str, ImportanceScore]:
        """
        Compute importance scores using all available criteria.
        
        Args:
            layer: The neural network layer
            layer_name: Name/identifier of the layer
            model: The complete model
            dataloader: DataLoader for gradient-based criteria
            criteria_subset: Subset of criteria to compute (None for all)
            **kwargs: Additional arguments for criteria
            
        Returns:
            Dictionary mapping criterion names to ImportanceScore objects
        """
        
        criteria_to_compute = criteria_subset or list(self.criteria.keys())
        results = {}
        
        logger.info(f"🎯 Computing importance scores for {layer_name} using {len(criteria_to_compute)} criteria")
        
        for criterion in criteria_to_compute:
            try:
                results[criterion] = self.compute_importance(
                    layer=layer,
                    layer_name=layer_name,
                    criterion=criterion,
                    model=model,
                    dataloader=dataloader,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to compute {criterion} for {layer_name}: {str(e)}")
                continue
        
        logger.info(f"✅ Computed {len(results)}/{len(criteria_to_compute)} importance criteria for {layer_name}")
        return results
    
    def get_best_criterion(self, layer: nn.Module, layer_name: str, 
                          model: nn.Module,
                          dataloader: Optional[torch.utils.data.DataLoader] = None,
                          **kwargs) -> str:
        """
        Determine the best importance criterion for a given layer.
        
        This is a heuristic-based selection that considers layer type,
        data availability, and computational efficiency.
        
        Args:
            layer: The neural network layer
            layer_name: Name/identifier of the layer
            model: The complete model
            dataloader: DataLoader availability
            **kwargs: Additional arguments
            
        Returns:
            Name of the recommended criterion
        """
        
        # Heuristic-based selection
        if dataloader is not None:
            # Prefer gradient-based methods when data is available
            if isinstance(layer, nn.Linear) and 'head' not in layer_name.lower():
                return 'taylor'  # Taylor works well for linear layers
            elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                return 'gradient'  # Gradient-based for conv layers
        
        # Fallback to magnitude-based methods
        if isinstance(layer, nn.Linear):
            return 'magnitude_l2'  # L2 norm for linear layers
        elif isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            return 'magnitude_l1'  # L1 norm for conv layers
        
        # Default fallback
        return 'magnitude_l2'
    
    def clear_cache(self, layer_name: Optional[str] = None, criterion: Optional[str] = None):
        """
        Clear importance score cache.
        
        Args:
            layer_name: Clear cache for specific layer (None for all layers)
            criterion: Clear cache for specific criterion (None for all criteria)
        """
        
        if layer_name is None and criterion is None:
            # Clear all cache
            cleared_count = len(self.cache)
            self.cache.clear()
            logger.info(f"🗑️ Cleared all cached importance scores ({cleared_count} entries)")
        else:
            # Clear specific entries
            keys_to_remove = []
            for key in self.cache.keys():
                key_layer, key_criterion = key.split('_', 1)
                if ((layer_name is None or key_layer == layer_name) and
                    (criterion is None or key_criterion == criterion)):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logger.info(f"🗑️ Cleared {len(keys_to_remove)} cached importance score entries")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about the importance score cache."""
        
        if not self.cache_enabled:
            return {'cache_enabled': False}
        
        criteria_counts = {}
        layer_counts = {}
        total_computation_time = 0.0
        
        for key, score in self.cache.items():
            layer_name, criterion = key.split('_', 1)
            
            criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1
            layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
            total_computation_time += score.computation_time
        
        return {
            'cache_enabled': True,
            'total_entries': len(self.cache),
            'unique_layers': len(layer_counts),
            'unique_criteria': len(criteria_counts),
            'criteria_distribution': criteria_counts,
            'layer_distribution': layer_counts,
            'total_computation_time': total_computation_time,
            'average_computation_time': total_computation_time / len(self.cache) if self.cache else 0.0
        }
    
    def print_cache_summary(self):
        """Print a summary of the importance score cache."""
        
        stats = self.get_cache_statistics()
        
        print(f"\n🎯 Importance Criteria Cache Summary")
        print("=" * 50)
        
        if not stats['cache_enabled']:
            print("   Cache is disabled")
            return
        
        print(f"📊 Cache Statistics:")
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Unique layers: {stats['unique_layers']}")
        print(f"   Unique criteria: {stats['unique_criteria']}")
        print(f"   Total computation time: {stats['total_computation_time']:.2f}s")
        print(f"   Average computation time: {stats['average_computation_time']:.3f}s")
        
        if stats['criteria_distribution']:
            print(f"\n📋 Criteria Distribution:")
            for criterion, count in stats['criteria_distribution'].items():
                print(f"   {criterion}: {count} entries")
        
        print("=" * 50)

