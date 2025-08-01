#!/usr/bin/env python3
"""
Metrics utilities for Multi-Agent LLM Pruning Framework

This module provides comprehensive metrics tracking, performance monitoring,
and evaluation utilities for the pruning pipeline.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class PruningMetrics:
    """
    Comprehensive metrics tracking for pruning experiments.
    Tracks accuracy, efficiency, and other performance indicators.
    """
    
    # Accuracy metrics
    accuracy_history: List[float] = field(default_factory=list)
    top5_accuracy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    
    # Efficiency metrics
    macs_history: List[int] = field(default_factory=list)
    params_history: List[int] = field(default_factory=list)
    inference_time_history: List[float] = field(default_factory=list)
    memory_usage_history: List[float] = field(default_factory=list)
    
    # Pruning-specific metrics
    sparsity_history: List[float] = field(default_factory=list)
    pruning_ratio_history: List[float] = field(default_factory=list)
    
    # Phase tracking
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Timestamps
    timestamps: List[float] = field(default_factory=list)
    
    def record_accuracy(self, accuracy: float, top5_accuracy: Optional[float] = None, 
                       loss: Optional[float] = None):
        """Record accuracy metrics."""
        self.accuracy_history.append(accuracy)
        if top5_accuracy is not None:
            self.top5_accuracy_history.append(top5_accuracy)
        if loss is not None:
            self.loss_history.append(loss)
        self.timestamps.append(time.time())
    
    def record_efficiency(self, macs: Optional[int] = None, params: Optional[int] = None,
                         inference_time: Optional[float] = None, memory_usage: Optional[float] = None):
        """Record efficiency metrics."""
        if macs is not None:
            self.macs_history.append(macs)
        if params is not None:
            self.params_history.append(params)
        if inference_time is not None:
            self.inference_time_history.append(inference_time)
        if memory_usage is not None:
            self.memory_usage_history.append(memory_usage)
    
    def record_pruning(self, sparsity: float, pruning_ratio: float):
        """Record pruning-specific metrics."""
        self.sparsity_history.append(sparsity)
        self.pruning_ratio_history.append(pruning_ratio)
    
    def record_phase_result(self, phase_name: str, results: Dict[str, Any]):
        """Record results for a specific phase."""
        self.phase_results[phase_name] = results.copy()
    
    def get_latest_accuracy(self) -> Optional[float]:
        """Get the most recent accuracy."""
        return self.accuracy_history[-1] if self.accuracy_history else None
    
    def get_accuracy_trend(self, window: int = 5) -> str:
        """Get accuracy trend over recent measurements."""
        if len(self.accuracy_history) < 2:
            return "insufficient_data"
        
        recent = self.accuracy_history[-min(window, len(self.accuracy_history)):]
        if len(recent) < 2:
            return "insufficient_data"
        
        trend = recent[-1] - recent[0]
        if abs(trend) < 0.001:  # Less than 0.1% change
            return "stable"
        elif trend > 0:
            return "improving"
        else:
            return "declining"
    
    def get_efficiency_summary(self) -> Dict[str, Any]:
        """Get summary of efficiency metrics."""
        summary = {}
        
        if self.macs_history:
            summary['macs'] = {
                'current': self.macs_history[-1],
                'reduction': (self.macs_history[0] - self.macs_history[-1]) / self.macs_history[0] if len(self.macs_history) > 1 else 0.0
            }
        
        if self.params_history:
            summary['params'] = {
                'current': self.params_history[-1],
                'reduction': (self.params_history[0] - self.params_history[-1]) / self.params_history[0] if len(self.params_history) > 1 else 0.0
            }
        
        if self.inference_time_history:
            summary['inference_time'] = {
                'current': self.inference_time_history[-1],
                'speedup': self.inference_time_history[0] / self.inference_time_history[-1] if len(self.inference_time_history) > 1 and self.inference_time_history[-1] > 0 else 1.0
            }
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'accuracy_history': self.accuracy_history,
            'top5_accuracy_history': self.top5_accuracy_history,
            'loss_history': self.loss_history,
            'macs_history': self.macs_history,
            'params_history': self.params_history,
            'inference_time_history': self.inference_time_history,
            'memory_usage_history': self.memory_usage_history,
            'sparsity_history': self.sparsity_history,
            'pruning_ratio_history': self.pruning_ratio_history,
            'phase_results': self.phase_results,
            'timestamps': self.timestamps
        }

def compute_macs(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> int:
    """
    Compute the number of Multiply-Accumulate operations (MACs) for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        
    Returns:
        Total number of MACs
    """
    
    def conv_mac_count(module, input, output):
        """Calculate MACs for convolution layers."""
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Get input and output dimensions
            batch_size = input[0].shape[0]
            output_dims = output.shape[2:]  # Spatial dimensions
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            # Calculate MACs
            kernel_flops = np.prod(kernel_dims) * in_channels // groups
            output_elements = batch_size * out_channels * np.prod(output_dims)
            macs = kernel_flops * output_elements
            
            module.__macs__ += macs
    
    def linear_mac_count(module, input, output):
        """Calculate MACs for linear layers."""
        if isinstance(module, nn.Linear):
            batch_size = input[0].shape[0]
            macs = batch_size * module.in_features * module.out_features
            module.__macs__ += macs
    
    def bn_mac_count(module, input, output):
        """Calculate MACs for batch normalization layers."""
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            batch_size = input[0].shape[0]
            num_features = input[0].numel() // batch_size
            # BN requires 2 operations per feature (subtract mean, divide by std)
            macs = 2 * batch_size * num_features
            module.__macs__ += macs
    
    def relu_mac_count(module, input, output):
        """Calculate MACs for ReLU layers (minimal)."""
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            # ReLU is essentially free in terms of MACs
            module.__macs__ += 0
    
    # Initialize MAC counters
    for module in model.modules():
        module.__macs__ = 0
    
    # Register hooks
    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(conv_mac_count))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_mac_count))
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            handles.append(module.register_forward_hook(bn_mac_count))
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            handles.append(module.register_forward_hook(relu_mac_count))
    
    # Create dummy input and run forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Sum up all MACs
    total_macs = sum(module.__macs__ for module in model.modules())
    
    # Clean up hooks and MAC counters
    for handle in handles:
        handle.remove()
    
    for module in model.modules():
        if hasattr(module, '__macs__'):
            delattr(module, '__macs__')
    
    return int(total_macs)

def compute_params(model: nn.Module) -> Dict[str, int]:
    """
    Compute detailed parameter counts for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter count details
    """
    
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # Count by layer type
    layer_params = {
        'conv': 0,
        'linear': 0,
        'bn': 0,
        'embedding': 0,
        'other': 0
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count
        
        # Categorize by layer type
        if 'conv' in name.lower():
            layer_params['conv'] += param_count
        elif 'linear' in name.lower() or 'fc' in name.lower():
            layer_params['linear'] += param_count
        elif 'bn' in name.lower() or 'norm' in name.lower():
            layer_params['bn'] += param_count
        elif 'embed' in name.lower():
            layer_params['embedding'] += param_count
        else:
            layer_params['other'] += param_count
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'by_layer_type': layer_params
    }

def compute_model_complexity(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
    """
    Compute comprehensive model complexity metrics.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        Dictionary with complexity metrics
    """
    
    # Compute parameters
    param_info = compute_params(model)
    
    # Compute MACs
    try:
        macs = compute_macs(model, input_shape)
        # Convert to GMACs (Giga MACs)
        gmacs = macs / 1e9
    except Exception as e:
        logger.warning(f"Failed to compute MACs: {e}")
        macs = 0
        gmacs = 0.0
    
    # Compute model size in MB
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    total_size_mb = param_size_mb + buffer_size_mb
    
    return {
        'parameters': param_info,
        'macs': macs,
        'gmacs': gmacs,
        'model_size_mb': total_size_mb,
        'param_size_mb': param_size_mb,
        'buffer_size_mb': buffer_size_mb,
        'input_shape': input_shape
    }

@dataclass
class AccuracyResult:
    """Container for accuracy measurement results."""
    top1_accuracy: float
    top5_accuracy: Optional[float] = None
    total_samples: int = 0
    correct_samples: int = 0
    loss: Optional[float] = None
    inference_time: Optional[float] = None

@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a specific time."""
    timestamp: float
    accuracy: float
    loss: float
    params_count: int
    model_size_mb: float
    inference_time_ms: float
    memory_usage_mb: Optional[float] = None
    flops: Optional[int] = None

class AccuracyTracker:
    """
    Comprehensive accuracy tracking with support for top-k accuracy,
    loss tracking, and performance monitoring.
    """
    
    def __init__(self, track_top5: bool = True, device: Optional[torch.device] = None):
        self.track_top5 = track_top5
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Reset tracking state
        self.reset()
        
        logger.info(f"📊 AccuracyTracker initialized (top5: {track_top5}, device: {device})")
    
    def reset(self):
        """Reset all tracking statistics."""
        self.total_samples = 0
        self.correct_top1 = 0
        self.correct_top5 = 0
        self.total_loss = 0.0
        self.batch_count = 0
        self.inference_times = []
        self.batch_accuracies = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor, 
              loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Update accuracy statistics with a batch of predictions.
        
        Args:
            outputs: Model outputs (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            loss: Optional loss tensor
            
        Returns:
            Dictionary with current batch metrics
        """
        
        batch_size = targets.size(0)
        self.total_samples += batch_size
        self.batch_count += 1
        
        # Calculate top-1 accuracy
        _, pred_top1 = outputs.topk(1, 1, True, True)
        pred_top1 = pred_top1.t()
        correct_top1 = pred_top1.eq(targets.view(1, -1).expand_as(pred_top1))
        
        batch_correct_top1 = correct_top1[:1].reshape(-1).float().sum(0, keepdim=True).item()
        self.correct_top1 += batch_correct_top1
        
        batch_metrics = {
            'batch_top1_accuracy': batch_correct_top1 / batch_size,
            'batch_size': batch_size
        }
        
        # Calculate top-5 accuracy if requested
        if self.track_top5 and outputs.size(1) >= 5:
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(targets.view(1, -1).expand_as(pred_top5))
            
            batch_correct_top5 = correct_top5[:5].reshape(-1).float().sum(0, keepdim=True).item()
            self.correct_top5 += batch_correct_top5
            
            batch_metrics['batch_top5_accuracy'] = batch_correct_top5 / batch_size
        
        # Track loss if provided
        if loss is not None:
            batch_loss = loss.item()
            self.total_loss += batch_loss
            batch_metrics['batch_loss'] = batch_loss
        
        # Store batch accuracy for variance calculation
        self.batch_accuracies.append(batch_metrics['batch_top1_accuracy'])
        
        return batch_metrics
    
    def get_accuracy(self) -> AccuracyResult:
        """Get current accuracy statistics."""
        
        if self.total_samples == 0:
            return AccuracyResult(
                top1_accuracy=0.0,
                top5_accuracy=0.0 if self.track_top5 else None,
                total_samples=0,
                correct_samples=0
            )
        
        top1_acc = self.correct_top1 / self.total_samples
        top5_acc = self.correct_top5 / self.total_samples if self.track_top5 else None
        avg_loss = self.total_loss / self.batch_count if self.batch_count > 0 else None
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else None
        
        return AccuracyResult(
            top1_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            total_samples=self.total_samples,
            correct_samples=int(self.correct_top1),
            loss=avg_loss,
            inference_time=avg_inference_time
        )
    
    def evaluate_model(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                      criterion: Optional[nn.Module] = None, 
                      max_batches: Optional[int] = None) -> AccuracyResult:
        """
        Evaluate a model on a dataset and return comprehensive results.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            criterion: Loss function (optional)
            max_batches: Maximum number of batches to evaluate (None for all)
            
        Returns:
            AccuracyResult with comprehensive evaluation metrics
        """
        
        model.eval()
        self.reset()
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass with timing
                inference_start = time.time()
                output = model(data)
                inference_time = (time.time() - inference_start) * 1000  # Convert to ms
                self.inference_times.append(inference_time)
                
                # Calculate loss if criterion provided
                loss = None
                if criterion is not None:
                    loss = criterion(output, target)
                
                # Update accuracy statistics
                self.update(output, target, loss)
        
        total_time = time.time() - start_time
        result = self.get_accuracy()
        result.inference_time = total_time
        
        logger.info(f"📊 Model evaluation completed in {total_time:.2f}s")
        logger.info(f"   Top-1 Accuracy: {result.top1_accuracy:.1%}")
        if result.top5_accuracy is not None:
            logger.info(f"   Top-5 Accuracy: {result.top5_accuracy:.1%}")
        if result.loss is not None:
            logger.info(f"   Average Loss: {result.loss:.4f}")
        
        return result
    
    def get_statistics(self) -> Dict[str, float]:
        """Get detailed statistics including variance and confidence intervals."""
        
        result = self.get_accuracy()
        stats = {
            'top1_accuracy': result.top1_accuracy,
            'total_samples': result.total_samples,
            'correct_samples': result.correct_samples
        }
        
        if result.top5_accuracy is not None:
            stats['top5_accuracy'] = result.top5_accuracy
        
        if result.loss is not None:
            stats['average_loss'] = result.loss
        
        # Calculate accuracy variance across batches
        if len(self.batch_accuracies) > 1:
            acc_variance = np.var(self.batch_accuracies)
            acc_std = np.std(self.batch_accuracies)
            stats.update({
                'accuracy_variance': acc_variance,
                'accuracy_std': acc_std,
                'accuracy_min': min(self.batch_accuracies),
                'accuracy_max': max(self.batch_accuracies)
            })
        
        # Inference time statistics
        if self.inference_times:
            stats.update({
                'avg_inference_time_ms': np.mean(self.inference_times),
                'min_inference_time_ms': min(self.inference_times),
                'max_inference_time_ms': max(self.inference_times),
                'std_inference_time_ms': np.std(self.inference_times)
            })
        
        return stats

class PerformanceMetrics:
    """
    Comprehensive performance metrics tracking for the entire pruning pipeline.
    
    This class tracks model performance, resource usage, and efficiency metrics
    throughout the pruning process.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.snapshots: deque = deque(maxlen=history_size)
        self.phase_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info(f"📈 PerformanceMetrics initialized (history_size: {history_size})")
    
    def take_snapshot(self, model: nn.Module, accuracy: float, loss: float = 0.0,
                     phase: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> PerformanceSnapshot:
        """
        Take a performance snapshot of the current model state.
        
        Args:
            model: Current model
            accuracy: Current accuracy
            loss: Current loss
            phase: Current phase/stage name
            metadata: Additional metadata
            
        Returns:
            PerformanceSnapshot object
        """
        
        # Count parameters
        params_count = sum(p.numel() for p in model.parameters())
        
        # Estimate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Measure inference time (approximate)
        inference_time_ms = self._measure_inference_time(model)
        
        # Get memory usage if CUDA is available
        memory_usage_mb = None
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            accuracy=accuracy,
            loss=loss,
            params_count=params_count,
            model_size_mb=model_size_mb,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
        # Store phase-specific metrics
        if phase:
            phase_data = {
                'snapshot': snapshot,
                'metadata': metadata or {}
            }
            self.phase_metrics[phase].append(phase_data)
        
        logger.debug(f"📸 Performance snapshot taken for phase: {phase}")
        logger.debug(f"   Accuracy: {accuracy:.1%}, Params: {params_count:,}, Size: {model_size_mb:.1f}MB")
        
        return snapshot
    
    def _measure_inference_time(self, model: nn.Module, num_runs: int = 10) -> float:
        """Measure approximate inference time for the model."""
        
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input (assuming image classification)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        
        return avg_time_ms
    
    def get_performance_trend(self, metric: str = 'accuracy', 
                            window_size: int = 10) -> List[float]:
        """
        Get performance trend for a specific metric.
        
        Args:
            metric: Metric name ('accuracy', 'loss', 'params_count', etc.)
            window_size: Moving average window size
            
        Returns:
            List of smoothed metric values
        """
        
        if not self.snapshots:
            return []
        
        # Extract metric values
        values = []
        for snapshot in self.snapshots:
            if hasattr(snapshot, metric):
                values.append(getattr(snapshot, metric))
        
        if not values:
            return []
        
        # Apply moving average smoothing
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window_values = values[start_idx:i+1]
            smoothed.append(sum(window_values) / len(window_values))
        
        return smoothed
    
    def get_phase_summary(self, phase: str) -> Dict[str, Any]:
        """Get performance summary for a specific phase."""
        
        if phase not in self.phase_metrics:
            return {}
        
        phase_data = self.phase_metrics[phase]
        if not phase_data:
            return {}
        
        # Extract metrics
        accuracies = [data['snapshot'].accuracy for data in phase_data]
        losses = [data['snapshot'].loss for data in phase_data]
        params = [data['snapshot'].params_count for data in phase_data]
        sizes = [data['snapshot'].model_size_mb for data in phase_data]
        inference_times = [data['snapshot'].inference_time_ms for data in phase_data]
        
        summary = {
            'phase': phase,
            'num_snapshots': len(phase_data),
            'accuracy': {
                'initial': accuracies[0],
                'final': accuracies[-1],
                'best': max(accuracies),
                'worst': min(accuracies),
                'average': sum(accuracies) / len(accuracies)
            },
            'loss': {
                'initial': losses[0],
                'final': losses[-1],
                'best': min(losses),
                'worst': max(losses),
                'average': sum(losses) / len(losses)
            },
            'parameters': {
                'initial': params[0],
                'final': params[-1],
                'reduction': 1.0 - (params[-1] / params[0]) if params[0] > 0 else 0.0,
                'reduction_count': params[0] - params[-1]
            },
            'model_size_mb': {
                'initial': sizes[0],
                'final': sizes[-1],
                'reduction': 1.0 - (sizes[-1] / sizes[0]) if sizes[0] > 0 else 0.0
            },
            'inference_time_ms': {
                'initial': inference_times[0],
                'final': inference_times[-1],
                'average': sum(inference_times) / len(inference_times),
                'speedup': inference_times[0] / inference_times[-1] if inference_times[-1] > 0 else 1.0
            }
        }
        
        return summary
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all phases."""
        
        if not self.snapshots:
            return {}
        
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        # Calculate overall metrics
        all_accuracies = [s.accuracy for s in self.snapshots]
        all_losses = [s.loss for s in self.snapshots]
        all_params = [s.params_count for s in self.snapshots]
        all_sizes = [s.model_size_mb for s in self.snapshots]
        all_inference_times = [s.inference_time_ms for s in self.snapshots]
        
        summary = {
            'total_snapshots': len(self.snapshots),
            'duration_seconds': last_snapshot.timestamp - first_snapshot.timestamp,
            'phases_tracked': list(self.phase_metrics.keys()),
            'accuracy': {
                'initial': first_snapshot.accuracy,
                'final': last_snapshot.accuracy,
                'best': max(all_accuracies),
                'worst': min(all_accuracies),
                'change': last_snapshot.accuracy - first_snapshot.accuracy
            },
            'parameters': {
                'initial': first_snapshot.params_count,
                'final': last_snapshot.params_count,
                'reduction_ratio': 1.0 - (last_snapshot.params_count / first_snapshot.params_count),
                'reduction_count': first_snapshot.params_count - last_snapshot.params_count
            },
            'model_size': {
                'initial_mb': first_snapshot.model_size_mb,
                'final_mb': last_snapshot.model_size_mb,
                'reduction_ratio': 1.0 - (last_snapshot.model_size_mb / first_snapshot.model_size_mb),
                'reduction_mb': first_snapshot.model_size_mb - last_snapshot.model_size_mb
            },
            'inference_speedup': first_snapshot.inference_time_ms / last_snapshot.inference_time_ms if last_snapshot.inference_time_ms > 0 else 1.0
        }
        
        return summary
    
    def print_summary(self, phase: Optional[str] = None):
        """Print a comprehensive performance summary."""
        
        if phase:
            summary = self.get_phase_summary(phase)
            if not summary:
                print(f"📈 No performance data available for phase: {phase}")
                return
            
            print(f"\n📈 Performance Summary - Phase: {phase}")
            print("=" * 60)
            
            print(f"📊 Accuracy:")
            print(f"   Initial: {summary['accuracy']['initial']:.1%}")
            print(f"   Final: {summary['accuracy']['final']:.1%}")
            print(f"   Best: {summary['accuracy']['best']:.1%}")
            print(f"   Change: {summary['accuracy']['final'] - summary['accuracy']['initial']:+.1%}")
            
            print(f"\n🔧 Parameters:")
            print(f"   Initial: {summary['parameters']['initial']:,}")
            print(f"   Final: {summary['parameters']['final']:,}")
            print(f"   Reduction: {summary['parameters']['reduction']:.1%} ({summary['parameters']['reduction_count']:,} params)")
            
            print(f"\n⚡ Performance:")
            print(f"   Initial inference: {summary['inference_time_ms']['initial']:.1f}ms")
            print(f"   Final inference: {summary['inference_time_ms']['final']:.1f}ms")
            print(f"   Speedup: {summary['inference_time_ms']['speedup']:.2f}x")
            
        else:
            summary = self.get_overall_summary()
            if not summary:
                print("📈 No performance data available")
                return
            
            print(f"\n📈 Overall Performance Summary")
            print("=" * 60)
            
            print(f"📊 Overview:")
            print(f"   Total snapshots: {summary['total_snapshots']}")
            print(f"   Duration: {summary['duration_seconds']:.1f}s")
            print(f"   Phases: {', '.join(summary['phases_tracked'])}")
            
            print(f"\n🎯 Accuracy:")
            print(f"   Initial: {summary['accuracy']['initial']:.1%}")
            print(f"   Final: {summary['accuracy']['final']:.1%}")
            print(f"   Best achieved: {summary['accuracy']['best']:.1%}")
            print(f"   Overall change: {summary['accuracy']['change']:+.1%}")
            
            print(f"\n🔧 Model Compression:")
            print(f"   Parameter reduction: {summary['parameters']['reduction_ratio']:.1%}")
            print(f"   Size reduction: {summary['model_size']['reduction_ratio']:.1%}")
            print(f"   Inference speedup: {summary['inference_speedup']:.2f}x")
        
        print("=" * 60)

