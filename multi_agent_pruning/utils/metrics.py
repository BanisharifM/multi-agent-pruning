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

