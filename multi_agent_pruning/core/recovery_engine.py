#!/usr/bin/env python3
"""
Recovery Engine for Multi-Agent LLM Pruning Framework

This module provides recovery mechanisms for handling pruning failures,
catastrophic accuracy drops, and model restoration capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import copy
import time
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RecoveryCheckpoint:
    """Represents a recovery checkpoint with model state and metadata."""
    model_state: Dict[str, torch.Tensor]
    accuracy: float
    params_count: int
    pruning_ratio: float
    timestamp: float
    checkpoint_id: str
    metadata: Dict[str, Any]

@dataclass
class RecoveryAction:
    """Represents a recovery action taken by the engine."""
    action_type: str  # 'restore', 'reduce_ratio', 'change_criterion', 'emergency_stop'
    description: str
    success: bool
    recovery_time: float
    restored_accuracy: Optional[float] = None
    restored_params: Optional[int] = None

class RecoveryEngine:
    """
    Recovery engine that handles pruning failures and provides fallback mechanisms.
    
    This engine monitors pruning operations, detects failures, and automatically
    applies recovery strategies to maintain model stability and performance.
    """
    
    def __init__(self, model: nn.Module, model_name: Optional[str] = None,
                 max_checkpoints: int = 5, min_accuracy_threshold: float = 0.1):
        self.model = model
        self.model_name = model_name or "unknown_model"
        self.max_checkpoints = max_checkpoints
        self.min_accuracy_threshold = min_accuracy_threshold
        
        # Recovery state
        self.checkpoints: List[RecoveryCheckpoint] = []
        self.recovery_history: List[RecoveryAction] = []
        self.is_monitoring = False
        self.failure_count = 0
        
        # Thresholds for failure detection
        self.catastrophic_threshold = 0.3  # 30% accuracy drop is catastrophic
        self.warning_threshold = 0.1       # 10% accuracy drop is concerning
        self.max_failures = 3              # Maximum failures before emergency stop
        
        logger.info(f"🛡️ Recovery engine initialized for {self.model_name}")
        logger.info(f"   Max checkpoints: {max_checkpoints}")
        logger.info(f"   Min accuracy threshold: {min_accuracy_threshold:.1%}")
        logger.info(f"   Catastrophic threshold: {self.catastrophic_threshold:.1%}")
    
    def create_checkpoint(self, accuracy: float, pruning_ratio: float = 0.0,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a recovery checkpoint with current model state.
        
        Args:
            accuracy: Current model accuracy
            pruning_ratio: Current pruning ratio
            metadata: Additional metadata to store
            
        Returns:
            Checkpoint ID
        """
        
        checkpoint_id = f"checkpoint_{len(self.checkpoints)}_{int(time.time())}"
        
        # Save model state
        model_state = {
            name: param.clone().detach().cpu()
            for name, param in self.model.named_parameters()
        }
        
        # Count parameters
        params_count = sum(p.numel() for p in self.model.parameters())
        
        # Create checkpoint
        checkpoint = RecoveryCheckpoint(
            model_state=model_state,
            accuracy=accuracy,
            params_count=params_count,
            pruning_ratio=pruning_ratio,
            timestamp=time.time(),
            checkpoint_id=checkpoint_id,
            metadata=metadata or {}
        )
        
        # Add to checkpoints list
        self.checkpoints.append(checkpoint)
        
        # Maintain maximum number of checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            removed_checkpoint = self.checkpoints.pop(0)
            logger.debug(f"🗑️ Removed old checkpoint: {removed_checkpoint.checkpoint_id}")
        
        logger.info(f"💾 Created checkpoint: {checkpoint_id}")
        logger.info(f"   Accuracy: {accuracy:.1%}, Params: {params_count:,}, Ratio: {pruning_ratio:.1%}")
        
        return checkpoint_id
    
    def detect_failure(self, current_accuracy: float, baseline_accuracy: float,
                      current_params: int, original_params: int) -> Dict[str, Any]:
        """
        Detect if the current state represents a failure condition.
        
        Args:
            current_accuracy: Current model accuracy
            baseline_accuracy: Baseline/reference accuracy
            current_params: Current parameter count
            original_params: Original parameter count
            
        Returns:
            Dictionary with failure detection results
        """
        
        accuracy_drop = baseline_accuracy - current_accuracy
        accuracy_drop_ratio = accuracy_drop / baseline_accuracy if baseline_accuracy > 0 else 0
        params_reduction = 1.0 - (current_params / original_params) if original_params > 0 else 0
        
        failure_info = {
            'is_failure': False,
            'failure_type': None,
            'severity': 'none',
            'accuracy_drop': accuracy_drop,
            'accuracy_drop_ratio': accuracy_drop_ratio,
            'params_reduction': params_reduction,
            'recommendations': []
        }
        
        # Check for catastrophic failure
        if accuracy_drop_ratio >= self.catastrophic_threshold:
            failure_info.update({
                'is_failure': True,
                'failure_type': 'catastrophic_accuracy_drop',
                'severity': 'critical',
                'recommendations': ['immediate_restore', 'reduce_pruning_ratio', 'change_strategy']
            })
            logger.error(f"💥 CATASTROPHIC FAILURE DETECTED!")
            logger.error(f"   Accuracy drop: {accuracy_drop:.1%} ({accuracy_drop_ratio:.1%} relative)")
            
        # Check for concerning accuracy drop
        elif accuracy_drop_ratio >= self.warning_threshold:
            failure_info.update({
                'is_failure': True,
                'failure_type': 'significant_accuracy_drop',
                'severity': 'warning',
                'recommendations': ['monitor_closely', 'consider_restore', 'adjust_strategy']
            })
            logger.warning(f"⚠️ Significant accuracy drop detected!")
            logger.warning(f"   Accuracy drop: {accuracy_drop:.1%} ({accuracy_drop_ratio:.1%} relative)")
        
        # Check for minimum accuracy threshold
        elif current_accuracy < self.min_accuracy_threshold:
            failure_info.update({
                'is_failure': True,
                'failure_type': 'below_minimum_threshold',
                'severity': 'critical',
                'recommendations': ['immediate_restore', 'emergency_stop']
            })
            logger.error(f"💥 ACCURACY BELOW MINIMUM THRESHOLD!")
            logger.error(f"   Current: {current_accuracy:.1%}, Minimum: {self.min_accuracy_threshold:.1%}")
        
        return failure_info
    
    def apply_recovery_strategy(self, failure_info: Dict[str, Any],
                              target_accuracy: Optional[float] = None) -> RecoveryAction:
        """
        Apply appropriate recovery strategy based on failure information.
        
        Args:
            failure_info: Failure detection results
            target_accuracy: Target accuracy to recover to
            
        Returns:
            RecoveryAction describing what was done
        """
        
        start_time = time.time()
        
        if not failure_info['is_failure']:
            return RecoveryAction(
                action_type='no_action',
                description='No failure detected, no recovery needed',
                success=True,
                recovery_time=0.0
            )
        
        failure_type = failure_info['failure_type']
        severity = failure_info['severity']
        
        logger.info(f"🛡️ Applying recovery strategy for: {failure_type} ({severity})")
        
        # Increment failure count
        self.failure_count += 1
        
        # Check if we've exceeded maximum failures
        if self.failure_count >= self.max_failures:
            return self._emergency_stop(start_time)
        
        # Apply recovery strategy based on severity
        if severity == 'critical':
            return self._critical_recovery(failure_info, target_accuracy, start_time)
        elif severity == 'warning':
            return self._warning_recovery(failure_info, target_accuracy, start_time)
        else:
            return RecoveryAction(
                action_type='unknown',
                description=f'Unknown failure severity: {severity}',
                success=False,
                recovery_time=time.time() - start_time
            )
    
    def _critical_recovery(self, failure_info: Dict[str, Any],
                          target_accuracy: Optional[float], start_time: float) -> RecoveryAction:
        """Handle critical failures with immediate restoration."""
        
        logger.error("🚨 CRITICAL FAILURE - Attempting immediate recovery")
        
        # Try to restore from the best available checkpoint
        best_checkpoint = self.get_best_checkpoint()
        
        if best_checkpoint is None:
            logger.error("❌ No checkpoints available for recovery!")
            return RecoveryAction(
                action_type='emergency_stop',
                description='Critical failure with no recovery checkpoints available',
                success=False,
                recovery_time=time.time() - start_time
            )
        
        # Restore from checkpoint
        success = self.restore_from_checkpoint(best_checkpoint.checkpoint_id)
        
        if success:
            logger.info(f"✅ Successfully restored from checkpoint: {best_checkpoint.checkpoint_id}")
            logger.info(f"   Restored accuracy: {best_checkpoint.accuracy:.1%}")
            logger.info(f"   Restored params: {best_checkpoint.params_count:,}")
            
            return RecoveryAction(
                action_type='restore',
                description=f'Restored from checkpoint {best_checkpoint.checkpoint_id}',
                success=True,
                recovery_time=time.time() - start_time,
                restored_accuracy=best_checkpoint.accuracy,
                restored_params=best_checkpoint.params_count
            )
        else:
            logger.error("❌ Failed to restore from checkpoint!")
            return self._emergency_stop(start_time)
    
    def _warning_recovery(self, failure_info: Dict[str, Any],
                         target_accuracy: Optional[float], start_time: float) -> RecoveryAction:
        """Handle warning-level failures with conservative recovery."""
        
        logger.warning("⚠️ WARNING LEVEL FAILURE - Applying conservative recovery")
        
        # For warning-level failures, we might not need to restore immediately
        # Instead, we can suggest strategy adjustments
        
        recommendations = failure_info.get('recommendations', [])
        
        if 'consider_restore' in recommendations and target_accuracy is not None:
            # Check if we have a checkpoint that meets the target accuracy
            suitable_checkpoint = None
            for checkpoint in reversed(self.checkpoints):  # Check recent checkpoints first
                if checkpoint.accuracy >= target_accuracy:
                    suitable_checkpoint = checkpoint
                    break
            
            if suitable_checkpoint:
                success = self.restore_from_checkpoint(suitable_checkpoint.checkpoint_id)
                if success:
                    return RecoveryAction(
                        action_type='restore',
                        description=f'Restored from checkpoint {suitable_checkpoint.checkpoint_id} to meet target accuracy',
                        success=True,
                        recovery_time=time.time() - start_time,
                        restored_accuracy=suitable_checkpoint.accuracy,
                        restored_params=suitable_checkpoint.params_count
                    )
        
        # If restoration is not needed or failed, suggest strategy adjustment
        return RecoveryAction(
            action_type='strategy_adjustment',
            description='Suggested strategy adjustment for warning-level failure',
            success=True,
            recovery_time=time.time() - start_time
        )
    
    def _emergency_stop(self, start_time: float) -> RecoveryAction:
        """Handle emergency stop condition."""
        
        logger.error("🛑 EMERGENCY STOP - Maximum failures exceeded!")
        logger.error(f"   Failure count: {self.failure_count}/{self.max_failures}")
        
        # Try one last restoration attempt
        best_checkpoint = self.get_best_checkpoint()
        if best_checkpoint:
            success = self.restore_from_checkpoint(best_checkpoint.checkpoint_id)
            if success:
                logger.warning("⚠️ Emergency restoration successful, but stopping further operations")
                return RecoveryAction(
                    action_type='emergency_stop',
                    description=f'Emergency stop with restoration to {best_checkpoint.checkpoint_id}',
                    success=True,
                    recovery_time=time.time() - start_time,
                    restored_accuracy=best_checkpoint.accuracy,
                    restored_params=best_checkpoint.params_count
                )
        
        return RecoveryAction(
            action_type='emergency_stop',
            description='Emergency stop due to maximum failures exceeded',
            success=False,
            recovery_time=time.time() - start_time
        )
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore model state from a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore from
            
        Returns:
            True if restoration was successful, False otherwise
        """
        
        # Find the checkpoint
        checkpoint = None
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                checkpoint = cp
                break
        
        if checkpoint is None:
            logger.error(f"❌ Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            # Restore model parameters
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in checkpoint.model_state:
                        # Move checkpoint data to the same device as current parameter
                        checkpoint_param = checkpoint.model_state[name].to(param.device)
                        param.copy_(checkpoint_param)
                    else:
                        logger.warning(f"⚠️ Parameter {name} not found in checkpoint")
            
            logger.info(f"✅ Successfully restored model from checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore from checkpoint {checkpoint_id}: {str(e)}")
            return False
    
    def get_best_checkpoint(self) -> Optional[RecoveryCheckpoint]:
        """Get the checkpoint with the highest accuracy."""
        
        if not self.checkpoints:
            return None
        
        return max(self.checkpoints, key=lambda cp: cp.accuracy)
    
    def get_latest_checkpoint(self) -> Optional[RecoveryCheckpoint]:
        """Get the most recent checkpoint."""
        
        if not self.checkpoints:
            return None
        
        return max(self.checkpoints, key=lambda cp: cp.timestamp)
    
    def cleanup_checkpoints(self, keep_best: int = 2, keep_recent: int = 2):
        """
        Clean up old checkpoints while keeping the best and most recent ones.
        
        Args:
            keep_best: Number of best-performing checkpoints to keep
            keep_recent: Number of most recent checkpoints to keep
        """
        
        if len(self.checkpoints) <= max(keep_best, keep_recent):
            return  # No cleanup needed
        
        # Get best checkpoints
        best_checkpoints = sorted(self.checkpoints, key=lambda cp: cp.accuracy, reverse=True)[:keep_best]
        
        # Get recent checkpoints
        recent_checkpoints = sorted(self.checkpoints, key=lambda cp: cp.timestamp, reverse=True)[:keep_recent]
        
        # Combine and deduplicate
        checkpoints_to_keep = list(set(best_checkpoints + recent_checkpoints))
        
        # Remove others
        removed_count = len(self.checkpoints) - len(checkpoints_to_keep)
        self.checkpoints = checkpoints_to_keep
        
        logger.info(f"🗑️ Cleaned up {removed_count} old checkpoints")
        logger.info(f"   Kept {len(checkpoints_to_keep)} checkpoints ({keep_best} best + {keep_recent} recent)")
    
    def reset_failure_count(self):
        """Reset the failure count (useful after successful operations)."""
        
        old_count = self.failure_count
        self.failure_count = 0
        
        if old_count > 0:
            logger.info(f"🔄 Reset failure count from {old_count} to 0")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about recovery operations."""
        
        stats = {
            'total_checkpoints': len(self.checkpoints),
            'total_recoveries': len(self.recovery_history),
            'current_failure_count': self.failure_count,
            'max_failures_threshold': self.max_failures,
            'recovery_success_rate': 0.0,
            'checkpoint_statistics': {},
            'recovery_types': {}
        }
        
        if self.checkpoints:
            accuracies = [cp.accuracy for cp in self.checkpoints]
            params = [cp.params_count for cp in self.checkpoints]
            
            stats['checkpoint_statistics'] = {
                'best_accuracy': max(accuracies),
                'worst_accuracy': min(accuracies),
                'average_accuracy': sum(accuracies) / len(accuracies),
                'best_params': max(params),
                'worst_params': min(params),
                'average_params': sum(params) / len(params)
            }
        
        if self.recovery_history:
            successful_recoveries = sum(1 for action in self.recovery_history if action.success)
            stats['recovery_success_rate'] = successful_recoveries / len(self.recovery_history)
            
            # Count recovery types
            for action in self.recovery_history:
                action_type = action.action_type
                stats['recovery_types'][action_type] = stats['recovery_types'].get(action_type, 0) + 1
        
        return stats
    
    def print_recovery_summary(self):
        """Print a comprehensive summary of the recovery engine state."""
        
        stats = self.get_recovery_statistics()
        
        print(f"\n🛡️ Recovery Engine Summary for {self.model_name}")
        print("=" * 60)
        
        print(f"📊 Current State:")
        print(f"   Failure count: {stats['current_failure_count']}/{stats['max_failures_threshold']}")
        print(f"   Total checkpoints: {stats['total_checkpoints']}")
        print(f"   Total recoveries: {stats['total_recoveries']}")
        print(f"   Recovery success rate: {stats['recovery_success_rate']:.1%}")
        
        if stats['checkpoint_statistics']:
            cs = stats['checkpoint_statistics']
            print(f"\n💾 Checkpoint Statistics:")
            print(f"   Best accuracy: {cs['best_accuracy']:.1%}")
            print(f"   Average accuracy: {cs['average_accuracy']:.1%}")
            print(f"   Worst accuracy: {cs['worst_accuracy']:.1%}")
            print(f"   Parameter range: {cs['worst_params']:,} - {cs['best_params']:,}")
        
        if stats['recovery_types']:
            print(f"\n🔄 Recovery Types:")
            for recovery_type, count in stats['recovery_types'].items():
                print(f"   {recovery_type}: {count}")
        
        if self.checkpoints:
            print(f"\n📋 Available Checkpoints:")
            for i, checkpoint in enumerate(self.checkpoints, 1):
                print(f"   {i}. {checkpoint.checkpoint_id}")
                print(f"      Accuracy: {checkpoint.accuracy:.1%}, "
                      f"Params: {checkpoint.params_count:,}, "
                      f"Ratio: {checkpoint.pruning_ratio:.1%}")
        
        print("=" * 60)

