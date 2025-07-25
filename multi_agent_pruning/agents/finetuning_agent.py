#!/usr/bin/env python3
"""
Fine-tuning Agent for Multi-Agent LLM Pruning Framework

This agent handles the fine-tuning of pruned models to recover accuracy
and optimize performance after structured pruning.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .base_agent import BaseAgent, AgentResponse
from ..core.state_manager import PruningState
from ..utils.profiler import TimingProfiler
from ..utils.metrics import AccuracyTracker

logger = logging.getLogger(__name__)

class FinetuningAgent(BaseAgent):
    """
    Fine-tuning Agent that recovers accuracy of pruned models through
    strategic fine-tuning with adaptive learning rates and early stopping.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None, profiler=None):
        """
        Initialize FinetuningAgent with proper BaseAgent inheritance.
        """
        # Call BaseAgent constructor with proper parameters
        super().__init__("FinetuningAgent", llm_client, profiler)
        
        # Store configuration
        self.config = config or {}
        
        # Initialize agent-specific components
        self._initialize_agent_components()
        
        logger.info("🎯 Fine-tuning Agent initialized with proper inheritance")
    
    def _initialize_agent_components(self):
        """Initialize agent-specific components based on configuration."""
        
        # Fine-tuning components - will be initialized when needed
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.accuracy_tracker: Optional[AccuracyTracker] = None
        
        # Training configuration
        training_config = self.config.get('training', {})
        self.epochs = training_config.get('epochs', 50)
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.batch_size = training_config.get('batch_size', 32)
        
        # Optimization configuration
        optimizer_config = self.config.get('optimizer', {})
        self.optimizer_type = optimizer_config.get('type', 'adamw')
        self.momentum = optimizer_config.get('momentum', 0.9)
        self.beta1 = optimizer_config.get('beta1', 0.9)
        self.beta2 = optimizer_config.get('beta2', 0.999)
        
        # Scheduler configuration
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler_type = scheduler_config.get('type', 'cosine')
        self.warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        self.min_lr = scheduler_config.get('min_lr', 1e-6)
        
        # Early stopping configuration
        early_stopping_config = self.config.get('early_stopping', {})
        self.enable_early_stopping = early_stopping_config.get('enabled', True)
        self.patience = early_stopping_config.get('patience', 10)
        self.min_delta = early_stopping_config.get('min_delta', 0.001)
        
        # Validation configuration
        validation_config = self.config.get('validation', {})
        self.validation_frequency = validation_config.get('frequency', 1)  # Every epoch
        self.save_best_model = validation_config.get('save_best', True)
        
        # Results storage
        self.finetuning_results = {}
        self.training_history = []
        
        logger.info("🎯 Fine-tuning Agent components initialized with configuration")
    
    def execute(self, state: PruningState) -> Dict[str, Any]:
        """
        Execute fine-tuning phase: recover accuracy of pruned model.
        
        Args:
            state: Current pruning state with pruned model
            
        Returns:
            Dictionary with fine-tuning results and fine-tuned model
        """
        
        with self.profiler.timer("finetuning_agent_execution"):
            logger.info("🎯 Starting Fine-tuning Agent execution")
            
            try:
                # Validate input state
                if not self._validate_input_state(state):
                    return self._create_error_result("Invalid input state for fine-tuning")
                
                # Initialize fine-tuning components
                self._initialize_finetuning_components(state)
                
                # Execute fine-tuning pipeline
                finetuning_results = self._execute_finetuning_pipeline(state)
                
                # Validate fine-tuned model
                validation_results = self._validate_finetuned_model(state, finetuning_results)
                
                # Get LLM analysis of fine-tuning results
                llm_analysis = self._get_llm_analysis(state, finetuning_results, validation_results)
                
                # Combine results
                final_results = {
                    'success': True,
                    'agent_name': self.agent_name,
                    'timestamp': datetime.now().isoformat(),
                    'finetuning_results': finetuning_results,
                    'validation_results': validation_results,
                    'llm_analysis': llm_analysis,
                    'next_agent': 'EvaluationAgent'
                }
                
                # Update state with fine-tuned model
                state.model = finetuning_results['finetuned_model']
                state.finetuning_results = finetuning_results
                
                # Store results
                self.finetuning_results = finetuning_results
                
                logger.info("✅ Fine-tuning Agent execution completed successfully")
                return final_results
                
            except Exception as e:
                logger.error(f"❌ Fine-tuning Agent execution failed: {str(e)}")
                return self._create_error_result(f"Fine-tuning execution failed: {str(e)}")
    
    def _validate_input_state(self, state: PruningState) -> bool:
        """Validate that the input state contains required pruning results."""
        
        required_fields = ['model', 'pruning_results']
        
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                logger.error(f"❌ Missing required field in state: {field}")
                return False
        
        # Check if model is pruned
        if not hasattr(state, 'pruning_results') or not state.pruning_results:
            logger.error("❌ No pruning results found - model may not be pruned")
            return False
        
        # Check for training data
        if not hasattr(state, 'train_dataloader') or state.train_dataloader is None:
            logger.warning("⚠️ No training dataloader found - fine-tuning may be limited")
        
        logger.info("✅ Input state validation passed")
        return True
    
    def _initialize_finetuning_components(self, state: PruningState):
        """Initialize optimizer, scheduler, and accuracy tracker."""
        
        with self.profiler.timer("finetuning_components_initialization"):
            model = state.model
            
            # Initialize accuracy tracker
            self.accuracy_tracker = AccuracyTracker(track_top5=True)
            
            # Get fine-tuning configuration
            finetuning_config = self._get_finetuning_config(state)
            
            # Initialize optimizer
            self.optimizer = self._create_optimizer(model, finetuning_config)
            
            # Initialize learning rate scheduler
            self.scheduler = self._create_scheduler(self.optimizer, finetuning_config)
            
            logger.info("🔧 Fine-tuning components initialized successfully")
    
    def _get_finetuning_config(self, state: PruningState) -> Dict[str, Any]:
        """Get fine-tuning configuration based on model and pruning results."""
        
        # Get pruning information
        pruning_results = state.pruning_results
        achieved_ratio = pruning_results.get('achieved_pruning_ratio', 0.0)
        
        # Base configuration
        base_config = {
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'epochs': 10,
            'warmup_epochs': 2,
            'patience': 3,
            'min_lr': 1e-6
        }
        
        # Adjust based on pruning ratio
        if achieved_ratio > 0.5:  # Heavy pruning
            base_config.update({
                'learning_rate': 5e-5,  # Lower LR for stability
                'epochs': 20,           # More epochs needed
                'warmup_epochs': 5,     # Longer warmup
                'patience': 5           # More patience
            })
        elif achieved_ratio > 0.3:  # Moderate pruning
            base_config.update({
                'learning_rate': 1e-4,
                'epochs': 15,
                'warmup_epochs': 3,
                'patience': 4
            })
        
        # Adjust based on model type
        model_name = getattr(state, 'model_name', '').lower()
        if 'vit' in model_name or 'deit' in model_name:
            # Vision transformers need different settings
            base_config.update({
                'learning_rate': base_config['learning_rate'] * 0.5,  # Lower LR for ViTs
                'weight_decay': 0.05,  # Higher weight decay
                'warmup_epochs': base_config['warmup_epochs'] + 1
            })
        
        logger.info(f"📋 Fine-tuning config: LR={base_config['learning_rate']}, Epochs={base_config['epochs']}")
        return base_config
    
    def _create_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer for fine-tuning."""
        
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': config['weight_decay']
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(f"🔧 Created AdamW optimizer with LR={config['learning_rate']}")
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer, 
                         config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        
        # Use cosine annealing with warmup
        total_steps = config['epochs']
        warmup_steps = config['warmup_epochs']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        logger.info(f"🔧 Created cosine scheduler with {warmup_steps} warmup epochs")
        return scheduler
    
    def _execute_finetuning_pipeline(self, state: PruningState) -> Dict[str, Any]:
        """Execute the complete fine-tuning pipeline."""
        
        with self.profiler.timer("finetuning_pipeline_execution"):
            logger.info("🔄 Executing fine-tuning pipeline")
            
            # Get configuration
            finetuning_config = self._get_finetuning_config(state)
            
            # Phase 1: Baseline evaluation
            baseline_results = self._evaluate_baseline_performance(state)
            
            # Phase 2: Fine-tuning execution
            training_results = self._execute_training_loop(state, finetuning_config)
            
            # Phase 3: Final evaluation
            final_evaluation = self._evaluate_final_performance(state)
            
            # Phase 4: Performance analysis
            performance_analysis = self._analyze_performance_improvement(
                baseline_results, final_evaluation, training_results
            )
            
            # Combine all results
            pipeline_results = {
                'baseline_performance': baseline_results,
                'training_results': training_results,
                'final_performance': final_evaluation,
                'performance_analysis': performance_analysis,
                'finetuned_model': state.model,
                'training_history': self.training_history,
                'config_used': finetuning_config,
                'pipeline_success': training_results['training_completed']
            }
            
            logger.info("✅ Fine-tuning pipeline execution completed")
            return pipeline_results
    
    def _extract_evaluation_metrics(self, result) -> Dict[str, float]:
        """Extract evaluation metrics from AccuracyTracker result with robust handling."""
        
        # Default values
        metrics = {
            'top1_accuracy': 0.0,
            'top5_accuracy': 0.0,
            'loss': float('inf'),
            'total_samples': 0,
            'inference_time': 0.0
        }
        
        try:
            # If result is a dictionary
            if isinstance(result, dict):
                for key in metrics.keys():
                    if key in result:
                        metrics[key] = float(result[key])
                return metrics
            
            # If result is an object with attributes
            if hasattr(result, 'top1_accuracy'):
                metrics['top1_accuracy'] = float(result.top1_accuracy)
            elif hasattr(result, 'accuracy'):
                metrics['top1_accuracy'] = float(result.accuracy)
            elif hasattr(result, 'acc'):
                metrics['top1_accuracy'] = float(result.acc)
            
            if hasattr(result, 'top5_accuracy'):
                metrics['top5_accuracy'] = float(result.top5_accuracy)
            elif hasattr(result, 'top5'):
                metrics['top5_accuracy'] = float(result.top5)
            
            if hasattr(result, 'loss'):
                metrics['loss'] = float(result.loss)
            elif hasattr(result, 'avg_loss'):
                metrics['loss'] = float(result.avg_loss)
            
            if hasattr(result, 'total_samples'):
                metrics['total_samples'] = int(result.total_samples)
            elif hasattr(result, 'num_samples'):
                metrics['total_samples'] = int(result.num_samples)
            
            if hasattr(result, 'inference_time'):
                metrics['inference_time'] = float(result.inference_time)
            elif hasattr(result, 'time'):
                metrics['inference_time'] = float(result.time)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting metrics: {e}, using defaults")
            return metrics

    def _evaluate_baseline_performance(self, state: PruningState) -> Dict[str, Any]:
        """Evaluate baseline performance of the pruned model."""
        
        with self.profiler.timer("baseline_evaluation"):
            logger.info("📊 Evaluating baseline performance")
            
            model = state.model
            
            eval_dataloader = self._get_evaluation_dataloader(state)
            
            if eval_dataloader is None:
                logger.warning("⚠️ No evaluation dataloader available")
                return {
                    'top1_accuracy': 0.0,
                    'top5_accuracy': 0.0,
                    'loss': float('inf'),
                    'total_samples': 0,
                    'inference_time': 0.0,
                    'message': 'No evaluation data available'
                }
            
            try:
                self.accuracy_tracker.reset()
                baseline_result = self.accuracy_tracker.evaluate_model(
                    model=model,
                    dataloader=eval_dataloader,
                    criterion=nn.CrossEntropyLoss(),
                    max_batches=50  # Limit for efficiency
                )
                
                baseline_performance = self._extract_evaluation_metrics(baseline_result)
                
                logger.info(f"📊 Baseline performance: {baseline_performance['top1_accuracy']:.1%} accuracy")
                return baseline_performance
                
            except Exception as e:
                logger.error(f"❌ Baseline evaluation failed: {str(e)}")
                return {
                    'top1_accuracy': 0.0,
                    'top5_accuracy': 0.0,
                    'loss': float('inf'),
                    'total_samples': 0,
                    'inference_time': 0.0,
                    'error': str(e)
                }

    def _execute_training_loop(self, state: PruningState, 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main training loop."""
        
        with self.profiler.timer("training_loop_execution"):
            logger.info("🏃 Starting training loop")
            
            model = state.model
            
            train_dataloader = self._get_training_dataloader(state)
            val_dataloader = self._get_evaluation_dataloader(state)
            
            if train_dataloader is None:
                logger.error("❌ No training dataloader available")
                return {
                    'training_completed': False,
                    'epochs_completed': 0,
                    'best_accuracy': 0.0,
                    'final_accuracy': 0.0,
                    'training_history': [],
                    'error': 'No training data available'
                }
            
            # Training setup
            model.train()
            criterion = nn.CrossEntropyLoss()
            device = next(model.parameters()).device
            
            # Training tracking
            best_accuracy = 0.0
            best_model_state = None
            patience_counter = 0
            epochs_completed = 0
            
            self.training_history = []
            
            # Training loop
            for epoch in range(config['epochs']):
                epoch_start_time = datetime.now()
                
                # Training epoch
                train_metrics = self._train_epoch(
                    model, train_dataloader, criterion, self.optimizer, device
                )
                
                # Validation epoch (if validation data available)
                val_metrics = {}
                if val_dataloader is not None:
                    val_metrics = self._validate_epoch(
                        model, val_dataloader, criterion, device
                    )
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Track progress
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics.get('loss', 0.0),
                    'train_accuracy': train_metrics.get('accuracy', 0.0),
                    'val_loss': val_metrics.get('loss', 0.0),
                    'val_accuracy': val_metrics.get('accuracy', 0.0),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': (datetime.now() - epoch_start_time).total_seconds()
                }
                
                self.training_history.append(epoch_metrics)
                epochs_completed = epoch + 1
                
                # Check for improvement
                current_accuracy = val_metrics.get('accuracy', train_metrics.get('accuracy', 0.0))
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if self.enable_early_stopping and patience_counter >= self.patience:
                    logger.info(f"🛑 Early stopping triggered after {epochs_completed} epochs")
                    break
                
                # Progress logging
                if epoch % max(1, config['epochs'] // 10) == 0:
                    logger.info(f"📊 Epoch {epoch + 1}/{config['epochs']}: "
                              f"Train Loss: {train_metrics.get('loss', 0.0):.4f}, "
                              f"Val Acc: {current_accuracy:.1%}")
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(f"✅ Restored best model with {best_accuracy:.1%} accuracy")
            
            return {
                'training_completed': True,
                'epochs_completed': epochs_completed,
                'best_accuracy': best_accuracy,
                'final_accuracy': best_accuracy,
                'training_history': self.training_history,
                'total_training_time': sum(h['epoch_time'] for h in self.training_history)
            }

    def _train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    criterion: nn.Module, optimizer: optim.Optimizer, 
                    device: torch.device) -> Dict[str, float]:
        """Train for one epoch."""
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Limit batches for efficiency during fine-tuning
            if batch_idx >= 100:  # Process max 100 batches per epoch
                break
        
        return {
            'loss': total_loss / (batch_idx + 1),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def _validate_epoch(self, model: nn.Module, dataloader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> Dict[str, float]:
        """Validate for one epoch."""
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Limit batches for efficiency
                if batch_idx >= 50:  # Process max 50 batches for validation
                    break
        
        return {
            'loss': total_loss / (batch_idx + 1),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def _evaluate_final_performance(self, state: PruningState, 
                                  training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate final performance after fine-tuning."""
        
        with self.profiler.timer("final_evaluation"):
            logger.info("📊 Evaluating final performance")
            
            model = state.model
            
            eval_dataloader = self._get_evaluation_dataloader(state)
            
            if eval_dataloader is None:
                logger.warning("⚠️ No evaluation dataloader available")
                return {
                    'top1_accuracy': training_results.get('best_accuracy', 0.0),
                    'top5_accuracy': 0.0,
                    'loss': float('inf'),
                    'total_samples': 0,
                    'inference_time': 0.0,
                    'message': 'No evaluation data available - using training accuracy'
                }
            
            try:
                self.accuracy_tracker.reset()
                final_result = self.accuracy_tracker.evaluate_model(
                    model=model,
                    dataloader=eval_dataloader,
                    criterion=nn.CrossEntropyLoss(),
                    max_batches=100  # More thorough evaluation
                )
                
                final_performance = self._extract_evaluation_metrics(final_result)
                
                logger.info(f"📊 Final performance: {final_performance['top1_accuracy']:.1%} accuracy")
                return final_performance
                
            except Exception as e:
                logger.error(f"❌ Final evaluation failed: {str(e)}")
                return {
                    'top1_accuracy': training_results.get('best_accuracy', 0.0),
                    'top5_accuracy': 0.0,
                    'loss': float('inf'),
                    'total_samples': 0,
                    'inference_time': 0.0,
                    'error': str(e)
                }

    def _analyze_performance_improvement(self, baseline: Dict[str, Any],
                                       final: Dict[str, Any],
                                       training: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance improvement from fine-tuning."""
        
        # Calculate improvements
        accuracy_improvement = final['top1_accuracy'] - baseline['top1_accuracy']
        loss_improvement = baseline['loss'] - final['loss']
        
        # Analyze training efficiency
        epochs_used = training['epochs_completed']
        training_time = training['total_training_time']
        
        # Calculate recovery metrics
        pruning_results = getattr(self, 'pruning_results', {})
        achieved_ratio = pruning_results.get('achieved_pruning_ratio', 0.0)
        
        analysis = {
            'accuracy_improvement': accuracy_improvement,
            'accuracy_improvement_percent': accuracy_improvement * 100,
            'loss_improvement': loss_improvement,
            'relative_accuracy_recovery': accuracy_improvement / achieved_ratio if achieved_ratio > 0 else 0.0,
            'training_efficiency': {
                'epochs_used': epochs_used,
                'training_time_minutes': training_time / 60,
                'accuracy_per_epoch': accuracy_improvement / epochs_used if epochs_used > 0 else 0.0,
                'early_stopping_used': training['early_stopping_triggered']
            },
            'overall_assessment': self._assess_finetuning_success(
                accuracy_improvement, achieved_ratio, epochs_used
            )
        }
        
        logger.info(f"📈 Performance analysis: {accuracy_improvement:+.1%} accuracy improvement")
        return analysis
    
    def _assess_finetuning_success(self, accuracy_improvement: float,
                                 pruning_ratio: float, epochs_used: int) -> str:
        """Assess overall success of fine-tuning."""
        
        # Define success criteria
        if accuracy_improvement > 0.05:  # >5% improvement
            return 'excellent'
        elif accuracy_improvement > 0.02:  # >2% improvement
            return 'good'
        elif accuracy_improvement > 0.0:   # Any improvement
            return 'moderate'
        elif accuracy_improvement > -0.02: # <2% degradation
            return 'acceptable'
        else:
            return 'poor'
    
    def _validate_finetuned_model(self, state: PruningState,
                                finetuning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the fine-tuned model."""
        
        with self.profiler.timer("finetuned_model_validation"):
            logger.info("🔍 Validating fine-tuned model")
            
            model = state.model
            
            validation_tests = {
                'forward_pass_test': self._test_forward_pass(model),
                'gradient_flow_test': self._test_gradient_flow(model),
                'performance_regression_test': self._test_performance_regression(finetuning_results),
                'stability_test': self._test_model_stability(model),
                'memory_efficiency_test': self._test_memory_efficiency(model)
            }
            
            # Overall validation status
            all_tests_passed = all(test['passed'] for test in validation_tests.values())
            
            validation_summary = {
                'model_is_valid': all_tests_passed,
                'validation_tests': validation_tests,
                'failed_tests': [
                    test_name for test_name, result in validation_tests.items()
                    if not result['passed']
                ],
                'validation_score': sum(1 for test in validation_tests.values() if test['passed']) / len(validation_tests)
            }
            
            if all_tests_passed:
                logger.info("✅ Fine-tuned model validation passed all tests")
            else:
                logger.warning(f"⚠️ Fine-tuned model validation failed {len(validation_summary['failed_tests'])} tests")
            
            return validation_summary
    
    def _get_llm_analysis(self, state: PruningState, finetuning_results: Dict[str, Any],
                         validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM analysis of fine-tuning results."""
        
        if not self.llm_client:
            return {'status': 'llm_not_available', 'message': 'LLM client not configured'}
        
        # Create prompt for LLM analysis
        prompt = self._create_llm_analysis_prompt(state, finetuning_results, validation_results)
        
        try:
            with self.profiler.timer("llm_analysis"):
                response = self.llm_client.generate(prompt)
                
                # Parse LLM response
                llm_analysis = self._parse_llm_analysis_response(response)
                
                logger.info("🤖 LLM analysis completed successfully")
                return llm_analysis
                
        except Exception as e:
            logger.warning(f"⚠️ LLM analysis failed: {str(e)}")
            return {
                'status': 'llm_analysis_failed',
                'error': str(e),
                'fallback_used': True
            }
    
    # Helper methods for validation tests
    def _test_forward_pass(self, model: nn.Module) -> Dict[str, Any]:
        """Test if the model can perform forward pass."""
        
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            return {
                'passed': True,
                'output_shape': list(output.shape),
                'message': 'Forward pass successful'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Forward pass failed'
            }
    
    def _test_gradient_flow(self, model: nn.Module) -> Dict[str, Any]:
        """Test if gradients can flow through the model."""
        
        try:
            model.train()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_target = torch.randint(0, 1000, (1,)).to(device)
            
            # Forward pass
            output = model(dummy_input)
            loss = nn.CrossEntropyLoss()(output, dummy_target)
            
            # Backward pass
            loss.backward()
            
            # Check if gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            
            return {
                'passed': has_gradients,
                'message': 'Gradient flow test successful' if has_gradients else 'No gradients found'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Gradient flow test failed'
            }
    
    def _test_performance_regression(self, finetuning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test for performance regression."""
        
        try:
            performance_analysis = finetuning_results['performance_analysis']
            accuracy_improvement = performance_analysis['accuracy_improvement']
            
            # Check if there's significant regression
            significant_regression = accuracy_improvement < -0.05  # >5% drop
            
            return {
                'passed': not significant_regression,
                'accuracy_change': accuracy_improvement,
                'message': 'No significant regression' if not significant_regression else f'Significant regression: {accuracy_improvement:.1%}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Performance regression test failed'
            }
    
    def _test_model_stability(self, model: nn.Module) -> Dict[str, Any]:
        """Test model stability with multiple forward passes."""
        
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            outputs = []
            with torch.no_grad():
                for _ in range(5):
                    output = model(dummy_input)
                    outputs.append(output.cpu())
            
            # Check consistency
            max_diff = max(torch.max(torch.abs(outputs[i] - outputs[0])).item() 
                          for i in range(1, len(outputs)))
            
            is_stable = max_diff < 1e-6  # Very small differences expected
            
            return {
                'passed': is_stable,
                'max_difference': max_diff,
                'message': 'Model is stable' if is_stable else f'Model instability detected: {max_diff}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Stability test failed'
            }
    
    def _test_memory_efficiency(self, model: nn.Module) -> Dict[str, Any]:
        """Test memory efficiency of the fine-tuned model."""
        
        try:
            # Get model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            # Test memory usage during inference
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model_copy = type(model)()
                model_copy.load_state_dict(model.state_dict())
                model_copy.to(device)
                
                # Measure memory usage
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                dummy_input = torch.randn(8, 3, 224, 224).to(device)  # Batch of 8
                with torch.no_grad():
                    _ = model_copy(dummy_input)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)
                
                return {
                    'passed': True,
                    'model_size_mb': model_size_mb,
                    'inference_memory_mb': memory_used_mb,
                    'message': 'Memory efficiency test passed'
                }
            else:
                return {
                    'passed': True,
                    'model_size_mb': model_size_mb,
                    'message': 'Memory efficiency test passed (CPU only)'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Memory efficiency test failed'
            }
    
    def _create_llm_analysis_prompt(self, state: PruningState, finetuning_results: Dict[str, Any],
                                  validation_results: Dict[str, Any]) -> str:
        """Create prompt for LLM analysis."""
        
        model_name = getattr(state, 'model_name', 'unknown')
        performance_analysis = finetuning_results['performance_analysis']
        
        prompt = f"""
You are an expert in neural network fine-tuning. Please analyze the following fine-tuning results:

## Model Information:
- Model: {model_name}
- Pruning Ratio: {getattr(state, 'target_pruning_ratio', 0.0):.1%}

## Fine-tuning Results:
{json.dumps(finetuning_results, indent=2, default=str)}

## Validation Results:
{json.dumps(validation_results, indent=2, default=str)}

Please provide analysis in JSON format:
{{
  "overall_assessment": "excellent|good|moderate|acceptable|poor",
  "key_insights": [list of key insights],
  "concerns": [list of concerns],
  "recommendations": [list of recommendations],
  "confidence_score": float between 0 and 1
}}
"""
        
        return prompt
    
    def _parse_llm_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                llm_analysis = json.loads(json_str)
                
                return {
                    'status': 'success',
                    'analysis': llm_analysis,
                    'raw_response': response
                }
            else:
                return {
                    'status': 'parsing_failed',
                    'message': 'Could not extract JSON from LLM response',
                    'raw_response': response
                }
                
        except json.JSONDecodeError as e:
            return {
                'status': 'json_parsing_failed',
                'error': str(e),
                'raw_response': response
            }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        
        return {
            'success': False,
            'agent_name': self.agent_name,
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'next_agent': None
        }

    def get_agent_role(self) -> str:
        """Return the role of this agent."""
        return "finetuning_agent"
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for the fine-tuning agent."""
        
        model_info = context.get('model_info', {})
        pruning_results = context.get('pruning_results', {})
        training_config = context.get('training_config', {})
        dataset_info = context.get('dataset_info', {})
        
        prompt = f"""You are an expert Fine-tuning Agent in a multi-agent neural network pruning system.

ROLE: Recover model performance after pruning through intelligent fine-tuning with adaptive strategies.

CURRENT CONTEXT:
- Model: {model_info.get('name', 'Unknown')} ({model_info.get('total_params', 0):,} parameters)
- Architecture: {model_info.get('architecture_type', 'Unknown')}
- Pruning applied: {pruning_results.get('compression_ratio', 0.0)*100:.1f}% compression
- Dataset: {dataset_info.get('name', 'Unknown')} ({dataset_info.get('num_classes', 'Unknown')} classes)

PRUNING IMPACT:
- Parameters removed: {pruning_results.get('params_removed', 0):,}
- Accuracy drop: {pruning_results.get('accuracy_drop', 0.0)*100:.1f}%
- Layers affected: {len(pruning_results.get('affected_layers', []))}
- Recovery target: {pruning_results.get('target_accuracy', 0.0)*100:.1f}%

FINE-TUNING RESPONSIBILITIES:
1. Design adaptive learning rate schedule based on pruning severity
2. Select optimal training strategy (full fine-tuning vs layer-wise)
3. Configure early stopping and convergence detection
4. Monitor recovery progress and adjust hyperparameters
5. Ensure stable training without overfitting

ARCHITECTURE-SPECIFIC STRATEGIES:
- Transformers: Lower LR for attention, higher for MLP, warmup essential
- CNNs: Layer-wise LR scaling, batch norm adaptation, data augmentation
- Hybrid: Component-specific strategies with careful coordination

TRAINING CONFIGURATION:
- Base learning rate: {training_config.get('base_lr', 1e-4)}
- Batch size: {training_config.get('batch_size', 32)}
- Max epochs: {training_config.get('max_epochs', 100)}
- Patience: {training_config.get('patience', 10)}

DECISION FRAMEWORK:
- ADAPTIVE: Adjust strategy based on pruning severity and architecture
- EFFICIENT: Minimize training time while maximizing recovery
- STABLE: Prevent overfitting and ensure convergence
- MONITORED: Track progress and intervene if needed

Provide structured fine-tuning plan with clear rationale for each decision."""
        
        return prompt
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response for fine-tuning decisions."""
        
        try:
            # Extract key fine-tuning decisions from response
            decisions = {}
            
            # Look for learning rate recommendations
            import re
            
            lr_match = re.search(r'learning.?rate.*?(\d+\.?\d*e?-?\d*)', response.lower())
            if lr_match:
                decisions['learning_rate'] = float(lr_match.group(1))
            else:
                decisions['learning_rate'] = 1e-4  # default
            
            # Extract training strategy
            if 'layer-wise' in response.lower() or 'layerwise' in response.lower():
                decisions['training_strategy'] = 'layer_wise'
            elif 'full' in response.lower() and 'fine' in response.lower():
                decisions['training_strategy'] = 'full_finetuning'
            elif 'progressive' in response.lower():
                decisions['training_strategy'] = 'progressive'
            else:
                decisions['training_strategy'] = 'full_finetuning'  # default
            
            # Extract epochs recommendation
            epochs_match = re.search(r'epoch.*?(\d+)', response.lower())
            if epochs_match:
                decisions['max_epochs'] = int(epochs_match.group(1))
            else:
                decisions['max_epochs'] = 50  # default
            
            # Extract patience for early stopping
            patience_match = re.search(r'patience.*?(\d+)', response.lower())
            if patience_match:
                decisions['patience'] = int(patience_match.group(1))
            else:
                decisions['patience'] = 10  # default
            
            # Extract warmup strategy
            if 'warmup' in response.lower():
                warmup_match = re.search(r'warmup.*?(\d+)', response.lower())
                if warmup_match:
                    decisions['warmup_epochs'] = int(warmup_match.group(1))
                else:
                    decisions['warmup_epochs'] = 5
            else:
                decisions['warmup_epochs'] = 0
            
            # Extract learning rate schedule
            if 'cosine' in response.lower():
                decisions['lr_schedule'] = 'cosine'
            elif 'step' in response.lower():
                decisions['lr_schedule'] = 'step'
            elif 'exponential' in response.lower():
                decisions['lr_schedule'] = 'exponential'
            else:
                decisions['lr_schedule'] = 'cosine'  # default
            
            # Extract data augmentation strategy
            if 'augmentation' in response.lower() or 'augment' in response.lower():
                decisions['use_augmentation'] = True
                if 'heavy' in response.lower() or 'aggressive' in response.lower():
                    decisions['augmentation_strength'] = 'heavy'
                elif 'light' in response.lower() or 'mild' in response.lower():
                    decisions['augmentation_strength'] = 'light'
                else:
                    decisions['augmentation_strength'] = 'medium'
            else:
                decisions['use_augmentation'] = False
                decisions['augmentation_strength'] = 'none'
            
            # Extract confidence and rationale
            if 'confident' in response.lower() or 'certain' in response.lower():
                confidence = 0.9
            elif 'uncertain' in response.lower() or 'unsure' in response.lower():
                confidence = 0.5
            else:
                confidence = 0.7
            
            # Extract key rationale points
            rationale = []
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['because', 'since', 'due to', 'reason', 'strategy']):
                    rationale.append(line.strip())
            
            decisions['rationale'] = rationale[:3]  # Keep top 3 reasons
            
            # Determine success based on content quality
            success = (
                len(decisions) >= 5 and 
                'error' not in response.lower() and
                decisions['learning_rate'] > 0 and
                decisions['max_epochs'] > 0
            )
            
            return AgentResponse(
                success=success,
                data=decisions,
                message=f"Fine-tuning plan: {decisions['training_strategy']}, LR={decisions['learning_rate']:.2e}, {decisions['max_epochs']} epochs",
                confidence=confidence
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={
                    'learning_rate': 1e-4,
                    'training_strategy': 'full_finetuning',
                    'max_epochs': 50,
                    'patience': 10,
                    'warmup_epochs': 5,
                    'lr_schedule': 'cosine',
                    'use_augmentation': True,
                    'augmentation_strength': 'medium'
                },
                message=f"Failed to parse fine-tuning response, using safe defaults: {str(e)}",
                confidence=0.3
            )

