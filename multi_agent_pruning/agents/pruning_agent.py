#!/usr/bin/env python3
"""
Pruning Agent for Multi-Agent LLM Pruning Framework

This agent executes the actual pruning based on analysis recommendations,
applying structured pruning with safety checks and constraint validation.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
import torch
import torch.nn as nn

from .base_agent import BaseAgent, AgentResponse
from ..core.state_manager import PruningState
from ..core.pruning_engine import PruningEngine
from ..core.importance_criteria import ImportanceCriteria
from ..utils.profiler import TimingProfiler
from ..utils.metrics import compute_model_complexity

logger = logging.getLogger(__name__)

class PruningAgent(BaseAgent):
    """
    Pruning Agent that executes structured pruning based on analysis recommendations.
    Applies safety checks, constraint validation, and creates checkpoints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None, profiler=None):
        """
        Initialize PruningAgent with proper BaseAgent inheritance.
        """
        # Call BaseAgent constructor with proper parameters
        super().__init__("PruningAgent", llm_client, profiler)
        
        # Store configuration
        self.config = config or {}
        
        # Initialize agent-specific components
        self._initialize_agent_components()
        
        logger.info("✂️ Pruning Agent initialized with proper inheritance")
    
    def _initialize_agent_components(self):
        """Initialize agent-specific components based on configuration."""
        
        # Pruning components - will be initialized when needed
        self.pruning_engine: Optional[PruningEngine] = None
        self.importance_criteria: Optional[ImportanceCriteria] = None
        
        # Pruning configuration
        pruning_config = self.config.get('pruning', {})
        self.enable_safety_checks = pruning_config.get('safety_checks', True)
        self.enable_checkpointing = pruning_config.get('checkpointing', True)
        self.validate_constraints = pruning_config.get('validate_constraints', True)
        
        # Safety configuration
        safety_config = self.config.get('safety', {})
        self.max_layer_pruning = safety_config.get('max_layer_pruning', 0.8)
        self.min_accuracy_threshold = safety_config.get('min_accuracy_threshold', 0.3)
        self.safety_margin = safety_config.get('safety_margin', 0.05)
        
        # Execution configuration
        execution_config = self.config.get('execution', {})
        self.dry_run_mode = execution_config.get('dry_run', False)
        self.verbose_logging = execution_config.get('verbose', True)
        self.backup_model = execution_config.get('backup_model', True)
        
        # Results storage
        self.pruning_results = {}
        self.checkpoints = {}
        
        logger.info("✂️ Pruning Agent components initialized with configuration")

    def execute(self, state: PruningState) -> Dict[str, Any]:
        """
        Execute pruning phase: apply structured pruning with safety checks.
        
        Args:
            state: Current pruning state with analysis results
            
        Returns:
            Dictionary with pruning results and pruned model
        """
        
        with self.profiler.timer("pruning_agent_execution"):
            logger.info("✂️ Starting Pruning Agent execution")
            
            try:
                # Validate input state
                if not self._validate_input_state(state):
                    return self._create_error_result("Invalid input state for pruning")
                
                # Create checkpoint before pruning
                self._create_checkpoint(state, "pre_pruning")
                
                # Initialize pruning components
                self._initialize_pruning_components(state)
                
                # Execute pruning pipeline
                pruning_results = self._execute_pruning_pipeline(state)
                
                # Validate pruned model
                validation_results = self._validate_pruned_model(state, pruning_results)
                
                # Get LLM validation if available
                llm_validation = self._get_llm_validation(state, pruning_results, validation_results)
                
                # Combine results
                final_results = {
                    'success': True,
                    'agent_name': self.agent_name,
                    'timestamp': datetime.now().isoformat(),
                    'pruning_results': pruning_results,
                    'validation_results': validation_results,
                    'llm_validation': llm_validation,
                    'next_agent': 'FinetuningAgent'
                }
                
                # Update state with pruned model
                state.model = pruning_results['pruned_model']
                state.pruning_results = pruning_results
                
                # Store results
                self.pruning_results = pruning_results
                
                logger.info("✅ Pruning Agent execution completed successfully")
                return final_results
                
            except Exception as e:
                logger.error(f"❌ Pruning Agent execution failed: {str(e)}")
                
                # Attempt recovery
                recovery_result = self._attempt_recovery(state, str(e))
                
                return self._create_error_result(
                    f"Pruning execution failed: {str(e)}",
                    recovery_info=recovery_result
                )
    
    def _validate_input_state(self, state: PruningState) -> bool:
        """Validate that the input state contains required analysis results."""
        
        required_fields = ['model', 'analysis_results']
        
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                logger.error(f"❌ Missing required field in state: {field}")
                return False
        
        # Check analysis results structure
        analysis_results = state.analysis_results
        if not isinstance(analysis_results, dict):
            logger.error("❌ Analysis results must be a dictionary")
            return False
        
        required_analysis_fields = ['strategic_recommendations']
        for field in required_analysis_fields:
            if field not in analysis_results:
                logger.error(f"❌ Missing required analysis field: {field}")
                return False
        
        logger.info("✅ Input state validation passed")
        return True
    
    def _create_checkpoint(self, state: PruningState, checkpoint_name: str):
        """Create a checkpoint of the current model state."""
        
        with self.profiler.timer("checkpoint_creation"):
            logger.info(f"💾 Creating checkpoint: {checkpoint_name}")
            
            checkpoint = {
                'model_state_dict': state.model.state_dict().copy(),
                'timestamp': datetime.now().isoformat(),
                'model_complexity': compute_model_complexity(state.model),
                'checkpoint_name': checkpoint_name
            }
            
            self.checkpoints[checkpoint_name] = checkpoint
            
            logger.info(f"✅ Checkpoint '{checkpoint_name}' created successfully")
    
    def _initialize_pruning_components(self, state: PruningState):
        """Initialize pruning engine and importance criteria."""
        
        with self.profiler.timer("pruning_components_initialization"):
            model = state.model
            model_name = getattr(state, 'model_name', 'unknown_model')
            
            # Get recommendations from analysis
            analysis_results = state.analysis_results
            recommendations = analysis_results['strategic_recommendations']
            
            # Initialize importance criteria
            importance_config = recommendations['importance_criterion']
            self.importance_criteria = ImportanceCriteria(
                criterion=importance_config['primary_criterion'],
                fallback_criterion=importance_config['fallback_criterion']
            )
            
            # Initialize pruning engine
            self.pruning_engine = PruningEngine(
                model=model,
                model_name=model_name,
                importance_criteria=self.importance_criteria
            )
            
            logger.info("🔧 Pruning components initialized successfully")
    
    def _execute_pruning_pipeline(self, state: PruningState) -> Dict[str, Any]:
        """Execute the complete pruning pipeline."""
        
        with self.profiler.timer("pruning_pipeline_execution"):
            logger.info("🔄 Executing pruning pipeline")
            
            # Get analysis recommendations
            analysis_results = state.analysis_results
            recommendations = analysis_results['strategic_recommendations']
            
            # Phase 1: Compute importance scores
            importance_results = self._compute_importance_scores(state, recommendations)
            
            # Phase 2: Apply structured pruning
            pruning_execution_results = self._apply_structured_pruning(state, recommendations, importance_results)
            
            # Phase 3: Validate constraints
            constraint_validation = self._validate_constraints(state, pruning_execution_results)
            
            # Phase 4: Compute final metrics
            final_metrics = self._compute_final_metrics(state, pruning_execution_results)
            
            # Combine all results
            pipeline_results = {
                'importance_computation': importance_results,
                'pruning_execution': pruning_execution_results,
                'constraint_validation': constraint_validation,
                'final_metrics': final_metrics,
                'pruned_model': pruning_execution_results['pruned_model'],
                'achieved_pruning_ratio': pruning_execution_results['achieved_pruning_ratio'],
                'pipeline_success': constraint_validation['all_constraints_satisfied']
            }
            
            logger.info("✅ Pruning pipeline execution completed")
            return pipeline_results
    
    def _compute_importance_scores(self, state: PruningState, 
                                 recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Compute importance scores for all prunable layers."""
        
        with self.profiler.timer("importance_computation"):
            logger.info("📊 Computing importance scores")
            
            model = state.model
            
            # Get importance criterion configuration
            importance_config = recommendations['importance_criterion']
            criterion = importance_config['primary_criterion']
            
            # Prepare data for gradient-based criteria
            data_loader = None
            if criterion == 'taylor' and hasattr(state, 'dataloader'):
                data_loader = state.dataloader
            
            # Compute importance scores
            try:
                importance_scores = self.importance_criteria.compute_importance(
                    model=model,
                    criterion=criterion,
                    data_loader=data_loader,
                    num_samples=100  # Use subset for efficiency
                )
                
                # Analyze importance distribution
                score_analysis = self._analyze_importance_distribution(importance_scores)
                
                results = {
                    'success': True,
                    'criterion_used': criterion,
                    'importance_scores': importance_scores,
                    'score_analysis': score_analysis,
                    'total_layers_analyzed': len(importance_scores)
                }
                
                logger.info(f"✅ Importance scores computed for {len(importance_scores)} layers")
                return results
                
            except Exception as e:
                logger.warning(f"⚠️ Primary criterion '{criterion}' failed: {str(e)}")
                
                # Fallback to secondary criterion
                fallback_criterion = importance_config['fallback_criterion']
                logger.info(f"🔄 Falling back to criterion: {fallback_criterion}")
                
                importance_scores = self.importance_criteria.compute_importance(
                    model=model,
                    criterion=fallback_criterion,
                    data_loader=None  # Fallback criteria don't need data
                )
                
                score_analysis = self._analyze_importance_distribution(importance_scores)
                
                return {
                    'success': True,
                    'criterion_used': fallback_criterion,
                    'fallback_used': True,
                    'fallback_reason': str(e),
                    'importance_scores': importance_scores,
                    'score_analysis': score_analysis,
                    'total_layers_analyzed': len(importance_scores)
                }
    
    def _apply_structured_pruning(self, state: PruningState, recommendations: Dict[str, Any],
                                importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply structured pruning based on importance scores and recommendations."""
        
        with self.profiler.timer("structured_pruning_application"):
            logger.info("✂️ Applying structured pruning")
            
            model = state.model
            target_ratio = state.target_pruning_ratio
            importance_scores = importance_results['importance_scores']
            
            # Get group ratio recommendations
            group_ratios = recommendations['group_ratios']['recommended_ratios']
            
            # Apply pruning using the engine
            try:
                pruning_result = self.pruning_engine.prune_model(
                    importance_scores=importance_scores,
                    target_pruning_ratio=target_ratio,
                    group_ratios=group_ratios,
                    safety_checks=True
                )
                
                # Calculate achieved metrics
                original_params = sum(p.numel() for p in model.parameters())
                pruned_params = sum(p.numel() for p in pruning_result['pruned_model'].parameters())
                achieved_ratio = 1.0 - (pruned_params / original_params)
                
                results = {
                    'success': True,
                    'pruned_model': pruning_result['pruned_model'],
                    'original_parameters': original_params,
                    'pruned_parameters': pruned_params,
                    'achieved_pruning_ratio': achieved_ratio,
                    'target_pruning_ratio': target_ratio,
                    'ratio_accuracy': abs(achieved_ratio - target_ratio) < 0.05,
                    'layers_pruned': pruning_result['layers_pruned'],
                    'pruning_statistics': pruning_result['statistics'],
                    'safety_checks_passed': pruning_result['safety_checks_passed']
                }
                
                logger.info(f"✅ Structured pruning applied successfully")
                logger.info(f"   Target ratio: {target_ratio:.1%}")
                logger.info(f"   Achieved ratio: {achieved_ratio:.1%}")
                logger.info(f"   Parameters: {original_params:,} → {pruned_params:,}")
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Structured pruning failed: {str(e)}")
                raise
    
    def _validate_constraints(self, state: PruningState, 
                            pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all constraints are satisfied after pruning."""
        
        with self.profiler.timer("constraint_validation"):
            logger.info("🔍 Validating constraints")
            
            pruned_model = pruning_results['pruned_model']
            
            # Get analysis results for constraint information
            analysis_results = state.analysis_results
            dependency_analysis = analysis_results['analysis_results']['dependency_analysis']
            safety_analysis = analysis_results['analysis_results']['safety_analysis']
            
            validation_results = {
                'dimension_constraints': self._validate_dimension_constraints(pruned_model, dependency_analysis),
                'safety_constraints': self._validate_safety_constraints(pruning_results, safety_analysis),
                'architectural_constraints': self._validate_architectural_constraints(pruned_model, state),
                'performance_constraints': self._validate_performance_constraints(pruning_results, state)
            }
            
            # Overall validation status
            all_passed = all(
                result['passed'] for result in validation_results.values()
            )
            
            validation_summary = {
                'all_constraints_satisfied': all_passed,
                'validation_details': validation_results,
                'violations': [
                    f"{constraint}: {result['violations']}"
                    for constraint, result in validation_results.items()
                    if not result['passed']
                ],
                'warnings': [
                    f"{constraint}: {result.get('warnings', [])}"
                    for constraint, result in validation_results.items()
                    if result.get('warnings')
                ]
            }
            
            if all_passed:
                logger.info("✅ All constraints validated successfully")
            else:
                logger.warning(f"⚠️ Constraint violations detected: {len(validation_summary['violations'])}")
            
            return validation_summary
    
    def _compute_final_metrics(self, state: PruningState, 
                             pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final metrics for the pruned model."""
        
        with self.profiler.timer("final_metrics_computation"):
            logger.info("📊 Computing final metrics")
            
            original_model = self.checkpoints['pre_pruning']['model_state_dict']
            pruned_model = pruning_results['pruned_model']
            
            # Compute complexity metrics
            pruned_complexity = compute_model_complexity(pruned_model)
            original_complexity = self.checkpoints['pre_pruning']['model_complexity']
            
            # Calculate improvements
            param_reduction = 1.0 - (pruned_complexity['parameters']['total'] / 
                                   original_complexity['parameters']['total'])
            size_reduction = 1.0 - (pruned_complexity['model_size_mb'] / 
                                  original_complexity['model_size_mb'])
            macs_reduction = 1.0 - (pruned_complexity['gmacs'] / 
                                  original_complexity['gmacs']) if original_complexity['gmacs'] > 0 else 0.0
            
            final_metrics = {
                'original_metrics': original_complexity,
                'pruned_metrics': pruned_complexity,
                'improvements': {
                    'parameter_reduction': param_reduction,
                    'size_reduction_mb': original_complexity['model_size_mb'] - pruned_complexity['model_size_mb'],
                    'size_reduction_ratio': size_reduction,
                    'macs_reduction_gmacs': original_complexity['gmacs'] - pruned_complexity['gmacs'],
                    'macs_reduction_ratio': macs_reduction
                },
                'efficiency_gains': {
                    'compression_ratio': param_reduction,
                    'estimated_speedup': 1.0 + (macs_reduction * 0.5),  # Conservative estimate
                    'memory_savings_mb': original_complexity['model_size_mb'] - pruned_complexity['model_size_mb']
                }
            }
            
            logger.info("✅ Final metrics computed")
            logger.info(f"   Parameter reduction: {param_reduction:.1%}")
            logger.info(f"   Size reduction: {size_reduction:.1%}")
            logger.info(f"   MACs reduction: {macs_reduction:.1%}")
            
            return final_metrics
    
    def _validate_pruned_model(self, state: PruningState, 
                             pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the pruned model for correctness and functionality."""
        
        with self.profiler.timer("pruned_model_validation"):
            logger.info("🔍 Validating pruned model")
            
            pruned_model = pruning_results['pruned_model']
            
            validation_tests = {
                'forward_pass_test': self._test_forward_pass(pruned_model),
                'gradient_flow_test': self._test_gradient_flow(pruned_model),
                'dimension_consistency_test': self._test_dimension_consistency(pruned_model),
                'parameter_count_test': self._test_parameter_count(pruning_results),
                'memory_test': self._test_memory_usage(pruned_model)
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
                logger.info("✅ Pruned model validation passed all tests")
            else:
                logger.warning(f"⚠️ Pruned model validation failed {len(validation_summary['failed_tests'])} tests")
            
            return validation_summary
    
    def _validate_safety_limits(self, state: PruningState) -> bool:
        """Validate that target ratio is within safety limits."""
        
        target_ratio = state.target_ratio
        model_name = state.model_name
        dataset = state.dataset
        
        # Define conservative safety limits
        safety_limits = {
            ('deit_small', 'imagenet'): 0.35,  # Max 35% for DeiT-Small on ImageNet
            ('deit_base', 'imagenet'): 0.30,   # Max 30% for DeiT-Base on ImageNet
            ('resnet50', 'imagenet'): 0.60,    # Max 60% for ResNet50 on ImageNet
            ('default', 'imagenet'): 0.40,     # Default max 40% for ImageNet
            ('default', 'cifar10'): 0.70,      # Default max 70% for CIFAR-10
        }
        
        # Get applicable limit
        key = (model_name.lower(), dataset.lower())
        if key not in safety_limits:
            key = ('default', dataset.lower())
        if key not in safety_limits:
            key = ('default', 'imagenet')  # Most conservative fallback
        
        max_safe_ratio = safety_limits[key]
        
        if target_ratio > max_safe_ratio:
            logger.error(f"❌ Target ratio {target_ratio:.1%} exceeds safety limit {max_safe_ratio:.1%} for {model_name} on {dataset}")
            return False
        
        return True
    def _get_llm_validation(self, state: PruningState, pruning_results: Dict[str, Any],
                          validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM-based validation of pruning results."""
        
        if not self.llm_client:
            return {'status': 'llm_not_available', 'message': 'LLM client not configured'}
        
        # Create prompt for LLM validation
        prompt = self._create_llm_validation_prompt(state, pruning_results, validation_results)
        
        try:
            with self.profiler.timer("llm_validation"):
                response = self.llm_client.generate(prompt)
                
                # Parse LLM response
                llm_validation = self._parse_llm_validation_response(response)
                
                logger.info("🤖 LLM validation completed successfully")
                return llm_validation
                
        except Exception as e:
            logger.warning(f"⚠️ LLM validation failed: {str(e)}")
            return {
                'status': 'llm_validation_failed',
                'error': str(e),
                'fallback_used': True
            }
    
    def _attempt_recovery(self, state: PruningState, error_message: str) -> Dict[str, Any]:
        """Attempt to recover from pruning failure."""
        
        logger.info("🔄 Attempting recovery from pruning failure")
        
        recovery_actions = []
        
        # Check if we have a checkpoint to restore
        if 'pre_pruning' in self.checkpoints:
            try:
                # Restore from checkpoint
                checkpoint = self.checkpoints['pre_pruning']
                state.model.load_state_dict(checkpoint['model_state_dict'])
                recovery_actions.append("Restored model from pre-pruning checkpoint")
                
                logger.info("✅ Model restored from checkpoint")
                
            except Exception as e:
                recovery_actions.append(f"Failed to restore from checkpoint: {str(e)}")
                logger.error(f"❌ Checkpoint restoration failed: {str(e)}")
        
        # Suggest alternative strategies
        alternative_strategies = [
            "Reduce target pruning ratio by 50%",
            "Use more conservative group ratios",
            "Switch to magnitude-based importance criterion",
            "Apply gradual pruning instead of one-shot"
        ]
        
        return {
            'recovery_attempted': True,
            'recovery_actions': recovery_actions,
            'alternative_strategies': alternative_strategies,
            'checkpoint_available': 'pre_pruning' in self.checkpoints,
            'error_context': error_message
        }
    
    # Helper methods for validation
    def _validate_dimension_constraints(self, model: nn.Module, 
                                      dependency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dimension constraints are satisfied."""
        
        violations = []
        warnings = []
        
        # Check coupling constraints
        coupling_constraints = dependency_analysis.get('coupling_constraints', [])
        
        for constraint in coupling_constraints:
            constraint_type = constraint['type']
            layers = constraint['layers']
            
            if constraint_type == 'mlp_coupling':
                # Validate MLP fc1-fc2 dimension matching
                if len(layers) >= 2:
                    try:
                        fc1_layer = dict(model.named_modules())[layers[0]]
                        fc2_layer = dict(model.named_modules())[layers[1]]
                        
                        if hasattr(fc1_layer, 'out_features') and hasattr(fc2_layer, 'in_features'):
                            if fc1_layer.out_features != fc2_layer.in_features:
                                violations.append(f"MLP dimension mismatch: {layers[0]} out_features != {layers[1]} in_features")
                    except KeyError:
                        warnings.append(f"Could not validate MLP coupling for layers: {layers}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }
    
    def _validate_safety_constraints(self, pruning_results: Dict[str, Any],
                                   safety_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety constraints are satisfied."""
        
        violations = []
        warnings = []
        
        # Check achieved pruning ratio against safety limits
        achieved_ratio = pruning_results['achieved_pruning_ratio']
        safety_thresholds = safety_analysis.get('safety_thresholds', {})
        
        max_safe_ratio = 0.8  # 80% maximum safe pruning ratio
        if achieved_ratio > max_safe_ratio:
            violations.append(f"Achieved pruning ratio {achieved_ratio:.1%} exceeds safety limit {max_safe_ratio:.1%}")
        
        # Check for catastrophic parameter reduction
        if achieved_ratio > 0.9:
            violations.append(f"Catastrophic pruning ratio detected: {achieved_ratio:.1%}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }
    
    def _validate_architectural_constraints(self, model: nn.Module, 
                                          state: PruningState) -> Dict[str, Any]:
        """Validate architectural constraints are satisfied."""
        
        violations = []
        warnings = []
        
        # Check for zero-dimension layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.in_features == 0 or module.out_features == 0:
                    violations.append(f"Linear layer {name} has zero dimensions")
            elif isinstance(module, nn.Conv2d):
                if module.in_channels == 0 or module.out_channels == 0:
                    violations.append(f"Conv2d layer {name} has zero channels")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }
    
    def _validate_performance_constraints(self, pruning_results: Dict[str, Any],
                                        state: PruningState) -> Dict[str, Any]:
        """Validate performance constraints are satisfied."""
        
        violations = []
        warnings = []
        
        # Check if achieved ratio is close to target
        target_ratio = state.target_pruning_ratio
        achieved_ratio = pruning_results['achieved_pruning_ratio']
        ratio_difference = abs(achieved_ratio - target_ratio)
        
        if ratio_difference > 0.1:  # 10% tolerance
            warnings.append(f"Achieved ratio {achieved_ratio:.1%} differs from target {target_ratio:.1%} by {ratio_difference:.1%}")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }
    
    # Helper methods for model testing
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
    
    def _test_dimension_consistency(self, model: nn.Module) -> Dict[str, Any]:
        """Test dimension consistency across layers."""
        
        try:
            inconsistencies = []
            
            # Check for dimension mismatches in sequential layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    for i in range(len(module) - 1):
                        current_layer = module[i]
                        next_layer = module[i + 1]
                        
                        # Check Linear layer connections
                        if isinstance(current_layer, nn.Linear) and isinstance(next_layer, nn.Linear):
                            if current_layer.out_features != next_layer.in_features:
                                inconsistencies.append(f"Dimension mismatch in {name}: layer {i} -> {i+1}")
            
            return {
                'passed': len(inconsistencies) == 0,
                'inconsistencies': inconsistencies,
                'message': 'Dimension consistency test passed' if len(inconsistencies) == 0 else f'{len(inconsistencies)} inconsistencies found'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Dimension consistency test failed'
            }
    
    def _test_parameter_count(self, pruning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test if parameter count matches expectations."""
        
        try:
            expected_params = pruning_results['pruned_parameters']
            actual_params = sum(p.numel() for p in pruning_results['pruned_model'].parameters())
            
            matches = expected_params == actual_params
            
            return {
                'passed': matches,
                'expected_params': expected_params,
                'actual_params': actual_params,
                'message': 'Parameter count matches' if matches else f'Parameter count mismatch: expected {expected_params}, got {actual_params}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Parameter count test failed'
            }
    
    def _test_memory_usage(self, model: nn.Module) -> Dict[str, Any]:
        """Test memory usage of the pruned model."""
        
        try:
            # Get model size
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            
            # Test if model can be moved to GPU (if available)
            if torch.cuda.is_available():
                device = torch.device('cuda')
                model_copy = type(model)()
                model_copy.load_state_dict(model.state_dict())
                model_copy.to(device)
                
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                
                return {
                    'passed': True,
                    'model_size_mb': model_size_mb,
                    'gpu_memory_mb': gpu_memory_mb,
                    'message': 'Memory usage test passed'
                }
            else:
                return {
                    'passed': True,
                    'model_size_mb': model_size_mb,
                    'message': 'Memory usage test passed (CPU only)'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'Memory usage test failed'
            }
    
    def _analyze_importance_distribution(self, importance_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the distribution of importance scores."""
        
        scores = list(importance_scores.values())
        
        if not scores:
            return {'error': 'No importance scores to analyze'}
        
        import numpy as np
        
        return {
            'total_layers': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'median_score': float(np.median(scores)),
            'score_range': float(np.max(scores) - np.min(scores))
        }
    
    def _create_llm_validation_prompt(self, state: PruningState, pruning_results: Dict[str, Any],
                                    validation_results: Dict[str, Any]) -> str:
        """Create prompt for LLM validation."""
        
        model_name = getattr(state, 'model_name', 'unknown')
        target_ratio = state.target_pruning_ratio
        achieved_ratio = pruning_results['achieved_pruning_ratio']
        
        prompt = f"""
You are an expert in neural network pruning. Please validate the following pruning results:

## Model Information:
- Model: {model_name}
- Target Pruning Ratio: {target_ratio:.1%}
- Achieved Pruning Ratio: {achieved_ratio:.1%}

## Pruning Results:
{json.dumps(pruning_results, indent=2, default=str)}

## Validation Results:
{json.dumps(validation_results, indent=2, default=str)}

Please provide validation in JSON format:
{{
  "overall_assessment": "success|warning|failure",
  "concerns": [list of concerns],
  "recommendations": [list of recommendations],
  "confidence_score": float between 0 and 1
}}
"""
        
        return prompt
    
    def _parse_llm_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM validation response."""
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                llm_validation = json.loads(json_str)
                
                return {
                    'status': 'success',
                    'validation': llm_validation,
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
    
    def _create_error_result(self, error_message: str, 
                           recovery_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized error result."""
        
        result = {
            'success': False,
            'agent_name': self.agent_name,
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'next_agent': None
        }
        
        if recovery_info:
            result['recovery_info'] = recovery_info
        
        return result

    def get_agent_role(self) -> str:
        """Return the role of this agent."""
        return "pruning_agent"
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for the pruning agent."""
        
        model_info = context.get('model_info', {})
        pruning_config = context.get('pruning_config', {})
        safety_constraints = context.get('safety_constraints', {})
        
        prompt = f"""You are an expert Pruning Agent in a multi-agent neural network pruning system.

ROLE: Execute structured pruning operations with safety guarantees and real-time monitoring.

CURRENT CONTEXT:
- Model: {model_info.get('name', 'Unknown')} ({model_info.get('total_params', 0):,} parameters)
- Architecture: {model_info.get('architecture_type', 'Unknown')}
- Target: {pruning_config.get('target_ratio', 0.5)*100:.1f}% compression
- Method: {pruning_config.get('method', 'structured')} pruning

SAFETY CONSTRAINTS:
- Max MLP pruning: {safety_constraints.get('max_mlp_ratio', 0.15)*100:.1f}%
- Max Attention pruning: {safety_constraints.get('max_attention_ratio', 0.10)*100:.1f}%
- Min accuracy threshold: {safety_constraints.get('min_accuracy', 0.70)*100:.1f}%

RESPONSIBILITIES:
1. Execute coordinated pruning for coupled layers (MLP fc1↔fc2, Attention qkv↔proj)
2. Apply importance-guided selection with safety validation
3. Monitor accuracy degradation and trigger rollback if needed
4. Maintain detailed statistics and progress tracking
5. Ensure architectural constraints are preserved

DECISION FRAMEWORK:
- SAFETY FIRST: Always validate constraints before pruning
- COORDINATED: Prune coupled layers together to maintain dimensions
- MONITORED: Track accuracy after each major pruning step
- RECOVERABLE: Maintain checkpoints for rollback capability

Respond with structured decisions including rationale, safety validation, and execution plan."""
        
        return prompt
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response for pruning decisions."""
        
        try:
            # Extract key decisions from response
            decisions = {}
            
            # Look for pruning ratios
            import re
            
            # Extract MLP pruning ratio
            mlp_match = re.search(r'mlp.*?(\d+\.?\d*)%', response.lower())
            if mlp_match:
                decisions['mlp_ratio'] = float(mlp_match.group(1)) / 100.0
            
            # Extract attention pruning ratio  
            attn_match = re.search(r'attention.*?(\d+\.?\d*)%', response.lower())
            if attn_match:
                decisions['attention_ratio'] = float(attn_match.group(1)) / 100.0
            
            # Extract overall strategy
            if 'aggressive' in response.lower():
                decisions['strategy'] = 'aggressive'
            elif 'conservative' in response.lower():
                decisions['strategy'] = 'conservative'
            else:
                decisions['strategy'] = 'balanced'
            
            # Extract safety validation
            decisions['safety_validated'] = 'safety' in response.lower() and 'validated' in response.lower()
            
            # Extract execution plan steps
            execution_steps = []
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['step', 'prune', 'execute', 'apply']):
                    execution_steps.append(line.strip())
            
            decisions['execution_steps'] = execution_steps[:5]  # Limit to 5 steps
            
            # Determine success based on content
            success = (
                len(decisions) > 2 and 
                'error' not in response.lower() and
                'fail' not in response.lower()
            )
            
            return AgentResponse(
                success=success,
                data=decisions,
                message=f"Parsed pruning strategy: {decisions.get('strategy', 'unknown')}",
                confidence=0.9 if success else 0.3
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                message=f"Failed to parse pruning response: {str(e)}",
                confidence=0.0
            )
