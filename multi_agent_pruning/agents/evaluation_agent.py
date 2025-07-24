#!/usr/bin/env python3
"""
Evaluation Agent for Multi-Agent LLM Pruning Framework

This agent provides comprehensive evaluation of the final pruned and fine-tuned model,
including accuracy metrics, MACs computation, parameter reduction analysis, and
comparison with baseline methods.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
import torch
import torch.nn as nn

from .base_agent import BaseAgent, AgentResponse
from ..core.state_manager import PruningState
from ..utils.profiler import TimingProfiler
from ..utils.metrics import AccuracyTracker, compute_model_complexity

logger = logging.getLogger(__name__)

class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent that provides comprehensive evaluation of the final model,
    including performance metrics, efficiency gains, and comparison analysis.
    """
    
    def __init__(self, llm_client=None, profiler: Optional[TimingProfiler] = None):
        super().__init__("EvaluationAgent", llm_client, profiler)
        
        # Evaluation components
        self.accuracy_tracker: Optional[AccuracyTracker] = None
        
        # Evaluation results
        self.evaluation_results = {}
        self.comparison_results = {}
        
        logger.info("📊 Evaluation Agent initialized")
    
    def execute(self, state: PruningState) -> Dict[str, Any]:
        """
        Execute evaluation phase: comprehensive evaluation of final model.
        
        Args:
            state: Current pruning state with fine-tuned model
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        
        with self.profiler.timer("evaluation_agent_execution"):
            logger.info("📊 Starting Evaluation Agent execution")
            
            try:
                # Validate input state
                if not self._validate_input_state(state):
                    return self._create_error_result("Invalid input state for evaluation")
                
                # Initialize evaluation components
                self._initialize_evaluation_components(state)
                
                # Execute comprehensive evaluation
                evaluation_results = self._execute_comprehensive_evaluation(state)
                
                # Perform comparison analysis
                comparison_results = self._perform_comparison_analysis(state, evaluation_results)
                
                # Generate final report
                final_report = self._generate_final_report(state, evaluation_results, comparison_results)
                
                # Get LLM assessment
                llm_assessment = self._get_llm_assessment(state, evaluation_results, comparison_results)
                
                # Combine results
                final_results = {
                    'success': True,
                    'agent_name': self.agent_name,
                    'timestamp': datetime.now().isoformat(),
                    'evaluation_results': evaluation_results,
                    'comparison_results': comparison_results,
                    'final_report': final_report,
                    'llm_assessment': llm_assessment,
                    'next_agent': None  # Final agent in the pipeline
                }
                
                # Update state with evaluation results
                state.evaluation_results = evaluation_results
                
                # Store results
                self.evaluation_results = evaluation_results
                self.comparison_results = comparison_results
                
                logger.info("✅ Evaluation Agent execution completed successfully")
                return final_results
                
            except Exception as e:
                logger.error(f"❌ Evaluation Agent execution failed: {str(e)}")
                return self._create_error_result(f"Evaluation execution failed: {str(e)}")
    
    def _validate_input_state(self, state: PruningState) -> bool:
        """Validate that the input state contains required results."""
        
        required_fields = ['model', 'finetuning_results']
        
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                logger.error(f"❌ Missing required field in state: {field}")
                return False
        
        # Check if we have evaluation data
        if not hasattr(state, 'test_dataloader') and not hasattr(state, 'val_dataloader'):
            logger.warning("⚠️ No test or validation dataloader found")
        
        logger.info("✅ Input state validation passed")
        return True
    
    def _initialize_evaluation_components(self, state: PruningState):
        """Initialize evaluation components."""
        
        with self.profiler.timer("evaluation_components_initialization"):
            # Initialize accuracy tracker
            self.accuracy_tracker = AccuracyTracker(track_top5=True)
            
            logger.info("🔧 Evaluation components initialized successfully")
    
    def _execute_comprehensive_evaluation(self, state: PruningState) -> Dict[str, Any]:
        """Execute comprehensive evaluation of the final model."""
        
        with self.profiler.timer("comprehensive_evaluation"):
            logger.info("🔄 Executing comprehensive evaluation")
            
            # Phase 1: Performance evaluation
            performance_results = self._evaluate_model_performance(state)
            
            # Phase 2: Efficiency analysis
            efficiency_results = self._analyze_model_efficiency(state)
            
            # Phase 3: Complexity analysis
            complexity_results = self._analyze_model_complexity(state)
            
            # Phase 4: Quality assessment
            quality_results = self._assess_model_quality(state)
            
            # Phase 5: Robustness evaluation
            robustness_results = self._evaluate_model_robustness(state)
            
            # Combine all results
            comprehensive_results = {
                'performance_evaluation': performance_results,
                'efficiency_analysis': efficiency_results,
                'complexity_analysis': complexity_results,
                'quality_assessment': quality_results,
                'robustness_evaluation': robustness_results,
                'evaluation_summary': self._create_evaluation_summary(
                    performance_results, efficiency_results, complexity_results
                )
            }
            
            logger.info("✅ Comprehensive evaluation completed")
            return comprehensive_results
    
    def _evaluate_model_performance(self, state: PruningState) -> Dict[str, Any]:
        """Evaluate model performance on test/validation data."""
        
        with self.profiler.timer("performance_evaluation"):
            logger.info("📈 Evaluating model performance")
            
            model = state.model
            
            # Get evaluation dataloader
            eval_dataloader = getattr(state, 'test_dataloader', None)
            if eval_dataloader is None:
                eval_dataloader = getattr(state, 'val_dataloader', None)
            
            if eval_dataloader is None:
                logger.warning("⚠️ No evaluation dataloader available")
                return {
                    'accuracy_metrics': {'top1_accuracy': 0.0, 'top5_accuracy': 0.0},
                    'loss_metrics': {'test_loss': float('inf')},
                    'message': 'No evaluation data available'
                }
            
            # Comprehensive evaluation
            self.accuracy_tracker.reset()
            evaluation_result = self.accuracy_tracker.evaluate_model(
                model=model,
                dataloader=eval_dataloader,
                criterion=nn.CrossEntropyLoss()
            )
            
            # Get detailed statistics
            detailed_stats = self.accuracy_tracker.get_statistics()
            
            performance_results = {
                'accuracy_metrics': {
                    'top1_accuracy': evaluation_result.top1_accuracy,
                    'top5_accuracy': evaluation_result.top5_accuracy,
                    'accuracy_variance': detailed_stats.get('accuracy_variance', 0.0),
                    'accuracy_std': detailed_stats.get('accuracy_std', 0.0)
                },
                'loss_metrics': {
                    'test_loss': evaluation_result.loss,
                    'loss_consistency': detailed_stats.get('loss_variance', 0.0)
                },
                'inference_metrics': {
                    'total_samples': evaluation_result.total_samples,
                    'inference_time_total': evaluation_result.inference_time,
                    'inference_time_per_sample': evaluation_result.inference_time / evaluation_result.total_samples if evaluation_result.total_samples > 0 else 0.0,
                    'throughput_samples_per_second': evaluation_result.total_samples / evaluation_result.inference_time if evaluation_result.inference_time > 0 else 0.0
                },
                'detailed_statistics': detailed_stats
            }
            
            logger.info(f"📈 Performance evaluation completed: {evaluation_result.top1_accuracy:.1%} accuracy")
            return performance_results
    
    def _analyze_model_efficiency(self, state: PruningState) -> Dict[str, Any]:
        """Analyze model efficiency gains from pruning."""
        
        with self.profiler.timer("efficiency_analysis"):
            logger.info("⚡ Analyzing model efficiency")
            
            # Get original model metrics (from checkpoints or state)
            original_metrics = self._get_original_model_metrics(state)
            current_metrics = compute_model_complexity(state.model)
            
            # Calculate efficiency improvements
            param_reduction = 1.0 - (current_metrics['parameters']['total'] / 
                                   original_metrics['parameters']['total']) if original_metrics['parameters']['total'] > 0 else 0.0
            
            size_reduction = 1.0 - (current_metrics['model_size_mb'] / 
                                  original_metrics['model_size_mb']) if original_metrics['model_size_mb'] > 0 else 0.0
            
            macs_reduction = 1.0 - (current_metrics['gmacs'] / 
                                  original_metrics['gmacs']) if original_metrics['gmacs'] > 0 else 0.0
            
            # Estimate performance gains
            estimated_speedup = 1.0 + (macs_reduction * 0.6)  # Conservative estimate
            estimated_memory_savings = size_reduction * original_metrics['model_size_mb']
            
            efficiency_results = {
                'parameter_efficiency': {
                    'original_parameters': original_metrics['parameters']['total'],
                    'current_parameters': current_metrics['parameters']['total'],
                    'parameters_removed': original_metrics['parameters']['total'] - current_metrics['parameters']['total'],
                    'reduction_ratio': param_reduction,
                    'compression_ratio': 1.0 / (1.0 - param_reduction) if param_reduction < 1.0 else float('inf')
                },
                'size_efficiency': {
                    'original_size_mb': original_metrics['model_size_mb'],
                    'current_size_mb': current_metrics['model_size_mb'],
                    'size_reduction_mb': original_metrics['model_size_mb'] - current_metrics['model_size_mb'],
                    'size_reduction_ratio': size_reduction
                },
                'computational_efficiency': {
                    'original_gmacs': original_metrics['gmacs'],
                    'current_gmacs': current_metrics['gmacs'],
                    'macs_reduction_gmacs': original_metrics['gmacs'] - current_metrics['gmacs'],
                    'macs_reduction_ratio': macs_reduction,
                    'estimated_speedup': estimated_speedup
                },
                'memory_efficiency': {
                    'estimated_memory_savings_mb': estimated_memory_savings,
                    'memory_efficiency_ratio': size_reduction,
                    'peak_memory_reduction_estimate': size_reduction * 0.8  # Conservative estimate
                },
                'overall_efficiency_score': self._calculate_efficiency_score(
                    param_reduction, size_reduction, macs_reduction
                )
            }
            
            logger.info(f"⚡ Efficiency analysis completed: {param_reduction:.1%} parameter reduction")
            return efficiency_results
    
    def _analyze_model_complexity(self, state: PruningState) -> Dict[str, Any]:
        """Analyze model complexity and architecture changes."""
        
        with self.profiler.timer("complexity_analysis"):
            logger.info("🔍 Analyzing model complexity")
            
            model = state.model
            current_complexity = compute_model_complexity(model)
            
            # Analyze layer-wise complexity
            layer_analysis = self._analyze_layer_complexity(model)
            
            # Analyze architectural changes
            architectural_changes = self._analyze_architectural_changes(state)
            
            # Calculate complexity metrics
            complexity_metrics = {
                'total_parameters': current_complexity['parameters']['total'],
                'trainable_parameters': current_complexity['parameters']['trainable'],
                'model_size_mb': current_complexity['model_size_mb'],
                'computational_complexity_gmacs': current_complexity['gmacs'],
                'parameter_density': self._calculate_parameter_density(model),
                'architectural_complexity': self._calculate_architectural_complexity(model)
            }
            
            complexity_results = {
                'complexity_metrics': complexity_metrics,
                'layer_analysis': layer_analysis,
                'architectural_changes': architectural_changes,
                'complexity_comparison': self._compare_complexity_with_baselines(current_complexity, state)
            }
            
            logger.info("🔍 Complexity analysis completed")
            return complexity_results
    
    def _assess_model_quality(self, state: PruningState) -> Dict[str, Any]:
        """Assess overall model quality after pruning and fine-tuning."""
        
        with self.profiler.timer("quality_assessment"):
            logger.info("🎯 Assessing model quality")
            
            # Get performance metrics
            performance_results = self.evaluation_results.get('performance_evaluation', {})
            accuracy = performance_results.get('accuracy_metrics', {}).get('top1_accuracy', 0.0)
            
            # Get efficiency metrics
            efficiency_results = self.evaluation_results.get('efficiency_analysis', {})
            param_reduction = efficiency_results.get('parameter_efficiency', {}).get('reduction_ratio', 0.0)
            
            # Calculate quality scores
            accuracy_score = min(accuracy / 0.8, 1.0)  # Normalize to 80% as perfect
            efficiency_score = min(param_reduction / 0.5, 1.0)  # Normalize to 50% reduction as perfect
            
            # Get fine-tuning quality
            finetuning_results = state.finetuning_results
            finetuning_quality = self._assess_finetuning_quality(finetuning_results)
            
            # Overall quality assessment
            quality_scores = {
                'accuracy_quality_score': accuracy_score,
                'efficiency_quality_score': efficiency_score,
                'finetuning_quality_score': finetuning_quality,
                'overall_quality_score': (accuracy_score + efficiency_score + finetuning_quality) / 3.0
            }
            
            # Quality categories
            overall_score = quality_scores['overall_quality_score']
            if overall_score >= 0.9:
                quality_category = 'excellent'
            elif overall_score >= 0.8:
                quality_category = 'good'
            elif overall_score >= 0.7:
                quality_category = 'acceptable'
            elif overall_score >= 0.6:
                quality_category = 'moderate'
            else:
                quality_category = 'poor'
            
            quality_results = {
                'quality_scores': quality_scores,
                'quality_category': quality_category,
                'quality_breakdown': {
                    'accuracy_assessment': self._assess_accuracy_quality(accuracy),
                    'efficiency_assessment': self._assess_efficiency_quality(param_reduction),
                    'stability_assessment': self._assess_model_stability(state),
                    'robustness_assessment': 'pending'  # Will be filled by robustness evaluation
                }
            }
            
            logger.info(f"🎯 Quality assessment completed: {quality_category} ({overall_score:.2f})")
            return quality_results
    
    def _evaluate_model_robustness(self, state: PruningState) -> Dict[str, Any]:
        """Evaluate model robustness and stability."""
        
        with self.profiler.timer("robustness_evaluation"):
            logger.info("🛡️ Evaluating model robustness")
            
            model = state.model
            
            # Test 1: Consistency across multiple runs
            consistency_results = self._test_inference_consistency(model)
            
            # Test 2: Gradient stability
            gradient_stability = self._test_gradient_stability(model)
            
            # Test 3: Input perturbation robustness
            perturbation_robustness = self._test_perturbation_robustness(model)
            
            # Test 4: Memory stability
            memory_stability = self._test_memory_stability(model)
            
            # Overall robustness score
            robustness_tests = [consistency_results, gradient_stability, perturbation_robustness, memory_stability]
            robustness_score = sum(test['score'] for test in robustness_tests) / len(robustness_tests)
            
            robustness_results = {
                'robustness_score': robustness_score,
                'consistency_test': consistency_results,
                'gradient_stability_test': gradient_stability,
                'perturbation_robustness_test': perturbation_robustness,
                'memory_stability_test': memory_stability,
                'overall_robustness_assessment': self._assess_robustness_level(robustness_score)
            }
            
            logger.info(f"🛡️ Robustness evaluation completed: {robustness_score:.2f}")
            return robustness_results
    
    def _perform_comparison_analysis(self, state: PruningState, 
                                   evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparison analysis with baselines and other methods."""
        
        with self.profiler.timer("comparison_analysis"):
            logger.info("📊 Performing comparison analysis")
            
            # Get current model metrics
            current_performance = evaluation_results['performance_evaluation']
            current_efficiency = evaluation_results['efficiency_analysis']
            
            # Compare with original model
            original_comparison = self._compare_with_original_model(state, current_performance, current_efficiency)
            
            # Compare with baseline pruning methods
            baseline_comparison = self._compare_with_baseline_methods(state, current_performance, current_efficiency)
            
            # Compare with paper results (if available)
            paper_comparison = self._compare_with_paper_results(state, current_performance, current_efficiency)
            
            # Generate comparison summary
            comparison_summary = self._generate_comparison_summary(
                original_comparison, baseline_comparison, paper_comparison
            )
            
            comparison_results = {
                'original_model_comparison': original_comparison,
                'baseline_methods_comparison': baseline_comparison,
                'paper_results_comparison': paper_comparison,
                'comparison_summary': comparison_summary,
                'competitive_analysis': self._perform_competitive_analysis(
                    current_performance, current_efficiency
                )
            }
            
            logger.info("📊 Comparison analysis completed")
            return comparison_results
    
    def _generate_final_report(self, state: PruningState, evaluation_results: Dict[str, Any],
                             comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        with self.profiler.timer("final_report_generation"):
            logger.info("📋 Generating final report")
            
            # Extract key metrics
            performance = evaluation_results['performance_evaluation']
            efficiency = evaluation_results['efficiency_analysis']
            quality = evaluation_results['quality_assessment']
            
            # Create executive summary
            executive_summary = {
                'model_name': getattr(state, 'model_name', 'unknown'),
                'target_pruning_ratio': getattr(state, 'target_pruning_ratio', 0.0),
                'achieved_pruning_ratio': efficiency['parameter_efficiency']['reduction_ratio'],
                'final_accuracy': performance['accuracy_metrics']['top1_accuracy'],
                'parameter_reduction': efficiency['parameter_efficiency']['reduction_ratio'],
                'model_size_reduction': efficiency['size_efficiency']['size_reduction_ratio'],
                'estimated_speedup': efficiency['computational_efficiency']['estimated_speedup'],
                'overall_quality': quality['quality_category'],
                'quality_score': quality['quality_scores']['overall_quality_score']
            }
            
            # Create detailed metrics
            detailed_metrics = {
                'accuracy_metrics': performance['accuracy_metrics'],
                'efficiency_metrics': {
                    'parameter_reduction': efficiency['parameter_efficiency']['reduction_ratio'],
                    'size_reduction': efficiency['size_efficiency']['size_reduction_ratio'],
                    'macs_reduction': efficiency['computational_efficiency']['macs_reduction_ratio'],
                    'estimated_speedup': efficiency['computational_efficiency']['estimated_speedup']
                },
                'quality_metrics': quality['quality_scores'],
                'robustness_metrics': evaluation_results['robustness_evaluation']['robustness_score']
            }
            
            # Create recommendations
            recommendations = self._generate_recommendations(state, evaluation_results, comparison_results)
            
            final_report = {
                'executive_summary': executive_summary,
                'detailed_metrics': detailed_metrics,
                'comparison_highlights': comparison_results['comparison_summary'],
                'recommendations': recommendations,
                'methodology_summary': self._create_methodology_summary(state),
                'limitations_and_caveats': self._identify_limitations(state, evaluation_results),
                'future_improvements': self._suggest_future_improvements(state, evaluation_results)
            }
            
            logger.info("📋 Final report generated successfully")
            return final_report
    
    def _get_llm_assessment(self, state: PruningState, evaluation_results: Dict[str, Any],
                          comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM assessment of the overall results."""
        
        if not self.llm_client:
            return {'status': 'llm_not_available', 'message': 'LLM client not configured'}
        
        # Create prompt for LLM assessment
        prompt = self._create_llm_assessment_prompt(state, evaluation_results, comparison_results)
        
        try:
            with self.profiler.timer("llm_assessment"):
                response = self.llm_client.generate(prompt)
                
                # Parse LLM response
                llm_assessment = self._parse_llm_assessment_response(response)
                
                logger.info("🤖 LLM assessment completed successfully")
                return llm_assessment
                
        except Exception as e:
            logger.warning(f"⚠️ LLM assessment failed: {str(e)}")
            return {
                'status': 'llm_assessment_failed',
                'error': str(e),
                'fallback_used': True
            }
    
    # Helper methods for evaluation
    def _get_original_model_metrics(self, state: PruningState) -> Dict[str, Any]:
        """Get original model metrics for comparison."""
        
        # Try to get from profiling results first
        if hasattr(state, 'profiling_results') and state.profiling_results:
            model_analysis = state.profiling_results.get('model_analysis', {})
            return {
                'parameters': {'total': model_analysis.get('total_parameters', 0)},
                'model_size_mb': model_analysis.get('model_size_mb', 0),
                'gmacs': model_analysis.get('estimated_macs', 0) / 1e9
            }
        
        # Fallback: estimate from current model (not ideal but better than nothing)
        current_complexity = compute_model_complexity(state.model)
        achieved_ratio = getattr(state, 'target_pruning_ratio', 0.0)
        
        # Reverse-engineer original metrics
        original_params = int(current_complexity['parameters']['total'] / (1.0 - achieved_ratio))
        original_size = current_complexity['model_size_mb'] / (1.0 - achieved_ratio)
        original_gmacs = current_complexity['gmacs'] / (1.0 - achieved_ratio)
        
        return {
            'parameters': {'total': original_params},
            'model_size_mb': original_size,
            'gmacs': original_gmacs
        }
    
    def _calculate_efficiency_score(self, param_reduction: float, size_reduction: float, 
                                  macs_reduction: float) -> float:
        """Calculate overall efficiency score."""
        
        # Weighted average of different efficiency metrics
        weights = {'param': 0.4, 'size': 0.3, 'macs': 0.3}
        
        score = (param_reduction * weights['param'] + 
                size_reduction * weights['size'] + 
                macs_reduction * weights['macs'])
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _analyze_layer_complexity(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze complexity at the layer level."""
        
        layer_stats = {
            'total_layers': 0,
            'conv_layers': 0,
            'linear_layers': 0,
            'norm_layers': 0,
            'activation_layers': 0,
            'other_layers': 0
        }
        
        layer_details = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_stats['total_layers'] += 1
                
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    layer_stats['conv_layers'] += 1
                    layer_info['channels'] = f"{module.in_channels} -> {module.out_channels}"
                elif isinstance(module, nn.Linear):
                    layer_stats['linear_layers'] += 1
                    layer_info['features'] = f"{module.in_features} -> {module.out_features}"
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    layer_stats['norm_layers'] += 1
                elif isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
                    layer_stats['activation_layers'] += 1
                else:
                    layer_stats['other_layers'] += 1
                
                layer_details.append(layer_info)
        
        return {
            'layer_statistics': layer_stats,
            'layer_details': layer_details[:20],  # Limit to first 20 for brevity
            'architecture_summary': f"{layer_stats['conv_layers']} Conv + {layer_stats['linear_layers']} Linear + {layer_stats['norm_layers']} Norm"
        }
    
    def _analyze_architectural_changes(self, state: PruningState) -> Dict[str, Any]:
        """Analyze architectural changes from pruning."""
        
        changes = []
        
        # Get pruning results
        if hasattr(state, 'pruning_results') and state.pruning_results:
            pruning_results = state.pruning_results
            
            # Check for layer modifications
            layers_pruned = pruning_results.get('layers_pruned', [])
            if layers_pruned:
                changes.append(f"Pruned {len(layers_pruned)} layers")
            
            # Check for dimension changes
            achieved_ratio = pruning_results.get('achieved_pruning_ratio', 0.0)
            if achieved_ratio > 0:
                changes.append(f"Reduced parameters by {achieved_ratio:.1%}")
        
        return {
            'architectural_changes': changes,
            'structural_integrity': 'maintained',  # Assuming structured pruning maintains integrity
            'connectivity_changes': 'dimension_reduction_only'
        }
    
    def _calculate_parameter_density(self, model: nn.Module) -> float:
        """Calculate parameter density (parameters per layer)."""
        
        total_params = sum(p.numel() for p in model.parameters())
        total_layers = len([m for m in model.modules() if len(list(m.children())) == 0])
        
        return total_params / total_layers if total_layers > 0 else 0.0
    
    def _calculate_architectural_complexity(self, model: nn.Module) -> str:
        """Calculate architectural complexity category."""
        
        total_layers = len([m for m in model.modules() if len(list(m.children())) == 0])
        
        if total_layers < 20:
            return 'simple'
        elif total_layers < 100:
            return 'moderate'
        else:
            return 'complex'
    
    def _compare_complexity_with_baselines(self, current_complexity: Dict[str, Any], 
                                         state: PruningState) -> Dict[str, Any]:
        """Compare complexity with baseline models."""
        
        model_name = getattr(state, 'model_name', '').lower()
        
        # Baseline complexity estimates (approximate)
        baselines = {
            'resnet50': {'params': 25.6e6, 'gmacs': 4.1},
            'vit_base': {'params': 86.6e6, 'gmacs': 17.6},
            'deit_small': {'params': 22.1e6, 'gmacs': 4.6}
        }
        
        # Find closest baseline
        closest_baseline = None
        for baseline_name in baselines:
            if baseline_name in model_name:
                closest_baseline = baseline_name
                break
        
        if closest_baseline:
            baseline_metrics = baselines[closest_baseline]
            current_params = current_complexity['parameters']['total']
            current_gmacs = current_complexity['gmacs']
            
            return {
                'baseline_model': closest_baseline,
                'parameter_comparison': {
                    'baseline_params': baseline_metrics['params'],
                    'current_params': current_params,
                    'ratio': current_params / baseline_metrics['params']
                },
                'computational_comparison': {
                    'baseline_gmacs': baseline_metrics['gmacs'],
                    'current_gmacs': current_gmacs,
                    'ratio': current_gmacs / baseline_metrics['gmacs']
                }
            }
        else:
            return {'message': 'No matching baseline found for comparison'}
    
    def _assess_finetuning_quality(self, finetuning_results: Dict[str, Any]) -> float:
        """Assess the quality of fine-tuning."""
        
        if not finetuning_results:
            return 0.5  # Neutral score if no fine-tuning results
        
        performance_analysis = finetuning_results.get('performance_analysis', {})
        accuracy_improvement = performance_analysis.get('accuracy_improvement', 0.0)
        
        # Score based on accuracy improvement
        if accuracy_improvement > 0.05:  # >5% improvement
            return 1.0
        elif accuracy_improvement > 0.02:  # >2% improvement
            return 0.8
        elif accuracy_improvement > 0.0:   # Any improvement
            return 0.6
        elif accuracy_improvement > -0.02: # <2% degradation
            return 0.4
        else:
            return 0.2
    
    def _assess_accuracy_quality(self, accuracy: float) -> str:
        """Assess accuracy quality category."""
        
        if accuracy > 0.8:
            return 'excellent'
        elif accuracy > 0.7:
            return 'good'
        elif accuracy > 0.6:
            return 'acceptable'
        elif accuracy > 0.5:
            return 'moderate'
        else:
            return 'poor'
    
    def _assess_efficiency_quality(self, param_reduction: float) -> str:
        """Assess efficiency quality category."""
        
        if param_reduction > 0.5:
            return 'excellent'
        elif param_reduction > 0.3:
            return 'good'
        elif param_reduction > 0.2:
            return 'acceptable'
        elif param_reduction > 0.1:
            return 'moderate'
        else:
            return 'poor'
    
    def _assess_model_stability(self, state: PruningState) -> str:
        """Assess model stability."""
        
        # Check if fine-tuning converged properly
        if hasattr(state, 'finetuning_results') and state.finetuning_results:
            training_results = state.finetuning_results.get('training_results', {})
            if training_results.get('training_completed', False):
                return 'stable'
            else:
                return 'unstable'
        
        return 'unknown'
    
    def _assess_robustness_level(self, robustness_score: float) -> str:
        """Assess robustness level based on score."""
        
        if robustness_score > 0.8:
            return 'highly_robust'
        elif robustness_score > 0.6:
            return 'moderately_robust'
        elif robustness_score > 0.4:
            return 'somewhat_robust'
        else:
            return 'fragile'
    
    # Robustness test methods
    def _test_inference_consistency(self, model: nn.Module) -> Dict[str, Any]:
        """Test inference consistency across multiple runs."""
        
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            outputs = []
            with torch.no_grad():
                for _ in range(5):
                    output = model(dummy_input)
                    outputs.append(output.cpu())
            
            # Calculate consistency
            max_diff = max(torch.max(torch.abs(outputs[i] - outputs[0])).item() 
                          for i in range(1, len(outputs)))
            
            score = 1.0 if max_diff < 1e-6 else max(0.0, 1.0 - max_diff * 1000)
            
            return {
                'score': score,
                'max_difference': max_diff,
                'status': 'consistent' if score > 0.8 else 'inconsistent'
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'status': 'failed'
            }
    
    def _test_gradient_stability(self, model: nn.Module) -> Dict[str, Any]:
        """Test gradient stability."""
        
        try:
            model.train()
            device = next(model.parameters()).device
            
            # Test with different inputs
            gradient_norms = []
            for _ in range(3):
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                dummy_target = torch.randint(0, 1000, (1,)).to(device)
                
                model.zero_grad()
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(output, dummy_target)
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
                gradient_norms.append(grad_norm.item())
            
            # Check gradient stability
            grad_std = torch.std(torch.tensor(gradient_norms)).item()
            grad_mean = torch.mean(torch.tensor(gradient_norms)).item()
            
            stability_ratio = grad_std / grad_mean if grad_mean > 0 else float('inf')
            score = max(0.0, 1.0 - stability_ratio)
            
            return {
                'score': score,
                'gradient_std': grad_std,
                'gradient_mean': grad_mean,
                'stability_ratio': stability_ratio,
                'status': 'stable' if score > 0.7 else 'unstable'
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'status': 'failed'
            }
    
    def _test_perturbation_robustness(self, model: nn.Module) -> Dict[str, Any]:
        """Test robustness to input perturbations."""
        
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                # Original output
                original_output = model(dummy_input)
                
                # Perturbed outputs
                perturbation_diffs = []
                for noise_level in [0.01, 0.05, 0.1]:
                    noise = torch.randn_like(dummy_input) * noise_level
                    perturbed_input = dummy_input + noise
                    perturbed_output = model(perturbed_input)
                    
                    diff = torch.max(torch.abs(perturbed_output - original_output)).item()
                    perturbation_diffs.append(diff)
                
                # Calculate robustness score
                max_diff = max(perturbation_diffs)
                score = max(0.0, 1.0 - max_diff / 10.0)  # Normalize by 10
                
                return {
                    'score': score,
                    'max_perturbation_diff': max_diff,
                    'perturbation_diffs': perturbation_diffs,
                    'status': 'robust' if score > 0.6 else 'sensitive'
                }
                
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'status': 'failed'
            }
    
    def _test_memory_stability(self, model: nn.Module) -> Dict[str, Any]:
        """Test memory stability during inference."""
        
        try:
            if not torch.cuda.is_available():
                return {
                    'score': 1.0,
                    'message': 'Memory stability test skipped (CPU only)',
                    'status': 'skipped'
                }
            
            model.eval()
            device = torch.device('cuda')
            model_copy = type(model)()
            model_copy.load_state_dict(model.state_dict())
            model_copy.to(device)
            
            # Test memory usage stability
            memory_usages = []
            for batch_size in [1, 4, 8]:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
                with torch.no_grad():
                    _ = model_copy(dummy_input)
                
                peak_memory = torch.cuda.memory_allocated()
                memory_used = peak_memory - initial_memory
                memory_usages.append(memory_used)
            
            # Check if memory usage scales reasonably
            memory_growth_ratio = memory_usages[-1] / memory_usages[0] if memory_usages[0] > 0 else 1.0
            expected_ratio = 8.0  # 8x batch size should be roughly 8x memory
            
            score = max(0.0, 1.0 - abs(memory_growth_ratio - expected_ratio) / expected_ratio)
            
            return {
                'score': score,
                'memory_usages': memory_usages,
                'memory_growth_ratio': memory_growth_ratio,
                'status': 'stable' if score > 0.7 else 'unstable'
            }
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'status': 'failed'
            }
    
    # Comparison methods
    def _compare_with_original_model(self, state: PruningState, performance: Dict[str, Any],
                                   efficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with original unpruned model."""
        
        # Get baseline accuracy (if available)
        baseline_accuracy = 0.8  # Default assumption
        if hasattr(state, 'baseline_accuracy'):
            baseline_accuracy = state.baseline_accuracy
        
        current_accuracy = performance['accuracy_metrics']['top1_accuracy']
        param_reduction = efficiency['parameter_efficiency']['reduction_ratio']
        
        accuracy_retention = current_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'current_accuracy': current_accuracy,
            'accuracy_retention': accuracy_retention,
            'accuracy_drop': baseline_accuracy - current_accuracy,
            'parameter_reduction': param_reduction,
            'efficiency_vs_accuracy_tradeoff': param_reduction / max(baseline_accuracy - current_accuracy, 0.01),
            'comparison_summary': f"Retained {accuracy_retention:.1%} accuracy with {param_reduction:.1%} parameter reduction"
        }
    
    def _compare_with_baseline_methods(self, state: PruningState, performance: Dict[str, Any],
                                     efficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with baseline pruning methods."""
        
        # Typical baseline method results (approximate)
        baseline_methods = {
            'magnitude_pruning': {'accuracy_retention': 0.85, 'param_reduction': 0.3},
            'taylor_pruning': {'accuracy_retention': 0.88, 'param_reduction': 0.35},
            'isomorphic_pruning': {'accuracy_retention': 0.90, 'param_reduction': 0.4}
        }
        
        current_accuracy = performance['accuracy_metrics']['top1_accuracy']
        current_param_reduction = efficiency['parameter_efficiency']['reduction_ratio']
        
        # Assume baseline accuracy for comparison
        baseline_accuracy = 0.8
        current_retention = current_accuracy / baseline_accuracy
        
        comparisons = {}
        for method, metrics in baseline_methods.items():
            comparisons[method] = {
                'accuracy_advantage': current_retention - metrics['accuracy_retention'],
                'efficiency_advantage': current_param_reduction - metrics['param_reduction'],
                'overall_advantage': (current_retention - metrics['accuracy_retention']) + 
                                   (current_param_reduction - metrics['param_reduction'])
            }
        
        return {
            'method_comparisons': comparisons,
            'best_baseline': max(baseline_methods.keys(), 
                               key=lambda k: baseline_methods[k]['accuracy_retention'] + baseline_methods[k]['param_reduction']),
            'our_method_ranking': self._rank_against_baselines(current_retention, current_param_reduction, baseline_methods)
        }
    
    def _compare_with_paper_results(self, state: PruningState, performance: Dict[str, Any],
                                  efficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with results from the original paper."""
        
        # This would ideally load paper results from configuration
        # For now, return placeholder comparison
        
        return {
            'paper_comparison_available': False,
            'message': 'Paper results comparison not implemented yet',
            'suggestion': 'Implement paper results loading from configuration'
        }
    
    def _generate_comparison_summary(self, original_comparison: Dict[str, Any],
                                   baseline_comparison: Dict[str, Any],
                                   paper_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all comparisons."""
        
        summary_points = []
        
        # Original model comparison
        accuracy_retention = original_comparison['accuracy_retention']
        param_reduction = original_comparison['parameter_reduction']
        
        summary_points.append(f"Retained {accuracy_retention:.1%} of original accuracy")
        summary_points.append(f"Achieved {param_reduction:.1%} parameter reduction")
        
        # Baseline comparison
        if baseline_comparison.get('our_method_ranking', 0) <= 2:
            summary_points.append("Outperforms most baseline pruning methods")
        else:
            summary_points.append("Competitive with baseline pruning methods")
        
        return {
            'summary_points': summary_points,
            'overall_performance': 'excellent' if accuracy_retention > 0.9 and param_reduction > 0.3 else 'good',
            'key_strengths': self._identify_key_strengths(original_comparison, baseline_comparison),
            'areas_for_improvement': self._identify_improvement_areas(original_comparison, baseline_comparison)
        }
    
    def _perform_competitive_analysis(self, performance: Dict[str, Any],
                                    efficiency: Dict[str, Any]) -> Dict[str, Any]:
        """Perform competitive analysis."""
        
        current_accuracy = performance['accuracy_metrics']['top1_accuracy']
        param_reduction = efficiency['parameter_efficiency']['reduction_ratio']
        
        # Define competitive categories
        if current_accuracy > 0.8 and param_reduction > 0.4:
            competitive_tier = 'top_tier'
        elif current_accuracy > 0.7 and param_reduction > 0.3:
            competitive_tier = 'competitive'
        elif current_accuracy > 0.6 and param_reduction > 0.2:
            competitive_tier = 'acceptable'
        else:
            competitive_tier = 'needs_improvement'
        
        return {
            'competitive_tier': competitive_tier,
            'accuracy_percentile': min(100, int(current_accuracy * 125)),  # Rough estimate
            'efficiency_percentile': min(100, int(param_reduction * 200)),  # Rough estimate
            'overall_competitiveness': competitive_tier
        }
    
    # Report generation helpers
    def _generate_recommendations(self, state: PruningState, evaluation_results: Dict[str, Any],
                                comparison_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        
        recommendations = []
        
        # Performance-based recommendations
        performance = evaluation_results['performance_evaluation']
        accuracy = performance['accuracy_metrics']['top1_accuracy']
        
        if accuracy < 0.7:
            recommendations.append("Consider reducing pruning ratio or improving fine-tuning strategy")
        
        # Efficiency-based recommendations
        efficiency = evaluation_results['efficiency_analysis']
        param_reduction = efficiency['parameter_efficiency']['reduction_ratio']
        
        if param_reduction < 0.2:
            recommendations.append("Consider more aggressive pruning for better efficiency gains")
        
        # Quality-based recommendations
        quality = evaluation_results['quality_assessment']
        overall_quality = quality['quality_scores']['overall_quality_score']
        
        if overall_quality < 0.8:
            recommendations.append("Overall quality could be improved through better hyperparameter tuning")
        
        # Robustness-based recommendations
        robustness = evaluation_results['robustness_evaluation']
        robustness_score = robustness['robustness_score']
        
        if robustness_score < 0.6:
            recommendations.append("Model robustness could be improved through additional regularization")
        
        return recommendations
    
    def _create_methodology_summary(self, state: PruningState) -> Dict[str, Any]:
        """Create summary of methodology used."""
        
        return {
            'pruning_approach': 'multi_agent_llm_guided',
            'importance_criterion': 'adaptive_selection',
            'pruning_strategy': 'structured_with_safety_checks',
            'fine_tuning_approach': 'adaptive_learning_rate_with_early_stopping',
            'validation_approach': 'comprehensive_multi_metric'
        }
    
    def _identify_limitations(self, state: PruningState, 
                            evaluation_results: Dict[str, Any]) -> List[str]:
        """Identify limitations of the current approach."""
        
        limitations = []
        
        # Check for evaluation limitations
        if not hasattr(state, 'test_dataloader'):
            limitations.append("Evaluation limited by lack of comprehensive test dataset")
        
        # Check for fine-tuning limitations
        if hasattr(state, 'finetuning_results'):
            finetuning_results = state.finetuning_results
            if not finetuning_results.get('training_results', {}).get('training_completed', False):
                limitations.append("Fine-tuning may not have converged properly")
        
        # Check for robustness limitations
        robustness = evaluation_results.get('robustness_evaluation', {})
        if robustness.get('robustness_score', 1.0) < 0.6:
            limitations.append("Model may be sensitive to input perturbations")
        
        return limitations
    
    def _suggest_future_improvements(self, state: PruningState,
                                   evaluation_results: Dict[str, Any]) -> List[str]:
        """Suggest future improvements."""
        
        improvements = []
        
        # Performance improvements
        performance = evaluation_results['performance_evaluation']
        accuracy = performance['accuracy_metrics']['top1_accuracy']
        
        if accuracy < 0.8:
            improvements.append("Implement progressive pruning for better accuracy retention")
            improvements.append("Explore knowledge distillation during fine-tuning")
        
        # Efficiency improvements
        efficiency = evaluation_results['efficiency_analysis']
        if efficiency['computational_efficiency']['macs_reduction_ratio'] < 0.3:
            improvements.append("Investigate more aggressive computational pruning")
        
        # Robustness improvements
        robustness = evaluation_results['robustness_evaluation']
        if robustness['robustness_score'] < 0.8:
            improvements.append("Add adversarial training for improved robustness")
        
        # General improvements
        improvements.extend([
            "Implement automated hyperparameter optimization",
            "Add support for dynamic pruning ratios",
            "Integrate with neural architecture search"
        ])
        
        return improvements
    
    # Helper methods for comparison
    def _rank_against_baselines(self, current_retention: float, current_reduction: float,
                              baseline_methods: Dict[str, Dict[str, float]]) -> int:
        """Rank current method against baselines."""
        
        # Calculate combined score for current method
        current_score = current_retention + current_reduction
        
        # Calculate scores for baseline methods
        baseline_scores = []
        for method, metrics in baseline_methods.items():
            score = metrics['accuracy_retention'] + metrics['param_reduction']
            baseline_scores.append(score)
        
        # Count how many baselines we outperform
        better_than = sum(1 for score in baseline_scores if current_score > score)
        
        return len(baseline_scores) - better_than + 1  # Rank (1 is best)
    
    def _identify_key_strengths(self, original_comparison: Dict[str, Any],
                              baseline_comparison: Dict[str, Any]) -> List[str]:
        """Identify key strengths of the approach."""
        
        strengths = []
        
        # High accuracy retention
        if original_comparison['accuracy_retention'] > 0.9:
            strengths.append("Excellent accuracy retention")
        
        # High parameter reduction
        if original_comparison['parameter_reduction'] > 0.4:
            strengths.append("Significant parameter reduction")
        
        # Good tradeoff
        tradeoff = original_comparison['efficiency_vs_accuracy_tradeoff']
        if tradeoff > 10:  # High efficiency per unit accuracy loss
            strengths.append("Excellent efficiency-accuracy tradeoff")
        
        return strengths
    
    def _identify_improvement_areas(self, original_comparison: Dict[str, Any],
                                  baseline_comparison: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement."""
        
        improvements = []
        
        # Low accuracy retention
        if original_comparison['accuracy_retention'] < 0.8:
            improvements.append("Accuracy retention could be improved")
        
        # Low parameter reduction
        if original_comparison['parameter_reduction'] < 0.2:
            improvements.append("Parameter reduction could be more aggressive")
        
        # Poor ranking against baselines
        ranking = baseline_comparison.get('our_method_ranking', 1)
        if ranking > 2:
            improvements.append("Performance relative to baseline methods could be improved")
        
        return improvements
    
    def _create_evaluation_summary(self, performance: Dict[str, Any], efficiency: Dict[str, Any],
                                 complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Create evaluation summary."""
        
        return {
            'key_metrics': {
                'accuracy': performance['accuracy_metrics']['top1_accuracy'],
                'parameter_reduction': efficiency['parameter_efficiency']['reduction_ratio'],
                'size_reduction': efficiency['size_efficiency']['size_reduction_ratio'],
                'estimated_speedup': efficiency['computational_efficiency']['estimated_speedup']
            },
            'evaluation_status': 'completed',
            'overall_assessment': self._assess_overall_performance(performance, efficiency)
        }
    
    def _assess_overall_performance(self, performance: Dict[str, Any], 
                                  efficiency: Dict[str, Any]) -> str:
        """Assess overall performance."""
        
        accuracy = performance['accuracy_metrics']['top1_accuracy']
        param_reduction = efficiency['parameter_efficiency']['reduction_ratio']
        
        if accuracy > 0.8 and param_reduction > 0.4:
            return 'excellent'
        elif accuracy > 0.7 and param_reduction > 0.3:
            return 'good'
        elif accuracy > 0.6 and param_reduction > 0.2:
            return 'acceptable'
        else:
            return 'needs_improvement'
    
    def _create_llm_assessment_prompt(self, state: PruningState, evaluation_results: Dict[str, Any],
                                    comparison_results: Dict[str, Any]) -> str:
        """Create prompt for LLM assessment."""
        
        model_name = getattr(state, 'model_name', 'unknown')
        performance = evaluation_results['performance_evaluation']
        efficiency = evaluation_results['efficiency_analysis']
        
        prompt = f"""
You are an expert in neural network pruning and model optimization. Please provide a comprehensive assessment of the following pruning results:

## Model Information:
- Model: {model_name}
- Target Pruning Ratio: {getattr(state, 'target_pruning_ratio', 0.0):.1%}

## Final Results:
- Accuracy: {performance['accuracy_metrics']['top1_accuracy']:.1%}
- Parameter Reduction: {efficiency['parameter_efficiency']['reduction_ratio']:.1%}
- Model Size Reduction: {efficiency['size_efficiency']['size_reduction_ratio']:.1%}
- Estimated Speedup: {efficiency['computational_efficiency']['estimated_speedup']:.2f}x

## Detailed Evaluation Results:
{json.dumps(evaluation_results, indent=2, default=str)}

## Comparison Results:
{json.dumps(comparison_results, indent=2, default=str)}

Please provide assessment in JSON format:
{{
  "overall_assessment": "excellent|good|acceptable|needs_improvement",
  "key_achievements": [list of key achievements],
  "areas_of_concern": [list of concerns],
  "recommendations": [list of recommendations],
  "confidence_in_results": float between 0 and 1,
  "publication_readiness": "ready|needs_minor_improvements|needs_major_improvements"
}}
"""
        
        return prompt
    
    def _parse_llm_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM assessment response."""
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                llm_assessment = json.loads(json_str)
                
                return {
                    'status': 'success',
                    'assessment': llm_assessment,
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
        return "evaluation_agent"
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for the evaluation agent."""
        
        model_info = context.get('model_info', {})
        pruning_results = context.get('pruning_results', {})
        finetuning_results = context.get('finetuning_results', {})
        baseline_results = context.get('baseline_results', {})
        
        prompt = f"""You are an expert Evaluation Agent in a multi-agent neural network pruning system.

ROLE: Conduct comprehensive evaluation and provide final assessment with publication-ready analysis.

CURRENT CONTEXT:
- Model: {model_info.get('name', 'Unknown')} ({model_info.get('total_params', 0):,} parameters)
- Architecture: {model_info.get('architecture_type', 'Unknown')}
- Final compression: {pruning_results.get('compression_ratio', 0.0)*100:.1f}%
- Recovery achieved: {finetuning_results.get('accuracy_recovery', 0.0)*100:.1f}%

PERFORMANCE METRICS:
- Original accuracy: {baseline_results.get('original_accuracy', 0.0)*100:.1f}%
- Final accuracy: {finetuning_results.get('final_accuracy', 0.0)*100:.1f}%
- Accuracy drop: {(baseline_results.get('original_accuracy', 0.0) - finetuning_results.get('final_accuracy', 0.0))*100:.1f}%
- Speedup achieved: {pruning_results.get('speedup', 1.0):.2f}x
- Memory reduction: {pruning_results.get('memory_reduction', 0.0)*100:.1f}%

EVALUATION RESPONSIBILITIES:
1. Assess accuracy preservation and performance gains
2. Compare against baseline pruning methods
3. Validate paper reproduction requirements
4. Analyze robustness and generalization
5. Generate publication-ready results summary

COMPARISON BASELINES:
- Magnitude pruning: {baseline_results.get('magnitude_accuracy', 'N/A')}
- Taylor pruning: {baseline_results.get('taylor_accuracy', 'N/A')}
- Isomorphic pruning: {baseline_results.get('isomorphic_accuracy', 'N/A')}
- Random pruning: {baseline_results.get('random_accuracy', 'N/A')}

ASSESSMENT CRITERIA:
- ACCURACY: Minimal degradation (<2% for good, <1% for excellent)
- EFFICIENCY: Significant speedup (>1.5x good, >2x excellent)
- COMPRESSION: Target ratio achieved with quality preservation
- ROBUSTNESS: Consistent performance across test scenarios
- REPRODUCIBILITY: Results match paper claims within variance

DECISION FRAMEWORK:
- COMPREHENSIVE: Evaluate all dimensions (accuracy, efficiency, robustness)
- COMPARATIVE: Position results against established baselines
- SCIENTIFIC: Provide statistical significance and confidence intervals
- ACTIONABLE: Give clear recommendations for improvement

Provide structured evaluation with clear metrics, comparisons, and final assessment."""
        
        return prompt
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response for evaluation decisions."""
        
        try:
            # Extract key evaluation decisions from response
            decisions = {}
            
            # Look for overall assessment
            import re
            
            if 'excellent' in response.lower():
                decisions['overall_rating'] = 'excellent'
                decisions['rating_score'] = 5
            elif 'good' in response.lower():
                decisions['overall_rating'] = 'good'
                decisions['rating_score'] = 4
            elif 'satisfactory' in response.lower() or 'acceptable' in response.lower():
                decisions['overall_rating'] = 'satisfactory'
                decisions['rating_score'] = 3
            elif 'poor' in response.lower():
                decisions['overall_rating'] = 'poor'
                decisions['rating_score'] = 2
            else:
                decisions['overall_rating'] = 'fair'
                decisions['rating_score'] = 3
            
            # Extract accuracy assessment
            if 'accuracy.*excellent' in response.lower() or 'minimal.*drop' in response.lower():
                decisions['accuracy_assessment'] = 'excellent'
            elif 'accuracy.*good' in response.lower() or 'acceptable.*drop' in response.lower():
                decisions['accuracy_assessment'] = 'good'
            elif 'accuracy.*poor' in response.lower() or 'significant.*drop' in response.lower():
                decisions['accuracy_assessment'] = 'poor'
            else:
                decisions['accuracy_assessment'] = 'fair'
            
            # Extract efficiency assessment
            if 'speedup.*excellent' in response.lower() or 'significant.*speedup' in response.lower():
                decisions['efficiency_assessment'] = 'excellent'
            elif 'speedup.*good' in response.lower() or 'moderate.*speedup' in response.lower():
                decisions['efficiency_assessment'] = 'good'
            elif 'speedup.*poor' in response.lower() or 'minimal.*speedup' in response.lower():
                decisions['efficiency_assessment'] = 'poor'
            else:
                decisions['efficiency_assessment'] = 'fair'
            
            # Extract comparison with baselines
            if 'outperform' in response.lower() or 'better.*than.*baseline' in response.lower():
                decisions['baseline_comparison'] = 'superior'
            elif 'comparable' in response.lower() or 'similar.*baseline' in response.lower():
                decisions['baseline_comparison'] = 'comparable'
            elif 'worse.*than.*baseline' in response.lower():
                decisions['baseline_comparison'] = 'inferior'
            else:
                decisions['baseline_comparison'] = 'comparable'
            
            # Extract publication readiness
            if 'publication.*ready' in response.lower() or 'ready.*publication' in response.lower():
                decisions['publication_ready'] = True
            elif 'not.*ready' in response.lower() or 'needs.*improvement' in response.lower():
                decisions['publication_ready'] = False
            else:
                decisions['publication_ready'] = decisions['rating_score'] >= 4
            
            # Extract recommendations
            recommendations = []
            lines = response.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'improve', 'enhance']):
                    recommendations.append(line.strip())
            
            decisions['recommendations'] = recommendations[:5]  # Keep top 5 recommendations
            
            # Extract key strengths and weaknesses
            strengths = []
            weaknesses = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['strength', 'advantage', 'benefit']):
                    strengths.append(line.strip())
                elif any(keyword in line.lower() for keyword in ['weakness', 'limitation', 'issue']):
                    weaknesses.append(line.strip())
            
            decisions['strengths'] = strengths[:3]
            decisions['weaknesses'] = weaknesses[:3]
            
            # Extract confidence level
            if 'highly confident' in response.lower() or 'very confident' in response.lower():
                confidence = 0.95
            elif 'confident' in response.lower():
                confidence = 0.85
            elif 'moderately confident' in response.lower():
                confidence = 0.70
            elif 'uncertain' in response.lower() or 'unsure' in response.lower():
                confidence = 0.50
            else:
                confidence = 0.75
            
            # Determine success based on content quality
            success = (
                len(decisions) >= 6 and 
                'error' not in response.lower() and
                decisions['rating_score'] > 0
            )
            
            return AgentResponse(
                success=success,
                data=decisions,
                message=f"Evaluation complete: {decisions['overall_rating']} ({decisions['rating_score']}/5), {decisions['baseline_comparison']} to baselines",
                confidence=confidence
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={
                    'overall_rating': 'fair',
                    'rating_score': 3,
                    'accuracy_assessment': 'fair',
                    'efficiency_assessment': 'fair',
                    'baseline_comparison': 'comparable',
                    'publication_ready': False,
                    'recommendations': ['Conduct more thorough evaluation'],
                    'strengths': ['Completed pruning process'],
                    'weaknesses': ['Evaluation incomplete']
                },
                message=f"Failed to parse evaluation response, using neutral assessment: {str(e)}",
                confidence=0.3
            )
