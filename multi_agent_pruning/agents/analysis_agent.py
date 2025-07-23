#!/usr/bin/env python3
"""
Analysis Agent for Multi-Agent LLM Pruning Framework

This agent analyzes the profiling results and provides strategic recommendations
for pruning configuration, importance criteria, and group ratio multipliers.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime

from .base_agent import BaseAgent
from ..core.state_manager import PruningState
from ..core.dependency_analyzer import DependencyAnalyzer
from ..core.isomorphic_analyzer import IsomorphicAnalyzer
from ..utils.profiler import TimingProfiler

logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """
    Analysis Agent that processes profiling results and generates strategic
    pruning recommendations based on model architecture and constraints.
    """
    
    def __init__(self, llm_client=None, profiler: Optional[TimingProfiler] = None):
        super().__init__("AnalysisAgent", llm_client, profiler)
        
        # Analysis components
        self.dependency_analyzer: Optional[DependencyAnalyzer] = None
        self.isomorphic_analyzer: Optional[IsomorphicAnalyzer] = None
        
        # Analysis results
        self.analysis_results = {}
        self.recommendations = {}
        
        logger.info("🔍 Analysis Agent initialized")
    
    def execute(self, state: PruningState) -> Dict[str, Any]:
        """
        Execute analysis phase: analyze profiling results and generate recommendations.
        
        Args:
            state: Current pruning state with profiling results
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        
        with self.profiler.timer("analysis_agent_execution"):
            logger.info("🔍 Starting Analysis Agent execution")
            
            try:
                # Validate input state
                if not self._validate_input_state(state):
                    return self._create_error_result("Invalid input state for analysis")
                
                # Initialize analyzers
                self._initialize_analyzers(state)
                
                # Perform comprehensive analysis
                analysis_results = self._perform_comprehensive_analysis(state)
                
                # Generate strategic recommendations
                recommendations = self._generate_strategic_recommendations(state, analysis_results)
                
                # Create LLM prompt for advanced analysis
                llm_analysis = self._get_llm_analysis(state, analysis_results, recommendations)
                
                # Combine results
                final_results = {
                    'success': True,
                    'agent_name': self.agent_name,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_results': analysis_results,
                    'strategic_recommendations': recommendations,
                    'llm_analysis': llm_analysis,
                    'next_agent': 'PruningAgent'
                }
                
                # Store results for future reference
                self.analysis_results = analysis_results
                self.recommendations = recommendations
                
                logger.info("✅ Analysis Agent execution completed successfully")
                return final_results
                
            except Exception as e:
                logger.error(f"❌ Analysis Agent execution failed: {str(e)}")
                return self._create_error_result(f"Analysis execution failed: {str(e)}")
    
    def _validate_input_state(self, state: PruningState) -> bool:
        """Validate that the input state contains required profiling results."""
        
        required_fields = ['model', 'profiling_results', 'master_results']
        
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                logger.error(f"❌ Missing required field in state: {field}")
                return False
        
        # Check profiling results structure
        profiling_results = state.profiling_results
        if not isinstance(profiling_results, dict):
            logger.error("❌ Profiling results must be a dictionary")
            return False
        
        required_profiling_fields = ['model_analysis', 'layer_analysis', 'dependency_graph']
        for field in required_profiling_fields:
            if field not in profiling_results:
                logger.error(f"❌ Missing required profiling field: {field}")
                return False
        
        logger.info("✅ Input state validation passed")
        return True
    
    def _initialize_analyzers(self, state: PruningState):
        """Initialize dependency and isomorphic analyzers."""
        
        with self.profiler.timer("analyzer_initialization"):
            model = state.model
            model_name = getattr(state, 'model_name', 'unknown_model')
            
            # Initialize dependency analyzer
            self.dependency_analyzer = DependencyAnalyzer(model, model_name)
            
            # Initialize isomorphic analyzer
            self.isomorphic_analyzer = IsomorphicAnalyzer(model, model_name)
            
            logger.info("🔧 Analyzers initialized successfully")
    
    def _perform_comprehensive_analysis(self, state: PruningState) -> Dict[str, Any]:
        """Perform comprehensive analysis of the model and profiling results."""
        
        with self.profiler.timer("comprehensive_analysis"):
            logger.info("📊 Performing comprehensive analysis")
            
            analysis = {
                'architecture_analysis': self._analyze_architecture(state),
                'dependency_analysis': self._analyze_dependencies(state),
                'constraint_analysis': self._analyze_constraints(state),
                'performance_analysis': self._analyze_performance(state),
                'safety_analysis': self._analyze_safety_constraints(state)
            }
            
            logger.info("✅ Comprehensive analysis completed")
            return analysis
    
    def _analyze_architecture(self, state: PruningState) -> Dict[str, Any]:
        """Analyze model architecture and characteristics."""
        
        profiling_results = state.profiling_results
        model_analysis = profiling_results['model_analysis']
        
        # Extract architecture information
        architecture_info = {
            'model_type': model_analysis.get('architecture_type', 'unknown'),
            'total_parameters': model_analysis.get('total_parameters', 0),
            'total_layers': model_analysis.get('total_layers', 0),
            'model_size_mb': model_analysis.get('model_size_mb', 0),
            'estimated_macs': model_analysis.get('estimated_macs', 0)
        }
        
        # Analyze layer distribution
        layer_analysis = profiling_results['layer_analysis']
        layer_distribution = {}
        
        for layer_name, layer_info in layer_analysis.items():
            layer_type = layer_info.get('layer_type', 'unknown')
            if layer_type not in layer_distribution:
                layer_distribution[layer_type] = {
                    'count': 0,
                    'total_params': 0,
                    'layers': []
                }
            
            layer_distribution[layer_type]['count'] += 1
            layer_distribution[layer_type]['total_params'] += layer_info.get('parameters', 0)
            layer_distribution[layer_type]['layers'].append(layer_name)
        
        # Identify critical layers
        critical_layers = []
        for layer_name, layer_info in layer_analysis.items():
            param_ratio = layer_info.get('parameters', 0) / architecture_info['total_parameters']
            if param_ratio > 0.05:  # Layers with >5% of total parameters
                critical_layers.append({
                    'name': layer_name,
                    'parameters': layer_info.get('parameters', 0),
                    'param_ratio': param_ratio,
                    'layer_type': layer_info.get('layer_type', 'unknown')
                })
        
        return {
            'architecture_info': architecture_info,
            'layer_distribution': layer_distribution,
            'critical_layers': critical_layers,
            'pruning_potential': self._assess_pruning_potential(architecture_info, layer_distribution)
        }
    
    def _analyze_dependencies(self, state: PruningState) -> Dict[str, Any]:
        """Analyze layer dependencies and coupling constraints."""
        
        if not self.dependency_analyzer:
            return {'error': 'Dependency analyzer not initialized'}
        
        # Get dependency information
        dependency_graph = self.dependency_analyzer.get_dependency_graph()
        coupling_constraints = self.dependency_analyzer.get_coupling_constraints()
        safe_pruning_groups = self.dependency_analyzer.get_safe_pruning_groups()
        critical_layers = self.dependency_analyzer.get_critical_layers()
        
        # Analyze dependency complexity
        dependency_complexity = {
            'total_dependencies': len([dep for layer_info in dependency_graph.values() 
                                     for dep in layer_info['depends_on']]),
            'highly_connected_layers': [],
            'isolated_layers': []
        }
        
        for layer_name, layer_info in dependency_graph.items():
            total_connections = len(layer_info['depends_on']) + len(layer_info['dependents'])
            
            if total_connections > 3:
                dependency_complexity['highly_connected_layers'].append({
                    'name': layer_name,
                    'connections': total_connections
                })
            elif total_connections == 0:
                dependency_complexity['isolated_layers'].append(layer_name)
        
        return {
            'dependency_graph_summary': {
                'total_layers': len(dependency_graph),
                'total_constraints': len(coupling_constraints),
                'safe_groups': len(safe_pruning_groups),
                'critical_layers': len(critical_layers)
            },
            'coupling_constraints': [
                {
                    'type': constraint.constraint_type,
                    'description': constraint.description,
                    'layers': constraint.layer_group,
                    'dimensions_affected': constraint.dimensions_affected
                }
                for constraint in coupling_constraints
            ],
            'safe_pruning_groups': safe_pruning_groups,
            'critical_layers': critical_layers,
            'dependency_complexity': dependency_complexity
        }
    
    def _analyze_constraints(self, state: PruningState) -> Dict[str, Any]:
        """Analyze pruning constraints and limitations."""
        
        # Get master agent recommendations
        master_results = state.master_results
        recommended_strategy = master_results.get('recommended_strategy', {})
        
        # Extract constraint information
        constraints = {
            'target_pruning_ratio': state.target_pruning_ratio,
            'dataset': getattr(state, 'dataset', 'unknown'),
            'safety_limits': master_results.get('safety_limits', {}),
            'architectural_constraints': []
        }
        
        # Analyze architectural constraints
        if hasattr(state, 'model_name'):
            model_name = state.model_name.lower()
            
            if 'vit' in model_name or 'deit' in model_name:
                constraints['architectural_constraints'].extend([
                    'Attention head pruning requires careful dimension matching',
                    'MLP layers have fc1-fc2 coupling constraints',
                    'Embedding dimensions should be preserved',
                    'Classification head should not be pruned'
                ])
            elif 'resnet' in model_name:
                constraints['architectural_constraints'].extend([
                    'Residual connections require dimension preservation',
                    'Batch normalization layers are sensitive to pruning',
                    'Shortcut connections limit pruning flexibility'
                ])
        
        # Analyze safety constraints
        safety_analysis = {
            'minimum_accuracy_threshold': 0.1,  # 10% minimum accuracy
            'maximum_single_layer_pruning': 0.8,  # 80% max for any single layer
            'critical_layer_protection': True,
            'gradient_flow_preservation': True
        }
        
        return {
            'pruning_constraints': constraints,
            'safety_analysis': safety_analysis,
            'constraint_violations_risk': self._assess_constraint_risk(constraints, state)
        }
    
    def _analyze_performance(self, state: PruningState) -> Dict[str, Any]:
        """Analyze expected performance impact of pruning."""
        
        profiling_results = state.profiling_results
        model_analysis = profiling_results['model_analysis']
        
        # Estimate performance gains
        target_ratio = state.target_pruning_ratio
        
        # Conservative estimates based on typical pruning results
        estimated_speedup = 1.0 + (target_ratio * 0.5)  # 50% of pruning ratio as speedup
        estimated_memory_reduction = target_ratio * 0.8  # 80% of pruning ratio as memory reduction
        estimated_accuracy_drop = target_ratio * 0.02   # 2% accuracy drop per 100% pruning (conservative)
        
        performance_analysis = {
            'baseline_metrics': {
                'parameters': model_analysis.get('total_parameters', 0),
                'model_size_mb': model_analysis.get('model_size_mb', 0),
                'estimated_macs': model_analysis.get('estimated_macs', 0)
            },
            'projected_metrics': {
                'parameters_after_pruning': int(model_analysis.get('total_parameters', 0) * (1 - target_ratio)),
                'model_size_mb_after_pruning': model_analysis.get('model_size_mb', 0) * (1 - target_ratio),
                'estimated_macs_after_pruning': int(model_analysis.get('estimated_macs', 0) * (1 - target_ratio))
            },
            'expected_improvements': {
                'inference_speedup': estimated_speedup,
                'memory_reduction_ratio': estimated_memory_reduction,
                'storage_reduction_ratio': target_ratio
            },
            'expected_degradation': {
                'accuracy_drop_estimate': estimated_accuracy_drop,
                'training_time_increase': 0.1,  # 10% increase due to fine-tuning
                'convergence_difficulty': target_ratio * 0.3  # Higher ratios are harder to recover
            }
        }
        
        return performance_analysis
    
    def _analyze_safety_constraints(self, state: PruningState) -> Dict[str, Any]:
        """Analyze safety constraints and failure prevention measures."""
        
        # Get master agent safety recommendations
        master_results = state.master_results
        safety_limits = master_results.get('safety_limits', {})
        
        # Define safety thresholds based on dataset and model type
        dataset = getattr(state, 'dataset', 'unknown').lower()
        
        if 'imagenet' in dataset:
            safety_thresholds = {
                'catastrophic_accuracy_threshold': 0.30,  # 30% accuracy is catastrophic for ImageNet
                'warning_accuracy_threshold': 0.50,       # 50% accuracy is concerning
                'maximum_mlp_pruning': 0.15,              # 15% max MLP pruning for safety
                'maximum_attention_pruning': 0.10         # 10% max attention pruning for safety
            }
        else:
            safety_thresholds = {
                'catastrophic_accuracy_threshold': 0.20,  # 20% accuracy is catastrophic for CIFAR
                'warning_accuracy_threshold': 0.40,       # 40% accuracy is concerning
                'maximum_mlp_pruning': 0.25,              # 25% max MLP pruning
                'maximum_attention_pruning': 0.15         # 15% max attention pruning
            }
        
        # Analyze current configuration against safety thresholds
        recommended_strategy = master_results.get('recommended_strategy', {})
        current_config_safety = {
            'is_safe': True,
            'violations': [],
            'warnings': []
        }
        
        # Check for safety violations
        if 'mlp_multiplier' in recommended_strategy:
            mlp_ratio = recommended_strategy['mlp_multiplier'] * state.target_pruning_ratio
            if mlp_ratio > safety_thresholds['maximum_mlp_pruning']:
                current_config_safety['violations'].append(
                    f"MLP pruning ratio {mlp_ratio:.1%} exceeds safety limit {safety_thresholds['maximum_mlp_pruning']:.1%}"
                )
                current_config_safety['is_safe'] = False
        
        if 'qkv_multiplier' in recommended_strategy:
            attn_ratio = recommended_strategy['qkv_multiplier'] * state.target_pruning_ratio
            if attn_ratio > safety_thresholds['maximum_attention_pruning']:
                current_config_safety['violations'].append(
                    f"Attention pruning ratio {attn_ratio:.1%} exceeds safety limit {safety_thresholds['maximum_attention_pruning']:.1%}"
                )
                current_config_safety['is_safe'] = False
        
        return {
            'safety_thresholds': safety_thresholds,
            'current_config_safety': current_config_safety,
            'recovery_strategies': [
                'Checkpoint creation before pruning',
                'Gradual pruning with validation',
                'Automatic rollback on catastrophic failure',
                'Conservative fine-tuning schedule'
            ],
            'monitoring_requirements': [
                'Accuracy tracking after each pruning step',
                'Loss monitoring during fine-tuning',
                'Gradient flow analysis',
                'Memory usage monitoring'
            ]
        }
    
    def _generate_strategic_recommendations(self, state: PruningState, 
                                          analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic recommendations based on analysis results."""
        
        with self.profiler.timer("strategic_recommendations"):
            logger.info("🎯 Generating strategic recommendations")
            
            # Get master agent recommendations as baseline
            master_results = state.master_results
            recommended_strategy = master_results.get('recommended_strategy', {})
            
            # Analyze architecture-specific recommendations
            arch_analysis = analysis_results['architecture_analysis']
            architecture_type = arch_analysis['architecture_info']['model_type']
            
            # Generate importance criterion recommendation
            importance_recommendation = self._recommend_importance_criterion(
                state, analysis_results
            )
            
            # Generate group ratio recommendations
            group_ratio_recommendation = self._recommend_group_ratios(
                state, analysis_results, recommended_strategy
            )
            
            # Generate pruning strategy recommendations
            strategy_recommendation = self._recommend_pruning_strategy(
                state, analysis_results
            )
            
            # Generate safety recommendations
            safety_recommendation = self._recommend_safety_measures(
                state, analysis_results
            )
            
            recommendations = {
                'importance_criterion': importance_recommendation,
                'group_ratios': group_ratio_recommendation,
                'pruning_strategy': strategy_recommendation,
                'safety_measures': safety_recommendation,
                'execution_plan': self._create_execution_plan(state, analysis_results),
                'confidence_score': self._calculate_confidence_score(analysis_results)
            }
            
            logger.info("✅ Strategic recommendations generated")
            return recommendations
    
    def _recommend_importance_criterion(self, state: PruningState, 
                                      analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend the best importance criterion based on analysis."""
        
        arch_analysis = analysis_results['architecture_analysis']
        architecture_type = arch_analysis['architecture_info']['model_type']
        
        # Architecture-specific recommendations
        if 'vit' in architecture_type.lower() or 'transformer' in architecture_type.lower():
            primary_criterion = 'taylor'
            fallback_criterion = 'magnitude_l2'
            rationale = "Taylor criterion works well for transformer architectures with gradient information"
        elif 'resnet' in architecture_type.lower() or 'conv' in architecture_type.lower():
            primary_criterion = 'magnitude_l1'
            fallback_criterion = 'taylor'
            rationale = "L1 magnitude is effective for convolutional architectures"
        else:
            primary_criterion = 'magnitude_l2'
            fallback_criterion = 'taylor'
            rationale = "L2 magnitude as safe default for unknown architectures"
        
        return {
            'primary_criterion': primary_criterion,
            'fallback_criterion': fallback_criterion,
            'rationale': rationale,
            'data_requirements': 'gradient' if primary_criterion == 'taylor' else 'none'
        }
    
    def _recommend_group_ratios(self, state: PruningState, analysis_results: Dict[str, Any],
                              baseline_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend group ratio multipliers based on analysis."""
        
        # Start with master agent recommendations
        recommended_ratios = baseline_strategy.copy()
        
        # Adjust based on safety analysis
        safety_analysis = analysis_results['safety_analysis']
        safety_thresholds = safety_analysis['safety_thresholds']
        
        # Apply safety constraints
        target_ratio = state.target_pruning_ratio
        
        if 'mlp_multiplier' in recommended_ratios:
            max_safe_mlp = safety_thresholds['maximum_mlp_pruning'] / target_ratio
            recommended_ratios['mlp_multiplier'] = min(
                recommended_ratios['mlp_multiplier'], 
                max_safe_mlp
            )
        
        if 'qkv_multiplier' in recommended_ratios:
            max_safe_attn = safety_thresholds['maximum_attention_pruning'] / target_ratio
            recommended_ratios['qkv_multiplier'] = min(
                recommended_ratios['qkv_multiplier'], 
                max_safe_attn
            )
        
        # Ensure projection and head layers are protected
        recommended_ratios['proj_multiplier'] = 0.0
        recommended_ratios['head_multiplier'] = 0.0
        
        return {
            'recommended_ratios': recommended_ratios,
            'safety_adjustments_applied': True,
            'rationale': 'Ratios adjusted to comply with safety thresholds while maintaining effectiveness'
        }
    
    def _recommend_pruning_strategy(self, state: PruningState, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend overall pruning strategy."""
        
        dependency_analysis = analysis_results['dependency_analysis']
        performance_analysis = analysis_results['performance_analysis']
        
        # Determine pruning approach
        if len(dependency_analysis['coupling_constraints']) > 5:
            approach = 'conservative_coupled'
            rationale = 'High coupling complexity requires conservative approach'
        elif performance_analysis['expected_degradation']['accuracy_drop_estimate'] > 0.05:
            approach = 'gradual_progressive'
            rationale = 'High accuracy risk requires gradual pruning'
        else:
            approach = 'standard_structured'
            rationale = 'Standard structured pruning is suitable'
        
        return {
            'approach': approach,
            'rationale': rationale,
            'execution_phases': self._define_execution_phases(approach),
            'validation_checkpoints': True,
            'rollback_strategy': 'checkpoint_restore'
        }
    
    def _recommend_safety_measures(self, state: PruningState, 
                                 analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend safety measures and monitoring."""
        
        return {
            'checkpoint_frequency': 'before_each_major_step',
            'validation_frequency': 'after_each_pruning_iteration',
            'early_stopping_criteria': {
                'accuracy_drop_threshold': 0.05,
                'loss_increase_threshold': 0.2,
                'gradient_norm_threshold': 10.0
            },
            'recovery_actions': [
                'automatic_checkpoint_restore',
                'reduce_pruning_ratio',
                'change_importance_criterion',
                'emergency_stop'
            ],
            'monitoring_metrics': [
                'accuracy',
                'loss',
                'gradient_norms',
                'parameter_statistics',
                'memory_usage'
            ]
        }
    
    def _create_execution_plan(self, state: PruningState, 
                             analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan for pruning."""
        
        return {
            'phases': [
                {
                    'name': 'preparation',
                    'description': 'Initialize analyzers and create checkpoints',
                    'estimated_time_minutes': 2
                },
                {
                    'name': 'importance_computation',
                    'description': 'Compute importance scores for all layers',
                    'estimated_time_minutes': 5
                },
                {
                    'name': 'pruning_execution',
                    'description': 'Apply structured pruning with safety checks',
                    'estimated_time_minutes': 3
                },
                {
                    'name': 'validation',
                    'description': 'Validate pruned model and check constraints',
                    'estimated_time_minutes': 2
                }
            ],
            'total_estimated_time_minutes': 12,
            'parallel_execution_opportunities': [
                'importance_computation_can_be_parallelized',
                'validation_can_overlap_with_next_phase_prep'
            ]
        }
    
    def _calculate_confidence_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis and recommendations."""
        
        confidence_factors = []
        
        # Architecture analysis confidence
        arch_analysis = analysis_results['architecture_analysis']
        if arch_analysis['architecture_info']['model_type'] != 'unknown':
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Dependency analysis confidence
        dep_analysis = analysis_results['dependency_analysis']
        if dep_analysis['dependency_graph_summary']['total_constraints'] > 0:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.7)
        
        # Safety analysis confidence
        safety_analysis = analysis_results['safety_analysis']
        if safety_analysis['current_config_safety']['is_safe']:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Calculate weighted average
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return round(overall_confidence, 2)
    
    def _get_llm_analysis(self, state: PruningState, analysis_results: Dict[str, Any],
                         recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM-based analysis and validation of recommendations."""
        
        if not self.llm_client:
            return {'status': 'llm_not_available', 'message': 'LLM client not configured'}
        
        # Create comprehensive prompt for LLM analysis
        prompt = self._create_llm_analysis_prompt(state, analysis_results, recommendations)
        
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
    
    def _create_llm_analysis_prompt(self, state: PruningState, analysis_results: Dict[str, Any],
                                  recommendations: Dict[str, Any]) -> str:
        """Create comprehensive prompt for LLM analysis."""
        
        # Extract key information
        model_name = getattr(state, 'model_name', 'unknown')
        target_ratio = state.target_pruning_ratio
        dataset = getattr(state, 'dataset', 'unknown')
        
        arch_info = analysis_results['architecture_analysis']['architecture_info']
        safety_info = analysis_results['safety_analysis']
        
        prompt = f"""
You are an expert in neural network pruning and optimization. Analyze the following pruning scenario and provide strategic insights.

## Model Information:
- Model: {model_name}
- Architecture: {arch_info['model_type']}
- Total Parameters: {arch_info['total_parameters']:,}
- Model Size: {arch_info['model_size_mb']:.1f} MB
- Target Pruning Ratio: {target_ratio:.1%}
- Dataset: {dataset}

## Analysis Results:
{json.dumps(analysis_results, indent=2)}

## Current Recommendations:
{json.dumps(recommendations, indent=2)}

## Please provide analysis on:

1. **Strategy Validation**: Are the recommended group ratios and importance criterion appropriate for this model and target ratio?

2. **Risk Assessment**: What are the main risks with this pruning configuration? How can they be mitigated?

3. **Alternative Approaches**: Are there alternative strategies that might work better?

4. **Expected Outcomes**: What accuracy retention and performance gains can realistically be expected?

5. **Implementation Recommendations**: Any specific implementation details or precautions?

Please provide your analysis in JSON format with the following structure:
{{
  "strategy_validation": {{
    "is_appropriate": boolean,
    "concerns": [list of concerns],
    "suggestions": [list of suggestions]
  }},
  "risk_assessment": {{
    "high_risks": [list],
    "medium_risks": [list],
    "mitigation_strategies": [list]
  }},
  "alternative_approaches": [list of alternatives],
  "expected_outcomes": {{
    "accuracy_retention_estimate": float,
    "performance_gain_estimate": float,
    "confidence_level": "high|medium|low"
  }},
  "implementation_recommendations": [list of recommendations]
}}
"""
        
        return prompt
    
    def _parse_llm_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM analysis response."""
        
        try:
            # Try to extract JSON from response
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
    
    # Helper methods
    def _assess_pruning_potential(self, arch_info: Dict[str, Any], 
                                layer_distribution: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the pruning potential of the model."""
        
        total_params = arch_info['total_parameters']
        
        # Calculate prunable parameters
        prunable_params = 0
        for layer_type, info in layer_distribution.items():
            if layer_type in ['Linear', 'Conv2d', 'Conv1d']:
                prunable_params += info['total_params']
        
        prunable_ratio = prunable_params / total_params if total_params > 0 else 0
        
        return {
            'prunable_parameters': prunable_params,
            'prunable_ratio': prunable_ratio,
            'pruning_difficulty': 'low' if prunable_ratio > 0.8 else 'medium' if prunable_ratio > 0.5 else 'high',
            'recommended_max_ratio': min(0.5, prunable_ratio * 0.8)  # Conservative estimate
        }
    
    def _assess_constraint_risk(self, constraints: Dict[str, Any], state: PruningState) -> str:
        """Assess the risk level of constraint violations."""
        
        target_ratio = state.target_pruning_ratio
        safety_limits = constraints.get('safety_limits', {})
        
        # Check if target ratio is within safe limits
        if target_ratio > 0.5:
            return 'high'
        elif target_ratio > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _define_execution_phases(self, approach: str) -> List[Dict[str, Any]]:
        """Define execution phases based on pruning approach."""
        
        if approach == 'conservative_coupled':
            return [
                {'name': 'dependency_analysis', 'description': 'Analyze coupling constraints'},
                {'name': 'safe_group_identification', 'description': 'Identify safe pruning groups'},
                {'name': 'gradual_pruning', 'description': 'Apply pruning in small increments'},
                {'name': 'validation_and_recovery', 'description': 'Validate and recover if needed'}
            ]
        elif approach == 'gradual_progressive':
            return [
                {'name': 'baseline_checkpoint', 'description': 'Create baseline checkpoint'},
                {'name': 'progressive_pruning', 'description': 'Apply pruning progressively'},
                {'name': 'intermediate_validation', 'description': 'Validate at each step'},
                {'name': 'final_optimization', 'description': 'Final optimization and cleanup'}
            ]
        else:  # standard_structured
            return [
                {'name': 'importance_computation', 'description': 'Compute importance scores'},
                {'name': 'structured_pruning', 'description': 'Apply structured pruning'},
                {'name': 'constraint_validation', 'description': 'Validate constraints'},
                {'name': 'performance_verification', 'description': 'Verify performance'}
            ]
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        
        return {
            'success': False,
            'agent_name': self.agent_name,
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'next_agent': None
        }

