"""
Enhanced Master Agent for Multi-Agent LLM Pruning

Improved version of the user's Master Agent with:
- Better convergence detection
- Smarter parameter exploration
- Enhanced history analysis
- More efficient decision making
- Improved safety constraints
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from .base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)

@dataclass
class PruningStrategy:
    """Structured pruning strategy recommendation."""
    
    importance_criterion: str
    pruning_ratio: float
    round_to: Optional[int]
    global_pruning: bool
    
    # Strategy metadata
    rationale: str
    confidence: float
    expected_accuracy_drop: float
    risk_level: str
    
    # Control decisions
    continue_exploration: bool
    stop_reason: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'importance_criterion': self.importance_criterion,
            'pruning_ratio': self.pruning_ratio,
            'round_to': self.round_to,
            'global_pruning': self.global_pruning,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'expected_accuracy_drop': self.expected_accuracy_drop,
            'risk_level': self.risk_level,
            'continue': self.continue_exploration,
            'stop_reason': self.stop_reason
        }

class MasterAgent(BaseAgent):
    """
    Enhanced Master Agent with intelligent strategy optimization.
    
    Key improvements:
    1. Better history analysis and pattern recognition
    2. Smarter parameter exploration with adaptive step sizes
    3. Enhanced convergence detection
    4. Risk-aware decision making
    5. More efficient LLM usage with structured reasoning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None, profiler=None):
        """
        Initialize MasterAgent with proper BaseAgent inheritance.
        """
        # Call BaseAgent constructor with proper parameters
        super().__init__("MasterAgent", llm_client, profiler)
        
        # Store configuration
        self.config = config or {}
        
        # Initialize agent-specific components
        self._initialize_agent_components()
        
        logger.info("🧠 Master Agent initialized with proper inheritance")
    
    def _initialize_agent_components(self):
        """Initialize agent-specific components based on configuration."""
        
        # Strategy optimization settings
        strategy_config = self.config.get('strategy_optimization', {})
        self.max_iterations = strategy_config.get('max_iterations', 5)
        self.convergence_threshold = strategy_config.get('convergence_threshold', 0.005)  # 0.5% accuracy
        self.target_tolerance = strategy_config.get('target_tolerance', 0.01)  # 1% parameter reduction tolerance
        
        # Exploration strategy settings
        exploration_config = self.config.get('exploration', {})
        self.exploration_strategies = exploration_config.get('strategies', ['conservative', 'moderate', 'aggressive'])
        self.current_strategy = exploration_config.get('initial_strategy', 'conservative')
        
        # History analysis
        self.history_analyzer = HistoryAnalyzer()
        
        # Risk assessment settings
        risk_config = self.config.get('risk_assessment', {})
        self.risk_tolerance = risk_config.get('tolerance', 'medium')
        self.safety_margin = risk_config.get('safety_margin', 0.05)
        
        logger.info(f"🧠 Master Agent components initialized with {len(self.exploration_strategies)} strategies")

    def _prepare_context(self, input_data) -> Dict[str, Any]:
        """Prepare context from input data (handles both PruningState objects and dictionaries)."""
        
        # Handle PruningState object
        if hasattr(input_data, 'model_name'):
            # It's a PruningState object - extract relevant information
            context = {
                'model_name': input_data.model_name,
                'dataset': input_data.dataset,
                'target_ratio': input_data.target_ratio,
                'revision_number': input_data.revision_number,
                'max_revisions': input_data.max_revisions,
                'num_classes': input_data.num_classes,
                'input_size': input_data.input_size,
                'history': input_data.history,
                'attempted_ratios': input_data.attempted_pruning_ratios,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add model info if available
            if hasattr(input_data, 'model') and input_data.model is not None:
                context['model_info'] = {
                    'architecture_type': self._detect_architecture_type(input_data.model_name),
                    'has_model': True
                }
            else:
                context['model_info'] = {
                    'architecture_type': self._detect_architecture_type(input_data.model_name),
                    'has_model': False
                }
            
            # Add dataset info
            context['dataset_info'] = self._get_dataset_info(input_data.dataset)
            
            # Add agent results if available
            if hasattr(input_data, 'profile_results') and input_data.profile_results:
                context['profile_results'] = input_data.profile_results
            
            return context
        
        # Handle dictionary input (fallback)
        elif isinstance(input_data, dict):
            # Create a copy to avoid modifying original
            import copy
            context = copy.deepcopy(input_data)
            
            # Ensure required fields exist
            context.setdefault('revision_number', 0)
            context.setdefault('max_revisions', self.max_iterations)
            context.setdefault('history', [])
            context.setdefault('attempted_ratios', [])
            context.setdefault('timestamp', datetime.now().isoformat())
            
            # Add dataset info if not present
            if 'dataset_info' not in context:
                dataset_name = context.get('dataset', 'imagenet')
                context['dataset_info'] = self._get_dataset_info(dataset_name)
            
            return context
        
        else:
            # Unknown input type - create minimal context
            logger.warning(f"⚠️ Unknown input type: {type(input_data)}")
            return {
                'revision_number': 0,
                'max_revisions': self.max_iterations,
                'target_ratio': 0.5,
                'history': [],
                'attempted_ratios': [],
                'timestamp': datetime.now().isoformat(),
                'dataset_info': self._get_dataset_info('imagenet'),
                'model_info': {'architecture_type': 'unknown', 'has_model': False}
            }
    
    def _detect_architecture_type(self, model_name: str) -> str:
        """Detect architecture type from model name."""
        
        model_name_lower = model_name.lower()
        
        if any(x in model_name_lower for x in ['deit', 'vit', 'swin', 'beit']):
            return 'vision_transformer'
        elif any(x in model_name_lower for x in ['resnet', 'resnext', 'densenet']):
            return 'cnn'
        elif any(x in model_name_lower for x in ['convnext', 'efficientnet']):
            return 'modern_cnn'
        elif any(x in model_name_lower for x in ['mobilenet', 'shufflenet']):
            return 'mobile_cnn'
        else:
            return 'unknown'
    
    def _get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset-specific information and safety limits."""
        
        dataset_name_lower = dataset_name.lower()
        
        if dataset_name_lower == 'imagenet':
            return {
                'num_classes': 1000,
                'complexity': 'high',
                'pruning_difficulty': 'hard',
                'recommended_approach': 'conservative',
                'importance_criterion': 'taylor',
                'safety_limits': {
                    'max_mlp_pruning': 0.15,
                    'max_attention_pruning': 0.10,
                    'max_overall_pruning': 0.60,
                    'min_accuracy_threshold': 0.40
                }
            }
        elif dataset_name_lower == 'cifar10':
            return {
                'num_classes': 10,
                'complexity': 'medium',
                'pruning_difficulty': 'medium',
                'recommended_approach': 'moderate',
                'importance_criterion': 'l1norm',
                'safety_limits': {
                    'max_mlp_pruning': 0.30,
                    'max_attention_pruning': 0.25,
                    'max_overall_pruning': 0.80,
                    'min_accuracy_threshold': 0.50
                }
            }
        elif dataset_name_lower == 'cifar100':
            return {
                'num_classes': 100,
                'complexity': 'medium-high',
                'pruning_difficulty': 'medium-hard',
                'recommended_approach': 'conservative',
                'importance_criterion': 'taylor',
                'safety_limits': {
                    'max_mlp_pruning': 0.20,
                    'max_attention_pruning': 0.15,
                    'max_overall_pruning': 0.70,
                    'min_accuracy_threshold': 0.35
                }
            }
        else:
            # Default/unknown dataset
            return {
                'num_classes': 1000,
                'complexity': 'unknown',
                'pruning_difficulty': 'unknown',
                'recommended_approach': 'conservative',
                'importance_criterion': 'l1norm',
                'safety_limits': {
                    'max_mlp_pruning': 0.15,
                    'max_attention_pruning': 0.10,
                    'max_overall_pruning': 0.60,
                    'min_accuracy_threshold': 0.40
                }
            }

    def get_agent_role(self) -> str:
        """Return the role description for this agent."""
        return """Strategic coordinator for multi-agent pruning workflow, specializing in 
        parameter optimization, convergence detection, and risk-aware decision making."""
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for strategic decision making."""
        
        dataset_info = context.get('dataset_info', {})
        target_ratio = context.get('target_ratio', 0.5)
        revision_number = context.get('revision_number', 0)
        
        return f"""You are the Master Agent coordinating a multi-agent neural network pruning workflow.

MISSION: Find the optimal pruning configuration that achieves EXACTLY {target_ratio:.1%} parameter reduction while maximizing accuracy.

CURRENT CONTEXT:
- Iteration: {revision_number + 1}/{self.max_iterations}
- Dataset: {context.get('dataset', 'unknown')} ({dataset_info.get('num_classes', 'unknown')} classes)
- Architecture: {context.get('model_info', {}).get('architecture_type', 'unknown')}
- Safety Approach: {dataset_info.get('recommended_approach', 'conservative')}

SAFETY CONSTRAINTS (CRITICAL):
- Max MLP Pruning: {dataset_info.get('safety_limits', {}).get('max_mlp_pruning', 0.2):.1%}
- Max Attention Pruning: {dataset_info.get('safety_limits', {}).get('max_attention_pruning', 0.15):.1%}
- Min Accuracy Threshold: {dataset_info.get('safety_limits', {}).get('min_accuracy_threshold', 0.5):.1%}

DECISION FRAMEWORK:
1. EXACT TARGET ACHIEVEMENT: Primary goal is {target_ratio:.1%} parameter reduction (±1% tolerance)
2. ACCURACY MAXIMIZATION: Secondary goal is highest possible accuracy
3. CONVERGENCE DETECTION: Stop when no meaningful progress is being made
4. SAFETY ENFORCEMENT: Never exceed architecture-specific safety limits

Your response must be a valid JSON object with strategic recommendations."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategic decision making with enhanced analysis."""
        
        try:
            # Analyze current context
            context = self._prepare_context(input_data)
            
            # Analyze pruning history for patterns
            history_analysis = self.history_analyzer.analyze_history(
                context.get('history', []),
                context.get('attempted_ratios', []),
                context.get('target_ratio', 0.5)
            )
            
            # Determine exploration strategy
            exploration_strategy = self._determine_exploration_strategy(
                context, history_analysis
            )
            
            # Check for early stopping conditions
            should_stop, stop_reason = self._check_stopping_conditions(
                context, history_analysis
            )
            
            if should_stop:
                return self._create_stop_response(stop_reason, context, history_analysis)
            
            # Generate next strategy
            strategy = self._generate_next_strategy(
                context, history_analysis, exploration_strategy
            )
            
            # Validate strategy against safety constraints
            strategy = self._apply_safety_validation(strategy, context)
            
            # Create response
            return self._create_strategy_response(strategy, context, history_analysis)
            
        except Exception as e:
            logger.error(f"❌ Master Agent execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.agent_name
            }
    
    def _determine_exploration_strategy(self, context: Dict[str, Any], 
                                      history_analysis: Dict[str, Any]) -> str:
        """Determine the exploration strategy based on current state."""
        
        revision_number = context.get('revision_number', 0)
        progress_rate = history_analysis.get('progress_rate', 0)
        target_achievement = history_analysis.get('target_achievement_rate', 0)
        
        # Early iterations: conservative
        if revision_number < 2:
            return 'conservative'
        
        # Good progress: continue current strategy
        if progress_rate > 0.01:  # 1% improvement per iteration
            return self.current_strategy
        
        # Poor target achievement: more aggressive
        if target_achievement < 0.5:
            return 'aggressive'
        
        # Default: moderate
        return 'moderate'
    
    def _check_stopping_conditions(self, context: Dict[str, Any], 
                                 history_analysis: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if we should stop the exploration process."""
        
        revision_number = context.get('revision_number', 0)
        target_ratio = context.get('target_ratio', 0.5)
        
        # Maximum iterations reached
        if revision_number >= self.max_iterations:
            return True, f"Maximum iterations ({self.max_iterations}) reached"
        
        # Target achieved with good accuracy
        best_result = history_analysis.get('best_result')
        if best_result:
            param_error = abs(best_result.get('achieved_ratio', 0) - target_ratio)
            accuracy = best_result.get('accuracy', 0)
            
            if param_error < self.target_tolerance and accuracy > 0.6:  # 60% minimum
                return True, f"Target achieved: {param_error:.1%} error, {accuracy:.1%} accuracy"
        
        # Convergence detected
        if history_analysis.get('converged', False):
            return True, f"Convergence detected: {history_analysis.get('convergence_reason', 'unknown')}"
        
        # Cycling behavior
        if history_analysis.get('cycling', False):
            return True, "Cycling behavior detected - no further progress expected"
        
        # Catastrophic failure
        if history_analysis.get('catastrophic_failure', False):
            return True, "Catastrophic accuracy loss detected"
        
        return False, None
    
    def _generate_next_strategy(self, context: Dict[str, Any], 
                              history_analysis: Dict[str, Any],
                              exploration_strategy: str) -> PruningStrategy:
        """Generate the next pruning strategy based on analysis."""
        
        target_ratio = context.get('target_ratio', 0.5)
        dataset_info = context.get('dataset_info', {})
        
        # Get parameter recommendations from history analysis
        param_recommendations = history_analysis.get('parameter_recommendations', {})
        
        # Determine pruning ratio
        if 'optimal_ratio' in param_recommendations:
            pruning_ratio = param_recommendations['optimal_ratio']
        else:
            pruning_ratio = self._estimate_pruning_ratio(target_ratio, history_analysis, exploration_strategy)
        
        # Determine importance criterion
        importance_criterion = param_recommendations.get(
            'best_importance_criterion',
            dataset_info.get('importance_criterion', 'taylor')
        )
        
        # Determine round_to parameter
        round_to = param_recommendations.get('optimal_round_to', 2)
        if exploration_strategy == 'aggressive':
            round_to = 1  # More fine-grained
        elif exploration_strategy == 'conservative':
            round_to = 4  # More hardware-friendly
        
        # Calculate confidence and risk
        confidence = self._calculate_confidence(history_analysis, exploration_strategy)
        risk_level = self._assess_risk_level(pruning_ratio, target_ratio, exploration_strategy)
        expected_accuracy_drop = self._estimate_accuracy_drop(pruning_ratio, history_analysis)
        
        # Generate rationale
        rationale = self._generate_rationale(
            pruning_ratio, importance_criterion, round_to, 
            exploration_strategy, history_analysis
        )
        
        return PruningStrategy(
            importance_criterion=importance_criterion,
            pruning_ratio=pruning_ratio,
            round_to=round_to,
            global_pruning=True,  # Always use global pruning
            rationale=rationale,
            confidence=confidence,
            expected_accuracy_drop=expected_accuracy_drop,
            risk_level=risk_level,
            continue_exploration=True,
            stop_reason=None
        )
    
    def _estimate_pruning_ratio(self, target_ratio: float, 
                              history_analysis: Dict[str, Any],
                              exploration_strategy: str) -> float:
        """Estimate the pruning ratio needed to achieve target parameter reduction."""
        
        # Use history to estimate scaling factor
        scaling_factor = history_analysis.get('ratio_scaling_factor', 1.2)
        
        # Base estimate
        estimated_ratio = target_ratio * scaling_factor
        
        # Adjust based on exploration strategy
        if exploration_strategy == 'conservative':
            estimated_ratio *= 0.9  # 10% more conservative
        elif exploration_strategy == 'aggressive':
            estimated_ratio *= 1.1  # 10% more aggressive
        
        # Clamp to reasonable bounds
        return max(0.1, min(0.9, estimated_ratio))
    
    def _calculate_confidence(self, history_analysis: Dict[str, Any], 
                            exploration_strategy: str) -> float:
        """Calculate confidence in the strategy."""
        
        base_confidence = 0.7
        
        # Increase confidence with more data
        num_attempts = history_analysis.get('num_attempts', 0)
        confidence_boost = min(0.2, num_attempts * 0.05)
        
        # Adjust based on progress
        progress_rate = history_analysis.get('progress_rate', 0)
        if progress_rate > 0.01:
            confidence_boost += 0.1
        
        # Adjust based on strategy
        if exploration_strategy == 'conservative':
            confidence_boost += 0.05
        elif exploration_strategy == 'aggressive':
            confidence_boost -= 0.05
        
        return min(0.95, base_confidence + confidence_boost)
    
    def _assess_risk_level(self, pruning_ratio: float, target_ratio: float, 
                         exploration_strategy: str) -> str:
        """Assess the risk level of the strategy."""
        
        if pruning_ratio > target_ratio * 1.5:
            return 'high'
        elif pruning_ratio > target_ratio * 1.2:
            return 'medium'
        elif exploration_strategy == 'aggressive':
            return 'medium'
        else:
            return 'low'
    
    def _estimate_accuracy_drop(self, pruning_ratio: float, 
                              history_analysis: Dict[str, Any]) -> float:
        """Estimate expected accuracy drop."""
        
        # Use historical data if available
        if 'accuracy_vs_ratio_trend' in history_analysis:
            trend = history_analysis['accuracy_vs_ratio_trend']
            return max(0, trend.get('slope', -0.1) * pruning_ratio)
        
        # Default estimation: roughly 2% accuracy drop per 10% pruning
        return pruning_ratio * 0.2
    
    def _generate_rationale(self, pruning_ratio: float, importance_criterion: str,
                          round_to: int, exploration_strategy: str,
                          history_analysis: Dict[str, Any]) -> str:
        """Generate human-readable rationale for the strategy."""
        
        rationale_parts = []
        
        # Ratio justification
        if 'ratio_scaling_factor' in history_analysis:
            scaling = history_analysis['ratio_scaling_factor']
            rationale_parts.append(f"Pruning ratio {pruning_ratio:.2f} based on observed {scaling:.2f}x scaling factor")
        else:
            rationale_parts.append(f"Conservative pruning ratio {pruning_ratio:.2f} for initial exploration")
        
        # Criterion justification
        if importance_criterion == 'taylor':
            rationale_parts.append("Taylor criterion for gradient-based importance")
        elif importance_criterion == 'l1norm':
            rationale_parts.append("L1 norm for efficient magnitude-based pruning")
        
        # Strategy justification
        rationale_parts.append(f"{exploration_strategy.title()} exploration strategy")
        
        # History-based adjustments
        if history_analysis.get('best_result'):
            best_acc = history_analysis['best_result'].get('accuracy', 0)
            rationale_parts.append(f"Building on previous best accuracy of {best_acc:.1%}")
        
        return ". ".join(rationale_parts)
    
    def _apply_safety_validation(self, strategy: PruningStrategy, 
                               context: Dict[str, Any]) -> PruningStrategy:
        """Apply safety constraints to the strategy."""
        
        safety_limits = context.get('dataset_info', {}).get('safety_limits', {})
        
        # Check overall pruning ratio
        max_overall = safety_limits.get('max_overall_pruning', 0.8)
        if strategy.pruning_ratio > max_overall:
            logger.warning(f"⚠️ Pruning ratio {strategy.pruning_ratio:.1%} exceeds limit {max_overall:.1%}")
            # FIXED: Use safety_margin instead of undefined safety_multiplier
            safety_margin = self.safety_margin  # Already defined in _initialize_agent_components
            strategy.pruning_ratio = max_overall * (1 - safety_margin)
            strategy.risk_level = 'high'
            strategy.rationale += f". Clamped to safety limit {max_overall:.1%}"
        
        return strategy

    def _create_strategy_response(self, strategy: PruningStrategy, 
                                context: Dict[str, Any],
                                history_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create the strategy response."""
        
        return {
            'success': True,
            'agent_name': self.agent_name,
            'strategy': strategy.to_dict(),
            'history_analysis': history_analysis,
            'exploration_strategy': self.current_strategy,
            'directives': {
                'importance_criterion': strategy.importance_criterion,
                'pruning_ratio': strategy.pruning_ratio,
                'round_to': strategy.round_to,
                'global_pruning': strategy.global_pruning
            },
            'continue_iterations': strategy.continue_exploration,
            'confidence': strategy.confidence,
            'timestamp': context.get('timestamp', '')
        }
    
    def _create_stop_response(self, stop_reason: str, context: Dict[str, Any],
                            history_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when stopping exploration."""
        
        best_result = history_analysis.get('best_result', {})
        
        return {
            'success': True,
            'agent_name': self.agent_name,
            'continue_iterations': False,
            'stop_reason': stop_reason,
            'best_result': best_result,
            'final_recommendation': {
                'achieved_ratio': best_result.get('achieved_ratio', 0),
                'accuracy': best_result.get('accuracy', 0),
                'parameters': best_result.get('parameters', {})
            },
            'history_analysis': history_analysis,
            'timestamp': context.get('timestamp', '')
        }
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response into structured format."""
        
        # This agent uses structured reasoning instead of LLM for efficiency
        return AgentResponse(
            success=True,
            reasoning="Structured analysis without LLM",
            recommendations={},
            confidence=0.8,
            safety_checks={'structured_analysis': True},
            warnings=[],
            timestamp=context.get('timestamp', ''),
            agent_name=self.agent_name
        )

class HistoryAnalyzer:
    """Analyzes pruning history to extract patterns and insights."""
    
    def analyze_history(self, history: List[Dict[str, Any]], 
                       attempted_ratios: List[float],
                       target_ratio: float) -> Dict[str, Any]:
        """Comprehensive history analysis."""
        
        if not history:
            return {
                'num_attempts': 0,
                'converged': False,
                'cycling': False,
                'catastrophic_failure': False,
                'progress_rate': 0,
                'target_achievement_rate': 0,
                'parameter_recommendations': {}
            }
        
        analysis = {
            'num_attempts': len(history),
            'attempted_ratios': attempted_ratios,
            'target_ratio': target_ratio
        }
        
        # Extract results from history
        results = []
        for entry in history:
            if 'results' in entry and 'evaluation' in entry['results']:
                eval_results = entry['results']['evaluation']
                results.append({
                    'accuracy': eval_results.get('final_accuracy', 0),
                    'achieved_ratio': eval_results.get('params_reduction', 0),
                    'macs_reduction': eval_results.get('macs_reduction', 0),
                    'parameters': entry['results'].get('master', {})
                })
        
        if results:
            analysis.update(self._analyze_results(results, target_ratio))
        
        return analysis
    
    def _analyze_results(self, results: List[Dict[str, Any]], 
                        target_ratio: float) -> Dict[str, Any]:
        """Analyze results for patterns and insights."""
        
        analysis = {}
        
        # Find best result
        best_result = max(results, key=lambda x: x['accuracy'])
        analysis['best_result'] = best_result
        
        # Calculate progress rate
        if len(results) > 1:
            accuracy_trend = [r['accuracy'] for r in results]
            progress_rate = (accuracy_trend[-1] - accuracy_trend[0]) / len(accuracy_trend)
            analysis['progress_rate'] = progress_rate
        else:
            analysis['progress_rate'] = 0
        
        # Target achievement analysis
        target_achievements = [
            abs(r['achieved_ratio'] - target_ratio) < 0.01 for r in results
        ]
        analysis['target_achievement_rate'] = sum(target_achievements) / len(target_achievements)
        
        # Convergence detection
        analysis['converged'] = self._detect_convergence(results)
        
        # Cycling detection
        analysis['cycling'] = self._detect_cycling(results)
        
        # Catastrophic failure detection
        analysis['catastrophic_failure'] = any(r['accuracy'] < 0.1 for r in results)
        
        # Parameter recommendations
        analysis['parameter_recommendations'] = self._generate_parameter_recommendations(results, target_ratio)
        
        # Scaling factor estimation
        achieved_ratios = [r['achieved_ratio'] for r in results if r['achieved_ratio'] > 0]
        if achieved_ratios and len(results) > 1:
            # Estimate the relationship between requested and achieved ratios
            # This would need the original requested ratios to be accurate
            analysis['ratio_scaling_factor'] = np.mean(achieved_ratios) / target_ratio
        else:
            analysis['ratio_scaling_factor'] = 1.2  # Default estimate
        
        return analysis
    
    def _detect_convergence(self, results: List[Dict[str, Any]]) -> bool:
        """Detect if the process has converged."""
        
        if len(results) < 3:
            return False
        
        # Check if accuracy improvements are diminishing
        recent_accuracies = [r['accuracy'] for r in results[-3:]]
        improvements = [recent_accuracies[i+1] - recent_accuracies[i] 
                       for i in range(len(recent_accuracies)-1)]
        
        # Converged if all recent improvements are small
        return all(imp < 0.005 for imp in improvements)  # 0.5% threshold
    
    def _detect_cycling(self, results: List[Dict[str, Any]]) -> bool:
        """Detect cycling behavior in results."""
        
        if len(results) < 4:
            return False
        
        # Check if we're seeing repeated accuracy values
        recent_accuracies = [round(r['accuracy'], 3) for r in results[-4:]]
        unique_accuracies = set(recent_accuracies)
        
        # Cycling if we see repeated values
        return len(unique_accuracies) < len(recent_accuracies) * 0.7
    
    def _generate_parameter_recommendations(self, results: List[Dict[str, Any]], 
                                          target_ratio: float) -> Dict[str, Any]:
        """Generate parameter recommendations based on results."""
        
        recommendations = {}
        
        # Find best performing parameters
        best_result = max(results, key=lambda x: x['accuracy'])
        best_params = best_result.get('parameters', {})
        
        if best_params:
            recommendations['best_importance_criterion'] = best_params.get('importance_criterion')
            recommendations['optimal_round_to'] = best_params.get('round_to')
        
        # Estimate optimal ratio for target achievement
        target_close_results = [
            r for r in results 
            if abs(r['achieved_ratio'] - target_ratio) < 0.05
        ]
        
        if target_close_results:
            best_target_result = max(target_close_results, key=lambda x: x['accuracy'])
            recommendations['optimal_ratio'] = best_target_result.get('achieved_ratio', target_ratio)
        
        return recommendations

