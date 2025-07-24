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

from .base_agent import BaseAgent, AgentResponse
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None, profiler=None):
        """
        Initialize AnalysisAgent with proper BaseAgent inheritance.
        """
        # Call BaseAgent constructor with proper parameters
        super().__init__("AnalysisAgent", llm_client, profiler)
        
        # Store configuration
        self.config = config or {}
        
        # Initialize agent-specific components
        self._initialize_agent_components()
        
        logger.info("🔍 Analysis Agent initialized with proper inheritance")
    
    def _initialize_agent_components(self):
        """Initialize agent-specific components based on configuration."""
        
        # Analysis components - will be initialized when needed
        self.dependency_analyzer: Optional[DependencyAnalyzer] = None
        self.isomorphic_analyzer: Optional[IsomorphicAnalyzer] = None
        
        # Analysis configuration
        analysis_config = self.config.get('analysis', {})
        self.enable_dependency_analysis = analysis_config.get('dependency_analysis', True)
        self.enable_isomorphic_analysis = analysis_config.get('isomorphic_analysis', True)
        self.enable_sensitivity_analysis = analysis_config.get('sensitivity_analysis', True)
        
        # Recommendation configuration
        recommendation_config = self.config.get('recommendations', {})
        self.default_importance_criterion = recommendation_config.get('default_importance', 'taylor')
        self.safety_margin = recommendation_config.get('safety_margin', 0.1)
        self.conservative_mode = recommendation_config.get('conservative_mode', True)
        
        # Analysis results storage
        self.analysis_results = {}
        self.recommendations = {}
        
        logger.info("🔍 Analysis Agent components initialized with configuration")

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
        """Validate that the input state contains required profiling results with flexible field checking."""
        
        required_fields = ['model', 'profile_results', 'master_results']
        
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                logger.error(f"❌ Missing required field in state: {field}")
                return False
        
        # Check profiling results structure
        profiling_results = state.profile_results
        if not isinstance(profiling_results, dict):
            logger.error("❌ Profiling results must be a dictionary")
            return False
        
        logger.info(f"🔍 Available profiling fields: {list(profiling_results.keys())}")
        
        # Check if we have any useful profiling data
        if not profiling_results:
            logger.error("❌ Profiling results dictionary is empty")
            return False
        
        # The profiling agent might return different field structures
        has_useful_data = any(
            isinstance(value, dict) and value 
            for value in profiling_results.values()
        )
        
        if not has_useful_data:
            logger.error("❌ No useful profiling data found")
            return False
        
        logger.info("✅ Input state validation passed with flexible checking")
        return True

    def _get_profiling_data(self, state: PruningState, field_name: str, default=None):
        """Safely get profiling data with enhanced structure handling to eliminate warnings."""
        
        profiling_results = state.profile_results
        
        # The profiling agent returns: {'success': True, 'agent_name': 'ProfilingAgent', 'profile': {...}, 'llm_analysis': {...}, ...}
        
        actual_profile_data = None
        
        # First, try to get the actual profile data from the nested structure
        if isinstance(profiling_results, dict):
            # Check for 'profile' field (most likely location)
            if 'profile' in profiling_results and isinstance(profiling_results['profile'], dict):
                actual_profile_data = profiling_results['profile']
                logger.info(f"🔍 Found profiling data in 'profile' field with keys: {list(actual_profile_data.keys())}")
            
            # Check for direct field access
            elif field_name in profiling_results:
                logger.info(f"✅ Found '{field_name}' directly in profiling results")
                return profiling_results[field_name]
        
        # If we found the actual profile data, search within it
        if actual_profile_data:
            # Try exact field name first
            if field_name in actual_profile_data:
                logger.info(f"✅ Found '{field_name}' in profile data")
                return actual_profile_data[field_name]
            
            # Try common alternative names within the profile data
            field_alternatives = {
                'model_analysis': ['model_info', 'architecture_analysis', 'model_profile', 'model_summary', 'architecture_info'],
                'layer_analysis': ['layers', 'layer_info', 'layer_profiles', 'layer_summary', 'layer_details'],
                'dependency_graph': ['dependencies', 'dependency_analysis', 'layer_dependencies', 'graph']
            }
            
            if field_name in field_alternatives:
                for alt_name in field_alternatives[field_name]:
                    if alt_name in actual_profile_data:
                        logger.info(f"✅ Using alternative field '{alt_name}' for '{field_name}' in profile data")
                        return actual_profile_data[alt_name]
        
        if isinstance(profiling_results, dict):
            logger.info(f"🔍 Searching for '{field_name}' in available top-level fields: {list(profiling_results.keys())}")
            
            # Try to find data in any nested dictionaries
            for key, value in profiling_results.items():
                if isinstance(value, dict) and value:
                    if field_name in value:
                        logger.info(f"✅ Found '{field_name}' in nested field '{key}'")
                        return value[field_name]
                    
                    # Try alternatives in nested fields
                    if field_name in field_alternatives:
                        for alt_name in field_alternatives[field_name]:
                            if alt_name in value:
                                logger.info(f"✅ Found alternative '{alt_name}' for '{field_name}' in nested field '{key}'")
                                return value[alt_name]
        
        logger.info(f"🔧 Creating enhanced default structure for '{field_name}' to eliminate warnings")
        
        if field_name == 'model_analysis':
            # Create comprehensive model analysis from available data
            model_name = getattr(state, 'model_name', 'unknown')
            model = getattr(state, 'model', None)
            
            enhanced_model_analysis = {
                'architecture_type': model_name,
                'model_name': model_name,
                'total_parameters': 0,
                'total_flops': 0,
                'complexity': 'unknown',
                'layer_count': 0,
                'prunable_layers': 0
            }
            
            # Extract detailed info from the model directly
            if model is not None:
                try:
                    total_params = sum(p.numel() for p in model.parameters())
                    enhanced_model_analysis['total_parameters'] = total_params
                    
                    # Count layers
                    layer_count = 0
                    prunable_count = 0
                    for name, module in model.named_modules():
                        if len(list(module.children())) == 0:  # Leaf modules only
                            layer_count += 1
                            if hasattr(module, 'weight') and module.weight is not None:
                                prunable_count += 1
                    
                    enhanced_model_analysis['layer_count'] = layer_count
                    enhanced_model_analysis['prunable_layers'] = prunable_count
                    
                    # Estimate complexity based on parameter count
                    if total_params > 100_000_000:  # 100M+
                        enhanced_model_analysis['complexity'] = 'high'
                    elif total_params > 10_000_000:  # 10M+
                        enhanced_model_analysis['complexity'] = 'medium'
                    else:
                        enhanced_model_analysis['complexity'] = 'low'
                        
                    logger.info(f"✅ Created enhanced model analysis: {total_params:,} params, {layer_count} layers, {prunable_count} prunable")
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract detailed model info: {e}")
            
            return enhanced_model_analysis
            
        elif field_name == 'layer_analysis':
            # Create comprehensive layer analysis from model
            model = getattr(state, 'model', None)
            if model is not None:
                try:
                    layer_analysis = {}
                    for name, module in model.named_modules():
                        if len(list(module.children())) == 0:  # Leaf modules only
                            param_count = sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
                            layer_analysis[name] = {
                                'type': type(module).__name__,
                                'parameters': param_count,
                                'prunable': hasattr(module, 'weight') and module.weight is not None,
                                'shape': list(module.weight.shape) if hasattr(module, 'weight') and module.weight is not None else [],
                                'bias': hasattr(module, 'bias') and module.bias is not None
                            }
                    
                    logger.info(f"✅ Created comprehensive layer analysis with {len(layer_analysis)} layers")
                    return layer_analysis
                except Exception as e:
                    logger.warning(f"⚠️ Could not create layer analysis: {e}")
            
            # Return empty dict if model analysis fails
            logger.info("🔧 Returning empty layer analysis structure")
            return {}
            
        elif field_name == 'dependency_graph':
            # Create basic dependency structure
            return {
                'dependencies': [], 
                'coupled_layers': [],
                'analysis_method': 'default_structure'
            }
        
        return default or {}

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

    def _analyze_isomorphic_groups(self, state: PruningState) -> Dict[str, Any]:
        """Analyze isomorphic layer groups for coordinated pruning using correct analyzer methods."""
        
        try:
            if self.isomorphic_analyzer is None:
                logger.warning("⚠️ Isomorphic analyzer not initialized, using basic analysis")
                return {
                    'isomorphic_groups': [],
                    'group_count': 0,
                    'coordination_opportunities': [],
                    'analysis_method': 'basic_fallback'
                }
            
            # The actual method is 'create_isomorphic_groups', not 'analyze_isomorphic_groups'
            target_ratio = 0.5  # Default target ratio
            
            # Get target ratio from master results if available
            if hasattr(state, 'master_results') and state.master_results:
                master_directives = state.master_results.get('directives', {})
                target_ratio = master_directives.get('pruning_ratio', 0.5)
            
            # Use the correct method
            isomorphic_groups = self.isomorphic_analyzer.create_isomorphic_groups(
                target_ratio=target_ratio,
                group_ratio_multiplier=1.0
            )
            
            # Get group statistics
            group_stats = self.isomorphic_analyzer.get_group_statistics(isomorphic_groups)
            
            # Convert to expected format
            groups_list = []
            for group_name, group in isomorphic_groups.items():
                groups_list.append({
                    'name': group_name,
                    'layers': group.layer_names,
                    'layer_count': len(group.layers),
                    'group_type': group.group_type,
                    'total_parameters': group.get_total_parameters(),
                    'pruning_ratio': group.pruning_ratio
                })
            
            logger.info(f"✅ Isomorphic analysis completed: {len(groups_list)} groups found")
            
            return {
                'isomorphic_groups': groups_list,
                'group_count': len(groups_list),
                'coordination_opportunities': [g['name'] for g in groups_list if g['layer_count'] > 1],
                'group_statistics': group_stats,
                'analysis_method': 'isomorphic_analyzer'
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Isomorphic analysis failed: {str(e)}, using enhanced fallback")
            
            try:
                model = state.model
                if model is not None:
                    basic_groups = []
                    linear_layers = []
                    conv_layers = []
                    
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            linear_layers.append(name)
                        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                            conv_layers.append(name)
                    
                    if linear_layers:
                        basic_groups.append({
                            'name': 'linear_group',
                            'layers': linear_layers,
                            'layer_count': len(linear_layers),
                            'group_type': 'linear',
                            'total_parameters': 0,
                            'pruning_ratio': 0.5
                        })
                    
                    if conv_layers:
                        basic_groups.append({
                            'name': 'conv_group',
                            'layers': conv_layers,
                            'layer_count': len(conv_layers),
                            'group_type': 'conv',
                            'total_parameters': 0,
                            'pruning_ratio': 0.5
                        })
                    
                    logger.info(f"✅ Created basic isomorphic groups: {len(basic_groups)} groups")
                    
                    return {
                        'isomorphic_groups': basic_groups,
                        'group_count': len(basic_groups),
                        'coordination_opportunities': [g['name'] for g in basic_groups],
                        'analysis_method': 'enhanced_fallback'
                    }
            except Exception as fallback_error:
                logger.warning(f"⚠️ Enhanced fallback also failed: {fallback_error}")
            
            return {
                'isomorphic_groups': [],
                'group_count': 0,
                'coordination_opportunities': [],
                'error': str(e),
                'analysis_method': 'error_fallback'
            }

    def _analyze_sensitivity(self, state: PruningState) -> Dict[str, Any]:
        """Analyze layer sensitivity to pruning for strategic recommendations."""
        
        try:
            model = state.model
            if model is None:
                return {
                    'sensitivity_scores': {},
                    'high_sensitivity_layers': [],
                    'low_sensitivity_layers': [],
                    'analysis_method': 'no_model'
                }
            
            # Basic sensitivity analysis based on layer types and parameters
            sensitivity_scores = {}
            high_sensitivity = []
            low_sensitivity = []
            
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    module_type = type(module).__name__
                    
                    # Assign sensitivity based on layer type
                    if 'Attention' in module_type or 'MultiheadAttention' in module_type:
                        sensitivity = 0.9  # High sensitivity
                    elif 'LayerNorm' in module_type or 'BatchNorm' in module_type:
                        sensitivity = 0.8  # High sensitivity
                    elif 'Linear' in module_type and hasattr(module, 'weight'):
                        # Check if it's a classifier (last layer)
                        if 'classifier' in name.lower() or 'head' in name.lower():
                            sensitivity = 0.95  # Very high sensitivity
                        else:
                            sensitivity = 0.6  # Medium sensitivity
                    elif 'Conv' in module_type:
                        sensitivity = 0.5  # Medium sensitivity
                    elif 'Dropout' in module_type or 'ReLU' in module_type:
                        sensitivity = 0.2  # Low sensitivity
                    else:
                        sensitivity = 0.5  # Default medium sensitivity
                    
                    sensitivity_scores[name] = sensitivity
                    
                    if sensitivity > 0.8:
                        high_sensitivity.append(name)
                    elif sensitivity < 0.4:
                        low_sensitivity.append(name)
            
            return {
                'sensitivity_scores': sensitivity_scores,
                'high_sensitivity_layers': high_sensitivity,
                'low_sensitivity_layers': low_sensitivity,
                'average_sensitivity': sum(sensitivity_scores.values()) / len(sensitivity_scores) if sensitivity_scores else 0,
                'analysis_method': 'layer_type_based'
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Sensitivity analysis failed: {str(e)}")
            return {
                'sensitivity_scores': {},
                'high_sensitivity_layers': [],
                'low_sensitivity_layers': [],
                'error': str(e),
                'analysis_method': 'error_fallback'
            }

    def _identify_pruning_opportunities(self, state: PruningState) -> Dict[str, Any]:
        """Identify specific pruning opportunities based on analysis results."""
        
        try:
            model = state.model
            if model is None:
                return {
                    'opportunities': [],
                    'total_opportunities': 0,
                    'potential_reduction': 0,
                    'analysis_method': 'no_model'
                }
            
            opportunities = []
            total_params = 0
            prunable_params = 0
            
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    module_params = module.weight.numel()
                    total_params += module_params
                    
                    module_type = type(module).__name__
                    
                    # Determine pruning potential
                    if 'Linear' in module_type:
                        if 'classifier' in name.lower() or 'head' in name.lower():
                            pruning_potential = 0.1  # Conservative for classifier
                        else:
                            pruning_potential = 0.6  # Good potential for other linear layers
                    elif 'Conv' in module_type:
                        pruning_potential = 0.5  # Medium potential for conv layers
                    else:
                        pruning_potential = 0.3  # Conservative for other types
                    
                    prunable_params += module_params * pruning_potential
                    
                    opportunities.append({
                        'layer_name': name,
                        'layer_type': module_type,
                        'parameters': module_params,
                        'pruning_potential': pruning_potential,
                        'estimated_reduction': int(module_params * pruning_potential)
                    })
            
            # Sort by potential reduction
            opportunities.sort(key=lambda x: x['estimated_reduction'], reverse=True)
            
            return {
                'opportunities': opportunities,
                'total_opportunities': len(opportunities),
                'potential_reduction': prunable_params / total_params if total_params > 0 else 0,
                'total_parameters': total_params,
                'prunable_parameters': int(prunable_params),
                'analysis_method': 'parameter_based'
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Pruning opportunities analysis failed: {str(e)}")
            return {
                'opportunities': [],
                'total_opportunities': 0,
                'potential_reduction': 0,
                'error': str(e),
                'analysis_method': 'error_fallback'
            }

    def _assess_architecture_complexity(self, layer_types: Dict[str, int], total_layers: int) -> str:
        """Assess the complexity of the model architecture."""
        
        if total_layers == 0:
            return 'unknown'
        
        # Count different layer types
        unique_types = len(layer_types)
        
        # Check for complex layer types
        complex_types = ['MultiheadAttention', 'Attention', 'TransformerBlock', 'LayerNorm']
        has_complex_layers = any(layer_type in str(layer_types.keys()) for layer_type in complex_types)
        
        if total_layers > 100 or (has_complex_layers and total_layers > 50):
            return 'high'
        elif total_layers > 50 or (has_complex_layers and total_layers > 20):
            return 'medium'
        elif total_layers > 20:
            return 'medium-low'
        else:
            return 'low'

    def _assess_constraint_risk(self, constraints: Dict[str, Any], state: PruningState) -> str:
        """Assess the risk level of constraint violations."""
        
        constraint_count = len(constraints.get('coupling_constraints', []))
        dependency_count = len(constraints.get('layer_dependencies', []))
        
        total_constraints = constraint_count + dependency_count
        
        if total_constraints > 50:
            return 'high'
        elif total_constraints > 20:
            return 'medium'
        elif total_constraints > 5:
            return 'low'
        else:
            return 'minimal'

    def _recommend_importance_criterion(self, model_analysis: Dict[str, Any], 
                                      analysis_results: Dict[str, Any],
                                      master_directives: Dict[str, Any]) -> str:
        """Recommend the best importance criterion based on analysis."""
        
        # Check master agent directive first
        if 'importance_criterion' in master_directives:
            return master_directives['importance_criterion']
        
        # Analyze architecture type
        arch_type = model_analysis.get('architecture_type', 'unknown')
        
        if 'transformer' in arch_type.lower() or 'attention' in arch_type.lower():
            return 'taylor'  # Taylor expansion works well for transformers
        elif 'cnn' in arch_type.lower() or 'conv' in arch_type.lower():
            return 'l1norm'  # L1 norm works well for CNNs
        else:
            return self.default_importance_criterion

    def _recommend_pruning_ratios(self, analysis_results: Dict[str, Any],
                                master_directives: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend layer-specific pruning ratios."""
        
        # Get master agent directive
        global_ratio = master_directives.get('pruning_ratio', 0.5)
        
        # Get sensitivity analysis
        sensitivity = analysis_results.get('sensitivity_analysis', {})
        sensitivity_scores = sensitivity.get('sensitivity_scores', {})
        
        # Create layer-specific ratios
        layer_ratios = {}
        
        for layer_name, sensitivity_score in sensitivity_scores.items():
            # Adjust ratio based on sensitivity (lower sensitivity = higher pruning)
            if sensitivity_score > 0.8:
                layer_ratios[layer_name] = global_ratio * 0.5  # Conservative for sensitive layers
            elif sensitivity_score > 0.6:
                layer_ratios[layer_name] = global_ratio * 0.8  # Moderate for medium sensitivity
            else:
                layer_ratios[layer_name] = global_ratio * 1.2  # Aggressive for low sensitivity
        
        return {
            'global_ratio': global_ratio,
            'layer_specific_ratios': layer_ratios,
            'adaptation_strategy': 'sensitivity_based'
        }

    def _recommend_group_multipliers(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend group multipliers for coordinated pruning."""
        
        isomorphic_analysis = analysis_results.get('isomorphic_analysis', {})
        groups = isomorphic_analysis.get('isomorphic_groups', [])
        
        group_multipliers = {}
        
        for i, group in enumerate(groups):
            group_id = f"group_{i}"
            group_size = len(group.get('layers', []))
            
            # Larger groups get more conservative multipliers
            if group_size > 10:
                multiplier = 0.8
            elif group_size > 5:
                multiplier = 0.9
            else:
                multiplier = 1.0
            
            group_multipliers[group_id] = {
                'multiplier': multiplier,
                'layers': group.get('layers', []),
                'rationale': f'Group size: {group_size}'
            }
        
        return {
            'group_multipliers': group_multipliers,
            'coordination_strategy': 'size_based'
        }

    def _recommend_safety_constraints(self, analysis_results: Dict[str, Any],
                                    state: PruningState) -> Dict[str, Any]:
        """Recommend safety constraints based on analysis."""
        
        sensitivity = analysis_results.get('sensitivity_analysis', {})
        high_sensitivity_layers = sensitivity.get('high_sensitivity_layers', [])
        
        return {
            'protected_layers': high_sensitivity_layers,
            'max_layer_pruning': 0.8,
            'min_accuracy_threshold': 0.4,
            'safety_margin': self.safety_margin,
            'constraint_enforcement': 'strict' if self.conservative_mode else 'moderate'
        }

    def _recommend_execution_strategy(self, analysis_results: Dict[str, Any],
                                    master_directives: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend execution strategy for pruning."""
        
        complexity = analysis_results.get('architecture_analysis', {}).get('architecture_complexity', 'medium')
        
        if complexity == 'high':
            approach = 'gradual'
            phases = 3
        elif complexity == 'medium':
            approach = 'moderate'
            phases = 2
        else:
            approach = 'direct'
            phases = 1
        
        return {
            'approach': approach,
            'phases': phases,
            'execution_phases': self._define_execution_phases(approach),
            'validation_strategy': 'per_phase' if phases > 1 else 'final'
        }

    def _define_execution_phases(self, approach: str) -> List[Dict[str, Any]]:
        """Define execution phases based on approach."""
        
        if approach == 'gradual':
            return [
                {'phase': 1, 'ratio': 0.3, 'focus': 'low_sensitivity_layers'},
                {'phase': 2, 'ratio': 0.6, 'focus': 'medium_sensitivity_layers'},
                {'phase': 3, 'ratio': 1.0, 'focus': 'final_adjustments'}
            ]
        elif approach == 'moderate':
            return [
                {'phase': 1, 'ratio': 0.7, 'focus': 'bulk_pruning'},
                {'phase': 2, 'ratio': 1.0, 'focus': 'fine_tuning'}
            ]
        else:
            return [
                {'phase': 1, 'ratio': 1.0, 'focus': 'direct_pruning'}
            ]

    def _perform_comprehensive_analysis(self, state: PruningState) -> Dict[str, Any]:
        """Perform comprehensive analysis of the model and profiling results."""
        
        with self.profiler.timer("comprehensive_analysis"):
            logger.info("🔍 Performing comprehensive analysis")
            
            model_analysis = self._get_profiling_data(state, 'model_analysis')
            
            analysis_results = {
                'model_info': model_analysis,
                'architecture_analysis': self._analyze_architecture(state),
                'dependency_analysis': self._analyze_dependencies(state),
                'isomorphic_analysis': self._analyze_isomorphic_groups(state),
                'sensitivity_analysis': self._analyze_sensitivity(state),
                'pruning_opportunities': self._identify_pruning_opportunities(state)
            }
            
            logger.info("✅ Comprehensive analysis completed")
            return analysis_results

    def _analyze_architecture(self, state: PruningState) -> Dict[str, Any]:
        """Analyze model architecture for pruning insights with enhanced fallback."""
        
        # Use enhanced data retrieval
        layer_analysis = self._get_profiling_data(state, 'layer_analysis', {})
        
        if not layer_analysis:
            logger.info("🔧 No layer analysis data available, using enhanced model introspection")
            # Use enhanced model introspection instead of just logging a warning
            return self._analyze_model_directly(state)
        
        # Process existing layer analysis
        total_layers = len(layer_analysis)
        prunable_layers = sum(1 for layer in layer_analysis.values() if layer.get('prunable', False))
        
        # Analyze layer types
        layer_types = {}
        for layer_info in layer_analysis.values():
            layer_type = layer_info.get('type', 'unknown')
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        # Calculate complexity metrics
        total_params = sum(layer.get('parameters', 0) for layer in layer_analysis.values())
        total_flops = sum(layer.get('flops', 0) for layer in layer_analysis.values())
        
        logger.info(f"✅ Architecture analysis completed: {total_layers} layers, {prunable_layers} prunable")
        
        return {
            'total_layers': total_layers,
            'prunable_layers': prunable_layers,
            'pruning_ratio': prunable_layers / total_layers if total_layers > 0 else 0,
            'layer_types': layer_types,
            'total_parameters': total_params,
            'total_flops': total_flops,
            'architecture_complexity': self._assess_architecture_complexity(layer_types, total_layers),
            'analysis_source': 'profiling_data'
        }

    def _analyze_model_directly(self, state: PruningState) -> Dict[str, Any]:
        """Fallback method to analyze model directly when profiling data is unavailable."""
        
        model = state.model
        if model is None:
            return {
                'total_layers': 0,
                'prunable_layers': 0,
                'pruning_ratio': 0,
                'layer_types': {},
                'total_parameters': 0,
                'total_flops': 0,
                'architecture_complexity': 'unknown'
            }
        
        # Basic model analysis
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count layers by type
        layer_types = {}
        prunable_layers = 0
        total_layers = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                total_layers += 1
                module_type = type(module).__name__
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
                
                # Check if layer is prunable
                if hasattr(module, 'weight') and module.weight is not None:
                    prunable_layers += 1
        
        return {
            'total_layers': total_layers,
            'prunable_layers': prunable_layers,
            'pruning_ratio': prunable_layers / total_layers if total_layers > 0 else 0,
            'layer_types': layer_types,
            'total_parameters': total_params,
            'total_flops': 0,  # Would need more complex calculation
            'architecture_complexity': 'medium' if total_layers > 50 else 'low'
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
        """Generate strategic pruning recommendations based on analysis."""
        
        with self.profiler.timer("strategic_recommendations"):
            logger.info("💡 Generating strategic recommendations")
            
            model_analysis = self._get_profiling_data(state, 'model_analysis')
            
            # Get master agent directives
            master_results = state.master_results
            master_directives = master_results.get('directives', {})
            
            recommendations = {
                'importance_criterion': self._recommend_importance_criterion(
                    model_analysis, analysis_results, master_directives
                ),
                'pruning_ratios': self._recommend_pruning_ratios(
                    analysis_results, master_directives
                ),
                'group_multipliers': self._recommend_group_multipliers(
                    analysis_results
                ),
                'safety_constraints': self._recommend_safety_constraints(
                    analysis_results, state
                ),
                'execution_strategy': self._recommend_execution_strategy(
                    analysis_results, master_directives
                )
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

    def get_agent_role(self) -> str:
        """Return the role of this agent."""
        return "analysis_agent"
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for the analysis agent."""
        
        model_info = context.get('model_info', {})
        profiling_results = context.get('profiling_results', {})
        pruning_config = context.get('pruning_config', {})
        history = context.get('history', [])
        
        prompt = f"""You are an expert Analysis Agent in a multi-agent neural network pruning system.

ROLE: Analyze profiling results and recommend optimal pruning strategies with intelligent parameter tuning.

CURRENT CONTEXT:
- Model: {model_info.get('name', 'Unknown')} ({model_info.get('total_params', 0):,} parameters)
- Architecture: {model_info.get('architecture_type', 'Unknown')}
- Target: {pruning_config.get('target_ratio', 0.5)*100:.1f}% compression
- Current iteration: {len(history) + 1}

PROFILING INSIGHTS:
- Layer dependencies: {len(profiling_results.get('dependencies', []))} detected
- Coupled layers: {len(profiling_results.get('coupled_layers', []))} groups
- Pruning opportunities: {profiling_results.get('pruning_potential', 'Unknown')}
- Architecture complexity: {profiling_results.get('complexity', 'Unknown')}

ANALYSIS RESPONSIBILITIES:
1. Select optimal importance criterion (magnitude, Taylor, gradient-based)
2. Recommend pruning ratios for different layer types
3. Tune exploration parameters based on model characteristics
4. Validate strategy against safety constraints
5. Learn from previous iteration outcomes

DECISION FRAMEWORK:
- DATA-DRIVEN: Use profiling results to guide decisions
- ADAPTIVE: Adjust strategy based on model architecture and history
- SAFE: Ensure recommendations respect safety constraints
- EFFICIENT: Balance accuracy preservation with compression goals

IMPORTANCE CRITERIA SELECTION:
- Magnitude: Fast, works well for over-parameterized models
- Taylor: More accurate, requires gradients, good for fine-tuned models
- Gradient-based: Most accurate, computationally expensive, best for critical layers

Provide structured analysis with clear rationale for each recommendation."""
        
        return prompt
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response for analysis decisions."""
        
        try:
            # Extract key analysis decisions from response
            decisions = {}
            
            # Look for importance criterion recommendation
            import re
            
            if 'taylor' in response.lower():
                decisions['importance_criterion'] = 'taylor'
            elif 'gradient' in response.lower():
                decisions['importance_criterion'] = 'gradient'
            elif 'magnitude' in response.lower():
                decisions['importance_criterion'] = 'magnitude'
            else:
                decisions['importance_criterion'] = 'magnitude'  # default
            
            # Extract recommended pruning ratios
            mlp_ratio_match = re.search(r'mlp.*?(\d+\.?\d*)%', response.lower())
            if mlp_ratio_match:
                decisions['recommended_mlp_ratio'] = float(mlp_ratio_match.group(1)) / 100.0
            else:
                decisions['recommended_mlp_ratio'] = 0.10  # default 10%
            
            attn_ratio_match = re.search(r'attention.*?(\d+\.?\d*)%', response.lower())
            if attn_ratio_match:
                decisions['recommended_attention_ratio'] = float(attn_ratio_match.group(1)) / 100.0
            else:
                decisions['recommended_attention_ratio'] = 0.05  # default 5%
            
            # Extract exploration strategy
            if 'aggressive' in response.lower():
                decisions['exploration_strategy'] = 'aggressive'
                decisions['round_to_values'] = [0.25, 0.5, 0.75]
            elif 'conservative' in response.lower():
                decisions['exploration_strategy'] = 'conservative'
                decisions['round_to_values'] = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                decisions['exploration_strategy'] = 'balanced'
                decisions['round_to_values'] = [0.2, 0.4, 0.6, 0.8]
            
            # Extract group ratio multipliers
            multiplier_match = re.search(r'multiplier.*?(\d+\.?\d*)', response.lower())
            if multiplier_match:
                decisions['group_ratio_multipliers'] = [1.0, float(multiplier_match.group(1)), 2.0]
            else:
                decisions['group_ratio_multipliers'] = [1.0, 1.5, 2.0]  # default
            
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
                if any(keyword in line.lower() for keyword in ['because', 'since', 'due to', 'reason']):
                    rationale.append(line.strip())
            
            decisions['rationale'] = rationale[:3]  # Keep top 3 reasons
            
            # Determine success based on content quality
            success = (
                len(decisions) >= 4 and 
                'error' not in response.lower() and
                decisions['importance_criterion'] in ['magnitude', 'taylor', 'gradient']
            )
            
            return AgentResponse(
                success=success,
                data=decisions,
                message=f"Analysis complete: {decisions['importance_criterion']} criterion, {decisions['exploration_strategy']} strategy",
                confidence=confidence
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={
                    'importance_criterion': 'magnitude',  # safe fallback
                    'recommended_mlp_ratio': 0.10,
                    'recommended_attention_ratio': 0.05,
                    'exploration_strategy': 'conservative',
                    'round_to_values': [0.1, 0.2, 0.3, 0.4, 0.5],
                    'group_ratio_multipliers': [1.0, 1.5, 2.0]
                },
                message=f"Failed to parse analysis response, using safe defaults: {str(e)}",
                confidence=0.3
            )
