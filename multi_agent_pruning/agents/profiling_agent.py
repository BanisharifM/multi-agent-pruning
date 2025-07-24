"""
Optimized Profiling Agent for Multi-Agent LLM Pruning - UPDATED VERSION

Enhanced version of the user's profiling agent with:
- Precomputed dependency analysis
- Cached architecture detection
- Improved safety constraints
- More efficient LLM usage
- Better error handling

"""

import torch
import torch.nn as nn
import timm
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .base_agent import BaseAgent, AgentResponse
from ..core.dependency_analyzer import DependencyAnalyzer
from ..core.isomorphic_analyzer import IsomorphicAnalyzer
from ..utils.profiler import TimingProfiler
from ..utils.metrics import compute_macs, compute_params

logger = logging.getLogger(__name__)

@dataclass
class ModelProfile:
    """Comprehensive model profile with precomputed analysis."""
    
    # Basic model info
    model_name: str
    architecture_type: str
    total_params: int
    total_macs: float
    
    # Layer analysis
    layer_count: int
    prunable_layers: List[str]
    critical_layers: List[str]
    sensitive_layers: List[str]
    
    # Dependency analysis
    dependency_graph: Dict[str, Any]
    isomorphic_groups: Dict[str, Any]
    coupling_constraints: List[str]
    
    # Pruning recommendations
    recommended_ratios: Dict[str, float]
    safety_limits: Dict[str, float]
    importance_criterion: str
    
    # Performance estimates
    memory_usage_mb: float
    inference_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'architecture_type': self.architecture_type,
            'total_params': self.total_params,
            'total_macs': self.total_macs,
            'layer_count': self.layer_count,
            'prunable_layers': self.prunable_layers,
            'critical_layers': self.critical_layers,
            'sensitive_layers': self.sensitive_layers,
            'dependency_graph': self.dependency_graph,
            'isomorphic_groups': self.isomorphic_groups,
            'coupling_constraints': self.coupling_constraints,
            'recommended_ratios': self.recommended_ratios,
            'safety_limits': self.safety_limits,
            'importance_criterion': self.importance_criterion,
            'memory_usage_mb': self.memory_usage_mb,
            'inference_time_ms': self.inference_time_ms
        }

class ProfilingAgent(BaseAgent):
    """
    Enhanced Profiling Agent with precomputation and caching.
    
    Improvements over original:
    1. Precomputed dependency analysis
    2. Cached architecture detection
    3. More efficient LLM usage
    4. Better safety constraint detection
    5. Comprehensive profiling with timing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None, profiler=None):
        """
        Initialize ProfilingAgent with proper BaseAgent inheritance.
        
        CHANGES MADE:
        - Added llm_client and profiler parameters to match BaseAgent signature
        - Call super().__init__ with proper parameters
        - Initialize agent-specific components in separate method
        """
        # Call BaseAgent constructor with proper parameters
        super().__init__("ProfilingAgent", llm_client, profiler)
        
        # Store configuration
        self.config = config or {}
        
        # Initialize profiling-specific components
        self._initialize_profiling_components()
        
        # Initialize analyzers
        self.dependency_analyzer = None
        self.isomorphic_analyzer = None
        
        # Initialize caching
        self.profile_cache = {}
        self.enable_caching = self.config.get('enable_caching', True)
        
        logger.info("📊 Profiling Agent initialized with proper inheritance")
    
# DELETE
    # def _initialize_profiling_components(self):
    #     """Initialize profiling-specific components based on configuration."""
        
    #     # Initialize dependency analyzer
    #     dependency_config = self.config.get('dependency_analysis', {})
    #     if dependency_config.get('enabled', True):
    #         self.dependency_analyzer = DependencyAnalyzer(
    #             cache_enabled=dependency_config.get('cache_enabled', True),
    #             cache_dir=dependency_config.get('cache_dir', './cache/dependencies')
    #         )
        
    #     # Initialize isomorphic analyzer
    #     isomorphic_config = self.config.get('isomorphic_analysis', {})
    #     if isomorphic_config.get('enabled', True):
    #         self.isomorphic_analyzer = IsomorphicAnalyzer(
    #             similarity_threshold=isomorphic_config.get('similarity_threshold', 0.95),
    #             cache_enabled=isomorphic_config.get('cache_enabled', True)
    #         )
        
    #     # Configure profiling behavior
    #     profiling_config = self.config.get('profiling', {})
    #     self.enable_detailed_profiling = profiling_config.get('detailed', True)
    #     self.enable_memory_profiling = profiling_config.get('memory', True)
    #     self.enable_flops_counting = profiling_config.get('flops', True)
    
    def _initialize_profiling_components(self):
        """Initialize profiling-specific components based on configuration."""
        
        # Initialize analyzers as None - will be created when needed with proper parameters
        self.dependency_analyzer = None
        self.isomorphic_analyzer = None
        
        # Initialize caching
        self.profile_cache = {}
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Configure profiling behavior
        profiling_config = self.config.get('profiling', {})
        self.enable_detailed_profiling = profiling_config.get('detailed', True)
        self.enable_memory_profiling = profiling_config.get('memory', True)
        self.enable_flops_counting = profiling_config.get('flops', True)
        
        # Store analyzer configurations for later use
        self.dependency_config = self.config.get('dependency_analysis', {})
        self.isomorphic_config = self.config.get('isomorphic_analysis', {})
        
        logger.info("📊 Profiling Agent components initialized with configuration")
        
    def get_agent_role(self) -> str:
        """Return the role description for this agent."""
        return """Expert model profiler specializing in neural network architecture analysis, 
        dependency detection, and pruning safety assessment."""
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for profiling analysis."""
        
        model_info = context.get('model_info', {})
        dataset_info = context.get('dataset_info', {})
        
        return f"""You are an expert neural network profiling agent with deep knowledge of:
- Vision Transformer (ViT) and CNN architectures
- Layer dependencies and coupling constraints
- Pruning sensitivity analysis
- Hardware-aware optimization

TASK: Analyze the model architecture and provide comprehensive profiling results.

MODEL CONTEXT:
- Architecture: {model_info.get('architecture_type', 'unknown')}
- Pruning Strategy: {model_info.get('pruning_strategy', 'unknown')}
- Sensitive Layers: {model_info.get('sensitive_layers', [])}

DATASET CONTEXT:
- Complexity: {dataset_info.get('complexity', 'unknown')}
- Classes: {dataset_info.get('num_classes', 'unknown')}
- Pruning Difficulty: {dataset_info.get('pruning_difficulty', 'unknown')}

SAFETY REQUIREMENTS:
- Apply {dataset_info.get('recommended_approach', 'conservative')} pruning approach
- Respect safety limits: MLP ≤ {dataset_info.get('safety_limits', {}).get('max_mlp_pruning', 0.2):.1%}, Attention ≤ {dataset_info.get('safety_limits', {}).get('max_attention_pruning', 0.15):.1%}
- Identify critical layers that must be preserved

Your analysis should be thorough but concise, focusing on actionable insights for the pruning workflow."""
    
    def execute(self, input_data) -> Dict[str, Any]:
        """Execute profiling with precomputation and caching."""
        
        # Handle both PruningState objects and dictionary contexts
        if hasattr(input_data, 'model'):
            # It's a PruningState object
            model = input_data.model
            model_name = input_data.model_name
            dataset = input_data.dataset
            context = {
                'input_size': getattr(input_data, 'input_size', 224),
                'num_classes': getattr(input_data, 'num_classes', 1000),
                'target_ratio': getattr(input_data, 'target_ratio', 0.5)
            }
        else:
            # It's a dictionary context
            model = input_data.get('model')
            model_name = input_data.get('model_name', 'unknown')
            dataset = input_data.get('dataset', 'unknown')
            context = input_data
        
        if model is None:
            return {
                'success': False,
                'error': 'No model provided for profiling',
                'agent_name': self.agent_name
            }
        
        # Check cache first
        cache_key = f"{model_name}_{dataset}"
        if self.enable_caching and cache_key in self.profile_cache:
            logger.info(f"📋 Using cached profile for {model_name}")
            cached_profile = self.profile_cache[cache_key]
            
            # Still run LLM analysis for context-specific insights
            llm_analysis = self._run_llm_analysis(cached_profile.to_dict(), context)
            
            return {
                'success': True,
                'agent_name': self.agent_name,
                'profile': cached_profile.to_dict(),
                'llm_analysis': llm_analysis,
                'cached': True,
                'timestamp': cached_profile.to_dict().get('timestamp')
            }
        
        # Run full profiling
        logger.info(f"🔍 Running full profiling for {model_name}")
        
        with self.profiler.timer("full_profiling"):
            profile = self._run_comprehensive_profiling(model, model_name, dataset, context)
        
        # Cache the profile
        if self.enable_caching:
            self.profile_cache[cache_key] = profile
        
        # Run LLM analysis
        llm_analysis = self._run_llm_analysis(profile.to_dict(), context)
        
        return {
            'success': True,
            'agent_name': self.agent_name,
            'profile': profile.to_dict(),
            'llm_analysis': llm_analysis,
            'cached': False,
            'timing': self.profiler.get_summary() if hasattr(self.profiler, 'get_summary') else {}
        }
    
    def _run_comprehensive_profiling(self, model: nn.Module, model_name: str, 
                                   dataset: str, context: Dict[str, Any]) -> ModelProfile:
        """Run comprehensive model profiling with all analyses."""
        
        # Basic model analysis
        with self.profiler.timer("basic_analysis"):
            total_params = sum(p.numel() for p in model.parameters())
            total_macs = compute_macs(model, (1, 3, context.get('input_size', 224), context.get('input_size', 224)))
            architecture_type = self._detect_architecture_type(model, model_name)
        
        # Layer analysis
        with self.profiler.timer("layer_analysis"):
            layer_info = self._analyze_layers(model, architecture_type)
        
        # Dependency analysis
        with self.profiler.timer("dependency_analysis"):
            if self.dependency_analyzer is None:
                self.dependency_analyzer = DependencyAnalyzer(model)
            dependency_graph = self.dependency_analyzer.get_dependency_graph()
            coupling_constraints = self.dependency_analyzer.get_coupling_constraints()
        
        # Isomorphic analysis
        with self.profiler.timer("isomorphic_analysis"):
            if self.isomorphic_analyzer is None:
                self.isomorphic_analyzer = IsomorphicAnalyzer(model)
            isomorphic_groups = self.isomorphic_analyzer.create_isomorphic_groups(0.5)  # Default ratio
        
        # Safety analysis
        with self.profiler.timer("safety_analysis"):
            safety_limits = self._compute_safety_limits(architecture_type, dataset)
            recommended_ratios = self._compute_recommended_ratios(architecture_type, dataset)
        
        # Performance analysis
        with self.profiler.timer("performance_analysis"):
            memory_usage = self._estimate_memory_usage(model)
            inference_time = self._estimate_inference_time(model, context.get('input_size', 224))
        
        # Importance criterion recommendation
        importance_criterion = self._recommend_importance_criterion(architecture_type, dataset)
        
        return ModelProfile(
            model_name=model_name,
            architecture_type=architecture_type,
            total_params=total_params,
            total_macs=total_macs,
            layer_count=layer_info['layer_count'],
            prunable_layers=layer_info['prunable_layers'],
            critical_layers=layer_info['critical_layers'],
            sensitive_layers=layer_info['sensitive_layers'],
            dependency_graph=dependency_graph,
            isomorphic_groups={k: v.to_dict() if hasattr(v, 'to_dict') else str(v) 
                             for k, v in isomorphic_groups.items()},
            coupling_constraints=coupling_constraints,
            recommended_ratios=recommended_ratios,
            safety_limits=safety_limits,
            importance_criterion=importance_criterion,
            memory_usage_mb=memory_usage,
            inference_time_ms=inference_time
        )
    
    def _detect_architecture_type(self, model: nn.Module, model_name: str) -> str:
        """Detect model architecture type with improved accuracy."""
        
        # Check model name first
        model_name_lower = model_name.lower()
        
        if any(x in model_name_lower for x in ['deit', 'vit', 'swin', 'beit']):
            return 'vision_transformer'
        elif any(x in model_name_lower for x in ['resnet', 'resnext', 'densenet']):
            return 'cnn'
        elif any(x in model_name_lower for x in ['convnext', 'efficientnet']):
            return 'modern_cnn'
        elif any(x in model_name_lower for x in ['mobilenet', 'shufflenet']):
            return 'mobile_cnn'
        
        # Analyze model structure
        has_attention = False
        has_conv = False
        has_linear = False
        
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                has_attention = True
            elif isinstance(module, nn.Conv2d):
                has_conv = True
            elif isinstance(module, nn.Linear):
                has_linear = True
        
        if has_attention and has_linear:
            return 'vision_transformer'
        elif has_conv and not has_attention:
            return 'cnn'
        elif has_conv and has_attention:
            return 'hybrid'
        else:
            return 'unknown'
    
    def _analyze_layers(self, model: nn.Module, architecture_type: str) -> Dict[str, Any]:
        """Analyze model layers for pruning characteristics."""
        
        prunable_layers = []
        critical_layers = []
        sensitive_layers = []
        layer_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_count += 1
                
                # Determine if layer is prunable
                if self._is_prunable_layer(name, module, architecture_type):
                    prunable_layers.append(name)
                
                # Identify critical layers (should not be pruned)
                if self._is_critical_layer(name, module, architecture_type):
                    critical_layers.append(name)
                
                # Identify sensitive layers (prune carefully)
                if self._is_sensitive_layer(name, module, architecture_type):
                    sensitive_layers.append(name)
        
        return {
            'layer_count': layer_count,
            'prunable_layers': prunable_layers,
            'critical_layers': critical_layers,
            'sensitive_layers': sensitive_layers
        }
    
    def _is_prunable_layer(self, name: str, module: nn.Module, arch_type: str) -> bool:
        """Determine if a layer can be safely pruned."""
        
        # Never prune these layers
        if any(x in name.lower() for x in ['head', 'classifier', 'fc', 'embed']):
            return False
        
        # Architecture-specific rules
        if arch_type == 'vision_transformer':
            # ViT: Can prune MLP and attention layers
            return any(x in name.lower() for x in ['mlp', 'fc1', 'fc2', 'qkv', 'proj'])
        elif arch_type in ['cnn', 'modern_cnn']:
            # CNN: Can prune most conv layers except first and last
            if isinstance(module, nn.Conv2d):
                return not (name.endswith('.0') or 'downsample' in name)
        
        return True
    
    def _is_critical_layer(self, name: str, module: nn.Module, arch_type: str) -> bool:
        """Identify critical layers that should never be pruned."""
        
        critical_patterns = [
            'head', 'classifier', 'fc', 'embed', 'pos_embed', 
            'cls_token', 'patch_embed', 'stem'
        ]
        
        return any(pattern in name.lower() for pattern in critical_patterns)
    
    def _is_sensitive_layer(self, name: str, module: nn.Module, arch_type: str) -> bool:
        """Identify layers that are sensitive to pruning."""
        
        if arch_type == 'vision_transformer':
            # First and last blocks are more sensitive
            if 'blocks.0.' in name or 'blocks.11.' in name:
                return True
        elif arch_type in ['cnn', 'modern_cnn']:
            # Early conv layers are sensitive
            if isinstance(module, nn.Conv2d) and any(x in name for x in ['conv1', 'layer1']):
                return True
        
        return False
    
    def _compute_safety_limits(self, arch_type: str, dataset: str) -> Dict[str, float]:
        """Compute architecture and dataset-specific safety limits."""
        
        base_limits = {
            'max_overall_pruning': 0.8,
            'max_layer_pruning': 0.5,
            'max_mlp_pruning': 0.3,
            'max_attention_pruning': 0.2
        }
        
        # Dataset-specific adjustments
        if dataset.lower() == 'imagenet':
            # More conservative for ImageNet
            base_limits.update({
                'max_overall_pruning': 0.6,
                'max_layer_pruning': 0.3,
                'max_mlp_pruning': 0.15,
                'max_attention_pruning': 0.10
            })
        elif dataset.lower() == 'cifar10':
            # More aggressive for CIFAR-10
            base_limits.update({
                'max_overall_pruning': 0.9,
                'max_layer_pruning': 0.8,
                'max_mlp_pruning': 0.7,
                'max_attention_pruning': 0.6
            })
        
        # Architecture-specific adjustments
        if arch_type == 'vision_transformer':
            # ViTs are more sensitive
            base_limits['max_mlp_pruning'] *= 0.8
            base_limits['max_attention_pruning'] *= 0.7
        
        return base_limits
    
    def _compute_recommended_ratios(self, arch_type: str, dataset: str) -> Dict[str, float]:
        """Compute recommended starting ratios."""
        
        if dataset.lower() == 'imagenet':
            if arch_type == 'vision_transformer':
                return {
                    'overall': 0.3,
                    'mlp': 0.1,
                    'attention': 0.05,
                    'increment': 0.05
                }
            else:
                return {
                    'overall': 0.4,
                    'conv': 0.2,
                    'fc': 0.3,
                    'increment': 0.1
                }
        else:  # CIFAR-10 or other
            return {
                'overall': 0.6,
                'mlp': 0.4,
                'attention': 0.3,
                'increment': 0.1
            }
    
    def _recommend_importance_criterion(self, arch_type: str, dataset: str) -> str:
        """Recommend importance criterion based on architecture and dataset."""
        
        if dataset.lower() == 'imagenet':
            return 'taylor'  # More accurate for complex datasets
        elif arch_type == 'vision_transformer':
            return 'taylor'  # Better for attention mechanisms
        else:
            return 'l1norm'  # Efficient for CNNs
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
    
    def _estimate_inference_time(self, model: nn.Module, input_size: int) -> float:
        """Estimate inference time in milliseconds."""
        
        # Rough estimation based on MACs
        macs = compute_macs(model, (1, 3, input_size, input_size))
        
        # Assume 1 GFLOP = 1ms on typical GPU (very rough estimate)
        return macs / 1e9
    
    def _run_llm_analysis(self, profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM analysis on the computed profile."""
        
        # Prepare context for LLM
        llm_context = context.copy()
        llm_context['computed_profile'] = profile
        
        # Generate focused prompt for LLM analysis
        user_prompt = f"""
COMPUTED PROFILE ANALYSIS:
- Architecture: {profile['architecture_type']}
- Total Parameters: {profile['total_params']:,}
- Total MACs: {profile['total_macs']/1e9:.2f}G
- Prunable Layers: {len(profile['prunable_layers'])}
- Critical Layers: {profile['critical_layers']}
- Recommended Importance: {profile['importance_criterion']}

SAFETY LIMITS:
- Max MLP Pruning: {profile['safety_limits']['max_mlp_pruning']:.1%}
- Max Attention Pruning: {profile['safety_limits']['max_attention_pruning']:.1%}

Based on this computed analysis, provide strategic insights for the pruning workflow:
1. Key architectural vulnerabilities
2. Recommended pruning sequence
3. Expected challenges and mitigation strategies
4. Performance vs accuracy trade-offs
        """
        
        try:
            # Query LLM for strategic analysis
            system_prompt = self.get_system_prompt(llm_context)
            llm_response = self._query_llm_with_retries(system_prompt, user_prompt)
            
            # Parse response
            agent_response = self.parse_llm_response(llm_response, llm_context)
            
            return agent_response.to_dict()
            
        except Exception as e:
            logger.warning(f"⚠️ LLM analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': 'Using computed profile only'
            }
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response into structured format."""
        
        return AgentResponse(
            success=True,
            reasoning=response,
            recommendations={
                'strategic_insights': response,
                'computed_profile': context.get('computed_profile', {})
            },
            confidence=0.8,
            safety_checks={'profile_computed': True},
            warnings=[],
            timestamp=context.get('timestamp', ''),
            agent_name=self.agent_name
        )