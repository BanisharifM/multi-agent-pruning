"""
Base Agent Class for Multi-Agent LLM Pruning

This implements the foundation for the user's unique multi-agent LLM approach,
where each agent uses LLM reasoning to make intelligent decisions about
neural network pruning.

Key Features:
- LLM-guided decision making
- Dataset-specific reasoning
- Safety constraints and limits
- Iterative refinement capability
- Comprehensive logging and reasoning traces
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import openai
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass  

@dataclass
class AgentResponse:
    """Structured response from an LLM agent."""
    
    success: bool
    reasoning: str
    recommendations: Dict[str, Any]
    confidence: float
    safety_checks: Dict[str, bool]
    warnings: List[str]
    timestamp: str
    agent_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'reasoning': self.reasoning,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'safety_checks': self.safety_checks,
            'warnings': self.warnings,
            'timestamp': self.timestamp,
            'agent_name': self.agent_name
        }

class BaseAgent(ABC):
    """
    Base class for all LLM-guided agents in the multi-agent pruning system.
    
    Each agent specializes in a specific aspect of the pruning workflow
    and uses LLM reasoning to make intelligent decisions.
    """
    
    def __init__(self, agent_name: str, llm_client=None, profiler=None):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Name/type of the agent
            llm_client: LLM client for reasoning (optional)
            profiler: Performance profiler (optional)
        """
        self.agent_name = agent_name
        self.llm_client = llm_client
        self.profiler = profiler
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        
        # Initialize LLM client if not provided
        if self.llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
                self.llm_model = "gpt-4o-mini"  # Default model
            except ImportError:
                self.logger.warning("OpenAI client not available, LLM features disabled")
                self.llm_client = None
                self.llm_model = None
        else:
            self.llm_model = "gpt-4o-mini"  # Default model
        
        # Initialize profiler if not provided
        if self.profiler is None:
            try:
                from ..utils.profiler import TimingProfiler
                self.profiler = TimingProfiler()
            except ImportError:
                self.logger.warning("Profiler not available")
                self.profiler = None
        
        self.logger.info(f"🤖 {agent_name} initialized with model {self.llm_model}")

    @abstractmethod
    def get_agent_role(self) -> str:
        """Return the role description for this agent."""
        pass
    
    @abstractmethod
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """Generate system prompt for this agent given context."""
        pass
    
    @abstractmethod
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response into structured agent response."""
        pass
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task using LLM reasoning.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent's response and recommendations
        """
        
        logger.info(f"🔄 Executing {self.agent_name}...")
        
        try:
            # Prepare context
            context = self._prepare_context(input_data)
            
            # Generate system prompt
            system_prompt = self.get_system_prompt(context)
            
            # Generate user prompt with input data
            user_prompt = self._generate_user_prompt(context)
            
            # Query LLM with retries
            llm_response = self._query_llm_with_retries(system_prompt, user_prompt)
            
            # Parse response
            agent_response = self.parse_llm_response(llm_response, context)
            
            # Apply safety checks
            if self.enable_safety_checks:
                agent_response = self._apply_safety_checks(agent_response, context)
            
            # Log reasoning trace
            self._log_reasoning_trace(agent_response, context)
            
            # Convert to output format
            output = self._format_output(agent_response, context)
            
            logger.info(f"✅ {self.agent_name} completed successfully")
            return output
            
        except Exception as e:
            error_msg = f"{self.agent_name} execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'agent_name': self.agent_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for LLM reasoning."""
        
        context = input_data.copy()
        
        # Add agent-specific context
        context['agent_name'] = self.agent_name
        context['agent_role'] = self.agent_role
        context['safety_multiplier'] = self.safety_multiplier
        
        # Add dataset-specific information
        dataset = context.get('dataset', 'unknown')
        context['dataset_info'] = self._get_dataset_info(dataset)
        
        # Add model-specific information
        model_name = context.get('model_name', 'unknown')
        context['model_info'] = self._get_model_info(model_name)
        
        return context
    
    def _get_dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get dataset-specific information for reasoning."""
        
        dataset_info = {
            'imagenet': {
                'num_classes': 1000,
                'complexity': 'very high',
                'typical_accuracy': '70-85%',
                'pruning_difficulty': 'very hard',
                'safety_limits': {
                    'max_mlp_pruning': 0.15,
                    'max_attention_pruning': 0.10,
                    'min_accuracy_threshold': 0.40
                },
                'recommended_approach': 'very conservative',
                'importance_criterion': 'taylor',
                'fine_tuning_epochs': '3-5'
            },
            'cifar10': {
                'num_classes': 10,
                'complexity': 'moderate',
                'typical_accuracy': '90-95%',
                'pruning_difficulty': 'moderate',
                'safety_limits': {
                    'max_mlp_pruning': 0.80,
                    'max_attention_pruning': 0.70,
                    'min_accuracy_threshold': 0.70
                },
                'recommended_approach': 'moderate',
                'importance_criterion': 'l1norm',
                'fine_tuning_epochs': '1-3'
            }
        }
        
        return dataset_info.get(dataset.lower(), {
            'num_classes': 1000,
            'complexity': 'unknown',
            'typical_accuracy': 'unknown',
            'pruning_difficulty': 'unknown',
            'safety_limits': {
                'max_mlp_pruning': 0.20,
                'max_attention_pruning': 0.15,
                'min_accuracy_threshold': 0.50
            },
            'recommended_approach': 'conservative',
            'importance_criterion': 'taylor',
            'fine_tuning_epochs': '3-5'
        })
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific information for reasoning."""
        
        model_info = {}
        
        if 'resnet' in model_name.lower():
            model_info = {
                'architecture_type': 'cnn',
                'pruning_strategy': 'structured_channels',
                'sensitive_layers': ['conv1', 'fc'],
                'pruning_difficulty': 'moderate'
            }
        elif 'deit' in model_name.lower() or 'vit' in model_name.lower():
            model_info = {
                'architecture_type': 'vision_transformer',
                'pruning_strategy': 'attention_mlp',
                'sensitive_layers': ['patch_embed', 'head'],
                'pruning_difficulty': 'hard'
            }
        elif 'convnext' in model_name.lower():
            model_info = {
                'architecture_type': 'modern_cnn',
                'pruning_strategy': 'structured_channels',
                'sensitive_layers': ['stem', 'head'],
                'pruning_difficulty': 'moderate'
            }
        else:
            model_info = {
                'architecture_type': 'unknown',
                'pruning_strategy': 'conservative',
                'sensitive_layers': ['first', 'last'],
                'pruning_difficulty': 'unknown'
            }
        
        return model_info
    
    def _generate_user_prompt(self, context: Dict[str, Any]) -> str:
        """Generate user prompt with context information."""
        
        # Base prompt with context
        prompt = f"""
TASK CONTEXT:
- Model: {context.get('model_name', 'unknown')}
- Dataset: {context.get('dataset', 'unknown')}
- Target Pruning Ratio: {context.get('target_ratio', 0.5):.1%}
- Architecture Type: {context.get('model_info', {}).get('architecture_type', 'unknown')}

DATASET CHARACTERISTICS:
- Classes: {context.get('dataset_info', {}).get('num_classes', 'unknown')}
- Complexity: {context.get('dataset_info', {}).get('complexity', 'unknown')}
- Pruning Difficulty: {context.get('dataset_info', {}).get('pruning_difficulty', 'unknown')}

SAFETY CONSTRAINTS:
- Max MLP Pruning: {context.get('dataset_info', {}).get('safety_limits', {}).get('max_mlp_pruning', 0.2):.1%}
- Max Attention Pruning: {context.get('dataset_info', {}).get('safety_limits', {}).get('max_attention_pruning', 0.15):.1%}
- Min Accuracy Threshold: {context.get('dataset_info', {}).get('safety_limits', {}).get('min_accuracy_threshold', 0.5):.1%}

Please analyze this context and provide your expert recommendations.
        """
        
        return prompt.strip()
    
    def _query_llm_with_retries(self, system_prompt: str, user_prompt: str) -> str:
        """Query LLM with retry logic."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"🔄 LLM query attempt {attempt + 1}/{self.max_retries}")
                
                response = openai.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content
                
                # Store conversation history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'system_prompt': system_prompt[:200] + "...",
                    'user_prompt': user_prompt[:200] + "...",
                    'response': content[:200] + "...",
                    'attempt': attempt + 1
                })
                
                return content
                
            except Exception as e:
                logger.warning(f"⚠️ LLM query attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise e
        
        raise Exception("All LLM query attempts failed")
    
    def _apply_safety_checks(self, response: AgentResponse, 
                           context: Dict[str, Any]) -> AgentResponse:
        """Apply safety checks to agent response."""
        
        safety_limits = context.get('dataset_info', {}).get('safety_limits', {})
        
        # Check pruning ratio limits
        recommendations = response.recommendations
        
        if 'pruning_ratios' in recommendations:
            ratios = recommendations['pruning_ratios']
            
            # Check MLP pruning limit
            mlp_ratio = ratios.get('mlp', 0)
            max_mlp = safety_limits.get('max_mlp_pruning', 0.2)
            
            if mlp_ratio > max_mlp:
                response.warnings.append(f"MLP pruning ratio {mlp_ratio:.1%} exceeds limit {max_mlp:.1%}")
                recommendations['pruning_ratios']['mlp'] = max_mlp * self.safety_multiplier
                response.safety_checks['mlp_limit'] = False
            else:
                response.safety_checks['mlp_limit'] = True
            
            # Check attention pruning limit
            attention_ratio = ratios.get('attention', 0)
            max_attention = safety_limits.get('max_attention_pruning', 0.15)
            
            if attention_ratio > max_attention:
                response.warnings.append(f"Attention pruning ratio {attention_ratio:.1%} exceeds limit {max_attention:.1%}")
                recommendations['pruning_ratios']['attention'] = max_attention * self.safety_multiplier
                response.safety_checks['attention_limit'] = False
            else:
                response.safety_checks['attention_limit'] = True
        
        return response
    
    def _log_reasoning_trace(self, response: AgentResponse, context: Dict[str, Any]):
        """Log detailed reasoning trace for analysis."""
        
        trace = {
            'timestamp': datetime.now().isoformat(),
            'agent_name': self.agent_name,
            'context_summary': {
                'model': context.get('model_name'),
                'dataset': context.get('dataset'),
                'target_ratio': context.get('target_ratio')
            },
            'response_summary': {
                'success': response.success,
                'confidence': response.confidence,
                'num_warnings': len(response.warnings),
                'safety_checks_passed': sum(response.safety_checks.values())
            },
            'reasoning_excerpt': response.reasoning[:500] + "..." if len(response.reasoning) > 500 else response.reasoning
        }
        
        self.reasoning_traces.append(trace)
        
        # Log to file for detailed analysis
        if self.config.get('save_reasoning_traces', False):
            trace_file = f"./logs/reasoning_traces_{self.agent_name.lower()}.jsonl"
            os.makedirs(os.path.dirname(trace_file), exist_ok=True)
            
            with open(trace_file, 'a') as f:
                f.write(json.dumps(trace) + '\n')
    
    def _format_output(self, response: AgentResponse, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format agent response for workflow consumption."""
        
        return {
            'success': response.success,
            'agent_name': self.agent_name,
            'reasoning': response.reasoning,
            'recommendations': response.recommendations,
            'confidence': response.confidence,
            'safety_checks': response.safety_checks,
            'warnings': response.warnings,
            'timestamp': response.timestamp,
            'context_hash': hash(str(sorted(context.items())))
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for debugging."""
        return self.conversation_history
    
    def get_reasoning_traces(self) -> List[Dict[str, Any]]:
        """Get reasoning traces for analysis."""
        return self.reasoning_traces
    
    def reset(self):
        """Reset agent state for new execution."""
        self.conversation_history = []
        self.reasoning_traces = []
        logger.info(f"🔄 {self.agent_name} reset for new execution")

