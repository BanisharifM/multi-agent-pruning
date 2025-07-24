"""
Agent Coordinator for Multi-Agent LLM Pruning Workflow - UPDATED VERSION

Implements the exact workflow shown in the user's diagram:
START → Profiling Agent → Master Agent → Analysis Agent → Pruning Agent → Fine-Tuning Agent → Evaluation Agent

This coordinator manages the sequential execution of agents and handles
the flow of information between them.

FIXES APPLIED:
- Fix 1: Coordinator State Management Unification
- Fix 3: Configuration Propagation System  
- Fix 4: Safety Configuration Enforcement
- Fix 5: Precomputation Activation
- Fix 6: Error Handling and Diagnostics
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
import json
from datetime import datetime

from .base_agent import BaseAgent
from .profiling_agent import ProfilingAgent
from .master_agent import MasterAgent
from .analysis_agent import AnalysisAgent
from .pruning_agent import PruningAgent
from .finetuning_agent import FinetuningAgent
from .evaluation_agent import EvaluationAgent
from ..core.state_manager import PruningState, StateManager
from ..utils.profiler import TimingProfiler

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """
    Coordinates the multi-agent pruning workflow following the user's diagram.
    
    Workflow:
    1. Profiling Agent: Analyzes model architecture and dependencies
    2. Master Agent: Makes high-level strategic decisions
    3. Analysis Agent: Provides architecture-specific pruning recommendations
    4. Pruning Agent: Executes the pruning based on recommendations
    5. Fine-Tuning Agent: Recovers model performance through fine-tuning
    6. Evaluation Agent: Measures final accuracy, MACs, and parameter reduction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.profiler = TimingProfiler()
        
        # Initialize all agents with proper configuration propagation
        self._initialize_agents()

        # Workflow state
        self.current_step = 0
        self.workflow_steps = [
            'profiling', 'master', 'analysis', 'pruning', 'finetuning', 'evaluation'
        ]
        
        logger.info("🤖 Agent Coordinator initialized with 6-agent workflow")
    
    def _initialize_agents(self):
        """Initialize all agents with proper constructor parameters and configuration propagation."""
        
        # Extract agent configurations
        agent_configs = self.config.get('agents', {})
        global_agent_config = agent_configs.get('global', {})
        
        # Create LLM client for agents
        llm_client = self._create_llm_client()
        
        # Initialize agents with standardized pattern
        self.agents = {
            'profiling': ProfilingAgent(
                config={**global_agent_config, **agent_configs.get('profiling_agent', {})},
                llm_client=llm_client,
                profiler=self.profiler
            ),
            'master': MasterAgent(
                config={**global_agent_config, **agent_configs.get('master_agent', {})},
                llm_client=llm_client,
                profiler=self.profiler
            ),
            'analysis': AnalysisAgent(
                config={**global_agent_config, **agent_configs.get('analysis_agent', {})},
                llm_client=llm_client,
                profiler=self.profiler
            ),
            'pruning': PruningAgent(
                config={**global_agent_config, **agent_configs.get('pruning_agent', {})},
                llm_client=llm_client,
                profiler=self.profiler
            ),
            'finetuning': FinetuningAgent(
                config={**global_agent_config, **agent_configs.get('finetuning_agent', {})},
                llm_client=llm_client,
                profiler=self.profiler
            ),
            'evaluation': EvaluationAgent(
                config={**global_agent_config, **agent_configs.get('evaluation_agent', {})},
                llm_client=llm_client,
                profiler=self.profiler
            )
        }

        # Create individual agent references for backward compatibility
        self.profiling_agent = self.agents['profiling']
        self.master_agent = self.agents['master']
        self.analysis_agent = self.agents['analysis']
        self.pruning_agent = self.agents['pruning']
        self.finetuning_agent = self.agents['finetuning']
        self.evaluation_agent = self.agents['evaluation']
        
        logger.info("🤖 All agents initialized with standardized pattern")

    def _create_llm_client(self):
        """Create LLM client based on configuration."""
        llm_config = self.config.get('llm', {})
        
        if llm_config.get('enabled', True):
            try:
                from openai import OpenAI
                return OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    base_url=llm_config.get('base_url'),
                    timeout=llm_config.get('timeout', 30)
                )
            except ImportError:
                logger.warning("OpenAI client not available")
                return None
        else:
            return None

    def _ensure_profiler_availability(self):
        """Ensure that profiler is available for agents that need it."""
        
        if self.profiler is None:
            logger.warning("No profiler provided, creating default profiler")
            try:
                from ..utils.profiler import TimingProfiler
                self.profiler = TimingProfiler(enabled=True)
            except ImportError:
                logger.error("Cannot create profiler, performance monitoring disabled")
                self.profiler = None
        
        # Validate profiler functionality
        if self.profiler is not None:
            try:
                with self.profiler.timer("profiler_test"):
                    pass
                logger.info("✅ Profiler validation successful")
            except Exception as e:
                logger.warning(f"Profiler validation failed: {e}, disabling profiler")
                self.profiler = None

    def run_pruning(self, model, train_loader, val_loader, test_loader):
        """
        Run the complete multi-agent pruning workflow with proper state management.
        
        Args:
            model: PyTorch model to prune
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            
        Returns:
            Dict containing pruning results and metrics
        """
        
        self.logger.info("🚀 Starting multi-agent pruning workflow")
        
        try:
            # Create proper state using StateManager
            state_manager = StateManager(cache_dir=self.config.get('cache_dir', './cache'))
            
            # Extract configuration parameters
            model_name = self.config.get('model', {}).get('name', 'unknown')
            dataset_name = self.config.get('dataset', {}).get('name', 'imagenet')
            target_ratio = self.config.get('pruning', {}).get('target_ratio', 0.5)
            
            # Create comprehensive query
            query = f"Prune {model_name} to {target_ratio:.1%} parameter reduction on {dataset_name}"
            
            # Create proper state object
            state = state_manager.create_state(
                query=query,
                model_name=model_name,
                dataset=dataset_name,
                target_ratio=target_ratio,
                num_classes=self.config.get('dataset', {}).get('num_classes', 1000),
                input_size=self.config.get('dataset', {}).get('input_size', 224),
                data_path=self.config.get('dataset', {}).get('data_path', '')
            )
            
            # Set the model reference properly
            state.model = model
            
            # Store data loaders in state for agent access
            state.train_loader = train_loader
            state.val_loader = val_loader
            state.test_loader = test_loader
            
            # Apply safety constraints
            safety_constraints = self._apply_safety_constraints(state)
            
            # Ensure precomputation is active
            self._ensure_precomputation_active(state)
            
            # Validate workflow state
            is_valid, errors = self._validate_workflow_state(state)
            if not is_valid:
                raise RuntimeError(f"Workflow state validation failed: {errors}")
            
            # Use the unified workflow
            return self.run_pruning_workflow(state)
            
        except Exception as e:
            self.logger.error(f"❌ Pruning workflow failed: {str(e)}")
            raise

    def _apply_safety_constraints(self, state: PruningState) -> Dict[str, Any]:
        """Apply safety constraints based on configuration and dataset."""
        
        dataset_name = state.dataset.lower()
        safety_config = self.config.get('datasets', {}).get(dataset_name, {}).get('safety_limits', {})
        
        # Apply dataset-specific safety limits
        safety_constraints = {
            'max_mlp_pruning': safety_config.get('max_mlp_pruning', 0.15),
            'max_attention_pruning': safety_config.get('max_attention_pruning', 0.10),
            'max_overall_pruning': safety_config.get('max_overall_pruning', 0.60),
            'min_accuracy_threshold': safety_config.get('min_accuracy_threshold', 0.40)
        }
        
        # Validate target ratio against safety limits
        if state.target_ratio > safety_constraints['max_overall_pruning']:
            logger.warning(f"Target ratio {state.target_ratio:.1%} exceeds safety limit {safety_constraints['max_overall_pruning']:.1%}")
            state.target_ratio = safety_constraints['max_overall_pruning']
        
        return safety_constraints

    def _ensure_precomputation_active(self, state: PruningState):
        """Ensure that precomputation is properly activated for the state."""
        
        if not hasattr(state, '_precomputed_cache') or not state._precomputed_cache:
            # Precomputation was not activated, activate it now
            state_manager = StateManager(cache_dir=state._cache_dir or './cache')
            state_manager._schedule_precomputation(state)
            
            logger.info("🚀 Precomputation activated for existing state")
        
        # Verify precomputation status
        precomputation_status = self._get_precomputation_status(state)
        missing_precomputation = [key for key, available in precomputation_status.items() if not available]
        
        if missing_precomputation:
            logger.warning(f"⚠️ Missing precomputation for: {missing_precomputation}")
        else:
            logger.info("✅ All precomputation data available")

    def _get_precomputation_status(self, state: PruningState) -> Dict[str, bool]:
        """Get status of precomputation data."""
        
        if not hasattr(state, '_precomputed_cache'):
            return {}
        
        return {
            'model_analysis': 'model_analysis' in state._precomputed_cache,
            'dependency_analysis': 'dependency_analysis' in state._precomputed_cache,
            'importance_scores': 'importance_scores' in state._precomputed_cache,
            'dataset_stats': 'dataset_stats' in state._precomputed_cache
        }

    def _validate_workflow_state(self, state: PruningState) -> Tuple[bool, List[str]]:
        """Comprehensive validation of workflow state with detailed diagnostics."""
        
        errors = []
        
        # Validate required fields
        required_fields = ['model', 'model_name', 'dataset', 'target_ratio']
        for field in required_fields:
            if not hasattr(state, field) or getattr(state, field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate model reference
        if hasattr(state, 'model') and state.model is not None:
            if not hasattr(state.model, 'parameters'):
                errors.append("Model object does not appear to be a valid PyTorch model")
        
        # Validate configuration consistency
        if state.target_ratio <= 0 or state.target_ratio >= 1:
            errors.append(f"Invalid target ratio: {state.target_ratio} (must be between 0 and 1)")
        
        # Validate precomputation status
        if hasattr(state, '_precomputed_cache'):
            cache_status = self._get_precomputation_status(state)
            missing_cache = [key for key, available in cache_status.items() if not available]
            if missing_cache:
                errors.append(f"Missing precomputed data: {missing_cache}")
        
        return len(errors) == 0, errors
    
    def _extract_response_data(self, response, operation_name: str):
        """Extract data from agent response (handles both dict and AgentResponse)."""
        
        if hasattr(response, 'success'):
            # It's an AgentResponse object
            success = response.success
            message = response.message
            data = response.data
        else:
            # It's a dictionary
            success = response.get('success', False)
            message = response.get('message', f'{operation_name} completed')
            data = response
        
        return success, message, data
    
    def run_pruning_workflow(self, state: PruningState) -> Dict[str, Any]:
        """
        Execute the complete multi-agent pruning workflow.
        
        Args:
            state: Initial pruning state
            
        Returns:
            Complete workflow results including all agent outputs
        """
        
        logger.info("🚀 Starting multi-agent pruning workflow...")
        logger.info(f"📋 Target: {state.query}")
        logger.info(f"🎯 Model: {state.model_name}, Dataset: {state.dataset}")
        logger.info(f"🎯 Target Ratio: {state.target_ratio:.1%}")
        
        workflow_results = {
            'workflow_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'initial_state': asdict(state),
            'agent_outputs': {},
            'workflow_timeline': [],
            'final_state': None,
            'success': False,
            'error': None
        }
        
        try:
            # Execute workflow steps sequentially
            for step_idx, agent_name in enumerate(self.workflow_steps):
                self.current_step = step_idx
                
                logger.info(f"🔄 Step {step_idx + 1}/{len(self.workflow_steps)}: {agent_name.title()} Agent")
                
                with self.profiler.timer(f"agent_{agent_name}"):
                    step_result = self._execute_agent_step(agent_name, state)
                
                # Record step in timeline
                workflow_results['workflow_timeline'].append({
                    'step': step_idx + 1,
                    'agent': agent_name,
                    'timestamp': datetime.now().isoformat(),
                    'duration': self.profiler.timings[f"agent_{agent_name}"][-1].duration if f"agent_{agent_name}" in self.profiler.timings else 0,
                    'success': step_result.get('success', False),
                    'error': step_result.get('error')
                })
                
                # Store agent output
                workflow_results['agent_outputs'][agent_name] = step_result
                
                # Check if step failed
                if not step_result.get('success', False):
                    error_msg = f"Agent {agent_name} failed: {step_result.get('error', 'Unknown error')}"
                    logger.error(f"❌ {error_msg}")
                    workflow_results['error'] = error_msg
                    break
                
                # Update state with agent results
                self._update_state_from_agent_result(state, agent_name, step_result)
                
                logger.info(f"✅ {agent_name.title()} Agent completed successfully")
            
            # Check if workflow completed successfully
            if self.current_step == len(self.workflow_steps) - 1:
                workflow_results['success'] = True
                workflow_results['final_state'] = asdict(state)
                logger.info("🎉 Multi-agent workflow completed successfully!")
            
        except Exception as e:
            error_msg = f"Workflow failed with exception: {str(e)}"
            logger.error(f"💥 {error_msg}")
            workflow_results['error'] = error_msg
            workflow_results['success'] = False
        
        # Add timing summary
        workflow_results['timing_summary'] = self._get_timing_summary()
        
        # Print workflow summary
        self._print_workflow_summary(workflow_results)
        
        return workflow_results
    
    def _execute_agent_step(self, agent_name: str, state: PruningState) -> Dict[str, Any]:
        """Execute a single agent step in the workflow."""
        
        agent = self.agents[agent_name]
        
        try:
            # Execute agent with state object directly
            logger.debug(f"🔧 Executing {agent_name} agent with PruningState")
            result = agent.execute(state)
            
            # Extract response data (handles both dict and AgentResponse)
            success, message, data = self._extract_response_data(result, agent_name)
            
            # Return standardized format
            return {
                'success': success,
                'message': message,
                'data': data,
                'agent': agent_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_name} execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': agent_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_state_from_agent_result(self, state: PruningState, 
                                      agent_name: str, result: Dict[str, Any]):
        """Update workflow state with agent results."""
        
        # Extract data from result
        data = result.get('data', result)
        
        if agent_name == 'profiling':
            state.profile_results = data
        elif agent_name == 'master':
            state.master_results = data
        elif agent_name == 'analysis':
            state.analysis_results = data
        elif agent_name == 'pruning':
            state.pruning_results = data
            # Track attempted ratio
            if 'achieved_ratio' in data:
                state.attempted_pruning_ratios.append(data['achieved_ratio'])
        elif agent_name == 'finetuning':
            state.fine_tuning_results = data
        elif agent_name == 'evaluation':
            state.evaluation_results = data
    
    def _get_timing_summary(self) -> Dict[str, Any]:
        """Get timing summary for the workflow."""
        
        timing_summary = {}
        total_time = 0
        
        for agent_name in self.workflow_steps:
            agent_key = f"agent_{agent_name}"
            if agent_key in self.profiler.timings:
                times = self.profiler.timings[agent_key]
                agent_time = times[-1].duration if times else 0
                timing_summary[agent_name] = {
                    'duration': agent_time,
                    'percentage': 0  # Will be calculated after total
                }
                total_time += agent_time
        
        # Calculate percentages
        for agent_name in timing_summary:
            if total_time > 0:
                timing_summary[agent_name]['percentage'] = (
                    timing_summary[agent_name]['duration'] / total_time * 100
                )
        
        timing_summary['total_duration'] = total_time
        
        return timing_summary
    
    def _print_workflow_summary(self, workflow_results: Dict[str, Any]):
        """Print comprehensive workflow summary."""
        
        print("\n" + "="*80)
        print("🤖 MULTI-AGENT WORKFLOW SUMMARY")
        print("="*80)
        
        print(f"🆔 Workflow ID: {workflow_results['workflow_id']}")
        print(f"✅ Success: {workflow_results['success']}")
        
        if workflow_results.get('error'):
            print(f"❌ Error: {workflow_results['error']}")
        
        print(f"\n📊 AGENT EXECUTION TIMELINE:")
        print("-" * 50)
        
        for step in workflow_results['workflow_timeline']:
            status = "✅" if step['success'] else "❌"
            duration = step['duration']
            print(f"{status} Step {step['step']}: {step['agent'].title()} Agent - {duration:.2f}s")
            if step.get('error'):
                print(f"    Error: {step['error']}")
        
        # Print timing breakdown
        timing = workflow_results.get('timing_summary', {})
        if timing:
            print(f"\n⏱️ TIMING BREAKDOWN:")
            print("-" * 30)
            
            for agent_name in self.workflow_steps:
                if agent_name in timing:
                    duration = timing[agent_name]['duration']
                    percentage = timing[agent_name]['percentage']
                    print(f"{agent_name.title():<12}: {duration:>6.2f}s ({percentage:>5.1f}%)")
            
            total_duration = timing.get('total_duration', 0)
            print(f"{'Total':<12}: {total_duration:>6.2f}s (100.0%)")
        
        # Print key results if available
        if workflow_results['success'] and 'evaluation' in workflow_results['agent_outputs']:
            eval_results = workflow_results['agent_outputs']['evaluation']
            eval_data = eval_results.get('data', eval_results)
            print(f"\n🎯 FINAL RESULTS:")
            print("-" * 20)
            print(f"Final Accuracy: {eval_data.get('final_accuracy', 0):.2f}%")
            print(f"MACs Reduction: {eval_data.get('macs_reduction', 0):.1%}")
            print(f"Params Reduction: {eval_data.get('params_reduction', 0):.1%}")
            print(f"Achieved MACs (G): {eval_data.get('final_macs_g', 0):.2f}")
            print(f"Achieved Params (M): {eval_data.get('final_params_m', 0):.2f}")
        
        print("="*80)
    
    def run_iterative_workflow(self, state: PruningState, 
                             max_iterations: int = 3) -> Dict[str, Any]:
        """
        Run iterative workflow with Master Agent feedback loop.
        
        The Master Agent can decide to continue/stop based on results,
        implementing the revision system from the original code.
        """
        
        logger.info(f"🔄 Starting iterative workflow (max {max_iterations} iterations)")
        
        all_results = []
        best_result = None
        best_accuracy = 0
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Update state for this iteration
            state.revision_number = iteration
            
            # Run single workflow
            result = self.run_pruning_workflow(state)
            all_results.append(result)
            
            # Check if this is the best result so far
            if result['success']:
                eval_results = result['agent_outputs'].get('evaluation', {})
                eval_data = eval_results.get('data', eval_results)
                accuracy = eval_data.get('final_accuracy', 0)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_result = result
            
            # Check Master Agent's decision to continue
            master_results = result['agent_outputs'].get('master', {})
            master_data = master_results.get('data', master_results)
            should_continue = master_data.get('continue_iterations', False)
            
            if not should_continue:
                logger.info("🛑 Master Agent decided to stop iterations")
                break
            
            # Update history for next iteration
            if result['success']:
                state.history.append({
                    'iteration': iteration,
                    'results': result['agent_outputs'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Compile final results
        final_results = {
            'iterative_workflow_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_iterations': len(all_results),
            'all_iterations': all_results,
            'best_result': best_result,
            'best_accuracy': best_accuracy,
            'convergence_achieved': not should_continue,
            'final_state': asdict(state)
        }
        
        logger.info(f"🏁 Iterative workflow completed after {len(all_results)} iterations")
        logger.info(f"🏆 Best accuracy achieved: {best_accuracy:.2f}%")
        
        return final_results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        
        status = {
            'coordinator_initialized': True,
            'current_step': self.current_step,
            'workflow_steps': self.workflow_steps,
            'agents_status': {}
        }
        
        for agent_name, agent in self.agents.items():
            status['agents_status'][agent_name] = {
                'initialized': agent is not None,
                'type': type(agent).__name__,
                'config': getattr(agent, 'config', {})
            }
        
        return status
    
    def reset_workflow(self):
        """Reset workflow state for new execution."""
        
        self.current_step = 0
        self.profiler = TimingProfiler()  # Reset timing
        
        # Reset individual agents if they have reset methods
        for agent in self.agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()
        
        logger.info("🔄 Workflow reset for new execution")