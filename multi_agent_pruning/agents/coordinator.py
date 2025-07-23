"""
Agent Coordinator for Multi-Agent LLM Pruning Workflow

Implements the exact workflow shown in the user's diagram:
START → Profiling Agent → Master Agent → Analysis Agent → Pruning Agent → Fine-Tuning Agent → Evaluation Agent

This coordinator manages the sequential execution of agents and handles
the flow of information between them.
"""

import logging
from typing import Dict, Any, Optional, List
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
        self.profiler = TimingProfiler()
        
        # Initialize all agents
        self.agents = {
            'profiling': ProfilingAgent(self.config.get('profiling_agent', {})),
            'master': MasterAgent(self.config.get('master_agent', {})),
            'analysis': AnalysisAgent(self.config.get('analysis_agent', {})),
            'pruning': PruningAgent(self.config.get('pruning_agent', {})),
            'finetuning': FinetuningAgent(self.config.get('finetuning_agent', {})),
            'evaluation': EvaluationAgent(self.config.get('evaluation_agent', {}))
        }
        
        # Workflow state
        self.current_step = 0
        self.workflow_steps = [
            'profiling', 'master', 'analysis', 'pruning', 'finetuning', 'evaluation'
        ]
        
        logger.info("🤖 Agent Coordinator initialized with 6-agent workflow")
    
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
                    'duration': self.profiler.timings[f"agent_{agent_name}"][-1],
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
            # Prepare agent input based on current state
            agent_input = self._prepare_agent_input(agent_name, state)
            
            # Execute agent
            logger.debug(f"🔧 Executing {agent_name} agent with input keys: {list(agent_input.keys())}")
            result = agent.execute(agent_input)
            
            # Validate result
            if not isinstance(result, dict):
                raise ValueError(f"Agent {agent_name} returned invalid result type: {type(result)}")
            
            result['success'] = True
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_name} execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'agent': agent_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_agent_input(self, agent_name: str, state: PruningState) -> Dict[str, Any]:
        """Prepare input for each agent based on workflow state."""
        
        base_input = {
            'model_name': state.model_name,
            'dataset': state.dataset,
            'target_ratio': state.target_ratio,
            'num_classes': state.num_classes,
            'input_size': state.input_size,
            'query': state.query
        }
        
        if agent_name == 'profiling':
            # Profiling Agent: Initial model analysis
            return {
                **base_input,
                'model': state.model,
                'data_path': state.data_path
            }
        
        elif agent_name == 'master':
            # Master Agent: Strategic coordination
            return {
                **base_input,
                'profile_results': state.profile_results,
                'history': state.history,
                'revision_number': state.revision_number,
                'attempted_ratios': state.attempted_pruning_ratios
            }
        
        elif agent_name == 'analysis':
            # Analysis Agent: Architecture-specific recommendations
            return {
                **base_input,
                'profile_results': state.profile_results,
                'master_results': state.master_results,
                'revision_number': state.revision_number
            }
        
        elif agent_name == 'pruning':
            # Pruning Agent: Execute pruning
            return {
                **base_input,
                'model': state.model,
                'analysis_results': state.analysis_results,
                'master_directives': state.master_results.get('directives', {}),
                'importance_criterion': state.analysis_results.get('importance_criterion', 'taylor'),
                'pruning_config': state.analysis_results.get('pruning_config', {})
            }
        
        elif agent_name == 'finetuning':
            # Fine-tuning Agent: Recover performance
            return {
                **base_input,
                'pruned_model': state.pruning_results.get('pruned_model'),
                'pruning_info': state.pruning_results.get('pruning_info', {}),
                'zero_shot_accuracy': state.pruning_results.get('zero_shot_accuracy', 0),
                'target_accuracy': state.master_results.get('target_accuracy', 75.0)
            }
        
        elif agent_name == 'evaluation':
            # Evaluation Agent: Final assessment
            return {
                **base_input,
                'original_model': state.model,
                'pruned_model': state.pruning_results.get('pruned_model'),
                'finetuned_model': state.fine_tuning_results.get('finetuned_model'),
                'pruning_info': state.pruning_results.get('pruning_info', {}),
                'finetuning_info': state.fine_tuning_results.get('finetuning_info', {})
            }
        
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
    
    def _update_state_from_agent_result(self, state: PruningState, 
                                      agent_name: str, result: Dict[str, Any]):
        """Update workflow state with agent results."""
        
        if agent_name == 'profiling':
            state.profile_results = result
        elif agent_name == 'master':
            state.master_results = result
        elif agent_name == 'analysis':
            state.analysis_results = result
        elif agent_name == 'pruning':
            state.pruning_results = result
            # Track attempted ratio
            if 'achieved_ratio' in result:
                state.attempted_pruning_ratios.append(result['achieved_ratio'])
        elif agent_name == 'finetuning':
            state.fine_tuning_results = result
        elif agent_name == 'evaluation':
            state.evaluation_results = result
    
    def _get_timing_summary(self) -> Dict[str, Any]:
        """Get timing summary for the workflow."""
        
        timing_summary = {}
        total_time = 0
        
        for agent_name in self.workflow_steps:
            agent_key = f"agent_{agent_name}"
            if agent_key in self.profiler.timings:
                times = self.profiler.timings[agent_key]
                agent_time = times[-1] if times else 0
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
            print(f"\n🎯 FINAL RESULTS:")
            print("-" * 20)
            print(f"Final Accuracy: {eval_results.get('final_accuracy', 0):.2f}%")
            print(f"MACs Reduction: {eval_results.get('macs_reduction', 0):.1%}")
            print(f"Params Reduction: {eval_results.get('params_reduction', 0):.1%}")
            print(f"Achieved MACs (G): {eval_results.get('final_macs_g', 0):.2f}")
            print(f"Achieved Params (M): {eval_results.get('final_params_m', 0):.2f}")
        
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
                accuracy = eval_results.get('final_accuracy', 0)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_result = result
            
            # Check Master Agent's decision to continue
            master_results = result['agent_outputs'].get('master', {})
            should_continue = master_results.get('continue_iterations', False)
            
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

