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
        self.logger = logging.getLogger(__name__)
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

        # Create individual agent references
        self.profiling_agent = self.agents['profiling']
        self.master_agent = self.agents['master']
        self.analysis_agent = self.agents['analysis']
        self.pruning_agent = self.agents['pruning']
        self.finetuning_agent = self.agents['finetuning']
        self.evaluation_agent = self.agents['evaluation']

        # Workflow state
        self.current_step = 0
        self.workflow_steps = [
            'profiling', 'master', 'analysis', 'pruning', 'finetuning', 'evaluation'
        ]
        
        logger.info("🤖 Agent Coordinator initialized with 6-agent workflow")
    
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
            # Prepare agent input based on current state
            agent_input = self._prepare_agent_input(agent_name, state)
            
            # Execute agent
            logger.debug(f"🔧 Executing {agent_name} agent with input keys: {list(agent_input.keys())}")
            result = agent.execute(agent_input)
            
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
                'master_directives': state.master_results.get('directives', {}) if state.master_results else {},
                'importance_criterion': state.analysis_results.get('importance_criterion', 'taylor') if state.analysis_results else 'taylor',
                'pruning_config': state.analysis_results.get('pruning_config', {}) if state.analysis_results else {}
            }
        
        elif agent_name == 'finetuning':
            # Fine-tuning Agent: Recover performance
            return {
                **base_input,
                'pruned_model': state.pruning_results.get('pruned_model') if state.pruning_results else None,
                'pruning_info': state.pruning_results.get('pruning_info', {}) if state.pruning_results else {},
                'zero_shot_accuracy': state.pruning_results.get('zero_shot_accuracy', 0) if state.pruning_results else 0,
                'target_accuracy': state.master_results.get('target_accuracy', 75.0) if state.master_results else 75.0
            }
        
        elif agent_name == 'evaluation':
            # Evaluation Agent: Final assessment
            return {
                **base_input,
                'original_model': state.model,
                'pruned_model': state.pruning_results.get('pruned_model') if state.pruning_results else None,
                'finetuned_model': state.fine_tuning_results.get('finetuned_model') if state.fine_tuning_results else None,
                'pruning_info': state.pruning_results.get('pruning_info', {}) if state.pruning_results else {},
                'finetuning_info': state.fine_tuning_results.get('finetuning_info', {}) if state.fine_tuning_results else {}
            }
        
        else:
            raise ValueError(f"Unknown agent: {agent_name}")
    
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

# DELETE
    # def run_pruning(self, model, train_loader, val_loader, test_loader):
    #     """
    #     Run the complete multi-agent pruning workflow.
        
    #     Args:
    #         model: PyTorch model to prune
    #         train_loader: Training data loader
    #         val_loader: Validation data loader  
    #         test_loader: Test data loader
            
    #     Returns:
    #         Dict containing pruning results and metrics
    #     """
        
    #     self.logger.info("🚀 Starting multi-agent pruning workflow")
        
    #     try:
    #         # Initialize workflow state
    #         workflow_state = {
    #             'model': model,
    #             'train_loader': train_loader,
    #             'val_loader': val_loader,
    #             'test_loader': test_loader,
    #             'iteration': 0,
    #             'converged': False,
    #             'results': {}
    #         }
            
    #         # Phase 1: Profiling
    #         self.logger.info("📊 Phase 1: Model Profiling")
    #         profiling_results = self._run_profiling_phase(workflow_state)
    #         workflow_state['profiling_results'] = profiling_results
            
    #         # Phase 2: Master Agent Planning
    #         self.logger.info("🧠 Phase 2: Master Agent Planning")
    #         master_plan = self._run_master_planning_phase(workflow_state)
    #         workflow_state['master_plan'] = master_plan
            
    #         # Phase 3: Iterative Pruning Loop
    #         self.logger.info("🔄 Phase 3: Iterative Pruning Loop")
    #         pruning_results = self._run_iterative_pruning_loop(workflow_state)
    #         workflow_state['pruning_results'] = pruning_results
            
    #         # Phase 4: Fine-tuning
    #         self.logger.info("🎯 Phase 4: Fine-tuning")
    #         finetuning_results = self._run_finetuning_phase(workflow_state)
    #         workflow_state['finetuning_results'] = finetuning_results
            
    #         # Phase 5: Final Evaluation
    #         self.logger.info("📊 Phase 5: Final Evaluation")
    #         evaluation_results = self._run_evaluation_phase(workflow_state)
    #         workflow_state['evaluation_results'] = evaluation_results
            
    #         # Compile final results
    #         final_results = self._compile_final_results(workflow_state)
            
    #         self.logger.info("✅ Multi-agent pruning workflow completed successfully")
    #         return final_results
            
    #     except Exception as e:
    #         self.logger.error(f"❌ Pruning workflow failed: {str(e)}")
    #         raise

    def run_pruning(self, model, train_loader, val_loader, test_loader):
        """
        Run the complete multi-agent pruning workflow with proper state management.
        """
        
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
        
        # Use the unified workflow
        return self.run_pruning_workflow(state)

    def _run_profiling_phase(self, workflow_state):
        """Run the profiling phase."""
        
        self.logger.info("🔍 Running profiling agent")
        
        context = {
            'model': workflow_state['model'],
            'model_info': {
                'name': self.config.get('model', {}).get('name', 'unknown'),
                'total_params': sum(p.numel() for p in workflow_state['model'].parameters()),
                'architecture_type': 'transformer'  # Will be detected by profiling agent
            },
            'target_config': self.config.get('pruning', {}),
            'safety_constraints': {
                'max_mlp_ratio': 0.15,
                'max_attention_ratio': 0.10,
                'min_accuracy': 0.70
            }
        }
        
        # Run profiling agent
        profiling_response = self.profiling_agent.execute(context)
        
        # Extract response data (handles both dict and AgentResponse)
        success, message, data = self._extract_response_data(profiling_response, "Profiling")
        
        if not success:
            raise RuntimeError(f"Profiling failed: {message}")
        
        self.logger.info(f"✅ Profiling completed: {message}")
        return data

    def _run_master_planning_phase(self, workflow_state):
        """Run the master planning phase."""
        
        self.logger.info("🧠 Running master agent")
        
        context = {
            'model_info': workflow_state.get('profiling_results', {}).get('model_info', {}),
            'profiling_results': workflow_state.get('profiling_results', {}),
            'target_config': self.config.get('pruning', {}),
            'history': []  # No history for first iteration
        }
        
        # Run master agent
        master_response = self.master_agent.execute(context)
        
        # Extract response data (handles both dict and AgentResponse)
        success, message, data = self._extract_response_data(master_response, "Master planning")
        
        if not success:
            raise RuntimeError(f"Master planning failed: {message}")
        
        self.logger.info(f"✅ Master planning completed: {message}")
        return data

    def _run_iterative_pruning_loop(self, workflow_state):
        """Run the iterative pruning loop."""
        
        max_iterations = self.config.get('multi_agent', {}).get('max_iterations', 10)
        convergence_threshold = self.config.get('multi_agent', {}).get('convergence_threshold', 0.001)
        
        iteration_results = []
        current_model = workflow_state['model']
        
        for iteration in range(max_iterations):
            self.logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Analysis phase
            self.logger.info("🔍 Running analysis agent")
            analysis_context = {
                'model_info': workflow_state.get('profiling_results', {}).get('model_info', {}),
                'profiling_results': workflow_state.get('profiling_results', {}),
                'pruning_config': self.config.get('pruning', {}),
                'history': iteration_results
            }
            
            analysis_response = self.analysis_agent.execute(analysis_context)
            success, message, analysis_data = self._extract_response_data(analysis_response, "Analysis")
            
            if not success:
                self.logger.warning(f"Analysis failed: {message}")
                continue
            
            # Pruning phase
            self.logger.info("✂️ Running pruning agent")
            pruning_context = {
                'model': current_model,
                'model_info': workflow_state.get('profiling_results', {}).get('model_info', {}),
                'analysis_results': analysis_data,
                'safety_constraints': {
                    'max_mlp_ratio': 0.15,
                    'max_attention_ratio': 0.10,
                    'min_accuracy': 0.70
                }
            }
            
            pruning_response = self.pruning_agent.execute(pruning_context)
            success, message, pruning_data = self._extract_response_data(pruning_response, "Pruning")
            
            if not success:
                self.logger.warning(f"Pruning failed: {message}")
                continue
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'analysis': analysis_data,
                'pruning': pruning_data,
                'timestamp': self._get_timestamp()
            }
            iteration_results.append(iteration_result)
            
            # Check convergence
            if self._check_convergence(iteration_results, convergence_threshold):
                self.logger.info(f"✅ Converged after {iteration + 1} iterations")
                break
        
        return {
            'iterations': iteration_results,
            'total_iterations': len(iteration_results),
            'converged': len(iteration_results) < max_iterations
        }

    def _run_finetuning_phase(self, workflow_state):
        """Run the fine-tuning phase."""
        
        self.logger.info("🎯 Running fine-tuning agent")
        
        context = {
            'model': workflow_state['model'],
            'model_info': workflow_state.get('profiling_results', {}).get('model_info', {}),
            'pruning_results': workflow_state.get('pruning_results', {}),
            'training_config': {
                'base_lr': 1e-4,
                'batch_size': self.config.get('dataset', {}).get('batch_size', 32),
                'max_epochs': 100,
                'patience': 10
            },
            'dataset_info': {
                'name': self.config.get('dataset', {}).get('name', 'imagenet'),
                'num_classes': 1000  # ImageNet default
            }
        }
        
        # Run fine-tuning agent
        finetuning_response = self.finetuning_agent.execute(context)
        
        # Extract response data (handles both dict and AgentResponse)
        success, message, data = self._extract_response_data(finetuning_response, "Fine-tuning")
        
        if not success:
            self.logger.warning(f"Fine-tuning planning failed: {message}")
            # Return default fine-tuning plan
            return {
                'learning_rate': 1e-4,
                'training_strategy': 'full_finetuning',
                'max_epochs': 50,
                'final_accuracy': 0.75  # Placeholder
            }
        
        self.logger.info(f"✅ Fine-tuning completed: {message}")
        return data

    def _run_evaluation_phase(self, workflow_state):
        """Run the evaluation phase."""
        
        self.logger.info("📊 Running evaluation agent")
        
        context = {
            'model_info': workflow_state.get('profiling_results', {}).get('model_info', {}),
            'pruning_results': workflow_state.get('pruning_results', {}),
            'finetuning_results': workflow_state.get('finetuning_results', {}),
            'baseline_results': {
                'original_accuracy': 0.80,  # Placeholder
                'magnitude_accuracy': 0.75,
                'taylor_accuracy': 0.76,
                'isomorphic_accuracy': 0.77
            }
        }
        
        # Run evaluation agent
        evaluation_response = self.evaluation_agent.execute(context)
        
        # Extract response data (handles both dict and AgentResponse)
        success, message, data = self._extract_response_data(evaluation_response, "Evaluation")
        
        if not success:
            self.logger.warning(f"Evaluation failed: {message}")
            # Return basic evaluation
            return {
                'overall_rating': 'fair',
                'rating_score': 3,
                'publication_ready': False
            }
        
        self.logger.info(f"✅ Evaluation completed: {message}")
        return data

    def _compile_final_results(self, workflow_state):
        """Compile final results from all phases."""
        
        return {
            'experiment_name': self.config.get('experiment', {}).get('name', 'unknown'),
            'model_name': self.config.get('model', {}).get('name', 'unknown'),
            'profiling_results': workflow_state.get('profiling_results', {}),
            'master_plan': workflow_state.get('master_plan', {}),
            'pruning_results': workflow_state.get('pruning_results', {}),
            'finetuning_results': workflow_state.get('finetuning_results', {}),
            'evaluation_results': workflow_state.get('evaluation_results', {}),
            'timestamp': self._get_timestamp(),
            'success': True
        }

    def _check_convergence(self, iteration_results, threshold):
        """Check if the pruning process has converged."""
        
        if len(iteration_results) < 2:
            return False
        
        # Simple convergence check based on pruning ratio changes
        last_ratio = iteration_results[-1].get('pruning', {}).get('compression_ratio', 0)
        prev_ratio = iteration_results[-2].get('pruning', {}).get('compression_ratio', 0)
        
        change = abs(last_ratio - prev_ratio)
        return change < threshold

    def _get_timestamp(self):
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()