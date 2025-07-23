#!/usr/bin/env python3
"""
Multi-Agent LLM Pruning Framework - Agents Package

This package contains all the specialized agents that work together to perform
intelligent neural network pruning using LLM guidance and coordination.
"""

from .base_agent import BaseAgent
from .profiling_agent import ProfilingAgent
from .master_agent import MasterAgent
from .analysis_agent import AnalysisAgent
from .pruning_agent import PruningAgent
from .finetuning_agent import FinetuningAgent
from .evaluation_agent import EvaluationAgent
from .coordinator import AgentCoordinator

__all__ = [
    'BaseAgent',
    'ProfilingAgent', 
    'MasterAgent',
    'AnalysisAgent',
    'PruningAgent',
    'FinetuningAgent',
    'EvaluationAgent',
    'AgentCoordinator'
]

# Agent workflow order
AGENT_WORKFLOW = [
    'ProfilingAgent',
    'MasterAgent', 
    'AnalysisAgent',
    'PruningAgent',
    'FinetuningAgent',
    'EvaluationAgent'
]

# Agent descriptions
AGENT_DESCRIPTIONS = {
    'ProfilingAgent': 'Analyzes model structure, dependencies, and constraints',
    'MasterAgent': 'Coordinates workflow and makes high-level pruning decisions',
    'AnalysisAgent': 'Determines optimal pruning strategy and parameters',
    'PruningAgent': 'Executes structured pruning with safety checks',
    'FinetuningAgent': 'Recovers accuracy through adaptive fine-tuning',
    'EvaluationAgent': 'Provides comprehensive evaluation and comparison'
}

