"""
Multi-Agent LLM Pruning Framework

A professional framework for neural network pruning using multi-agent LLM coordination.
Combines traditional pruning techniques with AI-driven decision making for optimal
model compression across different architectures (CNNs, Vision Transformers).

Key Features:
- Multi-agent coordination for pruning decisions
- LLM-guided strategy optimization
- Support for multiple architectures (ResNet, DeiT, ConvNext, etc.)
- Comprehensive baseline comparisons
- Hardware-aware optimization
- Reproducible experiments

Authors: Based on research in isomorphic pruning and multi-agent neural architecture search
"""

from .version import __version__

# Core imports
from .core.state_manager import PruningState, StateManager
from .core.pruning_engine import PruningEngine
from .core.dependency_analyzer import DependencyAnalyzer
from .core.isomorphic_analyzer import IsomorphicAnalyzer

# Agent imports
from .agents.coordinator import AgentCoordinator
from .agents.profiling_agent import ProfilingAgent
from .agents.master_agent import MasterAgent
from .agents.analysis_agent import AnalysisAgent
from .agents.pruning_agent import PruningAgent
from .agents.finetuning_agent import FinetuningAgent

# Model imports
from .models.model_factory import ModelFactory
from .models.architecture_detector import ArchitectureDetector

# Data imports
from .data.dataset_factory import DatasetFactory

# Utility imports
from .utils.logging import setup_logging
from .utils.metrics import PruningMetrics
from .utils.profiler import TimingProfiler

__all__ = [
    # Version
    '__version__',
    
    # Core classes
    'PruningState',
    'StateManager', 
    'PruningEngine',
    'DependencyAnalyzer',
    'IsomorphicAnalyzer',
    
    # Agent classes
    'AgentCoordinator',
    'ProfilingAgent',
    'MasterAgent',
    'AnalysisAgent', 
    'PruningAgent',
    'FinetuningAgent',
    
    # Model classes
    'ModelFactory',
    'ArchitectureDetector',
    
    # Data classes
    'DatasetFactory',
    
    # Utility classes
    'setup_logging',
    'PruningMetrics',
    'TimingProfiler',
]

