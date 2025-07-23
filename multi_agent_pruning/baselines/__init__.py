"""
Baseline Pruning Methods for Comparison

This module implements various baseline pruning methods for fair comparison
with the multi-agent LLM approach. All methods follow a unified interface
for consistent evaluation.

Implemented Methods:
- Magnitude Pruning (L1/L2 norm)
- Taylor Expansion Pruning
- Random Pruning
- Structured Pruning
- Original Isomorphic Pruning
- SNIP (Single-shot Network Pruning)
- GraSP (Gradient Signal Preservation)
"""

from .base_pruning import BasePruningMethod
from .magnitude_pruning import MagnitudePruning
from .taylor_pruning import TaylorPruning
from .random_pruning import RandomPruning
from .structured_pruning import StructuredPruning
from .isomorphic_pruning import IsomorphicPruning
from .snip_pruning import SNIPPruning
from .grasp_pruning import GraSPPruning

__all__ = [
    'BasePruningMethod',
    'MagnitudePruning',
    'TaylorPruning', 
    'RandomPruning',
    'StructuredPruning',
    'IsomorphicPruning',
    'SNIPPruning',
    'GraSPPruning',
]

