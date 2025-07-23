"""
State Manager for Multi-Agent Pruning Workflow

Handles workflow state management with extensive caching and precomputation
to accelerate experiments and avoid redundant calculations.
"""

import os
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PruningState:
    """Complete state of the pruning workflow with caching support."""
    
    # Basic workflow state
    query: str
    model_name: str
    dataset: str
    target_ratio: float
    revision_number: int = 0
    max_revisions: int = 5
    
    # Model and data info
    model: Optional[Any] = None
    num_classes: int = 1000
    input_size: int = 224
    data_path: str = ""
    
    # Agent results (cached)
    profile_results: Dict = None
    master_results: Dict = None
    analysis_results: Dict = None
    pruning_results: Dict = None
    fine_tuning_results: Dict = None
    evaluation_results: Dict = None
    
    # Experiment tracking
    attempted_pruning_ratios: List[float] = None
    history: List[Dict[str, Any]] = None
    
    # Precomputed data (cached to disk)
    _precomputed_cache: Dict[str, Any] = None
    _cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.attempted_pruning_ratios is None:
            self.attempted_pruning_ratios = []
        if self.history is None:
            self.history = []
        if self.profile_results is None:
            self.profile_results = {}
        if self.master_results is None:
            self.master_results = {}
        if self.analysis_results is None:
            self.analysis_results = {}
        if self.pruning_results is None:
            self.pruning_results = {}
        if self.fine_tuning_results is None:
            self.fine_tuning_results = {}
        if self.evaluation_results is None:
            self.evaluation_results = {}
        if self._precomputed_cache is None:
            self._precomputed_cache = {}

class StateManager:
    """
    Manages workflow state with extensive caching and precomputation capabilities.
    
    Key Features:
    - Persistent caching of expensive computations
    - Precomputation of common operations
    - Intelligent cache invalidation
    - Experiment reproducibility
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different types of cached data
        self.model_cache_dir = self.cache_dir / "models"
        self.importance_cache_dir = self.cache_dir / "importance_scores"
        self.dependency_cache_dir = self.cache_dir / "dependencies"
        self.profile_cache_dir = self.cache_dir / "profiles"
        self.results_cache_dir = self.cache_dir / "results"
        
        for cache_dir in [self.model_cache_dir, self.importance_cache_dir, 
                         self.dependency_cache_dir, self.profile_cache_dir, 
                         self.results_cache_dir]:
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"StateManager initialized with cache directory: {self.cache_dir}")
    
    def create_state(self, query: str, model_name: str, dataset: str, 
                    target_ratio: float, **kwargs) -> PruningState:
        """Create a new pruning state with precomputation support."""
        
        state = PruningState(
            query=query,
            model_name=model_name,
            dataset=dataset,
            target_ratio=target_ratio,
            _cache_dir=str(self.cache_dir),
            **kwargs
        )
        
        # Start precomputation for common operations
        self._schedule_precomputation(state)
        
        return state
    
    def _schedule_precomputation(self, state: PruningState):
        """Schedule precomputation of expensive operations."""
        
        logger.info("🚀 Starting precomputation for faster experiments...")
        
        # 1. Precompute model architecture analysis
        self._precompute_model_analysis(state)
        
        # 2. Precompute dependency graphs
        self._precompute_dependency_analysis(state)
        
        # 3. Precompute importance scores for different criteria
        self._precompute_importance_scores(state)
        
        # 4. Precompute dataset statistics
        self._precompute_dataset_stats(state)
        
        logger.info("✅ Precomputation completed!")
    
    def _precompute_model_analysis(self, state: PruningState):
        """Precompute model architecture analysis."""
        
        cache_key = f"model_analysis_{state.model_name}_{state.dataset}"
        cache_file = self.profile_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            logger.info(f"📋 Loading cached model analysis for {state.model_name}")
            with open(cache_file, 'rb') as f:
                analysis = pickle.load(f)
            state._precomputed_cache['model_analysis'] = analysis
            return
        
        logger.info(f"🔍 Precomputing model analysis for {state.model_name}...")
        
        try:
            # This would be implemented with actual model loading
            analysis = {
                'architecture_type': self._detect_architecture_type(state.model_name),
                'layer_count': self._count_layers(state.model_name),
                'parameter_count': self._count_parameters(state.model_name),
                'flops': self._estimate_flops(state.model_name, state.input_size),
                'memory_usage': self._estimate_memory(state.model_name, state.input_size),
                'critical_layers': self._identify_critical_layers(state.model_name),
                'prunable_layers': self._identify_prunable_layers(state.model_name),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the analysis
            with open(cache_file, 'wb') as f:
                pickle.dump(analysis, f)
            
            state._precomputed_cache['model_analysis'] = analysis
            logger.info(f"✅ Model analysis cached for {state.model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to precompute model analysis: {e}")
    
    def _precompute_dependency_analysis(self, state: PruningState):
        """Precompute layer dependency graphs."""
        
        cache_key = f"dependencies_{state.model_name}"
        cache_file = self.dependency_cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            logger.info(f"🔗 Loading cached dependency analysis for {state.model_name}")
            with open(cache_file, 'rb') as f:
                dependencies = pickle.load(f)
            state._precomputed_cache['dependencies'] = dependencies
            return
        
        logger.info(f"🔗 Precomputing dependency analysis for {state.model_name}...")
        
        try:
            # This would use the actual DependencyAnalyzer
            dependencies = {
                'layer_dependencies': self._analyze_layer_dependencies(state.model_name),
                'coupling_constraints': self._analyze_coupling_constraints(state.model_name),
                'isomorphic_groups': self._identify_isomorphic_groups(state.model_name),
                'pruning_constraints': self._identify_pruning_constraints(state.model_name),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the dependencies
            with open(cache_file, 'wb') as f:
                pickle.dump(dependencies, f)
            
            state._precomputed_cache['dependencies'] = dependencies
            logger.info(f"✅ Dependency analysis cached for {state.model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to precompute dependency analysis: {e}")
    
    def _precompute_importance_scores(self, state: PruningState):
        """Precompute importance scores for different criteria."""
        
        cache_key = f"importance_{state.model_name}_{state.dataset}"
        
        # Precompute for multiple importance criteria
        criteria = ['taylor', 'l1norm', 'l2norm', 'random']
        
        for criterion in criteria:
            criterion_cache_file = self.importance_cache_dir / f"{cache_key}_{criterion}.pkl"
            
            if criterion_cache_file.exists():
                logger.info(f"📊 Loading cached {criterion} importance scores")
                with open(criterion_cache_file, 'rb') as f:
                    scores = pickle.load(f)
                state._precomputed_cache[f'importance_{criterion}'] = scores
                continue
            
            logger.info(f"📊 Precomputing {criterion} importance scores...")
            
            try:
                # This would use actual importance calculation
                scores = self._calculate_importance_scores(state.model_name, 
                                                        state.dataset, criterion)
                
                # Cache the scores
                with open(criterion_cache_file, 'wb') as f:
                    pickle.dump(scores, f)
                
                state._precomputed_cache[f'importance_{criterion}'] = scores
                logger.info(f"✅ {criterion} importance scores cached")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to precompute {criterion} importance: {e}")
    
    def _precompute_dataset_stats(self, state: PruningState):
        """Precompute dataset statistics."""
        
        cache_key = f"dataset_stats_{state.dataset}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            logger.info(f"📈 Loading cached dataset statistics for {state.dataset}")
            with open(cache_file, 'r') as f:
                stats = json.load(f)
            state._precomputed_cache['dataset_stats'] = stats
            return
        
        logger.info(f"📈 Precomputing dataset statistics for {state.dataset}...")
        
        try:
            stats = {
                'num_classes': state.num_classes,
                'input_size': state.input_size,
                'train_samples': self._count_train_samples(state.dataset),
                'val_samples': self._count_val_samples(state.dataset),
                'class_distribution': self._analyze_class_distribution(state.dataset),
                'data_statistics': self._compute_data_statistics(state.dataset),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the statistics
            with open(cache_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            state._precomputed_cache['dataset_stats'] = stats
            logger.info(f"✅ Dataset statistics cached for {state.dataset}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to precompute dataset statistics: {e}")
    
    def get_cached_data(self, state: PruningState, key: str) -> Optional[Any]:
        """Retrieve cached precomputed data."""
        return state._precomputed_cache.get(key)
    
    def cache_result(self, state: PruningState, key: str, data: Any, 
                    persistent: bool = True):
        """Cache a result for future use."""
        
        # Store in memory cache
        state._precomputed_cache[key] = data
        
        if persistent:
            # Store to disk cache
            cache_file = self.results_cache_dir / f"{key}_{state.model_name}_{state.dataset}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug(f"💾 Cached {key} to disk")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache {key} to disk: {e}")
    
    def invalidate_cache(self, pattern: str = None):
        """Invalidate cached data matching pattern."""
        
        if pattern is None:
            # Clear all caches
            logger.info("🗑️ Clearing all caches...")
            for cache_dir in [self.model_cache_dir, self.importance_cache_dir,
                             self.dependency_cache_dir, self.profile_cache_dir,
                             self.results_cache_dir]:
                for file in cache_dir.glob("*"):
                    file.unlink()
        else:
            # Clear caches matching pattern
            logger.info(f"🗑️ Clearing caches matching pattern: {pattern}")
            for cache_dir in [self.model_cache_dir, self.importance_cache_dir,
                             self.dependency_cache_dir, self.profile_cache_dir,
                             self.results_cache_dir]:
                for file in cache_dir.glob(f"*{pattern}*"):
                    file.unlink()
    
    def save_state(self, state: PruningState, filename: str = None):
        """Save complete state to disk."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"state_{state.model_name}_{state.dataset}_{timestamp}.pkl"
        
        state_file = self.cache_dir / filename
        
        # Create a serializable copy of the state
        state_dict = asdict(state)
        state_dict['model'] = None  # Don't serialize the actual model
        
        with open(state_file, 'wb') as f:
            pickle.dump(state_dict, f)
        
        logger.info(f"💾 State saved to {state_file}")
        return str(state_file)
    
    def load_state(self, filename: str) -> PruningState:
        """Load state from disk."""
        
        state_file = Path(filename)
        if not state_file.exists():
            state_file = self.cache_dir / filename
        
        with open(state_file, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Reconstruct the state
        state = PruningState(**state_dict)
        state._cache_dir = str(self.cache_dir)
        
        logger.info(f"📂 State loaded from {state_file}")
        return state
    
    # Placeholder methods for actual implementations
    def _detect_architecture_type(self, model_name: str) -> str:
        """Detect model architecture type."""
        if 'resnet' in model_name.lower():
            return 'cnn'
        elif 'deit' in model_name.lower() or 'vit' in model_name.lower():
            return 'vit'
        elif 'convnext' in model_name.lower():
            return 'cnn'
        else:
            return 'unknown'
    
    def _count_layers(self, model_name: str) -> int:
        """Count model layers."""
        # Placeholder - would use actual model
        layer_counts = {
            'resnet50': 50,
            'resnet101': 101,
            'resnet152': 152,
            'deit_small': 12,
            'deit_tiny': 12,
            'convnext_small': 27,
            'convnext_tiny': 27
        }
        return layer_counts.get(model_name, 0)
    
    def _count_parameters(self, model_name: str) -> int:
        """Count model parameters."""
        # Placeholder - would use actual model
        param_counts = {
            'resnet50': 25557032,
            'resnet101': 44549160,
            'resnet152': 60192808,
            'deit_small': 22050664,
            'deit_tiny': 5717416,
            'convnext_small': 50223688,
            'convnext_tiny': 28589128
        }
        return param_counts.get(model_name, 0)
    
    def _estimate_flops(self, model_name: str, input_size: int) -> float:
        """Estimate model FLOPs."""
        # Placeholder - would use actual FLOP calculation
        return 0.0
    
    def _estimate_memory(self, model_name: str, input_size: int) -> float:
        """Estimate model memory usage."""
        # Placeholder - would use actual memory estimation
        return 0.0
    
    def _identify_critical_layers(self, model_name: str) -> List[str]:
        """Identify critical layers that shouldn't be pruned."""
        # Placeholder - would use actual analysis
        return ['classifier', 'head', 'fc']
    
    def _identify_prunable_layers(self, model_name: str) -> List[str]:
        """Identify layers that can be safely pruned."""
        # Placeholder - would use actual analysis
        return []
    
    def _analyze_layer_dependencies(self, model_name: str) -> Dict:
        """Analyze layer dependencies."""
        # Placeholder - would use DependencyAnalyzer
        return {}
    
    def _analyze_coupling_constraints(self, model_name: str) -> Dict:
        """Analyze coupling constraints."""
        # Placeholder - would use actual analysis
        return {}
    
    def _identify_isomorphic_groups(self, model_name: str) -> List:
        """Identify isomorphic layer groups."""
        # Placeholder - would use IsomorphicAnalyzer
        return []
    
    def _identify_pruning_constraints(self, model_name: str) -> Dict:
        """Identify pruning constraints."""
        # Placeholder - would use actual analysis
        return {}
    
    def _calculate_importance_scores(self, model_name: str, dataset: str, 
                                   criterion: str) -> Dict:
        """Calculate importance scores."""
        # Placeholder - would use actual importance calculation
        return {}
    
    def _count_train_samples(self, dataset: str) -> int:
        """Count training samples."""
        counts = {'imagenet': 1281167, 'cifar10': 50000}
        return counts.get(dataset, 0)
    
    def _count_val_samples(self, dataset: str) -> int:
        """Count validation samples."""
        counts = {'imagenet': 50000, 'cifar10': 10000}
        return counts.get(dataset, 0)
    
    def _analyze_class_distribution(self, dataset: str) -> Dict:
        """Analyze class distribution."""
        # Placeholder - would analyze actual dataset
        return {}
    
    def _compute_data_statistics(self, dataset: str) -> Dict:
        """Compute dataset statistics."""
        # Placeholder - would compute actual statistics
        return {}
    
    def get_precomputation_status(self, state: PruningState) -> Dict[str, bool]:
        """Get status of precomputed data."""
        
        expected_keys = [
            'model_analysis',
            'dependencies', 
            'importance_taylor',
            'importance_l1norm',
            'importance_l2norm',
            'importance_random',
            'dataset_stats'
        ]
        
        status = {}
        for key in expected_keys:
            status[key] = key in state._precomputed_cache
        
        return status
    
    def print_cache_summary(self, state: PruningState):
        """Print summary of cached data."""
        
        print("\n" + "="*60)
        print("📋 PRECOMPUTATION CACHE SUMMARY")
        print("="*60)
        
        status = self.get_precomputation_status(state)
        
        for key, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"{status_icon} {key:<25}: {'Available' if available else 'Missing'}")
        
        cache_size = sum(len(str(v)) for v in state._precomputed_cache.values())
        print(f"\n📊 Total cache size: {cache_size:,} bytes")
        print(f"🗂️ Cache directory: {self.cache_dir}")
        print("="*60)

