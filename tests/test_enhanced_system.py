"""
Comprehensive Test Suite for Enhanced Multi-Agent LLM Pruning Framework

Tests the improved system with precomputation, caching, and optimizations.
"""

import pytest
import torch
import torch.nn as nn
import timm
import tempfile
import shutil
from pathlib import Path
import json
import yaml
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, Any, List

# Import the enhanced framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_pruning.agents.profiling_agent import ProfilingAgent, ModelProfile
from multi_agent_pruning.agents.master_agent import MasterAgent, HistoryAnalyzer
from multi_agent_pruning.core.dependency_analyzer import DependencyAnalyzer
from multi_agent_pruning.core.isomorphic_analyzer import IsomorphicAnalyzer
from multi_agent_pruning.utils.profiler import TimingProfiler
from multi_agent_pruning.utils.metrics import compute_macs, compute_params

class TestEnhancedProfilingAgent:
    """Test the enhanced profiling agent with precomputation and caching."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def small_model(self):
        """Create a small test model."""
        return timm.create_model('resnet18', pretrained=False, num_classes=10)
    
    @pytest.fixture
    def profiling_agent(self, temp_cache_dir):
        """Create profiling agent with temporary cache."""
        config = {
            'enable_caching': True,
            'cache_dir': temp_cache_dir
        }
        return ProfilingAgent(config)
    
    def test_profiling_agent_initialization(self, profiling_agent):
        """Test profiling agent initializes correctly."""
        assert profiling_agent.agent_name == "ProfilingAgent"
        assert profiling_agent.enable_caching is True
        assert profiling_agent.dependency_analyzer is None  # Lazy initialization
        assert profiling_agent.isomorphic_analyzer is None
    
    def test_architecture_detection(self, profiling_agent, small_model):
        """Test architecture type detection."""
        arch_type = profiling_agent._detect_architecture_type(small_model, "resnet18")
        assert arch_type == "cnn"
        
        # Test ViT detection
        vit_model = timm.create_model('deit_tiny_patch16_224', pretrained=False)
        arch_type_vit = profiling_agent._detect_architecture_type(vit_model, "deit_tiny")
        assert arch_type_vit == "vision_transformer"
    
    def test_layer_analysis(self, profiling_agent, small_model):
        """Test layer analysis functionality."""
        layer_info = profiling_agent._analyze_layers(small_model, "cnn")
        
        assert 'layer_count' in layer_info
        assert 'prunable_layers' in layer_info
        assert 'critical_layers' in layer_info
        assert 'sensitive_layers' in layer_info
        
        assert layer_info['layer_count'] > 0
        assert len(layer_info['prunable_layers']) > 0
    
    def test_safety_limits_computation(self, profiling_agent):
        """Test safety limits computation for different architectures and datasets."""
        # Test ImageNet + ViT (most conservative)
        limits_imagenet_vit = profiling_agent._compute_safety_limits("vision_transformer", "imagenet")
        assert limits_imagenet_vit['max_mlp_pruning'] <= 0.15
        assert limits_imagenet_vit['max_attention_pruning'] <= 0.10
        
        # Test CIFAR-10 + CNN (more aggressive)
        limits_cifar_cnn = profiling_agent._compute_safety_limits("cnn", "cifar10")
        assert limits_cifar_cnn['max_overall_pruning'] >= 0.8
    
    def test_comprehensive_profiling(self, profiling_agent, small_model):
        """Test comprehensive profiling with all analyses."""
        profile = profiling_agent._run_comprehensive_profiling(
            small_model, "resnet18", "cifar10", {'input_size': 224}
        )
        
        assert isinstance(profile, ModelProfile)
        assert profile.model_name == "resnet18"
        assert profile.architecture_type == "cnn"
        assert profile.total_params > 0
        assert profile.total_macs > 0
        assert len(profile.prunable_layers) > 0
        assert len(profile.safety_limits) > 0
        assert profile.importance_criterion in ['taylor', 'l1norm', 'l2norm']
    
    def test_caching_functionality(self, profiling_agent, small_model):
        """Test that caching works correctly."""
        input_data = {
            'model': small_model,
            'model_name': 'resnet18',
            'dataset': 'cifar10'
        }
        
        # First execution - should compute and cache
        result1 = profiling_agent.execute(input_data)
        assert result1['success'] is True
        assert result1['cached'] is False
        
        # Second execution - should use cache
        result2 = profiling_agent.execute(input_data)
        assert result2['success'] is True
        assert result2['cached'] is True
        
        # Results should be identical
        assert result1['profile']['model_name'] == result2['profile']['model_name']
        assert result1['profile']['total_params'] == result2['profile']['total_params']
    
    def test_timing_profiler_integration(self, profiling_agent, small_model):
        """Test timing profiler integration."""
        input_data = {
            'model': small_model,
            'model_name': 'resnet18',
            'dataset': 'cifar10'
        }
        
        result = profiling_agent.execute(input_data)
        
        # Should have timing information
        assert 'timing' in result or profiling_agent.timing_profiler.timings
        
        # Check that timing profiler recorded some operations
        assert len(profiling_agent.timing_profiler.timings) > 0

class TestEnhancedMasterAgent:
    """Test the enhanced master agent with better decision making."""
    
    @pytest.fixture
    def master_agent(self):
        """Create master agent with test configuration."""
        config = {
            'max_iterations': 3,
            'convergence_threshold': 0.01,
            'target_tolerance': 0.02
        }
        return MasterAgent(config)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return {
            'target_ratio': 0.5,
            'dataset': 'cifar10',
            'revision_number': 0,
            'dataset_info': {
                'num_classes': 10,
                'recommended_approach': 'moderate',
                'safety_limits': {
                    'max_mlp_pruning': 0.3,
                    'max_attention_pruning': 0.2,
                    'min_accuracy_threshold': 0.6
                }
            },
            'model_info': {
                'architecture_type': 'vision_transformer'
            }
        }
    
    def test_master_agent_initialization(self, master_agent):
        """Test master agent initializes correctly."""
        assert master_agent.agent_name == "MasterAgent"
        assert master_agent.max_iterations == 3
        assert master_agent.convergence_threshold == 0.01
        assert isinstance(master_agent.history_analyzer, HistoryAnalyzer)
    
    def test_exploration_strategy_determination(self, master_agent, sample_context):
        """Test exploration strategy determination logic."""
        # Test early iteration - should be conservative
        history_analysis = {'progress_rate': 0, 'target_achievement_rate': 0}
        strategy = master_agent._determine_exploration_strategy(sample_context, history_analysis)
        assert strategy == 'conservative'
        
        # Test good progress - should continue current strategy
        sample_context['revision_number'] = 2
        history_analysis = {'progress_rate': 0.02, 'target_achievement_rate': 0.8}
        strategy = master_agent._determine_exploration_strategy(sample_context, history_analysis)
        assert strategy in ['conservative', 'moderate', 'aggressive']
    
    def test_stopping_conditions(self, master_agent, sample_context):
        """Test stopping condition detection."""
        # Test maximum iterations
        sample_context['revision_number'] = 5
        should_stop, reason = master_agent._check_stopping_conditions(sample_context, {})
        assert should_stop is True
        assert "Maximum iterations" in reason
        
        # Test target achievement
        sample_context['revision_number'] = 1
        history_analysis = {
            'best_result': {
                'achieved_ratio': 0.51,  # Close to target 0.5
                'accuracy': 0.75
            }
        }
        should_stop, reason = master_agent._check_stopping_conditions(sample_context, history_analysis)
        assert should_stop is True
        assert "Target achieved" in reason
    
    def test_pruning_ratio_estimation(self, master_agent):
        """Test pruning ratio estimation logic."""
        target_ratio = 0.5
        history_analysis = {'ratio_scaling_factor': 1.2}
        
        # Conservative strategy
        ratio = master_agent._estimate_pruning_ratio(target_ratio, history_analysis, 'conservative')
        assert 0.1 <= ratio <= 0.9
        assert ratio < target_ratio * 1.2  # Should be more conservative
        
        # Aggressive strategy
        ratio = master_agent._estimate_pruning_ratio(target_ratio, history_analysis, 'aggressive')
        assert ratio > target_ratio * 1.0  # Should be more aggressive
    
    def test_strategy_generation(self, master_agent, sample_context):
        """Test strategy generation with various inputs."""
        history_analysis = {
            'parameter_recommendations': {
                'best_importance_criterion': 'taylor',
                'optimal_round_to': 2
            },
            'ratio_scaling_factor': 1.1
        }
        
        strategy = master_agent._generate_next_strategy(
            sample_context, history_analysis, 'moderate'
        )
        
        assert strategy.importance_criterion in ['taylor', 'l1norm', 'l2norm']
        assert 0.1 <= strategy.pruning_ratio <= 0.9
        assert strategy.round_to in [1, 2, 4, 8, 16] or strategy.round_to is None
        assert strategy.global_pruning is True
        assert strategy.continue_exploration is True
        assert len(strategy.rationale) > 0
    
    def test_safety_validation(self, master_agent, sample_context):
        """Test safety constraint validation."""
        from multi_agent_pruning.agents.master_agent import PruningStrategy
        
        # Create strategy that exceeds safety limits
        unsafe_strategy = PruningStrategy(
            importance_criterion='taylor',
            pruning_ratio=0.95,  # Very high ratio
            round_to=2,
            global_pruning=True,
            rationale="Test strategy",
            confidence=0.8,
            expected_accuracy_drop=0.1,
            risk_level='high',
            continue_exploration=True,
            stop_reason=None
        )
        
        # Apply safety validation
        safe_strategy = master_agent._apply_safety_validation(unsafe_strategy, sample_context)
        
        # Should be clamped to safety limits
        max_overall = sample_context['dataset_info']['safety_limits'].get('max_overall_pruning', 0.8)
        assert safe_strategy.pruning_ratio <= max_overall
        assert safe_strategy.risk_level == 'high'

class TestHistoryAnalyzer:
    """Test the history analyzer for pattern detection."""
    
    @pytest.fixture
    def history_analyzer(self):
        """Create history analyzer."""
        return HistoryAnalyzer()
    
    @pytest.fixture
    def sample_history(self):
        """Create sample pruning history."""
        return [
            {
                'results': {
                    'evaluation': {
                        'final_accuracy': 0.75,
                        'params_reduction': 0.45,
                        'macs_reduction': 0.40
                    },
                    'master': {
                        'importance_criterion': 'taylor',
                        'pruning_ratio': 0.5,
                        'round_to': 2
                    }
                }
            },
            {
                'results': {
                    'evaluation': {
                        'final_accuracy': 0.78,
                        'params_reduction': 0.48,
                        'macs_reduction': 0.42
                    },
                    'master': {
                        'importance_criterion': 'taylor',
                        'pruning_ratio': 0.52,
                        'round_to': 2
                    }
                }
            },
            {
                'results': {
                    'evaluation': {
                        'final_accuracy': 0.76,
                        'params_reduction': 0.50,
                        'macs_reduction': 0.45
                    },
                    'master': {
                        'importance_criterion': 'l1norm',
                        'pruning_ratio': 0.55,
                        'round_to': 4
                    }
                }
            }
        ]
    
    def test_history_analysis_empty(self, history_analyzer):
        """Test analysis with empty history."""
        analysis = history_analyzer.analyze_history([], [], 0.5)
        
        assert analysis['num_attempts'] == 0
        assert analysis['converged'] is False
        assert analysis['cycling'] is False
        assert analysis['catastrophic_failure'] is False
        assert analysis['progress_rate'] == 0
    
    def test_history_analysis_with_data(self, history_analyzer, sample_history):
        """Test analysis with sample history data."""
        attempted_ratios = [0.5, 0.52, 0.55]
        target_ratio = 0.5
        
        analysis = history_analyzer.analyze_history(sample_history, attempted_ratios, target_ratio)
        
        assert analysis['num_attempts'] == 3
        assert 'best_result' in analysis
        assert analysis['best_result']['accuracy'] == 0.78  # Best accuracy
        assert 'progress_rate' in analysis
        assert 'target_achievement_rate' in analysis
        assert 'parameter_recommendations' in analysis
    
    def test_convergence_detection(self, history_analyzer):
        """Test convergence detection logic."""
        # Create results with diminishing improvements
        results = [
            {'accuracy': 0.70, 'achieved_ratio': 0.45},
            {'accuracy': 0.703, 'achieved_ratio': 0.47},  # Small improvement
            {'accuracy': 0.704, 'achieved_ratio': 0.48},  # Very small improvement
        ]
        
        converged = history_analyzer._detect_convergence(results)
        assert converged is True
        
        # Create results with significant improvements
        results_improving = [
            {'accuracy': 0.70, 'achieved_ratio': 0.45},
            {'accuracy': 0.75, 'achieved_ratio': 0.47},  # Large improvement
            {'accuracy': 0.78, 'achieved_ratio': 0.48},  # Large improvement
        ]
        
        converged = history_analyzer._detect_convergence(results_improving)
        assert converged is False
    
    def test_cycling_detection(self, history_analyzer):
        """Test cycling behavior detection."""
        # Create results with cycling accuracies
        results_cycling = [
            {'accuracy': 0.70, 'achieved_ratio': 0.45},
            {'accuracy': 0.75, 'achieved_ratio': 0.47},
            {'accuracy': 0.70, 'achieved_ratio': 0.45},  # Repeat
            {'accuracy': 0.75, 'achieved_ratio': 0.47},  # Repeat
        ]
        
        cycling = history_analyzer._detect_cycling(results_cycling)
        assert cycling is True
        
        # Create results without cycling
        results_no_cycling = [
            {'accuracy': 0.70, 'achieved_ratio': 0.45},
            {'accuracy': 0.75, 'achieved_ratio': 0.47},
            {'accuracy': 0.78, 'achieved_ratio': 0.48},
            {'accuracy': 0.80, 'achieved_ratio': 0.50},
        ]
        
        cycling = history_analyzer._detect_cycling(results_no_cycling)
        assert cycling is False

class TestDependencyAnalyzer:
    """Test the dependency analyzer for layer coupling detection."""
    
    @pytest.fixture
    def vit_model(self):
        """Create a small ViT model for testing."""
        return timm.create_model('deit_tiny_patch16_224', pretrained=False)
    
    @pytest.fixture
    def cnn_model(self):
        """Create a CNN model for testing."""
        return timm.create_model('resnet18', pretrained=False)
    
    def test_dependency_analyzer_initialization(self, vit_model):
        """Test dependency analyzer initialization."""
        analyzer = DependencyAnalyzer(vit_model)
        assert analyzer.model == vit_model
        assert hasattr(analyzer, 'dependency_graph')
    
    def test_vit_dependency_detection(self, vit_model):
        """Test dependency detection in Vision Transformers."""
        analyzer = DependencyAnalyzer(vit_model)
        
        # Should detect transformer block dependencies
        dependencies = analyzer.get_dependency_graph()
        assert isinstance(dependencies, dict)
        
        # Should find coupled layers in transformer blocks
        coupled_layers = analyzer.get_coupling_constraints()
        assert isinstance(coupled_layers, list)
    
    def test_cnn_dependency_detection(self, cnn_model):
        """Test dependency detection in CNNs."""
        analyzer = DependencyAnalyzer(cnn_model)
        
        dependencies = analyzer.get_dependency_graph()
        assert isinstance(dependencies, dict)
        
        # CNNs typically have fewer coupling constraints than ViTs
        coupled_layers = analyzer.get_coupling_constraints()
        assert isinstance(coupled_layers, list)

class TestIsomorphicAnalyzer:
    """Test the isomorphic analyzer for grouping similar layers."""
    
    @pytest.fixture
    def vit_model(self):
        """Create a ViT model for isomorphic analysis."""
        return timm.create_model('deit_tiny_patch16_224', pretrained=False)
    
    def test_isomorphic_analyzer_initialization(self, vit_model):
        """Test isomorphic analyzer initialization."""
        analyzer = IsomorphicAnalyzer(vit_model)
        assert analyzer.model == vit_model
    
    def test_isomorphic_group_creation(self, vit_model):
        """Test creation of isomorphic groups."""
        analyzer = IsomorphicAnalyzer(vit_model)
        
        groups = analyzer.create_isomorphic_groups(target_ratio=0.5)
        assert isinstance(groups, dict)
        assert len(groups) > 0
        
        # Should have different groups for different layer types
        group_names = list(groups.keys())
        assert any('mlp' in name.lower() for name in group_names)

class TestTimingProfiler:
    """Test the timing profiler for performance monitoring."""
    
    @pytest.fixture
    def profiler(self):
        """Create timing profiler."""
        return TimingProfiler()
    
    def test_timing_profiler_basic(self, profiler):
        """Test basic timing functionality."""
        import time
        
        with profiler.timer("test_operation"):
            time.sleep(0.01)  # 10ms
        
        assert "test_operation" in profiler.timings
        assert len(profiler.timings["test_operation"]) == 1
        assert profiler.timings["test_operation"][0] >= 0.01
    
    def test_timing_profiler_multiple_calls(self, profiler):
        """Test multiple calls to same operation."""
        import time
        
        for i in range(3):
            with profiler.timer("repeated_operation"):
                time.sleep(0.005)  # 5ms
        
        assert "repeated_operation" in profiler.timings
        assert len(profiler.timings["repeated_operation"]) == 3
        
        # All timings should be roughly 5ms
        for timing in profiler.timings["repeated_operation"]:
            assert 0.004 <= timing <= 0.01  # Allow some variance
    
    def test_timing_profiler_summary(self, profiler):
        """Test timing summary generation."""
        import time
        
        # Add some timed operations
        with profiler.timer("fast_operation"):
            time.sleep(0.001)
        
        with profiler.timer("slow_operation"):
            time.sleep(0.01)
        
        # Get summary (should not raise exception)
        summary = profiler.get_summary()
        # Summary is printed, not returned, so just check it doesn't crash

class TestIntegration:
    """Integration tests for the complete enhanced system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        return {
            'cache_dir': temp_dir,
            'enable_caching': True,
            'max_iterations': 2,
            'convergence_threshold': 0.01
        }
    
    def test_full_workflow_simulation(self, test_config):
        """Test a complete workflow simulation with mocked components."""
        
        # Create small test model
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        
        # Initialize agents
        profiling_agent = ProfilingAgent(test_config)
        master_agent = MasterAgent(test_config)
        
        # Step 1: Profiling
        profile_input = {
            'model': model,
            'model_name': 'resnet18',
            'dataset': 'cifar10',
            'input_size': 224
        }
        
        profile_result = profiling_agent.execute(profile_input)
        assert profile_result['success'] is True
        
        # Step 2: Master agent decision
        master_input = {
            'profile_results': profile_result['profile'],
            'target_ratio': 0.5,
            'dataset': 'cifar10',
            'revision_number': 0,
            'history': [],
            'dataset_info': {
                'num_classes': 10,
                'safety_limits': {
                    'max_overall_pruning': 0.8,
                    'max_mlp_pruning': 0.6,
                    'max_attention_pruning': 0.4
                }
            },
            'model_info': {
                'architecture_type': 'cnn'
            }
        }
        
        master_result = master_agent.execute(master_input)
        assert master_result['success'] is True
        assert 'strategy' in master_result
        assert master_result['continue_iterations'] is True
    
    def test_caching_across_agents(self, test_config):
        """Test that caching works correctly across multiple agent calls."""
        
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        profiling_agent = ProfilingAgent(test_config)
        
        profile_input = {
            'model': model,
            'model_name': 'resnet18',
            'dataset': 'cifar10'
        }
        
        # First call - should compute
        result1 = profiling_agent.execute(profile_input)
        assert result1['cached'] is False
        
        # Second call - should use cache
        result2 = profiling_agent.execute(profile_input)
        assert result2['cached'] is True
        
        # Results should be consistent
        assert result1['profile']['total_params'] == result2['profile']['total_params']
    
    def test_error_handling(self, test_config):
        """Test error handling in various scenarios."""
        
        profiling_agent = ProfilingAgent(test_config)
        
        # Test with invalid input
        invalid_input = {
            'model': None,  # Invalid model
            'model_name': 'invalid',
            'dataset': 'unknown'
        }
        
        result = profiling_agent.execute(invalid_input)
        # Should handle gracefully without crashing
        assert isinstance(result, dict)

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for the enhanced system."""
    
    def test_profiling_performance(self):
        """Benchmark profiling agent performance."""
        import time
        
        model = timm.create_model('resnet18', pretrained=False)
        profiling_agent = ProfilingAgent({'enable_caching': False})
        
        profile_input = {
            'model': model,
            'model_name': 'resnet18',
            'dataset': 'cifar10'
        }
        
        start_time = time.time()
        result = profiling_agent.execute(profile_input)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete profiling within reasonable time (5 seconds)
        assert execution_time < 5.0
        assert result['success'] is True
    
    def test_caching_performance_improvement(self):
        """Test that caching provides performance improvement."""
        import time
        
        model = timm.create_model('resnet18', pretrained=False)
        
        # Test with caching enabled
        profiling_agent_cached = ProfilingAgent({'enable_caching': True})
        profile_input = {
            'model': model,
            'model_name': 'resnet18',
            'dataset': 'cifar10'
        }
        
        # First call (compute)
        start_time = time.time()
        result1 = profiling_agent_cached.execute(profile_input)
        first_call_time = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        result2 = profiling_agent_cached.execute(profile_input)
        second_call_time = time.time() - start_time
        
        # Cached call should be significantly faster
        assert second_call_time < first_call_time * 0.5  # At least 50% faster
        assert result2['cached'] is True

# Fixtures for pytest
@pytest.fixture(scope="session")
def test_models():
    """Load test models once per session."""
    models = {
        'resnet18': timm.create_model('resnet18', pretrained=False, num_classes=10),
        'deit_tiny': timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=10)
    }
    return models

@pytest.fixture(scope="session") 
def test_datasets():
    """Create test dataset configurations."""
    return {
        'cifar10': {
            'num_classes': 10,
            'input_size': 32,
            'safety_limits': {
                'max_overall_pruning': 0.8,
                'max_mlp_pruning': 0.6,
                'max_attention_pruning': 0.4
            }
        },
        'imagenet': {
            'num_classes': 1000,
            'input_size': 224,
            'safety_limits': {
                'max_overall_pruning': 0.6,
                'max_mlp_pruning': 0.15,
                'max_attention_pruning': 0.10
            }
        }
    }

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

