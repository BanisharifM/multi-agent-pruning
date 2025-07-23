
# User Guide

This comprehensive guide will walk you through using the Enhanced Multi-Agent LLM Pruning Framework, from basic usage to advanced customization.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Configuration Guide](#configuration-guide)
4. [Advanced Features](#advanced-features)
5. [HPC Integration](#hpc-integration)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

Before using the framework, ensure you have:

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- OpenAI API key for LLM agents
- Access to ImageNet dataset (for full experiments)

### Installation Methods

#### Method 1: Automated HPC Setup (Recommended)

For HPC systems with SLURM, use our automated setup script:

```bash
git clone https://github.com/your-username/multi-agent-pruning.git
cd multi-agent-pruning
chmod +x tools/setup_hpc.sh
./tools/setup_hpc.sh
```

This script will:
- Detect your HPC environment
- Load appropriate modules
- Create conda environment
- Install all dependencies
- Download essential models
- Configure SLURM job scripts

#### Method 2: Manual Installation

For local development or custom environments:

```bash
# Clone repository
git clone https://github.com/your-username/multi-agent-pruning.git
cd multi-agent-pruning

# Create environment
conda create -n multi_agent_pruning python=3.10 -y
conda activate multi_agent_pruning

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Required: OpenAI API for LLM agents
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Alternative: OpenRouter API
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Dataset paths (adjust for your system)
IMAGENET_PATH=/path/to/imagenet
CIFAR10_PATH=/path/to/cifar10

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=multi_agent_pruning
WANDB_ENTITY=your_username

# Cache and output directories
CACHE_DIR=./cache
RESULTS_DIR=./results
MODELS_DIR=./models

# Hardware configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=16
```

### Model Setup

Download pretrained models for your experiments:

```bash
# Essential models for quick start
python tools/download_models.py --models deit_small,resnet50,convnext_tiny

# All models for comprehensive experiments
python tools/download_models.py --all

# Specific paper table models
python tools/download_models.py --table1  # DeiT models (Table 1)
python tools/download_models.py --table2  # ConvNext models (Table 2)

# Verify downloads
python tools/download_models.py --verify
```

## Basic Usage

### Simple Pruning Example

Here's the simplest way to prune a model:

```python
from multi_agent_pruning import MultiAgentPruner

# Initialize the framework
pruner = MultiAgentPruner(config_path="configs/experiments/enhanced_multi_agent.yaml")

# Prune DeiT-Small to 50% parameter reduction
result = pruner.prune_model(
    model_name="deit_small",
    dataset="imagenet",
    target_ratio=0.5,
    output_dir="./results/deit_small_50pct"
)

# Print results
print(f"✅ Pruning completed!")
print(f"Parameter reduction: {result['params_reduction']:.1%}")
print(f"MACs reduction: {result['macs_reduction']:.1%}")
print(f"Final accuracy: {result['final_accuracy']:.1%}")
print(f"Convergence iterations: {result['convergence_iterations']}")
print(f"Total time: {result['total_time_minutes']:.1f} minutes")
```

### Command Line Interface

The framework provides convenient CLI commands:

```bash
# Basic pruning
multi-agent-prune --model deit_small --dataset imagenet --target-ratio 0.5

# With custom output directory
multi-agent-prune --model resnet50 --dataset cifar10 --target-ratio 0.3 \
                  --output-dir ./results/resnet50_cifar10

# Enable verbose logging
multi-agent-prune --model convnext_tiny --dataset imagenet --target-ratio 0.4 \
                  --verbose

# Use custom configuration
multi-agent-prune --model deit_base --dataset imagenet --target-ratio 0.6 \
                  --config ./configs/custom_config.yaml
```

### Batch Processing

Process multiple models efficiently:

```python
from multi_agent_pruning import MultiAgentPruner

pruner = MultiAgentPruner()

# Define experiment configurations
experiments = [
    {"model_name": "deit_small", "dataset": "imagenet", "target_ratio": 0.3},
    {"model_name": "deit_small", "dataset": "imagenet", "target_ratio": 0.5},
    {"model_name": "deit_small", "dataset": "imagenet", "target_ratio": 0.7},
    {"model_name": "resnet50", "dataset": "imagenet", "target_ratio": 0.5},
    {"model_name": "convnext_tiny", "dataset": "imagenet", "target_ratio": 0.4},
]

# Run all experiments
results = {}
for exp in experiments:
    print(f"🚀 Running {exp['model_name']} @ {exp['target_ratio']:.0%} reduction...")
    
    result = pruner.prune_model(
        model_name=exp["model_name"],
        dataset=exp["dataset"],
        target_ratio=exp["target_ratio"],
        output_dir=f"./results/{exp['model_name']}_{exp['target_ratio']:.0%}"
    )
    
    results[f"{exp['model_name']}_{exp['target_ratio']:.0%}"] = result
    
    print(f"✅ Completed: {result['final_accuracy']:.1%} accuracy")

# Summary
print("\n📊 EXPERIMENT SUMMARY")
print("=" * 60)
for exp_name, result in results.items():
    print(f"{exp_name:<20}: {result['final_accuracy']:.1%} acc, "
          f"{result['params_reduction']:.1%} reduction")
```

## Configuration Guide

### Configuration File Structure

The framework uses hierarchical YAML configuration files. Here's the complete structure:

```yaml
# configs/experiments/my_experiment.yaml

# Experiment metadata
experiment:
  name: "my_pruning_experiment"
  description: "Custom pruning experiment"
  version: "1.0"
  seed: 42
  deterministic: true
  
  # Experiment tracking
  wandb:
    enabled: true
    project: "my_pruning_project"
    tags: ["custom", "experiment"]

# Dataset configurations
datasets:
  imagenet:
    path: "/path/to/imagenet"
    num_classes: 1000
    input_size: 224
    batch_size: 256
    num_workers: 16
    
    # Dataset-specific safety limits
    safety_limits:
      max_mlp_pruning: 0.15      # Conservative for ImageNet
      max_attention_pruning: 0.10
      max_overall_pruning: 0.60
      min_accuracy_threshold: 0.40
    
    # Evaluation settings
    evaluation:
      metrics: ["top1_accuracy", "top5_accuracy"]
      batch_size: 128
  
  cifar10:
    path: "/path/to/cifar10"
    num_classes: 10
    input_size: 32
    batch_size: 512
    
    # More aggressive limits for CIFAR-10
    safety_limits:
      max_mlp_pruning: 0.70
      max_attention_pruning: 0.60
      max_overall_pruning: 0.90
      min_accuracy_threshold: 0.70

# Model configurations
models:
  deit_small:
    name: "deit_small_patch16_224.fb_in1k"
    architecture_type: "vision_transformer"
    pretrained: true
    
    # Model-specific settings
    pruning_strategy: "attention_mlp"
    sensitive_layers: ["patch_embed", "head", "pos_embed"]
    
    # Recommended parameters
    recommended:
      importance_criterion: "taylor"
      round_to: 2
      initial_ratio: 0.3

# Multi-agent system configuration
agents:
  # Global settings
  global:
    llm_model: "gpt-4o-mini"
    temperature: 0.1
    max_tokens: 2000
    max_retries: 3
    enable_safety_checks: true
    safety_multiplier: 0.8
  
  # Individual agent configurations
  profiling_agent:
    enable_caching: true
    cache_duration: 3600  # 1 hour
    precompute_dependencies: true
    
    dependency_analysis:
      enabled: true
      coupling_detection: true
      isomorphic_grouping: true
  
  master_agent:
    max_iterations: 5
    convergence_threshold: 0.005  # 0.5% accuracy improvement
    target_tolerance: 0.01        # 1% parameter reduction tolerance
    
    exploration_strategies: ["conservative", "moderate", "aggressive"]
    adaptive_strategy: true
  
  analysis_agent:
    architecture_specific: true
    safety_enforcement: "strict"
    
    parameter_search:
      importance_criteria: ["taylor", "l1norm", "l2norm"]
      round_to_options: [1, 2, 4, 8]
  
  pruning_agent:
    backend: "torch_pruning"
    global_pruning: true
    validation_steps: true
    memory_efficient: true
  
  finetuning_agent:
    optimizer: "adamw"
    scheduler: "cosine"
    
    # Dataset-specific training settings
    imagenet:
      epochs: 5
      learning_rate: 0.001
      weight_decay: 0.05
      batch_size: 256
    
    cifar10:
      epochs: 3
      learning_rate: 0.01
      weight_decay: 0.0001
      batch_size: 512
    
    early_stopping:
      enabled: true
      patience: 2
      min_delta: 0.001
  
  evaluation_agent:
    comprehensive_metrics: true
    
    metrics:
      accuracy: true
      parameter_reduction: true
      macs_reduction: true
      memory_usage: true
      inference_time: true

# Precomputation settings
precomputation:
  enabled: true
  cache_dir: "./cache"
  
  model_analysis:
    enabled: true
    architecture_detection: true
    layer_analysis: true
    dependency_graph: true
    isomorphic_groups: true
  
  importance_scores:
    enabled: true
    criteria: ["taylor", "l1norm", "l2norm"]
    cache_duration: 7200  # 2 hours

# Hardware configuration
hardware:
  gpu:
    device: "cuda"
    memory_limit: "40GB"
    mixed_precision: true
    gradient_checkpointing: true
  
  cpu:
    num_workers: 16
    memory_limit: "64GB"

# Logging configuration
logging:
  level: "INFO"
  file_logging:
    enabled: true
    log_dir: "./logs"
  
  agent_logging:
    reasoning_traces: true
    conversation_history: true
    timing_profiler: true

# Output configuration
output:
  results_dir: "./results"
  save_models: true
  save_checkpoints: true
  save_plots: true
  
  export_formats:
    csv: true
    json: true
    wandb: true
```

### Environment-Specific Configurations

Create different configurations for different environments:

#### Development Configuration (`configs/dev.yaml`)

```yaml
# Lightweight configuration for development
experiment:
  name: "dev_experiment"
  
agents:
  master_agent:
    max_iterations: 2  # Faster for testing
    
  finetuning_agent:
    imagenet:
      epochs: 1  # Quick fine-tuning
    cifar10:
      epochs: 1

precomputation:
  enabled: true  # Still use caching for efficiency

logging:
  level: "DEBUG"  # Verbose logging for debugging
```

#### Production Configuration (`configs/production.yaml`)

```yaml
# Full configuration for production runs
experiment:
  name: "production_experiment"
  
agents:
  master_agent:
    max_iterations: 10  # Thorough optimization
    
  finetuning_agent:
    imagenet:
      epochs: 10  # Full fine-tuning
    cifar10:
      epochs: 5

precomputation:
  enabled: true
  
  # Extended cache durations for production
  model_analysis:
    cache_duration: 86400  # 24 hours
  importance_scores:
    cache_duration: 43200  # 12 hours

hardware:
  distributed:
    enabled: true  # Use multiple GPUs
    world_size: 4

logging:
  level: "INFO"
```

### Configuration Inheritance

You can create configurations that inherit from base configurations:

```yaml
# configs/my_experiment.yaml

# Inherit from base configuration
_base_: "enhanced_multi_agent.yaml"

# Override specific settings
experiment:
  name: "my_custom_experiment"

agents:
  master_agent:
    max_iterations: 8  # Custom iteration count
    
datasets:
  imagenet:
    batch_size: 128  # Smaller batch size for limited memory
```

## Advanced Features

### Custom Agent Implementation

You can extend the framework with custom agents for specialized pruning strategies:

```python
from multi_agent_pruning.agents import BaseAgent, AgentResponse
from typing import Dict, Any

class SpecializedPruningAgent(BaseAgent):
    """Custom agent for specialized pruning strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.specialized_param = config.get('specialized_param', 1.0)
    
    def get_agent_role(self) -> str:
        return """Specialized pruning agent that applies domain-specific 
        knowledge for optimal pruning decisions."""
    
    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        return f"""You are a specialized pruning agent with expertise in 
        {context.get('domain', 'general')} applications. Your specialized 
        parameter is {self.specialized_param}."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specialized pruning analysis."""
        
        # Custom analysis logic
        model = input_data.get('model')
        domain = input_data.get('domain', 'general')
        
        # Apply specialized analysis
        specialized_analysis = self._perform_specialized_analysis(model, domain)
        
        # Query LLM for strategic insights
        context = input_data.copy()
        context['specialized_analysis'] = specialized_analysis
        
        system_prompt = self.get_system_prompt(context)
        user_prompt = f"""
        Based on the specialized analysis for {domain} domain:
        {specialized_analysis}
        
        Provide strategic recommendations for pruning optimization.
        """
        
        try:
            llm_response = self._query_llm_with_retries(system_prompt, user_prompt)
            agent_response = self.parse_llm_response(llm_response, context)
            
            return {
                'success': True,
                'agent_name': self.agent_name,
                'specialized_analysis': specialized_analysis,
                'llm_insights': agent_response.to_dict(),
                'domain': domain
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'agent_name': self.agent_name
            }
    
    def _perform_specialized_analysis(self, model, domain: str) -> Dict[str, Any]:
        """Perform domain-specific analysis."""
        
        if domain == 'medical_imaging':
            # Medical imaging specific analysis
            return {
                'critical_features': ['early_conv_layers', 'attention_heads'],
                'safety_multiplier': 0.5,  # Very conservative
                'recommended_criterion': 'taylor'
            }
        elif domain == 'autonomous_driving':
            # Autonomous driving specific analysis
            return {
                'critical_features': ['object_detection_layers', 'depth_estimation'],
                'safety_multiplier': 0.6,
                'recommended_criterion': 'l1norm'
            }
        else:
            # General analysis
            return {
                'critical_features': ['classifier', 'embedding'],
                'safety_multiplier': 0.8,
                'recommended_criterion': 'taylor'
            }
    
    def parse_llm_response(self, response: str, context: Dict[str, Any]) -> AgentResponse:
        """Parse LLM response for specialized insights."""
        
        return AgentResponse(
            success=True,
            reasoning=response,
            recommendations={'specialized_insights': response},
            confidence=0.85,
            safety_checks={'domain_specific': True},
            warnings=[],
            timestamp=context.get('timestamp', ''),
            agent_name=self.agent_name
        )

# Usage example
specialized_agent = SpecializedPruningAgent(config={
    'specialized_param': 1.5
})

result = specialized_agent.execute({
    'model': model,
    'domain': 'medical_imaging'
})
```

### Custom Baseline Methods

Implement custom pruning baselines for comparison:

```python
from multi_agent_pruning.baselines import BasePruningMethod
import torch
import torch.nn as nn
from typing import Dict, Any

class GradientBasedPruning(BasePruningMethod):
    """Custom gradient-based pruning method."""
    
    def __init__(self, gradient_threshold: float = 0.01):
        super().__init__()
        self.gradient_threshold = gradient_threshold
        self.method_name = "gradient_based"
    
    def prune_to_target_macs(self, model_name: str, target_macs: float, 
                           dataset: str) -> Dict[str, Any]:
        """Prune model using gradient-based importance."""
        
        # Load model
        model = self.load_model(model_name)
        original_macs = self.compute_macs(model)
        
        # Compute gradient-based importance
        importance_scores = self._compute_gradient_importance(model, dataset)
        
        # Determine pruning ratio to achieve target MACs
        target_ratio = 1.0 - (target_macs / original_macs)
        
        # Apply pruning based on gradient importance
        pruned_model = self._apply_gradient_pruning(
            model, importance_scores, target_ratio
        )
        
        # Fine-tune the pruned model
        final_accuracy = self.fine_tune_and_evaluate(pruned_model, dataset)
        
        # Compute final metrics
        final_macs = self.compute_macs(pruned_model)
        final_params = sum(p.numel() for p in pruned_model.parameters())
        original_params = sum(p.numel() for p in model.parameters())
        
        return {
            'final_accuracy': final_accuracy,
            'final_macs': final_macs,
            'params_reduction': 1.0 - (final_params / original_params),
            'macs_reduction': 1.0 - (final_macs / original_macs),
            'method': self.method_name,
            'gradient_threshold': self.gradient_threshold
        }
    
    def _compute_gradient_importance(self, model: nn.Module, 
                                   dataset: str) -> Dict[str, torch.Tensor]:
        """Compute gradient-based importance scores."""
        
        # Set model to training mode
        model.train()
        
        # Load sample data
        dataloader = self.get_dataloader(dataset, batch_size=32, num_samples=1000)
        
        importance_scores = {}
        
        # Accumulate gradients over sample data
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit samples for efficiency
                break
                
            data, target = data.cuda(), target.cuda()
            
            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in importance_scores:
                        importance_scores[name] = torch.zeros_like(param)
                    importance_scores[name] += param.grad.abs()
            
            # Clear gradients
            model.zero_grad()
        
        # Normalize importance scores
        for name in importance_scores:
            importance_scores[name] /= min(10, len(dataloader))
        
        return importance_scores
    
    def _apply_gradient_pruning(self, model: nn.Module, 
                              importance_scores: Dict[str, torch.Tensor],
                              target_ratio: float) -> nn.Module:
        """Apply pruning based on gradient importance."""
        
        # Collect all importance values
        all_importances = []
        for name, scores in importance_scores.items():
            all_importances.extend(scores.flatten().cpu().numpy())
        
        # Determine threshold for target ratio
        threshold = np.percentile(all_importances, target_ratio * 100)
        
        # Apply pruning
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in importance_scores:
                    mask = importance_scores[name] > threshold
                    param.data *= mask.float()
        
        return model

# Register the custom method
custom_method = GradientBasedPruning(gradient_threshold=0.01)

# Use in comparison
from multi_agent_pruning import MultiAgentPruner

pruner = MultiAgentPruner()
comparison = pruner.compare_methods(
    model_name="deit_small",
    dataset="imagenet",
    target_ratio=0.5,
    methods=["multi_agent", "magnitude_l1", "gradient_based"],
    custom_methods={"gradient_based": custom_method}
)
```

### Advanced Caching Strategies

Implement sophisticated caching for complex workflows:

```python
from multi_agent_pruning.utils.caching import CacheManager
import hashlib
import pickle
from pathlib import Path

class AdvancedCacheManager(CacheManager):
    """Advanced caching with dependency tracking and invalidation."""
    
    def __init__(self, cache_dir: str = "./advanced_cache"):
        super().__init__(cache_dir)
        self.dependency_graph = {}
        self.cache_metadata = {}
    
    def cache_with_dependencies(self, key: str, value: Any, 
                              dependencies: List[str] = None,
                              ttl: int = 3600):
        """Cache value with dependency tracking."""
        
        # Store the value
        self.set(key, value, ttl)
        
        # Track dependencies
        if dependencies:
            self.dependency_graph[key] = dependencies
            
            # Store metadata
            self.cache_metadata[key] = {
                'created_at': time.time(),
                'dependencies': dependencies,
                'ttl': ttl,
                'checksum': self._compute_checksum(value)
            }
    
    def invalidate_dependents(self, key: str):
        """Invalidate all cache entries that depend on this key."""
        
        to_invalidate = []
        
        # Find all dependents
        for cache_key, deps in self.dependency_graph.items():
            if key in deps:
                to_invalidate.append(cache_key)
        
        # Recursively invalidate
        for dependent_key in to_invalidate:
            self.delete(dependent_key)
            self.invalidate_dependents(dependent_key)  # Recursive
    
    def validate_cache_integrity(self) -> Dict[str, bool]:
        """Validate integrity of cached data."""
        
        integrity_report = {}
        
        for key, metadata in self.cache_metadata.items():
            try:
                # Check if cache entry exists
                if not self.exists(key):
                    integrity_report[key] = False
                    continue
                
                # Check TTL
                if time.time() - metadata['created_at'] > metadata['ttl']:
                    integrity_report[key] = False
                    self.delete(key)
                    continue
                
                # Check checksum
                cached_value = self.get(key)
                current_checksum = self._compute_checksum(cached_value)
                
                if current_checksum != metadata['checksum']:
                    integrity_report[key] = False
                    self.delete(key)
                    continue
                
                integrity_report[key] = True
                
            except Exception as e:
                integrity_report[key] = False
                logger.warning(f"Cache integrity check failed for {key}: {e}")
        
        return integrity_report
    
    def _compute_checksum(self, value: Any) -> str:
        """Compute checksum for cache validation."""
        serialized = pickle.dumps(value)
        return hashlib.md5(serialized).hexdigest()

# Usage example
cache_manager = AdvancedCacheManager()

# Cache model analysis with dependencies
model_analysis = perform_model_analysis(model)
cache_manager.cache_with_dependencies(
    key=f"model_analysis_{model_name}",
    value=model_analysis,
    dependencies=[f"model_{model_name}", "analysis_config"],
    ttl=7200  # 2 hours
)

# If model changes, invalidate dependent caches
cache_manager.invalidate_dependents(f"model_{model_name}")

# Validate cache integrity
integrity_report = cache_manager.validate_cache_integrity()
print(f"Cache integrity: {sum(integrity_report.values())}/{len(integrity_report)} valid")
```

## HPC Integration

### SLURM Job Management

The framework provides comprehensive SLURM integration for HPC environments:

#### Basic SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=multi_agent_pruning
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load modules
module load python/3.10
module load cuda/11.8
module load gcc/9.3.0

# Activate environment
source ~/.bashrc
conda activate multi_agent_pruning

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Create output directories
mkdir -p logs results

# Run experiment
python experiments/paper_reproduction.py \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --output_dir results/paper_reproduction_$(date +%Y%m%d_%H%M%S)
```

#### Array Jobs for Parameter Sweeps

```bash
#!/bin/bash
#SBATCH --job-name=pruning_sweep
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --array=1-15  # 15 different configurations
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load environment
module load python/3.10 cuda/11.8
conda activate multi_agent_pruning

# Define parameter combinations
MODELS=("deit_small" "deit_base" "convnext_tiny" "convnext_small" "resnet50")
RATIOS=(0.3 0.5 0.7)

# Calculate indices
MODEL_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 3 ))
RATIO_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 3 ))

MODEL=${MODELS[$MODEL_IDX]}
RATIO=${RATIOS[$RATIO_IDX]}

echo "Running: $MODEL with ratio $RATIO"

# Run experiment
multi-agent-prune \
    --model $MODEL \
    --dataset imagenet \
    --target-ratio $RATIO \
    --output-dir results/${MODEL}_${RATIO}_${SLURM_ARRAY_TASK_ID} \
    --config configs/experiments/enhanced_multi_agent.yaml
```

#### Multi-GPU Distributed Jobs

```bash
#!/bin/bash
#SBATCH --job-name=distributed_pruning
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G

# Load modules
module load python/3.10 cuda/11.8 nccl/2.12

# Set up distributed environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Activate environment
conda activate multi_agent_pruning

# Run distributed experiment
srun python experiments/distributed_paper_reproduction.py \
    --config configs/experiments/distributed_multi_agent.yaml \
    --world-size $WORLD_SIZE \
    --rank $RANK
```

### HPC-Specific Optimizations

#### Memory Management

```python
# configs/hpc_optimized.yaml
hardware:
  gpu:
    memory_limit: "40GB"  # A100 GPU
    mixed_precision: true
    gradient_checkpointing: true
    memory_efficient_attention: true
  
  cpu:
    num_workers: 32  # High-core HPC nodes
    memory_limit: "256GB"
    
precomputation:
  # Aggressive caching for HPC
  enabled: true
  cache_dir: "/scratch/$USER/multi_agent_cache"  # Fast scratch storage
  
  model_analysis:
    cache_duration: 86400  # 24 hours
  importance_scores:
    cache_duration: 43200  # 12 hours

# Use fast local storage for temporary files
output:
  results_dir: "/scratch/$USER/results"
  temp_dir: "/scratch/$USER/temp"
```

#### Module Loading Automation

```python
# tools/hpc_utils.py
import subprocess
import os
from typing import List, Dict

class HPCModuleManager:
    """Manages HPC module loading and environment setup."""
    
    def __init__(self):
        self.loaded_modules = []
    
    def load_modules(self, modules: List[str]) -> Dict[str, bool]:
        """Load required HPC modules."""
        
        results = {}
        
        for module in modules:
            try:
                result = subprocess.run(
                    ['module', 'load', module],
                    capture_output=True,
                    text=True,
                    check=True
                )
                results[module] = True
                self.loaded_modules.append(module)
                print(f"✅ Loaded module: {module}")
                
            except subprocess.CalledProcessError as e:
                results[module] = False
                print(f"❌ Failed to load module {module}: {e.stderr}")
        
        return results
    
    def setup_cuda_environment(self, cuda_version: str = "11.8"):
        """Setup CUDA environment variables."""
        
        cuda_modules = [
            f"cuda/{cuda_version}",
            "cudnn/8.6.0",
            "nccl/2.12.12"
        ]
        
        self.load_modules(cuda_modules)
        
        # Set CUDA environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['NCCL_DEBUG'] = 'INFO'
        
    def setup_python_environment(self, python_version: str = "3.10"):
        """Setup Python environment."""
        
        python_modules = [
            f"python/{python_version}",
            "gcc/9.3.0",
            "cmake/3.20.0"
        ]
        
        self.load_modules(python_modules)
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get information about the current compute node."""
        
        info = {}
        
        # Get SLURM job information
        slurm_vars = [
            'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_NTASKS',
            'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE'
        ]
        
        for var in slurm_vars:
            info[var] = os.environ.get(var, 'N/A')
        
        # Get GPU information
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')
                info['gpus'] = [line.split(', ') for line in gpu_info]
        except:
            info['gpus'] = []
        
        return info

# Usage in HPC scripts
hpc_manager = HPCModuleManager()
hpc_manager.setup_cuda_environment("11.8")
hpc_manager.setup_python_environment("3.10")

node_info = hpc_manager.get_node_info()
print(f"Running on node with {len(node_info['gpus'])} GPUs")
```

### Data Management on HPC

#### Efficient Data Loading

```python
# utils/hpc_data_loader.py
import os
import shutil
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

class HPCDataManager:
    """Manages data efficiently on HPC systems."""
    
    def __init__(self, scratch_dir: Optional[str] = None):
        self.scratch_dir = Path(scratch_dir or f"/scratch/{os.environ.get('USER', 'unknown')}")
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        
    def copy_dataset_to_scratch(self, dataset_path: str, dataset_name: str) -> str:
        """Copy dataset to fast scratch storage."""
        
        source_path = Path(dataset_path)
        scratch_path = self.scratch_dir / dataset_name
        
        if scratch_path.exists():
            logger.info(f"Dataset {dataset_name} already in scratch: {scratch_path}")
            return str(scratch_path)
        
        logger.info(f"Copying {dataset_name} to scratch storage...")
        logger.info(f"Source: {source_path}")
        logger.info(f"Destination: {scratch_path}")
        
        # Use rsync for efficient copying
        import subprocess
        result = subprocess.run([
            'rsync', '-av', '--progress',
            str(source_path) + '/',
            str(scratch_path) + '/'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Dataset copied to scratch: {scratch_path}")
            return str(scratch_path)
        else:
            logger.error(f"❌ Failed to copy dataset: {result.stderr}")
            return dataset_path  # Fall back to original path
    
    def create_optimized_dataloader(self, dataset_path: str, 
                                  batch_size: int = 256,
                                  num_workers: Optional[int] = None) -> DataLoader:
        """Create optimized DataLoader for HPC environment."""
        
        if num_workers is None:
            # Use all available CPU cores
            num_workers = min(32, os.cpu_count() or 8)
        
        # Copy to scratch if beneficial
        if '/scratch/' not in dataset_path:
            dataset_path = self.copy_dataset_to_scratch(dataset_path, 'imagenet')
        
        # Create dataset
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        
        # Create optimized DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        logger.info(f"Created optimized DataLoader:")
        logger.info(f"  Dataset size: {len(dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num workers: {num_workers}")
        logger.info(f"  Data path: {dataset_path}")
        
        return dataloader
    
    def cleanup_scratch(self):
        """Clean up scratch directory."""
        
        if self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir)
            logger.info(f"Cleaned up scratch directory: {self.scratch_dir}")

# Usage in experiments
data_manager = HPCDataManager()

# Create optimized data loader
train_loader = data_manager.create_optimized_dataloader(
    dataset_path=os.environ['IMAGENET_PATH'],
    batch_size=256
)

# ... run experiments ...

# Cleanup when done
data_manager.cleanup_scratch()
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
```python
# In your config file
datasets:
  imagenet:
    batch_size: 128  # Reduce from 256
    
agents:
  finetuning_agent:
    imagenet:
      batch_size: 64  # Even smaller for fine-tuning
```

2. **Enable memory optimizations:**
```python
hardware:
  gpu:
    mixed_precision: true
    gradient_checkpointing: true
    memory_efficient_attention: true
```

3. **Use gradient accumulation:**
```python
agents:
  finetuning_agent:
    gradient_accumulation_steps: 4  # Effective batch size = batch_size * 4
```

#### Issue 2: LLM API Rate Limits

**Symptoms:**
```
openai.error.RateLimitError: Rate limit reached
```

**Solutions:**

1. **Add retry logic with exponential backoff:**
```python
agents:
  global:
    max_retries: 5
    retry_delay: 1.0
    exponential_backoff: true
```

2. **Use alternative API providers:**
```bash
# In .env file
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

3. **Reduce LLM usage frequency:**
```python
agents:
  profiling_agent:
    enable_caching: true  # Cache LLM responses
    
  master_agent:
    llm_frequency: "adaptive"  # Only query LLM when needed
```

#### Issue 3: Model Download Failures

**Symptoms:**
```
ConnectionError: Failed to download model weights
```

**Solutions:**

1. **Use manual download with verification:**
```bash
python tools/download_models.py --models deit_small --verify --force
```

2. **Set up offline model cache:**
```bash
# Download models on login node, then copy to compute nodes
python tools/download_models.py --all --cache-dir /shared/models
```

3. **Use alternative model sources:**
```python
# In config file
models:
  deit_small:
    source: "local"  # Use local checkpoint
    path: "/path/to/local/model.pth"
```

#### Issue 4: Slow Convergence

**Symptoms:**
- Many iterations without improvement
- High convergence times

**Solutions:**

1. **Adjust convergence thresholds:**
```python
agents:
  master_agent:
    convergence_threshold: 0.01  # Less strict
    max_iterations: 3  # Fewer iterations
```

2. **Use more aggressive exploration:**
```python
agents:
  master_agent:
    exploration_strategies: ["aggressive"]
    adaptive_strategy: false
```

3. **Enable precomputation:**
```python
precomputation:
  enabled: true
  importance_scores:
    enabled: true  # Precompute importance scores
```

#### Issue 5: Inconsistent Results

**Symptoms:**
- Different results across runs
- High variance in accuracy

**Solutions:**

1. **Set deterministic mode:**
```python
experiment:
  seed: 42
  deterministic: true
```

2. **Increase number of runs:**
```python
# For validation
validator.validate_all_targets(num_runs=5)  # More runs for stability
```

3. **Use ensemble methods:**
```python
agents:
  evaluation_agent:
    ensemble_evaluation: true
    num_ensemble_runs: 3
```

### Debugging Tools

#### Enable Detailed Logging

```python
# In config file
logging:
  level: "DEBUG"
  agent_logging:
    reasoning_traces: true
    conversation_history: true
    timing_profiler: true
    
  file_logging:
    enabled: true
    log_dir: "./debug_logs"
```

#### Performance Profiling

```python
from multi_agent_pruning.utils.profiler import TimingProfiler

# Enable global profiling
profiler = TimingProfiler()

# Profile specific operations
with profiler.timer("model_loading"):
    model = load_model("deit_small")

with profiler.timer("pruning_execution"):
    result = pruner.prune_model(...)

# Get detailed timing report
profiler.get_summary()
```

#### Memory Monitoring

```python
import psutil
import torch

def monitor_memory():
    """Monitor system and GPU memory usage."""
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent}% used ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")

# Use throughout your code
monitor_memory()
```

## Best Practices

### Configuration Management

1. **Use Environment-Specific Configs:**
```
configs/
├── base/
│   └── enhanced_multi_agent.yaml
├── dev/
│   └── quick_test.yaml
├── production/
│   └── full_experiment.yaml
└── hpc/
    └── slurm_optimized.yaml
```

2. **Version Control Your Configs:**
```bash
git add configs/
git commit -m "Add experiment configuration for paper reproduction"
```

3. **Use Configuration Inheritance:**
```yaml
# configs/my_experiment.yaml
_base_: "base/enhanced_multi_agent.yaml"

# Override only what you need
experiment:
  name: "my_custom_experiment"
```

### Experiment Management

1. **Use Descriptive Names:**
```python
experiment_name = f"deit_small_imagenet_{target_ratio:.0%}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

2. **Track Everything:**
```python
# Enable comprehensive tracking
wandb.init(
    project="multi_agent_pruning",
    name=experiment_name,
    config=config,
    tags=["paper_reproduction", "deit", "imagenet"]
)
```

3. **Save Intermediate Results:**
```python
# Save checkpoints during long runs
if iteration % 5 == 0:
    save_checkpoint(model, f"checkpoint_iter_{iteration}.pth")
```

### Performance Optimization

1. **Profile First, Optimize Second:**
```python
# Always profile to identify bottlenecks
with profiler.timer("full_workflow"):
    result = run_experiment()

profiler.get_summary()  # Identify slow components
```

2. **Use Appropriate Hardware:**
```python
# Match hardware to workload
if model_size > 100e6:  # Large models
    config['hardware']['gpu']['memory_limit'] = "80GB"  # A100 80GB
    config['hardware']['gpu']['mixed_precision'] = True
```

3. **Leverage Caching Aggressively:**
```python
# Cache everything that's reusable
precomputation:
  enabled: true
  model_analysis:
    cache_duration: 86400  # 24 hours
  importance_scores:
    cache_duration: 43200  # 12 hours
```

### Code Quality

1. **Write Tests for Custom Components:**
```python
def test_custom_agent():
    agent = CustomAgent()
    result = agent.execute(test_input)
    assert result['success'] == True
    assert 'custom_analysis' in result
```

2. **Use Type Hints:**
```python
def prune_model(model_name: str, target_ratio: float) -> Dict[str, Any]:
    """Prune model with type safety."""
    pass
```

3. **Document Your Configurations:**
```yaml
# configs/documented_config.yaml

# This configuration reproduces Table 1 from the paper
# with enhanced safety constraints for production use
experiment:
  name: "table1_reproduction_enhanced"
  description: |
    Reproduces DeiT pruning results from Table 1 with
    additional safety constraints and validation
```

### Reproducibility

1. **Set All Random Seeds:**
```python
import random
import numpy as np
import torch

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

2. **Document Your Environment:**
```bash
# Save environment information
pip freeze > requirements_exact.txt
conda env export > environment.yml
nvidia-smi > gpu_info.txt
```

3. **Use Version Control:**
```bash
# Tag important experiments
git tag -a v1.0-paper-reproduction -m "Paper reproduction results"
git push origin v1.0-paper-reproduction
```

This comprehensive user guide provides everything needed to effectively use the Enhanced Multi-Agent LLM Pruning Framework, from basic usage to advanced customization and HPC deployment.

