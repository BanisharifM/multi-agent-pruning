# API Reference

This document provides comprehensive API documentation for the Enhanced Multi-Agent LLM Pruning Framework.

## Core Components

### MultiAgentPruner

The main interface for the pruning framework.

```python
from multi_agent_pruning import MultiAgentPruner

pruner = MultiAgentPruner(config_path="path/to/config.yaml")
```

#### Methods

##### `prune_model(model_name, dataset, target_ratio, **kwargs)`

Prune a model to achieve the specified parameter reduction ratio.

**Parameters:**
- `model_name` (str): Name of the model to prune (e.g., "deit_small", "resnet50")
- `dataset` (str): Target dataset ("imagenet", "cifar10") 
- `target_ratio` (float): Target parameter reduction ratio (0.0 to 1.0)
- `output_dir` (str, optional): Directory to save results
- `max_iterations` (int, optional): Maximum optimization iterations
- `enable_caching` (bool, optional): Enable result caching

**Returns:**
- `dict`: Pruning results containing:
  - `final_accuracy` (float): Achieved accuracy after fine-tuning
  - `params_reduction` (float): Actual parameter reduction ratio
  - `macs_reduction` (float): MACs reduction ratio
  - `convergence_iterations` (int): Iterations to convergence
  - `total_time_minutes` (float): Total execution time

**Example:**
```python
result = pruner.prune_model(
    model_name="deit_small",
    dataset="imagenet",
    target_ratio=0.5,
    output_dir="./results/deit_small_50pct"
)

print(f"Achieved {result['params_reduction']:.1%} reduction")
print(f"Final accuracy: {result['final_accuracy']:.1%}")
```

##### `compare_methods(model_name, dataset, target_ratio, methods=None)`

Compare multiple pruning methods on the same target.

**Parameters:**
- `model_name` (str): Model to compare methods on
- `dataset` (str): Target dataset
- `target_ratio` (float): Target compression ratio
- `methods` (list, optional): List of methods to compare

**Returns:**
- `dict`: Comparison results for each method

**Example:**
```python
comparison = pruner.compare_methods(
    model_name="deit_small",
    dataset="imagenet", 
    target_ratio=0.5,
    methods=["multi_agent", "magnitude_l1", "taylor"]
)

for method, result in comparison.items():
    print(f"{method}: {result['final_accuracy']:.1%} accuracy")
```

## Agent Classes

### ProfilingAgent

Analyzes model architecture and computes safety constraints.

```python
from multi_agent_pruning.agents import ProfilingAgent

agent = ProfilingAgent(config={'enable_caching': True})
```

#### Methods

##### `execute(input_data)`

Execute comprehensive model profiling.

**Parameters:**
- `input_data` (dict): Input containing:
  - `model` (torch.nn.Module): Model to profile
  - `model_name` (str): Model identifier
  - `dataset` (str): Target dataset
  - `input_size` (int, optional): Input image size

**Returns:**
- `dict`: Profiling results containing:
  - `profile` (ModelProfile): Comprehensive model analysis
  - `llm_analysis` (dict): LLM strategic insights
  - `cached` (bool): Whether result was cached
  - `timing` (dict): Performance timing information

**Example:**
```python
import timm

model = timm.create_model('deit_small_patch16_224', pretrained=True)
result = agent.execute({
    'model': model,
    'model_name': 'deit_small',
    'dataset': 'imagenet'
})

profile = result['profile']
print(f"Architecture: {profile['architecture_type']}")
print(f"Prunable layers: {len(profile['prunable_layers'])}")
print(f"Safety limits: {profile['safety_limits']}")
```

### MasterAgent

Coordinates the pruning workflow and makes strategic decisions.

```python
from multi_agent_pruning.agents import MasterAgent

agent = MasterAgent(config={
    'max_iterations': 5,
    'convergence_threshold': 0.005
})
```

#### Methods

##### `execute(input_data)`

Generate strategic pruning recommendations.

**Parameters:**
- `input_data` (dict): Input containing:
  - `profile_results` (dict): Results from ProfilingAgent
  - `target_ratio` (float): Target parameter reduction
  - `history` (list): Previous pruning attempts
  - `dataset_info` (dict): Dataset-specific information
  - `revision_number` (int): Current iteration number

**Returns:**
- `dict`: Strategic recommendations containing:
  - `strategy` (PruningStrategy): Detailed pruning strategy
  - `continue_iterations` (bool): Whether to continue optimization
  - `confidence` (float): Confidence in recommendations
  - `directives` (dict): Specific parameter directives

**Example:**
```python
result = agent.execute({
    'profile_results': profile_result['profile'],
    'target_ratio': 0.5,
    'history': [],
    'dataset_info': {
        'num_classes': 1000,
        'safety_limits': {'max_overall_pruning': 0.8}
    },
    'revision_number': 0
})

strategy = result['strategy']
print(f"Recommended ratio: {strategy['pruning_ratio']:.2f}")
print(f"Importance criterion: {strategy['importance_criterion']}")
print(f"Continue: {result['continue_iterations']}")
```

### AnalysisAgent

Optimizes pruning parameters for specific architectures.

```python
from multi_agent_pruning.agents import AnalysisAgent

agent = AnalysisAgent(config={'safety_enforcement': 'strict'})
```

### PruningAgent

Executes the actual pruning operations.

```python
from multi_agent_pruning.agents import PruningAgent

agent = PruningAgent(config={
    'backend': 'torch_pruning',
    'global_pruning': True
})
```

### FineTuningAgent

Handles post-pruning model fine-tuning.

```python
from multi_agent_pruning.agents import FineTuningAgent

agent = FineTuningAgent(config={
    'optimizer': 'adamw',
    'scheduler': 'cosine'
})
```

### EvaluationAgent

Evaluates pruned models and computes comprehensive metrics.

```python
from multi_agent_pruning.agents import EvaluationAgent

agent = EvaluationAgent(config={'comprehensive_metrics': True})
```

## Utility Classes

### DependencyAnalyzer

Analyzes layer dependencies and coupling constraints.

```python
from multi_agent_pruning.core import DependencyAnalyzer

analyzer = DependencyAnalyzer(model)
dependencies = analyzer.get_dependency_graph()
constraints = analyzer.get_coupling_constraints()
```

#### Methods

##### `get_dependency_graph()`

Returns the complete dependency graph between layers.

**Returns:**
- `dict`: Dependency relationships between layers

##### `get_coupling_constraints()`

Returns coupling constraints that must be preserved during pruning.

**Returns:**
- `list`: List of coupling constraint descriptions

##### `get_coupled_layers(layer_name)`

Get all layers that must be pruned together with the specified layer.

**Parameters:**
- `layer_name` (str): Name of the layer to analyze

**Returns:**
- `list`: Names of coupled layers

### IsomorphicAnalyzer

Creates isomorphic groups of similar layers.

```python
from multi_agent_pruning.core import IsomorphicAnalyzer

analyzer = IsomorphicAnalyzer(model)
groups = analyzer.create_isomorphic_groups(target_ratio=0.5)
```

#### Methods

##### `create_isomorphic_groups(target_ratio, group_ratios=None)`

Create dependency-aware isomorphic groups.

**Parameters:**
- `target_ratio` (float): Target pruning ratio
- `group_ratios` (dict, optional): Custom ratios for different group types

**Returns:**
- `dict`: Dictionary of IsomorphicGroup objects

### TimingProfiler

Profiles execution time and identifies bottlenecks.

```python
from multi_agent_pruning.utils import TimingProfiler

profiler = TimingProfiler()

with profiler.timer("operation_name"):
    # Your code here
    pass

profiler.get_summary()  # Print timing summary
```

#### Methods

##### `timer(name)`

Context manager for timing code blocks.

**Parameters:**
- `name` (str): Name of the operation being timed

**Usage:**
```python
with profiler.timer("model_loading"):
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
```

##### `start_timer(name)` / `end_timer(name)`

Manual timer control for complex scenarios.

**Parameters:**
- `name` (str): Timer identifier

**Usage:**
```python
profiler.start_timer("training")
# ... training code ...
elapsed = profiler.end_timer("training")
```

##### `get_summary()`

Print comprehensive timing summary with bottleneck analysis.

## Configuration System

### Loading Configuration

```python
from multi_agent_pruning.config import load_config

config = load_config("configs/experiments/enhanced_multi_agent.yaml")
```

### Configuration Structure

The configuration system supports hierarchical YAML files:

```yaml
# Main experiment configuration
experiment:
  name: "experiment_name"
  seed: 42
  
# Agent-specific configurations  
agents:
  profiling_agent:
    enable_caching: true
    cache_duration: 3600
    
  master_agent:
    max_iterations: 5
    convergence_threshold: 0.005
    
# Dataset configurations
datasets:
  imagenet:
    path: "/path/to/imagenet"
    batch_size: 256
    safety_limits:
      max_mlp_pruning: 0.15
      max_attention_pruning: 0.10
```

### Environment Variables

The framework supports environment variable substitution:

```yaml
datasets:
  imagenet:
    path: ${IMAGENET_PATH}  # Reads from environment
    
openai:
  api_key: ${OPENAI_API_KEY}
```

## Baseline Methods

### MagnitudePruning

Traditional magnitude-based pruning.

```python
from multi_agent_pruning.baselines import MagnitudePruning

pruner = MagnitudePruning(criterion='l1')  # or 'l2'
result = pruner.prune_to_target_macs(
    model_name="deit_small",
    target_macs=2.3e9,
    dataset="imagenet"
)
```

### TaylorPruning

Taylor expansion-based importance pruning.

```python
from multi_agent_pruning.baselines import TaylorPruning

pruner = TaylorPruning()
result = pruner.prune_to_target_macs(
    model_name="deit_small", 
    target_macs=2.3e9,
    dataset="imagenet"
)
```

### IsomorphicOriginal

Faithful reproduction of the original isomorphic pruning method.

```python
from multi_agent_pruning.baselines import IsomorphicOriginal

pruner = IsomorphicOriginal()
result = pruner.prune_to_target_macs(
    model_name="deit_small",
    target_macs=2.3e9, 
    dataset="imagenet"
)
```

## Validation Framework

### PaperReproductionValidator

Validates paper reproduction and baseline comparisons.

```python
from multi_agent_pruning.validation import PaperReproductionValidator

validator = PaperReproductionValidator(
    config_path="configs/experiments/enhanced_multi_agent.yaml",
    output_dir="./validation_results"
)
```

#### Methods

##### `validate_all_targets(methods=None, num_runs=3)`

Validate all paper targets with specified methods.

**Parameters:**
- `methods` (list, optional): Methods to validate
- `num_runs` (int): Number of runs for statistical significance

**Returns:**
- `dict`: Validation results for each method

##### `analyze_results()`

Analyze validation results and generate comprehensive report.

**Returns:**
- `dict`: Analysis containing:
  - `summary`: Overall statistics
  - `method_comparison`: Method performance comparison
  - `statistical_tests`: Significance test results
  - `paper_reproduction`: Reproduction analysis

##### `generate_visualizations()`

Generate comprehensive visualizations of validation results.

Creates plots in the output directory:
- Success rate comparison
- Accuracy vs MACs scatter plots  
- Convergence time analysis
- Paper reproduction heatmaps

## Metrics and Evaluation

### compute_macs(model, input_shape)

Compute multiply-accumulate operations for a model.

**Parameters:**
- `model` (torch.nn.Module): Model to analyze
- `input_shape` (tuple): Input tensor shape

**Returns:**
- `float`: Total MACs

**Example:**
```python
from multi_agent_pruning.utils.metrics import compute_macs

macs = compute_macs(model, (1, 3, 224, 224))
print(f"Model MACs: {macs/1e9:.2f}G")
```

### compute_params(model)

Count total parameters in a model.

**Parameters:**
- `model` (torch.nn.Module): Model to analyze

**Returns:**
- `int`: Total parameter count

**Example:**
```python
from multi_agent_pruning.utils.metrics import compute_params

params = compute_params(model)
print(f"Model parameters: {params/1e6:.1f}M")
```

## Error Handling

### Common Exceptions

#### `PruningError`

Raised when pruning operations fail.

```python
from multi_agent_pruning.exceptions import PruningError

try:
    result = pruner.prune_model(...)
except PruningError as e:
    print(f"Pruning failed: {e}")
```

#### `ConfigurationError`

Raised for configuration-related issues.

```python
from multi_agent_pruning.exceptions import ConfigurationError

try:
    config = load_config("invalid_config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

#### `ValidationError`

Raised during validation failures.

```python
from multi_agent_pruning.exceptions import ValidationError

try:
    validator.validate_all_targets()
except ValidationError as e:
    print(f"Validation failed: {e}")
```

## Advanced Usage

### Custom Agent Implementation

You can extend the framework with custom agents:

```python
from multi_agent_pruning.agents import BaseAgent, AgentResponse

class CustomAgent(BaseAgent):
    def get_agent_role(self) -> str:
        return "Custom pruning agent"
    
    def execute(self, input_data: dict) -> dict:
        # Custom implementation
        return {
            'success': True,
            'agent_name': self.agent_name,
            'custom_result': 'value'
        }
    
    def parse_llm_response(self, response: str, context: dict) -> AgentResponse:
        # Custom LLM response parsing
        return AgentResponse(
            success=True,
            reasoning=response,
            recommendations={},
            confidence=0.8,
            safety_checks={},
            warnings=[],
            timestamp='',
            agent_name=self.agent_name
        )
```

### Custom Baseline Method

Implement custom baseline methods:

```python
from multi_agent_pruning.baselines import BasePruningMethod

class CustomPruningMethod(BasePruningMethod):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
    
    def prune_to_target_macs(self, model_name: str, target_macs: float, 
                           dataset: str) -> dict:
        # Custom pruning implementation
        model = self.load_model(model_name)
        
        # Apply custom pruning logic
        pruned_model = self.apply_custom_pruning(model, target_macs)
        
        # Fine-tune and evaluate
        final_accuracy = self.fine_tune_and_evaluate(pruned_model, dataset)
        
        return {
            'final_accuracy': final_accuracy,
            'final_macs': self.compute_macs(pruned_model),
            'method': 'custom_method'
        }
```

### Distributed Training

Configure distributed training for large models:

```python
# In your configuration file
hardware:
  distributed:
    enabled: true
    backend: "nccl"
    world_size: 4
    
  gpu:
    memory_limit: "40GB"
    mixed_precision: true
    gradient_checkpointing: true
```

### Custom Metrics

Add custom evaluation metrics:

```python
from multi_agent_pruning.utils.metrics import register_metric

@register_metric("custom_efficiency")
def compute_custom_efficiency(model, dataset_info):
    """Custom efficiency metric."""
    macs = compute_macs(model, (1, 3, 224, 224))
    params = compute_params(model)
    
    # Custom efficiency calculation
    efficiency = (params / 1e6) / (macs / 1e9)
    return efficiency

# Use in evaluation
evaluation_agent = EvaluationAgent(config={
    'custom_metrics': ['custom_efficiency']
})
```

## Performance Optimization

### Memory Optimization

```python
# Enable memory optimizations
config = {
    'agents': {
        'pruning_agent': {
            'memory_efficient': True,
            'gradient_checkpointing': True,
            'batch_processing': True
        }
    }
}
```

### Caching Configuration

```python
# Configure intelligent caching
config = {
    'precomputation': {
        'enabled': True,
        'cache_dir': './cache',
        'model_analysis': {
            'enabled': True,
            'cache_duration': 7200  # 2 hours
        },
        'importance_scores': {
            'enabled': True,
            'cache_duration': 3600  # 1 hour
        }
    }
}
```

### Parallel Processing

```python
# Enable parallel model downloading
from multi_agent_pruning.tools import ModelDownloader

downloader = ModelDownloader(num_workers=4)
results = downloader.download_models(
    ['deit_small', 'resnet50', 'convnext_tiny'],
    parallel=True
)
```

This API reference provides comprehensive documentation for all major components of the Enhanced Multi-Agent LLM Pruning Framework. For additional examples and tutorials, see the `examples/` directory in the repository.

