# Setup Instructions for Enhanced Multi-Agent LLM Pruning Framework

This document provides step-by-step instructions to set up and run the enhanced multi-agent pruning framework on your HPC system.

## 🎯 Quick Start (5 Minutes)

For immediate setup on HPC systems with SLURM:

```bash
# 1. Clone the repository
git clone https://github.com/BanisharifM/multi-agent-pruning.git
cd multi-agent-pruning

# 2. Run automated setup
chmod +x tools/setup_hpc.sh
./tools/setup_hpc.sh

# 3. Configure environment (edit with your details)
cp .env.template .env
nano .env

# 4. Test installation
conda activate multi_agent_pruning
python -c "import multi_agent_pruning; print('✅ Installation successful!')"

# 5. Run first experiment
multi-agent-prune --model deit_small --dataset imagenet --target-ratio 0.5
```

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+ or CentOS 7+)
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **Memory**: 64GB+ RAM recommended
- **Storage**: 100GB+ free space for models and datasets

### HPC Environment

- **SLURM**: Job scheduler (recommended)
- **Modules**: Environment module system
- **Shared Storage**: Access to shared filesystem for datasets
- **Internet Access**: For model downloads and API calls

### API Keys Required

- **OpenAI API Key**: For LLM agents (required)
- **WandB API Key**: For experiment tracking (optional)
- **OpenRouter API Key**: Alternative LLM provider (optional)

## 🔧 Detailed Installation

### Step 1: Environment Setup

#### Option A: Automated HPC Setup (Recommended)

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/your-username/multi-agent-pruning/main/tools/setup_hpc.sh
chmod +x setup_hpc.sh
./setup_hpc.sh
```

The script will:
- Detect your HPC environment
- Load appropriate modules (Python, CUDA, GCC)
- Create conda environment
- Install all dependencies
- Download essential models
- Create SLURM job scripts
- Verify installation

#### Option B: Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/BanisharifM/multi-agent-pruning.git
cd multi-agent-pruning

# 2. Load HPC modules (adjust for your system)
module load python/3.10
module load cuda/11.8
module load gcc/9.3.0

# 3. Create conda environment
conda create -n multi_agent_pruning python=3.10 -y
conda activate multi_agent_pruning

# 4. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install framework dependencies
pip install -r requirements.txt
pip install -e .

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import multi_agent_pruning; print('Framework installed successfully!')"
```

### Step 2: Configuration

#### Environment Variables

Create `.env` file with your configuration:

```bash
cp .env.template .env
nano .env
```

Required configuration:

```bash
# OpenAI API (required for LLM agents)
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1

# Dataset paths (adjust for your HPC system)
IMAGENET_PATH=/shared/datasets/imagenet
CIFAR10_PATH=/shared/datasets/cifar10

# Optional: Alternative LLM provider
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here

# Optional: Experiment tracking
WANDB_API_KEY=your-wandb-api-key-here
WANDB_PROJECT=multi_agent_pruning
WANDB_ENTITY=your-username

# Cache and results (use fast storage)
CACHE_DIR=/scratch/$USER/multi_agent_cache
RESULTS_DIR=/scratch/$USER/results
MODELS_DIR=/shared/models/multi_agent_pruning

# Hardware configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=16
```

#### HPC-Specific Settings

For SLURM systems, configure job defaults:

```bash
# Add to .env file
SLURM_PARTITION=gpu
SLURM_TIME=24:00:00
SLURM_NODES=1
SLURM_NTASKS_PER_NODE=1
SLURM_CPUS_PER_TASK=16
SLURM_GRES=gpu:a100:1
SLURM_MEM=64G
```

### Step 3: Model Setup

#### Download Pretrained Models

```bash
# Activate environment
conda activate multi_agent_pruning

# Download essential models for quick start
python tools/download_models.py --models deit_small,resnet50,convnext_tiny

# Download all models for comprehensive experiments
python tools/download_models.py --all

# Download specific paper table models
python tools/download_models.py --table1  # DeiT models (Table 1)
python tools/download_models.py --table2  # ConvNext models (Table 2)

# Verify downloads
python tools/download_models.py --verify --list
```

#### Model Storage Optimization

For HPC systems with shared storage:

```bash
# Create shared model cache
mkdir -p /shared/models/multi_agent_pruning
export MODELS_DIR=/shared/models/multi_agent_pruning

# Download to shared location
python tools/download_models.py --all --cache-dir $MODELS_DIR

# Verify all users can access
ls -la $MODELS_DIR
```

### Step 4: Dataset Preparation

#### ImageNet Setup

```bash
# If you have ImageNet downloaded
export IMAGENET_PATH=/path/to/your/imagenet

# Verify structure
ls $IMAGENET_PATH
# Should show: train/ val/

# Test data loading
python -c "
from torchvision.datasets import ImageFolder
dataset = ImageFolder('$IMAGENET_PATH/val')
print(f'✅ ImageNet validation set: {len(dataset)} images')
"
```

#### CIFAR-10 Setup (Auto-download)

```bash
# CIFAR-10 will be automatically downloaded
export CIFAR10_PATH=/scratch/$USER/cifar10

# Test download
python -c "
from torchvision.datasets import CIFAR10
dataset = CIFAR10('$CIFAR10_PATH', download=True)
print(f'✅ CIFAR-10 dataset: {len(dataset)} images')
"
```

## 🚀 Running Experiments

### Basic Usage

#### Single Model Pruning

```bash
# Prune DeiT-Small to 50% parameter reduction
multi-agent-prune \
    --model deit_small \
    --dataset imagenet \
    --target-ratio 0.5 \
    --output-dir ./results/deit_small_50pct

# With custom configuration
multi-agent-prune \
    --model resnet50 \
    --dataset cifar10 \
    --target-ratio 0.3 \
    --config configs/experiments/custom_config.yaml \
    --verbose
```

#### Paper Reproduction

```bash
# Reproduce Table 1 (DeiT results)
multi-agent-compare \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --table1 \
    --output-dir ./results/table1_reproduction

# Reproduce Table 2 (ConvNext results)
multi-agent-compare \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --table2 \
    --output-dir ./results/table2_reproduction

# Compare all methods
multi-agent-compare \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --methods multi_agent,magnitude_l1,taylor,isomorphic_original \
    --models deit_small,resnet50 \
    --target-ratios 0.3,0.5,0.7
```

### SLURM Job Submission

#### Single Job

```bash
# Submit basic pruning job
sbatch scripts/run_slurm.sh

# Submit with custom parameters
sbatch --job-name=deit_pruning \
       --time=12:00:00 \
       --gres=gpu:1 \
       scripts/run_slurm.sh deit_small imagenet 0.5
```

#### Array Jobs (Parameter Sweep)

```bash
# Submit array job for multiple configurations
sbatch scripts/run_array_job.sh

# Custom array job
sbatch --array=1-15 \
       --job-name=pruning_sweep \
       --time=8:00:00 \
       scripts/run_array_job.sh
```

#### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Monitor job output
tail -f logs/slurm_*.out

# Check job efficiency
seff <job_id>
```

### Python API Usage

#### Basic Pruning

```python
from multi_agent_pruning import MultiAgentPruner

# Initialize framework
pruner = MultiAgentPruner(
    config_path="configs/experiments/enhanced_multi_agent.yaml"
)

# Prune model
result = pruner.prune_model(
    model_name="deit_small",
    dataset="imagenet",
    target_ratio=0.5,
    output_dir="./results/deit_small_50pct"
)

print(f"Final accuracy: {result['final_accuracy']:.1%}")
print(f"Parameter reduction: {result['params_reduction']:.1%}")
```

#### Batch Processing

```python
# Define experiments
experiments = [
    {"model": "deit_small", "dataset": "imagenet", "ratio": 0.3},
    {"model": "deit_small", "dataset": "imagenet", "ratio": 0.5},
    {"model": "resnet50", "dataset": "imagenet", "ratio": 0.4},
]

# Run all experiments
results = {}
for exp in experiments:
    result = pruner.prune_model(
        model_name=exp["model"],
        dataset=exp["dataset"], 
        target_ratio=exp["ratio"]
    )
    results[f"{exp['model']}_{exp['ratio']}"] = result

# Print summary
for name, result in results.items():
    print(f"{name}: {result['final_accuracy']:.1%} accuracy")
```

## 🔍 Validation and Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_enhanced_system.py -v  # Enhanced system tests
pytest tests/test_baselines.py -v        # Baseline method tests
pytest tests/test_integration.py -v      # Integration tests

# Run with coverage
pytest tests/ --cov=multi_agent_pruning --cov-report=html
```

### Validate Paper Reproduction

```bash
# Run comprehensive validation
python validation/paper_reproduction_validator.py \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --methods multi_agent_enhanced,magnitude_l1,taylor \
    --num-runs 3 \
    --output-dir ./validation_results

# Quick validation (single run)
python validation/paper_reproduction_validator.py \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --methods multi_agent_enhanced \
    --num-runs 1
```

### Performance Benchmarks

```bash
# Run performance benchmarks
pytest tests/test_performance.py -v --benchmark

# Profile specific operations
python tools/profile_performance.py \
    --model deit_small \
    --operations profiling,pruning,finetuning
```

## 📊 Monitoring and Debugging

### Logging Configuration

Enable detailed logging for debugging:

```yaml
# In your config file
logging:
  level: "DEBUG"
  file_logging:
    enabled: true
    log_dir: "./debug_logs"
  
  agent_logging:
    reasoning_traces: true
    conversation_history: true
    timing_profiler: true
```

### Performance Monitoring

```python
# Enable timing profiler
from multi_agent_pruning.utils.profiler import TimingProfiler

profiler = TimingProfiler()

with profiler.timer("full_experiment"):
    result = pruner.prune_model(...)

profiler.get_summary()  # Print timing analysis
```

### Memory Monitoring

```bash
# Monitor GPU memory during experiments
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check disk usage
df -h
du -sh ./cache ./results
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Reduce batch sizes in config
datasets:
  imagenet:
    batch_size: 128  # Reduce from 256

# Enable memory optimizations
hardware:
  gpu:
    mixed_precision: true
    gradient_checkpointing: true
```

#### Model Download Failures

```bash
# Manual download with verification
python tools/download_models.py --models deit_small --force --verify

# Use alternative download location
python tools/download_models.py --models deit_small --cache-dir /tmp/models
```

#### LLM API Rate Limits

```bash
# Use alternative API provider
export OPENROUTER_API_KEY=your_key
export OPENAI_API_BASE=https://openrouter.ai/api/v1

# Reduce API usage
agents:
  global:
    max_retries: 3
    retry_delay: 2.0
```

#### Permission Issues

```bash
# Fix file permissions
chmod +x tools/*.sh scripts/*.sh

# Fix directory permissions
chmod -R 755 multi_agent_pruning/
```

### Getting Help

1. **Check logs**: Look in `./logs/` for detailed error messages
2. **Run tests**: `pytest tests/ -v` to identify issues
3. **Verify setup**: `python tools/verify_installation.py`
4. **Check documentation**: See `docs/` for detailed guides
5. **GitHub Issues**: Report bugs and ask questions

## 📈 Performance Optimization

### HPC Optimization

```yaml
# configs/hpc_optimized.yaml
hardware:
  gpu:
    memory_limit: "80GB"  # A100 80GB
    mixed_precision: true
    gradient_checkpointing: true
  
  cpu:
    num_workers: 32  # High-core HPC nodes
    
precomputation:
  enabled: true
  cache_dir: "/scratch/$USER/cache"  # Fast scratch storage
  
  model_analysis:
    cache_duration: 86400  # 24 hours
```

### Memory Optimization

```python
# Enable all memory optimizations
config = {
    'hardware': {
        'gpu': {
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'memory_efficient_attention': True
        }
    },
    'agents': {
        'pruning_agent': {
            'memory_efficient': True,
            'batch_processing': True
        }
    }
}
```

### Caching Optimization

```python
# Aggressive caching for repeated experiments
precomputation:
  enabled: true
  cache_dir: "/fast/scratch/cache"
  
  model_analysis:
    enabled: true
    cache_duration: 86400  # 24 hours
    
  importance_scores:
    enabled: true
    cache_duration: 43200  # 12 hours
```

## 🎯 Next Steps

After successful setup:

1. **Run Quick Test**: `multi-agent-prune --model deit_small --dataset cifar10 --target-ratio 0.5`
2. **Paper Reproduction**: `multi-agent-compare --table1`
3. **Custom Experiments**: Modify configs for your specific needs
4. **Baseline Comparison**: Compare with your original method
5. **Production Deployment**: Use HPC job arrays for large-scale experiments


**🎉 You're ready to run enhanced multi-agent pruning experiments!**

The framework is now set up and ready to deliver significantly improved results compared to your original implementation.

