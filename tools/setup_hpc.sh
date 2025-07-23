#!/bin/bash

# Multi-Agent LLM Pruning Framework - HPC Setup Script
# This script sets up the environment on HPC systems with SLURM

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="multi_agent_pruning"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect HPC system
detect_hpc_system() {
    if command_exists squeue; then
        echo "slurm"
    elif command_exists qstat; then
        echo "pbs"
    else
        echo "unknown"
    fi
}

# Function to load modules on HPC
load_hpc_modules() {
    local hpc_system=$(detect_hpc_system)
    
    print_status "Detected HPC system: $hpc_system"
    
    # Common modules to load
    if command_exists module; then
        print_status "Loading HPC modules..."
        
        # Try to load common modules (adjust based on your HPC system)
        module load python/3.10 2>/dev/null || print_warning "Python module not found"
        module load cuda/11.8 2>/dev/null || print_warning "CUDA module not found"
        module load gcc/9.3.0 2>/dev/null || print_warning "GCC module not found"
        module load cmake/3.20 2>/dev/null || print_warning "CMake module not found"
        
        print_status "Loaded modules:"
        module list
    else
        print_warning "Module system not available"
    fi
}

# Function to setup conda environment
setup_conda_env() {
    print_status "Setting up Conda environment: $CONDA_ENV_NAME"
    
    # Check if conda is available
    if ! command_exists conda; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
    
    # Remove existing environment if it exists
    if conda env list | grep -q "$CONDA_ENV_NAME"; then
        print_warning "Environment $CONDA_ENV_NAME already exists. Removing..."
        conda env remove -n "$CONDA_ENV_NAME" -y
    fi
    
    # Create new environment
    print_status "Creating new conda environment..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    print_success "Conda environment created and activated"
}

# Function to install PyTorch with CUDA support
install_pytorch() {
    print_status "Installing PyTorch with CUDA support..."
    
    # Detect CUDA version
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_status "Detected CUDA version: $CUDA_VERSION"
        
        # Install PyTorch based on CUDA version
        if [[ "$CUDA_VERSION" == "11.8" ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$CUDA_VERSION" == "12.1" ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            print_warning "Unsupported CUDA version. Installing CPU version."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_warning "CUDA not detected. Installing CPU version."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installed"
}

# Function to install requirements
install_requirements() {
    print_status "Installing Python requirements..."
    
    cd "$PROJECT_DIR"
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install requirements
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
    
    # Install the package in development mode
    pip install -e .
    
    print_success "Requirements installed"
}

# Function to setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    ENV_FILE="$PROJECT_DIR/.env"
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# Multi-Agent LLM Pruning Environment Configuration

# OpenAI API Configuration (required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# OpenRouter API Configuration (alternative)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# WandB Configuration (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=multi_agent_pruning
WANDB_ENTITY=your_wandb_entity_here

# Dataset Paths (adjust for your HPC system)
IMAGENET_PATH=/path/to/imagenet
CIFAR10_PATH=/path/to/cifar10

# Cache and Results Directories
CACHE_DIR=./cache
RESULTS_DIR=./results
MODELS_DIR=./models

# Hardware Configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=16

# HPC Configuration
SLURM_JOB_NAME=multi_agent_pruning
SLURM_PARTITION=gpu
SLURM_TIME=24:00:00
SLURM_NODES=1
SLURM_NTASKS_PER_NODE=1
SLURM_CPUS_PER_TASK=16
SLURM_GRES=gpu:a100:1
EOF
        print_success "Created .env file template"
        print_warning "Please edit $ENV_FILE with your actual API keys and paths"
    else
        print_status ".env file already exists"
    fi
}

# Function to download models
download_models() {
    print_status "Downloading pretrained models..."
    
    cd "$PROJECT_DIR"
    
    # Create models directory
    mkdir -p models
    
    # Download models using the setup script
    python tools/download_models.py --models resnet50,deit_small,convnext_tiny --cache_dir ./models
    
    print_success "Models downloaded"
}

# Function to run tests
run_tests() {
    print_status "Running tests to verify installation..."
    
    cd "$PROJECT_DIR"
    
    # Run basic tests
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import timm; print(f'TIMM version: {timm.__version__}')"
    python -c "import multi_agent_pruning; print('Multi-Agent Pruning imported successfully')"
    
    # Run unit tests if available
    if [[ -d "tests" ]]; then
        python -m pytest tests/ -v --tb=short
    fi
    
    print_success "Tests completed"
}

# Function to create SLURM job script
create_slurm_script() {
    print_status "Creating SLURM job script..."
    
    SLURM_SCRIPT="$PROJECT_DIR/scripts/run_slurm.sh"
    mkdir -p "$(dirname "$SLURM_SCRIPT")"
    
    cat > "$SLURM_SCRIPT" << 'EOF'
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

# Activate conda environment
source ~/.bashrc
conda activate multi_agent_pruning

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Create logs directory
mkdir -p logs

# Run the experiment
cd $SLURM_SUBMIT_DIR

# Example: Run paper reproduction experiments
python experiments/paper_reproduction.py \
    --config configs/experiments/enhanced_multi_agent.yaml \
    --output_dir results/paper_reproduction_$(date +%Y%m%d_%H%M%S)

# Example: Run single model pruning
# python scripts/prune.py \
#     --model deit_small \
#     --dataset imagenet \
#     --target_ratio 0.5 \
#     --config configs/experiments/enhanced_multi_agent.yaml

echo "Job completed at $(date)"
EOF
    
    chmod +x "$SLURM_SCRIPT"
    print_success "SLURM script created at $SLURM_SCRIPT"
}

# Function to print usage instructions
print_usage() {
    cat << EOF

${GREEN}🎉 Multi-Agent LLM Pruning Framework Setup Complete!${NC}

${BLUE}Next Steps:${NC}
1. Edit the .env file with your API keys:
   ${YELLOW}nano $PROJECT_DIR/.env${NC}

2. Activate the conda environment:
   ${YELLOW}conda activate $CONDA_ENV_NAME${NC}

3. Run a quick test:
   ${YELLOW}python -c "import multi_agent_pruning; print('Success!')"${NC}

4. Submit a SLURM job (if on HPC):
   ${YELLOW}sbatch scripts/run_slurm.sh${NC}

5. Or run locally:
   ${YELLOW}python experiments/paper_reproduction.py${NC}

${BLUE}Useful Commands:${NC}
- Run paper reproduction: ${YELLOW}multi-agent-compare${NC}
- Prune a single model: ${YELLOW}multi-agent-prune --model deit_small --dataset imagenet${NC}
- Download models: ${YELLOW}multi-agent-setup --download-models${NC}
- View help: ${YELLOW}multi-agent-prune --help${NC}

${BLUE}Configuration Files:${NC}
- Main config: ${YELLOW}configs/experiments/enhanced_multi_agent.yaml${NC}
- Environment: ${YELLOW}.env${NC}
- SLURM script: ${YELLOW}scripts/run_slurm.sh${NC}

${GREEN}Happy Pruning! 🚀${NC}

EOF
}

# Main setup function
main() {
    print_status "Starting Multi-Agent LLM Pruning Framework setup..."
    print_status "Project directory: $PROJECT_DIR"
    
    # Load HPC modules
    load_hpc_modules
    
    # Setup conda environment
    setup_conda_env
    
    # Install PyTorch
    install_pytorch
    
    # Install requirements
    install_requirements
    
    # Setup environment
    setup_environment
    
    # Download models (optional)
    read -p "Download pretrained models now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_models
    fi
    
    # Run tests
    run_tests
    
    # Create SLURM script
    create_slurm_script
    
    # Print usage instructions
    print_usage
    
    print_success "Setup completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env-name NAME          Conda environment name (default: multi_agent_pruning)"
            echo "  --python-version VERSION Python version (default: 3.10)"
            echo "  --skip-models           Skip model download"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main setup
main

