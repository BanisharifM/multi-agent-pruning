#!/usr/bin/env python3
"""
Main pruning script for Multi-Agent LLM Pruning Framework
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
import torch

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_agent_pruning import (
    AgentCoordinator, ModelFactory, DatasetFactory, 
    setup_logging, __version__
)
from multi_agent_pruning.utils.helpers import load_config, save_config

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the pruning CLI."""
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent LLM Pruning Framework - Main Pruning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pruning with DeiT-Small
  multi-agent-prune --model deit_small --dataset imagenet --target-ratio 0.5
  
  # Advanced pruning with custom config
  multi-agent-prune --config experiments/custom_config.yaml
  
  # Target specific MACs
  multi-agent-prune --model resnet50 --target-macs 2.0 --dataset imagenet
        """
    )
    
    # Version
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture (resnet50, deit_small, vit_base, etc.)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights (default: True)')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='imagenet',
                      help='Dataset name (imagenet, cifar10, cifar100)')
    parser.add_argument('--data-path', type=str,
                      help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training/evaluation')
    parser.add_argument('--subset-ratio', type=float,
                      help='Use subset of dataset (0.0-1.0)')
    
    # Pruning configuration
    parser.add_argument('--target-ratio', type=float,
                       help='Target pruning ratio (0.0-1.0)')
    parser.add_argument('--target-macs', type=float,
                       help='Target MACs in GMACs')
    parser.add_argument('--target-params', type=float,
                       help='Target parameters in millions')
    parser.add_argument('--pruning-method', type=str, default='multi_agent',
                       choices=['multi_agent', 'magnitude', 'taylor', 'isomorphic'],
                       help='Pruning method to use')
    
    # Multi-agent configuration
    parser.add_argument('--llm-model', type=str, default='gpt-3.5-turbo',
                       help='LLM model for agent reasoning')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum pruning iterations')
    parser.add_argument('--convergence-threshold', type=float, default=0.001,
                       help='Convergence threshold for stopping')
    
    # Experiment configuration
    parser.add_argument('--config', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--experiment-name', type=str,
                       help='Name for the experiment')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Logging and debugging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for log files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    
    errors = []
    
    # Check target specifications
    targets = [args.target_ratio, args.target_macs, args.target_params]
    if not args.config and not any(targets):
        errors.append("At least one target (--target-ratio, --target-macs, --target-params) must be specified")
    
    # Check target ranges
    if args.target_ratio is not None and not (0.0 < args.target_ratio < 1.0):
        errors.append("--target-ratio must be between 0.0 and 1.0")
    
    if args.target_macs is not None and args.target_macs <= 0:
        errors.append("--target-macs must be positive")
    
    if args.target_params is not None and args.target_params <= 0:
        errors.append("--target-params must be positive")
    
    # Check file paths
    if args.config and not Path(args.config).exists():
        errors.append(f"Config file not found: {args.config}")
    
    if args.checkpoint and not Path(args.checkpoint).exists():
        errors.append(f"Checkpoint file not found: {args.checkpoint}")
    
    if args.data_path and not Path(args.data_path).exists():
        errors.append(f"Data path not found: {args.data_path}")
    
    if errors:
        print("❌ Argument validation errors:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)

def load_or_create_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from file or create from arguments."""
    
    if args.config:
        # Load from file
        config = load_config(args.config)
        print(f"📁 Loaded configuration from {args.config}")
    else:
        # Create from arguments
        config = {
            'model': {
                'name': args.model,
                'pretrained': args.pretrained,
                'checkpoint': args.checkpoint
            },
            'dataset': {
                'name': args.dataset,
                'data_path': args.data_path,
                'batch_size': args.batch_size,
                'subset_ratio': args.subset_ratio
            },
            'pruning': {
                'method': args.pruning_method,
                'target_ratio': args.target_ratio,
                'target_macs': args.target_macs,
                'target_params': args.target_params
            },
            'multi_agent': {
                'llm_model': args.llm_model,
                'max_iterations': args.max_iterations,
                'convergence_threshold': args.convergence_threshold
            },
            'experiment': {
                'name': args.experiment_name or f"{args.model}_{args.pruning_method}",
                'output_dir': args.output_dir,
                'seed': args.seed
            },
            'hardware': {
                'device': args.device,
                'num_workers': args.num_workers
            },
            'logging': {
                'level': args.log_level,
                'log_dir': args.log_dir,
                'debug': args.debug
            }
        }
    
    return config

def setup_experiment_environment(config: Dict[str, Any]) -> None:
    """Setup the experiment environment."""
    
    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['experiment']['seed'])
    
    # Create output directory
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_config = setup_logging(
        log_level=config['logging']['level'],
        log_dir=config['logging']['log_dir'],
        enable_console=True,
        enable_file=True,
        enable_structured=True,
        experiment_id=config['experiment']['name']
    )
    
    # Save configuration
    config_file = output_dir / 'config.yaml'
    save_config(config, config_file)
    
    print(f"📁 Experiment directory: {output_dir}")
    print(f"📋 Configuration saved to: {config_file}")
    print(f"📝 Logs directory: {config['logging']['log_dir']}")

def run_pruning_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the main pruning experiment."""
    
    print(f"🚀 Starting pruning experiment: {config['experiment']['name']}")
    
    # Initialize components
    model_factory = ModelFactory()
    dataset_factory = DatasetFactory(config['dataset'].get('data_path', './data'))
    
    # Load model
    print(f"📦 Loading model: {config['model']['name']}")
    model = model_factory.create_model(
        config['model']['name'],
        pretrained=config['model']['pretrained'],
        checkpoint=config['model'].get('checkpoint')
    )
    
    # Load dataset
    print(f"📁 Loading dataset: {config['dataset']['name']}")
    train_loader, val_loader, test_loader = dataset_factory.create_dataloaders(
        config['dataset']['name'],
        batch_size=config['dataset']['batch_size'],
        subset_ratio=config['dataset'].get('subset_ratio')
    )
    
    # Initialize agent coordinator
    print("🤖 Initializing multi-agent system")
    coordinator = AgentCoordinator(config)
    
    # Run pruning
    print("✂️ Starting pruning process")
    results = coordinator.run_pruning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    print("✅ Pruning experiment completed successfully")
    return results

def save_results(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save experiment results."""
    
    output_dir = Path(config['experiment']['output_dir'])
    
    # Save results as JSON
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {config['experiment']['name']}\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Dataset: {config['dataset']['name']}\n")
        f.write(f"Method: {config['pruning']['method']}\n")
        f.write("\nResults:\n")
        
        if 'final_accuracy' in results:
            f.write(f"Final Accuracy: {results['final_accuracy']:.4f}\n")
        if 'compression_ratio' in results:
            f.write(f"Compression Ratio: {results['compression_ratio']:.4f}\n")
        if 'speedup' in results:
            f.write(f"Speedup: {results['speedup']:.2f}x\n")
    
    print(f"💾 Results saved to: {results_file}")
    print(f"📄 Summary saved to: {summary_file}")

def main():
    """Main entry point for the pruning script."""
    
    try:
        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Validate arguments
        validate_args(args)
        
        # Load configuration
        config = load_or_create_config(args)
        
        # Setup environment
        setup_experiment_environment(config)
        
        # Run pruning experiment
        results = run_pruning_experiment(config)
        
        # Save results
        save_results(results, config)
        
        print("🎉 Pruning experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
