#!/usr/bin/env python3
"""
Command Line Interface for Multi-Agent LLM Pruning Framework

This module provides the main CLI entry point for running pruning experiments.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_agent_pruning import (
    AgentCoordinator, ModelFactory, DatasetFactory, 
    setup_logging, __version__
)
from multi_agent_pruning.utils.helpers import load_config, save_config

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent LLM Pruning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pruning with DeiT-Small
  multi-agent-prune --model deit_small --dataset imagenet --target-ratio 0.5
  
  # Advanced pruning with custom config
  multi-agent-prune --config experiments/custom_config.yaml
  
  # Paper reproduction
  multi-agent-prune --reproduce-paper isomorphic --table 1
  
  # Compare with baselines
  multi-agent-prune --model resnet50 --compare-baselines --target-macs 2.0
        """
    )
    
    # Version
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, 
                           help='Model architecture (resnet50, deit_small, vit_base, etc.)')
    model_group.add_argument('--pretrained', action='store_true', default=True,
                           help='Use pretrained weights (default: True)')
    model_group.add_argument('--checkpoint', type=str,
                           help='Path to model checkpoint')
    
    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str, default='imagenet',
                          help='Dataset name (imagenet, cifar10, cifar100)')
    data_group.add_argument('--data-path', type=str,
                          help='Path to dataset directory')
    data_group.add_argument('--batch-size', type=int, default=32,
                          help='Batch size for training/evaluation')
    data_group.add_argument('--subset-ratio', type=float,
                          help='Use subset of dataset (0.0-1.0)')
    
    # Pruning configuration
    prune_group = parser.add_argument_group('Pruning Configuration')
    prune_group.add_argument('--target-ratio', type=float,
                           help='Target pruning ratio (0.0-1.0)')
    prune_group.add_argument('--target-macs', type=float,
                           help='Target MACs in GMACs')
    prune_group.add_argument('--target-params', type=float,
                           help='Target parameters in millions')
    prune_group.add_argument('--pruning-method', type=str, default='multi_agent',
                           choices=['multi_agent', 'magnitude', 'taylor', 'isomorphic'],
                           help='Pruning method to use')
    
    # Multi-agent configuration
    agent_group = parser.add_argument_group('Multi-Agent Configuration')
    agent_group.add_argument('--llm-model', type=str, default='gpt-3.5-turbo',
                           help='LLM model for agent reasoning')
    agent_group.add_argument('--max-iterations', type=int, default=10,
                           help='Maximum pruning iterations')
    agent_group.add_argument('--convergence-threshold', type=float, default=0.001,
                           help='Convergence threshold for stopping')
    
    # Experiment configuration
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument('--config', type=str,
                         help='Path to configuration YAML file')
    exp_group.add_argument('--output-dir', type=str, default='./results',
                         help='Output directory for results')
    exp_group.add_argument('--experiment-name', type=str,
                         help='Name for the experiment')
    exp_group.add_argument('--seed', type=int, default=42,
                         help='Random seed for reproducibility')
    
    # Comparison and reproduction
    comp_group = parser.add_argument_group('Comparison and Reproduction')
    comp_group.add_argument('--compare-baselines', action='store_true',
                          help='Compare with baseline methods')
    comp_group.add_argument('--reproduce-paper', type=str,
                          choices=['isomorphic', 'all'],
                          help='Reproduce paper results')
    comp_group.add_argument('--table', type=int,
                          help='Specific table to reproduce (1 or 2)')
    
    # Logging and debugging
    log_group = parser.add_argument_group('Logging and Debugging')
    log_group.add_argument('--log-level', type=str, default='INFO',
                         choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                         help='Logging level')
    log_group.add_argument('--log-dir', type=str, default='./logs',
                         help='Directory for log files')
    log_group.add_argument('--debug', action='store_true',
                         help='Enable debug mode')
    log_group.add_argument('--profile', action='store_true',
                         help='Enable performance profiling')
    
    # Hardware configuration
    hw_group = parser.add_argument_group('Hardware Configuration')
    hw_group.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu, cuda, auto)')
    hw_group.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    hw_group.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer')
    
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    
    errors = []
    
    # Check required arguments
    if not args.config and not args.model:
        errors.append("Either --config or --model must be specified")
    
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
    
    # Check reproduction arguments
    if args.reproduce_paper and not args.table:
        errors.append("--table must be specified when using --reproduce-paper")
    
    if args.table and not args.reproduce_paper:
        errors.append("--reproduce-paper must be specified when using --table")
    
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
                'seed': args.seed,
                'compare_baselines': args.compare_baselines
            },
            'hardware': {
                'device': args.device,
                'num_workers': args.num_workers,
                'pin_memory': args.pin_memory
            },
            'logging': {
                'level': args.log_level,
                'log_dir': args.log_dir,
                'debug': args.debug,
                'profile': args.profile
            }
        }
        
        # Handle reproduction
        if args.reproduce_paper:
            config['reproduction'] = {
                'paper': args.reproduce_paper,
                'table': args.table
            }
    
    return config

def setup_experiment_environment(config: Dict[str, Any]) -> None:
    """Setup the experiment environment."""
    
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

def run_paper_reproduction(config: Dict[str, Any]) -> None:
    """Run paper reproduction experiments."""
    
    paper = config['reproduction']['paper']
    table = config['reproduction']['table']
    
    print(f"📊 Reproducing {paper} paper, Table {table}")
    
    # Import reproduction module
    from multi_agent_pruning.experiments.paper_reproduction import PaperReproduction
    
    reproducer = PaperReproduction(config)
    
    if paper == 'isomorphic':
        if table == 1:
            results = reproducer.reproduce_table1()
        elif table == 2:
            results = reproducer.reproduce_table2()
        else:
            raise ValueError(f"Invalid table number: {table}")
    else:
        raise ValueError(f"Unknown paper: {paper}")
    
    print("✅ Paper reproduction completed")
    return results

def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single pruning experiment."""
    
    print(f"🚀 Starting experiment: {config['experiment']['name']}")
    
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
    
    print("✅ Experiment completed successfully")
    return results

def run_baseline_comparison(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run comparison with baseline methods."""
    
    print("📊 Running baseline comparison")
    
    # Import comparison module
    from multi_agent_pruning.experiments.baseline_comparison import BaselineComparison
    
    comparator = BaselineComparison(config)
    results = comparator.run_comparison()
    
    print("✅ Baseline comparison completed")
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
    """Main CLI entry point."""
    
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
        
        # Run experiment based on mode
        if 'reproduction' in config:
            results = run_paper_reproduction(config)
        elif config['experiment'].get('compare_baselines', False):
            results = run_baseline_comparison(config)
        else:
            results = run_single_experiment(config)
        
        # Save results
        save_results(results, config)
        
        print("🎉 All experiments completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        if args.debug if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()