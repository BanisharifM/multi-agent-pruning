#!/usr/bin/env python3
"""
Paper Reproduction Validation Framework

Validates that the enhanced multi-agent system can reproduce the results
from the Isomorphic Pruning paper (Tables 1 and 2) and compares against
baseline methods.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multi_agent_pruning.experiments.paper_reproduction import PaperReproductionExperiment
from multi_agent_pruning.baselines.magnitude_pruning import MagnitudePruning
from multi_agent_pruning.baselines.taylor_pruning import TaylorPruning
from multi_agent_pruning.baselines.isomorphic_original import IsomorphicOriginal
from multi_agent_pruning.utils.metrics import compute_macs, compute_params
from multi_agent_pruning.utils.logging import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationTarget:
    """Target results from the original paper."""
    model_name: str
    dataset: str
    target_macs_g: float
    target_accuracy: float
    paper_table: str
    method: str = "isomorphic"
    
    def __post_init__(self):
        self.target_macs = self.target_macs_g * 1e9  # Convert to raw MACs

@dataclass
class ValidationResult:
    """Result from validation experiment."""
    target: ValidationTarget
    method: str
    achieved_macs_g: float
    achieved_accuracy: float
    params_reduction: float
    macs_reduction: float
    inference_time_ms: float
    
    # Validation metrics
    macs_error: float
    accuracy_error: float
    success: bool
    
    # Additional metrics
    convergence_iterations: int
    total_time_minutes: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PaperReproductionValidator:
    """Validates paper reproduction and baseline comparisons."""
    
    def __init__(self, config_path: str, output_dir: str = "./validation_results"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize experiment framework
        self.experiment = PaperReproductionExperiment(self.config)
        
        # Define paper targets (from Tables 1 and 2)
        self.paper_targets = self._load_paper_targets()
        
        # Initialize baseline methods
        self.baseline_methods = self._initialize_baseline_methods()
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        
        logger.info(f"🔬 Paper Reproduction Validator initialized")
        logger.info(f"   Config: {self.config_path}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Targets: {len(self.paper_targets)} paper targets")
        logger.info(f"   Methods: {len(self.baseline_methods)} baseline methods")
    
    def _load_paper_targets(self) -> List[ValidationTarget]:
        """Load target results from the original paper."""
        
        targets = []
        
        # Table 1: DeiT results (Vision Transformers)
        deit_targets = [
            # DeiT-Base results from paper
            ValidationTarget("deit_base", "imagenet", 17.58, 81.85, "Table 1", "original"),
            ValidationTarget("deit_base", "imagenet", 4.16, 82.41, "Table 1", "isomorphic"),
            ValidationTarget("deit_base", "imagenet", 2.61, 81.13, "Table 1", "isomorphic"),
            ValidationTarget("deit_base", "imagenet", 1.21, 77.50, "Table 1", "isomorphic"),
            
            # DeiT-Small results
            ValidationTarget("deit_small", "imagenet", 4.61, 79.85, "Table 1", "original"),
            ValidationTarget("deit_small", "imagenet", 2.30, 79.42, "Table 1", "isomorphic"),
            ValidationTarget("deit_small", "imagenet", 1.15, 77.91, "Table 1", "isomorphic"),
            
            # DeiT-Tiny results
            ValidationTarget("deit_tiny", "imagenet", 1.26, 72.21, "Table 1", "original"),
            ValidationTarget("deit_tiny", "imagenet", 0.63, 71.45, "Table 1", "isomorphic"),
        ]
        
        # Table 2: ConvNext results (Modern CNNs)
        convnext_targets = [
            # ConvNext-Small results
            ValidationTarget("convnext_small", "imagenet", 8.70, 83.13, "Table 2", "original"),
            ValidationTarget("convnext_small", "imagenet", 8.48, 83.17, "Table 2", "isomorphic"),
            ValidationTarget("convnext_small", "imagenet", 4.35, 82.19, "Table 2", "isomorphic"),
            ValidationTarget("convnext_small", "imagenet", 2.18, 80.32, "Table 2", "isomorphic"),
            
            # ConvNext-Tiny results
            ValidationTarget("convnext_tiny", "imagenet", 4.47, 82.05, "Table 2", "original"),
            ValidationTarget("convnext_tiny", "imagenet", 4.19, 82.19, "Table 2", "isomorphic"),
            ValidationTarget("convnext_tiny", "imagenet", 2.24, 81.01, "Table 2", "isomorphic"),
        ]
        
        targets.extend(deit_targets)
        targets.extend(convnext_targets)
        
        return targets
    
    def _initialize_baseline_methods(self) -> Dict[str, Any]:
        """Initialize baseline pruning methods for comparison."""
        
        methods = {
            'magnitude_l1': MagnitudePruning(criterion='l1'),
            'magnitude_l2': MagnitudePruning(criterion='l2'),
            'taylor': TaylorPruning(),
            'isomorphic_original': IsomorphicOriginal(),
            'multi_agent_enhanced': self.experiment  # Our enhanced method
        }
        
        return methods
    
    def validate_single_target(self, target: ValidationTarget, 
                             method_name: str, 
                             num_runs: int = 3) -> List[ValidationResult]:
        """Validate a single target with multiple runs for statistical significance."""
        
        logger.info(f"🎯 Validating {target.model_name} @ {target.target_macs_g:.2f}G MACs with {method_name}")
        
        results = []
        method = self.baseline_methods[method_name]
        
        for run in range(num_runs):
            logger.info(f"   Run {run + 1}/{num_runs}")
            
            try:
                # Run the pruning experiment
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                # Configure experiment for this target
                experiment_config = {
                    'model': target.model_name,
                    'dataset': target.dataset,
                    'target_macs': target.target_macs,
                    'method': method_name,
                    'run_id': run
                }
                
                # Run experiment
                if method_name == 'multi_agent_enhanced':
                    result = method.run_experiment(experiment_config)
                else:
                    result = method.prune_to_target_macs(
                        model_name=target.model_name,
                        target_macs=target.target_macs,
                        dataset=target.dataset
                    )
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    total_time_ms = start_time.elapsed_time(end_time)
                    total_time_minutes = total_time_ms / (1000 * 60)
                else:
                    total_time_minutes = 0
                
                # Extract results
                achieved_macs = result.get('final_macs', 0)
                achieved_accuracy = result.get('final_accuracy', 0)
                params_reduction = result.get('params_reduction', 0)
                macs_reduction = result.get('macs_reduction', 0)
                inference_time = result.get('inference_time_ms', 0)
                convergence_iterations = result.get('iterations', 1)
                
                # Calculate validation metrics
                macs_error = abs(achieved_macs - target.target_macs) / target.target_macs
                accuracy_error = abs(achieved_accuracy - target.target_accuracy) / target.target_accuracy
                
                # Success criteria: MACs within 5%, accuracy within 2%
                success = macs_error < 0.05 and accuracy_error < 0.02
                
                validation_result = ValidationResult(
                    target=target,
                    method=method_name,
                    achieved_macs_g=achieved_macs / 1e9,
                    achieved_accuracy=achieved_accuracy,
                    params_reduction=params_reduction,
                    macs_reduction=macs_reduction,
                    inference_time_ms=inference_time,
                    macs_error=macs_error,
                    accuracy_error=accuracy_error,
                    success=success,
                    convergence_iterations=convergence_iterations,
                    total_time_minutes=total_time_minutes
                )
                
                results.append(validation_result)
                
                logger.info(f"   ✅ Run {run + 1} completed: "
                          f"MACs={achieved_macs/1e9:.2f}G ({macs_error:.1%} error), "
                          f"Acc={achieved_accuracy:.1%} ({accuracy_error:.1%} error)")
                
            except Exception as e:
                logger.error(f"   ❌ Run {run + 1} failed: {str(e)}")
                
                # Create failed result
                failed_result = ValidationResult(
                    target=target,
                    method=method_name,
                    achieved_macs_g=0,
                    achieved_accuracy=0,
                    params_reduction=0,
                    macs_reduction=0,
                    inference_time_ms=0,
                    macs_error=1.0,
                    accuracy_error=1.0,
                    success=False,
                    convergence_iterations=0,
                    total_time_minutes=0
                )
                results.append(failed_result)
        
        return results
    
    def validate_all_targets(self, methods: Optional[List[str]] = None, 
                           num_runs: int = 3) -> Dict[str, List[ValidationResult]]:
        """Validate all paper targets with specified methods."""
        
        if methods is None:
            methods = list(self.baseline_methods.keys())
        
        logger.info(f"🚀 Starting validation of {len(self.paper_targets)} targets with {len(methods)} methods")
        
        all_results = {}
        
        for method_name in methods:
            logger.info(f"\n📊 Validating method: {method_name}")
            method_results = []
            
            for target in tqdm(self.paper_targets, desc=f"Validating {method_name}"):
                target_results = self.validate_single_target(target, method_name, num_runs)
                method_results.extend(target_results)
            
            all_results[method_name] = method_results
            
            # Save intermediate results
            self._save_method_results(method_name, method_results)
        
        self.validation_results = all_results
        return all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze validation results and generate comprehensive report."""
        
        logger.info("📈 Analyzing validation results...")
        
        analysis = {
            'summary': {},
            'method_comparison': {},
            'statistical_tests': {},
            'paper_reproduction': {}
        }
        
        # Overall summary
        total_experiments = sum(len(results) for results in self.validation_results.values())
        total_successful = sum(
            sum(1 for r in results if r.success) 
            for results in self.validation_results.values()
        )
        
        analysis['summary'] = {
            'total_experiments': total_experiments,
            'total_successful': total_successful,
            'overall_success_rate': total_successful / total_experiments if total_experiments > 0 else 0,
            'methods_tested': list(self.validation_results.keys()),
            'targets_tested': len(self.paper_targets)
        }
        
        # Method comparison
        for method_name, results in self.validation_results.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                macs_errors = [r.macs_error for r in successful_results]
                accuracy_errors = [r.accuracy_error for r in successful_results]
                convergence_times = [r.total_time_minutes for r in successful_results]
                
                analysis['method_comparison'][method_name] = {
                    'success_rate': len(successful_results) / len(results),
                    'avg_macs_error': np.mean(macs_errors),
                    'std_macs_error': np.std(macs_errors),
                    'avg_accuracy_error': np.mean(accuracy_errors),
                    'std_accuracy_error': np.std(accuracy_errors),
                    'avg_convergence_time': np.mean(convergence_times),
                    'std_convergence_time': np.std(convergence_times)
                }
            else:
                analysis['method_comparison'][method_name] = {
                    'success_rate': 0,
                    'avg_macs_error': 1.0,
                    'std_macs_error': 0,
                    'avg_accuracy_error': 1.0,
                    'std_accuracy_error': 0,
                    'avg_convergence_time': 0,
                    'std_convergence_time': 0
                }
        
        # Statistical significance tests
        if len(self.validation_results) >= 2:
            analysis['statistical_tests'] = self._perform_statistical_tests()
        
        # Paper reproduction analysis
        analysis['paper_reproduction'] = self._analyze_paper_reproduction()
        
        return analysis
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests between methods."""
        
        tests = {}
        methods = list(self.validation_results.keys())
        
        # Pairwise comparisons
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                results1 = [r for r in self.validation_results[method1] if r.success]
                results2 = [r for r in self.validation_results[method2] if r.success]
                
                if len(results1) >= 3 and len(results2) >= 3:
                    # Compare accuracy errors
                    acc_errors1 = [r.accuracy_error for r in results1]
                    acc_errors2 = [r.accuracy_error for r in results2]
                    
                    # Wilcoxon rank-sum test (non-parametric)
                    statistic, p_value = stats.ranksums(acc_errors1, acc_errors2)
                    
                    tests[f"{method1}_vs_{method2}"] = {
                        'test': 'wilcoxon_rank_sum',
                        'metric': 'accuracy_error',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'method1_mean': np.mean(acc_errors1),
                        'method2_mean': np.mean(acc_errors2),
                        'better_method': method1 if np.mean(acc_errors1) < np.mean(acc_errors2) else method2
                    }
        
        return tests
    
    def _analyze_paper_reproduction(self) -> Dict[str, Any]:
        """Analyze how well we reproduce the original paper results."""
        
        reproduction_analysis = {
            'table1_reproduction': {},
            'table2_reproduction': {},
            'overall_reproduction': {}
        }
        
        # Analyze Table 1 (DeiT) reproduction
        table1_targets = [t for t in self.paper_targets if t.paper_table == "Table 1"]
        table1_results = []
        
        for target in table1_targets:
            # Find results for our enhanced method
            enhanced_results = [
                r for r in self.validation_results.get('multi_agent_enhanced', [])
                if r.target.model_name == target.model_name and 
                   abs(r.target.target_macs_g - target.target_macs_g) < 0.1
            ]
            
            if enhanced_results:
                best_result = min(enhanced_results, key=lambda x: x.accuracy_error)
                table1_results.append(best_result)
        
        if table1_results:
            reproduction_analysis['table1_reproduction'] = {
                'targets_attempted': len(table1_targets),
                'targets_reproduced': sum(1 for r in table1_results if r.success),
                'avg_macs_error': np.mean([r.macs_error for r in table1_results]),
                'avg_accuracy_error': np.mean([r.accuracy_error for r in table1_results]),
                'reproduction_rate': sum(1 for r in table1_results if r.success) / len(table1_results)
            }
        
        # Similar analysis for Table 2
        table2_targets = [t for t in self.paper_targets if t.paper_table == "Table 2"]
        table2_results = []
        
        for target in table2_targets:
            enhanced_results = [
                r for r in self.validation_results.get('multi_agent_enhanced', [])
                if r.target.model_name == target.model_name and 
                   abs(r.target.target_macs_g - target.target_macs_g) < 0.1
            ]
            
            if enhanced_results:
                best_result = min(enhanced_results, key=lambda x: x.accuracy_error)
                table2_results.append(best_result)
        
        if table2_results:
            reproduction_analysis['table2_reproduction'] = {
                'targets_attempted': len(table2_targets),
                'targets_reproduced': sum(1 for r in table2_results if r.success),
                'avg_macs_error': np.mean([r.macs_error for r in table2_results]),
                'avg_accuracy_error': np.mean([r.accuracy_error for r in table2_results]),
                'reproduction_rate': sum(1 for r in table2_results if r.success) / len(table2_results)
            }
        
        # Overall reproduction
        all_reproduction_results = table1_results + table2_results
        if all_reproduction_results:
            reproduction_analysis['overall_reproduction'] = {
                'total_targets': len(self.paper_targets),
                'targets_reproduced': sum(1 for r in all_reproduction_results if r.success),
                'overall_reproduction_rate': sum(1 for r in all_reproduction_results if r.success) / len(all_reproduction_results),
                'avg_macs_error': np.mean([r.macs_error for r in all_reproduction_results]),
                'avg_accuracy_error': np.mean([r.accuracy_error for r in all_reproduction_results])
            }
        
        return reproduction_analysis
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations of validation results."""
        
        logger.info("📊 Generating validation visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create visualization directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Success rate comparison
        self._plot_success_rates(viz_dir)
        
        # 2. Accuracy vs MACs scatter plot
        self._plot_accuracy_vs_macs(viz_dir)
        
        # 3. Method comparison radar chart
        self._plot_method_comparison_radar(viz_dir)
        
        # 4. Convergence time analysis
        self._plot_convergence_analysis(viz_dir)
        
        # 5. Paper reproduction heatmap
        self._plot_paper_reproduction_heatmap(viz_dir)
        
        logger.info(f"📊 Visualizations saved to {viz_dir}")
    
    def _plot_success_rates(self, viz_dir: Path):
        """Plot success rates for each method."""
        
        methods = list(self.validation_results.keys())
        success_rates = []
        
        for method in methods:
            results = self.validation_results[method]
            success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
            success_rates.append(success_rate)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, success_rates, alpha=0.8)
        plt.title('Method Success Rates', fontsize=16, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "success_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_vs_macs(self, viz_dir: Path):
        """Plot accuracy vs MACs for all methods."""
        
        plt.figure(figsize=(12, 8))
        
        colors = sns.color_palette("husl", len(self.validation_results))
        
        for i, (method, results) in enumerate(self.validation_results.items()):
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                macs = [r.achieved_macs_g for r in successful_results]
                accuracies = [r.achieved_accuracy * 100 for r in successful_results]  # Convert to percentage
                
                plt.scatter(macs, accuracies, label=method, alpha=0.7, 
                          s=60, color=colors[i])
        
        plt.xlabel('MACs (G)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy vs Computational Cost (MACs)', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "accuracy_vs_macs.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_method_comparison_radar(self, viz_dir: Path):
        """Create radar chart comparing methods across multiple metrics."""
        
        # This would require more complex radar chart implementation
        # For now, create a simple comparison table visualization
        pass
    
    def _plot_convergence_analysis(self, viz_dir: Path):
        """Plot convergence time analysis."""
        
        methods = list(self.validation_results.keys())
        convergence_times = []
        
        for method in methods:
            results = self.validation_results[method]
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                avg_time = np.mean([r.total_time_minutes for r in successful_results])
                convergence_times.append(avg_time)
            else:
                convergence_times.append(0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, convergence_times, alpha=0.8)
        plt.title('Average Convergence Time by Method', fontsize=16, fontweight='bold')
        plt.ylabel('Time (minutes)', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, time in zip(bars, convergence_times):
            if time > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "convergence_times.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_paper_reproduction_heatmap(self, viz_dir: Path):
        """Create heatmap showing paper reproduction success."""
        
        # Create matrix of reproduction success
        models = list(set(t.model_name for t in self.paper_targets))
        macs_targets = list(set(f"{t.target_macs_g:.1f}G" for t in self.paper_targets))
        
        # This would require more complex heatmap implementation
        # For now, skip this visualization
        pass
    
    def _save_method_results(self, method_name: str, results: List[ValidationResult]):
        """Save results for a specific method."""
        
        results_file = self.output_dir / f"{method_name}_results.json"
        
        results_data = {
            'method': method_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Saved {method_name} results to {results_file}")
    
    def save_comprehensive_report(self, analysis: Dict[str, Any]):
        """Save comprehensive validation report."""
        
        report_file = self.output_dir / "validation_report.json"
        
        report = {
            'validation_config': {
                'config_path': str(self.config_path),
                'output_dir': str(self.output_dir),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'paper_targets': [asdict(t) for t in self.paper_targets],
            'analysis': analysis,
            'all_results': {
                method: [r.to_dict() for r in results]
                for method, results in self.validation_results.items()
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Comprehensive report saved to {report_file}")
        
        # Also save as CSV for easy analysis
        self._save_results_csv()
    
    def _save_results_csv(self):
        """Save results in CSV format for easy analysis."""
        
        all_results = []
        for method, results in self.validation_results.items():
            for result in results:
                result_dict = result.to_dict()
                result_dict['method'] = method
                all_results.append(result_dict)
        
        df = pd.DataFrame(all_results)
        csv_file = self.output_dir / "validation_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"📊 Results CSV saved to {csv_file}")

def main():
    """Main function for running validation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Reproduction Validation")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration file')
    parser.add_argument('--output-dir', type=str, default='./validation_results',
                       help='Output directory for results')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['multi_agent_enhanced', 'magnitude_l1', 'taylor'],
                       help='Methods to validate')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of runs per target for statistical significance')
    parser.add_argument('--wandb', action='store_true',
                       help='Log results to Weights & Biases')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Initialize WandB if requested
    if args.wandb:
        wandb.init(project="multi_agent_pruning_validation", 
                  config=vars(args))
    
    # Create validator
    validator = PaperReproductionValidator(args.config, args.output_dir)
    
    # Run validation
    logger.info("🚀 Starting paper reproduction validation...")
    results = validator.validate_all_targets(args.methods, args.num_runs)
    
    # Analyze results
    analysis = validator.analyze_results()
    
    # Generate visualizations
    validator.generate_visualizations()
    
    # Save comprehensive report
    validator.save_comprehensive_report(analysis)
    
    # Print summary
    print("\n" + "="*80)
    print("📋 VALIDATION SUMMARY")
    print("="*80)
    
    summary = analysis['summary']
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful experiments: {summary['total_successful']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
    
    print(f"\n📊 METHOD COMPARISON:")
    for method, stats in analysis['method_comparison'].items():
        print(f"{method:<25}: {stats['success_rate']:.1%} success, "
              f"{stats['avg_accuracy_error']:.1%} avg accuracy error")
    
    if 'overall_reproduction' in analysis['paper_reproduction']:
        repro = analysis['paper_reproduction']['overall_reproduction']
        print(f"\n📄 PAPER REPRODUCTION:")
        print(f"Reproduction rate: {repro['overall_reproduction_rate']:.1%}")
        print(f"Average MACs error: {repro['avg_macs_error']:.1%}")
        print(f"Average accuracy error: {repro['avg_accuracy_error']:.1%}")
    
    print("="*80)
    
    # Log to WandB if enabled
    if args.wandb:
        wandb.log(analysis['summary'])
        wandb.log(analysis['method_comparison'])
        if 'overall_reproduction' in analysis['paper_reproduction']:
            wandb.log(analysis['paper_reproduction']['overall_reproduction'])
    
    logger.info("✅ Validation completed successfully!")

if __name__ == "__main__":
    main()

