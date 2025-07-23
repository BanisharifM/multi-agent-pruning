#!/usr/bin/env python3
"""
Model Download Utility for Multi-Agent LLM Pruning Framework

This script downloads and caches all the pretrained models needed for
baseline comparisons and experiments, optimized for HPC environments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import timm
import torch
import json
from tqdm import tqdm
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multi_agent_pruning.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Model configurations for paper reproduction
MODEL_CONFIGS = {
    # Vision Transformers (DeiT family)
    'deit_tiny': {
        'timm_name': 'deit_tiny_patch16_224.fb_in1k',
        'architecture': 'vision_transformer',
        'paper_table': 'Table 1',
        'description': 'DeiT Tiny for ViT pruning experiments',
        'expected_params': 5.7e6,
        'expected_macs': 1.3e9
    },
    'deit_small': {
        'timm_name': 'deit_small_patch16_224.fb_in1k',
        'architecture': 'vision_transformer', 
        'paper_table': 'Table 1',
        'description': 'DeiT Small for ViT pruning experiments',
        'expected_params': 22.1e6,
        'expected_macs': 4.6e9
    },
    'deit_base': {
        'timm_name': 'deit_base_patch16_224.fb_in1k',
        'architecture': 'vision_transformer',
        'paper_table': 'Table 1', 
        'description': 'DeiT Base for ViT pruning experiments',
        'expected_params': 86.6e6,
        'expected_macs': 17.6e9
    },
    'deit_base_distilled': {
        'timm_name': 'deit_base_distilled_patch16_224.fb_in1k',
        'architecture': 'vision_transformer',
        'paper_table': 'Table 1',
        'description': 'DeiT Base Distilled (exact paper model)',
        'expected_params': 86.6e6,
        'expected_macs': 17.6e9
    },
    
    # ConvNext family
    'convnext_tiny': {
        'timm_name': 'convnext_tiny.fb_in1k',
        'architecture': 'modern_cnn',
        'paper_table': 'Table 2',
        'description': 'ConvNext Tiny for modern CNN pruning',
        'expected_params': 28.6e6,
        'expected_macs': 4.5e9
    },
    'convnext_small': {
        'timm_name': 'convnext_small.fb_in1k', 
        'architecture': 'modern_cnn',
        'paper_table': 'Table 2',
        'description': 'ConvNext Small for modern CNN pruning',
        'expected_params': 50.2e6,
        'expected_macs': 8.7e9
    },
    
    # ResNet family (baseline CNNs)
    'resnet50': {
        'timm_name': 'resnet50.tv_in1k',
        'architecture': 'cnn',
        'paper_table': 'Baseline',
        'description': 'ResNet-50 baseline for CNN pruning',
        'expected_params': 25.6e6,
        'expected_macs': 4.1e9
    },
    'resnet101': {
        'timm_name': 'resnet101.tv_in1k',
        'architecture': 'cnn', 
        'paper_table': 'Baseline',
        'description': 'ResNet-101 for deeper CNN pruning',
        'expected_params': 44.5e6,
        'expected_macs': 7.8e9
    },
    'resnet152': {
        'timm_name': 'resnet152.tv_in1k',
        'architecture': 'cnn',
        'paper_table': 'Baseline', 
        'description': 'ResNet-152 for very deep CNN pruning',
        'expected_params': 60.2e6,
        'expected_macs': 11.6e9
    },
    
    # Mobile models
    'mobilenetv2': {
        'timm_name': 'mobilenetv2_100.ra_in1k',
        'architecture': 'mobile_cnn',
        'paper_table': 'Baseline',
        'description': 'MobileNetV2 for efficient model pruning',
        'expected_params': 3.5e6,
        'expected_macs': 0.3e9
    },
    
    # EfficientNet family
    'efficientnet_b0': {
        'timm_name': 'efficientnet_b0.ra_in1k',
        'architecture': 'efficient_cnn',
        'paper_table': 'Baseline',
        'description': 'EfficientNet-B0 for efficient pruning',
        'expected_params': 5.3e6,
        'expected_macs': 0.4e9
    }
}

class ModelDownloader:
    """Downloads and manages pretrained models for the framework."""
    
    def __init__(self, cache_dir: str = "./models", num_workers: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        
        # Create subdirectories
        self.model_cache_dir = self.cache_dir / "pretrained"
        self.metadata_dir = self.cache_dir / "metadata"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def download_model(self, model_key: str, force_redownload: bool = False) -> Dict:
        """Download a single model and return metadata."""
        
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_key]
        timm_name = config['timm_name']
        
        logger.info(f"📥 Downloading {model_key} ({timm_name})...")
        
        # Check if already cached
        metadata_file = self.metadata_dir / f"{model_key}.json"
        if metadata_file.exists() and not force_redownload:
            logger.info(f"✅ {model_key} already cached")
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        try:
            # Download model
            model = timm.create_model(timm_name, pretrained=True)
            
            # Compute actual statistics
            actual_params = sum(p.numel() for p in model.parameters())
            
            # Estimate MACs (requires a forward pass)
            try:
                from multi_agent_pruning.utils.metrics import compute_macs
                dummy_input = torch.randn(1, 3, 224, 224)
                actual_macs = compute_macs(model, dummy_input.shape)
            except Exception as e:
                logger.warning(f"Could not compute MACs for {model_key}: {e}")
                actual_macs = config.get('expected_macs', 0)
            
            # Create metadata
            metadata = {
                'model_key': model_key,
                'timm_name': timm_name,
                'architecture': config['architecture'],
                'paper_table': config['paper_table'],
                'description': config['description'],
                'actual_params': actual_params,
                'actual_macs': actual_macs,
                'expected_params': config['expected_params'],
                'expected_macs': config['expected_macs'],
                'param_match': abs(actual_params - config['expected_params']) / config['expected_params'] < 0.1,
                'download_success': True,
                'model_path': str(self.model_cache_dir / f"{model_key}.pth"),
                'timm_available': True
            }
            
            # Save model state dict (optional, for faster loading)
            if force_redownload or not Path(metadata['model_path']).exists():
                torch.save(model.state_dict(), metadata['model_path'])
                logger.info(f"💾 Saved model state dict to {metadata['model_path']}")
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ {model_key} downloaded successfully")
            logger.info(f"   Parameters: {actual_params:,} (expected: {config['expected_params']:,.0f})")
            logger.info(f"   MACs: {actual_macs/1e9:.2f}G (expected: {config['expected_macs']/1e9:.2f}G)")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_key}: {str(e)}")
            
            # Create error metadata
            error_metadata = {
                'model_key': model_key,
                'timm_name': timm_name,
                'download_success': False,
                'error': str(e),
                'timm_available': False
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(error_metadata, f, indent=2)
            
            return error_metadata
    
    def download_models(self, model_keys: List[str], 
                       force_redownload: bool = False,
                       parallel: bool = True) -> Dict[str, Dict]:
        """Download multiple models."""
        
        logger.info(f"📦 Downloading {len(model_keys)} models...")
        
        results = {}
        
        if parallel and len(model_keys) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_model = {
                    executor.submit(self.download_model, model_key, force_redownload): model_key
                    for model_key in model_keys
                }
                
                for future in tqdm(as_completed(future_to_model), 
                                 total=len(model_keys), 
                                 desc="Downloading models"):
                    model_key = future_to_model[future]
                    try:
                        results[model_key] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to download {model_key}: {e}")
                        results[model_key] = {'download_success': False, 'error': str(e)}
        else:
            # Sequential download
            for model_key in tqdm(model_keys, desc="Downloading models"):
                results[model_key] = self.download_model(model_key, force_redownload)
        
        return results
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their configurations."""
        return MODEL_CONFIGS
    
    def get_cached_models(self) -> List[str]:
        """Get list of cached models."""
        cached = []
        for model_key in MODEL_CONFIGS.keys():
            metadata_file = self.metadata_dir / f"{model_key}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('download_success', False):
                        cached.append(model_key)
        return cached
    
    def verify_models(self, model_keys: Optional[List[str]] = None) -> Dict[str, bool]:
        """Verify that models are correctly downloaded and loadable."""
        
        if model_keys is None:
            model_keys = self.get_cached_models()
        
        logger.info(f"🔍 Verifying {len(model_keys)} models...")
        
        verification_results = {}
        
        for model_key in tqdm(model_keys, desc="Verifying models"):
            try:
                # Load metadata
                metadata_file = self.metadata_dir / f"{model_key}.json"
                if not metadata_file.exists():
                    verification_results[model_key] = False
                    continue
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if not metadata.get('download_success', False):
                    verification_results[model_key] = False
                    continue
                
                # Try to load model
                config = MODEL_CONFIGS[model_key]
                model = timm.create_model(config['timm_name'], pretrained=True)
                
                # Quick forward pass test
                dummy_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    output = model(dummy_input)
                
                # Check output shape
                expected_classes = 1000  # ImageNet classes
                if output.shape[1] == expected_classes:
                    verification_results[model_key] = True
                    logger.debug(f"✅ {model_key} verified")
                else:
                    verification_results[model_key] = False
                    logger.warning(f"⚠️ {model_key} output shape mismatch: {output.shape}")
                
            except Exception as e:
                verification_results[model_key] = False
                logger.error(f"❌ {model_key} verification failed: {e}")
        
        return verification_results
    
    def print_summary(self):
        """Print summary of downloaded models."""
        
        cached_models = self.get_cached_models()
        
        print("\n" + "="*80)
        print("📋 MODEL DOWNLOAD SUMMARY")
        print("="*80)
        
        # Group by paper table
        tables = {}
        for model_key in MODEL_CONFIGS.keys():
            table = MODEL_CONFIGS[model_key]['paper_table']
            if table not in tables:
                tables[table] = []
            tables[table].append(model_key)
        
        for table, models in tables.items():
            print(f"\n{table}:")
            print("-" * 40)
            
            for model_key in models:
                status = "✅" if model_key in cached_models else "❌"
                config = MODEL_CONFIGS[model_key]
                params_m = config['expected_params'] / 1e6
                macs_g = config['expected_macs'] / 1e9
                
                print(f"{status} {model_key:<20} | {params_m:>6.1f}M params | {macs_g:>6.2f}G MACs | {config['architecture']}")
        
        print(f"\n📊 STATISTICS:")
        print(f"Total models available: {len(MODEL_CONFIGS)}")
        print(f"Models cached: {len(cached_models)}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cache size: {self._get_cache_size():.1f} MB")
        print("="*80)
    
    def _get_cache_size(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB

def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description="Download pretrained models for Multi-Agent LLM Pruning Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python download_models.py --all
  
  # Download specific models
  python download_models.py --models deit_small,resnet50,convnext_tiny
  
  # Download models for Table 1 reproduction
  python download_models.py --table1
  
  # Download models for Table 2 reproduction  
  python download_models.py --table2
  
  # Verify downloaded models
  python download_models.py --verify
  
  # List available models
  python download_models.py --list
        """
    )
    
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of models to download')
    parser.add_argument('--all', action='store_true',
                       help='Download all available models')
    parser.add_argument('--table1', action='store_true',
                       help='Download models for Table 1 reproduction (DeiT)')
    parser.add_argument('--table2', action='store_true', 
                       help='Download models for Table 2 reproduction (ConvNext)')
    parser.add_argument('--baselines', action='store_true',
                       help='Download baseline models (ResNet, MobileNet, etc.)')
    parser.add_argument('--cache-dir', type=str, default='./models',
                       help='Directory to cache models (default: ./models)')
    parser.add_argument('--force', action='store_true',
                       help='Force redownload even if cached')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded models')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel downloads (default: True)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level="INFO")
    
    # Create downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir, num_workers=args.num_workers)
    
    # List models
    if args.list:
        print("\n📋 Available Models:")
        print("="*60)
        for model_key, config in MODEL_CONFIGS.items():
            params_m = config['expected_params'] / 1e6
            macs_g = config['expected_macs'] / 1e9
            print(f"{model_key:<20} | {config['paper_table']:<10} | {params_m:>6.1f}M | {macs_g:>6.2f}G | {config['architecture']}")
        return
    
    # Verify models
    if args.verify:
        verification_results = downloader.verify_models()
        
        print("\n🔍 Model Verification Results:")
        print("="*50)
        for model_key, verified in verification_results.items():
            status = "✅ PASS" if verified else "❌ FAIL"
            print(f"{model_key:<20} | {status}")
        
        total_verified = sum(verification_results.values())
        print(f"\nVerified: {total_verified}/{len(verification_results)} models")
        return
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        models_to_download = list(MODEL_CONFIGS.keys())
    elif args.table1:
        models_to_download = [k for k, v in MODEL_CONFIGS.items() if v['paper_table'] == 'Table 1']
    elif args.table2:
        models_to_download = [k for k, v in MODEL_CONFIGS.items() if v['paper_table'] == 'Table 2']
    elif args.baselines:
        models_to_download = [k for k, v in MODEL_CONFIGS.items() if v['paper_table'] == 'Baseline']
    elif args.models:
        models_to_download = [m.strip() for m in args.models.split(',')]
    else:
        # Default: download essential models for quick start
        models_to_download = ['deit_small', 'resnet50', 'convnext_tiny']
        logger.info("No specific models specified. Downloading essential models for quick start.")
    
    # Validate model names
    invalid_models = [m for m in models_to_download if m not in MODEL_CONFIGS]
    if invalid_models:
        logger.error(f"Invalid model names: {invalid_models}")
        logger.error(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return 1
    
    # Download models
    logger.info(f"Starting download of {len(models_to_download)} models...")
    results = downloader.download_models(
        models_to_download, 
        force_redownload=args.force,
        parallel=args.parallel
    )
    
    # Print results
    successful = sum(1 for r in results.values() if r.get('download_success', False))
    failed = len(results) - successful
    
    logger.info(f"✅ Download completed: {successful} successful, {failed} failed")
    
    if failed > 0:
        logger.warning("Failed downloads:")
        for model_key, result in results.items():
            if not result.get('download_success', False):
                logger.warning(f"  - {model_key}: {result.get('error', 'Unknown error')}")
    
    # Print summary
    downloader.print_summary()
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

