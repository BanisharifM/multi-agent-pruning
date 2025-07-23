#!/usr/bin/env python3
"""
Helper utilities for Multi-Agent LLM Pruning Framework

This module provides various helper functions for formatting,
data processing, and common operations used throughout the framework.
"""

import os
import sys
import time
import json
import yaml
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def format_number(num: Union[int, float], precision: int = 2, 
                 use_thousands_separator: bool = True) -> str:
    """
    Format a number with appropriate precision and separators.
    
    Args:
        num: Number to format
        precision: Decimal precision for floats
        use_thousands_separator: Whether to use thousands separator
        
    Returns:
        Formatted number string
    """
    
    if isinstance(num, int):
        if use_thousands_separator:
            return f"{num:,}"
        else:
            return str(num)
    elif isinstance(num, float):
        if use_thousands_separator:
            return f"{num:,.{precision}f}"
        else:
            return f"{num:.{precision}f}"
    else:
        return str(num)

def format_time(seconds: float, precision: int = 2) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        precision: Decimal precision
        
    Returns:
        Formatted time string
    """
    
    if seconds < 1:
        return f"{seconds * 1000:.{precision}f}ms"
    elif seconds < 60:
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.{precision}f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.{precision}f}s"

def format_bytes(bytes_value: Union[int, float], precision: int = 2) -> str:
    """
    Format byte size in human-readable format.
    
    Args:
        bytes_value: Size in bytes
        precision: Decimal precision
        
    Returns:
        Formatted size string
    """
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    
    if bytes_value == 0:
        return "0B"
    
    unit_index = 0
    size = float(bytes_value)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{precision}f}{units[unit_index]}"

def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a decimal value as a percentage.
    
    Args:
        value: Decimal value (0.0 to 1.0)
        precision: Decimal precision
        
    Returns:
        Formatted percentage string
    """
    
    return f"{value * 100:.{precision}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    
    if denominator == 0:
        return default
    return numerator / denominator

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    logger.info(f"📄 Loaded configuration from: {config_path}")
    return config

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    
    config_path = Path(config_path)
    ensure_dir(config_path.parent)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    logger.info(f"💾 Saved configuration to: {config_path}")

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device object
    """
    
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("💻 Using CPU device")
    
    return device

def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive size information for a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory usage
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_bytes = param_size_bytes + buffer_size_bytes
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_param_count,
        'non_trainable_parameters': param_count - trainable_param_count,
        'parameter_size_bytes': param_size_bytes,
        'buffer_size_bytes': buffer_size_bytes,
        'total_size_bytes': total_size_bytes,
        'parameter_size_mb': param_size_bytes / (1024 * 1024),
        'total_size_mb': total_size_bytes / (1024 * 1024)
    }

def print_model_summary(model: torch.nn.Module, model_name: str = "Model") -> None:
    """
    Print a comprehensive model summary.
    
    Args:
        model: PyTorch model
        model_name: Name for the model
    """
    
    size_info = get_model_size(model)
    
    print(f"\n🔍 {model_name} Summary")
    print("=" * 50)
    print(f"📊 Parameters:")
    print(f"   Total: {format_number(size_info['total_parameters'])}")
    print(f"   Trainable: {format_number(size_info['trainable_parameters'])}")
    print(f"   Non-trainable: {format_number(size_info['non_trainable_parameters'])}")
    
    print(f"\n💾 Memory Usage:")
    print(f"   Parameters: {format_bytes(size_info['parameter_size_bytes'])}")
    print(f"   Buffers: {format_bytes(size_info['buffer_size_bytes'])}")
    print(f"   Total: {format_bytes(size_info['total_size_bytes'])}")
    
    print("=" * 50)

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"🎲 Set random seed to: {seed}")

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary with system information
    """
    
    info = {
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'platform': sys.platform,
        'cpu_count': os.cpu_count()
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
        })
    
    return info

def print_system_info() -> None:
    """Print comprehensive system information."""
    
    info = get_system_info()
    
    print(f"\n💻 System Information")
    print("=" * 50)
    print(f"🐍 Python: {info['python_version'].split()[0]}")
    print(f"🔥 PyTorch: {info['pytorch_version']}")
    print(f"🖥️ Platform: {info['platform']}")
    print(f"⚙️ CPU Cores: {info['cpu_count']}")
    
    if info['cuda_available']:
        print(f"\n🚀 CUDA Information:")
        print(f"   CUDA Version: {info['cuda_version']}")
        print(f"   cuDNN Version: {info['cudnn_version']}")
        print(f"   GPU Count: {info['gpu_count']}")
        
        for i, (name, memory) in enumerate(zip(info['gpu_names'], info['gpu_memory'])):
            print(f"   GPU {i}: {name} ({format_bytes(memory)})")
    else:
        print(f"\n❌ CUDA: Not available")
    
    print("=" * 50)

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that a configuration contains all required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    
    missing_keys = []
    
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        logger.error(f"❌ Configuration validation failed. Missing keys: {missing_keys}")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten a dictionary with nested keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator used in nested keys
        
    Returns:
        Unflattened nested dictionary
    """
    
    result = {}
    
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result

def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original_size: Original size
        compressed_size: Compressed size
        
    Returns:
        Compression ratio (0.0 to 1.0)
    """
    
    if original_size == 0:
        return 0.0
    
    return 1.0 - (compressed_size / original_size)

def calculate_speedup(original_time: float, optimized_time: float) -> float:
    """
    Calculate speedup ratio.
    
    Args:
        original_time: Original execution time
        optimized_time: Optimized execution time
        
    Returns:
        Speedup ratio (>1.0 means faster)
    """
    
    if optimized_time == 0:
        return float('inf')
    
    return original_time / optimized_time

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
        print(f"🚀 {description}: 0/{total} (0.0%)")
    
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current = min(self.current + increment, self.total)
        self._print_progress()
    
    def set_progress(self, current: int):
        """Set absolute progress."""
        self.current = min(current, self.total)
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress."""
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed_time = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        print(f"⏳ {self.description}: {self.current}/{self.total} "
              f"({percentage:.1f}%) - ETA: {eta_str}")
    
    def finish(self):
        """Mark progress as finished."""
        self.current = self.total
        elapsed_time = time.time() - self.start_time
        print(f"✅ {self.description}: Completed in {format_time(elapsed_time)}")

# Convenience functions for common operations
def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    
    usage = {}
    
    if torch.cuda.is_available():
        usage['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
        usage['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        usage['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # Add system memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        usage['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        usage['cpu_memory_percent'] = process.memory_percent()
    except ImportError:
        pass
    
    return usage

