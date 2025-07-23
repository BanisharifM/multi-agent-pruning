#!/usr/bin/env python3
"""
Profiler utilities for Multi-Agent LLM Pruning Framework

This module provides timing and performance profiling capabilities
for monitoring and optimizing the pruning pipeline.
"""

import time
import torch
import functools
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimingResult:
    """Container for timing measurement results."""
    name: str
    duration: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any]

class TimingProfiler:
    """
    Comprehensive timing profiler for performance monitoring.
    
    This profiler tracks execution times for different components
    of the pruning pipeline and provides detailed statistics.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, List[TimingResult]] = {}
        self.active_timers: Dict[str, float] = {}
        
        logger.info(f"⏱️ TimingProfiler initialized (enabled: {enabled})")
    
    @contextmanager
    def timer(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for timing code blocks.
        
        Usage:
            with profiler.timer("model_loading"):
                model = load_model()
        """
        
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        
        # Use CUDA events for GPU timing if available
        cuda_start = None
        cuda_end = None
        if torch.cuda.is_available():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        
        try:
            yield
        finally:
            end_time = time.time()
            
            # Get CUDA timing if available
            if cuda_start and cuda_end:
                cuda_end.record()
                torch.cuda.synchronize()
                gpu_duration = cuda_start.elapsed_time(cuda_end) / 1000.0  # Convert to seconds
            else:
                gpu_duration = None
            
            # Calculate CPU duration
            cpu_duration = end_time - start_time
            
            # Use GPU timing if available and significantly different
            duration = gpu_duration if gpu_duration is not None else cpu_duration
            
            # Create timing result
            result_metadata = metadata or {}
            if gpu_duration is not None:
                result_metadata.update({
                    'cpu_duration': cpu_duration,
                    'gpu_duration': gpu_duration,
                    'timing_source': 'cuda'
                })
            else:
                result_metadata['timing_source'] = 'cpu'
            
            result = TimingResult(
                name=name,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                metadata=result_metadata
            )
            
            # Store result
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(result)
            
            logger.debug(f"⏱️ {name}: {duration:.3f}s")
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        if not self.enabled:
            return
        
        self.active_timers[name] = time.time()
        logger.debug(f"⏱️ Started timer: {name}")
    
    def end_timer(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """End a named timer and return the duration."""
        if not self.enabled:
            return None
        
        if name not in self.active_timers:
            logger.warning(f"⚠️ Timer '{name}' was not started")
            return None
        
        start_time = self.active_timers.pop(name)
        end_time = time.time()
        duration = end_time - start_time
        
        # Create timing result
        result = TimingResult(
            name=name,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {}
        )
        
        # Store result
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(result)
        
        logger.debug(f"⏱️ {name}: {duration:.3f}s")
        return duration
    
    def get_statistics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get timing statistics.
        
        Args:
            name: Specific timer name (None for all timers)
            
        Returns:
            Dictionary with timing statistics
        """
        
        if name is not None:
            if name not in self.timings:
                return {}
            
            timings = self.timings[name]
            durations = [t.duration for t in timings]
            
            return {
                'name': name,
                'count': len(durations),
                'total_time': sum(durations),
                'average_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'last_time': durations[-1] if durations else 0.0
            }
        else:
            # Return statistics for all timers
            stats = {}
            total_time = 0.0
            total_calls = 0
            
            for timer_name in self.timings:
                timer_stats = self.get_statistics(timer_name)
                stats[timer_name] = timer_stats
                total_time += timer_stats['total_time']
                total_calls += timer_stats['count']
            
            stats['_summary'] = {
                'total_timers': len(self.timings),
                'total_time': total_time,
                'total_calls': total_calls,
                'average_call_time': total_time / total_calls if total_calls > 0 else 0.0
            }
            
            return stats
    
    def get_top_consumers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N time-consuming operations."""
        
        all_stats = []
        for name in self.timings:
            stats = self.get_statistics(name)
            if stats:
                all_stats.append(stats)
        
        # Sort by total time
        all_stats.sort(key=lambda x: x['total_time'], reverse=True)
        
        return all_stats[:n]
    
    def reset(self, name: Optional[str] = None):
        """
        Reset timing data.
        
        Args:
            name: Specific timer to reset (None for all timers)
        """
        
        if name is not None:
            if name in self.timings:
                del self.timings[name]
                logger.info(f"🗑️ Reset timer: {name}")
        else:
            self.timings.clear()
            self.active_timers.clear()
            logger.info("🗑️ Reset all timers")
    
    def print_summary(self, top_n: int = 10):
        """Print a comprehensive timing summary."""
        
        if not self.enabled:
            print("⏱️ Profiler is disabled")
            return
        
        print(f"\n⏱️ Timing Profiler Summary")
        print("=" * 60)
        
        stats = self.get_statistics()
        
        if '_summary' in stats:
            summary = stats['_summary']
            print(f"📊 Overall Statistics:")
            print(f"   Total timers: {summary['total_timers']}")
            print(f"   Total calls: {summary['total_calls']}")
            print(f"   Total time: {summary['total_time']:.3f}s")
            print(f"   Average call time: {summary['average_call_time']:.3f}s")
        
        print(f"\n🏆 Top {top_n} Time Consumers:")
        top_consumers = self.get_top_consumers(top_n)
        
        for i, timer_stats in enumerate(top_consumers, 1):
            print(f"   {i}. {timer_stats['name']}")
            print(f"      Total: {timer_stats['total_time']:.3f}s "
                  f"({timer_stats['count']} calls)")
            print(f"      Average: {timer_stats['average_time']:.3f}s "
                  f"(min: {timer_stats['min_time']:.3f}s, "
                  f"max: {timer_stats['max_time']:.3f}s)")
        
        print("=" * 60)
    
    def export_data(self) -> Dict[str, Any]:
        """Export all timing data for external analysis."""
        
        export_data = {
            'enabled': self.enabled,
            'timings': {},
            'statistics': self.get_statistics()
        }
        
        # Export raw timing data
        for name, timing_list in self.timings.items():
            export_data['timings'][name] = [
                {
                    'duration': t.duration,
                    'start_time': t.start_time,
                    'end_time': t.end_time,
                    'metadata': t.metadata
                }
                for t in timing_list
            ]
        
        return export_data

# Decorator for automatic function timing
def time_it(profiler: TimingProfiler, name: Optional[str] = None):
    """
    Decorator for timing function execution.
    
    Args:
        profiler: TimingProfiler instance
        name: Custom name for the timer (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__qualname__}"
            with profiler.timer(timer_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Async version
def time_it_async(profiler: TimingProfiler, name: Optional[str] = None):
    """
    Decorator for timing async function execution.
    
    Args:
        profiler: TimingProfiler instance
        name: Custom name for the timer (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__qualname__}"
            with profiler.timer(timer_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Global profiler instance
global_profiler = TimingProfiler()

# Convenience functions using global profiler
def timer(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function using global profiler."""
    return global_profiler.timer(name, metadata)

def start_timer(name: str):
    """Convenience function using global profiler."""
    return global_profiler.start_timer(name)

def end_timer(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function using global profiler."""
    return global_profiler.end_timer(name, metadata)

def get_timing_statistics(name: Optional[str] = None):
    """Convenience function using global profiler."""
    return global_profiler.get_statistics(name)

def print_timing_summary(top_n: int = 10):
    """Convenience function using global profiler."""
    return global_profiler.print_summary(top_n)

def reset_timers(name: Optional[str] = None):
    """Convenience function using global profiler."""
    return global_profiler.reset(name)

