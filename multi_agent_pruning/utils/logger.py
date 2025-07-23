#!/usr/bin/env python3
"""
Logger utilities for Multi-Agent LLM Pruning Framework

This module provides enhanced logging capabilities with structured
logging, performance tracking, and multi-agent coordination support.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        
        return super().format(record)

class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'agent_name'):
            log_entry['agent_name'] = record.agent_name
        if hasattr(record, 'phase'):
            log_entry['phase'] = record.phase
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
        
        return json.dumps(log_entry)

class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds agent-specific context."""
    
    def __init__(self, logger, agent_name: str, phase: Optional[str] = None):
        self.agent_name = agent_name
        self.phase = phase
        super().__init__(logger, {})
    
    def process(self, msg, kwargs):
        # Add agent context to log record
        extra = kwargs.get('extra', {})
        extra['agent_name'] = self.agent_name
        if self.phase:
            extra['phase'] = self.phase
        kwargs['extra'] = extra
        
        # Format message with agent prefix
        formatted_msg = f"[{self.agent_name}] {msg}"
        if self.phase:
            formatted_msg = f"[{self.agent_name}:{self.phase}] {msg}"
        
        return formatted_msg, kwargs
    
    def set_phase(self, phase: str):
        """Update the current phase."""
        self.phase = phase

def setup_logger(name: str = "multi_agent_pruning",
                level: str = "INFO",
                log_file: Optional[str] = None,
                console_output: bool = True,
                structured_logging: bool = False,
                log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a comprehensive logger for the multi-agent pruning framework.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output to console
        structured_logging: Whether to use structured JSON logging
        log_dir: Directory for log files (creates if doesn't exist)
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if structured_logging:
        formatter = StructuredFormatter()
        console_formatter = StructuredFormatter()
    else:
        # Standard format with timestamp and context
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        console_formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_dir:
            # Create log directory if it doesn't exist
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate log file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"📝 Logging to file: {log_file}")
    
    # Add system info to first log
    logger.info(f"🚀 Logger initialized: {name}")
    logger.info(f"   Level: {level}")
    logger.info(f"   Structured logging: {structured_logging}")
    logger.info(f"   Console output: {console_output}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)

def get_agent_logger(agent_name: str, base_logger: Optional[logging.Logger] = None,
                    phase: Optional[str] = None) -> AgentLoggerAdapter:
    """
    Get a logger adapter for a specific agent.
    
    Args:
        agent_name: Name of the agent
        base_logger: Base logger to adapt (uses default if None)
        phase: Current phase/stage of the agent
        
    Returns:
        AgentLoggerAdapter instance
    """
    
    if base_logger is None:
        base_logger = logging.getLogger("multi_agent_pruning")
    
    return AgentLoggerAdapter(base_logger, agent_name, phase)

class ExperimentLogger:
    """
    Specialized logger for experiment tracking and results.
    
    This logger provides structured logging for experiments with
    automatic result aggregation and performance tracking.
    """
    
    def __init__(self, experiment_id: str, log_dir: str = "logs/experiments"):
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific logger
        self.logger = setup_logger(
            name=f"experiment_{experiment_id}",
            log_file=str(self.log_dir / f"experiment_{experiment_id}.log"),
            structured_logging=True
        )
        
        # Experiment metadata
        self.start_time = datetime.now()
        self.metadata = {
            'experiment_id': experiment_id,
            'start_time': self.start_time.isoformat(),
            'status': 'running'
        }
        
        self.results = {}
        self.metrics = {}
        
        self.logger.info("🧪 Experiment started", extra=self.metadata)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.metadata['config'] = config
        self.logger.info("⚙️ Experiment configuration", extra={'config': config})
    
    def log_phase_start(self, phase: str, description: str = ""):
        """Log the start of an experiment phase."""
        phase_info = {
            'phase': phase,
            'description': description,
            'start_time': datetime.now().isoformat()
        }
        self.logger.info(f"🔄 Phase started: {phase}", extra=phase_info)
    
    def log_phase_end(self, phase: str, results: Optional[Dict[str, Any]] = None):
        """Log the end of an experiment phase."""
        phase_info = {
            'phase': phase,
            'end_time': datetime.now().isoformat(),
            'results': results or {}
        }
        self.logger.info(f"✅ Phase completed: {phase}", extra=phase_info)
        
        if results:
            self.results[phase] = results
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """Log a metric value."""
        metric_info = {
            'metric_name': name,
            'metric_value': value,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.logger.info(f"📊 Metric: {name} = {value}", extra=metric_info)
        
        # Store in metrics
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric_info)
    
    def log_error(self, error: Exception, phase: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """Log an experiment error."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'phase': phase,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.error(f"❌ Experiment error: {error}", extra=error_info)
    
    def log_warning(self, message: str, phase: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None):
        """Log an experiment warning."""
        warning_info = {
            'warning_message': message,
            'phase': phase,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.warning(f"⚠️ {message}", extra=warning_info)
    
    def finalize_experiment(self, status: str = "completed",
                          final_results: Optional[Dict[str, Any]] = None):
        """Finalize the experiment and log summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.metadata.update({
            'status': status,
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'final_results': final_results or {}
        })
        
        # Log experiment summary
        summary = {
            'experiment_id': self.experiment_id,
            'status': status,
            'duration': duration,
            'phases_completed': list(self.results.keys()),
            'metrics_tracked': list(self.metrics.keys()),
            'final_results': final_results or {}
        }
        
        self.logger.info(f"🏁 Experiment {status}: {self.experiment_id}", extra=summary)
        
        # Save experiment summary to separate file
        summary_file = self.log_dir / f"experiment_{self.experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'results': self.results,
                'metrics': self.metrics,
                'summary': summary
            }, f, indent=2)
        
        self.logger.info(f"💾 Experiment summary saved: {summary_file}")

# Global logger instance
_global_logger = None

def get_global_logger() -> logging.Logger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger

# Convenience functions
def debug(msg: str, **kwargs):
    """Log debug message using global logger."""
    get_global_logger().debug(msg, **kwargs)

def info(msg: str, **kwargs):
    """Log info message using global logger."""
    get_global_logger().info(msg, **kwargs)

def warning(msg: str, **kwargs):
    """Log warning message using global logger."""
    get_global_logger().warning(msg, **kwargs)

def error(msg: str, **kwargs):
    """Log error message using global logger."""
    get_global_logger().error(msg, **kwargs)

def critical(msg: str, **kwargs):
    """Log critical message using global logger."""
    get_global_logger().critical(msg, **kwargs)

