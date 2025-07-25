#!/usr/bin/env python3
"""
Logging Utilities for Multi-Agent LLM Pruning Framework

This module provides enhanced logging capabilities with structured logging,
agent-specific loggers, and experiment tracking.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import colorlog

class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""
    
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
        
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
        
        if hasattr(record, 'phase'):
            log_entry['phase'] = record.phase
        
        return json.dumps(log_entry)

class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds agent-specific context."""
    
    def __init__(self, logger, agent_name: str):
        super().__init__(logger, {'agent_name': agent_name})
        self.agent_name = agent_name
    
    def process(self, msg, kwargs):
        # Add agent name to log record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra']['agent_name'] = self.agent_name
        
        # Add agent emoji prefix
        agent_emojis = {
            'ProfilingAgent': '🔍',
            'MasterAgent': '🧠',
            'AnalysisAgent': '📊',
            'PruningAgent': '✂️',
            'FinetuningAgent': '🎯',
            'EvaluationAgent': '📈'
        }
        
        emoji = agent_emojis.get(self.agent_name, '🤖')
        msg = f"{emoji} [{self.agent_name}] {msg}"
        
        return msg, kwargs

class ExperimentLogger:
    """Logger for experiment tracking and results."""
    
    def __init__(self, experiment_id: str, log_dir: Union[str, Path] = "./logs"):
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific log file
        self.log_file = self.log_dir / f"experiment_{experiment_id}.jsonl"
        
        # Setup logger
        self.logger = logging.getLogger(f"experiment.{experiment_id}")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler for structured logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
    
    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment start with configuration."""
        self.logger.info("Experiment started", extra={
            'event_type': 'experiment_start',
            'config': config,
            'experiment_id': self.experiment_id
        })
    
    def log_phase_start(self, phase_name: str, phase_config: Dict[str, Any]):
        """Log phase start."""
        self.logger.info(f"Phase {phase_name} started", extra={
            'event_type': 'phase_start',
            'phase': phase_name,
            'phase_config': phase_config,
            'experiment_id': self.experiment_id
        })
    
    def log_phase_end(self, phase_name: str, results: Dict[str, Any]):
        """Log phase completion with results."""
        self.logger.info(f"Phase {phase_name} completed", extra={
            'event_type': 'phase_end',
            'phase': phase_name,
            'results': results,
            'experiment_id': self.experiment_id
        })
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        self.logger.info(f"Metric: {metric_name} = {value}", extra={
            'event_type': 'metric',
            'metric_name': metric_name,
            'metric_value': value,
            'step': step,
            'experiment_id': self.experiment_id
        })
    
    def log_error(self, error_message: str, error_details: Optional[Dict[str, Any]] = None):
        """Log an error."""
        self.logger.error(error_message, extra={
            'event_type': 'error',
            'error_details': error_details or {},
            'experiment_id': self.experiment_id
        })
    
    def log_experiment_end(self, final_results: Dict[str, Any]):
        """Log experiment completion."""
        self.logger.info("Experiment completed", extra={
            'event_type': 'experiment_end',
            'final_results': final_results,
            'experiment_id': self.experiment_id
        })

def setup_logging(log_level: str = "INFO", 
                 log_dir: Union[str, Path] = "./logs",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_structured: bool = False,
                 experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Setup comprehensive logging for the framework.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Enable structured JSON logging
        experiment_id: Experiment ID for tracking
        
    Returns:
        Dictionary with logger configuration
    """
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()

    if isinstance(log_level, int):
        root_logger.setLevel(log_level)
    else:
        root_logger.setLevel(getattr(logging, log_level.upper()))
 
    # Clear existing handlers
    root_logger.handlers.clear()
    
    loggers_created = []
    
    # Console logging with colors
    if enable_console:
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        loggers_created.append('console')
    
    # File logging
    if enable_file:
        # Main log file
        main_log_file = log_dir / "multi_agent_pruning.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        loggers_created.append('file')
        
        # Error log file
        error_log_file = log_dir / "errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        loggers_created.append('error_file')
    
    # Structured logging
    if enable_structured:
        structured_log_file = log_dir / "structured.jsonl"
        structured_handler = logging.FileHandler(structured_log_file)
        structured_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(structured_handler)
        loggers_created.append('structured')
    
    # Agent-specific loggers
    agent_loggers = {}
    agent_names = ['ProfilingAgent', 'MasterAgent', 'AnalysisAgent', 
                  'PruningAgent', 'FinetuningAgent', 'EvaluationAgent']
    
    for agent_name in agent_names:
        logger = logging.getLogger(f"agents.{agent_name}")
        agent_loggers[agent_name] = AgentLoggerAdapter(logger, agent_name)
        
        # Agent-specific log file
        if enable_file:
            agent_log_file = log_dir / f"{agent_name.lower()}.log"
            agent_handler = logging.FileHandler(agent_log_file)
            agent_handler.setFormatter(file_formatter)
            logger.addHandler(agent_handler)
    
    # Experiment logger
    experiment_logger = None
    if experiment_id:
        experiment_logger = ExperimentLogger(experiment_id, log_dir)
        loggers_created.append('experiment')
    
    # Log initial message
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"🚀 Logging system initialized: {', '.join(loggers_created)}")
    main_logger.info(f"📁 Log directory: {log_dir.absolute()}")
    
    return {
        'log_level': log_level,
        'log_dir': str(log_dir),
        'loggers_created': loggers_created,
        'agent_loggers': agent_loggers,
        'experiment_logger': experiment_logger,
        'main_log_file': str(log_dir / "multi_agent_pruning.log") if enable_file else None,
        'error_log_file': str(log_dir / "errors.log") if enable_file else None,
        'structured_log_file': str(log_dir / "structured.jsonl") if enable_structured else None
    }

def get_agent_logger(agent_name: str) -> AgentLoggerAdapter:
    """Get a logger adapter for a specific agent."""
    logger = logging.getLogger(f"agents.{agent_name}")
    return AgentLoggerAdapter(logger, agent_name)

def log_function_call(func_name: str, args: Dict[str, Any], 
                     results: Optional[Dict[str, Any]] = None,
                     execution_time: Optional[float] = None):
    """Log a function call with arguments and results."""
    logger = logging.getLogger(__name__)
    
    log_data = {
        'function': func_name,
        'arguments': args
    }
    
    if results is not None:
        log_data['results'] = results
    
    if execution_time is not None:
        log_data['execution_time'] = execution_time
    
    logger.debug(f"Function call: {func_name}", extra=log_data)

def log_model_info(model_name: str, model_info: Dict[str, Any]):
    """Log model information."""
    logger = logging.getLogger(__name__)
    logger.info(f"Model loaded: {model_name}", extra={
        'event_type': 'model_info',
        'model_name': model_name,
        'model_info': model_info
    })

def log_pruning_results(phase: str, results: Dict[str, Any]):
    """Log pruning results for a specific phase."""
    logger = logging.getLogger(__name__)
    logger.info(f"Pruning results - {phase}", extra={
        'event_type': 'pruning_results',
        'phase': phase,
        'results': results
    })

def configure_third_party_loggers():
    """Configure third-party library loggers to reduce noise."""
    
    # Reduce noise from common libraries
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'matplotlib',
        'PIL.PngImagePlugin',
        'transformers.tokenization_utils_base'
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

def create_run_logger(run_id: str, log_dir: Union[str, Path] = "./logs") -> logging.Logger:
    """Create a logger for a specific run."""
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"run.{run_id}")
    logger.setLevel(logging.INFO)
    
    # Run-specific log file
    run_log_file = log_dir / f"run_{run_id}.log"
    handler = logging.FileHandler(run_log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Convenience functions for common logging patterns
def log_phase_transition(from_phase: str, to_phase: str, context: Dict[str, Any] = None):
    """Log phase transition."""
    logger = logging.getLogger(__name__)
    logger.info(f"Phase transition: {from_phase} → {to_phase}", extra={
        'event_type': 'phase_transition',
        'from_phase': from_phase,
        'to_phase': to_phase,
        'context': context or {}
    })

def log_performance_metric(metric_name: str, value: float, 
                          baseline: Optional[float] = None,
                          improvement: Optional[float] = None):
    """Log performance metrics with optional comparison."""
    logger = logging.getLogger(__name__)
    
    log_data = {
        'event_type': 'performance_metric',
        'metric_name': metric_name,
        'value': value
    }
    
    if baseline is not None:
        log_data['baseline'] = baseline
    
    if improvement is not None:
        log_data['improvement'] = improvement
    
    message = f"Metric {metric_name}: {value}"
    if improvement is not None:
        message += f" (improvement: {improvement:+.2%})"
    
    logger.info(message, extra=log_data)

def log_system_info():
    """Log system information."""
    import platform
    import psutil
    import torch
    
    logger = logging.getLogger(__name__)
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        system_info['cuda_version'] = torch.version.cuda
        system_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
    
    logger.info("System information", extra={
        'event_type': 'system_info',
        'system_info': system_info
    })

# Initialize third-party logger configuration
configure_third_party_loggers()

