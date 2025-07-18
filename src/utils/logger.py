"""Logging utilities for portfolio optimization project"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class PortfolioLogger:
    """Custom logger for portfolio optimization"""
    
    def __init__(self, 
                 name: str = "portfolio_optimization",
                 level: str = "INFO",
                 log_dir: str = "logs",
                 console_output: bool = True,
                 file_output: bool = True):
        """
        Initialize portfolio logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directory for log files
            console_output: Whether to output to console
            file_output: Whether to output to file
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        self.simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Add handlers
        if console_output:
            self._add_console_handler()
        
        if file_output:
            self._add_file_handler()
    
    def _add_console_handler(self):
        """Add console handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add file handler"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{self.name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)


class MetricsLogger:
    """Logger for tracking portfolio metrics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
    def log_metrics(self, 
                   step: int,
                   episode: int,
                   metrics: Dict[str, Any],
                   timestamp: Optional[datetime] = None):
        """
        Log portfolio metrics.
        
        Args:
            step: Current step
            episode: Current episode
            metrics: Dictionary of metrics to log
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "step": step,
            "episode": episode,
            "metrics": metrics
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_portfolio_state(self,
                           step: int,
                           episode: int,
                           portfolio_value: float,
                           positions: Dict[str, float],
                           returns: float,
                           timestamp: Optional[datetime] = None):
        """Log portfolio state"""
        metrics = {
            "portfolio_value": portfolio_value,
            "positions": positions,
            "returns": returns
        }
        
        self.log_metrics(step, episode, metrics, timestamp)
    
    def log_training_metrics(self,
                           episode: int,
                           total_reward: float,
                           episode_length: int,
                           loss: Optional[float] = None,
                           timestamp: Optional[datetime] = None):
        """Log training metrics"""
        metrics = {
            "total_reward": total_reward,
            "episode_length": episode_length,
            "average_reward": total_reward / episode_length if episode_length > 0 else 0
        }
        
        if loss is not None:
            metrics["loss"] = loss
        
        self.log_metrics(0, episode, metrics, timestamp)


class PerformanceLogger:
    """Logger for tracking performance metrics"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.performance_file = self.log_dir / "performance.jsonl"
    
    def log_backtest_results(self,
                           strategy_name: str,
                           start_date: str,
                           end_date: str,
                           total_return: float,
                           annual_return: float,
                           volatility: float,
                           sharpe_ratio: float,
                           max_drawdown: float,
                           **additional_metrics):
        """Log backtesting results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "strategy_name": strategy_name,
            "start_date": start_date,
            "end_date": end_date,
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            **additional_metrics
        }
        
        with open(self.performance_file, 'a') as f:
            f.write(json.dumps(results) + '\n')


# Global logger instances
portfolio_logger = PortfolioLogger()
metrics_logger = MetricsLogger()
performance_logger = PerformanceLogger()


# Convenience functions
def get_logger(name: str = "portfolio_optimization") -> PortfolioLogger:
    """Get logger instance"""
    return PortfolioLogger(name)


def log_info(message: str, **kwargs):
    """Log info message using global logger"""
    portfolio_logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message using global logger"""
    portfolio_logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log error message using global logger"""
    portfolio_logger.error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message using global logger"""
    portfolio_logger.debug(message, **kwargs)
