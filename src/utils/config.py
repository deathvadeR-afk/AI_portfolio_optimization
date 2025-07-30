"""Configuration management utilities"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration settings"""
    equity_indices: list = field(default_factory=list)
    fixed_income: list = field(default_factory=list)
    commodities: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    frequency: str = "1d"
    lookback_period: str = "5y"
    storage_format: str = "parquet"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create DataConfig from dictionary"""
        market_data = config_dict.get('market_data', {})
        collection = config_dict.get('collection', {})
        storage = config_dict.get('storage', {})
        
        return cls(
            equity_indices=market_data.get('equity_indices', []),
            fixed_income=market_data.get('fixed_income', []),
            commodities=market_data.get('commodities', []),
            alternatives=market_data.get('alternatives', []),
            frequency=collection.get('frequency', '1d'),
            lookback_period=collection.get('lookback_period', '5y'),
            storage_format=storage.get('format', 'parquet')
        )


@dataclass
class EnvironmentConfig:
    """Environment configuration settings"""
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001
    lookback_window: int = 30
    episode_length: int = 252
    action_space_type: str = "continuous"
    reward_type: str = "multi_objective"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create EnvironmentConfig from dictionary"""
        env = config_dict.get('environment', {})
        action_space = env.get('action_space', {})
        reward = config_dict.get('reward', {})
        
        return cls(
            initial_balance=env.get('initial_balance', 100000.0),
            transaction_cost=env.get('transaction_cost', 0.001),
            lookback_window=env.get('lookback_window', 30),
            episode_length=env.get('episode_length', 252),
            action_space_type=action_space.get('type', 'continuous'),
            reward_type=reward.get('type', 'multi_objective')
        )


@dataclass
class AgentConfig:
    """Agent configuration settings"""
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dictionary"""
        return cls(
            algorithm=config_dict.get('algorithm', 'PPO'),
            learning_rate=config_dict.get('learning_rate', 3e-4),
            batch_size=config_dict.get('batch_size', 64),
            n_epochs=config_dict.get('n_epochs', 10),
            gamma=config_dict.get('gamma', 0.99),
            gae_lambda=config_dict.get('gae_lambda', 0.95),
            clip_range=config_dict.get('clip_range', 0.2)
        )


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._configs[config_name] = config
        return config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get configuration (load if not already loaded)"""
        if config_name not in self._configs:
            return self.load_config(config_name)
        return self._configs[config_name]
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        config_dict = self.get_config('data_config')
        return DataConfig.from_dict(config_dict)
    
    def get_env_config(self) -> EnvironmentConfig:
        """Get environment configuration"""
        config_dict = self.get_config('env_config')
        return EnvironmentConfig.from_dict(config_dict)
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        config_dict = self.get_config('agent_config')
        return AgentConfig.from_dict(config_dict)
    
    def save_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """Save configuration to YAML file"""
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        self._configs[config_name] = config
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """Update existing configuration"""
        config = self.get_config(config_name)
        config.update(updates)
        self.save_config(config_name, config)
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """Merge multiple configurations"""
        merged_config = {}
        
        for config_name in config_names:
            config = self.get_config(config_name)
            merged_config.update(config)
        
        return merged_config


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {}
    
    # Environment variables with CONFIG_ prefix
    for key, value in os.environ.items():
        if key.startswith('CONFIG_'):
            config_key = key[7:].lower()  # Remove CONFIG_ prefix
            
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                config[config_key] = value.lower() == 'true'
            elif value.isdigit():
                config[config_key] = int(value)
            else:
                try:
                    config[config_key] = float(value)
                except ValueError:
                    config[config_key] = value
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'data': {
            'frequency': '1d',
            'lookback_period': '5y',
            'storage_format': 'parquet'
        },
        'environment': {
            'initial_balance': 100000.0,
            'transaction_cost': 0.001,
            'lookback_window': 30,
            'episode_length': 252
        },
        'agent': {
            'algorithm': 'PPO',
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_epochs': 10
        }
    }


# Global config manager instance
config_manager = ConfigManager()


# Convenience functions
def get_data_config() -> DataConfig:
    """Get data configuration"""
    return config_manager.get_data_config()


def get_env_config() -> EnvironmentConfig:
    """Get environment configuration"""
    return config_manager.get_env_config()


def get_agent_config() -> AgentConfig:
    """Get agent configuration"""
    return config_manager.get_agent_config()
