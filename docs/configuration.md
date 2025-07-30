# Configuration Guide

This document provides a comprehensive guide to configuring the Portfolio Optimization system. All configuration files are located in the `configs/` directory and use YAML format for easy readability and modification.

## 📁 Configuration Files Overview

| File | Purpose | Lines | Key Sections |
|------|---------|-------|--------------|
| `main_config.yaml` | Master configuration file | 397 | Project metadata, execution modes, global settings |
| `data_config.yaml` | Data collection and processing | 193 | Market data sources, storage, feature engineering |
| `env_config.yaml` | RL environment settings | 214 | Portfolio environment, rewards, risk management |
| `agent_config.yaml` | RL agent configuration | 356 | PPO/DQN settings, neural networks, training |
| `training_config.yaml` | Training pipeline | 430 | Data preparation, training loops, evaluation |
| `backtest_config.yaml` | Backtesting framework | 530 | Test periods, benchmarks, performance metrics |

## 🚀 Quick Start

### 1. Basic Configuration
For a quick start with default settings:

```bash
# Copy environment variables template
cp .env.example .env

# Edit with your API keys
nano .env

# Test configurations
python scripts/test_configs.py
```

### 2. Essential Settings to Modify

**Data Collection (`data_config.yaml`):**
```yaml
# Add your FRED API key for economic data
economic_data:
  api_settings:
    fred:
      api_key: "your_fred_api_key_here"
```

**Training (`agent_config.yaml`):**
```yaml
# Adjust for your hardware
agent:
  device: "cuda"  # or "cpu" if no GPU

training:
  total_timesteps: 500000  # Reduce for faster training
```

**Portfolio (`env_config.yaml`):**
```yaml
environment:
  initial_balance: 100000.0  # Adjust starting capital
  transaction_cost: 0.001    # 0.1% transaction cost
```

## 📋 Detailed Configuration Reference

### 1. Main Configuration (`main_config.yaml`)

**Purpose:** Central coordination of all system components

**Key Sections:**
- `project`: Metadata and directory structure
- `modes`: Different execution modes (train, backtest, dashboard)
- `global`: System-wide settings (seed, logging, performance)
- `features`: Enable/disable experimental features

**Example:**
```yaml
project:
  name: "Multi-Asset Portfolio Optimization with RL"
  version: "1.0.0"

global:
  seed: 42
  logging:
    level: "INFO"
  
features:
  stable:
    basic_training: true
    standard_backtesting: true
```

### 2. Data Configuration (`data_config.yaml`)

**Purpose:** Define data sources, collection, and preprocessing

**Key Sections:**
- `market_data`: Asset symbols and data sources
- `collection`: Data frequency and validation rules
- `storage`: File formats and compression
- `features`: Technical indicators and risk metrics

**Asset Categories:**
```yaml
market_data:
  equity_indices: ["^GSPC", "^IXIC", "^DJI"]
  fixed_income: ["TLT", "IEF", "SHY"]
  commodities: ["GLD", "SLV", "USO"]
  alternatives: ["VNQ", "BTC-USD"]
```

**Feature Engineering:**
```yaml
features:
  technical:
    moving_averages: [5, 10, 20, 50, 200]
    rsi_periods: [14, 30]
  risk:
    volatility_windows: [10, 30, 60, 252]
```

### 3. Environment Configuration (`env_config.yaml`)

**Purpose:** RL environment settings and reward functions

**Key Sections:**
- `environment`: Basic portfolio parameters
- `action_space`: How agents can allocate capital
- `reward`: Reward function configuration
- `risk_management`: Position limits and risk controls

**Action Space Types:**
```yaml
action_space:
  type: "continuous"     # continuous, discrete, softmax, constrained
  min_weight: 0.0       # No short selling
  max_weight: 1.0       # Maximum 100% in any asset
  normalize: true       # Weights sum to 1
```

**Reward Functions:**
```yaml
reward:
  type: "multi_objective"
  multi_objective:
    return_weight: 0.6        # 60% weight on returns
    risk_weight: 0.3          # 30% weight on risk
    diversification_weight: 0.1  # 10% weight on diversification
```

### 4. Agent Configuration (`agent_config.yaml`)

**Purpose:** RL algorithm settings and neural network architecture

**Key Sections:**
- `agent`: General agent settings
- `ppo`: PPO-specific hyperparameters
- `dqn`: DQN-specific hyperparameters
- `network`: Neural network architecture
- `training`: Training loop configuration

**PPO Hyperparameters:**
```yaml
ppo:
  learning_rate: 3e-4      # Learning rate
  n_steps: 2048           # Steps per update
  batch_size: 64          # Minibatch size
  n_epochs: 10            # Optimization epochs
  gamma: 0.99             # Discount factor
  gae_lambda: 0.95        # GAE parameter
  clip_range: 0.2         # PPO clipping
```

**Network Architecture:**
```yaml
network:
  type: "mlp"
  mlp:
    hidden_dims: [256, 256, 128]
    activation: "relu"
    dropout: 0.0
```

### 5. Training Configuration (`training_config.yaml`)

**Purpose:** Training pipeline and data preparation

**Key Sections:**
- `training`: General training settings
- `data_preparation`: Asset selection and preprocessing
- `training_loop`: Multi-phase training
- `evaluation`: Evaluation during training

**Multi-Phase Training:**
```yaml
training_loop:
  phases:
    exploration:
      timesteps: 500000
      learning_rate: 5e-4
      exploration_noise: 0.3
    
    learning:
      timesteps: 1000000
      learning_rate: 3e-4
      exploration_noise: 0.2
```

**Curriculum Learning:**
```yaml
curriculum:
  enabled: true
  stages:
    simple:
      assets: ["SPY", "TLT"]
      timesteps: 200000
    full:
      assets: null  # All configured assets
      timesteps: 1300000
```

### 6. Backtest Configuration (`backtest_config.yaml`)

**Purpose:** Comprehensive backtesting and performance evaluation

**Key Sections:**
- `periods`: Train/validation/test periods
- `portfolio`: Portfolio settings and transaction costs
- `benchmarks`: Comparison strategies
- `metrics`: Performance metrics to calculate
- `reporting`: Output formats and visualizations

**Time Periods:**
```yaml
periods:
  train:
    start_date: "2018-01-01"
    end_date: "2022-12-31"
  test:
    start_date: "2024-01-01"
    end_date: "2024-12-31"
```

**Benchmark Strategies:**
```yaml
benchmarks:
  sp500:
    name: "S&P 500 Buy & Hold"
    type: "buy_and_hold"
    assets: ["^GSPC"]
  
  sixty_forty:
    name: "60/40 Portfolio"
    type: "static_allocation"
    assets: ["^GSPC", "TLT"]
    weights: [0.6, 0.4]
```

## ⚙️ Environment Variables

Set these in your `.env` file:

```bash
# API Keys
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Training Settings
TRAINING_DEVICE=cuda
BATCH_SIZE=64
LEARNING_RATE=0.0003

# Paths
DATA_PATH=data
MODEL_PATH=models
LOG_PATH=logs
```

## 🔧 Common Configuration Patterns

### Development vs Production

**Development:**
```yaml
global:
  logging:
    level: "DEBUG"
training:
  total_timesteps: 100000  # Faster training
evaluation:
  eval_freq: 5000         # More frequent evaluation
```

**Production:**
```yaml
global:
  logging:
    level: "INFO"
training:
  total_timesteps: 2000000  # Full training
evaluation:
  eval_freq: 25000         # Less frequent evaluation
```

### Hardware-Specific Settings

**CPU-Only:**
```yaml
agent:
  device: "cpu"
training:
  n_envs: 4              # Fewer parallel environments
resources:
  memory:
    max_memory_gb: 8     # Lower memory limit
```

**GPU-Enabled:**
```yaml
agent:
  device: "cuda"
training:
  n_envs: 16             # More parallel environments
resources:
  gpu:
    memory_fraction: 0.8
    mixed_precision: true
```

## 🧪 Testing Configurations

Always test your configurations before training:

```bash
# Validate all config files
python scripts/test_configs.py

# Test specific configuration loading
python -c "from src.utils.config import ConfigManager; cm = ConfigManager(); print(cm.get_data_config())"

# Dry run training with configs
python scripts/train.py --dry-run --episodes 1
```

## 🚨 Common Issues and Solutions

### 1. YAML Syntax Errors
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/agent_config.yaml'))"
```

### 2. Missing Required Sections
```bash
# Use the test script to identify missing sections
python scripts/test_configs.py
```

### 3. Invalid Parameter Values
```bash
# Check parameter constraints in main_config.yaml validation section
```

### 4. Memory Issues
```bash
# Reduce batch size and parallel environments
# In agent_config.yaml:
ppo:
  batch_size: 32
training:
  n_envs: 4
```

## 📚 Next Steps

1. **Customize for your use case**: Modify asset lists, time periods, and risk parameters
2. **Test configurations**: Run `python scripts/test_configs.py`
3. **Start with simple setup**: Use fewer assets and shorter training initially
4. **Monitor performance**: Use TensorBoard logging to track training progress
5. **Iterate and improve**: Adjust hyperparameters based on results

For more detailed information, see the individual configuration files which contain extensive comments explaining each parameter.
