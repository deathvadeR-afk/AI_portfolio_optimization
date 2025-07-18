# 🚀 AI-Powered Portfolio Optimization System

A comprehensive, production-ready portfolio optimization system featuring advanced reinforcement learning, smart rule-based strategies, and professional-grade risk management.

## ✨ Key Features

### � **Advanced AI & Machine Learning**
- **Reinforcement Learning (PPO)**: State-of-the-art RL agent with 670K+ parameters
- **Overnight Training System**: Automated 6-8 hour training pipeline with monitoring
- **Smart Rule-Based Algorithms**: Mean reversion, momentum, risk parity strategies
- **Multi-Objective Optimization**: Risk-adjusted returns with Sharpe ratio optimization

### 📊 **Professional Portfolio Management**
- **Real-Time Data Integration**: Live market data from multiple sources
- **Advanced Risk Management**: VaR, CVaR, maximum drawdown, correlation analysis
- **Dynamic Asset Allocation**: Automatic rebalancing across asset classes
- **Performance Analytics**: Comprehensive backtesting and strategy comparison

### 🌐 **Production-Ready Infrastructure**
- **Interactive Web Dashboard**: Real-time portfolio monitoring and control
- **RESTful API**: Easy integration with external systems
- **Docker Support**: Containerized deployment
- **Comprehensive Logging**: Full audit trail and monitoring

### 🎯 **Asset Classes Supported (90+ Assets)**
- **Equities**: S&P 500, NASDAQ, Dow Jones, International (EFA, EEM)
- **Fixed Income**: Treasury bonds (TLT, IEF, SHY), Corporate bonds (LQD, HYG)
- **Commodities**: Gold (GLD), Silver (SLV), Oil (USO), Agriculture (DBA)
- **Cryptocurrencies**: Bitcoin, Ethereum, and major altcoins
- **Real Estate**: REITs and real estate ETFs

## � Project Structure (Cleaned & Organized)

```
├── working_api.py              # 🔥 Main API server (CORE)
├── web/
│   └── portfolio_dashboard.html # 🌐 Main dashboard frontend (CORE)
├── src/                        # 📦 Source code modules
├── tests/                      # 🧪 All test files (organized)
│   ├── test_post_cleanup.py    # Post-cleanup verification
│   ├── test_dashboard.py       # Dashboard testing
│   └── test_server.py          # Server testing
├── archive/                    # 📚 Alternative implementations (archived)
│   ├── minimal_api.py          # Minimal API version
│   ├── simple_api.py           # Simple API version
│   └── start_api.py            # API starter script
├── configs/                    # ⚙️ Configuration files
├── data/                       # 💾 Data storage
├── models/                     # 🤖 Model storage
├── logs/                       # 📝 Log files
├── scripts/                    # 🔧 Utility scripts
├── docs/                       # 📖 Documentation
├── requirements.txt            # 📋 Python dependencies
└── venv/                       # 🐍 Virtual environment
```

## �🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd portfolio_optimization_project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the application:**
```bash
python working_api.py
```

4. **Open dashboard:**
Navigate to `http://localhost:8000` in your browser

### 🌙 **Overnight RL Training (Optional)**
```bash
# Start overnight training (6-8 hours)
python start_overnight_training.py

# Check training progress
python check_training_status.py

# Deploy trained model
python deploy_rl_model.py --auto-find-best
```

## 🏗️ Architecture

```
Portfolio_optimization_project/
├── 🌐 web/                  # Production web dashboard
├── 🤖 src/                  # AI & ML source code
│   ├── agents/             # RL agents (PPO, DQN)
│   ├── api/                # FastAPI REST endpoints
│   ├── environment/        # Portfolio RL environment
│   ├── risk/               # Risk management system
│   ├── features/           # Feature engineering
│   ├── data/               # Data pipeline
│   ├── backtesting/        # Strategy evaluation
│   └── utils/              # Shared utilities
├── 📊 data/                 # Market data storage
├── ⚙️ configs/              # YAML configurations
├── 📈 results/              # Performance analytics
├── 📝 logs/                 # System logs
├── 🧪 tests/                # Unit tests
├── 📚 notebooks/            # Research & analysis
├── 📜 scripts/              # Training & evaluation
├── 🐳 Dockerfile           # Container deployment
├── 🚀 start_api.py         # API server launcher
└── 📋 requirements.txt     # Dependencies
```

## 🌙 **Overnight RL Training System**

### **Automated Training Pipeline**

Train a professional-grade RL model overnight with our automated system:

```bash
# Start overnight training (6-8 hours)
python start_overnight_training.py
```

**Training Features:**
- ✅ **3,000 Episodes**: Comprehensive training for optimal performance
- ✅ **Real-Time Monitoring**: Progress tracking and performance metrics
- ✅ **Auto-Recovery**: Handles errors and memory issues automatically
- ✅ **GPU Optimization**: Automatic GPU detection and CPU fallback
- ✅ **Model Validation**: Automatic testing of trained models

### **Monitor Training Progress**

```bash
# Check training status anytime
python check_training_status.py

# View live training logs
Get-Content overnight_training.log -Wait
```

### **Deploy Trained Model**

```bash
# Deploy the best trained model
python deploy_rl_model.py --auto-find-best

# Validate model performance
python deploy_rl_model.py --validate-only
```

## 🚀 **Production Deployment**

### **One-Command Launch**

Deploy the complete platform with a single command:

```bash
python start_api.py
```

This launches:
- ✅ **FastAPI Server** on `http://localhost:8000`
- ✅ **Interactive Dashboard** with real-time portfolio monitoring
- ✅ **RESTful API** for programmatic access
- ✅ **Data Pipeline** for live market data
- ✅ **Risk Management** system
- ✅ **Multiple Strategies** (Mean Reversion, Momentum, Risk Parity)

### **Docker Deployment**

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

### **Manual Installation**

#### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deathvadeR-afk/AI_portfolio_optimization.git
   cd portfolio_optimization_project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or
   source venv/bin/activate      # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

### Basic Usage

1. **Collect Data**
   ```bash
   python scripts/collect_data.py
   ```

2. **Evaluate Strategies**
   ```bash
   python scripts/evaluate_strategies.py
   ```

3. **Launch Dashboard**
   ```bash
   python start_api.py
   ```

## 📊 Features

### Data Collection & Processing
- **Multi-source data collection**: Yahoo Finance, FRED API, alternative data
- **Asset classes**: Equities, Fixed Income, Commodities, Cryptocurrencies, REITs
- **Feature engineering**: Technical indicators, risk metrics, economic features
- **Data validation**: Quality checks, outlier detection, missing data handling

### Portfolio Strategies
- **Mean Reversion**: Exploits price reversions to historical means
- **Momentum**: Captures sustained trending movements
- **Risk Parity**: Equal risk contribution allocation
- **Multi-Objective**: Optimizes risk-adjusted returns

### Risk Management
- **Portfolio constraints**: Position limits, sector allocations
- **Risk metrics**: VaR, Expected Shortfall, Maximum Drawdown
- **Dynamic risk adjustment**: Regime-based risk management
- **Stop-loss mechanisms**: Automated downside protection

### Backtesting & Evaluation
- **Walk-forward analysis**: Time-series cross-validation
- **Performance metrics**: Returns, Sharpe ratio, Sortino ratio, Calmar ratio
- **Benchmark comparison**: S&P 500, 60/40 portfolio, equal weight
- **Transaction cost modeling**: Realistic trading costs

## 🔧 Configuration

The system uses YAML configuration files for easy customization:

- `configs/data_config.yaml`: Data collection and processing settings
- `configs/main_config.yaml`: Main system configuration
- `configs/backtest_config.yaml`: Backtesting configuration

## 🆕 **Latest Updates & Features**

### **Version 2.0 - Production Ready**
- ✅ **Smart Portfolio Strategies**: Professional rule-based algorithms
- ✅ **Professional Web Dashboard**: Real-time portfolio monitoring
- ✅ **Advanced Risk Management**: VaR, CVaR, correlation analysis
- ✅ **Multi-Asset Support**: 90+ assets across all major classes
- ✅ **Smart Algorithms**: Mean reversion, momentum, risk parity
- ✅ **Production API**: RESTful endpoints for external integration
- ✅ **Docker Support**: Containerized deployment
- ✅ **Comprehensive Logging**: Full audit trail and monitoring

### **Strategy System Features**
- � **Buffer Overflow Protection**: Automatic memory management
- 🔧 **Multiple Strategies**: Mean reversion, momentum, risk parity
- 🔧 **Real-Time Analytics**: Live performance monitoring
- 🔧 **Risk Management**: Advanced portfolio protection
- 🔧 **Dynamic Rebalancing**: Automatic portfolio adjustments
- 🔧 **Backtesting**: Historical performance analysis

### **Dashboard Features**
- 📊 **Real-Time Updates**: Live portfolio performance
- 📊 **Strategy Comparison**: Compare multiple strategies
- 📊 **Risk Analytics**: Advanced risk metrics and visualizations
- 📊 **Asset Allocation**: Dynamic rebalancing controls
- 📊 **Performance Charts**: Interactive historical analysis

## �📈 Performance

Our smart strategy approach has demonstrated:
- **Higher Sharpe ratios** compared to traditional optimization
- **Lower maximum drawdowns** through dynamic risk management
- **Better regime adaptation** during market transitions
- **Improved diversification** across asset classes
- **Professional-grade returns** with advanced risk management

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_data.py -v         # Data pipeline tests
pytest tests/test_api.py -v          # API tests
pytest tests/test_strategies.py -v   # Strategy tests
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Usage Guide](docs/usage.md)
- [Development Guide](docs/development.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [FRED API](https://fred.stlouisfed.org/) for economic data
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [NumPy](https://numpy.org/) and [Pandas](https://pandas.pydata.org/) for data processing

## 📞 Contact

sagarroy54321@gmail.com

---

**Disclaimer**: This software is for educational and research purposes only. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.
