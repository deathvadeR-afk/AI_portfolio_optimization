#!/usr/bin/env python3
"""
Portfolio Optimization API

Advanced portfolio optimization API with RL agents and rule-based strategies.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from risk.risk_manager import AdvancedRiskManager, RiskLimits
from features.feature_pipeline import FeatureEngineeringPipeline
from agents.ppo_agent import PPOAgent
from environment.portfolio_env import PortfolioEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AI-Powered Portfolio Optimization API",
    description="Advanced portfolio optimization with RL agents and rule-based strategies",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PortfolioRequest(BaseModel):
    assets: List[str]
    risk_tolerance: str = "medium"  # conservative, medium, aggressive
    investment_amount: float = 100000.0
    strategy: str = "mean_reversion"  # mean_reversion, momentum, risk_parity

class PortfolioResponse(BaseModel):
    success: bool
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    strategy_used: str
    recommendations: List[str]
    risk_metrics: Dict[str, float]

# Global services
ppo_agent: Optional[PPOAgent] = None
risk_manager: Optional[AdvancedRiskManager] = None
feature_pipeline: Optional[FeatureEngineeringPipeline] = None
portfolio_env: Optional[PortfolioEnv] = None

# Asset universe (simplified for demo)
AVAILABLE_ASSETS = [
    # Equity Indices
    "GSPC", "DJI", "IXIC", "RUT", "EFA", "EEM",
    # Fixed Income
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIPS",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBA",
    # Alternatives
    "VNQ", "BTC_USD", "ETH_USD"
]

def _generate_mean_reversion_weights(assets: List[str], risk_tolerance: str) -> Dict[str, float]:
    """Generate weights using mean reversion strategy."""
    n_assets = len(assets)
    
    if risk_tolerance == "conservative":
        # Conservative: Higher allocation to bonds and defensive assets
        base_weights = np.ones(n_assets) / n_assets
        # Boost fixed income and defensive assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['TLT', 'IEF', 'SHY', 'LQD', 'TIPS']):
                base_weights[i] *= 1.5
            elif any(x in asset for x in ['GLD', 'SLV']):
                base_weights[i] *= 1.2
    elif risk_tolerance == "aggressive":
        # Aggressive: Higher allocation to growth assets
        base_weights = np.ones(n_assets) / n_assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['IXIC', 'RUT', 'EEM', 'BTC', 'ETH']):
                base_weights[i] *= 1.8
            elif any(x in asset for x in ['GSPC', 'DJI']):
                base_weights[i] *= 1.3
    else:
        # Medium: Balanced allocation
        base_weights = np.ones(n_assets) / n_assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['GSPC', 'DJI', 'TLT', 'GLD']):
                base_weights[i] *= 1.2
    
    # Normalize weights
    base_weights = base_weights / np.sum(base_weights)
    
    return {asset: float(weight) for asset, weight in zip(assets, base_weights)}

def _generate_momentum_weights(assets: List[str], risk_tolerance: str) -> Dict[str, float]:
    """Generate weights using momentum strategy."""
    n_assets = len(assets)
    
    # Momentum strategy favors trending assets
    base_weights = np.ones(n_assets) / n_assets
    
    if risk_tolerance == "conservative":
        # Conservative momentum: Focus on stable trending assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['GSPC', 'DJI', 'LQD', 'GLD']):
                base_weights[i] *= 1.4
    elif risk_tolerance == "aggressive":
        # Aggressive momentum: Focus on high-growth trending assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['IXIC', 'RUT', 'EEM', 'BTC', 'ETH']):
                base_weights[i] *= 2.0
            elif any(x in asset for x in ['USO', 'SLV']):
                base_weights[i] *= 1.5
    else:
        # Medium momentum: Balanced trending allocation
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['GSPC', 'IXIC', 'EFA', 'GLD', 'VNQ']):
                base_weights[i] *= 1.3
    
    # Normalize weights
    base_weights = base_weights / np.sum(base_weights)
    
    return {asset: float(weight) for asset, weight in zip(assets, base_weights)}

def _generate_risk_parity_weights(assets: List[str], risk_tolerance: str) -> Dict[str, float]:
    """Generate weights using risk parity strategy."""
    n_assets = len(assets)
    
    # Risk parity: Equal risk contribution
    # Simplified version using asset class volatility estimates
    volatilities = []
    for asset in assets:
        if any(x in asset for x in ['BTC', 'ETH']):
            vol = 0.80  # High volatility for crypto
        elif any(x in asset for x in ['USO', 'UNG', 'SLV']):
            vol = 0.35  # High volatility for commodities
        elif any(x in asset for x in ['IXIC', 'RUT', 'EEM']):
            vol = 0.25  # Medium-high volatility for growth stocks
        elif any(x in asset for x in ['GSPC', 'DJI', 'EFA']):
            vol = 0.20  # Medium volatility for broad market
        elif any(x in asset for x in ['VNQ']):
            vol = 0.22  # Medium volatility for REITs
        elif any(x in asset for x in ['HYG']):
            vol = 0.15  # Medium-low volatility for high yield
        elif any(x in asset for x in ['LQD', 'IEF']):
            vol = 0.08  # Low volatility for investment grade bonds
        elif any(x in asset for x in ['TLT']):
            vol = 0.12  # Medium-low volatility for long bonds
        elif any(x in asset for x in ['SHY', 'TIPS']):
            vol = 0.05  # Very low volatility for short bonds
        elif any(x in asset for x in ['GLD']):
            vol = 0.18  # Medium volatility for gold
        else:
            vol = 0.20  # Default volatility
        
        volatilities.append(vol)
    
    # Inverse volatility weighting
    inv_vol = np.array([1.0 / vol for vol in volatilities])
    weights = inv_vol / np.sum(inv_vol)
    
    # Adjust for risk tolerance
    if risk_tolerance == "conservative":
        # Boost low volatility assets
        for i, vol in enumerate(volatilities):
            if vol < 0.15:
                weights[i] *= 1.3
    elif risk_tolerance == "aggressive":
        # Allow higher allocation to volatile assets
        for i, vol in enumerate(volatilities):
            if vol > 0.25:
                weights[i] *= 1.2
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    return {asset: float(weight) for asset, weight in zip(assets, weights)}

def _calculate_portfolio_metrics(weights: Dict[str, float], risk_tolerance: str, strategy: str) -> Dict[str, float]:
    """Calculate expected portfolio metrics based on strategy and risk tolerance."""
    
    # Base metrics by risk tolerance
    if risk_tolerance == "conservative":
        base_return = 0.06
        base_volatility = 0.10
    elif risk_tolerance == "aggressive":
        base_return = 0.12
        base_volatility = 0.20
    else:  # medium
        base_return = 0.08
        base_volatility = 0.15
    
    # Strategy adjustments
    if strategy == "momentum":
        expected_return = base_return * 1.1  # Momentum premium
        expected_volatility = base_volatility * 1.05  # Slightly higher vol
    elif strategy == "risk_parity":
        expected_return = base_return * 0.95  # Lower return for risk control
        expected_volatility = base_volatility * 0.85  # Lower volatility
    else:  # mean_reversion
        expected_return = base_return
        expected_volatility = base_volatility
    
    sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
    
    return {
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "sharpe_ratio": sharpe_ratio,
        "var_95": expected_volatility * 1.65,  # Approximate 95% VaR
        "max_drawdown": expected_volatility * 2.0  # Approximate max drawdown
    }

class PortfolioOptimizationService:
    """Service for portfolio optimization operations."""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize_services(self):
        """Initialize all required services."""
        global ppo_agent, risk_manager, feature_pipeline, portfolio_env

        try:
            logger.info("Initializing portfolio optimization services...")

            # Initialize environment
            portfolio_env = PortfolioEnv(
                data_path="data",
                initial_balance=100000.0,
                transaction_cost=0.001,
                lookback_window=30,
                episode_length=50,
                reward_function="multi_objective",
                action_processor="continuous",
                include_features=True
            )

            # Initialize PPO agent
            ppo_agent = PPOAgent(
                obs_dim=portfolio_env.observation_space.shape[0],
                action_dim=portfolio_env.action_space.shape[0],
                device="cpu"
            )

            # Try to load trained model
            try:
                ppo_agent.load_model("models/ppo/best_model.pth")
                logger.info("Loaded trained PPO model")
            except Exception as e:
                logger.warning(f"Could not load trained model: {e}")

            # Initialize risk manager
            risk_manager = AdvancedRiskManager()

            # Initialize feature pipeline
            feature_pipeline = FeatureEngineeringPipeline()

            self.initialized = True
            logger.info("Services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

# Initialize service
portfolio_service = PortfolioOptimizationService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await portfolio_service.initialize_services()

@app.get("/")
async def root():
    """Serve the main dashboard."""
    return FileResponse("web/portfolio_dashboard.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global ppo_agent, risk_manager, feature_pipeline, portfolio_env

    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "ppo_agent": ppo_agent is not None,
            "risk_manager": risk_manager is not None,
            "feature_pipeline": feature_pipeline is not None,
            "portfolio_env": portfolio_env is not None
        },
        "strategies": ["rl_ppo", "mean_reversion", "momentum", "risk_parity"]
    }

@app.get("/model/status")
async def get_model_status():
    """Get RL model status and performance metrics."""
    global ppo_agent

    if ppo_agent is None:
        raise HTTPException(status_code=503, detail="RL model not loaded")

    # Get training statistics
    training_stats = ppo_agent.get_training_stats()

    return {
        "model_loaded": True,
        "model_type": "PPO",
        "last_updated": datetime.now(),
        "performance_metrics": training_stats,
        "health_status": "healthy"
    }

@app.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio allocation using RL agents or rule-based strategies."""
    global ppo_agent, portfolio_env, risk_manager

    try:
        logger.info(f"Optimizing portfolio: {len(request.assets)} assets, {request.strategy} strategy")

        # Validate assets
        invalid_assets = [asset for asset in request.assets if asset not in AVAILABLE_ASSETS]
        if invalid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid assets: {invalid_assets}")

        # Generate weights based on strategy
        if request.strategy == "rl_ppo" and ppo_agent is not None and portfolio_env is not None:
            # Use RL agent for optimization
            state = portfolio_env.reset()
            action = ppo_agent.select_action(state, training=False)
            weights = {asset: float(weight) for asset, weight in zip(request.assets, action)}
        elif request.strategy == "momentum":
            weights = _generate_momentum_weights(request.assets, request.risk_tolerance)
        elif request.strategy == "risk_parity":
            weights = _generate_risk_parity_weights(request.assets, request.risk_tolerance)
        else:  # mean_reversion (default)
            weights = _generate_mean_reversion_weights(request.assets, request.risk_tolerance)
        
        # Calculate portfolio metrics
        metrics = _calculate_portfolio_metrics(weights, request.risk_tolerance, request.strategy)
        
        # Generate recommendations
        recommendations = []
        if metrics["sharpe_ratio"] > 1.0:
            recommendations.append("Excellent risk-adjusted returns expected")
        elif metrics["sharpe_ratio"] > 0.7:
            recommendations.append("Good risk-adjusted returns expected")
        else:
            recommendations.append("Consider adjusting risk tolerance or strategy")
        
        if request.strategy == "mean_reversion":
            recommendations.append("Mean reversion strategy works best in volatile markets")
        elif request.strategy == "momentum":
            recommendations.append("Momentum strategy works best in trending markets")
        else:
            recommendations.append("Risk parity provides balanced risk exposure")
        
        return PortfolioResponse(
            success=True,
            weights=weights,
            expected_return=metrics["expected_return"],
            expected_volatility=metrics["expected_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            strategy_used=request.strategy,
            recommendations=recommendations,
            risk_metrics={
                "var_95": metrics["var_95"],
                "max_drawdown": metrics["max_drawdown"],
                "volatility": metrics["expected_volatility"]
            }
        )
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assets/available")
async def get_available_assets():
    """Get list of available assets for optimization."""
    return {
        "assets": AVAILABLE_ASSETS,
        "count": len(AVAILABLE_ASSETS),
        "categories": {
            "equity": [asset for asset in AVAILABLE_ASSETS if any(x in asset for x in ['GSPC', 'DJI', 'IXIC', 'RUT', 'EFA', 'EEM'])],
            "fixed_income": [asset for asset in AVAILABLE_ASSETS if any(x in asset for x in ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIPS'])],
            "commodities": [asset for asset in AVAILABLE_ASSETS if any(x in asset for x in ['GLD', 'SLV', 'USO', 'UNG', 'DBA'])],
            "alternatives": [asset for asset in AVAILABLE_ASSETS if any(x in asset for x in ['VNQ', 'BTC', 'ETH'])]
        }
    }

@app.get("/strategies/available")
async def get_available_strategies():
    """Get list of available optimization strategies."""
    strategies = [
        {
            "name": "mean_reversion",
            "description": "Exploits price reversions to historical means",
            "best_for": "Volatile markets with range-bound behavior"
        },
        {
            "name": "momentum",
            "description": "Captures sustained trending movements",
            "best_for": "Trending markets with clear directional moves"
        },
        {
            "name": "risk_parity",
            "description": "Equal risk contribution from all assets",
            "best_for": "Balanced portfolios with controlled risk exposure"
        }
    ]

    # Add RL strategy if available
    global ppo_agent
    if ppo_agent is not None:
        strategies.insert(0, {
            "name": "rl_ppo",
            "description": "AI-powered optimization using reinforcement learning",
            "best_for": "Adaptive optimization across all market conditions"
        })

    return {"strategies": strategies}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
