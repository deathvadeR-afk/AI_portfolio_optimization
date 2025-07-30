#!/usr/bin/env python3
"""
Simple Portfolio API

A minimal, working portfolio optimization API for testing dashboard connection.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Simple Portfolio Optimization API",
    description="Minimal portfolio optimization API for testing",
    version="1.0.0"
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
    risk_tolerance: str = "medium"
    investment_amount: float = 100000.0
    strategy: str = "mean_reversion"

class PortfolioResponse(BaseModel):
    success: bool
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    strategy_used: str
    recommendations: List[str]
    risk_metrics: Dict[str, float]

# Simple asset list
AVAILABLE_ASSETS = [
    "GSPC", "DJI", "IXIC", "RUT", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIPS",
    "GLD", "SLV", "USO", "UNG", "DBA",
    "VNQ", "BTC_USD", "ETH_USD"
]

def generate_simple_weights(assets: List[str], risk_tolerance: str) -> Dict[str, float]:
    """Generate simple equal weights with risk adjustments."""
    n_assets = len(assets)
    base_weights = np.ones(n_assets) / n_assets
    
    # Simple risk adjustments
    if risk_tolerance == "conservative":
        # Boost bonds and defensive assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['TLT', 'IEF', 'SHY', 'GLD']):
                base_weights[i] *= 1.3
    elif risk_tolerance == "aggressive":
        # Boost growth assets
        for i, asset in enumerate(assets):
            if any(x in asset for x in ['IXIC', 'RUT', 'BTC', 'ETH']):
                base_weights[i] *= 1.5
    
    # Normalize
    base_weights = base_weights / np.sum(base_weights)
    return {asset: float(weight) for asset, weight in zip(assets, base_weights)}

@app.get("/")
async def root():
    """Serve the main dashboard."""
    try:
        return FileResponse("web/portfolio_dashboard.html")
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return {"message": "Dashboard not found", "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "message": "Simple API is running"
    }

@app.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Simple portfolio optimization."""
    try:
        logger.info(f"Optimizing portfolio: {len(request.assets)} assets")
        
        # Validate assets
        invalid_assets = [asset for asset in request.assets if asset not in AVAILABLE_ASSETS]
        if invalid_assets:
            raise HTTPException(status_code=400, detail=f"Invalid assets: {invalid_assets}")
        
        # Generate simple weights
        weights = generate_simple_weights(request.assets, request.risk_tolerance)
        
        # Simple metrics based on risk tolerance
        if request.risk_tolerance == "conservative":
            expected_return = 0.06
            expected_volatility = 0.10
        elif request.risk_tolerance == "aggressive":
            expected_return = 0.12
            expected_volatility = 0.20
        else:
            expected_return = 0.08
            expected_volatility = 0.15
        
        sharpe_ratio = expected_return / expected_volatility
        
        return PortfolioResponse(
            success=True,
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            strategy_used=request.strategy,
            recommendations=[
                f"Portfolio optimized for {request.risk_tolerance} risk tolerance",
                f"Expected annual return: {expected_return:.1%}",
                f"Expected volatility: {expected_volatility:.1%}"
            ],
            risk_metrics={
                "var_95": expected_volatility * 1.65,
                "max_drawdown": expected_volatility * 2.0,
                "volatility": expected_volatility
            }
        )
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assets/available")
async def get_available_assets():
    """Get available assets."""
    return {
        "assets": AVAILABLE_ASSETS,
        "count": len(AVAILABLE_ASSETS)
    }

@app.get("/strategies/available")
async def get_available_strategies():
    """Get available strategies."""
    return {
        "strategies": [
            {
                "name": "mean_reversion",
                "description": "Mean reversion strategy"
            },
            {
                "name": "momentum",
                "description": "Momentum strategy"
            },
            {
                "name": "risk_parity",
                "description": "Risk parity strategy"
            }
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Simple Portfolio API...")
    print("Dashboard will be available at: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
