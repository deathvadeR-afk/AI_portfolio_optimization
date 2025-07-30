"""
Advanced Risk Management System for Portfolio Optimization

This module provides comprehensive risk management capabilities including
VaR calculation, stress testing, portfolio constraints, position limits,
and real-time risk monitoring for production portfolio management.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size: float = 0.2  # Maximum position size (20%)
    max_sector_allocation: float = 0.4  # Maximum sector allocation (40%)
    max_portfolio_var: float = 0.05  # Maximum portfolio VaR (5%)
    max_drawdown: float = 0.15  # Maximum drawdown (15%)
    min_diversification_ratio: float = 1.2  # Minimum diversification ratio
    max_leverage: float = 1.0  # Maximum leverage (100%)
    max_turnover: float = 2.0  # Maximum annual turnover (200%)
    max_concentration: float = 0.5  # Maximum concentration index


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: pd.Timestamp
    alert_type: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    current_value: float
    limit_value: float
    asset: Optional[str] = None


class AdvancedRiskManager:
    """
    Advanced risk management system for portfolio optimization.
    
    Provides comprehensive risk monitoring, constraint enforcement,
    stress testing, and real-time risk analytics.
    """
    
    def __init__(self, 
                 risk_limits: Optional[RiskLimits] = None,
                 confidence_levels: List[float] = [0.95, 0.99],
                 lookback_window: int = 252):
        """
        Initialize the risk manager.
        
        Args:
            risk_limits: Risk limits configuration
            confidence_levels: VaR confidence levels
            lookback_window: Lookback window for risk calculations
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.confidence_levels = confidence_levels
        self.lookback_window = lookback_window
        
        # Risk monitoring
        self.alerts = []
        self.risk_metrics_history = []
        self.stress_test_results = {}
        
        logger.info(f"AdvancedRiskManager initialized with {len(confidence_levels)} VaR levels")
    
    def check_portfolio_constraints(self, 
                                  weights: np.ndarray,
                                  returns_data: pd.DataFrame,
                                  asset_names: List[str],
                                  sector_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Check portfolio against all risk constraints.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data
            asset_names: List of asset names
            sector_mapping: Mapping of assets to sectors
            
        Returns:
            Dictionary with constraint check results
        """
        logger.debug("Checking portfolio constraints...")
        
        results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'risk_metrics': {},
            'alerts': []
        }
        
        # Position size constraints
        position_violations = self._check_position_limits(weights, asset_names)
        if position_violations:
            results['violations'].extend(position_violations)
            results['passed'] = False
        
        # Sector concentration constraints
        if sector_mapping:
            sector_violations = self._check_sector_limits(weights, asset_names, sector_mapping)
            if sector_violations:
                results['violations'].extend(sector_violations)
                results['passed'] = False
        
        # Portfolio-level risk constraints
        portfolio_violations = self._check_portfolio_risk_limits(weights, returns_data)
        if portfolio_violations:
            results['violations'].extend(portfolio_violations)
            results['passed'] = False
        
        # Calculate comprehensive risk metrics
        risk_metrics = self._calculate_portfolio_risk_metrics(weights, returns_data)
        results['risk_metrics'] = risk_metrics
        
        # Generate alerts
        alerts = self._generate_risk_alerts(risk_metrics, weights, asset_names)
        results['alerts'] = alerts
        self.alerts.extend(alerts)
        
        logger.debug(f"Constraint check completed: {'PASSED' if results['passed'] else 'FAILED'}")
        return results
    
    def calculate_var_and_cvar(self, 
                              weights: np.ndarray,
                              returns_data: pd.DataFrame,
                              confidence_level: float = 0.95,
                              method: str = 'historical') -> Dict[str, float]:
        """
        Calculate Value at Risk and Conditional VaR.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data
            confidence_level: VaR confidence level
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dictionary with VaR and CVaR values
        """
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        if method == 'historical':
            var = self._calculate_historical_var(portfolio_returns, confidence_level)
            cvar = self._calculate_historical_cvar(portfolio_returns, confidence_level)
        
        elif method == 'parametric':
            var = self._calculate_parametric_var(portfolio_returns, confidence_level)
            cvar = self._calculate_parametric_cvar(portfolio_returns, confidence_level)
        
        elif method == 'monte_carlo':
            var, cvar = self._calculate_monte_carlo_var(portfolio_returns, confidence_level)
        
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'method': method
        }
    
    def stress_test_portfolio(self, 
                            weights: np.ndarray,
                            returns_data: pd.DataFrame,
                            scenarios: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive stress testing.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data
            scenarios: Custom stress test scenarios
            
        Returns:
            Stress test results
        """
        logger.info("Performing portfolio stress testing...")
        
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        stress_results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            logger.debug(f"Running stress scenario: {scenario_name}")
            
            scenario_result = self._run_stress_scenario(
                weights, returns_data, scenario_config
            )
            stress_results[scenario_name] = scenario_result
        
        # Historical stress testing
        historical_stress = self._run_historical_stress_test(weights, returns_data)
        stress_results['historical_worst_periods'] = historical_stress
        
        # Store results
        self.stress_test_results = stress_results
        
        logger.info(f"Stress testing completed for {len(scenarios)} scenarios")
        return stress_results
    
    def optimize_risk_adjusted_weights(self, 
                                     expected_returns: np.ndarray,
                                     cov_matrix: np.ndarray,
                                     risk_aversion: float = 1.0,
                                     constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights with risk constraints.
        
        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints
            
        Returns:
            Optimization results
        """
        logger.info("Optimizing risk-adjusted portfolio weights...")
        
        n_assets = len(expected_returns)
        
        # Objective function (negative utility)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Position limits
        bounds = [(0, self.risk_limits.max_position_size) for _ in range(n_assets)]
        
        # Additional constraints
        if constraints:
            if 'sector_limits' in constraints:
                sector_constraints = self._create_sector_constraints(constraints['sector_limits'])
                constraints_list.extend(sector_constraints)
            
            if 'turnover_limit' in constraints:
                turnover_constraint = self._create_turnover_constraint(
                    constraints['current_weights'], constraints['turnover_limit']
                )
                constraints_list.append(turnover_constraint)
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            optimization_result = {
                'success': True,
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_result': result
            }
        else:
            logger.warning(f"Portfolio optimization failed: {result.message}")
            optimization_result = {
                'success': False,
                'message': result.message,
                'optimization_result': result
            }
        
        return optimization_result
    
    def monitor_real_time_risk(self, 
                             current_weights: np.ndarray,
                             current_prices: pd.Series,
                             returns_data: pd.DataFrame,
                             asset_names: List[str]) -> Dict[str, Any]:
        """
        Monitor real-time portfolio risk.
        
        Args:
            current_weights: Current portfolio weights
            current_prices: Current asset prices
            returns_data: Historical returns data
            asset_names: Asset names
            
        Returns:
            Real-time risk monitoring results
        """
        timestamp = pd.Timestamp.now()
        
        # Calculate current risk metrics
        risk_metrics = self._calculate_portfolio_risk_metrics(current_weights, returns_data)
        
        # Check for violations
        violations = []
        
        # Position size violations
        for i, (weight, asset) in enumerate(zip(current_weights, asset_names)):
            if weight > self.risk_limits.max_position_size:
                violations.append({
                    'type': 'position_limit',
                    'asset': asset,
                    'current': weight,
                    'limit': self.risk_limits.max_position_size
                })
        
        # Portfolio VaR violation
        if 'var_95' in risk_metrics and abs(risk_metrics['var_95']) > self.risk_limits.max_portfolio_var:
            violations.append({
                'type': 'portfolio_var',
                'current': abs(risk_metrics['var_95']),
                'limit': self.risk_limits.max_portfolio_var
            })
        
        # Generate alerts
        alerts = []
        for violation in violations:
            severity = 'HIGH' if violation['current'] > violation['limit'] * 1.2 else 'MEDIUM'
            alert = RiskAlert(
                timestamp=timestamp,
                alert_type=violation['type'],
                severity=severity,
                message=f"{violation['type']} violation detected",
                current_value=violation['current'],
                limit_value=violation['limit'],
                asset=violation.get('asset')
            )
            alerts.append(alert)
        
        # Store monitoring results
        monitoring_result = {
            'timestamp': timestamp,
            'risk_metrics': risk_metrics,
            'violations': violations,
            'alerts': alerts,
            'portfolio_value': np.sum(current_weights * current_prices),
            'risk_score': self._calculate_risk_score(risk_metrics)
        }
        
        self.risk_metrics_history.append(monitoring_result)
        
        return monitoring_result
    
    # Helper methods for constraint checking
    def _check_position_limits(self, weights: np.ndarray, asset_names: List[str]) -> List[Dict]:
        """Check position size limits."""
        violations = []
        
        for i, (weight, asset) in enumerate(zip(weights, asset_names)):
            if weight > self.risk_limits.max_position_size:
                violations.append({
                    'type': 'position_limit',
                    'asset': asset,
                    'current': weight,
                    'limit': self.risk_limits.max_position_size,
                    'message': f"Position size {weight:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}"
                })
        
        return violations
    
    def _check_sector_limits(self, 
                           weights: np.ndarray, 
                           asset_names: List[str], 
                           sector_mapping: Dict[str, str]) -> List[Dict]:
        """Check sector concentration limits."""
        violations = []
        
        # Calculate sector allocations
        sector_allocations = {}
        for i, asset in enumerate(asset_names):
            sector = sector_mapping.get(asset, 'Unknown')
            sector_allocations[sector] = sector_allocations.get(sector, 0) + weights[i]
        
        # Check limits
        for sector, allocation in sector_allocations.items():
            if allocation > self.risk_limits.max_sector_allocation:
                violations.append({
                    'type': 'sector_limit',
                    'sector': sector,
                    'current': allocation,
                    'limit': self.risk_limits.max_sector_allocation,
                    'message': f"Sector {sector} allocation {allocation:.2%} exceeds limit {self.risk_limits.max_sector_allocation:.2%}"
                })
        
        return violations
    
    def _check_portfolio_risk_limits(self, weights: np.ndarray, returns_data: pd.DataFrame) -> List[Dict]:
        """Check portfolio-level risk limits."""
        violations = []
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # VaR check
        var_95 = self._calculate_historical_var(portfolio_returns, 0.95)
        if abs(var_95) > self.risk_limits.max_portfolio_var:
            violations.append({
                'type': 'portfolio_var',
                'current': abs(var_95),
                'limit': self.risk_limits.max_portfolio_var,
                'message': f"Portfolio VaR {abs(var_95):.2%} exceeds limit {self.risk_limits.max_portfolio_var:.2%}"
            })
        
        # Concentration check
        concentration = np.sum(weights**2)
        if concentration > self.risk_limits.max_concentration:
            violations.append({
                'type': 'concentration',
                'current': concentration,
                'limit': self.risk_limits.max_concentration,
                'message': f"Portfolio concentration {concentration:.3f} exceeds limit {self.risk_limits.max_concentration:.3f}"
            })
        
        return violations
    
    def _calculate_portfolio_risk_metrics(self, weights: np.ndarray, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio risk metrics."""
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'var_95': self._calculate_historical_var(portfolio_returns, 0.95),
            'var_99': self._calculate_historical_var(portfolio_returns, 0.99),
            'cvar_95': self._calculate_historical_cvar(portfolio_returns, 0.95),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'concentration': np.sum(weights**2),
            'effective_assets': 1 / np.sum(weights**2),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
        
        return metrics
    
    def _generate_risk_alerts(self, 
                            risk_metrics: Dict[str, float], 
                            weights: np.ndarray, 
                            asset_names: List[str]) -> List[RiskAlert]:
        """Generate risk alerts based on current metrics."""
        alerts = []
        timestamp = pd.Timestamp.now()
        
        # VaR alert
        if abs(risk_metrics.get('var_95', 0)) > self.risk_limits.max_portfolio_var * 0.8:
            severity = 'HIGH' if abs(risk_metrics['var_95']) > self.risk_limits.max_portfolio_var else 'MEDIUM'
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='var_warning',
                severity=severity,
                message='Portfolio VaR approaching limit',
                current_value=abs(risk_metrics['var_95']),
                limit_value=self.risk_limits.max_portfolio_var
            ))
        
        # Concentration alert
        if risk_metrics.get('concentration', 0) > self.risk_limits.max_concentration * 0.8:
            severity = 'HIGH' if risk_metrics['concentration'] > self.risk_limits.max_concentration else 'MEDIUM'
            alerts.append(RiskAlert(
                timestamp=timestamp,
                alert_type='concentration_warning',
                severity=severity,
                message='Portfolio concentration approaching limit',
                current_value=risk_metrics['concentration'],
                limit_value=self.risk_limits.max_concentration
            ))
        
        return alerts
    
    # VaR calculation methods
    def _calculate_historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical VaR."""
        return returns.quantile(1 - confidence_level)
    
    def _calculate_historical_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate historical CVaR."""
        var_threshold = self._calculate_historical_var(returns, confidence_level)
        return returns[returns <= var_threshold].mean()
    
    def _calculate_parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric VaR."""
        mean = returns.mean()
        std = returns.std()
        return mean + std * stats.norm.ppf(1 - confidence_level)
    
    def _calculate_parametric_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate parametric CVaR."""
        mean = returns.mean()
        std = returns.std()
        alpha = 1 - confidence_level
        var = self._calculate_parametric_var(returns, confidence_level)
        return mean - std * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    
    def _calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float, n_simulations: int = 10000) -> Tuple[float, float]:
        """Calculate Monte Carlo VaR and CVaR."""
        mean = returns.mean()
        std = returns.std()
        
        # Generate random scenarios
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        # Calculate VaR and CVaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()
        
        return var, cvar
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict]:
        """Get default stress test scenarios."""
        return {
            'market_crash': {
                'type': 'shock',
                'magnitude': -0.20,  # 20% market decline
                'correlation_increase': 0.3
            },
            'volatility_spike': {
                'type': 'volatility',
                'volatility_multiplier': 3.0
            },
            'interest_rate_shock': {
                'type': 'factor_shock',
                'factor': 'interest_rate',
                'magnitude': 0.02  # 200 bps increase
            },
            'liquidity_crisis': {
                'type': 'liquidity',
                'bid_ask_spread_increase': 5.0
            }
        }
    
    def _run_stress_scenario(self, weights: np.ndarray, returns_data: pd.DataFrame, scenario_config: Dict) -> Dict:
        """Run a single stress test scenario."""
        # Simplified stress testing implementation
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        if scenario_config['type'] == 'shock':
            # Apply market shock
            stressed_returns = portfolio_returns + scenario_config['magnitude']
            
        elif scenario_config['type'] == 'volatility':
            # Increase volatility
            mean_return = portfolio_returns.mean()
            stressed_returns = mean_return + (portfolio_returns - mean_return) * scenario_config['volatility_multiplier']
            
        else:
            # Default: no stress
            stressed_returns = portfolio_returns
        
        # Calculate stressed metrics
        stressed_var = self._calculate_historical_var(stressed_returns, 0.95)
        stressed_cvar = self._calculate_historical_cvar(stressed_returns, 0.95)
        stressed_max_dd = self._calculate_max_drawdown(stressed_returns)
        
        return {
            'scenario_config': scenario_config,
            'stressed_var': stressed_var,
            'stressed_cvar': stressed_cvar,
            'stressed_max_drawdown': stressed_max_dd,
            'portfolio_loss': stressed_returns.min()
        }
    
    def _run_historical_stress_test(self, weights: np.ndarray, returns_data: pd.DataFrame) -> Dict:
        """Run historical stress test using worst historical periods."""
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # Find worst periods
        worst_1d = portfolio_returns.min()
        worst_5d = portfolio_returns.rolling(5).sum().min()
        worst_20d = portfolio_returns.rolling(20).sum().min()
        
        return {
            'worst_1_day': worst_1d,
            'worst_5_day': worst_5d,
            'worst_20_day': worst_20d
        }
    
    def _calculate_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """Calculate overall risk score (0-100)."""
        # Simplified risk scoring
        var_score = min(abs(risk_metrics.get('var_95', 0)) / self.risk_limits.max_portfolio_var * 50, 50)
        concentration_score = min(risk_metrics.get('concentration', 0) / self.risk_limits.max_concentration * 30, 30)
        volatility_score = min(risk_metrics.get('volatility', 0) / 0.3 * 20, 20)  # Assume 30% vol as high
        
        return var_score + concentration_score + volatility_score
    
    def _create_sector_constraints(self, sector_limits: Dict[str, float]) -> List[Dict]:
        """Create sector constraint functions."""
        # Placeholder for sector constraints
        return []
    
    def _create_turnover_constraint(self, current_weights: np.ndarray, turnover_limit: float) -> Dict:
        """Create turnover constraint function."""
        def turnover_constraint(new_weights):
            turnover = np.sum(np.abs(new_weights - current_weights))
            return turnover_limit - turnover
        
        return {'type': 'ineq', 'fun': turnover_constraint}
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk management summary."""
        return {
            'risk_limits': self.risk_limits.__dict__,
            'active_alerts': len([a for a in self.alerts if a.severity in ['HIGH', 'CRITICAL']]),
            'total_alerts': len(self.alerts),
            'monitoring_history_length': len(self.risk_metrics_history),
            'stress_test_scenarios': len(self.stress_test_results),
            'last_monitoring': self.risk_metrics_history[-1] if self.risk_metrics_history else None
        }
