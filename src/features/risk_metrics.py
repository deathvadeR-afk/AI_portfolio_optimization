"""
Risk Metrics Calculator for Portfolio Optimization

This module provides comprehensive risk metrics and portfolio analytics
including VaR, CVaR, drawdown analysis, correlation metrics, and
advanced risk measures for portfolio optimization.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import minimize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """
    Professional-grade risk metrics calculator for portfolio analysis.
    
    Provides comprehensive risk measures including VaR, CVaR, drawdown analysis,
    correlation metrics, and advanced portfolio risk analytics.
    """
    
    def __init__(self, returns_data: pd.DataFrame, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize with returns data.
        
        Args:
            returns_data: DataFrame with asset returns (assets as columns)
            confidence_levels: List of confidence levels for VaR/CVaR calculations
        """
        self.returns = returns_data.copy()
        self.confidence_levels = confidence_levels
        self.risk_metrics = {}
        
        logger.info(f"RiskMetricsCalculator initialized with {len(returns_data)} observations, "
                   f"{len(returns_data.columns)} assets")
    
    def calculate_all_risk_metrics(self) -> Dict[str, any]:
        """
        Calculate all available risk metrics.
        
        Returns:
            Dictionary containing all risk metrics
        """
        logger.info("Calculating comprehensive risk metrics...")
        
        # Basic risk metrics
        self._calculate_basic_risk_metrics()
        
        # Value at Risk metrics
        self._calculate_var_metrics()
        
        # Drawdown analysis
        self._calculate_drawdown_metrics()
        
        # Correlation and dependence metrics
        self._calculate_correlation_metrics()
        
        # Higher moment risk metrics
        self._calculate_higher_moment_metrics()
        
        # Portfolio-specific metrics
        self._calculate_portfolio_metrics()
        
        # Tail risk metrics
        self._calculate_tail_risk_metrics()
        
        logger.info(f"Calculated {len(self.risk_metrics)} risk metric categories")
        return self.risk_metrics
    
    def _calculate_basic_risk_metrics(self):
        """Calculate basic risk metrics."""
        returns = self.returns
        
        basic_metrics = {
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'downside_volatility': self._calculate_downside_volatility(returns),
            'tracking_error': self._calculate_tracking_error(returns),
            'information_ratio': self._calculate_information_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns),
            'omega_ratio': self._calculate_omega_ratio(returns),
            'gain_loss_ratio': self._calculate_gain_loss_ratio(returns)
        }
        
        self.risk_metrics['basic'] = basic_metrics
    
    def _calculate_var_metrics(self):
        """Calculate Value at Risk metrics."""
        returns = self.returns
        var_metrics = {}
        
        for confidence in self.confidence_levels:
            alpha = 1 - confidence
            
            # Historical VaR
            var_hist = returns.quantile(alpha)
            
            # Parametric VaR (assuming normal distribution)
            var_param = returns.mean() + returns.std() * stats.norm.ppf(alpha)
            
            # Modified VaR (Cornish-Fisher expansion)
            var_modified = self._calculate_modified_var(returns, confidence)
            
            # Conditional VaR (Expected Shortfall)
            cvar = self._calculate_cvar(returns, confidence)
            
            var_metrics[f'var_historical_{confidence}'] = var_hist
            var_metrics[f'var_parametric_{confidence}'] = var_param
            var_metrics[f'var_modified_{confidence}'] = var_modified
            var_metrics[f'cvar_{confidence}'] = cvar
        
        self.risk_metrics['var'] = var_metrics
    
    def _calculate_drawdown_metrics(self):
        """Calculate drawdown analysis metrics."""
        returns = self.returns
        
        drawdown_metrics = {}
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            
            # Calculate cumulative returns
            cum_returns = (1 + asset_returns).cumprod()
            
            # Calculate running maximum
            running_max = cum_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max
            
            # Drawdown metrics
            max_drawdown = drawdown.min()
            avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
            
            # Drawdown duration
            drawdown_periods = self._calculate_drawdown_duration(drawdown)
            
            drawdown_metrics[col] = {
                'max_drawdown': max_drawdown,
                'average_drawdown': avg_drawdown,
                'max_drawdown_duration': drawdown_periods['max_duration'],
                'avg_drawdown_duration': drawdown_periods['avg_duration'],
                'recovery_time': drawdown_periods['recovery_time']
            }
        
        self.risk_metrics['drawdown'] = drawdown_metrics
    
    def _calculate_correlation_metrics(self):
        """Calculate correlation and dependence metrics."""
        returns = self.returns
        
        correlation_metrics = {
            'correlation_matrix': returns.corr(),
            'average_correlation': returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean(),
            'max_correlation': returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].max(),
            'min_correlation': returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].min(),
            'eigenvalues': np.linalg.eigvals(returns.corr()),
            'condition_number': np.linalg.cond(returns.corr())
        }
        
        # Rolling correlation analysis
        if len(returns) > 60:  # Need sufficient data
            rolling_corr = returns.rolling(window=60).corr()
            correlation_metrics['rolling_correlation_stability'] = rolling_corr.std().mean()
        
        self.risk_metrics['correlation'] = correlation_metrics
    
    def _calculate_higher_moment_metrics(self):
        """Calculate higher moment risk metrics."""
        returns = self.returns
        
        higher_moment_metrics = {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'jarque_bera_stat': {},
            'jarque_bera_pvalue': {}
        }
        
        # Jarque-Bera test for normality
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            if len(asset_returns) > 10:
                jb_stat, jb_pvalue = stats.jarque_bera(asset_returns)
                higher_moment_metrics['jarque_bera_stat'][col] = jb_stat
                higher_moment_metrics['jarque_bera_pvalue'][col] = jb_pvalue
        
        self.risk_metrics['higher_moments'] = higher_moment_metrics
    
    def _calculate_portfolio_metrics(self):
        """Calculate portfolio-specific risk metrics."""
        returns = self.returns
        
        # Equal weight portfolio
        equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_returns = (returns * equal_weights).sum(axis=1)
        
        portfolio_metrics = {
            'portfolio_volatility': portfolio_returns.std() * np.sqrt(252),
            'portfolio_var_95': portfolio_returns.quantile(0.05),
            'portfolio_cvar_95': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean(),
            'diversification_ratio': self._calculate_diversification_ratio(returns, equal_weights),
            'concentration_index': self._calculate_concentration_index(equal_weights),
            'effective_number_assets': self._calculate_effective_number_assets(equal_weights)
        }
        
        self.risk_metrics['portfolio'] = portfolio_metrics
    
    def _calculate_tail_risk_metrics(self):
        """Calculate tail risk metrics."""
        returns = self.returns
        
        tail_metrics = {}
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            
            # Tail ratio
            tail_ratio = self._calculate_tail_ratio(asset_returns)
            
            # Extreme value metrics
            extreme_metrics = self._calculate_extreme_value_metrics(asset_returns)
            
            tail_metrics[col] = {
                'tail_ratio': tail_ratio,
                'left_tail_mean': asset_returns[asset_returns <= asset_returns.quantile(0.05)].mean(),
                'right_tail_mean': asset_returns[asset_returns >= asset_returns.quantile(0.95)].mean(),
                **extreme_metrics
            }
        
        self.risk_metrics['tail_risk'] = tail_metrics
    
    # Helper methods for complex calculations
    def _calculate_downside_volatility(self, returns: pd.DataFrame, target_return: float = 0) -> pd.Series:
        """Calculate downside volatility."""
        downside_returns = returns[returns < target_return]
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_tracking_error(self, returns: pd.DataFrame, benchmark_col: Optional[str] = None) -> pd.Series:
        """Calculate tracking error relative to benchmark."""
        if benchmark_col is None:
            # Use equal weight portfolio as benchmark
            benchmark_returns = returns.mean(axis=1)
        else:
            benchmark_returns = returns[benchmark_col]
        
        tracking_errors = {}
        for col in returns.columns:
            if col != benchmark_col:
                excess_returns = returns[col] - benchmark_returns
                tracking_errors[col] = excess_returns.std() * np.sqrt(252)
        
        return pd.Series(tracking_errors)
    
    def _calculate_information_ratio(self, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate information ratio."""
        excess_returns = returns - risk_free_rate / 252
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    def _calculate_sortino_ratio(self, returns: pd.DataFrame, target_return: float = 0, risk_free_rate: float = 0.02) -> pd.Series:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_std = self._calculate_downside_volatility(returns, target_return)
        return (excess_returns.mean() * 252) / downside_std
    
    def _calculate_calmar_ratio(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate Calmar ratio."""
        annual_returns = returns.mean() * 252
        max_drawdowns = {}
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            cum_returns = (1 + asset_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdowns[col] = abs(drawdown.min())
        
        max_drawdowns = pd.Series(max_drawdowns)
        return annual_returns / max_drawdowns
    
    def _calculate_omega_ratio(self, returns: pd.DataFrame, threshold: float = 0) -> pd.Series:
        """Calculate Omega ratio."""
        omega_ratios = {}
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            gains = asset_returns[asset_returns > threshold] - threshold
            losses = threshold - asset_returns[asset_returns <= threshold]
            
            if losses.sum() > 0:
                omega_ratios[col] = gains.sum() / losses.sum()
            else:
                omega_ratios[col] = np.inf
        
        return pd.Series(omega_ratios)
    
    def _calculate_gain_loss_ratio(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate gain to loss ratio."""
        gain_loss_ratios = {}
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            gains = asset_returns[asset_returns > 0]
            losses = asset_returns[asset_returns < 0]
            
            if len(losses) > 0:
                avg_gain = gains.mean() if len(gains) > 0 else 0
                avg_loss = abs(losses.mean())
                gain_loss_ratios[col] = avg_gain / avg_loss
            else:
                gain_loss_ratios[col] = np.inf
        
        return pd.Series(gain_loss_ratios)
    
    def _calculate_modified_var(self, returns: pd.DataFrame, confidence: float) -> pd.Series:
        """Calculate Modified VaR using Cornish-Fisher expansion."""
        alpha = 1 - confidence
        z_alpha = stats.norm.ppf(alpha)
        
        modified_vars = {}
        
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            
            mean = asset_returns.mean()
            std = asset_returns.std()
            skew = asset_returns.skew()
            kurt = asset_returns.kurtosis()
            
            # Cornish-Fisher expansion
            cf_correction = (z_alpha**2 - 1) * skew / 6 + (z_alpha**3 - 3*z_alpha) * kurt / 24
            z_cf = z_alpha + cf_correction
            
            modified_vars[col] = mean + std * z_cf
        
        return pd.Series(modified_vars)
    
    def _calculate_cvar(self, returns: pd.DataFrame, confidence: float) -> pd.Series:
        """Calculate Conditional VaR (Expected Shortfall)."""
        alpha = 1 - confidence
        
        cvars = {}
        for col in returns.columns:
            asset_returns = returns[col].dropna()
            var_threshold = asset_returns.quantile(alpha)
            cvar = asset_returns[asset_returns <= var_threshold].mean()
            cvars[col] = cvar
        
        return pd.Series(cvars)
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict[str, float]:
        """Calculate drawdown duration metrics."""
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        
        start = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if start is not None:  # Still in drawdown at end
            drawdown_periods.append(len(drawdown) - start)
        
        if drawdown_periods:
            return {
                'max_duration': max(drawdown_periods),
                'avg_duration': np.mean(drawdown_periods),
                'recovery_time': drawdown_periods[-1] if in_drawdown.iloc[-1] else 0
            }
        else:
            return {'max_duration': 0, 'avg_duration': 0, 'recovery_time': 0}
    
    def _calculate_diversification_ratio(self, returns: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate diversification ratio."""
        weighted_avg_vol = np.sum(weights * returns.std() * np.sqrt(252))
        portfolio_vol = (returns * weights).sum(axis=1).std() * np.sqrt(252)
        return weighted_avg_vol / portfolio_vol
    
    def _calculate_concentration_index(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl concentration index."""
        return np.sum(weights**2)
    
    def _calculate_effective_number_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets."""
        return 1 / np.sum(weights**2)
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        return abs(returns.quantile(0.95) / returns.quantile(0.05))
    
    def _calculate_extreme_value_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate extreme value theory metrics."""
        # Simple extreme value metrics
        sorted_returns = returns.sort_values()
        n = len(sorted_returns)
        
        # Take top/bottom 5% as extreme values
        extreme_threshold = int(0.05 * n)
        
        left_extremes = sorted_returns[:extreme_threshold]
        right_extremes = sorted_returns[-extreme_threshold:]
        
        return {
            'extreme_left_mean': left_extremes.mean(),
            'extreme_right_mean': right_extremes.mean(),
            'extreme_left_std': left_extremes.std(),
            'extreme_right_std': right_extremes.std()
        }


def calculate_portfolio_risk_metrics(returns_data: pd.DataFrame, 
                                   weights: Optional[np.ndarray] = None) -> Dict[str, any]:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        returns_data: DataFrame with asset returns
        weights: Portfolio weights (equal weight if None)
        
    Returns:
        Dictionary with comprehensive risk metrics
    """
    logger.info("Calculating portfolio risk metrics...")
    
    if weights is None:
        weights = np.ones(len(returns_data.columns)) / len(returns_data.columns)
    
    # Initialize calculator
    risk_calc = RiskMetricsCalculator(returns_data)
    
    # Calculate all metrics
    risk_metrics = risk_calc.calculate_all_risk_metrics()
    
    # Add portfolio-specific calculations
    portfolio_returns = (returns_data * weights).sum(axis=1)
    
    portfolio_specific = {
        'portfolio_weights': weights,
        'portfolio_return_mean': portfolio_returns.mean() * 252,
        'portfolio_return_std': portfolio_returns.std() * np.sqrt(252),
        'portfolio_sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
        'portfolio_skewness': portfolio_returns.skew(),
        'portfolio_kurtosis': portfolio_returns.kurtosis()
    }
    
    risk_metrics['portfolio_specific'] = portfolio_specific
    
    logger.info("Portfolio risk metrics calculation completed")
    return risk_metrics
