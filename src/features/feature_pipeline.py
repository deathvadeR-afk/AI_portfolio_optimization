"""
Feature Engineering Pipeline for Portfolio Optimization

This module provides a comprehensive feature engineering pipeline that
combines technical indicators, risk metrics, market sentiment, and
macroeconomic features for enhanced portfolio optimization.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from features.technical_indicators import TechnicalIndicators, calculate_features_for_portfolio
from features.risk_metrics import RiskMetricsCalculator, calculate_portfolio_risk_metrics
from data.storage import DataStorage

# Configure logging
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for portfolio optimization.
    
    Combines multiple feature types:
    - Technical indicators (30+ indicators)
    - Risk metrics (VaR, drawdown, correlation)
    - Market microstructure features
    - Cross-asset features
    - Macroeconomic indicators
    """
    
    def __init__(self, 
                 lookback_window: int = 60,
                 feature_selection: bool = True,
                 normalize_features: bool = True,
                 handle_missing: str = "forward_fill"):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            lookback_window: Number of periods for rolling calculations
            feature_selection: Whether to perform feature selection
            normalize_features: Whether to normalize features
            handle_missing: How to handle missing values ('forward_fill', 'drop', 'interpolate')
        """
        self.lookback_window = lookback_window
        self.feature_selection = feature_selection
        self.normalize_features = normalize_features
        self.handle_missing = handle_missing
        
        # Feature storage
        self.features = {}
        self.feature_importance = {}
        self.feature_stats = {}
        
        logger.info(f"FeatureEngineeringPipeline initialized with lookback_window={lookback_window}")
    
    def process_portfolio_data(self, 
                             data_path: str,
                             assets: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process portfolio data and generate comprehensive features.
        
        Args:
            data_path: Path to data storage
            assets: List of assets to process (None for all)
            
        Returns:
            Dictionary with processed features for each asset
        """
        logger.info("Starting comprehensive feature engineering pipeline...")
        
        # Load data
        data_dict = self._load_portfolio_data(data_path, assets)
        
        # Generate features for each asset
        asset_features = {}
        
        for asset_name, asset_data in data_dict.items():
            logger.info(f"Processing features for {asset_name}")
            
            # Technical indicators
            technical_features = self._generate_technical_features(asset_data)
            
            # Risk metrics
            risk_features = self._generate_risk_features(asset_data)
            
            # Market microstructure features
            microstructure_features = self._generate_microstructure_features(asset_data)
            
            # Time-based features
            time_features = self._generate_time_features(asset_data)
            
            # Combine all features
            combined_features = pd.concat([
                technical_features,
                risk_features,
                microstructure_features,
                time_features
            ], axis=1)
            
            # Handle missing values
            combined_features = self._handle_missing_values(combined_features)
            
            # Normalize features if requested
            if self.normalize_features:
                combined_features = self._normalize_features(combined_features)
            
            asset_features[asset_name] = combined_features
            
            logger.info(f"Generated {len(combined_features.columns)} features for {asset_name}")
        
        # Generate cross-asset features
        cross_asset_features = self._generate_cross_asset_features(data_dict)
        
        # Add cross-asset features to each asset
        for asset_name in asset_features:
            asset_features[asset_name] = pd.concat([
                asset_features[asset_name],
                cross_asset_features
            ], axis=1)
        
        # Feature selection if requested
        if self.feature_selection:
            asset_features = self._perform_feature_selection(asset_features, data_dict)
        
        # Store features
        self.features = asset_features
        
        # Calculate feature statistics
        self._calculate_feature_statistics()
        
        logger.info(f"Feature engineering completed for {len(asset_features)} assets")
        return asset_features
    
    def _load_portfolio_data(self, data_path: str, assets: Optional[List[str]]) -> Dict[str, pd.DataFrame]:
        """Load portfolio data from storage."""
        logger.info(f"Loading portfolio data from {data_path}")
        
        storage = DataStorage(data_path)
        files = storage.list_files(data_type="raw")
        
        data_dict = {}
        
        for file_info in files:
            filename = file_info['filename']
            
            # Extract asset name
            if 'equity_index_' in filename:
                asset_name = filename.replace('equity_index_', '').replace('.parquet', '')
            elif 'fixed_income_' in filename:
                asset_name = filename.replace('fixed_income_', '').replace('.parquet', '')
            elif 'commodity_' in filename:
                asset_name = filename.replace('commodity_', '').replace('.parquet', '')
            elif 'alternative_' in filename:
                asset_name = filename.replace('alternative_', '').replace('.parquet', '')
            else:
                continue
            
            # Filter assets if specified
            if assets and asset_name not in assets:
                continue
            
            try:
                asset_data = storage.load_data(filename.replace('.parquet', ''), data_type="raw")
                
                # Ensure we have required columns
                if 'Close' in asset_data.columns:
                    data_dict[asset_name] = asset_data
                    logger.debug(f"Loaded {asset_name}: {len(asset_data)} rows")
                
            except Exception as e:
                logger.warning(f"Failed to load {asset_name}: {str(e)}")
        
        logger.info(f"Loaded data for {len(data_dict)} assets")
        return data_dict
    
    def _generate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        try:
            ti = TechnicalIndicators(data)
            features = ti.calculate_all_indicators()
            
            # Add prefix to avoid naming conflicts
            features.columns = [f"tech_{col}" for col in features.columns]
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to generate technical features: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _generate_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate risk-based features."""
        try:
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < self.lookback_window:
                logger.warning("Insufficient data for risk features")
                return pd.DataFrame(index=data.index)
            
            # Rolling risk metrics
            risk_features = pd.DataFrame(index=data.index)
            
            # Rolling volatility
            for window in [10, 20, 60]:
                risk_features[f'risk_volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
            
            # Rolling VaR
            for confidence in [0.95, 0.99]:
                for window in [20, 60]:
                    risk_features[f'risk_var_{confidence}_{window}d'] = returns.rolling(window).quantile(1 - confidence)
            
            # Rolling Sharpe ratio
            for window in [20, 60]:
                rolling_mean = returns.rolling(window).mean() * 252
                rolling_std = returns.rolling(window).std() * np.sqrt(252)
                risk_features[f'risk_sharpe_{window}d'] = rolling_mean / rolling_std
            
            # Rolling maximum drawdown
            cum_returns = (1 + returns).cumprod()
            for window in [20, 60]:
                rolling_max = cum_returns.rolling(window).max()
                rolling_dd = (cum_returns - rolling_max) / rolling_max
                risk_features[f'risk_max_dd_{window}d'] = rolling_dd.rolling(window).min()
            
            # Skewness and kurtosis
            for window in [20, 60]:
                risk_features[f'risk_skewness_{window}d'] = returns.rolling(window).skew()
                risk_features[f'risk_kurtosis_{window}d'] = returns.rolling(window).kurt()
            
            return risk_features
            
        except Exception as e:
            logger.error(f"Failed to generate risk features: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _generate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate market microstructure features."""
        try:
            micro_features = pd.DataFrame(index=data.index)
            
            close = data['Close']
            
            # Price-based microstructure features
            micro_features['micro_price_change'] = close.pct_change()
            micro_features['micro_price_acceleration'] = close.pct_change().diff()
            
            # Volatility clustering
            returns = close.pct_change()
            micro_features['micro_vol_clustering'] = returns.abs().rolling(5).mean()
            
            # Price momentum at different horizons
            for horizon in [1, 3, 5, 10]:
                micro_features[f'micro_momentum_{horizon}d'] = (close / close.shift(horizon)) - 1
            
            # Price reversal indicators
            micro_features['micro_reversal_1d'] = -returns.shift(1)
            micro_features['micro_reversal_3d'] = -(close.pct_change(3).shift(1))
            
            # Volume features (if available)
            if 'Volume' in data.columns:
                volume = data['Volume']
                
                # Volume-price relationship
                micro_features['micro_volume_price_corr'] = returns.rolling(20).corr(volume.pct_change())
                
                # Volume momentum
                micro_features['micro_volume_momentum'] = volume.pct_change()
                
                # Volume relative to average
                micro_features['micro_volume_ratio'] = volume / volume.rolling(20).mean()
            
            # High-low spread (if available)
            if 'High' in data.columns and 'Low' in data.columns:
                high = data['High']
                low = data['Low']
                
                micro_features['micro_hl_spread'] = (high - low) / close
                micro_features['micro_close_position'] = (close - low) / (high - low)
            
            return micro_features
            
        except Exception as e:
            logger.error(f"Failed to generate microstructure features: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _generate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        try:
            time_features = pd.DataFrame(index=data.index)
            
            # Extract time components
            time_features['time_day_of_week'] = data.index.dayofweek
            time_features['time_day_of_month'] = data.index.day
            time_features['time_month'] = data.index.month
            time_features['time_quarter'] = data.index.quarter
            
            # Cyclical encoding for time features
            time_features['time_day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            time_features['time_day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            time_features['time_month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            time_features['time_month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
            
            # Market regime indicators
            returns = data['Close'].pct_change()
            
            # High/low volatility regime
            vol_20d = returns.rolling(20).std()
            vol_60d = returns.rolling(60).std()
            time_features['time_high_vol_regime'] = (vol_20d > vol_60d * 1.5).astype(int)
            
            # Bull/bear market regime (simple trend following)
            ma_20 = data['Close'].rolling(20).mean()
            ma_60 = data['Close'].rolling(60).mean()
            time_features['time_bull_regime'] = (ma_20 > ma_60).astype(int)
            
            return time_features
            
        except Exception as e:
            logger.error(f"Failed to generate time features: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _generate_cross_asset_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate cross-asset features."""
        try:
            # Get common index
            common_index = None
            for asset_data in data_dict.values():
                if common_index is None:
                    common_index = asset_data.index
                else:
                    common_index = common_index.intersection(asset_data.index)
            
            if len(common_index) == 0:
                logger.warning("No common dates found for cross-asset features")
                return pd.DataFrame()
            
            cross_features = pd.DataFrame(index=common_index)
            
            # Collect close prices
            close_prices = pd.DataFrame()
            for asset_name, asset_data in data_dict.items():
                close_prices[asset_name] = asset_data.loc[common_index, 'Close']
            
            # Calculate returns
            returns = close_prices.pct_change()
            
            # Cross-asset correlation features
            rolling_corr = returns.rolling(20).corr()
            
            # Average correlation
            cross_features['cross_avg_correlation'] = self._calculate_avg_correlation(rolling_corr)
            
            # Market dispersion
            cross_features['cross_dispersion'] = returns.std(axis=1)
            
            # Market momentum
            cross_features['cross_momentum'] = returns.mean(axis=1)
            
            # Relative strength
            for asset_name in close_prices.columns:
                asset_returns = returns[asset_name]
                market_returns = returns.mean(axis=1)
                cross_features[f'cross_relative_strength_{asset_name}'] = (asset_returns - market_returns).rolling(10).mean()
            
            return cross_features
            
        except Exception as e:
            logger.error(f"Failed to generate cross-asset features: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_avg_correlation(self, rolling_corr: pd.DataFrame) -> pd.Series:
        """Calculate average correlation from rolling correlation matrix."""
        try:
            avg_corr = []
            
            for date in rolling_corr.index.get_level_values(0).unique():
                corr_matrix = rolling_corr.loc[date]
                
                if isinstance(corr_matrix, pd.Series):
                    avg_corr.append(np.nan)
                else:
                    # Get upper triangle values (excluding diagonal)
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    corr_values = corr_matrix.values[mask]
                    avg_corr.append(np.nanmean(corr_values))
            
            return pd.Series(avg_corr, index=rolling_corr.index.get_level_values(0).unique())
            
        except Exception as e:
            logger.error(f"Failed to calculate average correlation: {str(e)}")
            return pd.Series()
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        if self.handle_missing == "forward_fill":
            return features.fillna(method='ffill').fillna(0)
        elif self.handle_missing == "drop":
            return features.dropna()
        elif self.handle_missing == "interpolate":
            return features.interpolate().fillna(0)
        else:
            return features.fillna(0)
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using rolling z-score."""
        try:
            normalized_features = features.copy()
            
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    rolling_mean = features[col].rolling(self.lookback_window, min_periods=10).mean()
                    rolling_std = features[col].rolling(self.lookback_window, min_periods=10).std()
                    
                    normalized_features[col] = (features[col] - rolling_mean) / (rolling_std + 1e-8)
            
            return normalized_features.fillna(0)
            
        except Exception as e:
            logger.error(f"Failed to normalize features: {str(e)}")
            return features
    
    def _perform_feature_selection(self, 
                                 asset_features: Dict[str, pd.DataFrame], 
                                 data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Perform feature selection based on importance."""
        try:
            logger.info("Performing feature selection...")
            
            # Simple correlation-based feature selection
            selected_features = {}
            
            for asset_name, features in asset_features.items():
                # Calculate correlation with future returns
                if asset_name in data_dict:
                    future_returns = data_dict[asset_name]['Close'].pct_change().shift(-1)
                    
                    # Calculate feature importance (correlation with future returns)
                    feature_importance = {}
                    for col in features.columns:
                        if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            corr = features[col].corr(future_returns)
                            feature_importance[col] = abs(corr) if not np.isnan(corr) else 0
                    
                    # Select top features
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    selected_feature_names = [name for name, importance in top_features[:50]]  # Top 50 features
                    
                    selected_features[asset_name] = features[selected_feature_names]
                    self.feature_importance[asset_name] = dict(top_features[:50])
                else:
                    selected_features[asset_name] = features
            
            logger.info("Feature selection completed")
            return selected_features
            
        except Exception as e:
            logger.error(f"Failed to perform feature selection: {str(e)}")
            return asset_features
    
    def _calculate_feature_statistics(self):
        """Calculate comprehensive feature statistics."""
        try:
            self.feature_stats = {}
            
            for asset_name, features in self.features.items():
                asset_stats = {}
                
                for col in features.columns:
                    if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        asset_stats[col] = {
                            'mean': features[col].mean(),
                            'std': features[col].std(),
                            'min': features[col].min(),
                            'max': features[col].max(),
                            'null_pct': features[col].isnull().sum() / len(features) * 100
                        }
                
                self.feature_stats[asset_name] = asset_stats
            
            logger.info("Feature statistics calculated")
            
        except Exception as e:
            logger.error(f"Failed to calculate feature statistics: {str(e)}")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering summary."""
        return {
            'total_assets': len(self.features),
            'features_per_asset': {asset: len(features.columns) for asset, features in self.features.items()},
            'feature_importance': self.feature_importance,
            'feature_statistics': self.feature_stats,
            'pipeline_config': {
                'lookback_window': self.lookback_window,
                'feature_selection': self.feature_selection,
                'normalize_features': self.normalize_features,
                'handle_missing': self.handle_missing
            }
        }
