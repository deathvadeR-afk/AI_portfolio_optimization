"""
Technical Indicators for Portfolio Optimization

This module provides comprehensive technical indicators and market features
for enhanced portfolio optimization. Includes momentum, volatility, trend,
and volume-based indicators optimized for financial time series.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Professional-grade technical indicators calculator for financial data.
    
    Provides 30+ technical indicators including momentum, volatility, trend,
    volume, and statistical measures optimized for portfolio optimization.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.
        
        Args:
            data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)
        
        # Validate required columns
        required_cols = ['Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"TechnicalIndicators initialized with {len(data)} rows, {len(data.columns)} columns")
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all available technical indicators.
        
        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Calculating all technical indicators...")
        
        # Price-based indicators
        self._calculate_price_indicators()
        
        # Momentum indicators
        self._calculate_momentum_indicators()
        
        # Volatility indicators
        self._calculate_volatility_indicators()
        
        # Trend indicators
        self._calculate_trend_indicators()
        
        # Volume indicators (if available)
        if 'Volume' in self.data.columns:
            self._calculate_volume_indicators()
        
        # Statistical indicators
        self._calculate_statistical_indicators()
        
        # Market microstructure indicators
        self._calculate_microstructure_indicators()
        
        logger.info(f"Calculated {len(self.features.columns)} technical indicators")
        return self.features.fillna(method='ffill').fillna(0)
    
    def _calculate_price_indicators(self):
        """Calculate price-based indicators."""
        close = self.data['Close']
        
        # Returns
        self.features['returns_1d'] = close.pct_change()
        self.features['returns_5d'] = close.pct_change(5)
        self.features['returns_20d'] = close.pct_change(20)
        
        # Log returns
        self.features['log_returns_1d'] = np.log(close / close.shift(1))
        
        # Price relative to moving averages
        for window in [5, 10, 20, 50]:
            ma = close.rolling(window).mean()
            self.features[f'price_to_ma_{window}'] = (close / ma) - 1
        
        # Price momentum
        self.features['price_momentum_5'] = (close / close.shift(5)) - 1
        self.features['price_momentum_20'] = (close / close.shift(20)) - 1
    
    def _calculate_momentum_indicators(self):
        """Calculate momentum indicators."""
        close = self.data['Close']
        high = self.data.get('High', close)
        low = self.data.get('Low', close)
        
        # RSI (Relative Strength Index)
        for period in [14, 30]:
            self.features[f'rsi_{period}'] = self._calculate_rsi(close, period)
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = self._calculate_macd(close)
        self.features['macd'] = macd
        self.features['macd_signal'] = macd_signal
        self.features['macd_histogram'] = macd_hist
        
        # Stochastic Oscillator
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
            self.features['stoch_k'] = stoch_k
            self.features['stoch_d'] = stoch_d
        
        # Williams %R
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            self.features['williams_r'] = self._calculate_williams_r(high, low, close)
        
        # Rate of Change (ROC)
        for period in [10, 20]:
            self.features[f'roc_{period}'] = ((close - close.shift(period)) / close.shift(period)) * 100
    
    def _calculate_volatility_indicators(self):
        """Calculate volatility indicators."""
        close = self.data['Close']
        high = self.data.get('High', close)
        low = self.data.get('Low', close)
        
        # Historical volatility
        returns = close.pct_change()
        for window in [10, 20, 30]:
            self.features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            self.features['atr_14'] = self._calculate_atr(high, low, close, 14)
        
        # Bollinger Bands
        for window in [20, 50]:
            bb_upper, bb_middle, bb_lower, bb_width, bb_position = self._calculate_bollinger_bands(close, window)
            self.features[f'bb_width_{window}'] = bb_width
            self.features[f'bb_position_{window}'] = bb_position
        
        # Volatility ratio
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(30).std()
        self.features['volatility_ratio'] = vol_short / vol_long
    
    def _calculate_trend_indicators(self):
        """Calculate trend indicators."""
        close = self.data['Close']
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            self.features[f'sma_{window}'] = close.rolling(window).mean()
            self.features[f'ema_{window}'] = close.ewm(span=window).mean()
        
        # Moving average crossovers
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        self.features['ma_crossover_5_20'] = (sma_5 > sma_20).astype(int)
        
        # Trend strength
        self.features['trend_strength'] = self._calculate_trend_strength(close)
        
        # ADX (Average Directional Index)
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            self.features['adx_14'] = self._calculate_adx(self.data['High'], self.data['Low'], close)
    
    def _calculate_volume_indicators(self):
        """Calculate volume-based indicators."""
        close = self.data['Close']
        volume = self.data['Volume']
        
        # Volume moving averages
        self.features['volume_sma_20'] = volume.rolling(20).mean()
        self.features['volume_ratio'] = volume / self.features['volume_sma_20']
        
        # On-Balance Volume (OBV)
        self.features['obv'] = self._calculate_obv(close, volume)
        
        # Volume Price Trend (VPT)
        self.features['vpt'] = self._calculate_vpt(close, volume)
        
        # Accumulation/Distribution Line
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            self.features['ad_line'] = self._calculate_ad_line(
                self.data['High'], self.data['Low'], close, volume
            )
    
    def _calculate_statistical_indicators(self):
        """Calculate statistical indicators."""
        close = self.data['Close']
        returns = close.pct_change()
        
        # Rolling statistics
        for window in [20, 50]:
            self.features[f'skewness_{window}'] = returns.rolling(window).skew()
            self.features[f'kurtosis_{window}'] = returns.rolling(window).kurt()
        
        # Z-score
        for window in [20, 50]:
            mean = close.rolling(window).mean()
            std = close.rolling(window).std()
            self.features[f'zscore_{window}'] = (close - mean) / std
        
        # Percentile rank
        for window in [20, 50]:
            self.features[f'percentile_rank_{window}'] = close.rolling(window).rank(pct=True)
    
    def _calculate_microstructure_indicators(self):
        """Calculate market microstructure indicators."""
        close = self.data['Close']
        
        # Price gaps
        if 'Open' in self.data.columns:
            open_price = self.data['Open']
            self.features['gap'] = (open_price - close.shift(1)) / close.shift(1)
            self.features['gap_up'] = (self.features['gap'] > 0.01).astype(int)
            self.features['gap_down'] = (self.features['gap'] < -0.01).astype(int)
        
        # Intraday range
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            high = self.data['High']
            low = self.data['Low']
            self.features['intraday_range'] = (high - low) / close
            self.features['high_low_ratio'] = high / low
        
        # Close position within range
        if 'High' in self.data.columns and 'Low' in self.data.columns:
            high = self.data['High']
            low = self.data['Low']
            self.features['close_position'] = (close - low) / (high - low)
    
    # Helper methods for complex indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_width = (upper_band - lower_band) / sma
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return upper_band, sma, lower_band, bb_width, bb_position
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def calculate_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * r_value ** 2  # Slope weighted by R-squared
        
        return prices.rolling(window=period).apply(calculate_slope, raw=True)
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        # Simplified ADX calculation
        tr = self._calculate_atr(high, low, close, 1)
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        di_plus = 100 * pd.Series(dm_plus, index=high.index).rolling(period).mean() / tr.rolling(period).mean()
        di_minus = 100 * pd.Series(dm_minus, index=high.index).rolling(period).mean() / tr.rolling(period).mean()
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        return adx
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = volume.copy()
        obv[close.diff() < 0] = -volume[close.diff() < 0]
        return obv.cumsum()
    
    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend."""
        return (volume * close.pct_change()).cumsum()
    
    def _calculate_ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        return ad_line


def calculate_features_for_portfolio(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate technical indicators for multiple assets.
    
    Args:
        data_dict: Dictionary mapping asset names to price DataFrames
        
    Returns:
        Dictionary mapping asset names to feature DataFrames
    """
    logger.info(f"Calculating features for {len(data_dict)} assets")
    
    features_dict = {}
    
    for asset_name, asset_data in data_dict.items():
        try:
            logger.debug(f"Processing features for {asset_name}")
            
            # Calculate technical indicators
            ti = TechnicalIndicators(asset_data)
            features = ti.calculate_all_indicators()
            
            # Add asset identifier
            features['asset'] = asset_name
            
            features_dict[asset_name] = features
            
            logger.debug(f"Calculated {len(features.columns)} features for {asset_name}")
            
        except Exception as e:
            logger.error(f"Failed to calculate features for {asset_name}: {str(e)}")
            # Create empty features DataFrame to maintain consistency
            features_dict[asset_name] = pd.DataFrame(index=asset_data.index)
    
    logger.info(f"Feature calculation completed for {len(features_dict)} assets")
    return features_dict
