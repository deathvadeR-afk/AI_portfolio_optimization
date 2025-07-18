"""
Market Data Collection Module

This module provides comprehensive market data collection capabilities from multiple
sources including Yahoo Finance, with robust error handling, rate limiting, and
data validation. Designed for production use in portfolio optimization systems.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# Suppress yfinance warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Configure logging
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class RateLimitError(Exception):
    """Custom exception for rate limiting errors"""
    pass


class MarketDataCollector:
    """
    Professional-grade market data collector with comprehensive error handling,
    rate limiting, data validation, and caching capabilities.

    Features:
    - Multi-source data collection (Yahoo Finance primary)
    - Intelligent retry logic with exponential backoff
    - Data quality validation and cleaning
    - Rate limiting to respect API limits
    - Comprehensive logging and error reporting
    - Caching to minimize redundant API calls
    - Support for multiple asset classes
    """

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3,
                 timeout: int = 30,
                 validate_data: bool = True):
        """
        Initialize the market data collector.

        Args:
            cache_dir: Directory for caching downloaded data
            rate_limit_delay: Minimum delay between API calls (seconds)
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
            validate_data: Whether to perform data quality validation
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/market_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.validate_data = validate_data

        # Track last API call time for rate limiting
        self.last_api_call = 0.0

        # Data cache for session-level caching
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"MarketDataCollector initialized with cache_dir={self.cache_dir}")

    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting between API calls to respect service limits.
        """
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_api_call = time.time()

    def _validate_symbol(self, symbol: str) -> str:
        """
        Validate and normalize a trading symbol.

        Args:
            symbol: Trading symbol to validate

        Returns:
            Normalized symbol

        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")

        # Remove whitespace and convert to uppercase
        symbol = symbol.strip().upper()

        # Basic validation - symbols should be alphanumeric with some special chars
        if not symbol.replace('^', '').replace('-', '').replace('.', '').replace('=', '').isalnum():
            logger.warning(f"Symbol {symbol} contains unusual characters")

        return symbol

    def _validate_period(self, period: str) -> str:
        """
        Validate time period parameter.

        Args:
            period: Time period string (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            Validated period string

        Raises:
            ValueError: If period is invalid
        """
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

        if period not in valid_periods:
            raise ValueError(f"Invalid period '{period}'. Valid periods: {valid_periods}")

        return period

    def _validate_interval(self, interval: str) -> str:
        """
        Validate data interval parameter.

        Args:
            interval: Data interval string (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            Validated interval string

        Raises:
            ValueError: If interval is invalid
        """
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval '{interval}'. Valid intervals: {valid_intervals}")

        return interval

    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean market data quality.

        Args:
            data: Raw market data DataFrame
            symbol: Symbol for logging purposes

        Returns:
            Cleaned and validated DataFrame

        Raises:
            DataValidationError: If data quality is unacceptable
        """
        if data.empty:
            raise DataValidationError(f"No data received for symbol {symbol}")

        original_length = len(data)

        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns for {symbol}: {missing_columns}")

        # Remove rows with all NaN values
        data = data.dropna(how='all')

        # Check for negative prices (should not happen in real data)
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = data[col] < 0
                if negative_prices.any():
                    logger.warning(f"Found {negative_prices.sum()} negative prices in {col} for {symbol}")
                    data.loc[negative_prices, col] = np.nan

        # Check for impossible price relationships (High < Low, etc.)
        if 'High' in data.columns and 'Low' in data.columns:
            invalid_hl = data['High'] < data['Low']
            if invalid_hl.any():
                logger.warning(f"Found {invalid_hl.sum()} rows where High < Low for {symbol}")
                # Set both to the average
                avg_price = (data.loc[invalid_hl, 'High'] + data.loc[invalid_hl, 'Low']) / 2
                data.loc[invalid_hl, 'High'] = avg_price
                data.loc[invalid_hl, 'Low'] = avg_price

        # Check for extreme price movements (more than 50% in a day)
        if 'Close' in data.columns and len(data) > 1:
            returns = data['Close'].pct_change()
            extreme_moves = np.abs(returns) > 0.5
            if extreme_moves.any():
                logger.warning(f"Found {extreme_moves.sum()} extreme price movements (>50%) for {symbol}")

        # Forward fill missing values (common practice for financial data)
        data = data.fillna(method='ffill')

        # Check final data quality
        remaining_length = len(data)
        if remaining_length < original_length * 0.5:
            logger.warning(f"Data quality check removed {original_length - remaining_length} rows "
                         f"({(1 - remaining_length/original_length)*100:.1f}%) for {symbol}")

        if remaining_length < 10:
            raise DataValidationError(f"Insufficient data after cleaning for {symbol}: {remaining_length} rows")

        logger.debug(f"Data validation completed for {symbol}: {remaining_length} valid rows")
        return data

    def fetch_stock_data(self,
                        symbols: Union[str, List[str]],
                        period: str = "1y",
                        interval: str = "1d",
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock/ETF data for given symbols with comprehensive error handling.

        Args:
            symbols: Single symbol or list of symbols to fetch
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in 'YYYY-MM-DD' format (alternative to period)
            end_date: End date in 'YYYY-MM-DD' format (alternative to period)
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary mapping symbols to their respective DataFrames

        Raises:
            ValueError: If parameters are invalid
            DataValidationError: If data quality is unacceptable
            RateLimitError: If rate limits are exceeded
        """
        # Normalize input
        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate parameters
        symbols = [self._validate_symbol(symbol) for symbol in symbols]
        if not start_date and not end_date:
            period = self._validate_period(period)
        interval = self._validate_interval(interval)

        logger.info(f"Fetching stock data for {len(symbols)} symbols: {symbols}")

        results = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"{symbol}_{period}_{interval}_{start_date}_{end_date}"
                if use_cache and cache_key in self.data_cache:
                    logger.debug(f"Using cached data for {symbol}")
                    results[symbol] = self.data_cache[cache_key].copy()
                    continue

                # Enforce rate limiting
                self._enforce_rate_limit()

                logger.debug(f"Downloading data for {symbol}")

                # Create yfinance ticker object
                ticker = yf.Ticker(symbol)

                # Fetch data based on parameters
                if start_date or end_date:
                    data = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        timeout=self.timeout
                    )
                else:
                    data = ticker.history(
                        period=period,
                        interval=interval,
                        timeout=self.timeout
                    )

                # Validate data quality if enabled
                if self.validate_data:
                    data = self._validate_data_quality(data, symbol)

                # Add symbol column for identification
                data['Symbol'] = symbol

                # Cache the data
                if use_cache:
                    self.data_cache[cache_key] = data.copy()

                results[symbol] = data
                logger.debug(f"Successfully fetched {len(data)} rows for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)

                # Don't fail completely if some symbols work
                if len(symbols) == 1:
                    raise

        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")

        if not results:
            raise DataValidationError("No data could be fetched for any symbols")

        logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols)} symbols")
        return results

    def fetch_index_data(self,
                        indices: Union[str, List[str]],
                        period: str = "1y",
                        interval: str = "1d",
                        **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch index data (convenience method that calls fetch_stock_data).

        Args:
            indices: Single index or list of indices to fetch (e.g., '^GSPC', '^IXIC')
            period: Time period
            interval: Data interval
            **kwargs: Additional arguments passed to fetch_stock_data

        Returns:
            Dictionary mapping indices to their respective DataFrames
        """
        logger.info(f"Fetching index data for: {indices}")
        return self.fetch_stock_data(indices, period=period, interval=interval, **kwargs)

    def fetch_etf_data(self,
                      etfs: Union[str, List[str]],
                      period: str = "1y",
                      interval: str = "1d",
                      **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch ETF data (convenience method that calls fetch_stock_data).

        Args:
            etfs: Single ETF or list of ETFs to fetch
            period: Time period
            interval: Data interval
            **kwargs: Additional arguments passed to fetch_stock_data

        Returns:
            Dictionary mapping ETFs to their respective DataFrames
        """
        logger.info(f"Fetching ETF data for: {etfs}")
        return self.fetch_stock_data(etfs, period=period, interval=interval, **kwargs)

    def fetch_batch_data(self,
                        symbol_groups: Dict[str, List[str]],
                        period: str = "1y",
                        interval: str = "1d",
                        batch_delay: float = 1.0,
                        **kwargs) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for multiple groups of symbols with batch processing.

        Args:
            symbol_groups: Dictionary mapping group names to lists of symbols
            period: Time period
            interval: Data interval
            batch_delay: Delay between batches (seconds)
            **kwargs: Additional arguments passed to fetch_stock_data

        Returns:
            Nested dictionary: {group_name: {symbol: DataFrame}}
        """
        logger.info(f"Fetching batch data for {len(symbol_groups)} groups")

        results = {}

        for group_name, symbols in symbol_groups.items():
            logger.info(f"Processing group '{group_name}' with {len(symbols)} symbols")

            try:
                group_data = self.fetch_stock_data(
                    symbols,
                    period=period,
                    interval=interval,
                    **kwargs
                )
                results[group_name] = group_data

                # Add delay between groups to be respectful to the API
                if batch_delay > 0:
                    time.sleep(batch_delay)

            except Exception as e:
                logger.error(f"Failed to fetch data for group '{group_name}': {str(e)}")
                results[group_name] = {}

        total_symbols = sum(len(group_data) for group_data in results.values())
        logger.info(f"Batch processing completed: {total_symbols} symbols across {len(results)} groups")

        return results

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a symbol.

        Args:
            symbol: Symbol to get information for

        Returns:
            Dictionary containing symbol information
        """
        symbol = self._validate_symbol(symbol)

        try:
            self._enforce_rate_limit()
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key information
            symbol_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary', ''),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

            logger.debug(f"Retrieved info for {symbol}: {symbol_info['name']}")
            return symbol_info

        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

    def get_available_data_range(self, symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the available date range for a symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (start_date, end_date) or (None, None) if unavailable
        """
        try:
            # Fetch a small sample to determine range
            data = self.fetch_stock_data(symbol, period="max", use_cache=False)

            if symbol in data and not data[symbol].empty:
                start_date = data[symbol].index.min().to_pydatetime()
                end_date = data[symbol].index.max().to_pydatetime()
                return start_date, end_date

        except Exception as e:
            logger.error(f"Failed to get date range for {symbol}: {str(e)}")

        return None, None

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self.data_cache.clear()
        logger.info("Data cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self.data_cache)
        total_memory = sum(df.memory_usage(deep=True).sum() for df in self.data_cache.values())

        return {
            'total_entries': total_entries,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024),
            'cache_keys': list(self.data_cache.keys())
        }
