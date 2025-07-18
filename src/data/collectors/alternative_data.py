"""
Alternative Data Sources Collection Module

This module provides collection of alternative asset data including cryptocurrencies,
commodities, and other non-traditional assets. It integrates with multiple data
sources and provides fallback mechanisms for robust data collection.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import pandas as pd
import numpy as np
import logging
import time
import requests
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yfinance as yf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logger = logging.getLogger(__name__)


class AlternativeDataError(Exception):
    """Custom exception for alternative data collection errors"""
    pass


class AlternativeDataCollector:
    """
    Professional-grade alternative data collector supporting multiple asset classes
    and data sources with comprehensive error handling and fallback mechanisms.

    Features:
    - Cryptocurrency data collection
    - Commodity data collection
    - Real estate and REIT data
    - Multiple data source integration
    - Intelligent fallback mechanisms
    - Data validation and quality checks
    - Rate limiting and caching
    """

    # Cryptocurrency symbols mapping
    CRYPTO_SYMBOLS = {
        'bitcoin': 'BTC-USD',
        'ethereum': 'ETH-USD',
        'binancecoin': 'BNB-USD',
        'cardano': 'ADA-USD',
        'solana': 'SOL-USD',
        'polkadot': 'DOT-USD',
        'dogecoin': 'DOGE-USD',
        'avalanche': 'AVAX-USD',
        'polygon': 'MATIC-USD',
        'chainlink': 'LINK-USD',
        'litecoin': 'LTC-USD',
        'bitcoin-cash': 'BCH-USD',
        'stellar': 'XLM-USD',
        'vechain': 'VET-USD',
        'filecoin': 'FIL-USD'
    }

    # Commodity ETF symbols
    COMMODITY_ETFS = {
        'gold': 'GLD',                    # SPDR Gold Trust
        'silver': 'SLV',                  # iShares Silver Trust
        'oil': 'USO',                     # United States Oil Fund
        'natural_gas': 'UNG',             # United States Natural Gas Fund
        'copper': 'CPER',                 # United States Copper Index Fund
        'agriculture': 'DBA',             # Invesco DB Agriculture Fund
        'livestock': 'COW',               # iPath Series B Bloomberg Livestock
        'energy': 'DBE',                  # Invesco DB Energy Fund
        'precious_metals': 'DBP',         # Invesco DB Precious Metals Fund
        'base_metals': 'DBB',             # Invesco DB Base Metals Fund
        'commodities_broad': 'DBC',       # Invesco DB Commodity Index
        'palladium': 'PALL',              # Aberdeen Standard Physical Palladium
        'platinum': 'PPLT',               # Aberdeen Standard Physical Platinum
        'uranium': 'URA',                 # Global X Uranium ETF
        'timber': 'WOOD',                 # iShares Global Timber & Forestry ETF
    }

    # Real Estate and REIT symbols
    REAL_ESTATE_ETFS = {
        'us_reits': 'VNQ',                # Vanguard Real Estate ETF
        'global_reits': 'VNQI',           # Vanguard Global ex-US Real Estate ETF
        'residential_reits': 'REZ',       # iShares Residential Real Estate ETF
        'commercial_reits': 'FREL',       # Fidelity MSCI Real Estate ETF
        'mortgage_reits': 'REM',          # iShares Mortgage Real Estate ETF
        'infrastructure': 'IFRA',        # iShares Infrastructure ETF
        'real_estate_dev': 'HOMZ',       # Hoya Capital Housing ETF
    }

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3,
                 timeout: int = 30,
                 validate_data: bool = True):
        """
        Initialize the alternative data collector.

        Args:
            cache_dir: Directory for caching downloaded data
            rate_limit_delay: Minimum delay between API calls (seconds)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            validate_data: Whether to perform data quality validation
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/alternative_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.validate_data = validate_data

        # Track last API call time for rate limiting
        self.last_api_call = 0.0

        # Data cache for session-level caching
        self.data_cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"AlternativeDataCollector initialized with cache_dir={self.cache_dir}")

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_api_call = time.time()

    def _validate_crypto_symbol(self, symbol: str) -> str:
        """
        Validate and normalize cryptocurrency symbol.

        Args:
            symbol: Cryptocurrency symbol

        Returns:
            Normalized symbol for Yahoo Finance
        """
        symbol = symbol.lower().strip()

        # Check if it's a known crypto name
        if symbol in self.CRYPTO_SYMBOLS:
            return self.CRYPTO_SYMBOLS[symbol]

        # If it's already in Yahoo Finance format (XXX-USD), validate it
        if symbol.upper().endswith('-USD'):
            return symbol.upper()

        # Try to append -USD
        yahoo_symbol = f"{symbol.upper()}-USD"
        logger.debug(f"Converted crypto symbol {symbol} to {yahoo_symbol}")
        return yahoo_symbol

    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean alternative asset data quality.

        Args:
            data: Raw data DataFrame
            symbol: Symbol for logging purposes

        Returns:
            Cleaned and validated DataFrame
        """
        if data.empty:
            raise AlternativeDataError(f"No data received for symbol {symbol}")

        original_length = len(data)

        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing columns for {symbol}: {missing_columns}")

        # Remove rows with all NaN values
        data = data.dropna(how='all')

        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = data[col] < 0
                if negative_prices.any():
                    logger.warning(f"Found {negative_prices.sum()} negative prices in {col} for {symbol}")
                    data.loc[negative_prices, col] = np.nan

        # Forward fill missing values
        data = data.fillna(method='ffill')

        # Check final data quality
        remaining_length = len(data)
        if remaining_length < original_length * 0.5:
            logger.warning(f"Data quality check removed {original_length - remaining_length} rows "
                         f"({(1 - remaining_length/original_length)*100:.1f}%) for {symbol}")

        if remaining_length < 10:
            raise AlternativeDataError(f"Insufficient data after cleaning for {symbol}: {remaining_length} rows")

        logger.debug(f"Data validation completed for {symbol}: {remaining_length} valid rows")
        return data

    def fetch_crypto_data(self,
                         symbols: Union[str, List[str]],
                         period: str = "1y",
                         interval: str = "1d",
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch cryptocurrency data using Yahoo Finance.

        Args:
            symbols: Single symbol or list of cryptocurrency symbols
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        # Normalize input
        if isinstance(symbols, str):
            symbols = [symbols]

        # Validate and normalize crypto symbols
        normalized_symbols = []
        for symbol in symbols:
            try:
                normalized_symbol = self._validate_crypto_symbol(symbol)
                normalized_symbols.append(normalized_symbol)
            except Exception as e:
                logger.warning(f"Failed to normalize crypto symbol {symbol}: {str(e)}")

        if not normalized_symbols:
            raise AlternativeDataError("No valid cryptocurrency symbols provided")

        logger.info(f"Fetching cryptocurrency data for {len(normalized_symbols)} symbols")

        results = {}
        failed_symbols = []

        for symbol in normalized_symbols:
            try:
                # Check cache first
                cache_key = f"crypto_{symbol}_{period}_{interval}_{start_date}_{end_date}"
                if use_cache and cache_key in self.data_cache:
                    logger.debug(f"Using cached crypto data for {symbol}")
                    results[symbol] = self.data_cache[cache_key].copy()
                    continue

                # Enforce rate limiting
                self._enforce_rate_limit()

                logger.debug(f"Downloading crypto data for {symbol}")

                # Create yfinance ticker object
                ticker = yf.Ticker(symbol)

                # Fetch data
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

                # Add metadata
                data['Symbol'] = symbol
                data['AssetClass'] = 'Cryptocurrency'

                # Cache the data
                if use_cache:
                    self.data_cache[cache_key] = data.copy()

                results[symbol] = data
                logger.debug(f"Successfully fetched {len(data)} rows for crypto {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch crypto data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to fetch crypto data for: {failed_symbols}")

        if not results:
            raise AlternativeDataError("No cryptocurrency data could be fetched")

        logger.info(f"Successfully fetched crypto data for {len(results)} out of {len(normalized_symbols)} symbols")
        return results

    def fetch_commodity_data(self,
                           symbols: Union[str, List[str]],
                           period: str = "1y",
                           interval: str = "1d",
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch commodity data using ETF proxies.

        Args:
            symbols: Single symbol or list of commodity symbols/names
            period: Time period
            interval: Data interval
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        # Normalize input
        if isinstance(symbols, str):
            symbols = [symbols]

        # Map commodity names to ETF symbols
        normalized_symbols = []
        for symbol in symbols:
            symbol_lower = symbol.lower().strip()

            # Check if it's a known commodity name
            if symbol_lower in self.COMMODITY_ETFS:
                etf_symbol = self.COMMODITY_ETFS[symbol_lower]
                normalized_symbols.append(etf_symbol)
                logger.debug(f"Mapped commodity {symbol} to ETF {etf_symbol}")
            else:
                # Assume it's already an ETF symbol
                normalized_symbols.append(symbol.upper())

        logger.info(f"Fetching commodity data for {len(normalized_symbols)} symbols")

        results = {}
        failed_symbols = []

        for symbol in normalized_symbols:
            try:
                # Check cache first
                cache_key = f"commodity_{symbol}_{period}_{interval}_{start_date}_{end_date}"
                if use_cache and cache_key in self.data_cache:
                    logger.debug(f"Using cached commodity data for {symbol}")
                    results[symbol] = self.data_cache[cache_key].copy()
                    continue

                # Enforce rate limiting
                self._enforce_rate_limit()

                logger.debug(f"Downloading commodity data for {symbol}")

                # Create yfinance ticker object
                ticker = yf.Ticker(symbol)

                # Fetch data
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

                # Add metadata
                data['Symbol'] = symbol
                data['AssetClass'] = 'Commodity'

                # Cache the data
                if use_cache:
                    self.data_cache[cache_key] = data.copy()

                results[symbol] = data
                logger.debug(f"Successfully fetched {len(data)} rows for commodity {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch commodity data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to fetch commodity data for: {failed_symbols}")

        if not results:
            raise AlternativeDataError("No commodity data could be fetched")

        logger.info(f"Successfully fetched commodity data for {len(results)} out of {len(normalized_symbols)} symbols")
        return results

    def fetch_real_estate_data(self,
                             symbols: Union[str, List[str]],
                             period: str = "1y",
                             interval: str = "1d",
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch real estate and REIT data.

        Args:
            symbols: Single symbol or list of real estate symbols/names
            period: Time period
            interval: Data interval
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        # Normalize input
        if isinstance(symbols, str):
            symbols = [symbols]

        # Map real estate names to ETF symbols
        normalized_symbols = []
        for symbol in symbols:
            symbol_lower = symbol.lower().strip()

            # Check if it's a known real estate name
            if symbol_lower in self.REAL_ESTATE_ETFS:
                etf_symbol = self.REAL_ESTATE_ETFS[symbol_lower]
                normalized_symbols.append(etf_symbol)
                logger.debug(f"Mapped real estate {symbol} to ETF {etf_symbol}")
            else:
                # Assume it's already an ETF symbol
                normalized_symbols.append(symbol.upper())

        logger.info(f"Fetching real estate data for {len(normalized_symbols)} symbols")

        results = {}
        failed_symbols = []

        for symbol in normalized_symbols:
            try:
                # Check cache first
                cache_key = f"realestate_{symbol}_{period}_{interval}_{start_date}_{end_date}"
                if use_cache and cache_key in self.data_cache:
                    logger.debug(f"Using cached real estate data for {symbol}")
                    results[symbol] = self.data_cache[cache_key].copy()
                    continue

                # Enforce rate limiting
                self._enforce_rate_limit()

                logger.debug(f"Downloading real estate data for {symbol}")

                # Create yfinance ticker object
                ticker = yf.Ticker(symbol)

                # Fetch data
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

                # Add metadata
                data['Symbol'] = symbol
                data['AssetClass'] = 'RealEstate'

                # Cache the data
                if use_cache:
                    self.data_cache[cache_key] = data.copy()

                results[symbol] = data
                logger.debug(f"Successfully fetched {len(data)} rows for real estate {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch real estate data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to fetch real estate data for: {failed_symbols}")

        if not results:
            raise AlternativeDataError("No real estate data could be fetched")

        logger.info(f"Successfully fetched real estate data for {len(results)} out of {len(normalized_symbols)} symbols")
        return results

    def fetch_all_alternative_data(self,
                                 period: str = "1y",
                                 interval: str = "1d",
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 include_crypto: bool = True,
                                 include_commodities: bool = True,
                                 include_real_estate: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch all available alternative asset data.

        Args:
            period: Time period
            interval: Data interval
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            include_crypto: Whether to include cryptocurrency data
            include_commodities: Whether to include commodity data
            include_real_estate: Whether to include real estate data

        Returns:
            Nested dictionary with DataFrames for each asset class
        """
        logger.info("Fetching all alternative asset data")

        results = {}

        if include_crypto:
            try:
                # Fetch major cryptocurrencies
                major_cryptos = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
                crypto_data = self.fetch_crypto_data(
                    major_cryptos, period=period, interval=interval,
                    start_date=start_date, end_date=end_date
                )
                results['cryptocurrency'] = crypto_data
                logger.info(f"Fetched {len(crypto_data)} cryptocurrency assets")
            except Exception as e:
                logger.error(f"Failed to fetch cryptocurrency data: {str(e)}")

        if include_commodities:
            try:
                # Fetch major commodities
                major_commodities = ['gold', 'silver', 'oil', 'natural_gas', 'agriculture', 'commodities_broad']
                commodity_data = self.fetch_commodity_data(
                    major_commodities, period=period, interval=interval,
                    start_date=start_date, end_date=end_date
                )
                results['commodities'] = commodity_data
                logger.info(f"Fetched {len(commodity_data)} commodity assets")
            except Exception as e:
                logger.error(f"Failed to fetch commodity data: {str(e)}")

        if include_real_estate:
            try:
                # Fetch major real estate ETFs
                major_reits = ['us_reits', 'global_reits', 'residential_reits', 'commercial_reits']
                real_estate_data = self.fetch_real_estate_data(
                    major_reits, period=period, interval=interval,
                    start_date=start_date, end_date=end_date
                )
                results['real_estate'] = real_estate_data
                logger.info(f"Fetched {len(real_estate_data)} real estate assets")
            except Exception as e:
                logger.error(f"Failed to fetch real estate data: {str(e)}")

        if not results:
            raise AlternativeDataError("No alternative asset data could be fetched")

        total_assets = sum(len(asset_data) for asset_data in results.values())
        logger.info(f"Successfully fetched {total_assets} alternative assets across {len(results)} categories")

        return results

    def get_available_symbols(self) -> Dict[str, List[str]]:
        """
        Get all available symbols organized by asset class.

        Returns:
            Dictionary mapping asset classes to available symbols
        """
        return {
            'cryptocurrency': list(self.CRYPTO_SYMBOLS.keys()),
            'commodities': list(self.COMMODITY_ETFS.keys()),
            'real_estate': list(self.REAL_ESTATE_ETFS.keys())
        }

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self.data_cache.clear()
        logger.info("Alternative data cache cleared")

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
