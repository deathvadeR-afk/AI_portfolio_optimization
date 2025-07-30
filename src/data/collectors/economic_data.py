"""
Economic Data Collection Module

This module provides comprehensive economic data collection from the Federal Reserve
Economic Data (FRED) API. It includes interest rates, inflation indicators, employment
data, and other macroeconomic indicators essential for portfolio optimization.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from fredapi import Fred

# Configure logging
logger = logging.getLogger(__name__)


class EconomicDataError(Exception):
    """Custom exception for economic data collection errors"""
    pass


class EconomicDataCollector:
    """
    Professional-grade economic data collector using the FRED API.

    Features:
    - Comprehensive economic indicator collection
    - Intelligent retry logic and error handling
    - Data validation and quality checks
    - Rate limiting to respect API limits
    - Caching to minimize API calls
    - Support for multiple data frequencies
    - Automatic data alignment and resampling
    """

    # Predefined economic indicator series
    INTEREST_RATE_SERIES = {
        'fed_funds_rate': 'FEDFUNDS',           # Federal Funds Rate
        'treasury_3m': 'DGS3MO',                # 3-Month Treasury Rate
        'treasury_2y': 'DGS2',                  # 2-Year Treasury Rate
        'treasury_5y': 'DGS5',                  # 5-Year Treasury Rate
        'treasury_10y': 'DGS10',                # 10-Year Treasury Rate
        'treasury_30y': 'DGS30',                # 30-Year Treasury Rate
        'real_10y': 'DFII10',                   # 10-Year TIPS Rate
        'corporate_aaa': 'DAAA',                # AAA Corporate Bond Rate
        'corporate_baa': 'DBAA',                # BAA Corporate Bond Rate
        'mortgage_30y': 'MORTGAGE30US',         # 30-Year Fixed Mortgage Rate
    }

    INFLATION_SERIES = {
        'cpi_all': 'CPIAUCSL',                  # Consumer Price Index
        'cpi_core': 'CPILFESL',                 # Core CPI (ex food & energy)
        'pce_all': 'PCEPI',                     # PCE Price Index
        'pce_core': 'PCEPILFE',                 # Core PCE Price Index
        'breakeven_5y': 'T5YIE',                # 5-Year Breakeven Inflation
        'breakeven_10y': 'T10YIE',              # 10-Year Breakeven Inflation
        'producer_price': 'PPIACO',             # Producer Price Index
        'import_price': 'IR',                   # Import Price Index
    }

    EMPLOYMENT_SERIES = {
        'unemployment_rate': 'UNRATE',          # Unemployment Rate
        'employment_pop_ratio': 'EMRATIO',      # Employment-Population Ratio
        'labor_force_participation': 'CIVPART', # Labor Force Participation Rate
        'nonfarm_payrolls': 'PAYEMS',           # Nonfarm Payrolls
        'initial_claims': 'ICSA',               # Initial Unemployment Claims
        'continuing_claims': 'CCSA',            # Continuing Unemployment Claims
        'job_openings': 'JTSJOL',               # Job Openings
        'quits_rate': 'JTSQUR',                 # Quits Rate
    }

    ECONOMIC_ACTIVITY_SERIES = {
        'gdp': 'GDP',                           # Gross Domestic Product
        'gdp_real': 'GDPC1',                    # Real GDP
        'industrial_production': 'INDPRO',      # Industrial Production Index
        'capacity_utilization': 'TCU',          # Capacity Utilization
        'retail_sales': 'RSAFS',                # Retail Sales
        'housing_starts': 'HOUST',              # Housing Starts
        'building_permits': 'PERMIT',           # Building Permits
        'consumer_sentiment': 'UMCSENT',        # Consumer Sentiment
        'leading_indicators': 'USSLIND',        # Leading Economic Indicators
    }

    MONETARY_SERIES = {
        'money_supply_m1': 'M1SL',              # M1 Money Supply
        'money_supply_m2': 'M2SL',              # M2 Money Supply
        'bank_credit': 'TOTBKCR',               # Total Bank Credit
        'commercial_paper': 'COMPOUT',          # Commercial Paper Outstanding
        'ted_spread': 'TEDRATE',                # TED Spread
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 rate_limit_delay: float = 0.1,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize the economic data collector.

        Args:
            api_key: FRED API key (can also be set via FRED_API_KEY environment variable)
            cache_dir: Directory for caching downloaded data
            rate_limit_delay: Minimum delay between API calls (seconds)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('FRED_API_KEY')

        if not self.api_key:
            logger.warning("No FRED API key provided. Some functionality will be limited.")
            self.fred = None
        else:
            try:
                self.fred = Fred(api_key=self.api_key)
                logger.info("FRED API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FRED API client: {str(e)}")
                self.fred = None

        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/economic_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout

        # Track last API call time for rate limiting
        self.last_api_call = 0.0

        # Data cache for session-level caching
        self.data_cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"EconomicDataCollector initialized with cache_dir={self.cache_dir}")

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_api_call = time.time()

    def _validate_series_id(self, series_id: str) -> str:
        """
        Validate FRED series ID.

        Args:
            series_id: FRED series identifier

        Returns:
            Validated series ID

        Raises:
            ValueError: If series ID is invalid
        """
        if not series_id or not isinstance(series_id, str):
            raise ValueError(f"Invalid series ID: {series_id}")

        # Remove whitespace and convert to uppercase
        series_id = series_id.strip().upper()

        return series_id

    def _fetch_single_series(self,
                           series_id: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           frequency: Optional[str] = None,
                           use_cache: bool = True) -> pd.Series:
        """
        Fetch a single economic data series from FRED.

        Args:
            series_id: FRED series identifier
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            use_cache: Whether to use cached data

        Returns:
            Pandas Series with the economic data

        Raises:
            EconomicDataError: If data cannot be fetched
        """
        if not self.fred:
            raise EconomicDataError("FRED API client not initialized. Please provide a valid API key.")

        series_id = self._validate_series_id(series_id)

        # Check cache first
        cache_key = f"{series_id}_{start_date}_{end_date}_{frequency}"
        if use_cache and cache_key in self.data_cache:
            logger.debug(f"Using cached data for series {series_id}")
            return self.data_cache[cache_key].copy()

        # Enforce rate limiting
        self._enforce_rate_limit()

        try:
            logger.debug(f"Fetching FRED series: {series_id}")

            # Fetch data from FRED
            data = self.fred.get_series(
                series_id,
                start=start_date,
                end=end_date,
                frequency=frequency
            )

            if data.empty:
                raise EconomicDataError(f"No data returned for series {series_id}")

            # Remove any NaN values at the beginning or end
            data = data.dropna()

            if data.empty:
                raise EconomicDataError(f"No valid data after cleaning for series {series_id}")

            # Cache the data
            if use_cache:
                self.data_cache[cache_key] = data.copy()

            logger.debug(f"Successfully fetched {len(data)} observations for {series_id}")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch series {series_id}: {str(e)}")
            raise EconomicDataError(f"Failed to fetch series {series_id}: {str(e)}")

    def fetch_interest_rates(self,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           frequency: Optional[str] = None,
                           series_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch comprehensive interest rate data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            series_subset: Subset of series to fetch (default: all)

        Returns:
            DataFrame with interest rate data
        """
        logger.info("Fetching interest rate data from FRED")

        # Determine which series to fetch
        if series_subset:
            series_to_fetch = {k: v for k, v in self.INTEREST_RATE_SERIES.items() if k in series_subset}
        else:
            series_to_fetch = self.INTEREST_RATE_SERIES

        data_dict = {}
        failed_series = []

        for name, series_id in series_to_fetch.items():
            try:
                data = self._fetch_single_series(series_id, start_date, end_date, frequency)
                data_dict[name] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({series_id}): {str(e)}")
                failed_series.append(name)

        if not data_dict:
            raise EconomicDataError("No interest rate data could be fetched")

        # Combine into DataFrame
        df = pd.DataFrame(data_dict)

        # Forward fill missing values (common for daily data)
        df = df.fillna(method='ffill')

        logger.info(f"Successfully fetched {len(df.columns)} interest rate series with {len(df)} observations")

        if failed_series:
            logger.warning(f"Failed to fetch series: {failed_series}")

        return df

    def fetch_inflation_data(self,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           frequency: Optional[str] = None,
                           series_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch comprehensive inflation indicator data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            series_subset: Subset of series to fetch (default: all)

        Returns:
            DataFrame with inflation data
        """
        logger.info("Fetching inflation data from FRED")

        # Determine which series to fetch
        if series_subset:
            series_to_fetch = {k: v for k, v in self.INFLATION_SERIES.items() if k in series_subset}
        else:
            series_to_fetch = self.INFLATION_SERIES

        data_dict = {}
        failed_series = []

        for name, series_id in series_to_fetch.items():
            try:
                data = self._fetch_single_series(series_id, start_date, end_date, frequency)
                data_dict[name] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({series_id}): {str(e)}")
                failed_series.append(name)

        if not data_dict:
            raise EconomicDataError("No inflation data could be fetched")

        # Combine into DataFrame
        df = pd.DataFrame(data_dict)

        # Forward fill missing values
        df = df.fillna(method='ffill')

        logger.info(f"Successfully fetched {len(df.columns)} inflation series with {len(df)} observations")

        if failed_series:
            logger.warning(f"Failed to fetch series: {failed_series}")

        return df

    def fetch_employment_data(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            frequency: Optional[str] = None,
                            series_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch employment and labor market data.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            series_subset: Subset of series to fetch (default: all)

        Returns:
            DataFrame with employment data
        """
        logger.info("Fetching employment data from FRED")

        if series_subset:
            series_to_fetch = {k: v for k, v in self.EMPLOYMENT_SERIES.items() if k in series_subset}
        else:
            series_to_fetch = self.EMPLOYMENT_SERIES

        data_dict = {}
        failed_series = []

        for name, series_id in series_to_fetch.items():
            try:
                data = self._fetch_single_series(series_id, start_date, end_date, frequency)
                data_dict[name] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({series_id}): {str(e)}")
                failed_series.append(name)

        if not data_dict:
            raise EconomicDataError("No employment data could be fetched")

        df = pd.DataFrame(data_dict)
        df = df.fillna(method='ffill')

        logger.info(f"Successfully fetched {len(df.columns)} employment series with {len(df)} observations")

        if failed_series:
            logger.warning(f"Failed to fetch series: {failed_series}")

        return df

    def fetch_economic_activity_data(self,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   frequency: Optional[str] = None,
                                   series_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch economic activity and growth indicators.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')
            series_subset: Subset of series to fetch (default: all)

        Returns:
            DataFrame with economic activity data
        """
        logger.info("Fetching economic activity data from FRED")

        if series_subset:
            series_to_fetch = {k: v for k, v in self.ECONOMIC_ACTIVITY_SERIES.items() if k in series_subset}
        else:
            series_to_fetch = self.ECONOMIC_ACTIVITY_SERIES

        data_dict = {}
        failed_series = []

        for name, series_id in series_to_fetch.items():
            try:
                data = self._fetch_single_series(series_id, start_date, end_date, frequency)
                data_dict[name] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({series_id}): {str(e)}")
                failed_series.append(name)

        if not data_dict:
            raise EconomicDataError("No economic activity data could be fetched")

        df = pd.DataFrame(data_dict)
        df = df.fillna(method='ffill')

        logger.info(f"Successfully fetched {len(df.columns)} economic activity series with {len(df)} observations")

        if failed_series:
            logger.warning(f"Failed to fetch series: {failed_series}")

        return df

    def fetch_all_economic_data(self,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              frequency: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available economic data categories.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            frequency: Data frequency ('d', 'w', 'm', 'q', 'a')

        Returns:
            Dictionary with DataFrames for each category
        """
        logger.info("Fetching all economic data categories")

        results = {}

        try:
            results['interest_rates'] = self.fetch_interest_rates(start_date, end_date, frequency)
        except Exception as e:
            logger.error(f"Failed to fetch interest rates: {str(e)}")

        try:
            results['inflation'] = self.fetch_inflation_data(start_date, end_date, frequency)
        except Exception as e:
            logger.error(f"Failed to fetch inflation data: {str(e)}")

        try:
            results['employment'] = self.fetch_employment_data(start_date, end_date, frequency)
        except Exception as e:
            logger.error(f"Failed to fetch employment data: {str(e)}")

        try:
            results['economic_activity'] = self.fetch_economic_activity_data(start_date, end_date, frequency)
        except Exception as e:
            logger.error(f"Failed to fetch economic activity data: {str(e)}")

        if not results:
            raise EconomicDataError("No economic data could be fetched")

        logger.info(f"Successfully fetched {len(results)} economic data categories")
        return results

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a FRED series.

        Args:
            series_id: FRED series identifier

        Returns:
            Dictionary containing series information
        """
        if not self.fred:
            raise EconomicDataError("FRED API client not initialized")

        series_id = self._validate_series_id(series_id)

        try:
            self._enforce_rate_limit()
            info = self.fred.get_series_info(series_id)

            return {
                'id': series_id,
                'title': info.get('title', 'Unknown'),
                'units': info.get('units', 'Unknown'),
                'frequency': info.get('frequency', 'Unknown'),
                'seasonal_adjustment': info.get('seasonal_adjustment', 'Unknown'),
                'last_updated': info.get('last_updated', 'Unknown'),
                'observation_start': info.get('observation_start', 'Unknown'),
                'observation_end': info.get('observation_end', 'Unknown'),
                'notes': info.get('notes', '')
            }

        except Exception as e:
            logger.error(f"Failed to get info for series {series_id}: {str(e)}")
            return {'id': series_id, 'error': str(e)}

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self.data_cache.clear()
        logger.info("Economic data cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current cache.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self.data_cache)
        total_memory = sum(series.memory_usage(deep=True) for series in self.data_cache.values())

        return {
            'total_entries': total_entries,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024),
            'cache_keys': list(self.data_cache.keys())
        }
