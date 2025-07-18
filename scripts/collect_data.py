#!/usr/bin/env python3
"""
Comprehensive Data Collection Script

This script orchestrates the collection of market data, economic indicators,
and alternative assets using the data collection pipeline. It provides
command-line interface for flexible data collection with proper error
handling and progress tracking.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.collectors.market_data import MarketDataCollector
from data.collectors.economic_data import EconomicDataCollector
from data.collectors.alternative_data import AlternativeDataCollector
from data.storage import DataStorage
from utils.config import ConfigManager
from utils.logger import get_logger


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/data_collection.log')
        ]
    )
    return logging.getLogger(__name__)


def collect_market_data(config: Dict, 
                       storage: DataStorage, 
                       period: str = "1y",
                       verbose: bool = False) -> Dict[str, int]:
    """
    Collect market data for all configured assets.
    
    Args:
        config: Data configuration
        storage: Data storage instance
        period: Time period to collect
        verbose: Verbose logging
        
    Returns:
        Dictionary with collection statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting market data collection")
    
    collector = MarketDataCollector(
        cache_dir="cache/market_data",
        rate_limit_delay=0.1,
        validate_data=True
    )
    
    stats = {'total_symbols': 0, 'successful': 0, 'failed': 0}
    
    # Collect equity indices
    if 'equity_indices' in config.get('market_data', {}):
        symbols = config['market_data']['equity_indices']
        logger.info(f"Collecting {len(symbols)} equity indices")
        
        try:
            data = collector.fetch_stock_data(symbols, period=period)
            
            for symbol, df in data.items():
                try:
                    filename = f"equity_index_{symbol.replace('^', '').replace('-', '_')}"
                    storage.save_data(
                        df, 
                        filename, 
                        data_type="raw",
                        metadata={
                            'asset_class': 'equity_index',
                            'symbol': symbol,
                            'collection_date': datetime.now().isoformat(),
                            'period': period
                        }
                    )
                    stats['successful'] += 1
                    if verbose:
                        logger.info(f"Saved {symbol}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to save {symbol}: {str(e)}")
                    stats['failed'] += 1
            
            stats['total_symbols'] += len(symbols)
            
        except Exception as e:
            logger.error(f"Failed to collect equity indices: {str(e)}")
            stats['failed'] += len(symbols)
    
    # Collect fixed income ETFs
    if 'fixed_income' in config.get('market_data', {}):
        symbols = config['market_data']['fixed_income']
        logger.info(f"Collecting {len(symbols)} fixed income ETFs")
        
        try:
            data = collector.fetch_etf_data(symbols, period=period)
            
            for symbol, df in data.items():
                try:
                    filename = f"fixed_income_{symbol}"
                    storage.save_data(
                        df, 
                        filename, 
                        data_type="raw",
                        metadata={
                            'asset_class': 'fixed_income',
                            'symbol': symbol,
                            'collection_date': datetime.now().isoformat(),
                            'period': period
                        }
                    )
                    stats['successful'] += 1
                    if verbose:
                        logger.info(f"Saved {symbol}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to save {symbol}: {str(e)}")
                    stats['failed'] += 1
            
            stats['total_symbols'] += len(symbols)
            
        except Exception as e:
            logger.error(f"Failed to collect fixed income ETFs: {str(e)}")
            stats['failed'] += len(symbols)
    
    # Collect commodities
    if 'commodities' in config.get('market_data', {}):
        symbols = config['market_data']['commodities']
        logger.info(f"Collecting {len(symbols)} commodity ETFs")
        
        try:
            data = collector.fetch_etf_data(symbols, period=period)
            
            for symbol, df in data.items():
                try:
                    filename = f"commodity_{symbol}"
                    storage.save_data(
                        df, 
                        filename, 
                        data_type="raw",
                        metadata={
                            'asset_class': 'commodity',
                            'symbol': symbol,
                            'collection_date': datetime.now().isoformat(),
                            'period': period
                        }
                    )
                    stats['successful'] += 1
                    if verbose:
                        logger.info(f"Saved {symbol}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to save {symbol}: {str(e)}")
                    stats['failed'] += 1
            
            stats['total_symbols'] += len(symbols)
            
        except Exception as e:
            logger.error(f"Failed to collect commodity ETFs: {str(e)}")
            stats['failed'] += len(symbols)
    
    # Collect alternatives
    if 'alternatives' in config.get('market_data', {}):
        symbols = config['market_data']['alternatives']
        logger.info(f"Collecting {len(symbols)} alternative assets")
        
        try:
            data = collector.fetch_stock_data(symbols, period=period)
            
            for symbol, df in data.items():
                try:
                    filename = f"alternative_{symbol.replace('-', '_')}"
                    storage.save_data(
                        df, 
                        filename, 
                        data_type="raw",
                        metadata={
                            'asset_class': 'alternative',
                            'symbol': symbol,
                            'collection_date': datetime.now().isoformat(),
                            'period': period
                        }
                    )
                    stats['successful'] += 1
                    if verbose:
                        logger.info(f"Saved {symbol}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to save {symbol}: {str(e)}")
                    stats['failed'] += 1
            
            stats['total_symbols'] += len(symbols)
            
        except Exception as e:
            logger.error(f"Failed to collect alternative assets: {str(e)}")
            stats['failed'] += len(symbols)
    
    logger.info(f"Market data collection completed: {stats['successful']}/{stats['total_symbols']} successful")
    return stats


def collect_economic_data(config: Dict, 
                         storage: DataStorage, 
                         start_date: Optional[str] = None,
                         verbose: bool = False) -> Dict[str, int]:
    """
    Collect economic data from FRED API.
    
    Args:
        config: Data configuration
        storage: Data storage instance
        start_date: Start date for data collection
        verbose: Verbose logging
        
    Returns:
        Dictionary with collection statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting economic data collection")
    
    collector = EconomicDataCollector(
        cache_dir="cache/economic_data",
        rate_limit_delay=0.1
    )
    
    stats = {'total_categories': 0, 'successful': 0, 'failed': 0}
    
    # Collect different categories of economic data
    categories = [
        ('interest_rates', collector.fetch_interest_rates),
        ('inflation', collector.fetch_inflation_data),
        ('employment', collector.fetch_employment_data),
        ('economic_activity', collector.fetch_economic_activity_data)
    ]
    
    for category_name, fetch_method in categories:
        try:
            logger.info(f"Collecting {category_name} data")
            
            data = fetch_method(start_date=start_date)
            
            filename = f"economic_{category_name}"
            storage.save_data(
                data, 
                filename, 
                data_type="raw",
                metadata={
                    'data_category': 'economic',
                    'subcategory': category_name,
                    'collection_date': datetime.now().isoformat(),
                    'start_date': start_date,
                    'source': 'FRED'
                }
            )
            
            stats['successful'] += 1
            if verbose:
                logger.info(f"Saved {category_name}: {len(data)} rows, {len(data.columns)} series")
                
        except Exception as e:
            logger.error(f"Failed to collect {category_name}: {str(e)}")
            stats['failed'] += 1
        
        stats['total_categories'] += 1
    
    logger.info(f"Economic data collection completed: {stats['successful']}/{stats['total_categories']} successful")
    return stats


def collect_alternative_data(config: Dict, 
                           storage: DataStorage, 
                           period: str = "1y",
                           verbose: bool = False) -> Dict[str, int]:
    """
    Collect alternative asset data.
    
    Args:
        config: Data configuration
        storage: Data storage instance
        period: Time period to collect
        verbose: Verbose logging
        
    Returns:
        Dictionary with collection statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting alternative data collection")
    
    collector = AlternativeDataCollector(
        cache_dir="cache/alternative_data",
        rate_limit_delay=0.1,
        validate_data=True
    )
    
    stats = {'total_categories': 0, 'successful': 0, 'failed': 0}
    
    try:
        # Collect all alternative data
        alt_data = collector.fetch_all_alternative_data(
            period=period,
            include_crypto=True,
            include_commodities=True,
            include_real_estate=True
        )
        
        for category, asset_data in alt_data.items():
            for symbol, df in asset_data.items():
                try:
                    filename = f"alternative_{category}_{symbol.replace('-', '_')}"
                    storage.save_data(
                        df, 
                        filename, 
                        data_type="raw",
                        metadata={
                            'asset_class': 'alternative',
                            'subcategory': category,
                            'symbol': symbol,
                            'collection_date': datetime.now().isoformat(),
                            'period': period
                        }
                    )
                    stats['successful'] += 1
                    if verbose:
                        logger.info(f"Saved {category}/{symbol}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to save {category}/{symbol}: {str(e)}")
                    stats['failed'] += 1
        
        stats['total_categories'] = len(alt_data)
        
    except Exception as e:
        logger.error(f"Failed to collect alternative data: {str(e)}")
        stats['failed'] += 1
    
    logger.info(f"Alternative data collection completed: {stats['successful']} assets")
    return stats


def main():
    """Main data collection function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Collect market data, economic indicators, and alternative assets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all data for the past year
  python scripts/collect_data.py --all --period 1y

  # Collect only market data for the past 6 months
  python scripts/collect_data.py --market-data --period 6mo

  # Collect economic data from a specific start date
  python scripts/collect_data.py --economic-data --start-date 2020-01-01

  # Collect with verbose output
  python scripts/collect_data.py --all --period 2y --verbose
        """
    )

    # Data collection options
    parser.add_argument("--all", action="store_true",
                       help="Collect all types of data")
    parser.add_argument("--market-data", action="store_true",
                       help="Collect market data (stocks, ETFs, indices)")
    parser.add_argument("--economic-data", action="store_true",
                       help="Collect economic indicators from FRED")
    parser.add_argument("--alternative-data", action="store_true",
                       help="Collect alternative assets (crypto, commodities, REITs)")

    # Time period options
    parser.add_argument("--period", type=str, default="1y",
                       help="Time period for data collection (default: 1y)")
    parser.add_argument("--start-date", type=str,
                       help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str,
                       help="End date in YYYY-MM-DD format")

    # Configuration options
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                       help="Path to data configuration file")
    parser.add_argument("--storage-path", type=str, default="data",
                       help="Path for data storage")
    parser.add_argument("--format", type=str, default="parquet",
                       choices=["parquet", "hdf5", "csv"],
                       help="Storage format")

    # Output options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be collected without actually collecting")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    logger.info("Starting data collection pipeline")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config('data_config')
        logger.info(f"Loaded configuration from {args.config}")

        # Initialize storage
        storage = DataStorage(
            storage_path=args.storage_path,
            default_format=args.format,
            enable_versioning=True,
            enable_metadata=True
        )
        logger.info(f"Initialized storage at {args.storage_path}")

        # Determine what to collect
        collect_market = args.all or args.market_data
        collect_economic = args.all or args.economic_data
        collect_alternative = args.all or args.alternative_data

        if not any([collect_market, collect_economic, collect_alternative]):
            logger.error("No data collection options specified. Use --all or specific options.")
            return 1

        # Show collection plan
        logger.info("Data collection plan:")
        if collect_market:
            logger.info("  [x] Market data (stocks, ETFs, indices)")
        if collect_economic:
            logger.info("  [x] Economic indicators (FRED API)")
        if collect_alternative:
            logger.info("  [x] Alternative assets (crypto, commodities, REITs)")

        if args.dry_run:
            logger.info("DRY RUN: Would collect the above data but not saving anything")
            return 0

        # Collect data
        total_stats = {'successful': 0, 'failed': 0, 'total': 0}

        if collect_market:
            logger.info("=" * 50)
            market_stats = collect_market_data(
                config, storage, period=args.period, verbose=args.verbose
            )
            total_stats['successful'] += market_stats['successful']
            total_stats['failed'] += market_stats['failed']
            total_stats['total'] += market_stats['total_symbols']

        if collect_economic:
            logger.info("=" * 50)
            economic_stats = collect_economic_data(
                config, storage, start_date=args.start_date, verbose=args.verbose
            )
            total_stats['successful'] += economic_stats['successful']
            total_stats['failed'] += economic_stats['failed']
            total_stats['total'] += economic_stats['total_categories']

        if collect_alternative:
            logger.info("=" * 50)
            alt_stats = collect_alternative_data(
                config, storage, period=args.period, verbose=args.verbose
            )
            total_stats['successful'] += alt_stats['successful']
            total_stats['failed'] += alt_stats['failed']
            total_stats['total'] += alt_stats.get('total_categories', 0)

        # Final summary
        logger.info("=" * 50)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total items processed: {total_stats['total']}")
        logger.info(f"Successful: {total_stats['successful']}")
        logger.info(f"Failed: {total_stats['failed']}")

        if total_stats['total'] > 0:
            success_rate = (total_stats['successful'] / total_stats['total']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")

        # Show storage statistics
        storage_stats = storage.get_storage_stats()
        logger.info(f"Total storage used: {storage_stats['total_size_mb']:.2f} MB")
        logger.info(f"Files by type: {storage_stats['file_counts']}")

        if total_stats['failed'] > 0:
            logger.warning(f"Some data collection failed. Check logs for details.")
            return 1
        else:
            logger.info("All data collection completed successfully!")
            return 0

    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
