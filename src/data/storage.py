"""
Data Storage Utilities Module

This module provides comprehensive data storage and retrieval capabilities with
support for multiple formats, compression, metadata tracking, and data versioning.
Optimized for financial time series data with efficient storage and fast retrieval.

Author: Portfolio Optimization Team
Last Updated: 2025-07-10
"""

import pandas as pd
import numpy as np
import json
import logging
import shutil
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timezone
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logger = logging.getLogger(__name__)


class DataStorageError(Exception):
    """Custom exception for data storage errors"""
    pass


class DataStorage:
    """
    Professional-grade data storage system with comprehensive features for
    financial data management.

    Features:
    - Multiple storage formats (Parquet, HDF5, CSV)
    - Data compression and optimization
    - Metadata tracking and versioning
    - Efficient data partitioning
    - Data validation and integrity checks
    - Backup and recovery capabilities
    - Performance monitoring and optimization
    """

    SUPPORTED_FORMATS = ['parquet', 'hdf5', 'csv', 'json', 'pickle']

    def __init__(self,
                 storage_path: str = "data",
                 default_format: str = "parquet",
                 compression: str = "snappy",
                 enable_versioning: bool = True,
                 enable_metadata: bool = True,
                 backup_enabled: bool = False):
        """
        Initialize the data storage system.

        Args:
            storage_path: Base path for data storage
            default_format: Default storage format
            compression: Compression algorithm ('snappy', 'gzip', 'lz4', 'brotli')
            enable_versioning: Whether to enable data versioning
            enable_metadata: Whether to track metadata
            backup_enabled: Whether to enable automatic backups
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.raw_path = self.storage_path / "raw"
        self.processed_path = self.storage_path / "processed"
        self.features_path = self.storage_path / "features"
        self.metadata_path = self.storage_path / "metadata"
        self.backup_path = self.storage_path / "backups"

        for path in [self.raw_path, self.processed_path, self.features_path,
                    self.metadata_path, self.backup_path]:
            path.mkdir(exist_ok=True)

        self.default_format = default_format
        self.compression = compression
        self.enable_versioning = enable_versioning
        self.enable_metadata = enable_metadata
        self.backup_enabled = backup_enabled

        # Validate format
        if default_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {default_format}. "
                           f"Supported formats: {self.SUPPORTED_FORMATS}")

        logger.info(f"DataStorage initialized with path={self.storage_path}, "
                   f"format={default_format}, compression={compression}")

    def _generate_filename(self,
                          base_name: str,
                          format: str,
                          version: Optional[int] = None,
                          timestamp: bool = False) -> str:
        """
        Generate a filename with optional versioning and timestamps.

        Args:
            base_name: Base filename without extension
            format: File format
            version: Version number (if versioning enabled)
            timestamp: Whether to include timestamp

        Returns:
            Generated filename
        """
        # Clean base name
        base_name = base_name.replace(' ', '_').replace('-', '_')

        # Add timestamp if requested
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{ts}"

        # Add version if provided
        if version is not None and self.enable_versioning:
            base_name = f"{base_name}_v{version:03d}"

        # Add extension
        if format == 'parquet':
            extension = '.parquet'
        elif format == 'hdf5':
            extension = '.h5'
        elif format == 'csv':
            extension = '.csv'
        elif format == 'json':
            extension = '.json'
        elif format == 'pickle':
            extension = '.pkl'
        else:
            extension = f'.{format}'

        return f"{base_name}{extension}"

    def _save_metadata(self,
                      filename: str,
                      data: pd.DataFrame,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save metadata for a dataset.

        Args:
            filename: Data filename
            data: DataFrame to extract metadata from
            metadata: Additional metadata to save
        """
        if not self.enable_metadata:
            return

        # Extract basic metadata from DataFrame
        basic_metadata = {
            'filename': filename,
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'null_counts': data.isnull().sum().to_dict(),
            'date_range': {
                'start': data.index.min().isoformat() if hasattr(data.index, 'min') else None,
                'end': data.index.max().isoformat() if hasattr(data.index, 'max') else None
            } if isinstance(data.index, pd.DatetimeIndex) else None,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'compression': self.compression
        }

        # Add custom metadata
        if metadata:
            basic_metadata.update(metadata)

        # Save metadata as JSON
        metadata_filename = Path(filename).stem + '_metadata.json'
        metadata_path = self.metadata_path / metadata_filename

        with open(metadata_path, 'w') as f:
            json.dump(basic_metadata, f, indent=2, default=str)

        logger.debug(f"Saved metadata for {filename}")

    def _load_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a dataset.

        Args:
            filename: Data filename

        Returns:
            Metadata dictionary or None if not found
        """
        if not self.enable_metadata:
            return None

        metadata_filename = Path(filename).stem + '_metadata.json'
        metadata_path = self.metadata_path / metadata_filename

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {filename}: {str(e)}")
            return None

    def save_data(self,
                  data: pd.DataFrame,
                  filename: str,
                  format: Optional[str] = None,
                  data_type: str = "processed",
                  metadata: Optional[Dict[str, Any]] = None,
                  overwrite: bool = False) -> str:
        """
        Save data to storage with comprehensive options.

        Args:
            data: DataFrame to save
            filename: Base filename (without extension)
            format: Storage format (uses default if None)
            data_type: Data type ('raw', 'processed', 'features')
            metadata: Additional metadata to save
            overwrite: Whether to overwrite existing files

        Returns:
            Full path to saved file

        Raises:
            DataStorageError: If saving fails
        """
        if data.empty:
            raise DataStorageError("Cannot save empty DataFrame")

        # Determine format and path
        format = format or self.default_format
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")

        # Determine storage path based on data type
        if data_type == "raw":
            storage_dir = self.raw_path
        elif data_type == "processed":
            storage_dir = self.processed_path
        elif data_type == "features":
            storage_dir = self.features_path
        else:
            raise ValueError(f"Invalid data_type: {data_type}. "
                           f"Valid types: 'raw', 'processed', 'features'")

        # Generate filename
        full_filename = self._generate_filename(filename, format)
        file_path = storage_dir / full_filename

        # Check if file exists
        if file_path.exists() and not overwrite:
            if self.enable_versioning:
                # Find next version number
                version = 1
                while True:
                    versioned_filename = self._generate_filename(filename, format, version)
                    versioned_path = storage_dir / versioned_filename
                    if not versioned_path.exists():
                        file_path = versioned_path
                        full_filename = versioned_filename
                        break
                    version += 1
            else:
                raise DataStorageError(f"File {file_path} already exists and overwrite=False")

        try:
            logger.info(f"Saving data to {file_path} (format={format}, shape={data.shape})")

            # Save based on format
            if format == 'parquet':
                # Use PyArrow for better performance and compression
                table = pa.Table.from_pandas(data)
                pq.write_table(
                    table,
                    file_path,
                    compression=self.compression,
                    use_dictionary=True,  # Better compression for categorical data
                    write_statistics=True  # Enable column statistics
                )

            elif format == 'hdf5':
                data.to_hdf(
                    file_path,
                    key='data',
                    mode='w',
                    complevel=9 if self.compression else 0,
                    complib='blosc:zstd' if self.compression else None
                )

            elif format == 'csv':
                if self.compression == 'gzip':
                    data.to_csv(file_path, compression='gzip')
                else:
                    data.to_csv(file_path)

            elif format == 'json':
                if self.compression == 'gzip':
                    with gzip.open(f"{file_path}.gz", 'wt') as f:
                        data.to_json(f, orient='index', date_format='iso')
                    file_path = Path(f"{file_path}.gz")
                else:
                    data.to_json(file_path, orient='index', date_format='iso')

            elif format == 'pickle':
                if self.compression == 'gzip':
                    data.to_pickle(file_path, compression='gzip')
                else:
                    data.to_pickle(file_path)

            # Save metadata
            self._save_metadata(full_filename, data, metadata)

            # Create backup if enabled
            if self.backup_enabled:
                self._create_backup(file_path)

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"Successfully saved {file_path} ({file_size_mb:.2f} MB)")

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
            raise DataStorageError(f"Failed to save data: {str(e)}")

    def load_data(self,
                  filename: str,
                  format: Optional[str] = None,
                  data_type: str = "processed",
                  columns: Optional[List[str]] = None,
                  date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Load data from storage with filtering options.

        Args:
            filename: Filename to load (with or without extension)
            format: Storage format (auto-detected if None)
            data_type: Data type ('raw', 'processed', 'features')
            columns: Specific columns to load (if supported by format)
            date_range: Date range tuple (start, end) for filtering

        Returns:
            Loaded DataFrame

        Raises:
            DataStorageError: If loading fails
        """
        # Determine storage path based on data type
        if data_type == "raw":
            storage_dir = self.raw_path
        elif data_type == "processed":
            storage_dir = self.processed_path
        elif data_type == "features":
            storage_dir = self.features_path
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        # Find the file (handle different extensions)
        file_path = None
        if format:
            # Specific format requested
            full_filename = self._generate_filename(Path(filename).stem, format)
            potential_path = storage_dir / full_filename
            if potential_path.exists():
                file_path = potential_path
        else:
            # Auto-detect format
            base_name = Path(filename).stem
            for fmt in self.SUPPORTED_FORMATS:
                potential_filename = self._generate_filename(base_name, fmt)
                potential_path = storage_dir / potential_filename
                if potential_path.exists():
                    file_path = potential_path
                    format = fmt
                    break

        if not file_path or not file_path.exists():
            raise DataStorageError(f"File not found: {filename} in {storage_dir}")

        try:
            logger.info(f"Loading data from {file_path} (format={format})")

            # Load based on format
            if format == 'parquet':
                if columns:
                    data = pd.read_parquet(file_path, columns=columns)
                else:
                    data = pd.read_parquet(file_path)

            elif format == 'hdf5':
                data = pd.read_hdf(file_path, key='data')
                if columns:
                    data = data[columns]

            elif format == 'csv':
                if str(file_path).endswith('.gz'):
                    data = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
                else:
                    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if columns:
                    data = data[columns]

            elif format == 'json':
                if str(file_path).endswith('.gz'):
                    with gzip.open(file_path, 'rt') as f:
                        data = pd.read_json(f, orient='index')
                else:
                    data = pd.read_json(file_path, orient='index')
                if columns:
                    data = data[columns]

            elif format == 'pickle':
                data = pd.read_pickle(file_path)
                if columns:
                    data = data[columns]

            # Apply date range filter if specified
            if date_range and isinstance(data.index, pd.DatetimeIndex):
                start_date, end_date = date_range
                data = data.loc[start_date:end_date]

            logger.info(f"Successfully loaded data with shape {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise DataStorageError(f"Failed to load data: {str(e)}")

    def list_files(self,
                   data_type: str = "processed",
                   format_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all files in storage with metadata.

        Args:
            data_type: Data type to list ('raw', 'processed', 'features')
            format_filter: Filter by specific format

        Returns:
            List of file information dictionaries
        """
        if data_type == "raw":
            storage_dir = self.raw_path
        elif data_type == "processed":
            storage_dir = self.processed_path
        elif data_type == "features":
            storage_dir = self.features_path
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        files = []
        for file_path in storage_dir.iterdir():
            if file_path.is_file():
                # Determine format from extension
                if file_path.suffix == '.parquet':
                    file_format = 'parquet'
                elif file_path.suffix == '.h5':
                    file_format = 'hdf5'
                elif file_path.suffix == '.csv':
                    file_format = 'csv'
                elif file_path.suffix == '.json':
                    file_format = 'json'
                elif file_path.suffix == '.pkl':
                    file_format = 'pickle'
                else:
                    continue  # Skip unknown formats

                # Apply format filter
                if format_filter and file_format != format_filter:
                    continue

                # Get file stats
                stat = file_path.stat()
                file_info = {
                    'filename': file_path.name,
                    'format': file_format,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'data_type': data_type
                }

                # Add metadata if available
                metadata = self._load_metadata(file_path.name)
                if metadata:
                    file_info['metadata'] = metadata

                files.append(file_info)

        return sorted(files, key=lambda x: x['modified'], reverse=True)

    def delete_file(self, filename: str, data_type: str = "processed") -> bool:
        """
        Delete a file and its metadata.

        Args:
            filename: Filename to delete
            data_type: Data type ('raw', 'processed', 'features')

        Returns:
            True if successful, False otherwise
        """
        if data_type == "raw":
            storage_dir = self.raw_path
        elif data_type == "processed":
            storage_dir = self.processed_path
        elif data_type == "features":
            storage_dir = self.features_path
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        file_path = storage_dir / filename

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")

                # Delete metadata
                metadata_filename = Path(filename).stem + '_metadata.json'
                metadata_path = self.metadata_path / metadata_filename
                if metadata_path.exists():
                    metadata_path.unlink()
                    logger.info(f"Deleted metadata: {metadata_path}")

                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False

    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the file."""
        try:
            backup_filename = f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
            backup_path = self.backup_path / backup_filename
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {str(e)}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'storage_path': str(self.storage_path),
            'total_size_mb': 0,
            'file_counts': {},
            'data_types': {}
        }

        for data_type, path in [('raw', self.raw_path),
                               ('processed', self.processed_path),
                               ('features', self.features_path)]:
            files = self.list_files(data_type)
            total_size = sum(f['size_mb'] for f in files)

            stats['data_types'][data_type] = {
                'file_count': len(files),
                'total_size_mb': total_size,
                'formats': {}
            }

            # Count by format
            for file_info in files:
                format_name = file_info['format']
                if format_name not in stats['data_types'][data_type]['formats']:
                    stats['data_types'][data_type]['formats'][format_name] = {
                        'count': 0,
                        'size_mb': 0
                    }
                stats['data_types'][data_type]['formats'][format_name]['count'] += 1
                stats['data_types'][data_type]['formats'][format_name]['size_mb'] += file_info['size_mb']

            stats['total_size_mb'] += total_size

            # Update file counts by format
            for format_name in stats['data_types'][data_type]['formats']:
                if format_name not in stats['file_counts']:
                    stats['file_counts'][format_name] = 0
                stats['file_counts'][format_name] += stats['data_types'][data_type]['formats'][format_name]['count']

        return stats
