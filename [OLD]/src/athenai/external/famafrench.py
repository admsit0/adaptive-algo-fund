"""
Fama-French factor data fetcher.

Downloads and caches factor returns from Kenneth French Data Library:
- Mkt-RF: Market excess return
- SMB: Size factor (Small Minus Big)
- HML: Value factor (High Minus Low)
- Mom: Momentum factor (from separate dataset)

Used only during TRAIN phase to create target variables (Y) for factor proxy.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from athenai.core.logging import get_logger


FactorName = Literal["Mkt-RF", "SMB", "HML", "RF", "Mom"]


class FamaFrenchFetcher:
    """
    Fetch and cache Fama-French factor data.
    
    Data source: Kenneth French Data Library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    
    Downloads CSV files and parses them into clean DataFrames.
    
    Usage:
        fetcher = FamaFrenchFetcher(cache_dir=Path("data/external"))
        
        # Fetch daily or monthly factors
        factors = fetcher.fetch_factors(frequency="monthly")
        
        # Get specific date range
        factors = fetcher.fetch_factors(
            frequency="monthly",
            start="2020-01-01",
            end="2024-12-31"
        )
    """
    
    # Dataset URLs
    URLS = {
        # Daily Fama-French 3 factors
        "daily_3factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
        # Monthly Fama-French 3 factors
        "monthly_3factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
        # Monthly Momentum
        "monthly_momentum": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
        # Daily Momentum
        "daily_momentum": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    }
    
    def __init__(
        self,
        cache_dir: Path | str = Path("data/external"),
        use_cache: bool = True,
    ):
        """
        Initialize Fama-French fetcher.
        
        Args:
            cache_dir: Directory to store cached data.
            use_cache: Whether to use cached data if available.
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.logger = get_logger()
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, dataset: str, start: str, end: str) -> Path:
        """Generate cache path for a dataset."""
        key = f"ff_{dataset}_{start}_{end}"
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.cache_dir / f"ff_{dataset}_{hash_suffix}.parquet"
    
    def _load_cache(self, path: Path) -> pl.DataFrame | None:
        """Load cached data if exists."""
        if self.use_cache and path.exists():
            self.logger.info(f"Loading cached FF data from {path.name}")
            return pl.read_parquet(str(path))
        return None
    
    def _save_cache(self, df: pl.DataFrame, path: Path) -> None:
        """Save data to cache."""
        df.write_parquet(str(path), compression="zstd")
        self.logger.info(f"Cached FF data to {path.name}")
    
    def fetch_factors(
        self,
        frequency: Literal["daily", "monthly"] = "monthly",
        start: str | date = "2020-01-01",
        end: str | date = "2024-12-31",
        include_momentum: bool = True,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Fetch Fama-French factors.
        
        Args:
            frequency: "daily" or "monthly"
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            include_momentum: Whether to include momentum factor
            force_refresh: Ignore cache
            
        Returns:
            DataFrame with columns: date, mkt_rf, smb, hml, rf, mom (if included)
        """
        start_str = start if isinstance(start, str) else start.isoformat()
        end_str = end if isinstance(end, str) else end.isoformat()
        
        dataset_key = f"{frequency}_factors{'_mom' if include_momentum else ''}"
        cache_path = self._cache_path(dataset_key, start_str, end_str)
        
        if not force_refresh:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached
        
        # Try to download from web
        try:
            df = self._fetch_from_web(frequency, start_str, end_str, include_momentum)
        except Exception as e:
            self.logger.warning(f"Failed to fetch from web: {e}")
            self.logger.info("Using mock data for development")
            df = self._generate_mock_data(frequency, start_str, end_str, include_momentum)
        
        # Save to cache
        self._save_cache(df, cache_path)
        
        return df
    
    def _fetch_from_web(
        self,
        frequency: str,
        start: str,
        end: str,
        include_momentum: bool,
    ) -> pl.DataFrame:
        """Fetch data from Kenneth French Data Library."""
        import urllib.request
        import zipfile
        import io
        
        self.logger.info(f"Fetching {frequency} Fama-French factors from web...")
        
        # Fetch 3-factor data
        factor_key = f"{frequency}_3factors"
        url_3f = self.URLS.get(factor_key)
        
        if not url_3f:
            raise ValueError(f"No URL for {factor_key}")
        
        try:
            with urllib.request.urlopen(url_3f, timeout=60) as response:
                zip_data = response.read()
        except Exception as e:
            raise RuntimeError(f"Failed to download {factor_key}: {e}")
        
        # Parse zip file
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # Find the CSV file
            csv_files = [n for n in zf.namelist() if n.endswith('.CSV') or n.endswith('.csv')]
            if not csv_files:
                raise RuntimeError("No CSV file found in zip")
            
            with zf.open(csv_files[0]) as f:
                content = f.read().decode('utf-8', errors='ignore')
        
        # Parse the CSV content
        df_3f = self._parse_ff_csv(content, frequency)
        
        # Fetch momentum if requested
        if include_momentum:
            mom_key = f"{frequency}_momentum"
            url_mom = self.URLS.get(mom_key)
            
            if url_mom:
                try:
                    with urllib.request.urlopen(url_mom, timeout=60) as response:
                        zip_data_mom = response.read()
                    
                    with zipfile.ZipFile(io.BytesIO(zip_data_mom)) as zf:
                        csv_files = [n for n in zf.namelist() if n.endswith('.CSV') or n.endswith('.csv')]
                        if csv_files:
                            with zf.open(csv_files[0]) as f:
                                content_mom = f.read().decode('utf-8', errors='ignore')
                            
                            df_mom = self._parse_ff_csv(content_mom, frequency, is_momentum=True)
                            
                            # Merge momentum with 3-factor
                            df_3f = df_3f.join(
                                df_mom.select(["date", "mom"]),
                                on="date",
                                how="left"
                            )
                except Exception as e:
                    self.logger.warning(f"Failed to fetch momentum: {e}")
                    df_3f = df_3f.with_columns([
                        pl.lit(None).alias("mom").cast(pl.Float64)
                    ])
        
        # Filter to date range
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        
        df_3f = df_3f.filter(
            (pl.col("date") >= start_dt) & (pl.col("date") <= end_dt)
        )
        
        self.logger.info(f"Fetched {len(df_3f)} {frequency} factor observations")
        
        return df_3f
    
    def _parse_ff_csv(
        self,
        content: str,
        frequency: str,
        is_momentum: bool = False,
    ) -> pl.DataFrame:
        """
        Parse Fama-French CSV format.
        
        The format has a header section, then data rows with date in first column.
        """
        lines = content.strip().split('\n')
        
        # Find the data start (first line with numeric date)
        data_start = None
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if parts and parts[0].strip().isdigit() and len(parts[0].strip()) >= 6:
                data_start = i
                break
        
        if data_start is None:
            raise ValueError("Could not find data start in FF CSV")
        
        # Parse data rows
        records = []
        for line in lines[data_start:]:
            parts = [p.strip() for p in line.split(',')]
            
            if not parts or not parts[0]:
                continue
            
            # Check for annual data separator
            if 'annual' in parts[0].lower() or not parts[0].replace('-', '').isdigit():
                break
            
            try:
                date_str = parts[0]
                
                if frequency == "daily":
                    # Daily format: YYYYMMDD
                    if len(date_str) == 8:
                        dt = datetime.strptime(date_str, "%Y%m%d").date()
                    else:
                        continue
                else:
                    # Monthly format: YYYYMM
                    if len(date_str) == 6:
                        dt = datetime.strptime(date_str + "01", "%Y%m%d").date()
                    else:
                        continue
                
                if is_momentum:
                    # Momentum file: date, Mom
                    if len(parts) >= 2 and parts[1]:
                        mom_val = float(parts[1]) / 100  # Convert from percentage
                        records.append({"date": dt, "mom": mom_val})
                else:
                    # 3-factor file: date, Mkt-RF, SMB, HML, RF
                    if len(parts) >= 5:
                        records.append({
                            "date": dt,
                            "mkt_rf": float(parts[1]) / 100 if parts[1] else None,
                            "smb": float(parts[2]) / 100 if parts[2] else None,
                            "hml": float(parts[3]) / 100 if parts[3] else None,
                            "rf": float(parts[4]) / 100 if parts[4] else None,
                        })
            except (ValueError, IndexError):
                continue
        
        df = pl.DataFrame(records)
        
        if len(df) > 0:
            df = df.with_columns([
                pl.col("date").cast(pl.Date),
            ])
        
        return df.sort("date")
    
    def _generate_mock_data(
        self,
        frequency: str,
        start: str,
        end: str,
        include_momentum: bool,
    ) -> pl.DataFrame:
        """Generate mock factor data for development/testing."""
        self.logger.warning("Generating mock Fama-French data")
        
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        
        # Generate date range
        dates = []
        current = start_dt
        
        if frequency == "monthly":
            current = current.replace(day=1)
            while current <= end_dt:
                dates.append(current)
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        else:
            while current <= end_dt:
                if current.weekday() < 5:  # Business days
                    dates.append(current)
                # Advance one day
                days_in_month = 28 if current.month == 2 else (30 if current.month in [4, 6, 9, 11] else 31)
                if current.day < days_in_month:
                    current = date(current.year, current.month, current.day + 1)
                elif current.month < 12:
                    current = date(current.year, current.month + 1, 1)
                else:
                    current = date(current.year + 1, 1, 1)
        
        np.random.seed(42)
        n = len(dates)
        
        # Generate factor returns with realistic properties
        # Daily: mean ~0, vol ~1% for factors
        # Monthly: scaled up ~sqrt(21)
        scale = 1.0 if frequency == "daily" else 4.5
        
        data = {
            "date": dates,
            "mkt_rf": (0.0003 + np.random.randn(n) * 0.01) * scale,  # Market premium
            "smb": (0.0001 + np.random.randn(n) * 0.006) * scale,     # Size
            "hml": (0.0001 + np.random.randn(n) * 0.006) * scale,     # Value
            "rf": np.full(n, 0.00015) * scale,                         # Risk-free
        }
        
        if include_momentum:
            data["mom"] = (0.0002 + np.random.randn(n) * 0.008) * scale
        
        df = pl.DataFrame(data)
        
        return df
    
    def get_winning_factor(
        self,
        factors_df: pl.DataFrame,
        factor_cols: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Compute the winning factor each period.
        
        Args:
            factors_df: DataFrame with factor returns
            factor_cols: List of factor columns to compare (default: smb, hml, mom)
            
        Returns:
            DataFrame with winning_factor column added
        """
        if factor_cols is None:
            factor_cols = ["smb", "hml", "mom"]
        
        # Filter to existing columns
        factor_cols = [c for c in factor_cols if c in factors_df.columns]
        
        if not factor_cols:
            raise ValueError("No valid factor columns found")
        
        # Find argmax across factors
        df = factors_df.with_columns([
            # Stack factors and find max
            pl.concat_list([pl.col(c) for c in factor_cols]).alias("_factor_list")
        ])
        
        # Find index of max
        def get_winning_factor(row):
            values = row["_factor_list"]
            if values is None or all(v is None for v in values):
                return None
            max_idx = np.nanargmax([v if v is not None else -np.inf for v in values])
            return factor_cols[max_idx]
        
        # Use Polars native approach
        winning = []
        for row in df.iter_rows(named=True):
            vals = row["_factor_list"]
            if vals is None:
                winning.append(None)
                continue
            valid_vals = [(i, v) for i, v in enumerate(vals) if v is not None]
            if not valid_vals:
                winning.append(None)
            else:
                max_idx = max(valid_vals, key=lambda x: x[1])[0]
                winning.append(factor_cols[max_idx])
        
        df = factors_df.with_columns([
            pl.Series("winning_factor", winning)
        ])
        
        return df
