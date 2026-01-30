"""
FRED API fetcher for external market data.

Downloads and caches:
- VIXCLS: CBOE VIX close (daily)
- DGS10: 10Y Treasury yield (daily, % p.a.)
- USREC: NBER recession indicator (monthly, 0/1)
- SP500: S&P 500 index (daily)

Used only during TRAIN phase to create target variables (Y).
"""

from __future__ import annotations

import hashlib
import time
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from athenai.core.logging import get_logger


# Type alias for supported series
FREDSeriesId = Literal["VIXCLS", "DGS10", "USREC", "SP500"]


class FREDFetcher:
    """
    Fetch and cache FRED data series.
    
    Requires FRED API key set in environment variable FRED_API_KEY
    or passed directly.
    
    Usage:
        fetcher = FREDFetcher(api_key="xxx", cache_dir=Path("data/external"))
        
        # Fetch single series
        vix = fetcher.fetch("VIXCLS", start="2020-01-01", end="2024-12-31")
        
        # Fetch all series
        all_data = fetcher.fetch_all(start="2020-01-01", end="2024-12-31")
    """
    
    # Series configurations
    SERIES_CONFIG = {
        "VIXCLS": {
            "name": "CBOE VIX Close",
            "frequency": "daily",
            "units": "index",
            "description": "CBOE Volatility Index: VIX",
        },
        "DGS10": {
            "name": "10-Year Treasury Rate",
            "frequency": "daily",
            "units": "percent",
            "description": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
        },
        "USREC": {
            "name": "NBER Recession Indicator",
            "frequency": "monthly",
            "units": "dummy",
            "description": "NBER based Recession Indicators for the US",
        },
        "SP500": {
            "name": "S&P 500",
            "frequency": "daily",
            "units": "index",
            "description": "S&P 500 Index",
        },
    }
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str = Path("data/external"),
        use_cache: bool = True,
    ):
        """
        Initialize FRED fetcher.
        
        Args:
            api_key: FRED API key. If None, tries to read from FRED_API_KEY env var.
            cache_dir: Directory to store cached data.
            use_cache: Whether to use cached data if available.
        """
        import os
        
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.logger = get_logger()
        
        if not self.api_key:
            self.logger.warning(
                "No FRED API key found. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. Using mock data for development."
            )
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, series_id: str, start: str, end: str) -> Path:
        """Generate cache path for a series."""
        # Hash the request params for unique cache key
        key = f"{series_id}_{start}_{end}"
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.cache_dir / f"fred_{series_id.lower()}_{hash_suffix}.parquet"
    
    def _load_cache(self, path: Path) -> pl.DataFrame | None:
        """Load cached data if exists and use_cache is True."""
        if self.use_cache and path.exists():
            self.logger.info(f"Loading cached data from {path.name}")
            return pl.read_parquet(str(path))
        return None
    
    def _save_cache(self, df: pl.DataFrame, path: Path) -> None:
        """Save data to cache."""
        df.write_parquet(str(path), compression="zstd")
        self.logger.info(f"Cached data to {path.name}")
    
    def fetch(
        self,
        series_id: FREDSeriesId,
        start: str | date = "2020-01-01",
        end: str | date = "2024-12-31",
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """
        Fetch a single FRED series.
        
        Args:
            series_id: FRED series ID (e.g., "VIXCLS", "DGS10")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            force_refresh: Ignore cache and fetch fresh data
            
        Returns:
            DataFrame with columns: date, value, series_id
        """
        start_str = start if isinstance(start, str) else start.isoformat()
        end_str = end if isinstance(end, str) else end.isoformat()
        
        # Check cache
        cache_path = self._cache_path(series_id, start_str, end_str)
        if not force_refresh:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached
        
        # Fetch from API
        if self.api_key:
            df = self._fetch_from_api(series_id, start_str, end_str)
        else:
            df = self._generate_mock_data(series_id, start_str, end_str)
        
        # Save to cache
        self._save_cache(df, cache_path)
        
        return df
    
    def _fetch_from_api(
        self,
        series_id: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Fetch data from FRED API."""
        import urllib.request
        import urllib.parse
        import json
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        
        self.logger.info(f"Fetching {series_id} from FRED API...")
        
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            self.logger.error(f"Failed to fetch {series_id}: {e}")
            self.logger.info("Falling back to mock data")
            return self._generate_mock_data(series_id, start, end)
        
        # Parse observations
        observations = data.get("observations", [])
        
        records = []
        for obs in observations:
            try:
                dt = datetime.strptime(obs["date"], "%Y-%m-%d").date()
                val_str = obs["value"]
                # Handle FRED's "." for missing values
                value = float(val_str) if val_str != "." else None
                records.append({"date": dt, "value": value, "series_id": series_id})
            except (ValueError, KeyError):
                continue
        
        df = pl.DataFrame(records)
        
        # Convert date column
        if len(df) > 0:
            df = df.with_columns([
                pl.col("date").cast(pl.Date),
            ])
        
        self.logger.info(f"Fetched {len(df)} observations for {series_id}")
        
        # Rate limiting
        time.sleep(0.5)
        
        return df
    
    def _generate_mock_data(
        self,
        series_id: str,
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """
        Generate mock data for development/testing.
        
        This allows the pipeline to run without API keys for testing.
        """
        self.logger.warning(f"Generating mock data for {series_id}")
        
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        
        # Generate date range
        config = self.SERIES_CONFIG.get(series_id, {})
        freq = config.get("frequency", "daily")
        
        if freq == "monthly":
            # Monthly dates
            dates = []
            current = start_dt.replace(day=1)
            while current <= end_dt:
                dates.append(current)
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        else:
            # Daily dates (business days approximation)
            dates = []
            current = start_dt
            while current <= end_dt:
                # Skip weekends
                if current.weekday() < 5:
                    dates.append(current)
                current = date(
                    current.year,
                    current.month,
                    current.day + 1 if current.day < 28 else 1
                )
                if current.day == 1:
                    if current.month == 12:
                        current = date(current.year + 1, 1, 1)
                    else:
                        current = date(current.year, current.month + 1, 1)
        
        # Generate values based on series type
        np.random.seed(42 + hash(series_id) % 1000)
        n = len(dates)
        
        if series_id == "VIXCLS":
            # VIX: typically 10-80, mean ~20, with occasional spikes
            base = 20 + np.random.randn(n).cumsum() * 0.5
            spikes = np.random.exponential(5, n) * (np.random.rand(n) > 0.95)
            values = np.clip(base + spikes, 10, 80)
        
        elif series_id == "DGS10":
            # 10Y Treasury: typically 1-5%, trending
            values = 2.5 + np.random.randn(n).cumsum() * 0.02
            values = np.clip(values, 0.5, 6.0)
        
        elif series_id == "USREC":
            # Recession: mostly 0, with recession periods
            values = np.zeros(n)
            # Simulate 2020 recession
            for i, d in enumerate(dates):
                if date(2020, 2, 1) <= d <= date(2020, 6, 1):
                    values[i] = 1
        
        elif series_id == "SP500":
            # S&P 500: trending up with volatility
            returns = 0.0004 + np.random.randn(n) * 0.012  # ~10% annual, 19% vol
            prices = 3200 * np.exp(returns.cumsum())
            values = prices
        
        else:
            values = np.random.randn(n)
        
        df = pl.DataFrame({
            "date": dates,
            "value": values,
            "series_id": [series_id] * n,
        })
        
        return df
    
    def fetch_all(
        self,
        start: str | date = "2020-01-01",
        end: str | date = "2024-12-31",
        series: list[FREDSeriesId] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, pl.DataFrame]:
        """
        Fetch multiple FRED series.
        
        Args:
            start: Start date
            end: End date
            series: List of series IDs (default: all supported)
            force_refresh: Ignore cache
            
        Returns:
            Dictionary mapping series_id to DataFrame
        """
        if series is None:
            series = list(self.SERIES_CONFIG.keys())
        
        results = {}
        for sid in series:
            results[sid] = self.fetch(sid, start, end, force_refresh)
        
        return results
    
    def to_wide_format(
        self,
        data: dict[str, pl.DataFrame],
        align_dates: pl.Series | None = None,
    ) -> pl.DataFrame:
        """
        Convert multiple series to wide format.
        
        Args:
            data: Dictionary from fetch_all()
            align_dates: Optional date series to align to
            
        Returns:
            DataFrame with columns: date, VIXCLS, DGS10, USREC, SP500
        """
        # Start with first series
        series_ids = list(data.keys())
        
        if not series_ids:
            return pl.DataFrame()
        
        # Collect all dates if not provided
        if align_dates is None:
            all_dates = set()
            for df in data.values():
                all_dates.update(df["date"].to_list())
            align_dates = pl.Series("date", sorted(all_dates))
        
        # Create base with all dates
        result = pl.DataFrame({"date": align_dates})
        
        # Join each series
        for sid, df in data.items():
            df_renamed = df.select([
                pl.col("date"),
                pl.col("value").alias(sid.lower()),
            ])
            result = result.join(df_renamed, on="date", how="left")
        
        return result.sort("date")
    
    @staticmethod
    def get_series_info(series_id: FREDSeriesId) -> dict:
        """Get metadata about a series."""
        return FREDFetcher.SERIES_CONFIG.get(series_id, {})
