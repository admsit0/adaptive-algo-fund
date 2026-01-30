"""
Data alignment and normalization utilities for external data.

Handles:
- Aligning monthly data to daily calendar
- Forward fill with publication lag (anti-lookahead)
- Date normalization across different sources
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from athenai.core.logging import get_logger


def align_to_daily(
    df: pl.DataFrame,
    date_col: str = "date",
    value_cols: list[str] | None = None,
    target_dates: pl.Series | None = None,
    method: str = "forward_fill",
) -> pl.DataFrame:
    """
    Align data to a daily calendar.
    
    For monthly data (like USREC), forward-fills values to all business days.
    
    Args:
        df: Input DataFrame with date column
        date_col: Name of date column
        value_cols: Columns to carry forward (default: all except date)
        target_dates: Optional target date series to align to
        method: Fill method ("forward_fill", "asof")
        
    Returns:
        DataFrame with daily observations
    """
    logger = get_logger()
    
    if value_cols is None:
        value_cols = [c for c in df.columns if c != date_col]
    
    # Ensure date is Date type
    df = df.with_columns([
        pl.col(date_col).cast(pl.Date)
    ])
    
    # Get date range
    if target_dates is not None:
        all_dates = target_dates.cast(pl.Date).unique().sort()
    else:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        # Generate business day range
        dates = []
        current = min_date
        while current <= max_date:
            if current.weekday() < 5:  # Monday=0 to Friday=4
                dates.append(current)
            current += timedelta(days=1)
        
        all_dates = pl.Series(date_col, dates)
    
    # Create base calendar
    calendar = pl.DataFrame({date_col: all_dates})
    
    # Join with data
    result = calendar.join(
        df.select([date_col] + value_cols),
        on=date_col,
        how="left"
    )
    
    # Forward fill
    if method == "forward_fill":
        result = result.sort(date_col)
        for col in value_cols:
            result = result.with_columns([
                pl.col(col).forward_fill()
            ])
    
    logger.debug(f"Aligned {len(df)} rows to {len(result)} daily observations")
    
    return result


def align_to_monthly(
    df: pl.DataFrame,
    date_col: str = "date",
    value_cols: list[str] | None = None,
    agg_method: str = "last",
) -> pl.DataFrame:
    """
    Aggregate daily data to monthly frequency.
    
    Args:
        df: Input DataFrame with daily data
        date_col: Name of date column
        value_cols: Columns to aggregate (default: all numeric)
        agg_method: Aggregation method ("last", "mean", "sum")
        
    Returns:
        DataFrame with monthly observations (month-end dates)
    """
    logger = get_logger()
    
    if value_cols is None:
        value_cols = [c for c in df.columns if c != date_col and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    
    # Add year-month column
    df = df.with_columns([
        pl.col(date_col).dt.strftime("%Y-%m").alias("_ym")
    ])
    
    # Aggregate
    if agg_method == "last":
        agg_exprs = [pl.col(c).last().alias(c) for c in value_cols]
    elif agg_method == "mean":
        agg_exprs = [pl.col(c).mean().alias(c) for c in value_cols]
    elif agg_method == "sum":
        agg_exprs = [pl.col(c).sum().alias(c) for c in value_cols]
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")
    
    result = (
        df
        .group_by("_ym")
        .agg([pl.col(date_col).max().alias(date_col)] + agg_exprs)
        .drop("_ym")
        .sort(date_col)
    )
    
    logger.debug(f"Aggregated {len(df)} daily rows to {len(result)} monthly observations")
    
    return result


def forward_fill_with_lag(
    df: pl.DataFrame,
    date_col: str = "date",
    value_cols: list[str] | None = None,
    lag_days: int = 30,
    target_dates: pl.Series | None = None,
) -> pl.DataFrame:
    """
    Forward-fill data with publication lag to avoid lookahead bias.
    
    For data like USREC (NBER recession indicator), there's a publication
    delay of ~30 days. This function shifts the effective date forward
    by lag_days before forward-filling.
    
    Example:
        USREC for March 2020 is published in late April 2020.
        With lag_days=30, March data is only available from late April.
    
    Args:
        df: Input DataFrame
        date_col: Name of date column
        value_cols: Columns to forward-fill
        lag_days: Publication lag in days (default: 30)
        target_dates: Target calendar dates
        
    Returns:
        DataFrame aligned to target_dates with lagged values
    """
    logger = get_logger()
    
    if value_cols is None:
        value_cols = [c for c in df.columns if c != date_col]
    
    # Shift dates forward by lag
    df_lagged = df.with_columns([
        (pl.col(date_col) + pl.duration(days=lag_days)).alias(date_col)
    ])
    
    logger.debug(f"Applied {lag_days}-day publication lag")
    
    # Now align to daily calendar
    return align_to_daily(
        df_lagged,
        date_col=date_col,
        value_cols=value_cols,
        target_dates=target_dates,
        method="forward_fill"
    )


def compute_ma200(
    df: pl.DataFrame,
    price_col: str = "sp500",
    date_col: str = "date",
    window: int = 200,
) -> pl.DataFrame:
    """
    Compute 200-day moving average for risk-off detection.
    
    Adds columns:
    - ma200: 200-day moving average of price
    - below_ma200: Binary indicator (price < MA200)
    
    Args:
        df: DataFrame with price data
        price_col: Name of price column
        date_col: Name of date column
        window: MA window (default: 200)
        
    Returns:
        DataFrame with MA200 columns added
    """
    df = df.sort(date_col)
    
    df = df.with_columns([
        pl.col(price_col)
        .rolling_mean(window_size=window, min_samples=window // 2)
        .alias("ma200"),
    ])
    
    df = df.with_columns([
        (pl.col(price_col) < pl.col("ma200")).cast(pl.Int32).alias("below_ma200"),
    ])
    
    return df


def create_target_shift(
    df: pl.DataFrame,
    value_col: str,
    date_col: str = "date",
    horizon: int = 1,
    transform: str | None = None,
) -> pl.DataFrame:
    """
    Create target variable with future shift (for supervised learning).
    
    Creates y_{t+h} aligned to date t (no lookahead).
    
    Args:
        df: Input DataFrame sorted by date
        value_col: Column to shift
        date_col: Date column
        horizon: Forward shift in periods (default: 1)
        transform: Optional transform ("log", "diff", "pct_change")
        
    Returns:
        DataFrame with target column added
    """
    df = df.sort(date_col)
    
    if transform == "log":
        # Log transform first, then shift
        df = df.with_columns([
            pl.col(value_col).log().alias(f"_log_{value_col}")
        ])
        df = df.with_columns([
            pl.col(f"_log_{value_col}").shift(-horizon).alias(f"y_{value_col}_log_t{horizon}")
        ])
        df = df.drop(f"_log_{value_col}")
        
    elif transform == "diff":
        # Future value minus current
        df = df.with_columns([
            (pl.col(value_col).shift(-horizon) - pl.col(value_col)).alias(f"y_{value_col}_diff_t{horizon}")
        ])
        
    elif transform == "pct_change":
        # Percent change
        df = df.with_columns([
            ((pl.col(value_col).shift(-horizon) - pl.col(value_col)) / pl.col(value_col))
            .alias(f"y_{value_col}_pct_t{horizon}")
        ])
        
    else:
        # Simple shift
        df = df.with_columns([
            pl.col(value_col).shift(-horizon).alias(f"y_{value_col}_t{horizon}")
        ])
    
    return df
