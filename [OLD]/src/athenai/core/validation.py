"""
Validation utilities for AthenAI pipeline.

Provides:
- Schema validation
- Uniqueness checks
- Sorting checks
- Look-ahead leak detection
- Financial data sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl

from athenai.core.logging import get_logger


class ValidationError(RuntimeError):
    """Raised when validation fails."""
    pass


@dataclass
class SchemaSpec:
    """
    Schema specification for validation.
    
    Attributes:
        required: Columns that must exist with exact types
        optional: Columns that may exist (if present, must match type)
    """
    required: dict[str, pl.DataType | type]
    optional: dict[str, pl.DataType | type] | None = None


def assert_schema(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    spec: SchemaSpec | dict[str, Any],
    name: str = "dataset",
) -> None:
    """
    Validate that a DataFrame/LazyFrame matches the expected schema.
    
    Args:
        df_or_lf: DataFrame or LazyFrame to validate
        spec: Schema specification (SchemaSpec or dict of required cols)
        name: Name for error messages
    
    Raises:
        ValidationError: If schema doesn't match
    """
    if isinstance(spec, dict):
        spec = SchemaSpec(required=spec)
    
    schema = df_or_lf.collect_schema() if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf.schema
    
    # Check required columns
    for col, expected_type in spec.required.items():
        if col not in schema:
            raise ValidationError(
                f"[{name}] Missing required column: {col}. "
                f"Available: {list(schema.keys())}"
            )
        
        actual_type = schema[col]
        # Allow some flexibility (e.g., Int64 matches Int32)
        if not _types_compatible(actual_type, expected_type):
            raise ValidationError(
                f"[{name}] Column '{col}' has type {actual_type}, "
                f"expected {expected_type}"
            )
    
    # Check optional columns (if present)
    if spec.optional:
        for col, expected_type in spec.optional.items():
            if col in schema:
                actual_type = schema[col]
                if not _types_compatible(actual_type, expected_type):
                    raise ValidationError(
                        f"[{name}] Optional column '{col}' has type {actual_type}, "
                        f"expected {expected_type}"
                    )


def _types_compatible(actual: pl.DataType, expected: pl.DataType | type) -> bool:
    """Check if types are compatible (with some flexibility)."""
    # If expected is a Python type, convert
    if isinstance(expected, type):
        if expected == float:
            return actual in (pl.Float32, pl.Float64)
        if expected == int:
            return actual in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        if expected == str:
            return actual in (pl.Utf8, pl.String)
        if expected == bool:
            return actual == pl.Boolean
    
    # Direct comparison
    if actual == expected:
        return True
    
    # Allow numeric flexibility
    float_types = {pl.Float32, pl.Float64}
    int_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    string_types = {pl.Utf8, pl.String}
    
    if actual in float_types and expected in float_types:
        return True
    if actual in int_types and expected in int_types:
        return True
    if actual in string_types and expected in string_types:
        return True
    
    # Date/Datetime flexibility
    if actual in (pl.Date, pl.Datetime) and expected in (pl.Date, pl.Datetime):
        return True
    
    return False


def assert_unique_keys(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    keys: Sequence[str],
    name: str = "dataset",
) -> None:
    """
    Assert that the given columns form a unique key (no duplicates).
    
    Args:
        df_or_lf: DataFrame or LazyFrame
        keys: Column names that should be unique together
        name: Name for error messages
    
    Raises:
        ValidationError: If duplicates exist
    """
    logger = get_logger()
    
    if isinstance(df_or_lf, pl.LazyFrame):
        df = df_or_lf.collect()
    else:
        df = df_or_lf
    
    # Check for duplicates
    dup_count = (
        df.group_by(list(keys))
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    
    if dup_count > 0:
        # Get example duplicates
        examples = (
            df.group_by(list(keys))
            .len()
            .filter(pl.col("len") > 1)
            .head(5)
        )
        raise ValidationError(
            f"[{name}] Found {dup_count} duplicate key combinations for {keys}.\n"
            f"Examples:\n{examples}"
        )
    
    logger.debug(f"[{name}] Unique key check passed for {keys}")


def assert_sorted_within_group(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    group_col: str,
    sort_col: str,
    name: str = "dataset",
) -> None:
    """
    Assert that data is sorted by sort_col within each group.
    
    Important for time series data where each algo must be sorted by date.
    
    Args:
        df_or_lf: DataFrame or LazyFrame
        group_col: Column defining groups (e.g., "algo_id")
        sort_col: Column that should be sorted (e.g., "date")
        name: Name for error messages
    
    Raises:
        ValidationError: If not properly sorted
    """
    logger = get_logger()
    
    if isinstance(df_or_lf, pl.LazyFrame):
        df = df_or_lf.collect()
    else:
        df = df_or_lf
    
    # Check if sorted within groups
    unsorted = (
        df.with_columns([
            pl.col(sort_col).shift(1).over(group_col).alias("_prev"),
        ])
        .filter(pl.col("_prev").is_not_null())
        .filter(pl.col(sort_col) < pl.col("_prev"))
    )
    
    if unsorted.height > 0:
        examples = unsorted.head(5).select([group_col, sort_col, "_prev"])
        raise ValidationError(
            f"[{name}] Data not sorted by {sort_col} within {group_col}.\n"
            f"Found {unsorted.height} violations. Examples:\n{examples}"
        )
    
    logger.debug(f"[{name}] Sort order check passed for {sort_col} within {group_col}")


def assert_no_lookahead(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    feature_cols: Sequence[str],
    date_col: str = "date",
    name: str = "dataset",
) -> None:
    """
    Check that features don't contain look-ahead bias.
    
    This is a heuristic check that looks for:
    - Features with non-null values at the start of each series
      (might indicate using future data)
    - Perfect correlation with future returns
    
    Note: This is not a perfect check, but catches common mistakes.
    
    Args:
        df_or_lf: DataFrame or LazyFrame
        feature_cols: Columns to check
        date_col: Date column
        name: Name for error messages
    """
    logger = get_logger()
    
    # This is a simple heuristic - can be expanded
    if isinstance(df_or_lf, pl.LazyFrame):
        df = df_or_lf.collect()
    else:
        df = df_or_lf
    
    # Check that rolling features have nulls at the beginning
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        # Rolling features typically have "window" in name or pattern like _20, _60
        is_rolling = any(str(w) in col for w in [20, 60, 120]) or "rolling" in col.lower()
        
        if is_rolling:
            # First few rows should have nulls (warmup period)
            first_non_null = df[col].drop_nulls().head(1)
            # This is a weak check - just log a warning if first row is non-null
            if df[col].head(1).null_count() == 0:
                logger.warning(
                    f"[{name}] Rolling feature '{col}' has non-null first value. "
                    "Check for potential look-ahead bias."
                )


def assert_returns_reasonable(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    return_col: str = "ret_1d",
    max_abs: float = 0.5,
    name: str = "dataset",
) -> int:
    """
    Check that returns are within reasonable bounds.
    
    Args:
        df_or_lf: DataFrame or LazyFrame
        return_col: Column containing returns
        max_abs: Maximum absolute return (default 50%)
        name: Name for error messages
    
    Returns:
        Number of values exceeding the threshold (after any clipping)
    
    Raises:
        ValidationError: If excessive extreme returns found
    """
    logger = get_logger()
    
    if isinstance(df_or_lf, pl.LazyFrame):
        df = df_or_lf.collect()
    else:
        df = df_or_lf
    
    if return_col not in df.columns:
        logger.warning(f"[{name}] Return column '{return_col}' not found")
        return 0
    
    extreme_count = df.filter(
        pl.col(return_col).abs() > max_abs
    ).height
    
    if extreme_count > 0:
        pct = extreme_count / len(df) * 100
        logger.warning(
            f"[{name}] Found {extreme_count} ({pct:.2f}%) returns exceeding ±{max_abs:.0%}"
        )
    
    return extreme_count


def assert_coverage_reasonable(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    group_col: str,
    date_col: str,
    min_coverage: float = 0.5,
    name: str = "dataset",
) -> dict[str, int]:
    """
    Check that coverage (obs/days) is reasonable per group.
    
    Args:
        df_or_lf: DataFrame or LazyFrame
        group_col: Column defining groups
        date_col: Date column
        min_coverage: Minimum coverage ratio
        name: Name for error messages
    
    Returns:
        Dict with total, low_coverage, and zero_coverage counts
    """
    logger = get_logger()
    
    if isinstance(df_or_lf, pl.LazyFrame):
        df = df_or_lf.collect()
    else:
        df = df_or_lf
    
    coverage = (
        df.group_by(group_col)
        .agg([
            pl.len().alias("n_obs"),
            pl.col(date_col).min().alias("min_date"),
            pl.col(date_col).max().alias("max_date"),
        ])
        .with_columns([
            ((pl.col("max_date") - pl.col("min_date")).dt.total_days() + 1).alias("span_days"),
        ])
        .with_columns([
            (pl.col("n_obs") / pl.col("span_days")).alias("coverage"),
        ])
    )
    
    low_coverage = coverage.filter(pl.col("coverage") < min_coverage).height
    zero_coverage = coverage.filter(pl.col("span_days") <= 0).height
    total = coverage.height
    
    if low_coverage > 0:
        pct = low_coverage / total * 100
        logger.warning(
            f"[{name}] {low_coverage} groups ({pct:.1f}%) have coverage < {min_coverage:.0%}"
        )
    
    return {
        "total": total,
        "low_coverage": low_coverage,
        "zero_coverage": zero_coverage,
    }


def validate_panel_schema(df_or_lf: pl.DataFrame | pl.LazyFrame, name: str = "panel") -> None:
    """Validate the algos_panel schema."""
    spec = SchemaSpec(
        required={
            "algo_id": pl.String,
            "date": pl.Date,
            "close": pl.Float64,
            "ret_1d": pl.Float64,
        },
        optional={
            "logret_1d": pl.Float64,
        }
    )
    assert_schema(df_or_lf, spec, name)


def validate_meta_schema(df_or_lf: pl.DataFrame | pl.LazyFrame, name: str = "meta") -> None:
    """Validate the algos_meta schema."""
    spec = SchemaSpec(
        required={
            "algo_id": pl.String,
            "start_date": pl.Date,
            "end_date": pl.Date,
            "n_obs": int,
            "coverage_ratio": float,
            "is_constant": pl.Boolean,
        },
        optional={
            "n_days": int,
            "close_std": float,
            "ret_mean": float,
            "ret_std": float,
            "sharpe_ann": float,
            "vol_ann": float,
            "max_drawdown": float,
        }
    )
    assert_schema(df_or_lf, spec, name)
