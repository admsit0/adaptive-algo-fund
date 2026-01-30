"""
I/O utilities for AthenAI pipeline.

Provides:
- Atomic parquet writes
- Safe file operations
- Column detection helpers
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from athenai.core.logging import get_logger


class IOError(RuntimeError):
    """Raised on I/O errors."""
    pass


def pick_first_existing(cols: Iterable[str], candidates: Sequence[str]) -> str:
    """
    Find the first column name from candidates that exists in cols.
    
    Args:
        cols: Available column names
        candidates: Ordered list of candidate names to try
    
    Returns:
        First matching column name
    
    Raises:
        IOError: If no candidate found
    """
    col_set = set(cols)
    for c in candidates:
        if c in col_set:
            return c
    raise IOError(
        f"No column found among {candidates}. Available: {sorted(list(cols)[:30])}..."
    )


def safe_write_parquet(
    lf: pl.LazyFrame,
    path: Path,
    overwrite: bool = False,
    compression: str = "zstd",
) -> Path:
    """
    Write a LazyFrame to parquet atomically.
    
    Uses a temp file + rename to ensure partial writes don't corrupt data.
    
    Args:
        lf: LazyFrame to write
        path: Output path
        overwrite: Whether to overwrite existing file
        compression: Compression codec
    
    Returns:
        Path to written file
    """
    logger = get_logger()
    path = Path(path)
    
    if path.exists() and not overwrite:
        logger.debug(f"Skipping write (exists): {path}")
        return path
    
    # Ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first, then rename (atomic on most filesystems)
    temp_path = path.with_suffix(".parquet.tmp")
    
    try:
        lf.sink_parquet(str(temp_path), compression=compression, statistics=True)
        
        # Rename to final path
        if path.exists():
            path.unlink()
        shutil.move(str(temp_path), str(path))
        
        logger.debug(f"Written: {path}")
        return path
        
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to write {path}: {e}") from e


def safe_read_parquet(path: Path) -> pl.LazyFrame:
    """
    Read a parquet file as LazyFrame.
    
    Args:
        path: Path to parquet file
    
    Returns:
        LazyFrame
    
    Raises:
        IOError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise IOError(f"File not found: {path}")
    
    return pl.scan_parquet(str(path))


def collect_sample(lf: pl.LazyFrame, n: int = 1000) -> pl.DataFrame:
    """
    Collect a sample from a LazyFrame for inspection.
    
    Args:
        lf: LazyFrame
        n: Number of rows
    
    Returns:
        DataFrame sample
    """
    return lf.head(n).collect()


def count_rows(path: Path) -> int:
    """Count rows in a parquet file without loading all data."""
    return pl.scan_parquet(str(path)).select(pl.len()).collect().item()


def get_schema(path: Path) -> dict[str, pl.DataType]:
    """Get schema of a parquet file."""
    return pl.scan_parquet(str(path)).collect_schema()


def list_parquet_files(directory: Path, pattern: str = "*.parquet") -> list[Path]:
    """List all parquet files in a directory."""
    return sorted(directory.glob(pattern))


def build_dt_expr(colname: str, format_candidates: Sequence[str]) -> pl.Expr:
    """
    Build a datetime parsing expression that tries multiple formats.
    
    Uses coalesce to return the first successful parse.
    
    Args:
        colname: Column name to parse
        format_candidates: Datetime formats to try (in order)
    
    Returns:
        Polars expression that parses datetime
    """
    exprs = [
        pl.col(colname).str.strptime(pl.Datetime, format=fmt, strict=False)
        for fmt in format_candidates
    ]
    return pl.coalesce(exprs).alias("dt")
