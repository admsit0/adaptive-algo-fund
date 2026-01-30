"""
Dataset inspection utilities.

Provides quick diagnostics without opening notebooks:
- Head/tail
- Null distributions
- Date ranges
- Cardinality
- Duplicates detection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class InspectionResult:
    """Results from dataset inspection."""
    name: str
    path: str
    row_count: int
    col_count: int
    schema: dict[str, str]
    null_counts: dict[str, int]
    null_pcts: dict[str, float]
    min_date: str | None
    max_date: str | None
    unique_algo_ids: int | None
    sample_head: list[dict[str, Any]]
    sample_tail: list[dict[str, Any]]
    duplicates: list[dict[str, Any]] | None
    numeric_stats: dict[str, dict[str, float]] | None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "schema": self.schema,
            "null_counts": self.null_counts,
            "null_pcts": self.null_pcts,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "unique_algo_ids": self.unique_algo_ids,
            "sample_head": self.sample_head,
            "sample_tail": self.sample_tail,
            "duplicates": self.duplicates,
            "numeric_stats": self.numeric_stats,
        }
    
    def summary(self) -> str:
        """Get a text summary."""
        lines = [
            f"Dataset: {self.name}",
            f"Path: {self.path}",
            f"Rows: {self.row_count:,}",
            f"Columns: {self.col_count}",
        ]
        
        if self.min_date and self.max_date:
            lines.append(f"Date range: {self.min_date} to {self.max_date}")
        
        if self.unique_algo_ids:
            lines.append(f"Unique algo_ids: {self.unique_algo_ids:,}")
        
        # Null summary
        null_cols = {k: v for k, v in self.null_pcts.items() if v > 0}
        if null_cols:
            lines.append(f"\nColumns with nulls ({len(null_cols)}):")
            for col, pct in sorted(null_cols.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {col}: {pct:.1f}%")
        else:
            lines.append("\nNo null values")
        
        # Schema
        lines.append("\nSchema:")
        for col, dtype in self.schema.items():
            lines.append(f"  {col}: {dtype}")
        
        return "\n".join(lines)


class DatasetInspector:
    """
    Inspect parquet datasets for quick diagnostics.
    
    Usage:
        inspector = DatasetInspector()
        result = inspector.inspect(Path("data/cache/algos_panel.parquet"))
        print(result.summary())
    """
    
    def __init__(self, sample_size: int = 5):
        self.sample_size = sample_size
    
    def inspect(
        self,
        path: Path,
        name: str | None = None,
        key_cols: list[str] | None = None,
    ) -> InspectionResult:
        """
        Inspect a parquet file.
        
        Args:
            path: Path to parquet file
            name: Optional name (defaults to filename)
            key_cols: Columns to check for duplicates
        
        Returns:
            InspectionResult with all diagnostics
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        name = name or path.stem
        
        # Read data
        df = pl.read_parquet(str(path))
        
        # Schema
        schema = {col: str(dtype) for col, dtype in df.schema.items()}
        
        # Null counts
        null_counts = {col: df[col].null_count() for col in df.columns}
        null_pcts = {
            col: count / len(df) * 100 if len(df) > 0 else 0
            for col, count in null_counts.items()
        }
        
        # Date range
        min_date = None
        max_date = None
        if "date" in df.columns:
            dates = df["date"]
            if dates.dtype in (pl.Date, pl.Datetime):
                min_date = str(dates.min())
                max_date = str(dates.max())
        
        # Unique algo_ids
        unique_algo_ids = None
        if "algo_id" in df.columns:
            unique_algo_ids = df["algo_id"].n_unique()
        
        # Samples
        sample_head = df.head(self.sample_size).to_dicts()
        sample_tail = df.tail(self.sample_size).to_dicts()
        
        # Duplicates
        duplicates = None
        if key_cols:
            existing_keys = [c for c in key_cols if c in df.columns]
            if existing_keys:
                dups = (
                    df.group_by(existing_keys)
                    .len()
                    .filter(pl.col("len") > 1)
                    .head(10)
                )
                if dups.height > 0:
                    duplicates = dups.to_dicts()
        
        # Numeric stats
        numeric_cols = [
            col for col, dtype in df.schema.items()
            if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]
        numeric_stats = {}
        for col in numeric_cols[:10]:  # Limit to first 10
            stats = df.select([
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).median().alias("median"),
                pl.col(col).std().alias("std"),
            ]).to_dicts()[0]
            numeric_stats[col] = stats
        
        return InspectionResult(
            name=name,
            path=str(path),
            row_count=len(df),
            col_count=len(df.columns),
            schema=schema,
            null_counts=null_counts,
            null_pcts=null_pcts,
            min_date=min_date,
            max_date=max_date,
            unique_algo_ids=unique_algo_ids,
            sample_head=sample_head,
            sample_tail=sample_tail,
            duplicates=duplicates,
            numeric_stats=numeric_stats,
        )
    
    def compare(
        self,
        path1: Path,
        path2: Path,
    ) -> dict[str, Any]:
        """
        Compare two parquet files.
        
        Useful for verifying that refactored code produces same outputs.
        """
        r1 = self.inspect(path1)
        r2 = self.inspect(path2)
        
        differences = {}
        
        if r1.row_count != r2.row_count:
            differences["row_count"] = {
                "file1": r1.row_count,
                "file2": r2.row_count,
            }
        
        if r1.schema != r2.schema:
            differences["schema"] = {
                "file1": r1.schema,
                "file2": r2.schema,
            }
        
        if r1.min_date != r2.min_date or r1.max_date != r2.max_date:
            differences["date_range"] = {
                "file1": (r1.min_date, r1.max_date),
                "file2": (r2.min_date, r2.max_date),
            }
        
        if r1.unique_algo_ids != r2.unique_algo_ids:
            differences["unique_algo_ids"] = {
                "file1": r1.unique_algo_ids,
                "file2": r2.unique_algo_ids,
            }
        
        return {
            "identical": len(differences) == 0,
            "differences": differences,
        }
    
    def inspect_directory(self, directory: Path) -> list[InspectionResult]:
        """Inspect all parquet files in a directory."""
        results = []
        for path in sorted(directory.glob("*.parquet")):
            try:
                results.append(self.inspect(path))
            except Exception as e:
                print(f"Error inspecting {path}: {e}")
        return results
