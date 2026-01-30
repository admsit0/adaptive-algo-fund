"""
Artifact store and manifest management for AthenAI pipeline.

Handles:
- Versioned output directories (by run_id)
- Manifest JSON with full provenance
- Schema and stats tracking per artifact
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class ArtifactInfo:
    """Information about a single artifact (parquet file)."""
    name: str
    path: str
    row_count: int = 0
    col_count: int = 0
    schema: dict[str, str] = field(default_factory=dict)
    null_pcts: dict[str, float] = field(default_factory=dict)
    min_date: str | None = None
    max_date: str | None = None
    unique_algo_ids: int | None = None
    file_size_bytes: int = 0
    created_at: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArtifactInfo":
        return cls(**d)


@dataclass
class InputInfo:
    """Information about an input file."""
    path: str
    size_bytes: int
    mtime: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepTiming:
    """Timing information for a pipeline step."""
    step_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Manifest:
    """
    Pipeline run manifest with full provenance.
    
    Tracks:
    - run_id and config hash
    - input files with sizes and mtimes
    - output artifacts with schemas and stats
    - warnings and errors
    - timing per step
    """
    
    def __init__(
        self,
        run_id: str,
        config_hash: str,
        config: dict[str, Any],
    ):
        self.run_id = run_id
        self.config_hash = config_hash
        self.config = config
        self.created_at = datetime.now().isoformat()
        
        self.inputs: list[InputInfo] = []
        self.outputs: dict[str, ArtifactInfo] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.timings: list[StepTiming] = []
        self.total_duration_seconds: float = 0.0
    
    def add_input(self, path: Path) -> None:
        """Record an input file."""
        if path.exists():
            stat = path.stat()
            self.inputs.append(InputInfo(
                path=str(path),
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))
    
    def add_output(self, info: ArtifactInfo) -> None:
        """Record an output artifact."""
        self.outputs[info.name] = info
    
    def add_warning(self, msg: str) -> None:
        """Record a warning."""
        self.warnings.append(f"[{datetime.now().isoformat()}] {msg}")
    
    def add_error(self, msg: str) -> None:
        """Record an error."""
        self.errors.append(f"[{datetime.now().isoformat()}] {msg}")
    
    def add_timing(self, timing: StepTiming) -> None:
        """Record timing for a step."""
        self.timings.append(timing)
        self.total_duration_seconds += timing.duration_seconds
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "config": self.config,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "warnings": self.warnings,
            "errors": self.errors,
            "timings": [t.to_dict() for t in self.timings],
            "total_duration_seconds": self.total_duration_seconds,
        }
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class ArtifactStore:
    """
    Central store for pipeline artifacts.
    
    Manages:
    - Versioned output directories
    - Manifest generation and saving
    - Artifact info extraction from parquets
    
    Usage:
        store = ArtifactStore(root=Path("data/cache"), config=cfg)
        store.ensure_dirs()
        
        # After writing a parquet
        store.register_artifact("algos_panel", panel_path, lf)
        
        # At the end
        store.save_manifest()
    """
    
    def __init__(
        self,
        root: Path,
        config: Any,  # PreprocessConfig
        run_id: str | None = None,
    ):
        self.root = Path(root)
        self.config = config
        
        # Generate run_id if not provided
        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_hash = config.config_hash() if hasattr(config, "config_hash") else "unknown"
            self.run_id = f"{ts}_{config_hash}"
        else:
            self.run_id = run_id
        
        # Output directory for this run
        self.run_dir = self.root / self.run_id
        
        # Initialize manifest
        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        config_hash = config.config_hash() if hasattr(config, "config_hash") else ""
        self.manifest = Manifest(
            run_id=self.run_id,
            config_hash=config_hash,
            config=config_dict,
        )
        
        # Keep track of latest artifacts (for chaining steps)
        self._latest: dict[str, Path] = {}
    
    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "checks").mkdir(parents=True, exist_ok=True)
        
        # Reports and figures dirs from config
        if hasattr(self.config, "reports_dir"):
            self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.config, "figures_dir"):
            self.config.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def artifact_path(self, name: str) -> Path:
        """Get the path for an artifact in this run."""
        return self.run_dir / name
    
    def get_latest(self, name: str) -> Path | None:
        """Get the path to the latest version of an artifact."""
        return self._latest.get(name)
    
    def register_input(self, path: Path) -> None:
        """Register an input file for provenance tracking."""
        self.manifest.add_input(path)
    
    def register_artifact(
        self,
        name: str,
        path: Path,
        df_or_lf: pl.DataFrame | pl.LazyFrame | None = None,
    ) -> ArtifactInfo:
        """
        Register an output artifact and extract its stats.
        
        Args:
            name: Logical name (e.g., "algos_panel")
            path: Path to the parquet file
            df_or_lf: Optional DataFrame/LazyFrame to extract stats from
                      (if None, will read from path)
        """
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        
        # Get file stats
        stat = path.stat()
        
        # Read schema and stats
        if df_or_lf is None:
            lf = pl.scan_parquet(str(path))
        elif isinstance(df_or_lf, pl.LazyFrame):
            lf = df_or_lf
        else:
            lf = df_or_lf.lazy()
        
        schema = lf.collect_schema()
        schema_dict = {col: str(dtype) for col, dtype in schema.items()}
        
        # Collect basic stats (fast)
        df = lf.collect()
        row_count = len(df)
        col_count = len(df.columns)
        
        # Null percentages
        null_pcts = {}
        for col in df.columns:
            null_count = df[col].null_count()
            null_pcts[col] = round(null_count / row_count * 100, 2) if row_count > 0 else 0.0
        
        # Date range if "date" column exists
        min_date = None
        max_date = None
        if "date" in df.columns:
            dates = df["date"]
            if dates.dtype in (pl.Date, pl.Datetime):
                min_date = str(dates.min())
                max_date = str(dates.max())
        
        # Unique algo_ids if column exists
        unique_algo_ids = None
        if "algo_id" in df.columns:
            unique_algo_ids = df["algo_id"].n_unique()
        
        info = ArtifactInfo(
            name=name,
            path=str(path),
            row_count=row_count,
            col_count=col_count,
            schema=schema_dict,
            null_pcts=null_pcts,
            min_date=min_date,
            max_date=max_date,
            unique_algo_ids=unique_algo_ids,
            file_size_bytes=stat.st_size,
            created_at=datetime.now().isoformat(),
        )
        
        self.manifest.add_output(info)
        self._latest[name] = path
        
        return info
    
    def add_warning(self, msg: str) -> None:
        """Add a warning to the manifest."""
        self.manifest.add_warning(msg)
    
    def add_error(self, msg: str) -> None:
        """Add an error to the manifest."""
        self.manifest.add_error(msg)
    
    def add_timing(self, timing: StepTiming) -> None:
        """Add timing info for a step."""
        self.manifest.add_timing(timing)
    
    def save_manifest(self) -> Path:
        """Save the manifest to the run directory."""
        manifest_path = self.run_dir / "manifest.json"
        self.manifest.save(manifest_path)
        return manifest_path
    
    def summary(self) -> str:
        """Get a summary of the current run."""
        lines = [
            f"Run ID: {self.run_id}",
            f"Config Hash: {self.manifest.config_hash}",
            f"Output Dir: {self.run_dir}",
            f"Artifacts: {len(self.manifest.outputs)}",
            f"Warnings: {len(self.manifest.warnings)}",
            f"Errors: {len(self.manifest.errors)}",
        ]
        
        if self.manifest.outputs:
            lines.append("\nArtifacts:")
            for name, info in self.manifest.outputs.items():
                size_mb = info.file_size_bytes / (1024 * 1024)
                lines.append(f"  - {name}: {info.row_count:,} rows, {size_mb:.2f} MB")
        
        if self.manifest.warnings:
            lines.append(f"\nWarnings ({len(self.manifest.warnings)}):")
            for w in self.manifest.warnings[:5]:  # Show first 5
                lines.append(f"  - {w}")
            if len(self.manifest.warnings) > 5:
                lines.append(f"  ... and {len(self.manifest.warnings) - 5} more")
        
        if self.manifest.timings:
            lines.append(f"\nTimings (total: {self.manifest.total_duration_seconds:.2f}s):")
            for t in self.manifest.timings:
                lines.append(f"  - {t.step_name}: {t.duration_seconds:.2f}s")
        
        return "\n".join(lines)
