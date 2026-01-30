"""
Configuration dataclasses for AthenAI pipeline.

All configs are frozen dataclasses with sensible defaults.
Can be loaded from YAML via `from_yaml()` class method.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""
    pass


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for the preprocessing pipeline.
    
    Attributes:
        root_dir: Path to datos_competicion/ folder
        algos_subdir: Subfolder containing algorithm CSVs
        cache_dir: Where to write parquets and artifacts
        reports_dir: Where to write HTML/MD reports
        figures_dir: Where to write plots
        
        dt_candidates: Column name candidates for datetime
        close_candidates: Column name candidates for close price
        dt_format_candidates: Datetime parsing formats to try
        
        min_obs: Minimum observations required for "good" universe
        min_coverage: Minimum coverage ratio (n_obs / span_days)
        constant_close_std_eps: Threshold to detect constant close
        max_abs_ret_clip: Max absolute return for clipping
        
        feature_windows: Rolling window sizes for features
        annualization_factor: Trading days per year
        
        panel_name: Output filename for panel parquet
        meta_name: Output filename for meta parquet
        meta_good_name: Output filename for filtered meta
        features_name: Output filename for features
        features_good_name: Output filename for filtered features
        alive_intervals_name: Output filename for alive intervals
        
        benchmark_*: Output filenames for benchmark data
    """
    # Directories
    root_dir: Path = field(default_factory=lambda: Path("data/raw/datos_competicion"))
    algos_subdir: str = "algoritmos"
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    reports_dir: Path = field(default_factory=lambda: Path("data/reports"))
    figures_dir: Path = field(default_factory=lambda: Path("data/figures"))
    
    # Column detection (robust to variations)
    dt_candidates: tuple[str, ...] = ("datetime", "date", "timestamp", "time")
    close_candidates: tuple[str, ...] = ("close", "Close", "c", "price", "last")
    
    # Datetime parsing
    dt_format_candidates: tuple[str, ...] = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
    )
    
    # Quality filters
    min_obs: int = 60
    min_coverage: float = 0.70
    constant_close_std_eps: float = 1e-8
    max_abs_ret_clip: float = 0.50
    
    # Feature engineering
    feature_windows: tuple[int, ...] = (20, 60, 120)
    annualization_factor: int = 252
    
    # Output filenames
    panel_name: str = "algos_panel.parquet"
    meta_name: str = "algos_meta.parquet"
    meta_good_name: str = "algos_meta_good.parquet"
    features_name: str = "algos_features.parquet"
    features_good_name: str = "algos_features_good.parquet"
    alive_intervals_name: str = "alive_intervals.parquet"
    
    benchmark_trades_name: str = "benchmark_trades_clean.parquet"
    benchmark_monthly_name: str = "benchmark_monthly_clean.parquet"
    benchmark_yearly_name: str = "benchmark_yearly_clean.parquet"
    benchmark_monthly_stats_name: str = "benchmark_monthly_stats.parquet"
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        # Use object.__setattr__ because dataclass is frozen
        if isinstance(self.root_dir, str):
            object.__setattr__(self, "root_dir", Path(self.root_dir))
        if isinstance(self.cache_dir, str):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if isinstance(self.reports_dir, str):
            object.__setattr__(self, "reports_dir", Path(self.reports_dir))
        if isinstance(self.figures_dir, str):
            object.__setattr__(self, "figures_dir", Path(self.figures_dir))
        
        # Convert lists to tuples for hashability
        if isinstance(self.dt_candidates, list):
            object.__setattr__(self, "dt_candidates", tuple(self.dt_candidates))
        if isinstance(self.close_candidates, list):
            object.__setattr__(self, "close_candidates", tuple(self.close_candidates))
        if isinstance(self.dt_format_candidates, list):
            object.__setattr__(self, "dt_format_candidates", tuple(self.dt_format_candidates))
        if isinstance(self.feature_windows, list):
            object.__setattr__(self, "feature_windows", tuple(self.feature_windows))
    
    @property
    def algos_dir(self) -> Path:
        """Path to the algorithms CSV folder."""
        return self.root_dir / self.algos_subdir
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.root_dir.exists():
            raise ConfigError(f"root_dir does not exist: {self.root_dir}")
        if not self.algos_dir.exists():
            raise ConfigError(f"algos_dir does not exist: {self.algos_dir}")
        if self.min_obs < 1:
            raise ConfigError(f"min_obs must be >= 1, got {self.min_obs}")
        if not (0 < self.min_coverage <= 1):
            raise ConfigError(f"min_coverage must be in (0, 1], got {self.min_coverage}")
        if self.max_abs_ret_clip <= 0:
            raise ConfigError(f"max_abs_ret_clip must be > 0, got {self.max_abs_ret_clip}")
        if not self.feature_windows:
            raise ConfigError("feature_windows cannot be empty")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (with paths as strings)."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
    
    def config_hash(self) -> str:
        """Generate a short hash of this config for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreprocessConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "PreprocessConfig":
        """Load config from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Handle nested structure (e.g., preprocess: {...})
        if "preprocess" in data:
            data = data["preprocess"]
        
        return cls.from_dict(data)


def load_config(path: Path | str) -> PreprocessConfig:
    """Load PreprocessConfig from YAML file."""
    return PreprocessConfig.from_yaml(path)
