"""
Configuration dataclasses for factor analysis and clustering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class FactorConfig:
    """
    Configuration for factor time series construction.
    
    Attributes:
        use_good_universe: Whether to use only GOOD algos for factors
        pca_k: Number of PCA components to extract
        pca_date_sampling: Date sampling for PCA ("D" daily, "W" weekly)
        pca_max_dates: Max dates to use for PCA (controls memory/compute)
        standardize_returns: Whether to standardize returns before PCA
        missing_policy: How to handle missing returns ("zero", "drop_algo", "drop_date")
        asof_date: Optional cutoff date (YYYY-MM-DD)
        min_algos_pca: Minimum algos required for PCA
    """
    use_good_universe: bool = True
    pca_k: int = 10
    pca_date_sampling: Literal["D", "W", "M"] = "W"
    pca_max_dates: int = 600
    standardize_returns: bool = True
    missing_policy: Literal["zero", "drop_algo", "drop_date"] = "zero"
    asof_date: str | None = None
    min_algos_pca: int = 500
    
    # Output names
    factor_timeseries_name: str = "factor_timeseries.parquet"
    pca_model_name: str = "pca_model.npz"
    factor_ts_name: str = "factor_timeseries.parquet"  # alias
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureConfig:
    """
    Configuration for factor exposure computation.
    
    Attributes:
        method: Regression method ("ols", "ridge_ols")
        ridge_lambda: Regularization parameter for ridge
        min_obs: Minimum observations for valid exposure
        annualization: Trading days per year
    """
    method: Literal["ols", "ridge_ols"] = "ridge_ols"
    ridge_lambda: float = 1e-3
    min_obs: int = 120
    annualization: int = 252
    
    # Output names
    exposures_all_name: str = "algo_factor_exposures.parquet"
    exposures_good_name: str = "algo_factor_exposures_good.parquet"
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterSetConfig:
    """
    Configuration for a single cluster set.
    
    Attributes:
        cluster_set_id: Unique identifier (e.g., "behavioral_k100_v1")
        family: Clustering family type
        k: Number of clusters
        feature_source: Which features to use
        scaler: Scaling method
        algorithm: Clustering algorithm
        seed: Random seed for reproducibility
        min_cluster_size: Minimum size for warning
    """
    cluster_set_id: str
    family: Literal["behavioral", "correlation_embedding", "regime_specialists"] = "behavioral"
    k: int = 100
    feature_source: Literal["static_features", "pca_embedding", "regime_signature"] = "static_features"
    scaler: Literal["robust", "zscore", "none"] = "robust"
    algorithm: Literal["minibatch_kmeans", "kmeans", "gmm"] = "minibatch_kmeans"
    seed: int = 42
    min_cluster_size: int = 20
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterTimeseriesConfig:
    """
    Configuration for cluster time series construction.
    
    Attributes:
        weighting: How to weight algo returns ("equal", "vol_inverse")
        min_alive_per_cluster: Minimum algos alive for valid cluster day
        rolling_windows: Window sizes for rolling metrics
        vol_annualization: Trading days per year for vol annualization
        min_samples_ratio: Min samples ratio for rolling calcs
    """
    weighting: Literal["equal", "vol_inverse"] = "equal"
    min_alive_per_cluster: int = 5
    rolling_windows: tuple[int, ...] = (5, 10, 20, 60)
    vol_annualization: int = 252
    min_samples_ratio: float = 0.25
    
    def __post_init__(self) -> None:
        if isinstance(self.rolling_windows, list):
            object.__setattr__(self, "rolling_windows", tuple(self.rolling_windows))
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusteringConfig:
    """
    Main configuration for clustering pipeline.
    
    Attributes:
        preprocess_run_id: Run ID of preprocessing outputs
        personality_run_id: Run ID of personality outputs
        factor_config: Factor construction config
        exposure_config: Exposure computation config
        cluster_sets: List of cluster set configurations
        timeseries_config: Cluster timeseries config
        cache_dir: Cache directory
        reports_dir: Reports directory
        asof_date: Optional cutoff date
    """
    run_id: str = "clustering_v1"
    preprocess_run_id: str = ""
    personality_run_id: str = ""
    factor_config: FactorConfig = field(default_factory=FactorConfig)
    exposure_config: ExposureConfig = field(default_factory=ExposureConfig)
    cluster_sets: list[ClusterSetConfig] = field(default_factory=list)
    timeseries_config: ClusterTimeseriesConfig = field(default_factory=ClusterTimeseriesConfig)
    
    # Directories
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    reports_dir: Path = field(default_factory=lambda: Path("data/reports"))
    
    # Optional macro run (for regime features)
    macro_run_id: str | None = None
    
    def __post_init__(self) -> None:
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.reports_dir, str):
            self.reports_dir = Path(self.reports_dir)
        
        # Convert dicts to dataclasses if needed
        if isinstance(self.factor_config, dict):
            self.factor_config = FactorConfig(**self.factor_config)
        if isinstance(self.exposure_config, dict):
            self.exposure_config = ExposureConfig(**self.exposure_config)
        if isinstance(self.timeseries_config, dict):
            self.timeseries_config = ClusterTimeseriesConfig(**self.timeseries_config)
        
        # Convert cluster_sets dicts to ClusterSetConfig
        if self.cluster_sets and isinstance(self.cluster_sets[0], dict):
            self.cluster_sets = [ClusterSetConfig(**cs) for cs in self.cluster_sets]
        
        # Add default cluster sets if empty
        if not self.cluster_sets:
            self.cluster_sets = [
                ClusterSetConfig(
                    cluster_set_id="behavioral_k100_v1",
                    family="behavioral",
                    k=100,
                    feature_source="static_features",
                    scaler="robust",
                    algorithm="minibatch_kmeans",
                    seed=42,
                ),
                ClusterSetConfig(
                    cluster_set_id="corrpca_k60_v1",
                    family="correlation_embedding",
                    k=60,
                    feature_source="pca_embedding",
                    scaler="zscore",
                    algorithm="kmeans",
                    seed=42,
                ),
            ]
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.preprocess_run_id:
            raise ValueError("preprocess_run_id is required")
        if not self.personality_run_id:
            raise ValueError("personality_run_id is required")
        
        preprocess_dir = self.cache_dir / self.preprocess_run_id
        if not preprocess_dir.exists():
            raise ValueError(f"Preprocess run dir not found: {preprocess_dir}")
        
        personality_dir = self.cache_dir / self.personality_run_id
        if not personality_dir.exists():
            raise ValueError(f"Personality run dir not found: {personality_dir}")
        
        for cs in self.cluster_sets:
            if cs.k < 2:
                raise ValueError(f"k must be >= 2 for cluster set {cs.cluster_set_id}")
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "preprocess_run_id": self.preprocess_run_id,
            "personality_run_id": self.personality_run_id,
            "factor_config": self.factor_config.to_dict(),
            "exposure_config": self.exposure_config.to_dict(),
            "cluster_sets": [cs.to_dict() for cs in self.cluster_sets],
            "timeseries_config": self.timeseries_config.to_dict(),
            "cache_dir": str(self.cache_dir),
            "reports_dir": str(self.reports_dir),
            "macro_run_id": self.macro_run_id,
        }
    
    def config_hash(self) -> str:
        """Generate a short hash of this config for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def save(self, path: Path | str) -> None:
        """Save config to YAML file."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClusteringConfig":
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "ClusteringConfig":
        """Load config from YAML file."""
        import yaml
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if "clustering" in data:
            data = data["clustering"]
        
        return cls.from_dict(data)
