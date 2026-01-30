"""
Configuration dataclasses for proxy layers and enriched RL state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ProxyVIXConfig:
    """
    Configuration for VIX synthetic proxy (Proxy A).
    
    Multiple targets for different use cases:
    - Regression: Δlog(VIX) - noisy, hard to predict
    - Spike detection: VIX spike (Δlog > q90) - more useful for RL
    - Multi-horizon: t+1 (fast) and t+5 (macro) horizons
    
    For RL, spike detection is more valuable than exact prediction.
    """
    enabled: bool = True
    target_transform: Literal["log", "level", "diff", "both", "spike"] = "spike"
    horizon: int = 1  # t+1
    high_threshold: float = 25.0  # VIX > 25 = high volatility
    classification_enabled: bool = True  # Also train binary classifier
    
    # === SPIKE DETECTION (PRIMARY FOR RL) ===
    use_spike_detection: bool = True  # Train spike classifier
    spike_quantile: float = 0.90  # top 10% = spike
    spike_threshold: float | None = None  # Or explicit threshold (e.g., 0.05)
    
    # === MULTI-HORIZON ===
    use_multi_horizon: bool = True
    horizons: tuple[int, ...] = (1, 5)  # t+1 (fast shock), t+5 (macro trend)
    
    # === AUTOREGRESSIVE FEATURE (for inference without real VIX) ===
    use_autoregressive: bool = True  # Use pred_vix_* from previous day
    ar_feature_name: str = "f_pred_vix_spike_lag1"  # State carries forward
    
    # Enhanced stress features (computed in universe features)
    use_stress_features: bool = True
    realized_vol_window: int = 20  # rolling std of f_mkt_cluster
    jumpiness_window: int = 10  # rolling max |ret|
    corr_spike_window: int = 60  # compare avg_corr_20 vs avg_corr_60
    
    # Regularization
    ridge_alpha: float = 1.0
    logreg_C: float = 1.0  # For spike classifier
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyRatesConfig:
    """
    Configuration for rate oracle proxy (Proxy B).
    
    Predicts DGS10 change at longer horizons with categorical buckets.
    Includes leakage tests and drift monitoring.
    """
    enabled: bool = True
    horizon: int = 10  # t+10 days (more macro trend)
    classification_enabled: bool = True  # prob_rate_up
    
    # Categorical mode: predict down/flat/up instead of exact bps
    use_categorical: bool = True
    up_threshold_bps: float = 5.0  # > +5 bps = up
    down_threshold_bps: float = -5.0  # < -5 bps = down
    
    # Duration proxy features (added in universe features)
    use_duration_features: bool = True
    
    # === LEAKAGE TESTS ===
    run_leakage_test: bool = True  # Verify no forward info in features
    leakage_test_shuffle_target: bool = True  # Shuffle y, expect ~random
    
    # === DRIFT MONITORING ===
    monitor_drift: bool = True
    drift_window_months: int = 6  # Rolling window for monitoring
    drift_alert_threshold: float = 0.15  # Alert if perf drops > 15%
    
    # Regularization
    ridge_alpha: float = 1.0
    softmax_C: float = 1.0  # For categorical mode
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyRecessionConfig:
    """
    Configuration for recession/risk-off proxy (Proxy C).
    
    Three modes:
    - NBER: Uses USREC with publication lag (often trivial)
    - Market: Uses SP500 < MA200 (no lag needed)
    - Internal: Uses internal drawdown (always available, no lookahead)
    
    CRITICAL: Internal mode must use FUTURE window (t+1..t+H) for target,
    NOT a backward-looking window that overlaps with features.
    """
    enabled: bool = True
    mode: Literal["nber", "market", "internal", "both"] = "internal"  # Prefer internal
    
    # NBER settings
    usrec_publication_lag_days: int = 30
    
    # Market risk-off settings
    ma_window: int = 200
    
    # Internal risk-off settings (from cluster data only - no external)
    internal_dd_threshold: float = -0.05  # DD > 5% = risk-off
    internal_window: int = 60  # Look-back window for DD
    
    # === FUTURE WINDOW VALIDATION (CRITICAL) ===
    # The target MUST be computed from a FUTURE window, not past!
    # y[t] = any(DD[t+1:t+H] > threshold)
    use_future_window: bool = True  # Target looks forward (required!)
    future_horizon: int = 10  # Predict risk-off in next 10 days
    
    # === QUANTILE-BASED THRESHOLD ===
    # Instead of fixed threshold, use bottom X% of forward returns
    use_quantile_threshold: bool = True  # Recommended for stability
    risk_quantile: float = 0.10  # Bottom 10% = risk-off events
    
    # === FEATURE INDEPENDENCE CHECK ===
    # Features used CANNOT be the same calculation as target
    verify_feature_independence: bool = True
    
    # Continuous stress index (alternative to binary)
    use_stress_index: bool = True  # z(corr) + z(vol) - z(breadth)
    
    # Regularization
    logreg_C: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyFactorsConfig:
    """
    Configuration for factor monitor proxy (Proxy D).
    
    Two modes:
    - Multiclass: predict winning factor (requires more data)
    - Binary: predict MOM > 0 or MOM > HML (more robust with 54 samples)
    """
    enabled: bool = True
    factors: tuple[str, ...] = ("smb", "hml", "mom")
    include_market: bool = False  # Include mkt_rf as class
    frequency: Literal["monthly", "weekly"] = "monthly"
    
    # Binary mode (more robust with limited data)
    use_binary_mode: bool = True  # Predict MOM > 0 instead of multiclass
    binary_target: str = "mom"  # Which factor to predict > 0
    
    # Persistence baseline (strong baseline for factors)
    use_persistence_baseline: bool = True  # Last month's winner persists
    
    # Regularization
    softmax_C: float = 1.0
    
    def __post_init__(self) -> None:
        if isinstance(self.factors, list):
            object.__setattr__(self, "factors", tuple(self.factors))
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CVConfig:
    """
    Cross-validation configuration for proxy training.
    
    Uses time-series walk-forward CV:
    - Train up to time T
    - Validate T to T + val_months
    - Repeat with expanding window
    """
    n_folds: int = 5
    val_months: int = 6  # 6-month validation window
    min_train_months: int = 12  # Minimum training data
    gap_days: int = 1  # Gap between train and val to avoid leakage
    
    # Calibration settings
    enable_calibration: bool = True  # Temperature scaling / Platt
    calibration_method: Literal["platt", "temperature", "isotonic"] = "platt"
    
    # Uncertainty estimation
    compute_uncertainty: bool = True  # Store residual_std per fold
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UniverseFeaturesConfig:
    """
    Configuration for universe-level features (X internal).
    
    These features are computed from cluster timeseries and are
    available during both training and inference (no external data).
    
    Enhanced with stress features for VIX prediction:
    - realized_vol: rolling std of market return
    - jumpiness: rolling max |ret|
    - corr_spike: change in correlation vs longer window
    - skew_cs: cross-sectional skewness (crash indicator)
    """
    # Correlation features
    corr_window: int = 20
    corr_window_long: int = 60  # For corr_spike = corr_20 - corr_60
    
    # Volatility features
    vol_window: int = 20
    
    # Stress features (for VIX proxy)
    stress_window: int = 60
    realized_vol_window: int = 20  # rolling std of f_mkt_cluster
    jumpiness_window: int = 10  # rolling max |ret| of f_mkt_cluster
    compute_skew_cs: bool = True  # cross-sectional skewness
    
    # Breadth features
    breadth_threshold: float = 0.0  # ret > threshold counts as positive
    
    # PCA factors to include
    include_pca_factors: bool = True
    n_pca_factors: int = 10
    
    # Defensive vs aggressive spread (for rates proxy)
    defensive_vol_quantile: float = 0.1  # Bottom 10% vol = defensive
    aggressive_vol_quantile: float = 0.9  # Top 10% vol = aggressive
    
    # Low vol minus high vol spread (duration proxy for rates)
    compute_lowvol_highvol_spread: bool = True
    compute_momentum_spread: bool = True  # High mom - low mom clusters
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayersConfig:
    """
    Main configuration for proxy layers pipeline.
    
    Orchestrates:
    1. External data fetching (TRAIN only)
    2. Universe features computation
    3. Proxy model training (walk-forward CV)
    4. Prediction generation
    5. Enriched state assembly
    """
    run_id: str = "layers_v1"
    
    # Input runs
    clustering_run_id: str = ""
    cluster_set_id: str = ""  # Which cluster set to use
    
    # Date ranges
    train_start: str = "2020-01-01"
    train_end: str = "2024-12-31"
    
    # Proxy configs
    proxy_vix: ProxyVIXConfig = field(default_factory=ProxyVIXConfig)
    proxy_rates: ProxyRatesConfig = field(default_factory=ProxyRatesConfig)
    proxy_recession: ProxyRecessionConfig = field(default_factory=ProxyRecessionConfig)
    proxy_factors: ProxyFactorsConfig = field(default_factory=ProxyFactorsConfig)
    
    # CV config
    cv_config: CVConfig = field(default_factory=CVConfig)
    
    # Universe features config
    universe_features: UniverseFeaturesConfig = field(default_factory=UniverseFeaturesConfig)
    
    # External data config
    fred_api_key: str | None = None  # Or from env FRED_API_KEY
    external_cache_dir: Path = field(default_factory=lambda: Path("data/external"))
    
    # Output paths
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    models_dir: Path = field(default_factory=lambda: Path("data/cache/models"))
    reports_dir: Path = field(default_factory=lambda: Path("data/reports"))
    
    # Output filenames
    universe_features_name: str = "universe_features_daily_{cluster_set_id}.parquet"
    proxy_preds_name: str = "proxy_preds_{cluster_set_id}.parquet"
    enriched_features_name: str = "cluster_features_enriched_{cluster_set_id}.parquet"
    
    def __post_init__(self) -> None:
        # Convert string paths to Path
        if isinstance(self.external_cache_dir, str):
            self.external_cache_dir = Path(self.external_cache_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
        if isinstance(self.reports_dir, str):
            self.reports_dir = Path(self.reports_dir)
        
        # Convert nested dicts to dataclasses
        if isinstance(self.proxy_vix, dict):
            self.proxy_vix = ProxyVIXConfig(**self.proxy_vix)
        if isinstance(self.proxy_rates, dict):
            self.proxy_rates = ProxyRatesConfig(**self.proxy_rates)
        if isinstance(self.proxy_recession, dict):
            self.proxy_recession = ProxyRecessionConfig(**self.proxy_recession)
        if isinstance(self.proxy_factors, dict):
            self.proxy_factors = ProxyFactorsConfig(**self.proxy_factors)
        if isinstance(self.cv_config, dict):
            self.cv_config = CVConfig(**self.cv_config)
        if isinstance(self.universe_features, dict):
            self.universe_features = UniverseFeaturesConfig(**self.universe_features)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.clustering_run_id:
            raise ValueError("clustering_run_id is required")
        if not self.cluster_set_id:
            raise ValueError("cluster_set_id is required")
        
        # Check directories exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "clustering_run_id": self.clustering_run_id,
            "cluster_set_id": self.cluster_set_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "proxy_vix": self.proxy_vix.to_dict(),
            "proxy_rates": self.proxy_rates.to_dict(),
            "proxy_recession": self.proxy_recession.to_dict(),
            "proxy_factors": self.proxy_factors.to_dict(),
            "cv_config": self.cv_config.to_dict(),
            "universe_features": self.universe_features.to_dict(),
            "external_cache_dir": str(self.external_cache_dir),
            "cache_dir": str(self.cache_dir),
            "models_dir": str(self.models_dir),
            "reports_dir": str(self.reports_dir),
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
    def from_dict(cls, d: dict[str, Any]) -> "LayersConfig":
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "LayersConfig":
        """Load config from YAML file."""
        import yaml
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if "layers" in data:
            data = data["layers"]
        
        return cls.from_dict(data)
