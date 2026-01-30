"""
Dataset builders for proxy model training.

Builds:
- Universe features (X): Internal features from clusters
- Target datasets (X, y): Aligned features and targets
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from athenai.core.logging import get_logger

if TYPE_CHECKING:
    from athenai.layers.config import LayersConfig, UniverseFeaturesConfig


def build_universe_features(
    cluster_ts: pl.DataFrame,
    cluster_features: pl.DataFrame,
    factor_ts: pl.DataFrame | None = None,
    cfg: "UniverseFeaturesConfig" | None = None,
) -> pl.DataFrame:
    """
    Build universe-level features from cluster timeseries.
    
    These features aggregate cluster-level data to a single row per date,
    capturing market-wide dynamics that can predict external indicators.
    
    Features built:
    - f_mkt_cluster: Mean return of clusters (market return proxy)
    - breadth_pos_ret: % of clusters with positive return
    - dispersion_ret_cs: Cross-sectional std of returns
    - avg_vol_20: Mean cluster volatility
    - tail_q05_cs: 5th percentile of returns (left tail)
    - stress_dd_min_60: Worst drawdown across clusters
    - avg_corr_20: Average off-diagonal correlation
    - defensive_vs_aggressive_spread: Return spread
    
    Args:
        cluster_ts: cluster_timeseries DataFrame with columns:
            date, cluster_id, ret_ew, vol_ew_20, ...
        cluster_features: cluster_features_daily DataFrame
        factor_ts: Optional factor_timeseries for PCA features
        cfg: Configuration for feature computation
        
    Returns:
        DataFrame with columns: date, f_mkt_cluster, breadth_pos_ret, ...
    """
    logger = get_logger()
    
    if cfg is None:
        from athenai.layers.config import UniverseFeaturesConfig
        cfg = UniverseFeaturesConfig()
    
    logger.info("Building universe features from cluster timeseries...")
    
    # Get all unique dates
    all_dates = cluster_ts["date"].unique().sort()
    logger.info(f"  Processing {len(all_dates)} dates")
    
    # -----------------------------------------------------
    # Feature 1: f_mkt_cluster (mean cluster return)
    # -----------------------------------------------------
    f_mkt = (
        cluster_ts
        .group_by("date")
        .agg([
            pl.col("ret_ew").mean().alias("f_mkt_cluster"),
        ])
    )
    
    # -----------------------------------------------------
    # Feature 2: breadth_pos_ret (% clusters with ret > 0)
    # -----------------------------------------------------
    breadth = (
        cluster_ts
        .group_by("date")
        .agg([
            (pl.col("ret_ew") > cfg.breadth_threshold).mean().alias("breadth_pos_ret"),
        ])
    )
    
    # -----------------------------------------------------
    # Feature 3: dispersion_ret_cs (cross-sectional return std)
    # -----------------------------------------------------
    dispersion = (
        cluster_ts
        .group_by("date")
        .agg([
            pl.col("ret_ew").std().alias("dispersion_ret_cs"),
        ])
    )
    
    # -----------------------------------------------------
    # Feature 4: avg_vol_20 (mean cluster volatility)
    # -----------------------------------------------------
    avg_vol = (
        cluster_ts
        .group_by("date")
        .agg([
            pl.col("vol_ew_20").mean().alias("avg_vol_20"),
        ])
    )
    
    # -----------------------------------------------------
    # Feature 5: tail_q05_cs (5th percentile of returns)
    # -----------------------------------------------------
    tail = (
        cluster_ts
        .group_by("date")
        .agg([
            pl.col("ret_ew").quantile(0.05).alias("tail_q05_cs"),
            pl.col("ret_ew").quantile(0.95).alias("tail_q95_cs"),
        ])
    )
    
    # -----------------------------------------------------
    # Feature 6: stress_dd_min_60 (worst drawdown proxy)
    # -----------------------------------------------------
    # Use rolling sum of returns as equity proxy
    # Compute per cluster first, then take min across clusters
    
    cluster_ts_sorted = cluster_ts.sort(["cluster_id", "date"])
    
    # Add cumulative return (equity proxy) per cluster
    cluster_ts_with_cum = cluster_ts_sorted.with_columns([
        pl.col("ret_ew")
        .cum_sum()
        .over("cluster_id")
        .alias("_cum_ret"),
    ])
    
    # Rolling max (high water mark)
    cluster_ts_with_dd = cluster_ts_with_cum.with_columns([
        pl.col("_cum_ret")
        .rolling_max(window_size=cfg.stress_window, min_samples=10)
        .over("cluster_id")
        .alias("_hwm"),
    ])
    
    # Drawdown = current - hwm
    cluster_ts_with_dd = cluster_ts_with_dd.with_columns([
        (pl.col("_cum_ret") - pl.col("_hwm")).alias("_dd"),
    ])
    
    # Min drawdown across clusters per date
    stress_dd = (
        cluster_ts_with_dd
        .group_by("date")
        .agg([
            pl.col("_dd").min().alias("stress_dd_min_60"),
        ])
    )
    
    # -----------------------------------------------------
    # Feature 7: avg_corr_20 (average return correlation)
    # -----------------------------------------------------
    avg_corr = _compute_rolling_correlation(cluster_ts, cfg.corr_window)

    # -----------------------------------------------------
    # Feature 8: defensive_vs_aggressive_spread
    # -----------------------------------------------------
    # Defensive = low vol clusters, Aggressive = high vol clusters
    spread = _compute_defensive_aggressive_spread(
        cluster_ts,
        cfg.defensive_vol_quantile,
        cfg.aggressive_vol_quantile
    )

    # -----------------------------------------------------
    # ENHANCED STRESS FEATURES (for VIX proxy)
    # -----------------------------------------------------
    
    # Feature 9: skew_cs (cross-sectional skewness)
    # During crashes, returns become more negatively skewed
    if getattr(cfg, 'compute_skew_cs', True):
        skew_cs = _compute_cross_sectional_skew(cluster_ts)
    else:
        skew_cs = None
    
    # Feature 10: avg_corr_60 for corr_spike
    corr_long_window = getattr(cfg, 'corr_window_long', 60)
    avg_corr_60 = _compute_rolling_correlation(cluster_ts, corr_long_window)
    avg_corr_60 = avg_corr_60.rename({"avg_corr_20": "avg_corr_60"})
    
    # Feature 11-12: lowvol_highvol_spread and momentum_spread
    if getattr(cfg, 'compute_lowvol_highvol_spread', True):
        lvhv_spread = _compute_lowvol_highvol_spread(cluster_ts)
    else:
        lvhv_spread = None
    
    if getattr(cfg, 'compute_momentum_spread', True):
        mom_spread = _compute_momentum_spread(cluster_ts)
    else:
        mom_spread = None

    # -----------------------------------------------------
    # Join all features
    # -----------------------------------------------------
    result = (
        pl.DataFrame({"date": all_dates})
        .join(f_mkt, on="date", how="left")
        .join(breadth, on="date", how="left")
        .join(dispersion, on="date", how="left")
        .join(avg_vol, on="date", how="left")
        .join(tail, on="date", how="left")
        .join(stress_dd, on="date", how="left")
        .join(avg_corr, on="date", how="left")
        .join(avg_corr_60, on="date", how="left")
        .join(spread, on="date", how="left")
    )
    
    # Add optional enhanced features
    if skew_cs is not None:
        result = result.join(skew_cs, on="date", how="left")
    
    if lvhv_spread is not None:
        result = result.join(lvhv_spread, on="date", how="left")
    
    if mom_spread is not None:
        result = result.join(mom_spread, on="date", how="left")
    
    # -----------------------------------------------------
    # Compute derived stress features AFTER joining f_mkt_cluster
    # -----------------------------------------------------
    result = result.sort("date")
    
    # realized_vol_mkt: rolling std of market return
    realized_vol_window = getattr(cfg, 'realized_vol_window', 20)
    result = result.with_columns([
        pl.col("f_mkt_cluster")
        .rolling_std(window_size=realized_vol_window, min_samples=5)
        .alias("realized_vol_mkt")
    ])
    
    # jumpiness: rolling max |ret| (captures sudden moves)
    jumpiness_window = getattr(cfg, 'jumpiness_window', 10)
    result = result.with_columns([
        pl.col("f_mkt_cluster").abs()
        .rolling_max(window_size=jumpiness_window, min_samples=3)
        .alias("jumpiness_mkt")
    ])
    
    # corr_spike: increase in correlation = stress
    result = result.with_columns([
        (pl.col("avg_corr_20") - pl.col("avg_corr_60")).alias("corr_spike")
    ])
    
    # -----------------------------------------------------
    # Add PCA factors if available
    # -----------------------------------------------------
    if factor_ts is not None and cfg.include_pca_factors:
        pca_cols = [c for c in factor_ts.columns if c.startswith("f_pca_")]
        if pca_cols:
            # Limit to n_pca_factors
            pca_cols = pca_cols[:cfg.n_pca_factors]
            factor_subset = factor_ts.select(["date"] + pca_cols)
            result = result.join(factor_subset, on="date", how="left")
            logger.info(f"  Added {len(pca_cols)} PCA factors")
    
    # -----------------------------------------------------
    # Fill nulls and clean up
    # -----------------------------------------------------
    feature_cols = [c for c in result.columns if c != "date"]
    result = result.with_columns([
        pl.col(c).fill_nan(None).fill_null(0).alias(c)
        for c in feature_cols
    ])
    
    result = result.sort("date")
    
    logger.info(f"  Universe features: {len(result)} dates, {len(feature_cols)} features")
    
    return result


def _compute_rolling_correlation(
    cluster_ts: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """
    Compute average off-diagonal correlation of cluster returns.
    
    For efficiency, samples clusters and uses rolling covariance.
    """
    logger = get_logger()
    
    # Pivot to wide format: dates x clusters
    wide = cluster_ts.pivot(
        on="cluster_id",
        index="date", 
        values="ret_ew"
    ).sort("date")
    
    dates = wide["date"].to_list()
    cluster_cols = [c for c in wide.columns if c != "date"]
    
    if len(cluster_cols) < 2:
        logger.warning("Not enough clusters for correlation")
        return pl.DataFrame({
            "date": dates,
            "avg_corr_20": [0.5] * len(dates)
        })
    
    # Convert to numpy for correlation computation
    ret_matrix = wide.select(cluster_cols).to_numpy()
    n_dates, n_clusters = ret_matrix.shape
    
    # Sample clusters if too many (for speed)
    max_clusters = 50
    if n_clusters > max_clusters:
        np.random.seed(42)
        cluster_idx = np.random.choice(n_clusters, max_clusters, replace=False)
        ret_matrix = ret_matrix[:, cluster_idx]
        n_clusters = max_clusters
    
    # Compute rolling correlation
    avg_corrs = []
    
    for i in range(n_dates):
        start_idx = max(0, i - window + 1)
        window_data = ret_matrix[start_idx:i+1]
        
        if len(window_data) < window // 2:
            avg_corrs.append(0.5)  # Default
            continue
        
        # Remove all-nan columns
        valid_cols = ~np.all(np.isnan(window_data), axis=0)
        window_data = window_data[:, valid_cols]
        
        if window_data.shape[1] < 2:
            avg_corrs.append(0.5)
            continue
        
        # Fill nans with column mean
        col_means = np.nanmean(window_data, axis=0)
        for j in range(window_data.shape[1]):
            mask = np.isnan(window_data[:, j])
            window_data[mask, j] = col_means[j]
        
        # Compute correlation matrix
        try:
            corr = np.corrcoef(window_data.T)
            # Average off-diagonal
            mask = ~np.eye(corr.shape[0], dtype=bool)
            avg_corr = np.nanmean(corr[mask])
            avg_corrs.append(float(avg_corr) if not np.isnan(avg_corr) else 0.5)
        except:
            avg_corrs.append(0.5)
    
    return pl.DataFrame({
        "date": dates,
        "avg_corr_20": avg_corrs,
    })


def _compute_defensive_aggressive_spread(
    cluster_ts: pl.DataFrame,
    def_quantile: float = 0.1,
    agg_quantile: float = 0.9,
) -> pl.DataFrame:
    """
    Compute return spread between defensive and aggressive clusters.
    
    Defensive = bottom quantile by volatility
    Aggressive = top quantile by volatility
    """
    # Get latest volatility per cluster
    latest_vol = (
        cluster_ts
        .sort("date")
        .group_by("cluster_id")
        .agg([
            pl.col("vol_ew_20").last().alias("vol_last"),
        ])
    )
    
    # Compute thresholds
    vol_def_thresh = latest_vol["vol_last"].quantile(def_quantile)
    vol_agg_thresh = latest_vol["vol_last"].quantile(agg_quantile)
    
    # Classify clusters
    defensive_clusters = (
        latest_vol
        .filter(pl.col("vol_last") <= vol_def_thresh)
        ["cluster_id"].to_list()
    )
    
    aggressive_clusters = (
        latest_vol
        .filter(pl.col("vol_last") >= vol_agg_thresh)
        ["cluster_id"].to_list()
    )
    
    # Compute returns for each group
    def_ret = (
        cluster_ts
        .filter(pl.col("cluster_id").is_in(defensive_clusters))
        .group_by("date")
        .agg([pl.col("ret_ew").mean().alias("ret_defensive")])
    )
    
    agg_ret = (
        cluster_ts
        .filter(pl.col("cluster_id").is_in(aggressive_clusters))
        .group_by("date")
        .agg([pl.col("ret_ew").mean().alias("ret_aggressive")])
    )
    
    # Compute spread
    spread = (
        def_ret
        .join(agg_ret, on="date", how="outer")
        .with_columns([
            (pl.col("ret_defensive").fill_null(0) - pl.col("ret_aggressive").fill_null(0))
            .alias("defensive_vs_aggressive_spread")
        ])
        .select(["date", "defensive_vs_aggressive_spread"])
    )
    
    return spread


def _compute_cross_sectional_skew(cluster_ts: pl.DataFrame) -> pl.DataFrame:
    """
    Compute cross-sectional skewness of cluster returns.
    
    During crashes, returns become more negatively skewed.
    High negative skew = stress.
    """
    from scipy.stats import skew
    
    # Group by date and compute skew manually
    dates = cluster_ts["date"].unique().sort().to_list()
    results = []
    
    for dt in dates:
        day_data = cluster_ts.filter(pl.col("date") == dt)
        rets = day_data["ret_ew"].drop_nulls().to_numpy()
        
        if len(rets) > 2:
            sk = float(skew(rets, nan_policy='omit'))
        else:
            sk = 0.0
        
        results.append({"date": dt, "skew_cs": sk})
    
    return pl.DataFrame(results)


def _compute_lowvol_highvol_spread(
    cluster_ts: pl.DataFrame,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> pl.DataFrame:
    """
    Compute return spread between low-vol and high-vol clusters.
    
    Low-vol minus high-vol: positive when defensive outperforms.
    Useful as duration proxy for rates prediction.
    """
    # Get rolling volatility per cluster
    latest_vol = (
        cluster_ts
        .sort("date")
        .group_by("cluster_id")
        .agg([
            pl.col("vol_ew_20").mean().alias("vol_avg"),
        ])
    )
    
    # Compute thresholds
    vol_low_thresh = latest_vol["vol_avg"].quantile(low_quantile)
    vol_high_thresh = latest_vol["vol_avg"].quantile(high_quantile)
    
    # Classify clusters
    low_vol_clusters = (
        latest_vol
        .filter(pl.col("vol_avg") <= vol_low_thresh)
        ["cluster_id"].to_list()
    )
    
    high_vol_clusters = (
        latest_vol
        .filter(pl.col("vol_avg") >= vol_high_thresh)
        ["cluster_id"].to_list()
    )
    
    # Compute returns for each group
    low_ret = (
        cluster_ts
        .filter(pl.col("cluster_id").is_in(low_vol_clusters))
        .group_by("date")
        .agg([pl.col("ret_ew").mean().alias("ret_lowvol")])
    )
    
    high_ret = (
        cluster_ts
        .filter(pl.col("cluster_id").is_in(high_vol_clusters))
        .group_by("date")
        .agg([pl.col("ret_ew").mean().alias("ret_highvol")])
    )
    
    # Compute spread
    spread = (
        low_ret
        .join(high_ret, on="date", how="outer")
        .with_columns([
            (pl.col("ret_lowvol").fill_null(0) - pl.col("ret_highvol").fill_null(0))
            .alias("lowvol_highvol_spread")
        ])
        .select(["date", "lowvol_highvol_spread"])
    )
    
    return spread


def _compute_momentum_spread(
    cluster_ts: pl.DataFrame,
    window: int = 20,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> pl.DataFrame:
    """
    Compute return spread between high-momentum and low-momentum clusters.
    
    High momentum minus low momentum clusters.
    Useful for factor regime detection.
    """
    # Compute rolling momentum per cluster (cumulative return over window)
    cluster_mom = (
        cluster_ts
        .sort(["cluster_id", "date"])
        .with_columns([
            pl.col("ret_ew")
            .rolling_sum(window_size=window, min_samples=window//2)
            .over("cluster_id")
            .alias("momentum_20")
        ])
    )
    
    # Get latest momentum per cluster
    latest_mom = (
        cluster_mom
        .group_by("cluster_id")
        .agg([
            pl.col("momentum_20").last().alias("mom_last"),
        ])
        .drop_nulls()
    )
    
    if len(latest_mom) < 5:
        # Not enough clusters
        dates = cluster_ts["date"].unique().sort().to_list()
        return pl.DataFrame({
            "date": dates,
            "momentum_spread": [0.0] * len(dates)
        })
    
    # Compute thresholds
    mom_high_thresh = latest_mom["mom_last"].quantile(1 - top_quantile)
    mom_low_thresh = latest_mom["mom_last"].quantile(bottom_quantile)
    
    # Classify clusters
    high_mom_clusters = (
        latest_mom
        .filter(pl.col("mom_last") >= mom_high_thresh)
        ["cluster_id"].to_list()
    )
    
    low_mom_clusters = (
        latest_mom
        .filter(pl.col("mom_last") <= mom_low_thresh)
        ["cluster_id"].to_list()
    )
    
    # Compute returns for each group
    high_ret = (
        cluster_ts
        .filter(pl.col("cluster_id").is_in(high_mom_clusters))
        .group_by("date")
        .agg([pl.col("ret_ew").mean().alias("ret_highmom")])
    )
    
    low_ret = (
        cluster_ts
        .filter(pl.col("cluster_id").is_in(low_mom_clusters))
        .group_by("date")
        .agg([pl.col("ret_ew").mean().alias("ret_lowmom")])
    )
    
    # Compute spread
    spread = (
        high_ret
        .join(low_ret, on="date", how="outer")
        .with_columns([
            (pl.col("ret_highmom").fill_null(0) - pl.col("ret_lowmom").fill_null(0))
            .alias("momentum_spread")
        ])
        .select(["date", "momentum_spread"])
    )
    
    return spread
class ProxyDatasetBuilder:
    """
    Build aligned (X, y) datasets for proxy training.
    
    Handles:
    - Feature/target alignment with proper time shifts
    - Anti-lookahead verification
    - Train/val splitting for CV
    """
    
    def __init__(
        self,
        universe_features: pl.DataFrame,
        cfg: "LayersConfig",
    ):
        """
        Initialize dataset builder.
        
        Args:
            universe_features: DataFrame with date and feature columns
            cfg: Layers configuration
        """
        self.universe_features = universe_features.sort("date")
        self.cfg = cfg
        self.logger = get_logger()
        
        # Extract feature columns
        self.feature_cols = [
            c for c in universe_features.columns 
            if c not in ["date", "cluster_set_id"]
        ]
        
        self.logger.info(f"Dataset builder: {len(self.feature_cols)} features")
    
    def build_vix_dataset(
        self,
        vix_data: pl.DataFrame,
    ) -> tuple[pl.DataFrame, str]:
        """
        Build dataset for VIX proxy (Proxy A).
        
        Enhanced targets for RL:
        - spike: Binary classification when Δlog(VIX) > q90 (BEST FOR RL)
        - multi-horizon: t+1 (fast shock) and t+5 (macro trend)
        - autoregressive: Include lagged predictions for inference
        
        For RL, spike detection is more valuable than exact prediction.
        
        Args:
            vix_data: DataFrame with date, vixcls columns
            
        Returns:
            (aligned_df, target_col_name)
        """
        cfg = self.cfg.proxy_vix
        
        # Prepare target: log(VIX_{t+1}) and Δlog(VIX)
        vix = vix_data.select(["date", "vixcls"]).sort("date")
        
        # Always compute log level
        vix = vix.with_columns([
            pl.col("vixcls").log().alias("_log_vix"),
            pl.col("vixcls").log().shift(-cfg.horizon).alias("y_vix_log"),
        ])
        
        # Compute Δlog(VIX) for primary horizon
        vix = vix.with_columns([
            (pl.col("y_vix_log") - pl.col("_log_vix")).alias("y_vix_dlog")
        ])
        
        # Also add current VIX as feature
        vix = vix.with_columns([
            pl.col("_log_vix").alias("f_vix_log_current")
        ])
        
        # === SPIKE DETECTION (PRIMARY FOR RL) ===
        if cfg.use_spike_detection:
            # Compute spike threshold from data
            if cfg.spike_threshold is not None:
                spike_thresh = cfg.spike_threshold
            else:
                # Use quantile
                spike_thresh = vix["y_vix_dlog"].drop_nulls().quantile(cfg.spike_quantile)
            
            vix = vix.with_columns([
                (pl.col("y_vix_dlog") > spike_thresh).cast(pl.Int32).alias("y_vix_spike")
            ])
            
            self.logger.info(f"VIX spike threshold (q{cfg.spike_quantile:.0%}): {spike_thresh:.4f}")
        
        # === MULTI-HORIZON ===
        if cfg.use_multi_horizon:
            for h in cfg.horizons:
                if h == cfg.horizon:
                    continue  # Already computed
                vix = vix.with_columns([
                    (pl.col("vixcls").log().shift(-h) - pl.col("_log_vix")).alias(f"y_vix_dlog_h{h}"),
                ])
                # Spike at each horizon
                if cfg.use_spike_detection:
                    vix = vix.with_columns([
                        (pl.col(f"y_vix_dlog_h{h}") > spike_thresh).cast(pl.Int32).alias(f"y_vix_spike_h{h}")
                    ])
        
        # Select primary target based on config
        if cfg.target_transform == "spike":
            target_col = "y_vix_spike"
        elif cfg.target_transform == "both":
            target_col = "y_vix_dlog"
        elif cfg.target_transform == "diff" or cfg.target_transform == "dlog":
            target_col = "y_vix_dlog"
        elif cfg.target_transform == "log":
            target_col = "y_vix_log"
        else:
            # Level transform
            vix = vix.with_columns([
                pl.col("vixcls").shift(-cfg.horizon).alias("y_vix_level")
            ])
            target_col = "y_vix_level"
        
        # Optional: high VIX classification target
        if cfg.classification_enabled:
            vix = vix.with_columns([
                (pl.col("vixcls").shift(-cfg.horizon) > cfg.high_threshold)
                .cast(pl.Int32)
                .alias("y_vix_high")
            ])
        
        # Build select columns
        select_cols = ["date", target_col, "f_vix_log_current"]
        if cfg.classification_enabled and "y_vix_high" not in select_cols:
            select_cols.append("y_vix_high")
        if cfg.use_spike_detection and "y_vix_spike" not in select_cols:
            select_cols.append("y_vix_spike")
        if cfg.target_transform == "both":
            if "y_vix_log" not in select_cols:
                select_cols.append("y_vix_log")
        
        # Add multi-horizon targets
        if cfg.use_multi_horizon:
            for h in cfg.horizons:
                if h != cfg.horizon:
                    dlog_col = f"y_vix_dlog_h{h}"
                    if dlog_col in vix.columns and dlog_col not in select_cols:
                        select_cols.append(dlog_col)
                    if cfg.use_spike_detection:
                        spike_col = f"y_vix_spike_h{h}"
                        if spike_col in vix.columns and spike_col not in select_cols:
                            select_cols.append(spike_col)
        
        # Join with features
        dataset = (
            self.universe_features
            .join(vix.select(select_cols), on="date", how="inner")
            .drop_nulls(subset=[target_col])
        )
        
        # Store extra features for this dataset
        self._vix_extra_features = ["f_vix_log_current"]
        
        self.logger.info(f"VIX dataset: {len(dataset)} samples, target={target_col}")
        if cfg.use_spike_detection:
            spike_rate = dataset["y_vix_spike"].mean() if "y_vix_spike" in dataset.columns else 0
            self.logger.info(f"  Spike rate: {spike_rate:.1%}")
        
        return dataset, target_col
    
    def build_rates_dataset(
        self,
        rates_data: pl.DataFrame,
    ) -> tuple[pl.DataFrame, str]:
        """
        Build dataset for rates proxy (Proxy B).
        
        Enhanced with:
        - Longer horizons (h=10 or h=20) for more macro trend
        - Categorical classification (down/flat/up) for stability
        
        Target: DGS10 change in basis points over horizon.
        """
        cfg = self.cfg.proxy_rates
        
        # Prepare target: Δ DGS10 in bps
        rates = rates_data.select(["date", "dgs10"]).sort("date")
        
        rates = rates.with_columns([
            ((pl.col("dgs10").shift(-cfg.horizon) - pl.col("dgs10")) * 100)
            .alias("y_rate_bps")
        ])
        
        # Categorical mode: down / flat / up
        use_categorical = getattr(cfg, 'use_categorical', False)
        
        if use_categorical:
            up_thresh = getattr(cfg, 'up_threshold_bps', 5.0)
            down_thresh = getattr(cfg, 'down_threshold_bps', -5.0)
            
            # Create categorical target: 0=down, 1=flat, 2=up
            rates = rates.with_columns([
                pl.when(pl.col("y_rate_bps") > up_thresh).then(2)
                .when(pl.col("y_rate_bps") < down_thresh).then(0)
                .otherwise(1)
                .cast(pl.Int32)
                .alias("y_rate_cat")
            ])
            target_col = "y_rate_cat"
        else:
            target_col = "y_rate_bps"
        
        # Optional: binary rate direction classification
        if cfg.classification_enabled:
            rates = rates.with_columns([
                (pl.col("y_rate_bps") > 0).cast(pl.Int32).alias("y_rate_up")
            ])
        
        # Build select columns
        select_cols = ["date", target_col]
        if use_categorical:
            select_cols.append("y_rate_bps")  # Keep original for reference
        if cfg.classification_enabled:
            select_cols.append("y_rate_up")
        
        # Remove duplicates in select_cols
        select_cols = list(dict.fromkeys(select_cols))
        
        # Join with features
        dataset = (
            self.universe_features
            .join(rates.select(select_cols), on="date", how="inner")
            .drop_nulls(subset=[target_col])
        )
        
        self.logger.info(f"Rates dataset: {len(dataset)} samples, target={target_col}")
        if use_categorical:
            # Log class distribution
            dist = dataset.group_by(target_col).agg(pl.count().alias("n"))
            self.logger.info(f"  Class distribution: {dist.to_dict()}")
        
        return dataset, target_col
    
    def build_recession_dataset(
        self,
        usrec_data: pl.DataFrame | None = None,
        sp500_data: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, str]:
        """
        Build dataset for recession/risk-off proxy (Proxy C).
        
        Enhanced with INTERNAL risk-off label (no external data needed):
        - "internal": Uses drawdown from cluster market proxy (always available)
        - "stress_index": Continuous stress index z(corr) + z(vol) - z(breadth)
        - "nber": Uses USREC with publication lag
        - "market": Uses SP500 < MA200
        """
        cfg = self.cfg.proxy_recession
        
        targets = []
        
        # NEW: Internal risk-off target (from cluster data only)
        if cfg.mode in ["internal", "both"]:
            internal_target = self._build_internal_riskoff_target(cfg)
            targets.append(internal_target)
        
        # NEW: Continuous stress index (alternative to binary)
        use_stress_index = getattr(cfg, 'use_stress_index', False)
        if use_stress_index:
            stress_target = self._build_stress_index_target()
            targets.append(stress_target)
        
        # NBER recession target
        if cfg.mode in ["nber"] and usrec_data is not None:
            from athenai.external.schemas import forward_fill_with_lag
            
            usrec_lagged = forward_fill_with_lag(
                usrec_data.select(["date", "usrec"]),
                lag_days=cfg.usrec_publication_lag_days,
                target_dates=self.universe_features["date"],
            )
            usrec_lagged = usrec_lagged.rename({"usrec": "y_usrec"})
            targets.append(usrec_lagged)
        
        # Market risk-off target
        if cfg.mode in ["market"] and sp500_data is not None:
            from athenai.external.schemas import compute_ma200
            
            sp500_with_ma = compute_ma200(
                sp500_data.select(["date", "sp500"]),
                window=cfg.ma_window
            )
            sp500_with_ma = sp500_with_ma.select([
                "date",
                pl.col("below_ma200").alias("y_risk_off_market")
            ])
            targets.append(sp500_with_ma)
        
        if not targets:
            raise ValueError("No recession/risk-off targets available")
        
        # Combine targets
        combined = self.universe_features.clone()
        for target_df in targets:
            combined = combined.join(target_df, on="date", how="left")
        
        # Determine primary target based on mode
        if cfg.mode == "internal" and "y_risk_off_internal" in combined.columns:
            target_col = "y_risk_off_internal"
        elif use_stress_index and "y_stress_index" in combined.columns:
            # For stress index, keep as regression target
            target_col = "y_stress_index"
        elif "y_usrec" in combined.columns:
            target_col = "y_usrec"
        elif "y_risk_off_internal" in combined.columns:
            target_col = "y_risk_off_internal"
        else:
            target_col = "y_risk_off_market"
        
        dataset = combined.drop_nulls(subset=[target_col])
        
        self.logger.info(f"Recession dataset: {len(dataset)} samples, target={target_col}")
        
        # Log class distribution for binary targets
        if target_col in ["y_risk_off_internal", "y_usrec", "y_risk_off_market"]:
            dist = dataset.group_by(target_col).agg(pl.count().alias("n"))
            self.logger.info(f"  Class distribution: {dist.to_dict()}")
        
        return dataset, target_col
    
    def _build_internal_riskoff_target(self, cfg) -> pl.DataFrame:
        """
        Build internal risk-off label from cluster data only.
        
        IMPROVED: Uses FUTURE window for proper prediction task:
        - Risk-off = 1 when market proxy will have drawdown > threshold
          in the next H days (forward-looking label)
        - This is the correct formulation for RL: predict future stress
        
        When use_future_window=True (default):
          y_t = 1 if min(ret_{t+1:t+H}) < threshold
        
        When use_future_window=False (original):
          y_t = 1 if current_drawdown < threshold
        """
        internal_dd_thresh = getattr(cfg, 'internal_dd_threshold', -0.05)
        internal_window = getattr(cfg, 'internal_window', 60)
        
        # NEW: Future window configuration
        use_future_window = getattr(cfg, 'use_future_window', True)
        future_horizon = getattr(cfg, 'future_horizon', 10)
        
        # NEW: Use quantile-based threshold if fixed threshold gives no positives
        use_quantile_threshold = getattr(cfg, 'use_quantile_threshold', True)
        risk_quantile = getattr(cfg, 'risk_quantile', 0.10)  # Bottom 10% = risk
        
        features = self.universe_features.sort("date")
        
        if use_future_window:
            # FORWARD-LOOKING: Predict if drawdown will occur in next H days
            self.logger.info(f"Risk-off using FUTURE window (h={future_horizon})")
            
            # Compute forward cumulative min return (proxy for max drawdown)
            features = features.with_columns([
                pl.col("f_mkt_cluster")
                .shift(-1)  # Start from tomorrow
                .rolling_sum(window_size=future_horizon, min_samples=1)
                .shift(-(future_horizon - 1))  # Align to current date
                .alias("_fwd_cumret")
            ])
            
            # Alternative: compute forward min of cumulative returns
            # This captures the worst point in the future window
            cum_ret = features["f_mkt_cluster"].cum_sum().to_numpy()
            n = len(cum_ret)
            fwd_min = np.zeros(n)
            for i in range(n):
                end_idx = min(i + future_horizon, n)
                if i < end_idx:
                    fwd_window = cum_ret[i+1:end_idx+1] - cum_ret[i]
                    fwd_min[i] = fwd_window.min() if len(fwd_window) > 0 else 0
                else:
                    fwd_min[i] = np.nan
            
            features = features.with_columns([
                pl.Series("_fwd_min_ret", fwd_min)
            ])
            
            # Determine threshold
            if use_quantile_threshold:
                # Use quantile-based threshold (bottom X% are risk-off)
                valid_fwd_min = fwd_min[~np.isnan(fwd_min)]
                threshold = np.percentile(valid_fwd_min, risk_quantile * 100)
                self.logger.info(f"Using quantile threshold: q{risk_quantile:.0%} = {threshold:.4f}")
            else:
                threshold = internal_dd_thresh
            
            target = features.select([
                "date",
                (pl.col("_fwd_min_ret") < threshold)
                .cast(pl.Int32)
                .alias("y_risk_off_internal")
            ])
            
        else:
            # BACKWARD-LOOKING: Current drawdown state (original implementation)
            self.logger.info("Risk-off using BACKWARD window (current state)")
            
            # Use stress_dd_min_60 from universe features if available
            if "stress_dd_min_60" in features.columns:
                target = features.select([
                    "date",
                    (pl.col("stress_dd_min_60") < internal_dd_thresh)
                    .cast(pl.Int32)
                    .alias("y_risk_off_internal")
                ])
            else:
                # Fallback: compute from f_mkt_cluster
                features = features.with_columns([
                    pl.col("f_mkt_cluster")
                    .cum_sum()
                    .alias("_cum_ret"),
                ])
                features = features.with_columns([
                    pl.col("_cum_ret")
                    .rolling_max(window_size=internal_window, min_samples=10)
                    .alias("_hwm"),
                ])
                features = features.with_columns([
                    (pl.col("_cum_ret") - pl.col("_hwm")).alias("_dd"),
                ])
                target = features.select([
                    "date",
                    (pl.col("_dd") < internal_dd_thresh)
                    .cast(pl.Int32)
                    .alias("y_risk_off_internal")
                ])
        
        # === FEATURE INDEPENDENCE CHECK ===
        verify_independence = getattr(cfg, 'verify_feature_independence', True)
        if verify_independence:
            self._verify_riskoff_feature_independence(target)
        
        return target
    
    def _verify_riskoff_feature_independence(self, target: pl.DataFrame) -> None:
        """
        Verify that features don't directly compute the same thing as target.
        
        Checks correlation between target and each feature.
        High correlation (>0.9) with any feature suggests potential leakage.
        """
        target_col = "y_risk_off_internal"
        if target_col not in target.columns:
            return
        
        # Merge target with features
        merged = self.universe_features.join(target, on="date", how="inner")
        
        # Check correlation with each feature
        suspicious_features = []
        y = merged[target_col].to_numpy()
        
        for col in self.feature_cols:
            if col not in merged.columns:
                continue
            x = merged[col].to_numpy()
            
            # Handle NaN values
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            if valid_mask.sum() < 100:
                continue
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            # Compute correlation
            corr = np.corrcoef(x_valid, y_valid)[0, 1]
            
            if abs(corr) > 0.90:
                suspicious_features.append((col, corr))
        
        if suspicious_features:
            self.logger.warning("[!] Features with high correlation to risk-off target:")
            for feat, corr in suspicious_features:
                self.logger.warning(f"  {feat}: r={corr:.3f}")
            self.logger.warning("This may indicate feature leakage!")
        else:
            self.logger.info("[OK] No feature leakage detected (all features r < 0.9 with target)")
    
    def _build_stress_index_target(self) -> pl.DataFrame:
        """
        Build continuous stress index from internal features.
        
        stress_t = z(avg_corr_20) + z(avg_vol_20) - z(breadth_pos_ret)
        
        Higher = more stress. RL can use this as "survival mode" signal.
        """
        features = self.universe_features.clone()
        
        # Z-score each component
        stress_cols = []
        
        if "avg_corr_20" in features.columns:
            features = features.with_columns([
                ((pl.col("avg_corr_20") - pl.col("avg_corr_20").mean()) / 
                 (pl.col("avg_corr_20").std() + 1e-9)).alias("_z_corr")
            ])
            stress_cols.append("_z_corr")
        
        if "avg_vol_20" in features.columns:
            features = features.with_columns([
                ((pl.col("avg_vol_20") - pl.col("avg_vol_20").mean()) / 
                 (pl.col("avg_vol_20").std() + 1e-9)).alias("_z_vol")
            ])
            stress_cols.append("_z_vol")
        
        if "breadth_pos_ret" in features.columns:
            # Negative for breadth (low breadth = high stress)
            features = features.with_columns([
                (-(pl.col("breadth_pos_ret") - pl.col("breadth_pos_ret").mean()) / 
                 (pl.col("breadth_pos_ret").std() + 1e-9)).alias("_z_breadth_neg")
            ])
            stress_cols.append("_z_breadth_neg")
        
        if not stress_cols:
            # Fallback to constant
            return features.select([
                "date",
                pl.lit(0.0).alias("y_stress_index")
            ])
        
        # Sum z-scores
        features = features.with_columns([
            sum([pl.col(c) for c in stress_cols]).alias("y_stress_index")
        ])
        
        return features.select(["date", "y_stress_index"])
    
    def build_factors_dataset(
        self,
        factors_data: pl.DataFrame,
    ) -> tuple[pl.DataFrame, str]:
        """
        Build dataset for factor monitor proxy (Proxy D).
        
        Enhanced with:
        - Binary mode: Predict MOM > 0 instead of multiclass (more robust with 54 samples)
        - Persistence baseline: Last month's winner persists
        
        Target: winning factor next month (multiclass) OR binary MOM > 0.
        """
        cfg = self.cfg.proxy_factors
        
        # Aggregate features to monthly (using end-of-month date)
        features_monthly = (
            self.universe_features
            .with_columns([
                pl.col("date").dt.strftime("%Y-%m").alias("_ym"),
            ])
            .group_by("_ym")
            .agg([
                pl.col("date").max().alias("date"),  # End of month date
                *[pl.col(c).mean().alias(c) for c in self.feature_cols]
            ])
            .drop("_ym")
            .sort("date")
        )
        
        # Prepare factor data
        factor_cols = list(cfg.factors)
        
        from athenai.external.famafrench import FamaFrenchFetcher
        fetcher = FamaFrenchFetcher()
        factors_with_winner = fetcher.get_winning_factor(
            factors_data, 
            factor_cols=factor_cols
        )
        
        # Shift to get next month's values
        factors_with_winner = factors_with_winner.sort("date")
        factors_with_winner = factors_with_winner.with_columns([
            pl.col("winning_factor").shift(-1).alias("y_winning_factor")
        ])
        
        # Check binary mode
        use_binary_mode = getattr(cfg, 'use_binary_mode', False)
        binary_target = getattr(cfg, 'binary_target', 'mom')
        
        if use_binary_mode:
            # Binary: predict if target factor > 0 next month
            if binary_target in factors_data.columns:
                factors_with_winner = factors_with_winner.with_columns([
                    (pl.col(binary_target).shift(-1) > 0)
                    .cast(pl.Int32)
                    .alias("y_factor_positive")
                ])
                target_col = "y_factor_positive"
            else:
                # Fallback: predict if MOM is the winner
                factors_with_winner = factors_with_winner.with_columns([
                    (pl.col("y_winning_factor") == binary_target)
                    .cast(pl.Int32)
                    .alias("y_factor_positive")
                ])
                target_col = "y_factor_positive"
        else:
            # Multiclass: predict winning factor
            label_map = {f: i for i, f in enumerate(factor_cols)}
            factors_with_winner = factors_with_winner.with_columns([
                pl.col("y_winning_factor")
                .map_elements(lambda x: label_map.get(x, -1), return_dtype=pl.Int32)
                .alias("y_winning_factor_int")
            ])
            target_col = "y_winning_factor_int"
        
        # Add current winner as feature (for persistence baseline comparison)
        use_persistence = getattr(cfg, 'use_persistence_baseline', True)
        if use_persistence and not use_binary_mode:
            label_map = {f: i for i, f in enumerate(factor_cols)}
            factors_with_winner = factors_with_winner.with_columns([
                pl.col("winning_factor")
                .map_elements(lambda x: label_map.get(x, -1), return_dtype=pl.Int32)
                .alias("f_current_winner_int")
            ])
        
        # Add year-month key to both dataframes for joining
        features_monthly = features_monthly.with_columns([
            pl.col("date").dt.strftime("%Y-%m").alias("_join_ym")
        ])
        
        factors_with_winner = factors_with_winner.with_columns([
            pl.col("date").dt.strftime("%Y-%m").alias("_join_ym")
        ])
        
        # Build select columns
        select_cols = ["_join_ym", target_col]
        if not use_binary_mode:
            select_cols.append("y_winning_factor")
        if use_persistence and not use_binary_mode:
            select_cols.append("f_current_winner_int")
        
        # Remove duplicates
        select_cols = list(dict.fromkeys(select_cols))
        
        # Join on year-month key instead of exact date
        dataset = (
            features_monthly
            .join(
                factors_with_winner.select(select_cols),
                on="_join_ym",
                how="inner"
            )
            .drop("_join_ym")
        )
        
        # Filter valid targets
        if use_binary_mode:
            dataset = dataset.drop_nulls(subset=[target_col])
        else:
            dataset = dataset.filter(pl.col(target_col) >= 0)
        
        self.logger.info(f"Factors dataset: {len(dataset)} samples, target={target_col}")
        
        # Log class distribution
        dist = dataset.group_by(target_col).agg(pl.count().alias("n"))
        self.logger.info(f"  Class distribution: {dist.to_dict()}")
        
        return dataset, target_col
    
    def get_cv_splits(
        self,
        dataset: pl.DataFrame,
    ) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate walk-forward CV splits.
        
        Returns:
            List of (train_df, val_df) tuples
        """
        cv_cfg = self.cfg.cv_config
        
        dates = dataset["date"].unique().sort().to_list()
        n_dates = len(dates)
        
        # Calculate split points
        val_days = cv_cfg.val_months * 21  # ~21 trading days per month
        min_train_days = cv_cfg.min_train_months * 21
        
        splits = []
        
        for fold_idx in range(cv_cfg.n_folds):
            # Expanding window: train end moves forward each fold
            train_end_idx = min_train_days + fold_idx * val_days
            val_start_idx = train_end_idx + cv_cfg.gap_days
            val_end_idx = val_start_idx + val_days
            
            if val_end_idx > n_dates:
                break
            
            train_dates = dates[:train_end_idx]
            val_dates = dates[val_start_idx:val_end_idx]
            
            train_df = dataset.filter(pl.col("date").is_in(train_dates))
            val_df = dataset.filter(pl.col("date").is_in(val_dates))
            
            if len(train_df) > 0 and len(val_df) > 0:
                splits.append((train_df, val_df))
                self.logger.debug(
                    f"Fold {fold_idx}: train {len(train_df)} samples "
                    f"({train_dates[0]} to {train_dates[-1]}), "
                    f"val {len(val_df)} samples"
                )
        
        self.logger.info(f"Generated {len(splits)} CV folds")
        
        return splits
    
    def to_numpy(
        self,
        dataset: pl.DataFrame,
        target_col: str,
        extra_features: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[date]]:
        """
        Convert dataset to numpy arrays.
        
        Args:
            dataset: The dataset DataFrame
            target_col: Name of target column
            extra_features: Additional feature columns specific to this dataset
        
        Returns:
            (X, y, dates)
        """
        # Get base features that exist in this dataset
        feature_cols = [c for c in self.feature_cols if c in dataset.columns]
        
        # Add extra features if provided (e.g., f_vix_log_current for VIX proxy)
        if extra_features:
            for f in extra_features:
                if f in dataset.columns and f not in feature_cols:
                    feature_cols.append(f)
        
        X = dataset.select(feature_cols).to_numpy().astype(np.float64)
        y = dataset[target_col].to_numpy()
        dates = dataset["date"].to_list()
        
        # Handle any remaining NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y, dates
