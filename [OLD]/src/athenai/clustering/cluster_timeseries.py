"""
Build cluster-level time series and artifacts.

Produces:
- cluster_timeseries_<id>.parquet: Daily returns + stats per cluster
- cluster_alive_mask_<id>.parquet: Binary mask of active algos per date/cluster
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from athenai.data.base import PipelineStep

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore
    from athenai.factors.config import ClusteringConfig


class BuildClusterArtifactsStep(PipelineStep):
    """
    Build cluster-level time series and alive masks.
    
    For each date:
    - Identify which algos are alive (had data on that date)
    - Compute cluster stats: EW return, vol, n_alive
    
    Outputs per cluster_set:
    - cluster_timeseries_<id>.parquet:
        date, cluster_id, ret_ew, vol_ew_20, sharpe_rolling_60, n_alive, pct_alive
    - cluster_alive_mask_<id>.parquet:
        date, algo_id, cluster_id, is_alive (optional for RL)
    """
    
    name = "build_cluster_artifacts"
    
    def inputs(self) -> list[str]:
        return [
            "algos_panel",
        ]
    
    def outputs(self) -> list[str]:
        return []  # Dynamic per cluster_set
    
    def run(
        self,
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Build cluster timeseries for all cluster_sets."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        outputs = {}
        
        # Load algos_panel
        panel_path = cfg.cache_dir / cfg.preprocess_run_id / "algos_panel.parquet"
        logger.info(f"Loading algos_panel from {panel_path}")
        algos_panel = pl.read_parquet(str(panel_path))
        
        # Get all unique dates
        all_dates = algos_panel["date"].unique().sort()
        logger.info(f"Panel has {len(all_dates)} dates, {algos_panel['algo_id'].n_unique()} algos")
        
        # Process each cluster_set
        for cs in cfg.cluster_sets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Building timeseries for: {cs.cluster_set_id}")
            
            cs_outputs = self._build_single_cluster_timeseries(
                cs, store, cfg, algos_panel, all_dates, overwrite, logger
            )
            outputs.update(cs_outputs)
        
        return outputs
    
    def _build_single_cluster_timeseries(
        self,
        cs,
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        algos_panel: pl.DataFrame,
        all_dates: pl.Series,
        overwrite: bool,
        logger,
    ) -> dict[str, Path]:
        """Build timeseries for one cluster_set."""
        
        ts_path = store.artifact_path(f"cluster_timeseries_{cs.cluster_set_id}.parquet")
        mask_path = store.artifact_path(f"cluster_alive_mask_{cs.cluster_set_id}.parquet")
        
        if ts_path.exists() and mask_path.exists() and not overwrite:
            logger.info(f"  Outputs exist, skipping")
            return {
                f"cluster_timeseries_{cs.cluster_set_id}": ts_path,
                f"cluster_alive_mask_{cs.cluster_set_id}": mask_path,
            }
        
        # Load cluster_map
        map_path = store.artifact_path(f"cluster_map_{cs.cluster_set_id}.parquet")
        if not map_path.exists():
            raise FileNotFoundError(f"cluster_map not found: {map_path}")
        
        cluster_map = pl.read_parquet(str(map_path))
        logger.info(f"  Loaded cluster_map: {len(cluster_map)} algos")
        
        # Join panel with cluster assignments
        panel_clustered = algos_panel.join(
            cluster_map.select(["algo_id", "cluster_id"]),
            on="algo_id",
            how="inner"  # Only algos in cluster_map
        )
        
        logger.info(f"  Panel with clusters: {len(panel_clustered)} rows")
        
        # Mark alive: algo had return on date (not null)
        panel_clustered = panel_clustered.with_columns([
            pl.col("ret_1d").is_not_null().alias("is_alive"),
        ])
        
        # -----------------------------------------------------------------
        # Build cluster_timeseries: aggregate per (date, cluster_id)
        # -----------------------------------------------------------------
        
        # First: compute EW return per cluster/date
        cluster_ts = (
            panel_clustered
            .filter(pl.col("ret_1d").is_not_null())
            .group_by(["date", "cluster_id"])
            .agg([
                pl.col("ret_1d").mean().alias("ret_ew"),
                pl.col("ret_1d").std().alias("ret_std"),
                pl.col("ret_1d").count().alias("n_alive"),
            ])
            .sort(["date", "cluster_id"])
        )
        
        # Get total members per cluster from cluster_map
        total_members = (
            cluster_map
            .group_by("cluster_id")
            .agg(pl.len().alias("n_total"))
        )
        
        cluster_ts = cluster_ts.join(total_members, on="cluster_id", how="left")
        cluster_ts = cluster_ts.with_columns([
            (pl.col("n_alive") / pl.col("n_total")).alias("pct_alive"),
        ])
        
        # Rolling metrics per cluster
        # Sort by cluster_id, date for rolling
        cluster_ts = cluster_ts.sort(["cluster_id", "date"])
        
        # Rolling vol (20d)
        cluster_ts = cluster_ts.with_columns([
            pl.col("ret_ew")
            .rolling_std(window_size=20, min_samples=10)
            .over("cluster_id")
            .mul(np.sqrt(252))
            .alias("vol_ew_20"),
        ])
        
        # Rolling Sharpe (60d)
        cluster_ts = cluster_ts.with_columns([
            (
                pl.col("ret_ew").rolling_mean(window_size=60, min_samples=20).over("cluster_id")
                / pl.col("ret_ew").rolling_std(window_size=60, min_samples=20).over("cluster_id").add(1e-9)
            ).mul(np.sqrt(252)).alias("sharpe_rolling_60"),
        ])
        
        # Add cluster_set_id
        cluster_ts = cluster_ts.with_columns([
            pl.lit(cs.cluster_set_id).alias("cluster_set_id"),
        ])
        
        # Select final columns
        cluster_ts = cluster_ts.select([
            "cluster_set_id", "date", "cluster_id",
            "ret_ew", "vol_ew_20", "sharpe_rolling_60",
            "n_alive", "n_total", "pct_alive",
        ])
        
        logger.info(f"  cluster_timeseries: {len(cluster_ts)} rows")
        
        # Write
        cluster_ts.write_parquet(str(ts_path), compression="zstd")
        store.register_artifact(f"cluster_timeseries_{cs.cluster_set_id}", ts_path, cluster_ts)
        
        # -----------------------------------------------------------------
        # Build cluster_alive_mask
        # -----------------------------------------------------------------
        
        # For RL, we need to know which algos are alive on each date
        # This is a sparse representation: only rows where is_alive=True
        
        alive_mask = (
            panel_clustered
            .filter(pl.col("is_alive"))
            .select(["date", "algo_id", "cluster_id"])
            .with_columns([
                pl.lit(cs.cluster_set_id).alias("cluster_set_id"),
            ])
            .sort(["date", "cluster_id", "algo_id"])
        )
        
        logger.info(f"  cluster_alive_mask: {len(alive_mask)} rows (sparse)")
        
        alive_mask.write_parquet(str(mask_path), compression="zstd")
        store.register_artifact(f"cluster_alive_mask_{cs.cluster_set_id}", mask_path, alive_mask)
        
        return {
            f"cluster_timeseries_{cs.cluster_set_id}": ts_path,
            f"cluster_alive_mask_{cs.cluster_set_id}": mask_path,
        }
    
    def validate(self, store: "ArtifactStore") -> None:
        """Validate cluster timeseries."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        for key, path in store._latest.items():
            if key.startswith("cluster_timeseries_"):
                df = pl.read_parquet(str(path))
                
                # Check for nulls in critical columns
                null_ret = df["ret_ew"].null_count()
                if null_ret > 0:
                    logger.warning(f"{key}: {null_ret} null ret_ew values")
                
                # Check cluster coverage
                n_clusters = df["cluster_id"].n_unique()
                n_dates = df["date"].n_unique()
                expected = n_clusters * n_dates
                actual = len(df)
                
                coverage = actual / expected if expected > 0 else 0
                logger.info(f"{key}: {n_clusters} clusters, {n_dates} dates, coverage={coverage:.1%}")
        
        logger.info("Cluster timeseries validation passed!")


class BuildClusterFeaturesDailyStep(PipelineStep):
    """
    Build RL-ready features per cluster per day.
    
    Creates a state representation for each (date, cluster_id):
    - Rolling stats from cluster_timeseries
    - Cluster composition features (from cluster_meta)
    - Optional: macro regime features
    
    Output:
    - cluster_features_daily_<id>.parquet:
        date, cluster_id, f_ret_20d, f_vol_20d, f_sharpe_60d, f_pct_alive, ...
    """
    
    name = "build_cluster_features_daily"
    
    def inputs(self) -> list[str]:
        return []  # Depends on cluster_timeseries dynamically
    
    def outputs(self) -> list[str]:
        return []  # Dynamic
    
    def run(
        self,
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Build daily features for all cluster_sets."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        outputs = {}
        ts_cfg = cfg.timeseries_config
        
        for cs in cfg.cluster_sets:
            logger.info(f"\nBuilding features_daily for: {cs.cluster_set_id}")
            
            feat_path = store.artifact_path(f"cluster_features_daily_{cs.cluster_set_id}.parquet")
            
            if feat_path.exists() and not overwrite:
                logger.info(f"  Output exists, skipping")
                outputs[f"cluster_features_daily_{cs.cluster_set_id}"] = feat_path
                continue
            
            # Load cluster_timeseries
            ts_path = store.artifact_path(f"cluster_timeseries_{cs.cluster_set_id}.parquet")
            if not ts_path.exists():
                raise FileNotFoundError(f"cluster_timeseries not found: {ts_path}")
            
            cluster_ts = pl.read_parquet(str(ts_path))
            
            # Load cluster_meta
            meta_path = store.artifact_path(f"cluster_meta_{cs.cluster_set_id}.parquet")
            cluster_meta = None
            if meta_path.exists():
                cluster_meta = pl.read_parquet(str(meta_path))
            
            # Build features
            features_df = self._build_features(
                cluster_ts, cluster_meta, ts_cfg, logger
            )
            
            features_df = features_df.with_columns([
                pl.lit(cs.cluster_set_id).alias("cluster_set_id"),
            ])
            
            logger.info(f"  cluster_features_daily: {len(features_df)} rows, {len(features_df.columns)} features")
            
            features_df.write_parquet(str(feat_path), compression="zstd")
            store.register_artifact(f"cluster_features_daily_{cs.cluster_set_id}", feat_path, features_df)
            
            outputs[f"cluster_features_daily_{cs.cluster_set_id}"] = feat_path
        
        return outputs
    
    def _build_features(
        self,
        cluster_ts: pl.DataFrame,
        cluster_meta: pl.DataFrame | None,
        ts_cfg,
        logger,
    ) -> pl.DataFrame:
        """Build rolling features from cluster timeseries."""
        
        # Sort for rolling
        df = cluster_ts.sort(["cluster_id", "date"])
        
        # Rolling return features
        for w in ts_cfg.rolling_windows:
            df = df.with_columns([
                pl.col("ret_ew")
                .rolling_mean(window_size=w, min_samples=max(1, w//4))
                .over("cluster_id")
                .alias(f"f_ret_{w}d"),
                
                pl.col("ret_ew")
                .rolling_std(window_size=w, min_samples=max(1, w//4))
                .over("cluster_id")
                .mul(np.sqrt(252))
                .alias(f"f_vol_{w}d"),
            ])
        
        # Momentum features
        df = df.with_columns([
            pl.col("ret_ew")
            .rolling_sum(window_size=20, min_samples=10)
            .over("cluster_id")
            .alias("f_mom_20d"),
            
            pl.col("ret_ew")
            .rolling_sum(window_size=60, min_samples=30)
            .over("cluster_id")
            .alias("f_mom_60d"),
        ])
        
        # Existing columns as features
        df = df.with_columns([
            pl.col("vol_ew_20").alias("f_vol_ew_20"),
            pl.col("sharpe_rolling_60").alias("f_sharpe_60d"),
            pl.col("pct_alive").alias("f_pct_alive"),
            pl.col("n_alive").alias("f_n_alive"),
        ])
        
        # Drawdown proxy: distance from rolling max
        df = df.with_columns([
            (
                1 - pl.col("ret_ew").rolling_sum(window_size=60, min_samples=10).over("cluster_id")
                / pl.col("ret_ew").rolling_sum(window_size=60, min_samples=10).over("cluster_id")
                  .rolling_max(window_size=60, min_samples=10).over("cluster_id")
            ).clip(lower_bound=0, upper_bound=1).alias("f_drawdown_proxy"),
        ])
        
        # Join cluster_meta static features if available
        if cluster_meta is not None and len(cluster_meta) > 0:
            meta_cols = [c for c in cluster_meta.columns if c not in ["cluster_set_id", "cluster_id"]]
            if meta_cols:
                # Prefix with "f_meta_"
                meta_renamed = cluster_meta.select(
                    ["cluster_id"] + [pl.col(c).alias(f"f_meta_{c}") for c in meta_cols]
                )
                df = df.join(meta_renamed, on="cluster_id", how="left")
        
        # Select feature columns
        feature_cols = [c for c in df.columns if c.startswith("f_")]
        base_cols = ["cluster_set_id", "date", "cluster_id"] if "cluster_set_id" in df.columns else ["date", "cluster_id"]
        
        df = df.select(base_cols + sorted(feature_cols))
        
        # Fill nulls for early dates
        df = df.with_columns([
            pl.col(c).fill_nan(None).fill_null(0) for c in feature_cols
        ])
        
        return df
    
    def validate(self, store: "ArtifactStore") -> None:
        """Validate features."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        for key, path in store._latest.items():
            if key.startswith("cluster_features_daily_"):
                df = pl.read_parquet(str(path))
                
                # Check feature columns
                f_cols = [c for c in df.columns if c.startswith("f_")]
                logger.info(f"{key}: {len(f_cols)} features, {len(df)} rows")
                
                # Check for all-null features
                for c in f_cols:
                    null_pct = df[c].null_count() / len(df)
                    if null_pct > 0.5:
                        logger.warning(f"{key}: {c} has {null_pct:.1%} nulls")
        
        logger.info("Cluster features validation passed!")
