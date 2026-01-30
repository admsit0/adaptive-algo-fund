"""
Factor time series construction: market factor + PCA factors.

Builds:
- f_mkt: Equal-weight market return (mean of universe)
- f_pca_1..K: PCA factors from returns matrix

Outputs:
- factor_timeseries.parquet
- pca_model.npz (components, mean, std, algo_ids)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from athenai.data.base import PipelineStep

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore
    from athenai.factors.config import ClusteringConfig


class BuildFactorTimeseriesStep(PipelineStep):
    """
    Build factor time series from returns panel.
    
    Produces:
    - factor_timeseries.parquet: (date, f_mkt, f_pca_1..K)
    - pca_model.npz: PCA model for reuse
    
    Strategy:
    1. Build f_mkt as equal-weight mean return per date
    2. Build returns matrix (algo x date) with optional sampling
    3. Run PCA via SVD (no sklearn needed)
    4. Project returns onto PCA to get factor time series
    """
    
    name = "build_factor_timeseries"
    
    def inputs(self) -> list[str]:
        return ["algos_panel", "algos_meta_good"]
    
    def outputs(self) -> list[str]:
        return ["factor_timeseries", "pca_model"]
    
    def run(
        self,
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Build factor time series."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        fcfg = cfg.factor_config
        
        # Paths
        preprocess_dir = cfg.cache_dir / cfg.preprocess_run_id
        panel_path = preprocess_dir / "algos_panel.parquet"
        meta_good_path = preprocess_dir / "algos_meta_good.parquet"
        
        out_factors_path = store.artifact_path(fcfg.factor_timeseries_name)
        out_pca_path = store.artifact_path(fcfg.pca_model_name)
        
        if out_factors_path.exists() and out_pca_path.exists() and not overwrite:
            logger.info("Factor outputs exist, skipping")
            return {
                "factor_timeseries": out_factors_path,
                "pca_model": out_pca_path,
            }
        
        logger.info("Building factor time series...")
        
        # Load panel
        panel_lf = pl.scan_parquet(str(panel_path))
        
        # Apply asof_date filter
        if fcfg.asof_date:
            asof = date.fromisoformat(fcfg.asof_date)
            panel_lf = panel_lf.filter(pl.col("date") <= asof)
        
        # Filter to GOOD universe if requested
        if fcfg.use_good_universe:
            good_ids = pl.scan_parquet(str(meta_good_path)).select("algo_id").collect()
            panel_lf = panel_lf.join(good_ids.lazy(), on="algo_id", how="inner")
        
        # Collect for processing
        panel_df = panel_lf.select(["algo_id", "date", "ret_1d"]).collect()
        logger.info(f"  Panel loaded: {len(panel_df):,} rows")
        
        # ========================================
        # Step 1: Build f_mkt (market factor)
        # ========================================
        logger.info("  Computing market factor (f_mkt)...")
        
        f_mkt = (
            panel_df
            .group_by("date")
            .agg([
                pl.col("ret_1d").mean().alias("f_mkt"),
                pl.len().alias("n_algos"),
            ])
            .sort("date")
        )
        
        logger.info(f"    {len(f_mkt)} dates, avg {f_mkt['n_algos'].mean():.0f} algos/day")
        
        # ========================================
        # Step 2: Build returns matrix for PCA
        # ========================================
        logger.info("  Building returns matrix for PCA...")
        
        # Sample dates if needed
        all_dates = f_mkt["date"].to_list()
        
        if fcfg.pca_date_sampling == "W":
            # Weekly sampling: take every 5th date
            sampled_dates = all_dates[::5]
        elif fcfg.pca_date_sampling == "M":
            # Monthly sampling: take every 21st date
            sampled_dates = all_dates[::21]
        else:
            sampled_dates = all_dates
        
        # Limit to max dates
        if len(sampled_dates) > fcfg.pca_max_dates:
            # Take most recent
            sampled_dates = sampled_dates[-fcfg.pca_max_dates:]
        
        logger.info(f"    Using {len(sampled_dates)} dates for PCA")
        
        # Filter panel to sampled dates
        panel_sampled = panel_df.filter(pl.col("date").is_in(sampled_dates))
        
        # Pivot to wide format: rows=algos, cols=dates
        returns_wide = (
            panel_sampled
            .pivot(
                on="date",
                index="algo_id",
                values="ret_1d",
            )
        )
        
        algo_ids = returns_wide["algo_id"].to_list()
        date_cols = [c for c in returns_wide.columns if c != "algo_id"]
        
        # Extract matrix (handle missing with policy)
        X = returns_wide.select(date_cols).to_numpy()
        
        if fcfg.missing_policy == "zero":
            X = np.nan_to_num(X, nan=0.0)
        elif fcfg.missing_policy == "drop_algo":
            # Keep only algos with no missing
            valid_mask = ~np.any(np.isnan(X), axis=1)
            X = X[valid_mask]
            algo_ids = [a for a, v in zip(algo_ids, valid_mask) if v]
        
        logger.info(f"    Returns matrix: {X.shape[0]} algos x {X.shape[1]} dates")
        
        # Standardize if requested
        if fcfg.standardize_returns:
            X_mean = np.nanmean(X, axis=1, keepdims=True)
            X_std = np.nanstd(X, axis=1, keepdims=True)
            X_std[X_std < 1e-10] = 1.0  # Avoid division by zero
            X = (X - X_mean) / X_std
        else:
            X_mean = np.zeros((X.shape[0], 1))
            X_std = np.ones((X.shape[0], 1))
        
        # ========================================
        # Step 3: PCA via SVD
        # ========================================
        logger.info(f"  Running PCA (k={fcfg.pca_k})...")
        
        # Center columns (dates) for PCA
        X_centered = X - np.nanmean(X, axis=0, keepdims=True)
        X_centered = np.nan_to_num(X_centered, nan=0.0)
        
        # SVD: X = U @ S @ Vt
        # U: (n_algos, k) - algo loadings
        # S: (k,) - singular values
        # Vt: (k, n_dates) - date scores (transposed)
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.warning("SVD failed, using randomized SVD fallback")
            # Simple randomized SVD approximation
            k = min(fcfg.pca_k, min(X_centered.shape) - 1)
            random_state = np.random.RandomState(42)
            Q = random_state.randn(X_centered.shape[1], k + 10)
            Y = X_centered @ Q
            Q, _ = np.linalg.qr(Y)
            B = Q.T @ X_centered
            Uhat, S, Vt = np.linalg.svd(B, full_matrices=False)
            U = Q @ Uhat
        
        # Keep top k components
        k = min(fcfg.pca_k, len(S))
        components = Vt[:k, :]  # (k, n_dates)
        loadings = U[:, :k]     # (n_algos, k)
        explained_var = S[:k] ** 2 / np.sum(S ** 2)
        
        logger.info(f"    Top {k} PCA components explain {explained_var.sum()*100:.1f}% variance")
        
        # ========================================
        # Step 4: Project ALL dates onto PCA
        # ========================================
        logger.info("  Projecting all dates onto PCA factors...")
        
        # Build full returns matrix for all dates
        returns_full = (
            panel_df
            .pivot(
                on="date",
                index="algo_id",
                values="ret_1d",
            )
        )
        
        full_algo_ids = returns_full["algo_id"].to_list()
        full_date_cols = [c for c in returns_full.columns if c != "algo_id"]
        full_dates = [date.fromisoformat(str(d)) for d in full_date_cols]
        
        X_full = returns_full.select(full_date_cols).to_numpy()
        X_full = np.nan_to_num(X_full, nan=0.0)
        
        # Standardize same way
        if fcfg.standardize_returns:
            # Use stored mean/std for algos that were in PCA
            algo_id_to_idx = {a: i for i, a in enumerate(algo_ids)}
            X_full_std = np.zeros_like(X_full)
            for i, aid in enumerate(full_algo_ids):
                if aid in algo_id_to_idx:
                    j = algo_id_to_idx[aid]
                    X_full_std[i] = (X_full[i] - X_mean[j]) / X_std[j]
                else:
                    # For algos not in PCA, use their own mean/std
                    m = np.nanmean(X_full[i])
                    s = np.nanstd(X_full[i])
                    s = s if s > 1e-10 else 1.0
                    X_full_std[i] = (X_full[i] - m) / s
            X_full = X_full_std
        
        # Center by date
        X_full_centered = X_full - np.nanmean(X_full, axis=0, keepdims=True)
        X_full_centered = np.nan_to_num(X_full_centered, nan=0.0)
        
        # Build algo loadings for full algo set
        algo_id_to_loading = {a: loadings[i] for i, a in enumerate(algo_ids)}
        full_loadings = np.zeros((len(full_algo_ids), k))
        for i, aid in enumerate(full_algo_ids):
            if aid in algo_id_to_loading:
                full_loadings[i] = algo_id_to_loading[aid]
            # else: zeros (new algo)
        
        # Project: factor_scores = loadings.T @ X_centered
        # Shape: (k, n_all_dates)
        factor_scores = full_loadings.T @ X_full_centered
        
        # ========================================
        # Step 5: Build factor_timeseries DataFrame
        # ========================================
        logger.info("  Building factor_timeseries DataFrame...")
        
        factor_data = {"date": full_dates}
        
        # Add f_mkt (join from earlier computation)
        f_mkt_dict = dict(zip(f_mkt["date"].to_list(), f_mkt["f_mkt"].to_list()))
        factor_data["f_mkt"] = [f_mkt_dict.get(d, 0.0) for d in full_dates]
        
        # Add PCA factors
        for i in range(k):
            factor_data[f"f_pca_{i+1}"] = factor_scores[i, :].tolist()
        
        factor_df = pl.DataFrame(factor_data).sort("date")
        
        # Validate: no nulls, factors have variance
        for col in factor_df.columns:
            if col == "date":
                continue
            null_count = factor_df[col].null_count()
            var = factor_df[col].var()
            if null_count > 0:
                store.add_warning(f"Factor {col} has {null_count} nulls")
            if var is not None and var < 1e-12:
                store.add_warning(f"Factor {col} has near-zero variance")
        
        # Write outputs
        logger.info(f"  Writing {len(factor_df)} rows to {out_factors_path}")
        factor_df.write_parquet(str(out_factors_path), compression="zstd")
        store.register_artifact("factor_timeseries", out_factors_path, factor_df)
        
        # Save PCA model
        logger.info(f"  Saving PCA model to {out_pca_path}")
        np.savez_compressed(
            out_pca_path,
            components=components,
            loadings=loadings,
            explained_var=explained_var,
            algo_ids=np.array(algo_ids, dtype=object),
            X_mean=X_mean.flatten(),
            X_std=X_std.flatten(),
            pca_k=k,
            standardize=fcfg.standardize_returns,
        )
        
        # Register PCA model (just track path, not as parquet)
        store._latest["pca_model"] = out_pca_path
        
        logger.info("Factor time series complete!")
        
        return {
            "factor_timeseries": out_factors_path,
            "pca_model": out_pca_path,
        }
    
    def validate(self, store: "ArtifactStore") -> None:
        """Validate factor outputs."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        out_path = store.get_latest("factor_timeseries")
        if out_path is None:
            raise ValueError("factor_timeseries not found in store")
        
        df = pl.read_parquet(str(out_path))
        
        # Check schema
        if "date" not in df.columns:
            raise ValueError("factor_timeseries missing 'date' column")
        if "f_mkt" not in df.columns:
            raise ValueError("factor_timeseries missing 'f_mkt' column")
        
        pca_cols = [c for c in df.columns if c.startswith("f_pca_")]
        if not pca_cols:
            store.add_warning("No PCA factors found in factor_timeseries")
        
        # Check for nulls
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                store.add_warning(f"factor_timeseries.{col} has {null_count} nulls")
        
        # Check dates are sorted
        dates = df["date"].to_list()
        if dates != sorted(dates):
            raise ValueError("factor_timeseries dates not sorted")
        
        logger.info("Factor timeseries validation passed!")
