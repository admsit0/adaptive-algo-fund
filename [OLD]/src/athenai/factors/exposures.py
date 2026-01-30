"""
Algo factor exposures computation.

For each algo, run regression:
    r_algo = beta_mkt * f_mkt + sum(beta_pca_k * f_pca_k) + epsilon

Outputs:
- algo_factor_exposures.parquet (ALL)
- algo_factor_exposures_good.parquet (GOOD)

Schema:
- algo_id, n_obs_used, beta_mkt, beta_pca_1..K, r2, resid_vol_ann
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


class BuildAlgoFactorExposuresStep(PipelineStep):
    """
    Compute factor exposures (betas) for each algo.
    
    For each algo:
    1. Join algo returns with factor time series
    2. Run regression (OLS or ridge)
    3. Extract betas, R², residual volatility
    
    Produces:
    - algo_factor_exposures.parquet (ALL algos)
    - algo_factor_exposures_good.parquet (GOOD only)
    """
    
    name = "build_algo_factor_exposures"
    
    def inputs(self) -> list[str]:
        return ["algos_panel", "factor_timeseries", "algos_meta_good"]
    
    def outputs(self) -> list[str]:
        return ["algo_factor_exposures", "algo_factor_exposures_good"]
    
    def run(
        self,
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Compute factor exposures for all algos."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        ecfg = cfg.exposure_config
        fcfg = cfg.factor_config
        
        # Paths
        preprocess_dir = cfg.cache_dir / cfg.preprocess_run_id
        panel_path = preprocess_dir / "algos_panel.parquet"
        meta_good_path = preprocess_dir / "algos_meta_good.parquet"
        factor_path = store.get_latest("factor_timeseries")
        
        if factor_path is None:
            factor_path = store.artifact_path(fcfg.factor_timeseries_name)
        
        out_all_path = store.artifact_path(ecfg.exposures_all_name)
        out_good_path = store.artifact_path(ecfg.exposures_good_name)
        
        if out_all_path.exists() and out_good_path.exists() and not overwrite:
            logger.info("Exposure outputs exist, skipping")
            return {
                "algo_factor_exposures": out_all_path,
                "algo_factor_exposures_good": out_good_path,
            }
        
        logger.info("Computing factor exposures...")
        
        # Load data
        panel_df = pl.read_parquet(str(panel_path)).select(["algo_id", "date", "ret_1d"])
        factor_df = pl.read_parquet(str(factor_path))
        meta_good = pl.read_parquet(str(meta_good_path))
        good_ids_set = set(meta_good["algo_id"].to_list())
        
        # Apply asof_date if set
        if fcfg.asof_date:
            asof = date.fromisoformat(fcfg.asof_date)
            panel_df = panel_df.filter(pl.col("date") <= asof)
            factor_df = factor_df.filter(pl.col("date") <= asof)
        
        # Join panel with factors
        joined = panel_df.join(factor_df, on="date", how="inner")
        
        # Identify factor columns
        factor_cols = [c for c in factor_df.columns if c != "date"]
        logger.info(f"  Factor columns: {factor_cols}")
        
        # Get unique algos
        algo_ids = joined["algo_id"].unique().to_list()
        logger.info(f"  Computing exposures for {len(algo_ids)} algos...")
        
        # Prepare output
        ann_sqrt = np.sqrt(ecfg.annualization)
        results = []
        
        # Ridge lambda
        ridge_lambda = ecfg.ridge_lambda if ecfg.method == "ridge_ols" else 0.0
        
        # Process each algo
        for i, algo_id in enumerate(algo_ids):
            if (i + 1) % 2000 == 0:
                logger.info(f"    Processed {i+1}/{len(algo_ids)} algos...")
            
            algo_data = joined.filter(pl.col("algo_id") == algo_id)
            n_obs = len(algo_data)
            
            if n_obs < ecfg.min_obs:
                # Skip, but still record with nulls
                result = {
                    "algo_id": algo_id,
                    "n_obs_used": n_obs,
                    "r2": None,
                    "resid_vol_ann": None,
                }
                for fc in factor_cols:
                    result[f"beta_{fc}"] = None
                results.append(result)
                continue
            
            # Extract y (returns) and X (factors)
            y = algo_data["ret_1d"].to_numpy()
            X = algo_data.select(factor_cols).to_numpy()
            
            # Handle NaN
            valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
            y = y[valid_mask]
            X = X[valid_mask]
            n_valid = len(y)
            
            if n_valid < ecfg.min_obs:
                result = {
                    "algo_id": algo_id,
                    "n_obs_used": n_valid,
                    "r2": None,
                    "resid_vol_ann": None,
                }
                for fc in factor_cols:
                    result[f"beta_{fc}"] = None
                results.append(result)
                continue
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(n_valid), X])
            
            # Solve (X'X + lambda*I)^-1 X'y
            XtX = X_with_intercept.T @ X_with_intercept
            if ridge_lambda > 0:
                XtX += ridge_lambda * np.eye(XtX.shape[0])
            
            try:
                betas = np.linalg.solve(XtX, X_with_intercept.T @ y)
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                betas = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            # Predictions and residuals
            y_pred = X_with_intercept @ betas
            residuals = y - y_pred
            
            # R²
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r2 = np.clip(r2, 0.0, 1.0)
            
            # Residual volatility (annualized)
            resid_vol = np.std(residuals) * ann_sqrt
            
            # Build result
            result = {
                "algo_id": algo_id,
                "n_obs_used": n_valid,
                "r2": float(r2),
                "resid_vol_ann": float(resid_vol),
            }
            
            # Betas (skip intercept at index 0)
            for j, fc in enumerate(factor_cols):
                beta_val = betas[j + 1]
                result[f"beta_{fc}"] = float(beta_val) if np.isfinite(beta_val) else None
            
            results.append(result)
        
        # Build DataFrame
        exposures_df = pl.DataFrame(results)
        
        logger.info(f"  Exposures computed for {len(exposures_df)} algos")
        
        # Write ALL
        logger.info(f"  Writing ALL exposures to {out_all_path}")
        exposures_df.write_parquet(str(out_all_path), compression="zstd")
        store.register_artifact("algo_factor_exposures", out_all_path, exposures_df)
        
        # Build GOOD
        exposures_good = exposures_df.filter(pl.col("algo_id").is_in(list(good_ids_set)))
        
        logger.info(f"  Writing GOOD exposures ({len(exposures_good)} algos) to {out_good_path}")
        exposures_good.write_parquet(str(out_good_path), compression="zstd")
        store.register_artifact("algo_factor_exposures_good", out_good_path, exposures_good)
        
        # Warnings
        null_r2 = exposures_df["r2"].null_count()
        if null_r2 > 0:
            store.add_warning(f"Exposures: {null_r2} algos with null r2 (insufficient obs)")
        
        # Check beta distribution
        if "beta_f_mkt" in exposures_df.columns:
            beta_mkt_vals = exposures_df["beta_f_mkt"].drop_nulls()
            if len(beta_mkt_vals) > 0:
                extreme = (beta_mkt_vals.abs() > 10).sum()
                if extreme > 0:
                    store.add_warning(f"Exposures: {extreme} algos with |beta_mkt| > 10")
        
        logger.info("Factor exposures complete!")
        
        return {
            "algo_factor_exposures": out_all_path,
            "algo_factor_exposures_good": out_good_path,
        }
    
    def validate(self, store: "ArtifactStore") -> None:
        """Validate exposure outputs."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        for name in ["algo_factor_exposures", "algo_factor_exposures_good"]:
            path = store.get_latest(name)
            if path is None:
                raise ValueError(f"{name} not found in store")
            
            df = pl.read_parquet(str(path))
            
            # Check schema
            required = ["algo_id", "n_obs_used", "r2", "resid_vol_ann"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")
            
            # Check unique algo_id
            if df["algo_id"].n_unique() != len(df):
                raise ValueError(f"{name} has duplicate algo_ids")
            
            # Check r2 in [0, 1] where not null
            r2_valid = df.filter(pl.col("r2").is_not_null())["r2"]
            if len(r2_valid) > 0:
                r2_out = r2_valid.filter((r2_valid < 0) | (r2_valid > 1.01))
                if len(r2_out) > 0:
                    store.add_warning(f"{name}: {len(r2_out)} algos with r2 out of [0,1]")
            
            # Check resid_vol >= 0
            rv_valid = df.filter(pl.col("resid_vol_ann").is_not_null())["resid_vol_ann"]
            if len(rv_valid) > 0:
                rv_neg = rv_valid.filter(rv_valid < 0)
                if len(rv_neg) > 0:
                    raise ValueError(f"{name}: {len(rv_neg)} algos with negative resid_vol_ann")
        
        logger.info("Exposure validation passed!")
