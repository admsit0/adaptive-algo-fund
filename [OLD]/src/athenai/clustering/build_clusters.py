"""
Fit cluster models and assign algos to clusters.

For each cluster_set in config:
1. Build feature matrix (personality + exposures + optional regime)
2. Scale features
3. Fit clustering model
4. Assign cluster_id to each algo
5. Save model and cluster_map

Outputs per cluster_set:
- cluster_model_<id>.pkl
- cluster_map_<id>.parquet
- cluster_meta_<id>.parquet
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from athenai.data.base import PipelineStep
from athenai.clustering.cluster_models import get_cluster_model, get_scaler

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore
    from athenai.factors.config import ClusteringConfig, ClusterSetConfig


class FitClusterModelsStep(PipelineStep):
    """
    Fit clustering models and create cluster_map for each cluster_set.
    
    Families:
    - behavioral: personality + exposures (+ regime if available)
    - correlation_embedding: PCA embedding of returns
    - regime_specialists: regime signature features
    
    Produces per cluster_set:
    - cluster_model_<id>.pkl: Trained model + scaler
    - cluster_map_<id>.parquet: (algo_id, cluster_id, score)
    - cluster_meta_<id>.parquet: Per-cluster statistics
    """
    
    name = "fit_cluster_models"
    
    def inputs(self) -> list[str]:
        return [
            "algo_personality_static_good",
            "algo_factor_exposures_good",
            "algos_meta_good",
        ]
    
    def outputs(self) -> list[str]:
        # Dynamic based on cluster_sets
        return []  # Populated at runtime
    
    def run(
        self,
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        overwrite: bool = False,
        macro_features: pl.DataFrame = None,
        macro_mode: str = "train",  # "train" (usa reales), "predict" (usa proxies)
    ) -> dict[str, Path]:
        """Fit all cluster models."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        outputs = {}
        
        # Load input data
        personality_path = cfg.cache_dir / cfg.personality_run_id / "algo_personality_static_good.parquet"
        exposures_path = store.get_latest("algo_factor_exposures_good")
        meta_good_path = cfg.cache_dir / cfg.preprocess_run_id / "algos_meta_good.parquet"
        
        if exposures_path is None:
            exposures_path = store.artifact_path(cfg.exposure_config.exposures_good_name)
        
        logger.info("Loading feature data...")
        personality_df = pl.read_parquet(str(personality_path))
        exposures_df = pl.read_parquet(str(exposures_path))
        meta_df = pl.read_parquet(str(meta_good_path))
        
        logger.info(f"  Personality: {len(personality_df)} algos")
        logger.info(f"  Exposures: {len(exposures_df)} algos")
        # --- Añadir variables macroeconómicas ---
        if macro_features is not None:
            logger.info(f"  Añadiendo variables macroeconómicas ({macro_mode}) a features de clustering: {macro_features.columns}")
            # Unir por fecha (o por la clave adecuada)
            # Suponemos que macro_features tiene columna 'date' y personality_df también
            if 'date' in personality_df.columns and 'date' in macro_features.columns:
                # Merge por fecha (para clustering diario)
                personality_df = personality_df.join(macro_features, on="date", how="left")
            else:
                # Si no hay fecha, hacer merge por columna común o añadir broadcast
                for col in macro_features.columns:
                    if col != 'date':
                        personality_df = personality_df.with_columns([
                            pl.lit(macro_features[col][0]).alias(col)
                        ])
        else:
            logger.info("  No se añaden variables macroeconómicas adicionales al clustering.")
        
        # Load PCA model for correlation_embedding family
        pca_model_path = store.get_latest("pca_model")
        if pca_model_path is None:
            pca_model_path = store.artifact_path(cfg.factor_config.pca_model_name)
        
        pca_data = None
        if pca_model_path.exists():
            pca_data = np.load(pca_model_path, allow_pickle=True)
        
        # Process each cluster set
        for cs in cfg.cluster_sets:
            logger.info(f"\n{'='*50}")
            logger.info(f"Fitting cluster set: {cs.cluster_set_id}")
            logger.info(f"  Family: {cs.family}, K={cs.k}, Algorithm: {cs.algorithm}")
            
            cs_outputs = self._fit_single_cluster_set(
                cs, store, cfg, 
                personality_df, exposures_df, meta_df, 
                pca_data, overwrite, logger
            )
            outputs.update(cs_outputs)
        
        return outputs
    
    def _fit_single_cluster_set(
        self,
        cs: "ClusterSetConfig",
        store: "ArtifactStore",
        cfg: "ClusteringConfig",
        personality_df: pl.DataFrame,
        exposures_df: pl.DataFrame,
        meta_df: pl.DataFrame,
        pca_data: dict | None,
        overwrite: bool,
        logger,
    ) -> dict[str, Path]:
        """Fit a single cluster set."""
        
        model_path = store.artifact_path(f"cluster_model_{cs.cluster_set_id}.pkl")
        map_path = store.artifact_path(f"cluster_map_{cs.cluster_set_id}.parquet")
        meta_path = store.artifact_path(f"cluster_meta_{cs.cluster_set_id}.parquet")
        
        if model_path.exists() and map_path.exists() and meta_path.exists() and not overwrite:
            logger.info(f"  Outputs exist, skipping")
            return {
                f"cluster_model_{cs.cluster_set_id}": model_path,
                f"cluster_map_{cs.cluster_set_id}": map_path,
                f"cluster_meta_{cs.cluster_set_id}": meta_path,
            }
        
        # Build feature matrix based on family
        X, algo_ids, feature_names = self._build_feature_matrix(
            cs, personality_df, exposures_df, meta_df, pca_data, logger
        )
        
        logger.info(f"  Feature matrix: {X.shape[0]} algos x {X.shape[1]} features")
        
        # Handle NaN
        nan_mask = np.any(np.isnan(X), axis=1)
        if nan_mask.sum() > 0:
            logger.warning(f"  {nan_mask.sum()} algos with NaN features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        scaler = get_scaler(cs.scaler)
        if scaler is not None:
            X_scaled = scaler.fit_transform(X)
            logger.info(f"  Applied {cs.scaler} scaling")
        else:
            X_scaled = X
        
        # Fit clustering model
        logger.info(f"  Fitting {cs.algorithm} with k={cs.k}...")
        model = get_cluster_model(cs.algorithm, k=cs.k, seed=cs.seed)
        model.fit(X_scaled)
        
        labels = model.labels_
        distances = model.get_distances(X_scaled)
        
        logger.info(f"  Inertia: {model.inertia_:.2f}")
        
        # Check cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_empty = cs.k - len(unique_labels)
        min_size = counts.min()
        max_size = counts.max()
        
        logger.info(f"  Cluster sizes: min={min_size}, max={max_size}, empty={n_empty}")
        
        if n_empty > 0:
            store.add_warning(f"{cs.cluster_set_id}: {n_empty} empty clusters")
        
        small_clusters = (counts < cs.min_cluster_size).sum()
        if small_clusters > 0:
            store.add_warning(f"{cs.cluster_set_id}: {small_clusters} clusters with size < {cs.min_cluster_size}")
        
        # Build cluster_map
        cluster_map = pl.DataFrame({
            "algo_id": algo_ids,
            "cluster_id": labels.astype(np.int32),
            "cluster_set_id": [cs.cluster_set_id] * len(algo_ids),
            "score": distances,
        })
        
        logger.info(f"  Writing cluster_map to {map_path}")
        cluster_map.write_parquet(str(map_path), compression="zstd")
        store.register_artifact(f"cluster_map_{cs.cluster_set_id}", map_path, cluster_map)
        
        # Build cluster_meta
        cluster_meta = self._build_cluster_meta(
            cs, cluster_map, personality_df, exposures_df, X_scaled, labels
        )
        
        logger.info(f"  Writing cluster_meta to {meta_path}")
        cluster_meta.write_parquet(str(meta_path), compression="zstd")
        store.register_artifact(f"cluster_meta_{cs.cluster_set_id}", meta_path, cluster_meta)
        
        # Save model + scaler
        model_bundle = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "config": cs.to_dict(),
        }
        
        logger.info(f"  Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(model_bundle, f)
        
        store._latest[f"cluster_model_{cs.cluster_set_id}"] = model_path
        
        return {
            f"cluster_model_{cs.cluster_set_id}": model_path,
            f"cluster_map_{cs.cluster_set_id}": map_path,
            f"cluster_meta_{cs.cluster_set_id}": meta_path,
        }
    
    def _build_feature_matrix(
        self,
        cs: "ClusterSetConfig",
        personality_df: pl.DataFrame,
        exposures_df: pl.DataFrame,
        meta_df: pl.DataFrame,
        pca_data: dict | None,
        logger,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Build feature matrix based on cluster family."""
        
        if cs.family == "behavioral" or cs.feature_source == "static_features":
            return self._build_behavioral_features(
                personality_df, exposures_df, logger
            )
        
        elif cs.family == "correlation_embedding" or cs.feature_source == "pca_embedding":
            if pca_data is None:
                raise ValueError("PCA model required for correlation_embedding family")
            return self._build_pca_embedding_features(
                personality_df, pca_data, logger
            )
        
        elif cs.family == "regime_specialists" or cs.feature_source == "regime_signature":
            # Placeholder: use behavioral features if no regime data
            logger.warning("  Regime data not available, falling back to behavioral features")
            return self._build_behavioral_features(
                personality_df, exposures_df, logger
            )
        
        else:
            raise ValueError(f"Unknown family/feature_source: {cs.family}/{cs.feature_source}")
    
    def _build_behavioral_features(
        self,
        personality_df: pl.DataFrame,
        exposures_df: pl.DataFrame,
        logger,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Build features from personality + exposures."""
        
        # Personality features to use
        pers_cols = [
            "sharpe_ann", "vol_ann", "max_drawdown", "hit_rate",
            "ulcer_index", "time_in_drawdown", "max_drawdown_duration",
            "skew", "excess_kurtosis", "tail_ratio_95_05", "sortino_ann",
            "autocorr_ret_1", "autocorr_absret_1", "momentum_log_120",
            "trend_slope", "trend_r2",
            "sharpe_drift", "vol_drift",
        ]
        
        # Filter to existing columns
        pers_cols = [c for c in pers_cols if c in personality_df.columns]
        
        # Exposure features to use
        exp_cols = [c for c in exposures_df.columns if c.startswith("beta_") or c in ["r2", "resid_vol_ann"]]
        
        # Join personality and exposures
        joined = personality_df.select(["algo_id"] + pers_cols).join(
            exposures_df.select(["algo_id"] + exp_cols),
            on="algo_id",
            how="inner"
        )
        
        feature_names = pers_cols + exp_cols
        algo_ids = joined["algo_id"].to_list()
        X = joined.select(feature_names).to_numpy()
        
        logger.info(f"    Behavioral features: {len(feature_names)}")
        
        return X, algo_ids, feature_names
    
    def _build_pca_embedding_features(
        self,
        personality_df: pl.DataFrame,
        pca_data: dict,
        logger,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Build features from PCA loadings (algo-level)."""
        
        pca_algo_ids = pca_data["algo_ids"].tolist()
        loadings = pca_data["loadings"]  # (n_algos, k)
        
        # Filter to algos in personality (GOOD universe)
        good_ids = set(personality_df["algo_id"].to_list())
        
        valid_indices = [i for i, a in enumerate(pca_algo_ids) if a in good_ids]
        algo_ids = [pca_algo_ids[i] for i in valid_indices]
        X = loadings[valid_indices]
        
        k = X.shape[1]
        feature_names = [f"pca_loading_{i+1}" for i in range(k)]
        
        logger.info(f"    PCA embedding features: {k} dimensions, {len(algo_ids)} algos")
        
        return X, algo_ids, feature_names
    
    def _build_cluster_meta(
        self,
        cs: "ClusterSetConfig",
        cluster_map: pl.DataFrame,
        personality_df: pl.DataFrame,
        exposures_df: pl.DataFrame,
        X_scaled: np.ndarray,
        labels: np.ndarray,
    ) -> pl.DataFrame:
        """Build cluster_meta with per-cluster statistics."""
        
        # Join cluster_map with personality for stats
        joined = cluster_map.join(personality_df, on="algo_id", how="left")
        
        if "beta_f_mkt" in exposures_df.columns:
            joined = joined.join(
                exposures_df.select(["algo_id", "beta_f_mkt"]),
                on="algo_id",
                how="left"
            )
        
        # Aggregate per cluster
        meta = (
            joined
            .group_by("cluster_id")
            .agg([
                pl.len().alias("n_members_total"),
                pl.col("sharpe_ann").mean().alias("avg_sharpe"),
                pl.col("vol_ann").mean().alias("avg_vol_ann"),
                pl.col("max_drawdown").mean().alias("avg_mdd"),
                pl.col("beta_f_mkt").mean().alias("avg_beta_mkt") if "beta_f_mkt" in joined.columns else pl.lit(None).alias("avg_beta_mkt"),
            ])
            .with_columns([
                pl.lit(cs.cluster_set_id).alias("cluster_set_id"),
            ])
            .sort("cluster_id")
        )
        
        # Compute dispersion (variance of scaled features within cluster)
        dispersions = []
        unique_clusters = sorted(np.unique(labels))
        for c in unique_clusters:
            mask = labels == c
            if mask.sum() > 1:
                cluster_X = X_scaled[mask]
                dispersion = np.mean(np.var(cluster_X, axis=0))
            else:
                dispersion = 0.0
            dispersions.append(dispersion)
        
        # Add dispersion to meta
        dispersion_df = pl.DataFrame({
            "cluster_id": unique_clusters,
            "dispersion": dispersions,
        })
        
        meta = meta.join(dispersion_df, on="cluster_id", how="left")
        
        # Reorder columns
        meta = meta.select([
            "cluster_set_id", "cluster_id", "n_members_total",
            "avg_sharpe", "avg_vol_ann", "avg_mdd", "avg_beta_mkt", "dispersion"
        ])
        
        return meta
    
    def validate(self, store: "ArtifactStore") -> None:
        """Validate cluster outputs."""
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        # Find all cluster_map outputs
        for key, path in store._latest.items():
            if key.startswith("cluster_map_"):
                df = pl.read_parquet(str(path))
                
                # Check unique algo_id
                if df["algo_id"].n_unique() != len(df):
                    raise ValueError(f"{key}: duplicate algo_ids")
                
                # Check cluster_id range
                k = df["cluster_id"].max() + 1
                if df["cluster_id"].min() < 0:
                    raise ValueError(f"{key}: negative cluster_id")
                
                logger.info(f"{key}: {len(df)} algos, {k} clusters")
        
        logger.info("Cluster validation passed!")
