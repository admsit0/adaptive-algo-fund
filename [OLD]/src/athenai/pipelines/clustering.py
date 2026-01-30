"""
Clustering Pipeline orchestrator.

Runs all steps to build cluster universes:
1. Build factor timeseries (PCA + market factor)
2. Compute algo factor exposures (betas)
3. Fit cluster models + create cluster_map
4. Build cluster timeseries + alive masks
5. Build cluster features daily (RL-ready)

Usage:
    python -m athenai.scripts.run_clustering --config configs/clustering.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from athenai.factors.factor_timeseries import BuildFactorTimeseriesStep
from athenai.factors.exposures import BuildAlgoFactorExposuresStep
from athenai.clustering.build_clusters import FitClusterModelsStep
from athenai.clustering.cluster_timeseries import (
    BuildClusterArtifactsStep,
    BuildClusterFeaturesDailyStep,
)

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore
    from athenai.factors.config import ClusteringConfig


class ClusteringPipeline:
    """
    Orchestrator for the full clustering pipeline.
    
    Steps:
    1. BuildFactorTimeseriesStep: f_mkt, f_pca_1..K
    2. BuildAlgoFactorExposuresStep: beta_mkt, beta_pca_*, r2
    3. FitClusterModelsStep: cluster_model, cluster_map, cluster_meta
    4. BuildClusterArtifactsStep: cluster_timeseries, cluster_alive_mask
    5. BuildClusterFeaturesDailyStep: cluster_features_daily
    """
    
    def __init__(self, cfg: "ClusteringConfig", store: "ArtifactStore"):
        self.cfg = cfg
        self.store = store
        
        # Initialize steps
        self.steps = [
            BuildFactorTimeseriesStep(),
            BuildAlgoFactorExposuresStep(),
            FitClusterModelsStep(),
            BuildClusterArtifactsStep(),
            BuildClusterFeaturesDailyStep(),
        ]
    
    def run(
        self,
        skip_factors: bool = False,
        skip_exposures: bool = False,
        skip_clustering: bool = False,
        cluster_sets: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """
        Run the full pipeline.
        
        Args:
            skip_factors: Skip factor timeseries step (use cached)
            skip_exposures: Skip exposures step (use cached)
            skip_clustering: Skip cluster fitting (use cached)
            cluster_sets: List of cluster_set_ids to process (None = all)
            overwrite: Overwrite existing outputs
            
        Returns:
            Dictionary of all output paths
        """
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        all_outputs = {}
        
        # Filter cluster_sets if specified
        if cluster_sets is not None:
            original_sets = self.cfg.cluster_sets
            self.cfg.cluster_sets = [
                cs for cs in original_sets 
                if cs.cluster_set_id in cluster_sets
            ]
            logger.info(f"Filtering to cluster_sets: {cluster_sets}")
        
        # Step 1: Factor timeseries
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Build Factor Timeseries")
        logger.info("="*60)
        
        if skip_factors:
            logger.info("Skipping (--skip-factors)")
        else:
            step1 = self.steps[0]
            outputs = step1.run(self.store, self.cfg, overwrite=overwrite)
            step1.validate(self.store)
            all_outputs.update(outputs)
        
        # Step 2: Algo factor exposures
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Compute Algo Factor Exposures")
        logger.info("="*60)
        
        if skip_exposures:
            logger.info("Skipping (--skip-exposures)")
        else:
            step2 = self.steps[1]
            outputs = step2.run(self.store, self.cfg, overwrite=overwrite)
            step2.validate(self.store)
            all_outputs.update(outputs)
        
        # Step 3: Fit cluster models
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Fit Cluster Models")
        logger.info("="*60)
        
        if skip_clustering:
            logger.info("Skipping (--skip-clustering)")
        else:
            step3 = self.steps[2]
            outputs = step3.run(self.store, self.cfg, overwrite=overwrite)
            step3.validate(self.store)
            all_outputs.update(outputs)
        
        # Step 4: Cluster timeseries
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Build Cluster Timeseries")
        logger.info("="*60)
        
        step4 = self.steps[3]
        outputs = step4.run(self.store, self.cfg, overwrite=overwrite)
        step4.validate(self.store)
        all_outputs.update(outputs)
        
        # Step 5: Cluster features daily
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Build Cluster Features Daily")
        logger.info("="*60)
        
        step5 = self.steps[4]
        outputs = step5.run(self.store, self.cfg, overwrite=overwrite)
        step5.validate(self.store)
        all_outputs.update(outputs)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        self._print_summary(logger, all_outputs)
        
        return all_outputs
    
    def _print_summary(self, logger, outputs: dict) -> None:
        """Print summary of outputs."""
        
        logger.info(f"Total outputs: {len(outputs)}")
        
        # Group by type
        by_type = {}
        for key in outputs:
            if key.startswith("cluster_map_"):
                by_type.setdefault("cluster_map", []).append(key)
            elif key.startswith("cluster_meta_"):
                by_type.setdefault("cluster_meta", []).append(key)
            elif key.startswith("cluster_model_"):
                by_type.setdefault("cluster_model", []).append(key)
            elif key.startswith("cluster_timeseries_"):
                by_type.setdefault("cluster_timeseries", []).append(key)
            elif key.startswith("cluster_alive_mask_"):
                by_type.setdefault("cluster_alive_mask", []).append(key)
            elif key.startswith("cluster_features_daily_"):
                by_type.setdefault("cluster_features_daily", []).append(key)
            else:
                by_type.setdefault("other", []).append(key)
        
        for type_name, keys in sorted(by_type.items()):
            logger.info(f"  {type_name}: {len(keys)} files")
        
        # List cluster_sets processed
        cluster_sets = set()
        for key in outputs:
            for prefix in ["cluster_map_", "cluster_timeseries_"]:
                if key.startswith(prefix):
                    cluster_sets.add(key.replace(prefix, ""))
        
        if cluster_sets:
            logger.info(f"\nCluster sets processed: {sorted(cluster_sets)}")
        
        # Warnings
        if self.store.manifest.warnings:
            logger.warning(f"\nWarnings ({len(self.store.manifest.warnings)}):")
            for w in self.store.manifest.warnings:
                logger.warning(f"  - {w}")


def run_clustering_pipeline(
    config_path: str | Path,
    cache_dir: str | Path | None = None,
    preprocess_run_id: str | None = None,
    personality_run_id: str | None = None,
    cluster_sets: list[str] | None = None,
    skip_factors: bool = False,
    skip_exposures: bool = False,
    skip_clustering: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Main entry point for clustering pipeline.
    
    Args:
        config_path: Path to clustering.yaml config
        cache_dir: Override cache directory
        preprocess_run_id: Run ID from preprocessing pipeline
        personality_run_id: Run ID from personality pipeline
        cluster_sets: List of cluster_set_ids to process (None = all)
        skip_factors: Skip factor timeseries step
        skip_exposures: Skip exposures step
        skip_clustering: Skip cluster fitting step
        overwrite: Overwrite existing outputs
        verbose: Enable verbose logging
        
    Returns:
        Dictionary of output paths
    """
    from athenai.core.artifacts import ArtifactStore
    from athenai.core.logging import setup_logging, get_logger
    from athenai.factors.config import ClusteringConfig
    
    import logging
    level = logging.INFO if verbose else logging.WARNING
    setup_logging(level=level)
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("CLUSTERING PIPELINE")
    logger.info("="*60)
    
    # Load config
    config_path = Path(config_path)
    logger.info(f"Loading config from {config_path}")
    cfg = ClusteringConfig.from_yaml(config_path)
    
    # Override paths if provided
    if cache_dir is not None:
        cfg.cache_dir = Path(cache_dir)
    
    if preprocess_run_id is not None:
        cfg.preprocess_run_id = preprocess_run_id
    
    if personality_run_id is not None:
        cfg.personality_run_id = personality_run_id
    
    logger.info(f"  Cache dir: {cfg.cache_dir}")
    logger.info(f"  Preprocess run: {cfg.preprocess_run_id}")
    logger.info(f"  Personality run: {cfg.personality_run_id}")
    logger.info(f"  Cluster sets: {[cs.cluster_set_id for cs in cfg.cluster_sets]}")
    
    # Create output directory
    output_dir = cfg.cache_dir / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize artifact store
    store = ArtifactStore(root=cfg.cache_dir, config=cfg, run_id=cfg.run_id)
    store.ensure_dirs()
    
    # Save config
    cfg.save(output_dir / "clustering_config.yaml")
    
    # Run pipeline
    pipeline = ClusteringPipeline(cfg, store)
    outputs = pipeline.run(
        skip_factors=skip_factors,
        skip_exposures=skip_exposures,
        skip_clustering=skip_clustering,
        cluster_sets=cluster_sets,
        overwrite=overwrite,
    )
    
    logger.info(f"\nAll outputs in: {output_dir}")
    
    return outputs
