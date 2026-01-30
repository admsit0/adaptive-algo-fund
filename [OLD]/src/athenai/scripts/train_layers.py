#!/usr/bin/env python
"""
CLI for training and running the proxy layers pipeline.

Usage:
    # Train models (requires external data):
    python -m athenai.scripts.train_layers --config configs/layers.yaml --train
    
    # Predict only (no external data needed):
    python -m athenai.scripts.train_layers --config configs/layers.yaml --predict
    
    # Build enriched RL state:
    python -m athenai.scripts.train_layers --config configs/layers.yaml --build-enriched
    
    # Skip steps:
    python -m athenai.scripts.train_layers --config configs/layers.yaml --train --skip-fetch
    
    # Specify cluster set:
    python -m athenai.scripts.train_layers --config configs/layers.yaml --train \\
        --cluster-set-id behavioral_k100_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train proxy models and build enriched RL state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/layers.yaml"),
        help="Path to layers config YAML (default: configs/layers.yaml)"
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Run full training pipeline (fetch data, train models, generate predictions)"
    )
    mode_group.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction only (no external data, uses trained models)"
    )
    mode_group.add_argument(
        "--build-enriched",
        action="store_true",
        help="Build enriched RL state from existing predictions"
    )
    
    # Override config values
    parser.add_argument(
        "--clustering-run-id",
        type=str,
        default=None,
        help="Override clustering_run_id from config"
    )
    
    parser.add_argument(
        "--cluster-set-id",
        type=str,
        default=None,
        help="Override cluster_set_id from config"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory from config"
    )
    
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override models directory from config"
    )
    
    # Skip flags
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip external data fetching (use cached)"
    )
    
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip universe features computation (use cached)"
    )
    
    # Other flags
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs"
    )
    
    parser.add_argument(
        "--fred-api-key",
        type=str,
        default=None,
        help="FRED API key (or set FRED_API_KEY env var)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    import logging
    args = parse_args()
    
    # Setup logging
    from athenai.core.logging import setup_logging, get_logger
    log_level = logging.WARNING if args.quiet else logging.INFO
    setup_logging(level=log_level, use_rich=False)  # Disable rich for terminal compatibility
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("AthenAI Proxy Layers Pipeline (Paso 3)")
    logger.info("="*60)
    
    # Load config
    from athenai.layers.config import LayersConfig
    
    if args.config.exists():
        logger.info(f"Loading config from {args.config}")
        cfg = LayersConfig.from_yaml(args.config)
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        cfg = LayersConfig()
    
    # Apply overrides
    if args.clustering_run_id:
        cfg.clustering_run_id = args.clustering_run_id
    
    if args.cluster_set_id:
        cfg.cluster_set_id = args.cluster_set_id
    
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    
    if args.models_dir:
        cfg.models_dir = args.models_dir
    
    if args.fred_api_key:
        cfg.fred_api_key = args.fred_api_key
    
    # Validate config
    try:
        cfg.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    logger.info(f"Clustering run: {cfg.clustering_run_id}")
    logger.info(f"Cluster set: {cfg.cluster_set_id}")
    
    # Create artifact store
    from athenai.core.artifacts import ArtifactStore
    
    store = ArtifactStore(
        root=cfg.cache_dir,
        config=cfg,
        run_id=cfg.run_id,
    )
    store.ensure_dirs()
    
    # Create pipeline
    from athenai.pipelines.layers import LayersPipeline
    
    pipeline = LayersPipeline(cfg=cfg, store=store)
    
    try:
        if args.train:
            logger.info("\nRunning TRAINING pipeline...")
            outputs = pipeline.run_train(
                skip_fetch=args.skip_fetch,
                skip_features=args.skip_features,
                overwrite=args.overwrite,
            )
            
            logger.info("\n" + "="*60)
            logger.info("Output artifacts:")
            for name, path in outputs.items():
                logger.info(f"  {name}: {path}")
            
        elif args.predict:
            logger.info("\nRunning PREDICTION pipeline (no external data)...")
            enriched = pipeline.run_predict()
            
            logger.info(f"\nEnriched features shape: {enriched.shape}")
            
        elif args.build_enriched:
            logger.info("\nBuilding enriched RL state from existing predictions...")
            
            # Load existing predictions
            preds_path = store.artifact_path(
                cfg.proxy_preds_name.format(cluster_set_id=cfg.cluster_set_id)
            )
            
            if not preds_path.exists():
                logger.error(f"Predictions not found: {preds_path}")
                logger.error("Run with --train or --predict first")
                return 1
            
            import polars as pl
            predictions_df = pl.read_parquet(str(preds_path))
            
            enriched_path = pipeline._build_enriched_state(predictions_df)
            logger.info(f"Built enriched state: {enriched_path}")
        
        logger.info("\nPipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
