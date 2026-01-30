#!/usr/bin/env python
"""
CLI for running the clustering pipeline.

Usage:
    python -m athenai.scripts.run_clustering --config configs/clustering.yaml
    
    # Skip steps (use cached):
    python -m athenai.scripts.run_clustering --config configs/clustering.yaml --skip-factors
    
    # Process specific cluster sets:
    python -m athenai.scripts.run_clustering --config configs/clustering.yaml \\
        --cluster-sets behavioral_k100_v1 corrpca_k60_v1
    
    # Overwrite existing outputs:
    python -m athenai.scripts.run_clustering --config configs/clustering.yaml --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the clustering pipeline to create cluster universes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/clustering.yaml"),
        help="Path to clustering config YAML (default: configs/clustering.yaml)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory from config"
    )
    
    parser.add_argument(
        "--preprocess-run-id",
        type=str,
        default=None,
        help="Run ID from preprocessing pipeline (required input)"
    )
    
    parser.add_argument(
        "--personality-run-id",
        type=str,
        default=None,
        help="Run ID from personality pipeline (required input)"
    )
    
    parser.add_argument(
        "--cluster-sets",
        nargs="+",
        type=str,
        default=None,
        help="List of cluster_set_ids to process (default: all)"
    )
    
    parser.add_argument(
        "--skip-factors",
        action="store_true",
        help="Skip factor timeseries step (use cached)"
    )
    
    parser.add_argument(
        "--skip-exposures",
        action="store_true",
        help="Skip exposures step (use cached)"
    )
    
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip cluster fitting (use cached models)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Import here to avoid slow startup
    from athenai.pipelines.clustering import run_clustering_pipeline
    
    try:
        outputs = run_clustering_pipeline(
            config_path=args.config,
            cache_dir=args.cache_dir,
            preprocess_run_id=args.preprocess_run_id,
            personality_run_id=args.personality_run_id,
            cluster_sets=args.cluster_sets,
            skip_factors=args.skip_factors,
            skip_exposures=args.skip_exposures,
            skip_clustering=args.skip_clustering,
            overwrite=args.overwrite,
            verbose=not args.quiet,
        )
        
        print(f"\n✅ Pipeline complete! {len(outputs)} outputs generated.")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"\n❌ Validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
