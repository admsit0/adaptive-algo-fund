#!/usr/bin/env python3
"""
CLI for running personality feature engineering pipeline.

Usage:
    # With preprocess run ID
    python -m athenai.scripts.run_personality --preprocess-run-id 20260119_225417_8512cdac4218
    
    # With latest preprocess run
    python -m athenai.scripts.run_personality --latest
    
    # With config file
    python -m athenai.scripts.run_personality --config configs/personality.yaml
    
    # With custom min_obs
    python -m athenai.scripts.run_personality --latest --min-obs-personality 120
    
    # Overwrite existing
    python -m athenai.scripts.run_personality --latest --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_latest_preprocess_run(cache_dir: Path) -> str | None:
    """Find the most recent preprocess run ID by timestamp in name."""
    if not cache_dir.exists():
        return None
    
    # List directories that look like run IDs (YYYYMMDD_HHMMSS_...)
    run_dirs = []
    for d in cache_dir.iterdir():
        if d.is_dir() and len(d.name) > 15 and d.name[8] == "_":
            # Check if it has preprocess outputs
            if (d / "algos_panel.parquet").exists():
                run_dirs.append(d.name)
    
    if not run_dirs:
        return None
    
    # Sort by name (timestamp prefix makes this chronological)
    run_dirs.sort(reverse=True)
    return run_dirs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run personality feature engineering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with specific preprocess run
    python -m athenai.scripts.run_personality --preprocess-run-id 20260119_225417_8512cdac4218
    
    # Run with latest preprocess run
    python -m athenai.scripts.run_personality --latest
    
    # Run with config file
    python -m athenai.scripts.run_personality --config configs/personality.yaml
    
    # Custom parameters
    python -m athenai.scripts.run_personality --latest --min-obs-personality 200 --overwrite
        """,
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--preprocess-run-id",
        type=str,
        help="Run ID of preprocessing outputs to use",
    )
    input_group.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent preprocess run",
    )
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to personality.yaml config file",
    )
    
    # Parameters
    parser.add_argument(
        "--min-obs-personality",
        type=int,
        default=120,
        help="Minimum observations for GOOD output (default: 120)",
    )
    parser.add_argument(
        "--asof-date",
        type=str,
        default=None,
        help="Only use data up to this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory (default: data/cache)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/reports",
        help="Reports directory (default: data/reports)",
    )
    
    # Control
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation step",
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from athenai.features.personality import PersonalityConfig
    from athenai.pipelines.personality import PersonalityPipeline
    
    cache_dir = Path(args.cache_dir)
    reports_dir = Path(args.reports_dir)
    
    # Determine preprocess run ID
    if args.config:
        cfg = PersonalityConfig.from_yaml(args.config)
        # Override cache_dir if specified
        if args.cache_dir != "data/cache":
            cfg = PersonalityConfig(
                preprocess_run_id=cfg.preprocess_run_id,
                min_obs_personality=cfg.min_obs_personality,
                asof_date=cfg.asof_date,
                cache_dir=cache_dir,
                reports_dir=reports_dir,
            )
    else:
        if args.latest:
            preprocess_run_id = find_latest_preprocess_run(cache_dir)
            if preprocess_run_id is None:
                print(f"ERROR: No preprocess runs found in {cache_dir}", file=sys.stderr)
                sys.exit(1)
            print(f"Using latest preprocess run: {preprocess_run_id}")
        else:
            preprocess_run_id = args.preprocess_run_id
        
        # Verify preprocess run exists
        preprocess_dir = cache_dir / preprocess_run_id
        if not preprocess_dir.exists():
            print(f"ERROR: Preprocess run not found: {preprocess_dir}", file=sys.stderr)
            sys.exit(1)
        
        required_files = ["algos_panel.parquet", "algos_meta.parquet", "algos_meta_good.parquet"]
        for f in required_files:
            if not (preprocess_dir / f).exists():
                print(f"ERROR: Required file not found: {preprocess_dir / f}", file=sys.stderr)
                sys.exit(1)
        
        cfg = PersonalityConfig(
            preprocess_run_id=preprocess_run_id,
            min_obs_personality=args.min_obs_personality,
            asof_date=args.asof_date,
            cache_dir=cache_dir,
            reports_dir=reports_dir,
        )
    
    # Run pipeline
    print(f"\n{'='*60}")
    print("PERSONALITY FEATURE ENGINEERING PIPELINE")
    print(f"{'='*60}")
    print(f"Preprocess run ID: {cfg.preprocess_run_id}")
    print(f"min_obs_personality: {cfg.min_obs_personality}")
    print(f"asof_date: {cfg.asof_date or 'None (use all data)'}")
    print(f"Cache dir: {cfg.cache_dir}")
    print(f"Reports dir: {cfg.reports_dir}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Validate: {not args.no_validate}")
    print(f"{'='*60}\n")
    
    pipeline = PersonalityPipeline(cfg)
    
    try:
        results = pipeline.run(
            overwrite=args.overwrite,
            validate=not args.no_validate,
        )
        
        print("\n✓ Pipeline completed successfully!")
        print("\nOutputs:")
        for name, path in results.items():
            print(f"  - {name}: {path}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
