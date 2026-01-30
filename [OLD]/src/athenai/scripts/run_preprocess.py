#!/usr/bin/env python
"""
CLI script to run the preprocessing pipeline.

Usage:
    python -m athenai.scripts.run_preprocess --config configs/preprocess.yaml
    python -m athenai.scripts.run_preprocess --root-dir data/raw/datos_competicion
    python -m athenai.scripts.run_preprocess --config configs/preprocess.yaml --overwrite
    
Or via entry point (after pip install):
    athenai-preprocess --config configs/preprocess.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AthenAI preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configs/preprocess.yaml
  %(prog)s --root-dir datos_competicion --cache-dir data/cache
  %(prog)s --config configs/preprocess.yaml --overwrite --no-validate
        """,
    )
    
    # Config file (mutually exclusive with manual paths)
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to YAML config file",
    )
    
    # Manual path overrides
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="Path to datos_competicion folder (overrides config)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Path to cache/output folder (overrides config)",
    )
    
    # Pipeline options
    parser.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after each step",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Run only a specific step by name",
    )
    
    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Import here to avoid slow startup
    from athenai.core.config import PreprocessConfig, load_config
    from athenai.core.logging import setup_logging
    from athenai.pipelines.preprocess import PreprocessPipeline
    
    import logging
    
    # Setup logging level
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    setup_logging(level=level)
    
    # Load or build config
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        cfg = load_config(args.config)
    else:
        # Default config
        cfg = PreprocessConfig()
    
    # Apply overrides
    if args.root_dir:
        cfg = PreprocessConfig(
            root_dir=args.root_dir,
            **{k: v for k, v in cfg.to_dict().items() if k != "root_dir"}
        )
    if args.cache_dir:
        cfg = PreprocessConfig(
            cache_dir=args.cache_dir,
            **{k: v for k, v in cfg.to_dict().items() if k != "cache_dir"}
        )
    
    # Validate config
    try:
        cfg.validate()
    except Exception as e:
        print(f"Config validation error: {e}", file=sys.stderr)
        return 1
    
    # Run pipeline
    pipeline = PreprocessPipeline(cfg)
    
    try:
        if args.step:
            outputs = pipeline.run_step(
                args.step,
                overwrite=args.overwrite,
                validate=not args.no_validate,
            )
        else:
            outputs = pipeline.run(
                overwrite=args.overwrite,
                validate=not args.no_validate,
            )
        
        if not args.quiet:
            print("\n✓ Pipeline completed successfully!")
            print(f"  Outputs: {len(outputs)} artifacts")
            for name, path in outputs.items():
                print(f"    - {name}: {path}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
