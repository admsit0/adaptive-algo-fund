"""
Personality feature engineering pipeline orchestrator.

Runs the BuildAlgoPersonalityStaticStep with:
- Timing and monitoring
- Validation
- Manifest generation
- Report generation

Usage:
    cfg = PersonalityConfig(preprocess_run_id="20260119_225417_8512cdac4218")
    pipeline = PersonalityPipeline(cfg)
    results = pipeline.run(overwrite=False)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from athenai.core.artifacts import ArtifactStore, StepTiming
from athenai.core.logging import get_logger, set_run_context, setup_logging
from athenai.features.personality import BuildAlgoPersonalityStaticStep, PersonalityConfig

if TYPE_CHECKING:
    pass


class PersonalityPipeline:
    """
    Orchestrates personality feature engineering.
    
    Steps:
    1. build_algo_personality_static - Compute all personality features
    
    Each step:
    - Checks if outputs exist (skip if not overwrite)
    - Runs with timing
    - Validates outputs
    - Logs progress
    
    After completion:
    - Saves manifest.json
    - Generates report_personality_<run_id>.md
    
    Usage:
        cfg = PersonalityConfig(
            preprocess_run_id="20260119_225417_8512cdac4218",
            min_obs_personality=120,
        )
        pipeline = PersonalityPipeline(cfg)
        results = pipeline.run(overwrite=False)
    """
    
    def __init__(self, cfg: PersonalityConfig):
        self.cfg = cfg
        self.logger = get_logger()
    
    def run(
        self,
        overwrite: bool = False,
        validate: bool = True,
        run_id: str | None = None,
    ) -> dict[str, Path]:
        """
        Run the personality feature engineering pipeline.
        
        Args:
            overwrite: Whether to overwrite existing outputs
            validate: Whether to run validation after the step
            run_id: Optional run ID (auto-generated if None)
        
        Returns:
            Dict mapping artifact names to paths
        """
        # Create artifact store with new run_id (child of preprocess)
        store = ArtifactStore(
            root=self.cfg.cache_dir,
            config=self.cfg,
            run_id=run_id,
        )
        store.ensure_dirs()
        
        # Add parent reference to manifest
        store.manifest.config["parent_run_id"] = self.cfg.preprocess_run_id
        
        set_run_context(run_id=store.run_id)
        setup_logging(log_file=store.run_dir / "pipeline.log")
        
        self.logger.info("Starting personality feature engineering pipeline")
        self.logger.info(f"Run ID: {store.run_id}")
        self.logger.info(f"Parent (preprocess) run ID: {self.cfg.preprocess_run_id}")
        self.logger.info(f"Output dir: {store.run_dir}")
        
        all_outputs: dict[str, Path] = {}
        
        # Run the personality step
        step = BuildAlgoPersonalityStaticStep(cfg=self.cfg)
        
        start_time = datetime.now()
        try:
            set_run_context(step_name=step.name)
            
            # Run step
            outputs = step.run(store, self.cfg, overwrite=overwrite)
            all_outputs.update(outputs)
            
            # Validate
            if validate:
                step.validate(store)
            
        except Exception as e:
            store.add_error(f"Step {step.name} failed: {e}")
            self.logger.error(f"Step {step.name} failed: {e}")
            raise
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Record timing
        timing = StepTiming(
            step_name=step.name,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
        )
        store.add_timing(timing)
        
        # Save manifest
        manifest_path = store.save_manifest()
        self.logger.info(f"Saved manifest: {manifest_path}")
        
        # Generate report
        self._generate_report(store)
        
        # Print summary
        print("\n" + "=" * 60)
        print(store.summary())
        print("=" * 60)
        print(f"\nTotal duration: {duration:.2f}s")
        
        return all_outputs
    
    def _generate_report(self, store: ArtifactStore) -> Path:
        """Generate personality report with percentiles and extremes."""
        import polars as pl
        
        report_path = self.cfg.reports_dir / f"report_personality_{store.run_id}.md"
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        
        out_all_path = store.get_latest("algo_personality_static")
        out_good_path = store.get_latest("algo_personality_static_good")
        
        if out_all_path is None or out_good_path is None:
            self.logger.warning("Cannot generate report: outputs not found")
            return report_path
        
        df_all = pl.read_parquet(str(out_all_path))
        df_good = pl.read_parquet(str(out_good_path))
        
        lines = [
            f"# Personality Feature Report",
            f"",
            f"**Run ID:** {store.run_id}",
            f"**Parent Run ID:** {self.cfg.preprocess_run_id}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**min_obs_personality:** {self.cfg.min_obs_personality}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total algos (ALL) | {len(df_all):,} |",
            f"| Good algos (GOOD) | {len(df_good):,} |",
            f"| Date range | {df_all['start_date'].min()} to {df_all['end_date'].max()} |",
            f"",
        ]
        
        # Percentile tables
        percentile_cols = [
            "sharpe_ann", "vol_ann", "max_drawdown",
            "ulcer_index", "max_drawdown_duration",
            "skew", "excess_kurtosis", "tail_ratio_95_05",
            "autocorr_ret_1", "trend_r2",
        ]
        
        lines.append("## Percentiles (GOOD universe)")
        lines.append("")
        lines.append("| Metric | p1 | p5 | p50 | p95 | p99 |")
        lines.append("|--------|---:|---:|----:|----:|----:|")
        
        for col in percentile_cols:
            if col not in df_good.columns:
                continue
            vals = df_good[col].drop_nulls()
            if len(vals) == 0:
                continue
            p1 = vals.quantile(0.01)
            p5 = vals.quantile(0.05)
            p50 = vals.quantile(0.50)
            p95 = vals.quantile(0.95)
            p99 = vals.quantile(0.99)
            lines.append(f"| {col} | {p1:.4f} | {p5:.4f} | {p50:.4f} | {p95:.4f} | {p99:.4f} |")
        
        lines.append("")
        
        # Top extremes
        lines.append("## Top 10 Extremes (GOOD universe)")
        lines.append("")
        
        # Top 10 Sharpe
        lines.append("### Highest Sharpe")
        lines.append("")
        top_sharpe = df_good.sort("sharpe_ann", descending=True, nulls_last=True).head(10)
        lines.append("| algo_id | sharpe_ann | vol_ann | max_drawdown |")
        lines.append("|---------|----------:|--------:|-------------:|")
        for row in top_sharpe.iter_rows(named=True):
            lines.append(f"| {row['algo_id']} | {row['sharpe_ann']:.3f} | {row['vol_ann']:.3f} | {row['max_drawdown']:.3f} |")
        lines.append("")
        
        # Top 10 Ulcer
        lines.append("### Highest Ulcer Index (most pain)")
        lines.append("")
        top_ulcer = df_good.sort("ulcer_index", descending=True, nulls_last=True).head(10)
        lines.append("| algo_id | ulcer_index | max_drawdown | time_in_drawdown |")
        lines.append("|---------|------------:|-------------:|-----------------:|")
        for row in top_ulcer.iter_rows(named=True):
            lines.append(f"| {row['algo_id']} | {row['ulcer_index']:.4f} | {row['max_drawdown']:.3f} | {row['time_in_drawdown']:.3f} |")
        lines.append("")
        
        # Worst 10 Max Drawdown
        lines.append("### Worst Max Drawdown")
        lines.append("")
        worst_dd = df_good.sort("max_drawdown").head(10)
        lines.append("| algo_id | max_drawdown | max_drawdown_duration | sharpe_ann |")
        lines.append("|---------|-------------:|----------------------:|-----------:|")
        for row in worst_dd.iter_rows(named=True):
            sharpe = row['sharpe_ann'] if row['sharpe_ann'] is not None else 0
            lines.append(f"| {row['algo_id']} | {row['max_drawdown']:.3f} | {row['max_drawdown_duration']} | {sharpe:.3f} |")
        lines.append("")
        
        # Warnings
        lines.append("## Warnings")
        lines.append("")
        
        # Count issues
        zero_std = len(df_all.filter(pl.col("ret_std") < 1e-10))
        null_tail = df_all["tail_ratio_95_05"].null_count()
        null_autocorr = df_all["autocorr_ret_1"].null_count()
        low_obs = len(df_all.filter(pl.col("n_obs") < self.cfg.min_obs_personality))
        
        lines.append(f"- Algos with ret_std ≈ 0: {zero_std} ({zero_std/len(df_all)*100:.1f}%)")
        lines.append(f"- Algos with null tail_ratio: {null_tail} ({null_tail/len(df_all)*100:.1f}%)")
        lines.append(f"- Algos with null autocorr_ret_1: {null_autocorr} ({null_autocorr/len(df_all)*100:.1f}%)")
        lines.append(f"- Algos with n_obs < {self.cfg.min_obs_personality}: {low_obs} ({low_obs/len(df_all)*100:.1f}%)")
        lines.append("")
        
        # Manifest warnings
        if store.manifest.warnings:
            lines.append("### Pipeline Warnings")
            lines.append("")
            for w in store.manifest.warnings:
                lines.append(f"- {w}")
            lines.append("")
        
        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        self.logger.info(f"Report written to {report_path}")
        return report_path
