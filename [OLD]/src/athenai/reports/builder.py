"""
Report builder for preprocessing pipeline.

Generates Markdown reports with:
- Artifact summaries
- Filtering statistics
- Data quality metrics
- Alive algorithms timeline
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore, Manifest


class ReportBuilder:
    """
    Build Markdown reports from pipeline runs.
    
    Usage:
        builder = ReportBuilder(store)
        report = builder.build_preprocess_report()
        builder.save(report, Path("data/reports/preprocess_run123.md"))
    """
    
    def __init__(self, store: "ArtifactStore"):
        self.store = store
        self.manifest = store.manifest
    
    def build_preprocess_report(self) -> str:
        """Build a comprehensive preprocessing report."""
        lines = [
            f"# Preprocessing Report",
            f"",
            f"**Run ID:** {self.manifest.run_id}",
            f"**Config Hash:** {self.manifest.config_hash}",
            f"**Created:** {self.manifest.created_at}",
            f"**Total Duration:** {self.manifest.total_duration_seconds:.2f}s",
            f"",
        ]
        
        # Artifacts summary
        lines.append("## Artifacts")
        lines.append("")
        lines.append("| Artifact | Rows | Columns | Size | Date Range |")
        lines.append("|----------|------|---------|------|------------|")
        
        for name, info in self.manifest.outputs.items():
            size_mb = info.file_size_bytes / (1024 * 1024)
            date_range = f"{info.min_date} - {info.max_date}" if info.min_date else "-"
            lines.append(
                f"| {name} | {info.row_count:,} | {info.col_count} | "
                f"{size_mb:.2f} MB | {date_range} |"
            )
        lines.append("")
        
        # Filtering statistics
        lines.extend(self._filtering_section())
        
        # Data quality
        lines.extend(self._quality_section())
        
        # Timings
        lines.extend(self._timing_section())
        
        # Warnings
        if self.manifest.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.manifest.warnings:
                lines.append(f"- {w}")
            lines.append("")
        
        # Config
        lines.append("## Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.manifest.config, indent=2))
        lines.append("```")
        
        return "\n".join(lines)
    
    def _filtering_section(self) -> list[str]:
        """Build filtering statistics section."""
        lines = [
            "## Filtering Statistics",
            "",
        ]
        
        meta_info = self.manifest.outputs.get("algos_meta")
        good_info = self.manifest.outputs.get("algos_meta_good")
        
        if meta_info and good_info:
            total = meta_info.row_count
            good = good_info.row_count
            filtered = total - good
            pct = filtered / total * 100 if total > 0 else 0
            
            lines.append(f"- **Total algorithms:** {total:,}")
            lines.append(f"- **Good universe:** {good:,}")
            lines.append(f"- **Filtered out:** {filtered:,} ({pct:.1f}%)")
            lines.append("")
            
            # Try to get more details from meta
            meta_path = self.store.get_latest("algos_meta")
            if meta_path and meta_path.exists():
                df = pl.read_parquet(str(meta_path))
                
                n_constant = df.filter(pl.col("is_constant")).height
                n_low_obs = df.filter(
                    pl.col("n_obs") < self.manifest.config.get("min_obs", 60)
                ).height
                n_low_cov = df.filter(
                    pl.col("coverage_ratio") < self.manifest.config.get("min_coverage", 0.7)
                ).height
                
                lines.append("### Filtering Reasons")
                lines.append("")
                lines.append(f"- Constant prices: {n_constant:,}")
                lines.append(f"- Low observations: {n_low_obs:,}")
                lines.append(f"- Low coverage: {n_low_cov:,}")
                lines.append("")
        
        return lines
    
    def _quality_section(self) -> list[str]:
        """Build data quality section."""
        lines = [
            "## Data Quality",
            "",
        ]
        
        # Panel stats
        panel_path = self.store.get_latest("algos_panel")
        if panel_path and panel_path.exists():
            df = pl.read_parquet(str(panel_path))
            
            # Return distribution
            ret_stats = df.select([
                pl.col("ret_1d").min().alias("min"),
                pl.col("ret_1d").max().alias("max"),
                pl.col("ret_1d").mean().alias("mean"),
                pl.col("ret_1d").std().alias("std"),
                pl.col("ret_1d").quantile(0.01).alias("p01"),
                pl.col("ret_1d").quantile(0.99).alias("p99"),
            ]).to_dicts()[0]
            
            lines.append("### Daily Returns (ret_1d)")
            lines.append("")
            lines.append(f"- Min: {ret_stats['min']:.4f}")
            lines.append(f"- Max: {ret_stats['max']:.4f}")
            lines.append(f"- Mean: {ret_stats['mean']:.6f}")
            lines.append(f"- Std: {ret_stats['std']:.4f}")
            lines.append(f"- P01: {ret_stats['p01']:.4f}")
            lines.append(f"- P99: {ret_stats['p99']:.4f}")
            lines.append("")
        
        # Meta stats
        meta_path = self.store.get_latest("algos_meta")
        if meta_path and meta_path.exists():
            df = pl.read_parquet(str(meta_path))
            
            cov_stats = df.select([
                pl.col("coverage_ratio").min().alias("min"),
                pl.col("coverage_ratio").max().alias("max"),
                pl.col("coverage_ratio").median().alias("median"),
                pl.col("n_obs").min().alias("min_obs"),
                pl.col("n_obs").max().alias("max_obs"),
                pl.col("n_obs").median().alias("med_obs"),
            ]).to_dicts()[0]
            
            lines.append("### Coverage Statistics")
            lines.append("")
            lines.append(f"- Coverage ratio: {cov_stats['min']:.2f} - {cov_stats['max']:.2f} (median: {cov_stats['median']:.2f})")
            lines.append(f"- Observations: {cov_stats['min_obs']:.0f} - {cov_stats['max_obs']:.0f} (median: {cov_stats['med_obs']:.0f})")
            lines.append("")
        
        return lines
    
    def _timing_section(self) -> list[str]:
        """Build timing section."""
        lines = [
            "## Execution Timings",
            "",
            "| Step | Duration (s) |",
            "|------|--------------|",
        ]
        
        for timing in self.manifest.timings:
            lines.append(f"| {timing.step_name} | {timing.duration_seconds:.2f} |")
        
        lines.append(f"| **Total** | **{self.manifest.total_duration_seconds:.2f}** |")
        lines.append("")
        
        return lines
    
    def build_alive_summary(self) -> str:
        """Build summary of alive algorithms over time."""
        lines = [
            "# Alive Algorithms Summary",
            "",
        ]
        
        alive_path = self.store.get_latest("alive_intervals")
        if not alive_path or not alive_path.exists():
            lines.append("No alive_intervals data available.")
            return "\n".join(lines)
        
        df = pl.read_parquet(str(alive_path))
        
        # Get date range
        min_date = df["start_date"].min()
        max_date = df["end_date"].max()
        
        lines.append(f"**Date range:** {min_date} to {max_date}")
        lines.append(f"**Total algorithms:** {len(df):,}")
        lines.append("")
        
        # Monthly alive counts
        lines.append("## Monthly Alive Counts")
        lines.append("")
        
        # Generate monthly dates
        dates = pl.date_range(min_date, max_date, interval="1mo", eager=True)
        
        monthly_counts = []
        for d in dates:
            count = df.filter(
                (pl.col("start_date") <= d) & (pl.col("end_date") >= d)
            ).height
            monthly_counts.append({"month": d, "alive_count": count})
        
        lines.append("| Month | Alive Algos |")
        lines.append("|-------|-------------|")
        for row in monthly_counts[-24:]:  # Last 24 months
            lines.append(f"| {row['month']} | {row['alive_count']:,} |")
        
        return "\n".join(lines)
    
    def save(self, content: str, path: Path) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    
    def generate_all(self, output_dir: Path) -> list[Path]:
        """Generate all reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        
        # Main preprocessing report
        report_path = output_dir / f"preprocess_{self.manifest.run_id}.md"
        self.save(self.build_preprocess_report(), report_path)
        paths.append(report_path)
        
        # Alive summary
        alive_path = output_dir / f"alive_summary_{self.manifest.run_id}.md"
        self.save(self.build_alive_summary(), alive_path)
        paths.append(alive_path)
        
        return paths


def generate_report(store: "ArtifactStore", output_dir: Path) -> list[Path]:
    """Convenience function to generate all reports."""
    builder = ReportBuilder(store)
    return builder.generate_all(output_dir)
