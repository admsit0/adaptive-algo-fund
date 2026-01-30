"""
Personality feature engineering for algorithms.

Builds static per-algorithm features capturing:
- Performance & risk (Sharpe, vol, hit rate, max drawdown)
- Drawdown dynamics (Ulcer index, time in drawdown, duration)
- Distribution / tails (skew, kurtosis, tail ratios, sortino)
- Temporal behaviour (autocorr, momentum, trend fit)
- Stability (sharpe drift between halves)

Outputs:
- algo_personality_static.parquet (ALL algos)
- algo_personality_static_good.parquet (GOOD universe filtered)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from athenai.data.base import PipelineStep
from athenai.core.validation import assert_schema, assert_unique_keys

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore


@dataclass(frozen=True)
class PersonalityConfig:
    """
    Configuration for personality feature engineering.
    
    Attributes:
        preprocess_run_id: Run ID of preprocessing outputs to use
        min_obs_personality: Minimum observations for GOOD output (default 120)
        annualization: Trading days per year (default 252)
        quantiles: Quantiles to compute (default 0.01, 0.05, 0.95, 0.99)
        eps: Small value for safe division (default 1e-12)
        momentum_lag: Lag for momentum calculation (default 120)
        clip_returns: Max abs return clip (default 0.50, should match preprocess)
        asof_date: If set, only use data up to this date (YYYY-MM-DD)
        
        cache_dir: Where preprocessed parquets are stored
        reports_dir: Where to write reports
    """
    preprocess_run_id: str = ""
    min_obs_personality: int = 120
    annualization: int = 252
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99)
    eps: float = 1e-12
    momentum_lag: int = 120
    clip_returns: float | None = 0.50
    asof_date: str | None = None  # YYYY-MM-DD or None
    
    # Directories
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    reports_dir: Path = field(default_factory=lambda: Path("data/reports"))
    
    # Output names
    personality_all_name: str = "algo_personality_static.parquet"
    personality_good_name: str = "algo_personality_static_good.parquet"
    
    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.cache_dir, str):
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if isinstance(self.reports_dir, str):
            object.__setattr__(self, "reports_dir", Path(self.reports_dir))
        if isinstance(self.quantiles, list):
            object.__setattr__(self, "quantiles", tuple(self.quantiles))
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.preprocess_run_id:
            raise ValueError("preprocess_run_id is required")
        preprocess_dir = self.cache_dir / self.preprocess_run_id
        if not preprocess_dir.exists():
            raise ValueError(f"Preprocess run dir does not exist: {preprocess_dir}")
        if self.min_obs_personality < 1:
            raise ValueError(f"min_obs_personality must be >= 1, got {self.min_obs_personality}")
        if self.annualization < 1:
            raise ValueError(f"annualization must be >= 1, got {self.annualization}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (with paths as strings)."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
    
    def config_hash(self) -> str:
        """Generate a short hash of this config for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonalityConfig":
        """Create config from dictionary."""
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: Path | str) -> "PersonalityConfig":
        """Load config from YAML file."""
        import yaml
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # Handle nested structure
        if "personality" in data:
            data = data["personality"]
        
        return cls.from_dict(data)


class BuildAlgoPersonalityStaticStep(PipelineStep):
    """
    Build static personality features per algorithm.
    
    Inputs:
        - algos_panel.parquet: (algo_id, date, close, ret_1d, logret_1d)
        - algos_meta.parquet: (algo_id, start_date, end_date, n_obs, ...)
        - algos_meta_good.parquet: (algo_id, ...) for good universe
    
    Outputs:
        - algo_personality_static.parquet: ALL algos
        - algo_personality_static_good.parquet: GOOD universe filtered
    
    Features computed:
        - Performance: ret_mean, ret_std, vol_ann, sharpe_ann, hit_rate, max_drawdown
        - Drawdown dynamics: ulcer_index, time_in_drawdown, avg_drawdown_depth, max_drawdown_duration
        - Distribution: skew, excess_kurtosis, quantiles, tail_ratio, downside_dev, sortino_ann
        - Temporal: autocorr_ret_1, autocorr_absret_1, momentum_log_120, trend_slope, trend_r2
        - Stability: sharpe_first_half, sharpe_second_half, sharpe_drift, vol_drift, return_drift
    """
    
    name = "build_algo_personality_static"
    
    def __init__(self, cfg: PersonalityConfig | None = None):
        self.cfg = cfg
    
    def inputs(self) -> list[str]:
        return ["algos_panel", "algos_meta", "algos_meta_good"]
    
    def outputs(self) -> list[str]:
        return ["algo_personality_static", "algo_personality_static_good"]
    
    def run(
        self,
        store: "ArtifactStore",
        cfg: PersonalityConfig,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """
        Run personality feature engineering.
        
        Strategy (3 passes for efficiency):
        1. Base stats: mean, std, quantiles, skew, kurtosis, hit_rate, autocorr, momentum
        2. Drawdown dynamics: max_dd, ulcer, time_in_dd, avg_depth, max_duration
        3. Stability: first/second half stats, drift metrics
        Then join all and output.
        """
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        # Paths
        preprocess_dir = cfg.cache_dir / cfg.preprocess_run_id
        panel_path = preprocess_dir / "algos_panel.parquet"
        meta_path = preprocess_dir / "algos_meta.parquet"
        meta_good_path = preprocess_dir / "algos_meta_good.parquet"
        
        out_all_path = store.artifact_path(cfg.personality_all_name)
        out_good_path = store.artifact_path(cfg.personality_good_name)
        
        if out_all_path.exists() and out_good_path.exists() and not overwrite:
            logger.info("Personality outputs exist, skipping (use --overwrite to regenerate)")
            return {
                "algo_personality_static": out_all_path,
                "algo_personality_static_good": out_good_path,
            }
        
        logger.info("Building algo personality features...")
        ann_sqrt = float(np.sqrt(cfg.annualization))
        
        # ===============================
        # PASS 1: Base return statistics
        # ===============================
        logger.info("Pass 1/3: Computing base return statistics...")
        
        panel_lf = pl.scan_parquet(str(panel_path))
        
        # Apply asof_date filter if specified
        if cfg.asof_date:
            asof = date.fromisoformat(cfg.asof_date)
            panel_lf = panel_lf.filter(pl.col("date") <= asof)
        
        # Sort for rolling/shift operations
        panel_lf = panel_lf.sort(["algo_id", "date"])
        
        # Create lagged return and abs return columns for autocorrelation
        panel_with_lags = (
            panel_lf
            .with_columns([
                pl.col("ret_1d").shift(1).over("algo_id").alias("ret_1d_lag1"),
                pl.col("ret_1d").abs().alias("abs_ret_1d"),
            ])
            .with_columns([
                pl.col("abs_ret_1d").shift(1).over("algo_id").alias("abs_ret_1d_lag1"),
                # Momentum: log(close) - log(close.shift(momentum_lag))
                (pl.col("close").log() - pl.col("close").shift(cfg.momentum_lag).log())
                .over("algo_id")
                .alias("momentum_raw"),
                # For trend: time index
                pl.cum_count("date").over("algo_id").alias("t_idx"),
                pl.col("close").log().alias("log_close"),
                # Positive/negative returns for downside
                pl.when(pl.col("ret_1d") < 0).then(pl.col("ret_1d")).otherwise(None).alias("ret_negative"),
            ])
        )
        
        # Aggregate base stats
        base_stats = (
            panel_with_lags
            .group_by("algo_id")
            .agg([
                # Count
                pl.len().alias("n_obs_calc"),
                
                # Return moments
                pl.col("ret_1d").mean().alias("ret_mean"),
                pl.col("ret_1d").std().alias("ret_std"),
                pl.col("ret_1d").skew().alias("skew"),
                pl.col("ret_1d").kurtosis().alias("kurtosis_raw"),  # excess kurtosis in polars
                
                # Quantiles
                pl.col("ret_1d").quantile(cfg.quantiles[0]).alias("ret_q01"),
                pl.col("ret_1d").quantile(cfg.quantiles[1]).alias("ret_q05"),
                pl.col("ret_1d").quantile(cfg.quantiles[2]).alias("ret_q95"),
                pl.col("ret_1d").quantile(cfg.quantiles[3]).alias("ret_q99"),
                
                # Hit rate
                (pl.col("ret_1d") > 0).mean().alias("hit_rate"),
                
                # Downside deviation
                pl.col("ret_negative").std().alias("downside_dev"),
                
                # Autocorrelation via Pearson correlation
                pl.corr("ret_1d", "ret_1d_lag1").alias("autocorr_ret_1"),
                pl.corr("abs_ret_1d", "abs_ret_1d_lag1").alias("autocorr_absret_1"),
                
                # Momentum (mean of momentum_raw over history)
                pl.col("momentum_raw").mean().alias("momentum_log_120"),
                
                # Trend regression sums (for manual slope/r2)
                pl.col("t_idx").mean().alias("t_mean"),
                pl.col("log_close").mean().alias("y_mean"),
                pl.col("t_idx").var().alias("t_var"),
                pl.col("log_close").var().alias("y_var"),
                ((pl.col("t_idx") - pl.col("t_idx").mean()) * (pl.col("log_close") - pl.col("log_close").mean())).mean().alias("cov_ty"),
            ])
            .with_columns([
                # Annualized volatility
                (pl.col("ret_std") * ann_sqrt).alias("vol_ann"),
                
                # Sharpe annualized
                pl.when(pl.col("ret_std") > cfg.eps)
                .then((pl.col("ret_mean") / pl.col("ret_std")) * ann_sqrt)
                .otherwise(None)
                .alias("sharpe_ann"),
                
                # Excess kurtosis (polars kurtosis is already Fisher's excess)
                pl.col("kurtosis_raw").alias("excess_kurtosis"),
                
                # Tail ratio: q95 / |q05|
                pl.when(pl.col("ret_q05").abs() > cfg.eps)
                .then(pl.col("ret_q95") / pl.col("ret_q05").abs())
                .otherwise(None)
                .alias("tail_ratio_95_05"),
                
                # Tail spread
                (pl.col("ret_q99") - pl.col("ret_q01")).alias("tail_spread_99_01"),
                
                # Sortino annualized
                pl.when(pl.col("downside_dev") > cfg.eps)
                .then((pl.col("ret_mean") / pl.col("downside_dev")) * ann_sqrt)
                .otherwise(None)
                .alias("sortino_ann"),
                
                # Trend slope: cov_ty / var_t
                pl.when(pl.col("t_var") > cfg.eps)
                .then(pl.col("cov_ty") / pl.col("t_var"))
                .otherwise(None)
                .alias("trend_slope"),
                
                # Trend R²: corr(t, y)^2 = cov_ty^2 / (var_t * var_y)
                pl.when((pl.col("t_var") > cfg.eps) & (pl.col("y_var") > cfg.eps))
                .then((pl.col("cov_ty") ** 2) / (pl.col("t_var") * pl.col("y_var")))
                .otherwise(None)
                .alias("trend_r2"),
            ])
            .select([
                "algo_id", "n_obs_calc", "ret_mean", "ret_std", "vol_ann", "sharpe_ann",
                "hit_rate", "ret_q01", "ret_q05", "ret_q95", "ret_q99",
                "tail_ratio_95_05", "tail_spread_99_01",
                "skew", "excess_kurtosis", "downside_dev", "sortino_ann",
                "autocorr_ret_1", "autocorr_absret_1", "momentum_log_120",
                "trend_slope", "trend_r2",
            ])
        ).collect()
        
        logger.info(f"  Base stats computed for {len(base_stats)} algos")
        
        # ===============================
        # PASS 2: Drawdown dynamics
        # ===============================
        logger.info("Pass 2/3: Computing drawdown dynamics...")
        
        # Reload panel (fresh scan for drawdown pass)
        panel_lf2 = pl.scan_parquet(str(panel_path))
        if cfg.asof_date:
            asof = date.fromisoformat(cfg.asof_date)
            panel_lf2 = panel_lf2.filter(pl.col("date") <= asof)
        
        panel_lf2 = panel_lf2.sort(["algo_id", "date"])
        
        # Compute drawdown per row
        dd_panel = (
            panel_lf2
            .with_columns([
                # Running max close
                pl.col("close").cum_max().over("algo_id").alias("cummax_close"),
            ])
            .with_columns([
                # Drawdown: close / cummax - 1 (negative or zero)
                (pl.col("close") / pl.col("cummax_close") - 1.0).alias("dd"),
            ])
            .with_columns([
                # Is in drawdown?
                (pl.col("dd") < 0).alias("in_dd"),
                # For ulcer index: dd^2
                (pl.col("dd") ** 2).alias("dd_sq"),
                # Abs drawdown for avg depth
                pl.col("dd").abs().alias("dd_abs"),
            ])
        )
        
        # For max drawdown duration, we need segment IDs
        # Segment starts when we transition from not-in-dd to in-dd
        dd_panel = dd_panel.with_columns([
            pl.col("in_dd").shift(1).fill_null(False).over("algo_id").alias("in_dd_prev"),
        ]).with_columns([
            (pl.col("in_dd") & ~pl.col("in_dd_prev")).alias("dd_start"),
        ]).with_columns([
            pl.col("dd_start").cum_sum().over("algo_id").alias("dd_segment_id"),
        ])
        
        # Aggregate drawdown stats per algo
        dd_stats = (
            dd_panel
            .group_by("algo_id")
            .agg([
                # Max drawdown (min of dd, which is negative)
                pl.col("dd").min().alias("max_drawdown"),
                
                # Ulcer index: sqrt(mean(dd^2))
                pl.col("dd_sq").mean().sqrt().alias("ulcer_index"),
                
                # Time in drawdown: proportion of days with dd < 0
                pl.col("in_dd").mean().alias("time_in_drawdown"),
                
                # Avg drawdown depth (when in drawdown)
                pl.when(pl.col("in_dd")).then(pl.col("dd_abs")).otherwise(None).mean().alias("avg_drawdown_depth"),
            ])
        ).collect()
        
        # Max drawdown duration requires a separate aggregation
        # Group by (algo_id, dd_segment_id) and count rows where in_dd=True
        # Then get max per algo
        dd_duration = (
            dd_panel
            .filter(pl.col("in_dd"))  # Only rows in drawdown
            .group_by(["algo_id", "dd_segment_id"])
            .agg(pl.len().alias("segment_length"))
            .group_by("algo_id")
            .agg(pl.col("segment_length").max().alias("max_drawdown_duration"))
        ).collect()
        
        # Join duration to dd_stats
        dd_stats = dd_stats.join(dd_duration, on="algo_id", how="left").with_columns([
            pl.col("max_drawdown_duration").fill_null(0).cast(pl.Int64),
        ])
        
        logger.info(f"  Drawdown stats computed for {len(dd_stats)} algos")
        
        # ===============================
        # PASS 3: Stability (first/second half)
        # ===============================
        logger.info("Pass 3/3: Computing stability metrics...")
        
        # Load meta for mid_date calculation
        meta_lf = pl.scan_parquet(str(meta_path))
        meta_df = meta_lf.select(["algo_id", "start_date", "end_date", "n_obs", "is_constant"]).collect()
        
        # Calculate mid_date per algo
        meta_with_mid = meta_df.with_columns([
            (pl.col("start_date") + (pl.col("end_date") - pl.col("start_date")) / 2).alias("mid_date"),
        ])
        
        # Reload panel
        panel_lf3 = pl.scan_parquet(str(panel_path))
        if cfg.asof_date:
            asof = date.fromisoformat(cfg.asof_date)
            panel_lf3 = panel_lf3.filter(pl.col("date") <= asof)
        
        # Join mid_date to panel
        panel_with_mid = (
            panel_lf3
            .join(meta_with_mid.lazy().select(["algo_id", "mid_date"]), on="algo_id", how="left")
            .with_columns([
                (pl.col("date") <= pl.col("mid_date")).alias("is_first_half"),
            ])
        )
        
        # Compute first half stats
        first_half_stats = (
            panel_with_mid
            .filter(pl.col("is_first_half"))
            .group_by("algo_id")
            .agg([
                pl.col("ret_1d").mean().alias("ret_mean_first"),
                pl.col("ret_1d").std().alias("ret_std_first"),
                pl.len().alias("n_first"),
            ])
            .with_columns([
                pl.when(pl.col("ret_std_first") > cfg.eps)
                .then((pl.col("ret_mean_first") / pl.col("ret_std_first")) * ann_sqrt)
                .otherwise(None)
                .alias("sharpe_first_half"),
                (pl.col("ret_std_first") * ann_sqrt).alias("vol_first_half"),
            ])
        ).collect()
        
        # Compute second half stats
        second_half_stats = (
            panel_with_mid
            .filter(~pl.col("is_first_half"))
            .group_by("algo_id")
            .agg([
                pl.col("ret_1d").mean().alias("ret_mean_second"),
                pl.col("ret_1d").std().alias("ret_std_second"),
                pl.len().alias("n_second"),
            ])
            .with_columns([
                pl.when(pl.col("ret_std_second") > cfg.eps)
                .then((pl.col("ret_mean_second") / pl.col("ret_std_second")) * ann_sqrt)
                .otherwise(None)
                .alias("sharpe_second_half"),
                (pl.col("ret_std_second") * ann_sqrt).alias("vol_second_half"),
            ])
        ).collect()
        
        # Join halves
        stability_stats = (
            first_half_stats
            .join(second_half_stats, on="algo_id", how="full", coalesce=True)
            .with_columns([
                (pl.col("sharpe_second_half") - pl.col("sharpe_first_half")).alias("sharpe_drift"),
                (pl.col("vol_second_half") - pl.col("vol_first_half")).alias("vol_drift"),
                (pl.col("ret_mean_second") - pl.col("ret_mean_first")).alias("return_drift"),
            ])
            .select([
                "algo_id",
                "sharpe_first_half", "sharpe_second_half", "sharpe_drift",
                "vol_first_half", "vol_second_half", "vol_drift",
                "ret_mean_first", "ret_mean_second", "return_drift",
            ])
        )
        
        logger.info(f"  Stability stats computed for {len(stability_stats)} algos")
        
        # ===============================
        # JOIN ALL & BUILD FINAL
        # ===============================
        logger.info("Joining all stats and building final output...")
        
        # Start with meta for authoritative n_obs, start_date, end_date
        final_df = (
            meta_with_mid
            .select(["algo_id", "start_date", "end_date", "n_obs", "is_constant"])
            .join(base_stats, on="algo_id", how="left")
            .join(dd_stats, on="algo_id", how="left")
            .join(stability_stats, on="algo_id", how="left")
        )
        
        # Final column selection (schema as specified)
        final_df = final_df.select([
            # Identity
            "algo_id",
            "n_obs",
            "start_date",
            "end_date",
            
            # Performance / risk
            "ret_mean",
            "ret_std",
            "vol_ann",
            "sharpe_ann",
            "hit_rate",
            "max_drawdown",
            
            # Drawdown dynamics
            "ulcer_index",
            "time_in_drawdown",
            "avg_drawdown_depth",
            "max_drawdown_duration",
            
            # Distribution / tails
            "ret_q01",
            "ret_q05",
            "ret_q95",
            "ret_q99",
            "tail_ratio_95_05",
            "tail_spread_99_01",
            "skew",
            "excess_kurtosis",
            "downside_dev",
            "sortino_ann",
            
            # Temporal behaviour
            "autocorr_ret_1",
            "autocorr_absret_1",
            "momentum_log_120",
            "trend_slope",
            "trend_r2",
            
            # Stability
            "sharpe_first_half",
            "sharpe_second_half",
            "sharpe_drift",
            "vol_drift",
            "return_drift",
            
            # Keep is_constant for filtering
            "is_constant",
        ])
        
        # Write ALL
        logger.info(f"Writing {len(final_df)} algos to {out_all_path}")
        final_df.write_parquet(str(out_all_path), compression="zstd")
        store.register_artifact("algo_personality_static", out_all_path, final_df)
        
        # Build GOOD: intersect with algos_meta_good + filter by min_obs_personality + exclude constant
        meta_good_ids = pl.scan_parquet(str(meta_good_path)).select(["algo_id"]).collect()
        
        good_df = (
            final_df
            .join(meta_good_ids, on="algo_id", how="inner")
            .filter(pl.col("n_obs") >= cfg.min_obs_personality)
            .filter(~pl.col("is_constant"))
        )
        
        # Drop is_constant from output (was just for filtering)
        good_df = good_df.drop("is_constant")
        final_df_out = final_df.drop("is_constant")
        
        # Re-write ALL without is_constant
        final_df_out.write_parquet(str(out_all_path), compression="zstd")
        
        logger.info(f"Writing {len(good_df)} good algos to {out_good_path}")
        good_df.write_parquet(str(out_good_path), compression="zstd")
        store.register_artifact("algo_personality_static_good", out_good_path, good_df)
        
        logger.info("Personality feature engineering complete!")
        
        return {
            "algo_personality_static": out_all_path,
            "algo_personality_static_good": out_good_path,
        }
    
    def validate(self, store: "ArtifactStore") -> None:
        """
        Validate personality outputs.
        
        Checks:
        - Required columns exist
        - Unique algo_id
        - Value ranges (drawdown, hit_rate, time_in_dd, etc.)
        - Warning counts for nulls
        """
        from athenai.core.logging import get_logger
        logger = get_logger()
        
        cfg = self.cfg
        out_all_path = store.get_latest("algo_personality_static")
        out_good_path = store.get_latest("algo_personality_static_good")
        
        if out_all_path is None or out_good_path is None:
            raise ValueError("Personality outputs not found in store")
        
        # Required schema
        required_cols = [
            "algo_id", "n_obs", "start_date", "end_date",
            "ret_mean", "ret_std", "vol_ann", "sharpe_ann", "hit_rate", "max_drawdown",
            "ulcer_index", "time_in_drawdown", "avg_drawdown_depth", "max_drawdown_duration",
            "ret_q01", "ret_q05", "ret_q95", "ret_q99", "tail_ratio_95_05", "tail_spread_99_01",
            "skew", "excess_kurtosis", "downside_dev", "sortino_ann",
            "autocorr_ret_1", "autocorr_absret_1", "momentum_log_120", "trend_slope", "trend_r2",
            "sharpe_first_half", "sharpe_second_half", "sharpe_drift", "vol_drift", "return_drift",
        ]
        
        for path, name in [(out_all_path, "ALL"), (out_good_path, "GOOD")]:
            df = pl.read_parquet(str(path))
            logger.info(f"Validating {name}: {len(df)} rows")
            
            # Schema check
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")
            
            # Unique algo_id
            assert_unique_keys(df, ["algo_id"], name)
            
            # Range validations
            issues = []
            
            # max_drawdown ∈ [-1, 0]
            dd_out = df.filter((pl.col("max_drawdown") > 0) | (pl.col("max_drawdown") < -1))
            if len(dd_out) > 0:
                issues.append(f"max_drawdown out of [-1,0]: {len(dd_out)} rows")
            
            # time_in_drawdown ∈ [0, 1]
            tid_out = df.filter((pl.col("time_in_drawdown") < 0) | (pl.col("time_in_drawdown") > 1))
            if len(tid_out) > 0:
                issues.append(f"time_in_drawdown out of [0,1]: {len(tid_out)} rows")
            
            # ulcer_index >= 0
            ulcer_neg = df.filter(pl.col("ulcer_index") < 0)
            if len(ulcer_neg) > 0:
                issues.append(f"ulcer_index negative: {len(ulcer_neg)} rows")
            
            # vol_ann >= 0
            vol_neg = df.filter(pl.col("vol_ann") < 0)
            if len(vol_neg) > 0:
                issues.append(f"vol_ann negative: {len(vol_neg)} rows")
            
            # hit_rate ∈ [0, 1]
            hr_out = df.filter((pl.col("hit_rate") < 0) | (pl.col("hit_rate") > 1))
            if len(hr_out) > 0:
                issues.append(f"hit_rate out of [0,1]: {len(hr_out)} rows")
            
            # Warnings for nulls
            null_cols = ["sharpe_ann", "tail_ratio_95_05", "autocorr_ret_1", "trend_r2", "sortino_ann"]
            for col in null_cols:
                null_pct = df[col].null_count() / len(df) * 100
                if null_pct > 5:
                    store.add_warning(f"{name}: {col} has {null_pct:.1f}% nulls")
            
            # Extreme skew warning
            extreme_skew = df.filter(pl.col("skew").abs() > 20)
            if len(extreme_skew) > 0:
                store.add_warning(f"{name}: {len(extreme_skew)} algos with |skew| > 20")
            
            for issue in issues:
                logger.warning(f"Validation issue ({name}): {issue}")
                store.add_warning(f"{name}: {issue}")
        
        logger.info("Personality validation passed!")
