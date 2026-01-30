"""
Smoke tests for personality feature engineering.

Tests:
1. Synthetic panel with known behavior (trend vs mean-revert)
2. Validates feature values are in expected ranges
3. Validates schema and constraints
"""

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def generate_synthetic_panel(
    n_algos: int = 3,
    n_days: int = 200,
    seed: int = 42,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """
    Generate synthetic panel with known behaviors.
    
    Returns:
        panel: DataFrame with algo_id, date, close, ret_1d, logret_1d
        behaviors: Dict mapping algo_id to behavior type
    """
    np.random.seed(seed)
    
    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    rows = []
    behaviors = {}
    
    # Algo 1: Strong upward trend (positive autocorr, high trend_r2)
    algo_id = "TREND_UP"
    behaviors[algo_id] = "trend"
    close = 100.0
    for i, d in enumerate(dates):
        if i > 0:
            ret = 0.001 + 0.005 * np.random.randn()  # drift + small noise
            close = close * (1 + ret)
            ret_1d = ret
            logret_1d = np.log(1 + ret)
        else:
            ret_1d = None
            logret_1d = None
        rows.append({
            "algo_id": algo_id,
            "date": d,
            "close": close,
            "ret_1d": ret_1d,
            "logret_1d": logret_1d,
        })
    
    # Algo 2: Mean-reverting (negative autocorr)
    algo_id = "MEAN_REVERT"
    behaviors[algo_id] = "mean_revert"
    close = 100.0
    last_ret = 0.0
    for i, d in enumerate(dates):
        if i > 0:
            # Mean reversion: opposite of last return plus noise
            ret = -0.5 * last_ret + 0.01 * np.random.randn()
            close = close * (1 + ret)
            ret_1d = ret
            logret_1d = np.log(1 + ret)
            last_ret = ret
        else:
            ret_1d = None
            logret_1d = None
        rows.append({
            "algo_id": algo_id,
            "date": d,
            "close": close,
            "ret_1d": ret_1d,
            "logret_1d": logret_1d,
        })
    
    # Algo 3: Vol clustering (high autocorr_absret)
    algo_id = "VOL_CLUSTER"
    behaviors[algo_id] = "vol_cluster"
    close = 100.0
    vol = 0.01
    for i, d in enumerate(dates):
        if i > 0:
            # GARCH-like: vol depends on recent vol
            vol = 0.01 + 0.7 * (abs(rows[-1]["ret_1d"]) if rows[-1]["ret_1d"] else 0.01)
            vol = min(vol, 0.05)  # cap vol
            ret = 0.0001 + vol * np.random.randn()
            close = close * (1 + ret)
            ret_1d = ret
            logret_1d = np.log(1 + ret)
        else:
            ret_1d = None
            logret_1d = None
        rows.append({
            "algo_id": algo_id,
            "date": d,
            "close": close,
            "ret_1d": ret_1d,
            "logret_1d": logret_1d,
        })
    
    panel = pl.DataFrame(rows)
    return panel, behaviors


def generate_synthetic_meta(panel: pl.DataFrame) -> pl.DataFrame:
    """Generate meta DataFrame from panel."""
    meta = (
        panel
        .group_by("algo_id")
        .agg([
            pl.min("date").alias("start_date"),
            pl.max("date").alias("end_date"),
            pl.len().alias("n_obs"),
            pl.col("close").n_unique().alias("n_unique_close"),
            pl.col("close").std().alias("close_std"),
            pl.col("ret_1d").mean().alias("ret_mean"),
            pl.col("ret_1d").std().alias("ret_std"),
        ])
        .with_columns([
            ((pl.col("end_date") - pl.col("start_date")).dt.total_days() + 1).alias("n_days"),
        ])
        .with_columns([
            (pl.col("n_obs") / pl.col("n_days")).alias("coverage_ratio"),
            (pl.col("close_std") <= 1e-8).alias("is_constant"),
        ])
    )
    return meta


class TestPersonalityFeatures:
    """Test personality feature computation."""
    
    @pytest.fixture
    def temp_artifacts(self, tmp_path):
        """Create temporary artifact structure."""
        # Generate synthetic data
        panel, behaviors = generate_synthetic_panel()
        meta = generate_synthetic_meta(panel)
        
        # Create preprocess-like structure
        run_id = "test_preprocess_run"
        run_dir = tmp_path / "cache" / run_id
        run_dir.mkdir(parents=True)
        
        # Save parquets
        panel.write_parquet(run_dir / "algos_panel.parquet")
        meta.write_parquet(run_dir / "algos_meta.parquet")
        meta.write_parquet(run_dir / "algos_meta_good.parquet")  # All are "good" for test
        
        return {
            "tmp_path": tmp_path,
            "run_id": run_id,
            "panel": panel,
            "meta": meta,
            "behaviors": behaviors,
        }
    
    def test_personality_step_runs(self, temp_artifacts):
        """Test that personality step runs without error."""
        from athenai.features.personality import PersonalityConfig, BuildAlgoPersonalityStaticStep
        from athenai.core.artifacts import ArtifactStore
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        # Create output store
        out_run_id = "test_personality_run"
        store = ArtifactStore(
            root=tmp_path / "cache",
            config=cfg,
            run_id=out_run_id,
        )
        store.ensure_dirs()
        
        # Run step
        step = BuildAlgoPersonalityStaticStep(cfg=cfg)
        outputs = step.run(store, cfg, overwrite=True)
        
        assert "algo_personality_static" in outputs
        assert "algo_personality_static_good" in outputs
        assert outputs["algo_personality_static"].exists()
        assert outputs["algo_personality_static_good"].exists()
    
    def test_personality_schema(self, temp_artifacts):
        """Test that output has correct schema."""
        from athenai.features.personality import PersonalityConfig, BuildAlgoPersonalityStaticStep
        from athenai.core.artifacts import ArtifactStore
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        out_run_id = "test_personality_schema"
        store = ArtifactStore(root=tmp_path / "cache", config=cfg, run_id=out_run_id)
        store.ensure_dirs()
        
        step = BuildAlgoPersonalityStaticStep(cfg=cfg)
        outputs = step.run(store, cfg, overwrite=True)
        
        df = pl.read_parquet(outputs["algo_personality_static"])
        
        required_cols = [
            "algo_id", "n_obs", "start_date", "end_date",
            "ret_mean", "ret_std", "vol_ann", "sharpe_ann", "hit_rate", "max_drawdown",
            "ulcer_index", "time_in_drawdown", "avg_drawdown_depth", "max_drawdown_duration",
            "ret_q01", "ret_q05", "ret_q95", "ret_q99", "tail_ratio_95_05", "tail_spread_99_01",
            "skew", "excess_kurtosis", "downside_dev", "sortino_ann",
            "autocorr_ret_1", "autocorr_absret_1", "momentum_log_120", "trend_slope", "trend_r2",
            "sharpe_first_half", "sharpe_second_half", "sharpe_drift", "vol_drift", "return_drift",
        ]
        
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_drawdown_ranges(self, temp_artifacts):
        """Test that drawdown metrics are in valid ranges."""
        from athenai.features.personality import PersonalityConfig, BuildAlgoPersonalityStaticStep
        from athenai.core.artifacts import ArtifactStore
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        out_run_id = "test_personality_ranges"
        store = ArtifactStore(root=tmp_path / "cache", config=cfg, run_id=out_run_id)
        store.ensure_dirs()
        
        step = BuildAlgoPersonalityStaticStep(cfg=cfg)
        outputs = step.run(store, cfg, overwrite=True)
        
        df = pl.read_parquet(outputs["algo_personality_static"])
        
        # max_drawdown ∈ [-1, 0]
        assert df["max_drawdown"].max() <= 0, "max_drawdown should be <= 0"
        assert df["max_drawdown"].min() >= -1, "max_drawdown should be >= -1"
        
        # ulcer_index >= 0
        assert df["ulcer_index"].min() >= 0, "ulcer_index should be >= 0"
        
        # time_in_drawdown ∈ [0, 1]
        assert df["time_in_drawdown"].min() >= 0, "time_in_drawdown should be >= 0"
        assert df["time_in_drawdown"].max() <= 1, "time_in_drawdown should be <= 1"
        
        # hit_rate ∈ [0, 1]
        assert df["hit_rate"].min() >= 0, "hit_rate should be >= 0"
        assert df["hit_rate"].max() <= 1, "hit_rate should be <= 1"
        
        # vol_ann >= 0
        assert df["vol_ann"].min() >= 0, "vol_ann should be >= 0"
    
    def test_trend_algo_has_high_r2(self, temp_artifacts):
        """Test that trending algo has higher trend_r2 than mean-reverting."""
        from athenai.features.personality import PersonalityConfig, BuildAlgoPersonalityStaticStep
        from athenai.core.artifacts import ArtifactStore
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        behaviors = temp_artifacts["behaviors"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        out_run_id = "test_personality_trend"
        store = ArtifactStore(root=tmp_path / "cache", config=cfg, run_id=out_run_id)
        store.ensure_dirs()
        
        step = BuildAlgoPersonalityStaticStep(cfg=cfg)
        outputs = step.run(store, cfg, overwrite=True)
        
        df = pl.read_parquet(outputs["algo_personality_static"])
        
        trend_r2 = df.filter(pl.col("algo_id") == "TREND_UP")["trend_r2"][0]
        revert_r2 = df.filter(pl.col("algo_id") == "MEAN_REVERT")["trend_r2"][0]
        
        assert trend_r2 is not None, "trend_r2 should not be null for TREND_UP"
        assert trend_r2 > revert_r2, f"TREND_UP should have higher trend_r2 ({trend_r2}) than MEAN_REVERT ({revert_r2})"
    
    def test_unique_algo_ids(self, temp_artifacts):
        """Test that algo_id is unique in output."""
        from athenai.features.personality import PersonalityConfig, BuildAlgoPersonalityStaticStep
        from athenai.core.artifacts import ArtifactStore
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        out_run_id = "test_personality_unique"
        store = ArtifactStore(root=tmp_path / "cache", config=cfg, run_id=out_run_id)
        store.ensure_dirs()
        
        step = BuildAlgoPersonalityStaticStep(cfg=cfg)
        outputs = step.run(store, cfg, overwrite=True)
        
        df = pl.read_parquet(outputs["algo_personality_static"])
        
        n_rows = len(df)
        n_unique = df["algo_id"].n_unique()
        
        assert n_rows == n_unique, f"algo_id should be unique: {n_rows} rows, {n_unique} unique"
    
    def test_validation_passes(self, temp_artifacts):
        """Test that validation passes on valid data."""
        from athenai.features.personality import PersonalityConfig, BuildAlgoPersonalityStaticStep
        from athenai.core.artifacts import ArtifactStore
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        out_run_id = "test_personality_validate"
        store = ArtifactStore(root=tmp_path / "cache", config=cfg, run_id=out_run_id)
        store.ensure_dirs()
        
        step = BuildAlgoPersonalityStaticStep(cfg=cfg)
        step.run(store, cfg, overwrite=True)
        
        # Should not raise
        step.validate(store)


class TestPersonalityPipeline:
    """Test full personality pipeline."""
    
    @pytest.fixture
    def temp_artifacts(self, tmp_path):
        """Create temporary artifact structure."""
        panel, behaviors = generate_synthetic_panel()
        meta = generate_synthetic_meta(panel)
        
        run_id = "test_preprocess_run"
        run_dir = tmp_path / "cache" / run_id
        run_dir.mkdir(parents=True)
        
        panel.write_parquet(run_dir / "algos_panel.parquet")
        meta.write_parquet(run_dir / "algos_meta.parquet")
        meta.write_parquet(run_dir / "algos_meta_good.parquet")
        
        return {
            "tmp_path": tmp_path,
            "run_id": run_id,
        }
    
    def test_pipeline_runs(self, temp_artifacts):
        """Test that full pipeline runs successfully."""
        from athenai.features.personality import PersonalityConfig
        from athenai.pipelines.personality import PersonalityPipeline
        
        tmp_path = temp_artifacts["tmp_path"]
        run_id = temp_artifacts["run_id"]
        
        cfg = PersonalityConfig(
            preprocess_run_id=run_id,
            min_obs_personality=50,
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        )
        
        pipeline = PersonalityPipeline(cfg)
        results = pipeline.run(overwrite=True, validate=True)
        
        assert "algo_personality_static" in results
        assert "algo_personality_static_good" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
