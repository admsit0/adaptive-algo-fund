"""
Smoke tests for preprocessing pipeline.

Quick tests to verify the pipeline runs without errors.
Uses a small subset of data for speed.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

import polars as pl


# Skip if data not available
DATA_DIR = Path("datos_competicion")
SKIP_NO_DATA = not DATA_DIR.exists()


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="athenai_test_"))
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config(temp_cache_dir):
    """Create a config for testing."""
    from athenai.core.config import PreprocessConfig
    
    return PreprocessConfig(
        root_dir=DATA_DIR,
        cache_dir=temp_cache_dir,
        reports_dir=temp_cache_dir / "reports",
        figures_dir=temp_cache_dir / "figures",
        min_obs=10,  # Lower threshold for testing
        min_coverage=0.5,
        feature_windows=(5, 10),  # Smaller windows for speed
    )


@pytest.mark.skipif(SKIP_NO_DATA, reason="Data directory not found")
class TestPreprocessPipeline:
    """Test the preprocessing pipeline."""
    
    def test_config_validation(self, sample_config):
        """Test that config validation works."""
        sample_config.validate()
    
    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        from athenai.core.config import PreprocessConfig
        
        yaml_path = Path("configs/preprocess.yaml")
        if yaml_path.exists():
            cfg = PreprocessConfig.from_yaml(yaml_path)
            assert cfg.root_dir is not None
    
    def test_artifact_store_creation(self, sample_config, temp_cache_dir):
        """Test artifact store initialization."""
        from athenai.core.artifacts import ArtifactStore
        
        store = ArtifactStore(root=temp_cache_dir, config=sample_config)
        store.ensure_dirs()
        
        assert store.run_dir.exists()
        assert store.run_id is not None
    
    def test_build_panel_step(self, sample_config, temp_cache_dir):
        """Test building algos_panel."""
        from athenai.core.artifacts import ArtifactStore
        from athenai.data.algos_panel import BuildAlgosPanelStep
        
        store = ArtifactStore(root=temp_cache_dir, config=sample_config)
        store.ensure_dirs()
        
        step = BuildAlgosPanelStep()
        outputs = step.run(store, sample_config, overwrite=True)
        
        assert "algos_panel" in outputs
        assert outputs["algos_panel"].exists()
        
        # Validate schema
        df = pl.read_parquet(str(outputs["algos_panel"]))
        assert "algo_id" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns
        assert "ret_1d" in df.columns
    
    def test_full_pipeline(self, sample_config, temp_cache_dir):
        """Test running the full pipeline."""
        from athenai.pipelines.preprocess import PreprocessPipeline
        
        pipeline = PreprocessPipeline(sample_config)
        outputs = pipeline.run(overwrite=True, validate=True)
        
        # Check core outputs exist
        assert "algos_panel" in outputs
        assert "algos_meta" in outputs
        assert "algos_meta_good" in outputs
        
        # Check files exist
        for name, path in outputs.items():
            assert path.exists(), f"Output {name} not found at {path}"
    
    def test_dataset_inspector(self, sample_config, temp_cache_dir):
        """Test dataset inspection."""
        from athenai.pipelines.preprocess import PreprocessPipeline
        from athenai.reports.inspector import DatasetInspector
        
        # First run pipeline
        pipeline = PreprocessPipeline(sample_config)
        outputs = pipeline.run(overwrite=True, validate=False)
        
        # Then inspect
        inspector = DatasetInspector()
        result = inspector.inspect(
            outputs["algos_panel"],
            key_cols=["algo_id", "date"]
        )
        
        assert result.row_count > 0
        assert result.schema is not None
        assert "algo_id" in result.schema


class TestValidation:
    """Test validation utilities."""
    
    def test_schema_validation(self):
        """Test schema validation."""
        from athenai.core.validation import assert_schema, SchemaSpec
        
        df = pl.DataFrame({
            "algo_id": ["a", "b"],
            "value": [1.0, 2.0],
        })
        
        spec = SchemaSpec(required={"algo_id": pl.String, "value": pl.Float64})
        assert_schema(df, spec, "test")  # Should not raise
    
    def test_unique_keys(self):
        """Test unique key validation."""
        from athenai.core.validation import assert_unique_keys, ValidationError
        
        df = pl.DataFrame({
            "id": ["a", "b", "c"],
            "value": [1, 2, 3],
        })
        
        assert_unique_keys(df, ["id"], "test")  # Should not raise
        
        df_dup = pl.DataFrame({
            "id": ["a", "a", "b"],
            "value": [1, 2, 3],
        })
        
        with pytest.raises(ValidationError):
            assert_unique_keys(df_dup, ["id"], "test")


class TestConfig:
    """Test configuration classes."""
    
    def test_config_hash(self):
        """Test config hash is deterministic."""
        from athenai.core.config import PreprocessConfig
        
        cfg1 = PreprocessConfig(root_dir=Path("test"), min_obs=60)
        cfg2 = PreprocessConfig(root_dir=Path("test"), min_obs=60)
        cfg3 = PreprocessConfig(root_dir=Path("test"), min_obs=100)
        
        assert cfg1.config_hash() == cfg2.config_hash()
        assert cfg1.config_hash() != cfg3.config_hash()
    
    def test_config_to_dict(self):
        """Test config serialization."""
        from athenai.core.config import PreprocessConfig
        
        cfg = PreprocessConfig(root_dir=Path("test"))
        d = cfg.to_dict()
        
        assert d["root_dir"] == "test"
        assert "min_obs" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
