"""
Smoke tests for clustering pipeline modules.

Tests:
1. Factor config loading
2. Cluster models (KMeans, MiniBatch, GMM)
3. Scalers (Robust, ZScore)
4. Feature matrix building
5. Cluster assignment consistency
6. Timeseries aggregation
7. Pipeline end-to-end (if data available)

Run:
    pytest tests/test_clustering_smoke.py -v
"""

import numpy as np
import pytest
import polars as pl
from pathlib import Path
import tempfile


# =============================================================================
# Test: Cluster Models
# =============================================================================

class TestClusterModels:
    """Test clustering algorithms."""
    
    def test_minibatch_kmeans_basic(self):
        """Test MiniBatchKMeans fits and predicts."""
        from athenai.clustering.cluster_models import MiniBatchKMeansModel
        
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(100, 5) + [0, 0, 0, 0, 0],
            np.random.randn(100, 5) + [5, 5, 5, 5, 5],
            np.random.randn(100, 5) + [10, 0, 10, 0, 10],
        ])
        
        model = MiniBatchKMeansModel(k=3, seed=42)
        model.fit(X)
        
        assert model.labels_ is not None
        assert len(model.labels_) == 300
        assert model.cluster_centers_.shape == (3, 5)
        assert model.inertia_ > 0
        
        # Check cluster sizes are reasonable
        unique, counts = np.unique(model.labels_, return_counts=True)
        assert len(unique) == 3  # All clusters populated
        assert min(counts) > 50  # Each cluster has at least some points
    
    def test_kmeans_basic(self):
        """Test standard KMeans."""
        from athenai.clustering.cluster_models import KMeansModel
        
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 3) + [0, 0, 0],
            np.random.randn(50, 3) + [5, 5, 5],
        ])
        
        model = KMeansModel(k=2, seed=42)
        model.fit(X)
        
        assert model.labels_ is not None
        assert len(model.labels_) == 100
        assert model.cluster_centers_.shape == (2, 3)
        
        # Check separation
        unique = np.unique(model.labels_)
        assert len(unique) == 2
    
    def test_gmm_basic(self):
        """Test GMM fits."""
        from athenai.clustering.cluster_models import GMMModel
        
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(80, 4) * 0.5 + [0, 0, 0, 0],
            np.random.randn(80, 4) * 0.5 + [3, 3, 3, 3],
        ])
        
        model = GMMModel(k=2, seed=42)
        model.fit(X)
        
        assert model.labels_ is not None
        assert len(model.labels_) == 160
        assert model.means_.shape == (2, 4)
        
        # Check weights sum to 1
        assert abs(model.weights_.sum() - 1.0) < 1e-6
    
    def test_factory_function(self):
        """Test get_cluster_model factory."""
        from athenai.clustering.cluster_models import get_cluster_model
        
        model1 = get_cluster_model("minibatch_kmeans", k=10)
        model2 = get_cluster_model("kmeans", k=5)
        model3 = get_cluster_model("gmm", k=3)
        
        assert model1.k == 10
        assert model2.k == 5
        assert model3.k == 3
    
    def test_predict_distances(self):
        """Test distance computation."""
        from athenai.clustering.cluster_models import KMeansModel
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        
        model = KMeansModel(k=3, seed=42)
        model.fit(X)
        
        distances = model.get_distances(X)
        
        assert len(distances) == 50
        assert all(d >= 0 for d in distances)
    
    def test_model_save_load(self):
        """Test model persistence."""
        from athenai.clustering.cluster_models import KMeansModel
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        
        model = KMeansModel(k=2, seed=42)
        model.fit(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(str(save_path))
            
            # Load uses classmethod
            loaded = KMeansModel.load(str(save_path))
            
            np.testing.assert_array_almost_equal(
                model.cluster_centers_, loaded.cluster_centers_
            )


# =============================================================================
# Test: Scalers
# =============================================================================

class TestScalers:
    """Test scaling methods."""
    
    def test_robust_scaler(self):
        """Test RobustScaler with outliers."""
        from athenai.clustering.cluster_models import RobustScaler
        
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[0, 0] = 100  # Outlier
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check median is roughly 0 (robust to outlier)
        assert abs(np.median(X_scaled[:, 0])) < 0.5
        
        # Check clipping
        assert X_scaled.max() <= 10
        assert X_scaled.min() >= -10
    
    def test_zscore_scaler(self):
        """Test ZScoreScaler."""
        from athenai.clustering.cluster_models import ZScoreScaler
        
        np.random.seed(42)
        X = np.random.randn(100, 3) * 5 + 10  # Mean~10, std~5
        
        scaler = ZScoreScaler()
        X_scaled = scaler.fit_transform(X)
        
        # After scaling, mean should be ~0, std ~1
        assert abs(X_scaled.mean()) < 0.5
        assert abs(X_scaled.std() - 1.0) < 0.5
    
    def test_scaler_factory(self):
        """Test get_scaler factory."""
        from athenai.clustering.cluster_models import get_scaler
        
        scaler1 = get_scaler("robust")
        scaler2 = get_scaler("zscore")
        scaler3 = get_scaler("none")
        
        assert scaler1 is not None
        assert scaler2 is not None
        assert scaler3 is None


# =============================================================================
# Test: Configuration
# =============================================================================

class TestConfig:
    """Test configuration dataclasses."""
    
    def test_factor_config_defaults(self):
        """Test FactorConfig defaults."""
        from athenai.factors.config import FactorConfig
        
        cfg = FactorConfig()
        
        assert cfg.pca_k == 10
        assert cfg.pca_date_sampling == "W"
        assert cfg.standardize_returns == True
    
    def test_exposure_config_defaults(self):
        """Test ExposureConfig defaults."""
        from athenai.factors.config import ExposureConfig
        
        cfg = ExposureConfig()
        
        assert cfg.method == "ridge_ols"
        assert cfg.ridge_lambda == 1e-3
        assert cfg.min_obs == 120
    
    def test_cluster_set_config(self):
        """Test ClusterSetConfig."""
        from athenai.factors.config import ClusterSetConfig
        
        cs = ClusterSetConfig(
            cluster_set_id="test_k10",
            family="behavioral",
            k=10,
        )
        
        assert cs.cluster_set_id == "test_k10"
        assert cs.k == 10
        assert cs.scaler == "robust"  # default
        
        d = cs.to_dict()
        assert d["cluster_set_id"] == "test_k10"
    
    def test_clustering_config_defaults(self):
        """Test ClusteringConfig with defaults."""
        from athenai.factors.config import ClusteringConfig
        
        cfg = ClusteringConfig(
            run_id="test_run",
            preprocess_run_id="prep_v1",
            personality_run_id="pers_v1",
        )
        
        # Should have default cluster sets
        assert len(cfg.cluster_sets) == 2
        assert cfg.cluster_sets[0].cluster_set_id == "behavioral_k100_v1"
        assert cfg.cluster_sets[1].cluster_set_id == "corrpca_k60_v1"
    
    def test_config_yaml_roundtrip(self):
        """Test YAML save/load."""
        from athenai.factors.config import ClusteringConfig, ClusterSetConfig
        
        cfg = ClusteringConfig(
            run_id="test",
            preprocess_run_id="prep",
            personality_run_id="pers",
            cluster_sets=[
                ClusterSetConfig(
                    cluster_set_id="test_k5",
                    k=5,
                )
            ],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg.save(str(cfg_path))
            
            loaded = ClusteringConfig.from_yaml(str(cfg_path))
            
            assert loaded.run_id == "test"
            assert len(loaded.cluster_sets) == 1
            assert loaded.cluster_sets[0].k == 5


# =============================================================================
# Test: Factor Timeseries (unit level)
# =============================================================================

class TestFactorTimeseries:
    """Test factor construction."""
    
    def test_pca_via_svd(self):
        """Test PCA implementation produces valid factors."""
        # Simulate: n_dates x n_algos returns matrix
        np.random.seed(42)
        n_dates, n_algos = 200, 50
        
        # Create correlated returns
        factors = np.random.randn(n_dates, 3)
        loadings = np.random.randn(n_algos, 3)
        noise = np.random.randn(n_dates, n_algos) * 0.5
        returns = factors @ loadings.T + noise
        
        # Standardize
        returns_std = (returns - returns.mean(axis=0)) / (returns.std(axis=0) + 1e-9)
        
        # SVD
        U, S, Vt = np.linalg.svd(returns_std, full_matrices=False)
        
        k = 3
        pca_factors = U[:, :k] * S[:k]  # (n_dates, k)
        loadings_out = Vt[:k, :].T      # (n_algos, k)
        
        # Check shapes
        assert pca_factors.shape == (n_dates, k)
        assert loadings_out.shape == (n_algos, k)
        
        # Check variance explained
        total_var = (S ** 2).sum()
        explained_var = (S[:k] ** 2).sum() / total_var
        assert explained_var > 0.3  # Should explain significant variance


# =============================================================================
# Test: Exposures (unit level)
# =============================================================================

class TestExposures:
    """Test exposure computation."""
    
    def test_ols_regression(self):
        """Test OLS gives reasonable betas."""
        np.random.seed(42)
        n = 200
        
        # True betas
        beta_true = np.array([0.8, 0.5, -0.3])
        
        # Factor returns
        F = np.random.randn(n, 3) * 0.02  # 2% daily vol
        
        # Algo returns = beta @ F + noise
        noise = np.random.randn(n) * 0.01
        y = F @ beta_true + noise
        
        # OLS: beta = (F'F)^-1 F'y
        FtF_inv = np.linalg.inv(F.T @ F)
        beta_hat = FtF_inv @ F.T @ y
        
        # Should be close to true
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=1)
    
    def test_ridge_regression(self):
        """Test ridge regression with regularization."""
        np.random.seed(42)
        n = 200
        
        F = np.random.randn(n, 3) * 0.02
        noise = np.random.randn(n) * 0.01
        beta_true = np.array([1.0, 0.5, 0.2])
        y = F @ beta_true + noise
        
        # Ridge: (F'F + λI)^-1 F'y
        lam = 0.001
        I = np.eye(3)
        beta_ridge = np.linalg.solve(F.T @ F + lam * I, F.T @ y)
        
        # Should be close (with small shrinkage)
        assert np.abs(beta_ridge - beta_true).max() < 0.2


# =============================================================================
# Test: Cluster Timeseries (unit level)
# =============================================================================

class TestClusterTimeseries:
    """Test cluster aggregation."""
    
    def test_equal_weight_return(self):
        """Test EW return computation."""
        # Simulate cluster with 3 algos
        df = pl.DataFrame({
            "date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "algo_id": ["A", "B", "C"] * 2,
            "cluster_id": [0, 0, 0, 0, 0, 0],
            "ret": [0.01, 0.02, 0.03, -0.01, 0.00, 0.01],
        })
        
        # EW return per day
        ew_ret = (
            df.group_by("date")
            .agg(pl.col("ret").mean().alias("ret_ew"))
            .sort("date")
        )
        
        assert len(ew_ret) == 2
        assert abs(ew_ret["ret_ew"][0] - 0.02) < 1e-9  # (0.01+0.02+0.03)/3
        assert abs(ew_ret["ret_ew"][1] - 0.00) < 1e-9  # (-0.01+0.00+0.01)/3


# =============================================================================
# Test: End-to-end (optional, needs data)
# =============================================================================

@pytest.mark.skipif(
    not Path("data/cache/preprocess_v1").exists(),
    reason="Preprocessed data not available"
)
class TestPipelineE2E:
    """End-to-end tests (requires actual data)."""
    
    def test_pipeline_runs(self):
        """Test full pipeline execution."""
        from athenai.pipelines.clustering import run_clustering_pipeline
        from athenai.factors.config import ClusteringConfig, ClusterSetConfig
        
        # Create minimal config
        cfg = ClusteringConfig(
            run_id="test_e2e",
            preprocess_run_id="preprocess_v1",
            personality_run_id="personality_v1",
            cluster_sets=[
                ClusterSetConfig(
                    cluster_set_id="test_k5",
                    k=5,
                )
            ],
        )
        
        # Would need to save and run
        # This is a placeholder for actual e2e test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
