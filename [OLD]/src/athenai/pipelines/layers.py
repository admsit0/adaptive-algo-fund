"""
Layers Pipeline orchestrator.

Runs all steps for Proxy Layers (Paso 3):
1. Fetch external data (TRAIN only)
2. Build universe features from clusters
3. Train proxy models with walk-forward CV
4. Generate predictions
5. Build enriched RL state

Usage:
    python -m athenai.scripts.train_layers --config configs/layers.yaml --train
    python -m athenai.scripts.train_layers --config configs/layers.yaml --predict
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from athenai.core.logging import get_logger
from athenai.layers.proxies.datasets import ProxyDatasetBuilder, build_universe_features
from athenai.layers.proxies.trainer import ProxyTrainer, generate_training_report
from athenai.layers.proxies.predictor import ProxyPredictor

if TYPE_CHECKING:
    from athenai.core.artifacts import ArtifactStore
    from athenai.layers.config import LayersConfig


class LayersPipeline:
    """
    Orchestrator for the proxy layers pipeline.
    
    Steps:
    1. FetchExternalDataStep: Download FRED/FF data (TRAIN only)
    2. BuildUniverseFeaturesStep: Create internal features
    3. TrainProxyModelsStep: Walk-forward CV training
    4. GeneratePredictionsStep: Apply models
    5. BuildEnrichedStateStep: Combine features + predictions
    """
    
    def __init__(self, cfg: "LayersConfig", store: "ArtifactStore"):
        self.cfg = cfg
        self.store = store
        self.logger = get_logger()
    
    def run_train(
        self,
        skip_fetch: bool = False,
        skip_features: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """
        Run training pipeline.
        
        Args:
            skip_fetch: Skip external data fetching (use cached)
            skip_features: Skip universe features (use cached)
            overwrite: Overwrite existing outputs
            
        Returns:
            Dictionary of output paths
        """
        outputs = {}
        
        # ============================================================
        # Step 1: Fetch external data
        # ============================================================
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 1: Fetch External Data (TRAIN only)")
        self.logger.info("="*60)
        
        if skip_fetch:
            self.logger.info("Skipping (--skip-fetch)")
            external_data = self._load_cached_external_data()
        else:
            external_data = self._fetch_external_data()
            outputs.update(self._save_external_data(external_data))
        
        # ============================================================
        # Step 2: Build universe features
        # ============================================================
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 2: Build Universe Features")
        self.logger.info("="*60)
        
        universe_features_path = self.store.artifact_path(
            self.cfg.universe_features_name.format(cluster_set_id=self.cfg.cluster_set_id)
        )
        
        if skip_features and universe_features_path.exists():
            self.logger.info("Skipping (--skip-features)")
            universe_features = pl.read_parquet(str(universe_features_path))
        else:
            universe_features = self._build_universe_features()
            universe_features.write_parquet(str(universe_features_path), compression="zstd")
            outputs["universe_features"] = universe_features_path
            self.logger.info(f"Saved universe features to {universe_features_path}")
        
        # ============================================================
        # Step 3: Train proxy models
        # ============================================================
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 3: Train Proxy Models")
        self.logger.info("="*60)
        
        train_results = self._train_proxy_models(universe_features, external_data)
        
        # Save training report
        report_path = self.cfg.reports_dir / f"proxy_training_{self.cfg.cluster_set_id}.md"
        generate_training_report(train_results, report_path)
        outputs["training_report"] = report_path
        self.logger.info(f"Saved training report to {report_path}")
        
        # ============================================================
        # Step 4: Generate predictions on training data
        # ============================================================
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 4: Generate Predictions")
        self.logger.info("="*60)
        
        # Pass external data for proxies that need it (e.g., VIX for f_vix_log_current)
        # Map vixcls -> vix for consistency with predictor
        pred_external_data = {}
        if "vixcls" in external_data:
            pred_external_data["vix"] = external_data["vixcls"]
        
        predictions_df = self._generate_predictions(universe_features, external_data=pred_external_data)
        
        preds_path = self.store.artifact_path(
            self.cfg.proxy_preds_name.format(cluster_set_id=self.cfg.cluster_set_id)
        )
        predictions_df.write_parquet(str(preds_path), compression="zstd")
        outputs["proxy_predictions"] = preds_path
        self.logger.info(f"Saved predictions to {preds_path}")
        
        # ============================================================
        # Step 5: Build enriched state
        # ============================================================
        self.logger.info("\n" + "="*60)
        self.logger.info("STEP 5: Build Enriched RL State")
        self.logger.info("="*60)
        
        enriched_path = self._build_enriched_state(predictions_df)
        outputs["enriched_features"] = enriched_path
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Training pipeline complete!")
        self.logger.info("="*60)
        
        return outputs
    
    def run_predict(
        self,
        universe_features: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """
        Run inference-only pipeline.
        
        No external API calls - uses only trained models and internal features.
        
        Args:
            universe_features: Pre-computed features (or will load from cache)
            
        Returns:
            Enriched features DataFrame
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Running Inference Pipeline (no external data)")
        self.logger.info("="*60)
        
        # Load universe features if not provided
        if universe_features is None:
            universe_features = self._build_universe_features()
        
        # Generate predictions
        predictions_df = self._generate_predictions(universe_features)
        
        # Build enriched state
        enriched_path = self._build_enriched_state(predictions_df)
        enriched = pl.read_parquet(str(enriched_path))
        
        self.logger.info("Inference complete!")
        
        return enriched
    
    def _fetch_external_data(self) -> dict[str, pl.DataFrame]:
        """Fetch all external data for training."""
        from athenai.external.fred import FREDFetcher
        from athenai.external.famafrench import FamaFrenchFetcher
        
        external_data = {}
        
        # FRED data
        fred = FREDFetcher(
            api_key=self.cfg.fred_api_key,
            cache_dir=self.cfg.external_cache_dir,
        )
        
        fred_series = fred.fetch_all(
            start=self.cfg.train_start,
            end=self.cfg.train_end,
        )
        
        # Convert to wide format for each series
        for series_id, df in fred_series.items():
            external_data[series_id.lower()] = df.select([
                "date",
                pl.col("value").alias(series_id.lower())
            ])
        
        # Fama-French data
        ff = FamaFrenchFetcher(cache_dir=self.cfg.external_cache_dir)
        
        ff_data = ff.fetch_factors(
            frequency="monthly",
            start=self.cfg.train_start,
            end=self.cfg.train_end,
            include_momentum=True,
        )
        external_data["factors"] = ff_data
        
        self.logger.info(f"Fetched external data: {list(external_data.keys())}")
        
        return external_data
    
    def _load_cached_external_data(self) -> dict[str, pl.DataFrame]:
        """Load external data from cache."""
        external_data = {}
        
        cache_dir = self.cfg.external_cache_dir
        
        # Try to load cached FRED files and convert to wide format
        for name in ["vixcls", "dgs10", "usrec", "sp500"]:
            pattern = f"fred_{name}_*.parquet"
            files = list(cache_dir.glob(pattern))
            if files:
                df = pl.read_parquet(str(files[0]))
                # Convert from long format (date, value, series_id) to wide (date, {name})
                if "value" in df.columns:
                    df = df.select([
                        "date",
                        pl.col("value").alias(name)
                    ])
                external_data[name] = df
                self.logger.info(f"Loaded cached {name} from {files[0].name}")
        
        # Load factors
        factor_files = list(cache_dir.glob("ff_*_factors*.parquet"))
        if factor_files:
            external_data["factors"] = pl.read_parquet(str(factor_files[0]))
            self.logger.info(f"Loaded cached factors from {factor_files[0].name}")
        
        return external_data
    
    def _save_external_data(self, data: dict[str, pl.DataFrame]) -> dict[str, Path]:
        """Save external data to standard locations."""
        outputs = {}
        
        for name, df in data.items():
            path = self.store.artifact_path(f"external_{name}.parquet")
            df.write_parquet(str(path), compression="zstd")
            outputs[f"external_{name}"] = path
            self.logger.info(f"Saved {name} to {path}")
        
        return outputs
    
    def _build_universe_features(self) -> pl.DataFrame:
        """Build universe features from cluster data."""
        # Load cluster timeseries
        cluster_ts_path = self.cfg.cache_dir / self.cfg.clustering_run_id / f"cluster_timeseries_{self.cfg.cluster_set_id}.parquet"
        
        if not cluster_ts_path.exists():
            raise FileNotFoundError(f"Cluster timeseries not found: {cluster_ts_path}")
        
        cluster_ts = pl.read_parquet(str(cluster_ts_path))
        self.logger.info(f"Loaded cluster timeseries: {len(cluster_ts)} rows")
        
        # Load cluster features
        cluster_features_path = self.cfg.cache_dir / self.cfg.clustering_run_id / f"cluster_features_daily_{self.cfg.cluster_set_id}.parquet"
        
        cluster_features = None
        if cluster_features_path.exists():
            cluster_features = pl.read_parquet(str(cluster_features_path))
            self.logger.info(f"Loaded cluster features: {len(cluster_features)} rows")
        
        # Load factor timeseries (optional)
        factor_ts_path = self.cfg.cache_dir / self.cfg.clustering_run_id / "factor_timeseries.parquet"
        
        factor_ts = None
        if factor_ts_path.exists():
            factor_ts = pl.read_parquet(str(factor_ts_path))
            self.logger.info(f"Loaded factor timeseries: {len(factor_ts)} rows")
        
        # Build universe features
        universe_features = build_universe_features(
            cluster_ts=cluster_ts,
            cluster_features=cluster_features if cluster_features is not None else cluster_ts,
            factor_ts=factor_ts,
            cfg=self.cfg.universe_features,
        )
        
        return universe_features
    
    def _train_proxy_models(
        self,
        universe_features: pl.DataFrame,
        external_data: dict[str, pl.DataFrame],
    ) -> dict[str, any]:
        """Train all proxy models."""
        # Create dataset builder
        dataset_builder = ProxyDatasetBuilder(
            universe_features=universe_features,
            cfg=self.cfg,
        )
        
        # Create trainer
        trainer = ProxyTrainer(
            dataset_builder=dataset_builder,
            models_dir=self.cfg.models_dir,
        )
        
        # Prepare external data
        vix_data = external_data.get("vixcls")
        rates_data = external_data.get("dgs10")
        usrec_data = external_data.get("usrec")
        sp500_data = external_data.get("sp500")
        factors_data = external_data.get("factors")
        
        # Train all models
        results = trainer.train_all(
            vix_data=vix_data,
            rates_data=rates_data,
            usrec_data=usrec_data,
            sp500_data=sp500_data,
            factors_data=factors_data,
        )
        
        return results
    
    def _generate_predictions(
        self,
        universe_features: pl.DataFrame,
        external_data: dict[str, pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Generate proxy predictions."""
        predictor = ProxyPredictor.from_config(self.cfg)
        predictions = predictor.predict(universe_features, external_data=external_data)
        
        # Validate
        warnings = predictor.validate_predictions(predictions)
        for w in warnings:
            self.logger.warning(f"Prediction warning: {w}")
        
        return predictions.to_dataframe()
    
    def _build_enriched_state(
        self,
        predictions_df: pl.DataFrame,
    ) -> Path:
        """Build enriched cluster features."""
        # Load cluster features
        cluster_features_path = self.cfg.cache_dir / self.cfg.clustering_run_id / f"cluster_features_daily_{self.cfg.cluster_set_id}.parquet"
        
        if not cluster_features_path.exists():
            raise FileNotFoundError(f"Cluster features not found: {cluster_features_path}")
        
        cluster_features = pl.read_parquet(str(cluster_features_path))
        
        # Create predictor and build enriched state
        predictor = ProxyPredictor.from_config(self.cfg)
        
        # Convert predictions_df back to ProxyPredictions
        from athenai.layers.proxies.predictor import ProxyPredictions
        
        dates = predictions_df["date"].to_list()
        
        preds = ProxyPredictions(dates=dates)
        
        if "pred_vix_log_tplus1" in predictions_df.columns:
            preds.pred_vix_log_tplus1 = predictions_df["pred_vix_log_tplus1"].to_numpy()
            preds.pred_vix_level_tplus1 = predictions_df["pred_vix_level_tplus1"].to_numpy()
        
        if "prob_vix_high" in predictions_df.columns:
            preds.prob_vix_high = predictions_df["prob_vix_high"].to_numpy()
        
        if "pred_rate_bps_tplus5" in predictions_df.columns:
            preds.pred_rate_bps_tplus5 = predictions_df["pred_rate_bps_tplus5"].to_numpy()
        
        if "prob_rate_up" in predictions_df.columns:
            preds.prob_rate_up = predictions_df["prob_rate_up"].to_numpy()
        
        if "prob_recession" in predictions_df.columns:
            preds.prob_recession = predictions_df["prob_recession"].to_numpy()
        
        if "prob_risk_off" in predictions_df.columns:
            preds.prob_risk_off = predictions_df["prob_risk_off"].to_numpy()
        
        if "p_factor_smb" in predictions_df.columns:
            preds.p_factor_smb = predictions_df["p_factor_smb"].to_numpy()
            preds.p_factor_hml = predictions_df["p_factor_hml"].to_numpy()
            preds.p_factor_mom = predictions_df["p_factor_mom"].to_numpy()
        
        if "winning_factor_hat" in predictions_df.columns:
            preds.winning_factor_hat = predictions_df["winning_factor_hat"].to_list()
        
        # Build enriched features
        enriched = predictor.build_enriched_state(cluster_features, preds)
        
        # Save
        enriched_path = self.store.artifact_path(
            self.cfg.enriched_features_name.format(cluster_set_id=self.cfg.cluster_set_id)
        )
        enriched.write_parquet(str(enriched_path), compression="zstd")
        
        self.logger.info(f"Saved enriched features to {enriched_path}")
        self.logger.info(f"  Shape: {enriched.shape}")
        self.logger.info(f"  Columns: {len(enriched.columns)}")
        
        # Log column groups
        f_cols = [c for c in enriched.columns if c.startswith("f_")]
        pred_cols = [c for c in enriched.columns if c.startswith("pred_") or c.startswith("prob_") or c.startswith("p_factor")]
        
        self.logger.info(f"  Feature columns (f_*): {len(f_cols)}")
        self.logger.info(f"  Prediction columns: {len(pred_cols)}")
        
        return enriched_path


def run_layers_pipeline(
    cfg: "LayersConfig",
    mode: str = "train",
    skip_fetch: bool = False,
    skip_features: bool = False,
    overwrite: bool = False,
) -> dict[str, Path] | pl.DataFrame:
    """
    Convenience function to run layers pipeline.
    
    Args:
        cfg: Layers configuration
        mode: "train" or "predict"
        skip_fetch: Skip external data fetching
        skip_features: Skip universe features
        overwrite: Overwrite existing outputs
        
    Returns:
        Output paths (train) or enriched DataFrame (predict)
    """
    from athenai.core.artifacts import ArtifactStore
    
    # Create artifact store
    store = ArtifactStore(
        base_dir=cfg.cache_dir / cfg.run_id,
        run_id=cfg.run_id,
    )
    
    # Create and run pipeline
    pipeline = LayersPipeline(cfg=cfg, store=store)
    
    if mode == "train":
        return pipeline.run_train(
            skip_fetch=skip_fetch,
            skip_features=skip_features,
            overwrite=overwrite,
        )
    else:
        return pipeline.run_predict()
