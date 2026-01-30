"""
Proxy model predictor for inference.

Loads trained models and generates predictions on new data.
Produces enriched features for RL state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from athenai.core.logging import get_logger
from athenai.layers.proxies.base import ProxyTaskType
from athenai.layers.proxies.models_linear import (
    RidgeRegressor,
    LogisticClassifier,
    SoftmaxClassifier,
)

if TYPE_CHECKING:
    from athenai.layers.config import LayersConfig


@dataclass
class ProxyPredictions:
    """Container for all proxy predictions."""
    
    dates: list[date]
    
    # VIX predictions
    pred_vix_log_tplus1: np.ndarray | None = None
    pred_vix_level_tplus1: np.ndarray | None = None
    prob_vix_high: np.ndarray | None = None
    
    # Rates predictions
    pred_rate_bps_tplus5: np.ndarray | None = None
    prob_rate_up: np.ndarray | None = None
    
    # Recession predictions
    prob_recession: np.ndarray | None = None
    prob_risk_off: np.ndarray | None = None
    
    # Factor predictions
    p_factor_smb: np.ndarray | None = None
    p_factor_hml: np.ndarray | None = None
    p_factor_mom: np.ndarray | None = None
    winning_factor_hat: list[str] | None = None
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert predictions to DataFrame."""
        data = {"date": self.dates}
        
        if self.pred_vix_log_tplus1 is not None:
            data["pred_vix_log_tplus1"] = self.pred_vix_log_tplus1
            data["pred_vix_level_tplus1"] = self.pred_vix_level_tplus1
        
        if self.prob_vix_high is not None:
            data["prob_vix_high"] = self.prob_vix_high
        
        if self.pred_rate_bps_tplus5 is not None:
            data["pred_rate_bps_tplus5"] = self.pred_rate_bps_tplus5
        
        if self.prob_rate_up is not None:
            data["prob_rate_up"] = self.prob_rate_up
        
        if self.prob_recession is not None:
            data["prob_recession"] = self.prob_recession
        
        if self.prob_risk_off is not None:
            data["prob_risk_off"] = self.prob_risk_off
        
        if self.p_factor_smb is not None:
            data["p_factor_smb"] = self.p_factor_smb
            data["p_factor_hml"] = self.p_factor_hml
            data["p_factor_mom"] = self.p_factor_mom
        
        if self.winning_factor_hat is not None:
            data["winning_factor_hat"] = self.winning_factor_hat
        
        return pl.DataFrame(data)


class ProxyPredictor:
    """
    Load trained proxy models and generate predictions.
    
    Usage:
        predictor = ProxyPredictor.from_config(cfg, models_dir)
        predictions = predictor.predict(universe_features)
        enriched = predictor.build_enriched_state(cluster_features, predictions)
    """
    
    def __init__(
        self,
        models_dir: Path,
        cluster_set_id: str,
        cfg: "LayersConfig",
    ):
        self.models_dir = Path(models_dir)
        self.cluster_set_id = cluster_set_id
        self.cfg = cfg
        self.logger = get_logger()
        
        # Load models
        self.models = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all trained proxy models."""
        
        # VIX model (sklearn saves as .pkl)
        if self.cfg.proxy_vix.enabled:
            vix_path = self.models_dir / f"proxy_vix_{self.cluster_set_id}__ridge_regressor.pkl"
            if vix_path.exists():
                self.models["vix"] = RidgeRegressor.load(vix_path)
                self.logger.info(f"Loaded VIX model from {vix_path.name}")
            else:
                self.logger.warning(f"VIX model not found: {vix_path}")
        
        # Rates model
        if self.cfg.proxy_rates.enabled:
            rates_path = self.models_dir / f"proxy_rates_{self.cluster_set_id}__ridge_regressor.pkl"
            if rates_path.exists():
                self.models["rates"] = RidgeRegressor.load(rates_path)
                self.logger.info(f"Loaded rates model from {rates_path.name}")
            else:
                self.logger.warning(f"Rates model not found: {rates_path}")
        
        # Recession model
        if self.cfg.proxy_recession.enabled:
            rec_path = self.models_dir / f"proxy_recession_{self.cluster_set_id}__logistic_classifier.pkl"
            if rec_path.exists():
                self.models["recession"] = LogisticClassifier.load(rec_path)
                self.logger.info(f"Loaded recession model from {rec_path.name}")
            else:
                self.logger.warning(f"Recession model not found: {rec_path}")
        
        # Factors model
        if self.cfg.proxy_factors.enabled:
            factors_path = self.models_dir / f"proxy_factors_{self.cluster_set_id}__softmax_classifier.pkl"
            if factors_path.exists():
                self.models["factors"] = SoftmaxClassifier.load(factors_path)
                self.logger.info(f"Loaded factors model from {factors_path.name}")
            else:
                self.logger.warning(f"Factors model not found: {factors_path}")
    
    @classmethod
    def from_config(cls, cfg: "LayersConfig") -> "ProxyPredictor":
        """Create predictor from config."""
        return cls(
            models_dir=cfg.models_dir,
            cluster_set_id=cfg.cluster_set_id,
            cfg=cfg,
        )
    
    def predict(
        self,
        universe_features: pl.DataFrame,
        external_data: dict[str, pl.DataFrame] | None = None,
    ) -> ProxyPredictions:
        """
        Generate proxy predictions from universe features.
        
        Args:
            universe_features: DataFrame with date and feature columns
            external_data: Optional dict with 'vix', 'rates', etc. DataFrames
                          needed for proxies that use external data as features
            
        Returns:
            ProxyPredictions with all proxy outputs
        """
        self.logger.info("Generating proxy predictions...")
        
        # Extract feature columns (exclude date)
        feature_cols = [c for c in universe_features.columns if c not in ["date", "cluster_set_id"]]
        
        X = universe_features.select(feature_cols).to_numpy().astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        dates = universe_features["date"].to_list()
        n_samples = len(dates)
        
        predictions = ProxyPredictions(dates=dates)
        
        # VIX predictions
        if "vix" in self.models:
            self.logger.info("  Predicting VIX...")
            vix_model = self.models["vix"]
            
            # Check if model needs VIX as feature (for "both" mode)
            # Get expected number of features from scaler
            n_model_features = vix_model.scaler_.n_features_in_ if vix_model.scaler_ is not None else X.shape[1]
            
            if n_model_features > X.shape[1]:
                # Model was trained with f_vix_log_current - need to add it
                if external_data and "vix" in external_data:
                    vix_df = external_data["vix"]
                    # Join VIX with universe features (left join to keep all dates)
                    vix_log = (
                        vix_df
                        .sort("date")
                        .with_columns([pl.col("vixcls").log().alias("f_vix_log_current")])
                        .select(["date", "f_vix_log_current"])
                    )
                    features_with_vix = (
                        universe_features
                        .join(vix_log, on="date", how="left")
                    )
                    # Fill missing VIX with median (or forward fill)
                    features_with_vix = features_with_vix.with_columns([
                        pl.col("f_vix_log_current").fill_null(
                            pl.col("f_vix_log_current").median()
                        )
                    ])
                    X_vix = features_with_vix.select(feature_cols + ["f_vix_log_current"]).to_numpy().astype(np.float64)
                    X_vix = np.nan_to_num(X_vix, nan=0.0)
                    pred_log = vix_model.predict(X_vix)
                else:
                    # No VIX data available - pad with zeros (will be less accurate)
                    self.logger.warning("VIX model needs f_vix_log_current but no VIX data provided")
                    X_padded = np.column_stack([X, np.zeros(n_samples)])
                    pred_log = vix_model.predict(X_padded)
            else:
                pred_log = vix_model.predict(X)
            
            predictions.pred_vix_log_tplus1 = pred_log
            predictions.pred_vix_level_tplus1 = np.exp(pred_log)
            
            # Clip to reasonable range
            predictions.pred_vix_level_tplus1 = np.clip(
                predictions.pred_vix_level_tplus1, 8, 100
            )
            
            # High VIX probability (simple threshold)
            threshold = self.cfg.proxy_vix.high_threshold
            # Approximate probability using sigmoid around threshold
            z = (predictions.pred_vix_level_tplus1 - threshold) / 5
            predictions.prob_vix_high = 1 / (1 + np.exp(-z))
        
        # Rates predictions
        if "rates" in self.models:
            self.logger.info("  Predicting rates...")
            rates_model = self.models["rates"]
            pred_bps = rates_model.predict(X)
            
            predictions.pred_rate_bps_tplus5 = pred_bps
            
            # Clip to reasonable range (-200, 200 bps)
            predictions.pred_rate_bps_tplus5 = np.clip(
                predictions.pred_rate_bps_tplus5, -200, 200
            )
            
            # Rate up probability
            z = pred_bps / 10  # Scale for sigmoid
            predictions.prob_rate_up = 1 / (1 + np.exp(-z))
        
        # Recession predictions
        if "recession" in self.models:
            self.logger.info("  Predicting recession/risk-off...")
            rec_model = self.models["recession"]
            proba = rec_model.predict_proba(X)
            
            predictions.prob_recession = proba[:, 1]
            predictions.prob_risk_off = proba[:, 1]  # Same for now
        
        # Factor predictions
        if "factors" in self.models:
            self.logger.info("  Predicting factors...")
            factors_model = self.models["factors"]
            proba = factors_model.predict_proba(X)
            
            factor_labels = list(self.cfg.proxy_factors.factors)
            
            # Check if binary mode (2 classes: negative=0, positive=1)
            if proba.shape[1] == 2:
                # Binary mode: predict P(MOM > 0)
                predictions.p_factor_mom = proba[:, 1]  # P(positive)
                predictions.p_factor_smb = np.zeros(n_samples)  # Not predicted in binary mode
                predictions.p_factor_hml = np.zeros(n_samples)  # Not predicted in binary mode
                # Winning factor is MOM if positive, else "NONE"
                predictions.winning_factor_hat = [
                    "MOM" if p > 0.5 else "NONE" for p in proba[:, 1]
                ]
            elif len(factor_labels) >= 3:
                # Multiclass mode
                predictions.p_factor_smb = proba[:, 0]
                predictions.p_factor_hml = proba[:, 1]
                predictions.p_factor_mom = proba[:, 2]
                # Winning factor
                winning_idx = np.argmax(proba, axis=1)
                predictions.winning_factor_hat = [factor_labels[i] for i in winning_idx]
        
        self.logger.info(f"  Generated predictions for {n_samples} dates")
        
        return predictions
    
    def build_enriched_state(
        self,
        cluster_features: pl.DataFrame,
        predictions: ProxyPredictions,
    ) -> pl.DataFrame:
        """
        Combine cluster features with proxy predictions.
        
        Creates the enriched RL state that includes:
        - Original cluster features (f_*)
        - Proxy predictions (pred_*, prob_*)
        
        Args:
            cluster_features: cluster_features_daily DataFrame
            predictions: Proxy predictions
            
        Returns:
            Enriched DataFrame with all features
        """
        self.logger.info("Building enriched cluster features...")
        
        # Convert predictions to DataFrame
        pred_df = predictions.to_dataframe()
        
        # Join with cluster features
        enriched = cluster_features.join(
            pred_df,
            on="date",
            how="left"
        )
        
        # Fill nulls in prediction columns
        pred_cols = [c for c in pred_df.columns if c != "date"]
        
        for col in pred_cols:
            if col in enriched.columns:
                # Forward fill for continuity
                enriched = enriched.with_columns([
                    pl.col(col).forward_fill().alias(col)
                ])
                # Then fill remaining with neutral value
                if "prob" in col:
                    fill_val = 0.5
                elif "factor" in col and "winning" not in col:
                    fill_val = 1.0 / 3  # Uniform probability
                elif col == "winning_factor_hat":
                    fill_val = "smb"  # Default
                else:
                    fill_val = 0.0
                
                if col == "winning_factor_hat":
                    enriched = enriched.with_columns([
                        pl.col(col).fill_null(fill_val)
                    ])
                else:
                    enriched = enriched.with_columns([
                        pl.col(col).fill_null(fill_val).fill_nan(fill_val)
                    ])
        
        self.logger.info(
            f"Enriched features: {len(enriched)} rows, "
            f"{len(enriched.columns)} columns"
        )
        
        return enriched
    
    def validate_predictions(
        self,
        predictions: ProxyPredictions,
    ) -> list[str]:
        """
        Validate prediction quality.
        
        Returns list of warnings if issues found.
        """
        warnings = []
        
        # Check VIX predictions
        if predictions.pred_vix_level_tplus1 is not None:
            vix_levels = predictions.pred_vix_level_tplus1
            
            # Check for extreme values
            if np.any(vix_levels < 5):
                warnings.append(f"VIX predictions below 5: {np.sum(vix_levels < 5)} samples")
            if np.any(vix_levels > 90):
                warnings.append(f"VIX predictions above 90: {np.sum(vix_levels > 90)} samples")
            
            # Check for NaNs
            if np.any(np.isnan(vix_levels)):
                warnings.append(f"VIX predictions contain NaN: {np.sum(np.isnan(vix_levels))} samples")
        
        # Check rate predictions
        if predictions.pred_rate_bps_tplus5 is not None:
            rate_bps = predictions.pred_rate_bps_tplus5
            
            if np.any(np.abs(rate_bps) > 150):
                warnings.append(
                    f"Rate predictions exceed ±150 bps: {np.sum(np.abs(rate_bps) > 150)} samples"
                )
        
        # Check probabilities
        for prob_arr, name in [
            (predictions.prob_vix_high, "prob_vix_high"),
            (predictions.prob_rate_up, "prob_rate_up"),
            (predictions.prob_recession, "prob_recession"),
            (predictions.prob_risk_off, "prob_risk_off"),
        ]:
            if prob_arr is not None:
                if np.any((prob_arr < 0) | (prob_arr > 1)):
                    warnings.append(f"{name} outside [0,1]: check model")
        
        # Check factor probabilities sum to 1
        if predictions.p_factor_smb is not None:
            factor_sum = (
                predictions.p_factor_smb + 
                predictions.p_factor_hml + 
                predictions.p_factor_mom
            )
            if not np.allclose(factor_sum, 1.0, atol=0.01):
                warnings.append("Factor probabilities don't sum to 1")
        
        return warnings


def generate_proxy_predictions(
    universe_features: pl.DataFrame,
    cfg: "LayersConfig",
) -> pl.DataFrame:
    """
    Convenience function to generate proxy predictions.
    
    Args:
        universe_features: DataFrame with date and feature columns
        cfg: Layers configuration
        
    Returns:
        DataFrame with proxy predictions
    """
    predictor = ProxyPredictor.from_config(cfg)
    predictions = predictor.predict(universe_features)
    
    # Validate
    warnings = predictor.validate_predictions(predictions)
    if warnings:
        logger = get_logger()
        for w in warnings:
            logger.warning(f"Prediction validation: {w}")
    
    return predictions.to_dataframe()
