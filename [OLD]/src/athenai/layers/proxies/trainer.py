"""
Proxy model trainer with walk-forward cross-validation.

Handles:
- Walk-forward CV splitting
- Model fitting and evaluation
- Baseline comparison
- Metrics computation
- Model persistence
- Calibration (Platt scaling, temperature scaling)
- Uncertainty estimation (residual std)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from athenai.core.logging import get_logger
from athenai.layers.proxies.base import (
    ProxyTask,
    ProxyTaskType,
    ProxyModel,
    ProxyTrainResult,
    CVFoldResult,
)
from athenai.layers.proxies.models_linear import (
    RidgeRegressor,
    LogisticClassifier,
    SoftmaxClassifier,
    BaselineModel,
)

if TYPE_CHECKING:
    from athenai.layers.proxies.datasets import ProxyDatasetBuilder


# ==============================================================================
# Calibration functions
# ==============================================================================

def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual frequencies.
    Lower is better; 0 = perfectly calibrated.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (0 to 1)
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_proba[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return float(ece)


def calibrate_platt(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """
    Fit Platt scaling parameters (sigmoid calibration).
    
    P(y=1|x) = 1 / (1 + exp(A*f(x) + B))
    
    Returns:
        (A, B) parameters for sigmoid calibration
    """
    from scipy.optimize import minimize
    
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
    
    # Convert probabilities to log-odds (avoid extreme values)
    eps = 1e-7
    y_proba = np.clip(y_proba, eps, 1 - eps)
    f = np.log(y_proba / (1 - y_proba))
    
    def loss(params):
        A, B = params
        p = 1 / (1 + np.exp(A * f + B))
        # Cross-entropy loss
        return -np.mean(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))
    
    result = minimize(loss, [1.0, 0.0], method='BFGS')
    return float(result.x[0]), float(result.x[1])


def apply_platt_calibration(y_proba: np.ndarray, A: float, B: float) -> np.ndarray:
    """Apply Platt scaling to probabilities."""
    y_proba = np.asarray(y_proba).ravel()
    
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
    
    eps = 1e-7
    y_proba = np.clip(y_proba, eps, 1 - eps)
    f = np.log(y_proba / (1 - y_proba))
    
    return 1 / (1 + np.exp(A * f + B))


def calibrate_temperature(
    y_true: np.ndarray,
    y_logits: np.ndarray,
) -> float:
    """
    Fit temperature scaling parameter.
    
    Temperature scaling divides logits by T before softmax.
    T > 1: softer probabilities, T < 1: sharper probabilities
    
    Returns:
        Optimal temperature T
    """
    from scipy.optimize import minimize_scalar
    
    y_true = np.asarray(y_true).ravel().astype(int)
    y_logits = np.asarray(y_logits)
    
    if y_logits.ndim == 1:
        # Binary case: convert to 2-class logits
        y_logits = np.stack([-y_logits, y_logits], axis=1)
    
    def nll_loss(T):
        scaled_logits = y_logits / T
        # Softmax
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Negative log-likelihood
        return -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-10))
    
    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    return float(result.x)


# ==============================================================================
# Metrics functions
# ==============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression metrics.
    
    Returns:
        rmse: Root mean squared error
        mae: Mean absolute error
        r2: R-squared
        spearman: Spearman correlation
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    # Spearman correlation
    from scipy.stats import spearmanr
    try:
        spearman, _ = spearmanr(y_true, y_pred)
    except:
        spearman = 0.0
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "spearman": float(spearman) if not np.isnan(spearman) else 0.0,
    }


def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute classification metrics.
    
    Returns:
        accuracy: Overall accuracy
        balanced_accuracy: Balanced accuracy
        f1: F1 score (macro)
        auc: ROC AUC (binary) or averaged (multiclass)
        brier: Brier score (if proba available)
        ece: Expected Calibration Error (if proba available)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Balanced accuracy
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(np.mean(y_pred[mask] == c))
    balanced_accuracy = np.mean(recalls) if recalls else 0.0
    
    # F1 (macro)
    f1_scores = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)
    f1_macro = np.mean(f1_scores) if f1_scores else 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_macro": float(f1_macro),
    }
    
    # AUC (binary only for now)
    if y_proba is not None and len(classes) == 2:
        try:
            from scipy.special import expit
            # Compute ROC AUC manually
            pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            
            # Sort by predicted probability
            sorted_idx = np.argsort(pos_proba)[::-1]
            y_sorted = y_true[sorted_idx]
            
            # Compute AUC via trapezoid rule
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            
            if n_pos > 0 and n_neg > 0:
                tpr = np.cumsum(y_sorted == 1) / n_pos
                fpr = np.cumsum(y_sorted == 0) / n_neg
                auc = np.trapz(tpr, fpr)
                metrics["auc"] = float(auc)
            
            # Brier score
            brier = np.mean((pos_proba - y_true) ** 2)
            metrics["brier"] = float(brier)
            
            # Expected Calibration Error (ECE)
            ece = compute_ece(y_true, pos_proba)
            metrics["ece"] = ece
        except:
            pass
    
    return metrics


def compute_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sign accuracy (useful for directional predictions)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# ==============================================================================
# Leakage Test
# ==============================================================================

def run_leakage_test(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_params: dict,
    feature_names: list[str] | None = None,
    n_shuffles: int = 5,
    metric: str = "r2",
    logger = None,
) -> dict[str, float]:
    """
    Detect feature leakage by comparing real vs shuffled target performance.
    
    If model performs well on shuffled targets, features likely contain
    forward-looking information (leakage).
    
    Args:
        X: Features
        y: Real target
        model_class: Model class to use
        model_params: Model parameters
        feature_names: Optional feature names
        n_shuffles: Number of shuffle iterations
        metric: Metric to use ("r2", "accuracy")
        logger: Optional logger
        
    Returns:
        Dict with real_metric, shuffle_mean, shuffle_std, leakage_detected
    """
    from athenai.core.logging import get_logger
    log = logger or get_logger()
    
    # Fit on real target
    model = model_class(**model_params)
    model.fit(X, y, feature_names=feature_names)
    y_pred = model.predict(X)
    
    if metric == "r2":
        real_metric = compute_regression_metrics(y, y_pred)["r2"]
    else:
        real_metric = float(np.mean(y == y_pred))
    
    # Fit on shuffled targets
    shuffle_metrics = []
    for i in range(n_shuffles):
        y_shuffled = np.random.permutation(y)
        shuffle_model = model_class(**model_params)
        shuffle_model.fit(X, y_shuffled, feature_names=feature_names)
        y_shuffle_pred = shuffle_model.predict(X)
        
        if metric == "r2":
            shuffle_score = compute_regression_metrics(y_shuffled, y_shuffle_pred)["r2"]
        else:
            shuffle_score = float(np.mean(y_shuffled == y_shuffle_pred))
        
        shuffle_metrics.append(shuffle_score)
    
    shuffle_mean = float(np.mean(shuffle_metrics))
    shuffle_std = float(np.std(shuffle_metrics))
    
    # Leakage detected if shuffle performance is too good (> expected random + 2 std)
    # For R2: expect ~0 on shuffled data
    # For accuracy: expect ~50% (or 1/n_classes)
    if metric == "r2":
        expected_shuffle = 0.0
    else:
        n_classes = len(np.unique(y))
        expected_shuffle = 1.0 / n_classes
    
    leakage_detected = shuffle_mean > expected_shuffle + 2 * shuffle_std + 0.05
    
    result = {
        "real_metric": real_metric,
        "shuffle_mean": shuffle_mean,
        "shuffle_std": shuffle_std,
        "expected_shuffle": expected_shuffle,
        "leakage_detected": leakage_detected,
    }
    
    log.info(f"Leakage test ({metric}): real={real_metric:.4f}, shuffle={shuffle_mean:.4f}+/-{shuffle_std:.4f}")
    if leakage_detected:
        log.warning("[!] POTENTIAL LEAKAGE DETECTED - shuffled performance is suspiciously high!")
    else:
        log.info("[OK] No leakage detected")
    
    return result


# ==============================================================================
# Drift Monitoring
# ==============================================================================

def compute_drift_metrics(
    cv_folds: list,
    dates: np.ndarray | None = None,
    window_months: int = 6,
    alert_threshold: float = 0.15,
    logger = None,
) -> dict[str, Any]:
    """
    Monitor performance drift over time using CV fold results.
    
    Detects if model performance degrades over time (concept drift).
    
    Args:
        cv_folds: List of CVFoldResult
        dates: Optional dates for temporal analysis
        window_months: Size of rolling window in months
        alert_threshold: Relative change that triggers alert
        logger: Optional logger
        
    Returns:
        Dict with drift analysis results
    """
    from athenai.core.logging import get_logger
    log = logger or get_logger()
    
    if not cv_folds:
        return {"error": "No CV folds for drift analysis"}
    
    # Extract metrics by fold (time-ordered)
    fold_metrics = []
    for fold in cv_folds:
        fold_metrics.append({
            "fold_idx": fold.fold_idx,
            "val_start": fold.val_start,
            "val_end": fold.val_end,
            **fold.metrics,
        })
    
    # Find primary metric
    primary_metric = None
    for key in ["r2", "accuracy", "auc", "sign_accuracy"]:
        if key in fold_metrics[0]:
            primary_metric = key
            break
    
    if primary_metric is None:
        return {"error": "No primary metric found"}
    
    # Compute rolling window statistics
    metric_values = [f[primary_metric] for f in fold_metrics]
    
    # Early folds vs late folds comparison
    n_folds = len(metric_values)
    if n_folds >= 4:
        half = n_folds // 2
        early_mean = np.mean(metric_values[:half])
        late_mean = np.mean(metric_values[half:])
        
        drift_amount = late_mean - early_mean
        relative_drift = drift_amount / (abs(early_mean) + 1e-10)
        
        drift_detected = abs(relative_drift) > alert_threshold
    else:
        early_mean = metric_values[0] if metric_values else 0
        late_mean = metric_values[-1] if metric_values else 0
        drift_amount = late_mean - early_mean
        relative_drift = drift_amount / (abs(early_mean) + 1e-10) if early_mean else 0
        drift_detected = False
    
    result = {
        "primary_metric": primary_metric,
        "metric_by_fold": metric_values,
        "early_mean": float(early_mean),
        "late_mean": float(late_mean),
        "drift_amount": float(drift_amount),
        "relative_drift": float(relative_drift),
        "drift_detected": drift_detected,
    }
    
    log.info(f"Drift analysis ({primary_metric}):")
    log.info(f"  Early folds avg: {early_mean:.4f}")
    log.info(f"  Late folds avg: {late_mean:.4f}")
    log.info(f"  Drift: {drift_amount:+.4f} ({relative_drift:+.1%})")
    
    if drift_detected:
        log.warning(f"[!] DRIFT DETECTED - performance changed by {relative_drift:.1%}")
    else:
        log.info("[OK] No significant drift detected")
    
    return result


class ProxyTrainer:
    """
    Train proxy models with walk-forward cross-validation.
    
    Workflow:
    1. Build dataset with features and targets
    2. Generate walk-forward CV splits
    3. Train model on each fold
    4. Compute metrics and compare to baseline
    5. Train final model on all data
    6. Save model and results
    """
    
    def __init__(
        self,
        dataset_builder: "ProxyDatasetBuilder",
        models_dir: Path,
    ):
        self.dataset_builder = dataset_builder
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
    
    def train_vix_proxy(
        self,
        vix_data: pl.DataFrame,
        alpha: float = 1.0,
        C: float = 1.0,
    ) -> ProxyTrainResult:
        """
        Train VIX proxy model (Proxy A).
        
        Enhanced modes:
        - Regression: predict Δlog(VIX_{t+1}) - continuous
        - Spike detection: predict VIX spike (Δlog > q90) - BEST FOR RL
        - Multi-horizon: train separate models for t+1 and t+5
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Training VIX Proxy (Proxy A)")
        self.logger.info("="*60)
        
        cfg = self.dataset_builder.cfg.proxy_vix
        
        # Build dataset
        dataset, target_col = self.dataset_builder.build_vix_dataset(vix_data)
        
        # Get extra features specific to VIX (set by build_vix_dataset)
        extra_features = getattr(self.dataset_builder, '_vix_extra_features', [])
        
        # Create task
        feature_cols = self.dataset_builder.feature_cols.copy()
        if extra_features:
            feature_cols.extend(extra_features)
        
        # Determine if classification (spike) or regression
        is_spike = target_col == "y_vix_spike" or cfg.target_transform == "spike"
        
        task = ProxyTask(
            task_type=ProxyTaskType.VIX,
            name="vix_synthetic",
            target_col=target_col,
            target_transform="spike" if is_spike else "dlog",
            horizon=cfg.horizon,
            model_type="binary" if is_spike else "regression",
            feature_cols=feature_cols,
        )
        
        # Choose model based on task
        if is_spike:
            self.logger.info("Using spike detection (binary classification)")
            model = LogisticClassifier(C=cfg.logreg_C if hasattr(cfg, 'logreg_C') else C)
            metric_fn = compute_classification_metrics
            is_classification = True
        else:
            self.logger.info("Using regression (Δlog prediction)")
            model = RidgeRegressor(alpha=alpha)
            metric_fn = compute_regression_metrics
            is_classification = False
        
        # Train with CV
        result = self._train_with_cv(
            task=task,
            dataset=dataset,
            target_col=target_col,
            model=model,
            metric_fn=metric_fn,
            is_classification=is_classification,
            extra_features=extra_features,
        )
        
        # === MULTI-HORIZON TRAINING ===
        # Train additional models for other horizons
        multi_horizon_results = {}
        if cfg.use_multi_horizon and not is_spike:
            for h in cfg.horizons:
                if h == cfg.horizon:
                    continue  # Already trained
                alt_target = f"y_vix_dlog_h{h}"
                if alt_target in dataset.columns:
                    self.logger.info(f"\n--- Training horizon h={h} model ---")
                    alt_model = RidgeRegressor(alpha=alpha)
                    alt_result = self._train_with_cv(
                        task=task,
                        dataset=dataset,
                        target_col=alt_target,
                        model=alt_model,
                        metric_fn=compute_regression_metrics,
                        extra_features=extra_features,
                    )
                    multi_horizon_results[f"h{h}"] = alt_result.metrics_mean
        
        # Store multi-horizon results in metadata
        if multi_horizon_results:
            result.metadata = result.metadata or {}
            result.metadata["multi_horizon_metrics"] = multi_horizon_results
        
        return result
    
    def train_rates_proxy(
        self,
        rates_data: pl.DataFrame,
        alpha: float = 1.0,
    ) -> ProxyTrainResult:
        """
        Train rates proxy model (Proxy B).
        
        Regression: predict ΔDGS10 in bps
        Includes leakage test and drift monitoring.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Rates Proxy (Proxy B)")
        self.logger.info("="*60)
        
        dataset, target_col = self.dataset_builder.build_rates_dataset(rates_data)
        
        cfg = self.dataset_builder.cfg.proxy_rates
        
        task = ProxyTask(
            task_type=ProxyTaskType.RATES,
            name="rate_oracle",
            target_col=target_col,
            target_transform="bps",
            horizon=cfg.horizon,
            model_type="regression",
            feature_cols=self.dataset_builder.feature_cols.copy(),
        )
        
        model = RidgeRegressor(alpha=alpha)
        
        # === LEAKAGE TEST ===
        leakage_result = None
        if cfg.run_leakage_test and len(dataset) > 100:
            self.logger.info("\n--- Running Leakage Test ---")
            X_test, y_test, _ = self.dataset_builder.to_numpy(dataset, target_col)
            leakage_result = run_leakage_test(
                X=X_test,
                y=y_test,
                model_class=RidgeRegressor,
                model_params={"alpha": alpha},
                feature_names=self.dataset_builder.feature_cols,
                metric="r2",
                logger=self.logger,
            )
        
        # Train with CV
        result = self._train_with_cv(
            task=task,
            dataset=dataset,
            target_col=target_col,
            model=model,
            metric_fn=compute_regression_metrics,
        )
        
        # === DRIFT MONITORING ===
        drift_result = None
        if cfg.monitor_drift and len(result.cv_folds) >= 3:
            self.logger.info("\n--- Drift Monitoring ---")
            drift_result = compute_drift_metrics(
                cv_folds=result.cv_folds,
                window_months=cfg.drift_window_months,
                alert_threshold=cfg.drift_alert_threshold,
                logger=self.logger,
            )
        
        # Store leakage and drift results in result metadata
        if leakage_result or drift_result:
            result.metadata = {
                "leakage_test": leakage_result,
                "drift_analysis": drift_result,
            }
        
        return result
    
    def train_recession_proxy(
        self,
        usrec_data: pl.DataFrame | None = None,
        sp500_data: pl.DataFrame | None = None,
        C: float = 1.0,
    ) -> ProxyTrainResult:
        """
        Train recession/risk-off proxy (Proxy C).
        
        Binary classification: predict recession/risk-off probability.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Recession Proxy (Proxy C)")
        self.logger.info("="*60)
        
        dataset, target_col = self.dataset_builder.build_recession_dataset(
            usrec_data, sp500_data
        )
        
        task = ProxyTask(
            task_type=ProxyTaskType.RECESSION,
            name="recession_detector",
            target_col=target_col,
            model_type="binary",
            feature_cols=self.dataset_builder.feature_cols.copy(),
        )
        
        model = LogisticClassifier(C=C)
        result = self._train_with_cv(
            task=task,
            dataset=dataset,
            target_col=target_col,
            model=model,
            metric_fn=compute_classification_metrics,
            is_classification=True,
        )
        
        return result
    
    def train_factors_proxy(
        self,
        factors_data: pl.DataFrame,
        C: float = 1.0,
    ) -> ProxyTrainResult:
        """
        Train factor monitor proxy (Proxy D).
        
        Multiclass: predict winning factor next month.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Factors Proxy (Proxy D)")
        self.logger.info("="*60)
        
        dataset, target_col = self.dataset_builder.build_factors_dataset(factors_data)
        
        cfg = self.dataset_builder.cfg.proxy_factors
        
        task = ProxyTask(
            task_type=ProxyTaskType.FACTORS,
            name="factor_monitor",
            target_col=target_col,
            model_type="multiclass",
            feature_cols=self.dataset_builder.feature_cols.copy(),
            class_labels=list(cfg.factors),
        )
        
        model = SoftmaxClassifier(C=C)
        result = self._train_with_cv(
            task=task,
            dataset=dataset,
            target_col=target_col,
            model=model,
            metric_fn=compute_classification_metrics,
            is_classification=True,
        )
        
        return result
    
    def _train_with_cv(
        self,
        task: ProxyTask,
        dataset: pl.DataFrame,
        target_col: str,
        model: ProxyModel,
        metric_fn: callable,
        is_classification: bool = False,
        extra_features: list[str] | None = None,
    ) -> ProxyTrainResult:
        """
        Train model with walk-forward CV.
        
        Args:
            extra_features: Additional feature columns specific to this dataset
                           (e.g., f_vix_log_current for VIX proxy)
        
        Returns complete training result with metrics.
        """
        feature_cols = self.dataset_builder.feature_cols.copy()
        
        # Add extra features if provided
        if extra_features:
            for f in extra_features:
                if f not in feature_cols:
                    feature_cols.append(f)
        
        # Handle empty dataset
        if len(dataset) == 0:
            self.logger.warning(f"Empty dataset for {task.name}, skipping training")
            return ProxyTrainResult(
                task=task,
                model=model,
                cv_folds=[],
                metrics_mean={},
                metrics_std={},
                baseline_metrics={},
                feature_importance={},
                model_path=None,
            )
        
        # Get CV splits
        splits = self.dataset_builder.get_cv_splits(dataset)
        
        cv_folds = []
        all_metrics = []
        
        for fold_idx, (train_df, val_df) in enumerate(splits):
            self.logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
            
            # Convert to numpy (pass extra_features for this specific proxy)
            X_train, y_train, train_dates = self.dataset_builder.to_numpy(
                train_df, target_col, extra_features=extra_features
            )
            X_val, y_val, val_dates = self.dataset_builder.to_numpy(
                val_df, target_col, extra_features=extra_features
            )
            
            self.logger.info(f"  Train: {len(X_train)} samples ({train_dates[0]} to {train_dates[-1]})")
            self.logger.info(f"  Val: {len(X_val)} samples ({val_dates[0]} to {val_dates[-1]})")
            
            # Clone model for this fold
            fold_model = type(model)(**model.get_params())
            
            # Fit
            fold_model.fit(X_train, y_train, feature_names=feature_cols)
            
            # Predict
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val) if is_classification else None
            
            # Compute metrics
            if is_classification:
                metrics = metric_fn(y_val, y_pred, y_proba)
            else:
                metrics = metric_fn(y_val, y_pred)
                # Add sign accuracy for regression
                metrics["sign_accuracy"] = compute_sign_accuracy(y_val, y_pred)
            
            # Compute residual std for uncertainty estimation (regression only)
            residual_std = None
            if not is_classification:
                residuals = y_val - y_pred
                residual_std = float(np.std(residuals))
                metrics["residual_std"] = residual_std
            
            self.logger.info(f"  Metrics: {metrics}")
            
            all_metrics.append(metrics)
            
            cv_folds.append(CVFoldResult(
                fold_idx=fold_idx,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                val_start=val_dates[0],
                val_end=val_dates[-1],
                metrics=metrics,
                val_dates=[d if isinstance(d, date) else d.date() for d in val_dates],
                val_true=y_val.tolist(),
                val_pred=y_pred.tolist(),
                val_proba=[p.tolist() for p in y_proba] if y_proba is not None else None,
                residual_std=residual_std,
            ))
        
        # Aggregate metrics (handle missing metrics in some folds)
        metrics_mean = {}
        metrics_std = {}
        if len(all_metrics) > 0:
            # Collect all possible metric keys
            all_keys = set()
            for m in all_metrics:
                all_keys.update(m.keys())
            
            for key in all_keys:
                values = [m[key] for m in all_metrics if key in m]
                if len(values) > 0:
                    metrics_mean[key] = float(np.mean(values))
                    metrics_std[key] = float(np.std(values))
            
            self.logger.info(f"\nAggregated metrics (mean ± std):")
            for key in metrics_mean:
                self.logger.info(f"  {key}: {metrics_mean[key]:.4f} ± {metrics_std[key]:.4f}")
        else:
            self.logger.warning("No CV folds generated - not enough data")
        
        # Train baseline
        self.logger.info("\nTraining baselines...")
        X_all, y_all, _ = self.dataset_builder.to_numpy(
            dataset, target_col, extra_features=extra_features
        )
        
        baseline_metrics = {}
        for baseline_type in ["mean", "persistence", "ewma"]:
            baseline = BaselineModel(method=baseline_type)
            baseline.fit(X_all, y_all)
            y_baseline = baseline.predict(X_all)
            
            if is_classification:
                bl_metrics = compute_classification_metrics(y_all, (y_baseline > 0.5).astype(int))
            else:
                bl_metrics = compute_regression_metrics(y_all, y_baseline)
            
            for k, v in bl_metrics.items():
                baseline_metrics[f"{baseline_type}_{k}"] = v
        
        # Train final model on all data
        self.logger.info("\nTraining final model on all data...")
        final_model = type(model)(**model.get_params())
        final_model.fit(X_all, y_all, feature_names=feature_cols)
        
        # Save model
        cluster_set_id = self.dataset_builder.cfg.cluster_set_id
        model_name = f"proxy_{task.task_type.value}_{cluster_set_id}__{model.name}"
        model_path = self.models_dir / f"{model_name}.npz"
        final_model.save(model_path)
        self.logger.info(f"Saved model to {model_path}")
        
        # Get feature importance
        feature_importance = final_model.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:10]
            self.logger.info("\nTop 10 features:")
            for feat, imp in top_features:
                self.logger.info(f"  {feat}: {imp:.4f}")
        
        # ==============================================================
        # Calibration and Uncertainty estimation
        # ==============================================================
        calibration_params = None
        calibration_metrics = None
        residual_std_mean = None
        residual_std_per_fold = None
        
        cv_cfg = self.dataset_builder.cfg.cv_config
        enable_calibration = getattr(cv_cfg, 'enable_calibration', False)
        compute_uncertainty = getattr(cv_cfg, 'compute_uncertainty', False)
        
        # Calibration for binary classifiers
        if is_classification and enable_calibration and len(cv_folds) > 0:
            # Collect all OOF predictions and true labels
            all_y_true = []
            all_y_proba = []
            for fold in cv_folds:
                all_y_true.extend(fold.val_true)
                if fold.val_proba is not None:
                    # Get positive class probability
                    for p in fold.val_proba:
                        if isinstance(p, list) and len(p) > 1:
                            all_y_proba.append(p[1])
                        else:
                            all_y_proba.append(p)
            
            if len(all_y_proba) > 0:
                all_y_true = np.array(all_y_true)
                all_y_proba = np.array(all_y_proba)
                
                # Compute ECE before calibration
                ece_before = compute_ece(all_y_true, all_y_proba)
                
                # Fit Platt scaling
                try:
                    A, B = calibrate_platt(all_y_true, all_y_proba)
                    calibrated_proba = apply_platt_calibration(all_y_proba, A, B)
                    ece_after = compute_ece(all_y_true, calibrated_proba)
                    
                    calibration_params = {"A": A, "B": B}
                    calibration_metrics = {
                        "ece_before": ece_before,
                        "ece_after": ece_after,
                        "ece_improvement": ece_before - ece_after,
                    }
                    
                    self.logger.info(f"\nCalibration (Platt scaling):")
                    self.logger.info(f"  ECE before: {ece_before:.4f}")
                    self.logger.info(f"  ECE after: {ece_after:.4f}")
                    self.logger.info(f"  Params: A={A:.4f}, B={B:.4f}")
                except Exception as e:
                    self.logger.warning(f"Calibration failed: {e}")
        
        # Uncertainty estimation for regression
        if not is_classification and compute_uncertainty and len(cv_folds) > 0:
            residual_std_per_fold = [f.residual_std for f in cv_folds if f.residual_std is not None]
            if residual_std_per_fold:
                residual_std_mean = float(np.mean(residual_std_per_fold))
                self.logger.info(f"\nUncertainty estimation:")
                self.logger.info(f"  Mean residual std: {residual_std_mean:.4f}")
                self.logger.info(f"  Per fold: {[f'{s:.4f}' for s in residual_std_per_fold]}")
        
        return ProxyTrainResult(
            task=task,
            model_path=model_path,
            cv_folds=cv_folds,
            metrics_mean=metrics_mean,
            metrics_std=metrics_std,
            baseline_metrics=baseline_metrics,
            n_features=len(feature_cols),
            feature_names=feature_cols,
            feature_importance=feature_importance,
            calibration_params=calibration_params,
            calibration_metrics=calibration_metrics,
            residual_std_mean=residual_std_mean,
            residual_std_per_fold=residual_std_per_fold,
        )
    
    def train_all(
        self,
        vix_data: pl.DataFrame | None = None,
        rates_data: pl.DataFrame | None = None,
        usrec_data: pl.DataFrame | None = None,
        sp500_data: pl.DataFrame | None = None,
        factors_data: pl.DataFrame | None = None,
    ) -> dict[str, ProxyTrainResult]:
        """
        Train all enabled proxy models.
        
        Returns dictionary of results by task type.
        """
        cfg = self.dataset_builder.cfg
        results = {}
        
        if cfg.proxy_vix.enabled and vix_data is not None:
            results["vix"] = self.train_vix_proxy(
                vix_data,
                alpha=cfg.proxy_vix.ridge_alpha,
            )
        
        if cfg.proxy_rates.enabled and rates_data is not None:
            results["rates"] = self.train_rates_proxy(
                rates_data,
                alpha=cfg.proxy_rates.ridge_alpha,
            )
        
        if cfg.proxy_recession.enabled and (usrec_data is not None or sp500_data is not None):
            results["recession"] = self.train_recession_proxy(
                usrec_data,
                sp500_data,
                C=cfg.proxy_recession.logreg_C,
            )
        
        if cfg.proxy_factors.enabled and factors_data is not None:
            results["factors"] = self.train_factors_proxy(
                factors_data,
                C=cfg.proxy_factors.softmax_C,
            )
        
        return results


def generate_training_report(
    results: dict[str, ProxyTrainResult],
    output_path: Path,
) -> str:
    """
    Generate markdown report of training results.
    
    Returns:
        Report content as string
    """
    lines = [
        "# Proxy Layers Training Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
    ]
    
    for task_name, result in results.items():
        lines.append(f"## {result.task.name} ({task_name})")
        lines.append("")
        lines.append(f"**Model**: {result.model_path.name}")
        lines.append(f"**Features**: {result.n_features}")
        lines.append(f"**CV Folds**: {len(result.cv_folds)}")
        lines.append("")
        
        lines.append("### Metrics (mean ± std)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        for metric, mean_val in result.metrics_mean.items():
            std_val = result.metrics_std.get(metric, 0)
            lines.append(f"| {metric} | {mean_val:.4f} ± {std_val:.4f} |")
        
        lines.append("")
        
        # Baseline comparison
        if result.baseline_metrics:
            lines.append("### Baseline Comparison")
            lines.append("")
            lines.append("| Baseline | Metric | Value |")
            lines.append("|----------|--------|-------|")
            
            for key, val in result.baseline_metrics.items():
                parts = key.split("_", 1)
                if len(parts) == 2:
                    lines.append(f"| {parts[0]} | {parts[1]} | {val:.4f} |")
            
            lines.append("")
        
        # Top features
        if result.feature_importance:
            lines.append("### Top Features")
            lines.append("")
            top_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:10]
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for feat, imp in top_features:
                lines.append(f"| {feat} | {imp:.4f} |")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    content = "\n".join(lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return content
