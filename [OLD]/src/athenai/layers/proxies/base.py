"""
Base classes for proxy models.

Defines:
- ProxyTask: Enum for proxy types
- ProxyModel: Abstract base for trainable proxies
- ProxyPrediction: Output structure
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl


class ProxyTaskType(Enum):
    """Types of proxy prediction tasks."""
    
    VIX = "vix"               # Proxy A: VIX synthetic
    RATES = "rates"           # Proxy B: Rate oracle
    RECESSION = "recession"   # Proxy C: Recession/risk-off
    FACTORS = "factors"       # Proxy D: Factor monitor


@dataclass
class ProxyTask:
    """
    Definition of a proxy prediction task.
    
    Specifies:
    - Task type (VIX, rates, recession, factors)
    - Target variable name and transform
    - Feature columns to use
    - Prediction horizon
    - Model type (regression/classification)
    """
    task_type: ProxyTaskType
    name: str
    target_col: str
    target_transform: str | None = None  # "log", "diff", "bps"
    horizon: int = 1
    model_type: Literal["regression", "binary", "multiclass"] = "regression"
    feature_cols: list[str] = field(default_factory=list)
    class_labels: list[str] | None = None  # For multiclass
    
    # Training metadata
    train_start: date | None = None
    train_end: date | None = None
    
    def __post_init__(self):
        if self.model_type == "multiclass" and not self.class_labels:
            raise ValueError("multiclass tasks require class_labels")


@dataclass
class ProxyPrediction:
    """
    Output from a proxy model prediction.
    
    Contains both point predictions and probabilities where applicable.
    """
    task_type: ProxyTaskType
    date: date
    
    # Point prediction (regression or argmax class)
    prediction: float | str
    
    # For classification: probabilities per class
    probabilities: dict[str, float] | None = None
    
    # Confidence/uncertainty estimate
    confidence: float | None = None
    
    # Model info
    model_name: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "date": self.date.isoformat() if isinstance(self.date, date) else self.date,
            "prediction": self.prediction,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }


class ProxyModel(ABC):
    """
    Abstract base class for proxy models.
    
    All proxy models must implement:
    - fit(): Train on (X, y) data
    - predict(): Generate predictions
    - save()/load(): Model persistence
    
    Models should be "sklearn-free" (pure numpy/polars).
    """
    
    name: str = "base_proxy"
    task_type: ProxyTaskType | None = None
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ProxyModel":
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) or (n_samples, n_classes)
            sample_weight: Optional sample weights
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """
        Generate class probabilities (for classifiers).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probabilities (n_samples, n_classes) or None if not applicable
        """
        return None
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Output path (e.g., "model.npz" or "model.pkl")
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ProxyModel":
        """
        Load model from disk.
        
        Args:
            path: Path to saved model
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importances if available.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return None
    
    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params) -> "ProxyModel":
        """Set model parameters."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self


@dataclass
class CVFoldResult:
    """Result from one cross-validation fold."""
    fold_idx: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    
    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Predictions on validation set
    val_dates: list[date] = field(default_factory=list)
    val_true: list[float] = field(default_factory=list)
    val_pred: list[float] = field(default_factory=list)
    val_proba: list[list[float]] | None = None
    
    # Uncertainty estimation (for regression)
    residual_std: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
            "metrics": self.metrics,
            "n_val_samples": len(self.val_true),
            "residual_std": self.residual_std,
        }


@dataclass
class ProxyTrainResult:
    """Complete result from proxy model training."""
    task: ProxyTask
    model_path: Path
    
    # CV results
    cv_folds: list[CVFoldResult] = field(default_factory=list)
    
    # Aggregated metrics
    metrics_mean: dict[str, float] = field(default_factory=dict)
    metrics_std: dict[str, float] = field(default_factory=dict)
    
    # Baseline comparison
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    
    # Training info
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] | None = None
    
    # Calibration info (for classifiers)
    calibration_params: dict[str, float] | None = None  # e.g., {"A": 1.0, "B": 0.0} for Platt
    calibration_metrics: dict[str, float] | None = None  # e.g., {"ece_before": 0.1, "ece_after": 0.05}
    
    # Uncertainty estimation (for regression)
    residual_std_mean: float | None = None  # Average residual std across folds
    residual_std_per_fold: list[float] | None = None
    
    # Metadata for additional analyses (leakage test, drift monitoring, etc.)
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task.task_type.value,
            "task_name": self.task.name,
            "model_path": str(self.model_path),
            "cv_folds": [f.to_dict() for f in self.cv_folds],
            "metrics_mean": self.metrics_mean,
            "metrics_std": self.metrics_std,
            "baseline_metrics": self.baseline_metrics,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "calibration_params": self.calibration_params,
            "calibration_metrics": self.calibration_metrics,
            "residual_std_mean": self.residual_std_mean,
            "metadata": self.metadata,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== {self.task.name} ({self.task.task_type.value}) ===",
            f"Model: {self.model_path.name}",
            f"Features: {self.n_features}",
            f"CV Folds: {len(self.cv_folds)}",
            "",
            "Metrics (mean ± std):",
        ]
        
        for metric, mean_val in self.metrics_mean.items():
            std_val = self.metrics_std.get(metric, 0)
            baseline = self.baseline_metrics.get(metric, None)
            
            line = f"  {metric}: {mean_val:.4f} ± {std_val:.4f}"
            if baseline is not None:
                improvement = ((mean_val - baseline) / abs(baseline) * 100 
                              if baseline != 0 else 0)
                line += f" (baseline: {baseline:.4f}, Δ={improvement:+.1f}%)"
            
            lines.append(line)
        
        return "\n".join(lines)
