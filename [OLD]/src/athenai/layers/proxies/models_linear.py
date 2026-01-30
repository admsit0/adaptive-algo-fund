"""
Linear models for proxy predictions using sklearn.

Wraps sklearn models with unified interface for save/load and feature importance.

Implements:
- RidgeRegressor: sklearn Ridge regression
- LogisticClassifier: sklearn LogisticRegression (binary)
- SoftmaxClassifier: sklearn LogisticRegression (multiclass)
- BaselineModel: Persistence, mean, EWMA baselines
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

from athenai.layers.proxies.base import ProxyModel


class RidgeRegressor(ProxyModel):
    """
    Ridge regression using sklearn.
    
    Wraps sklearn.linear_model.Ridge with standard scaling.
    
    Attributes:
        alpha: Regularization strength (λ)
        fit_intercept: Whether to fit intercept term
        normalize: Whether to standardize features before fitting
    """
    
    name = "ridge_regressor"
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize: bool = True,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        
        # sklearn model
        self.model_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        
        # Target scaling
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        
        # Feature names (optional)
        self.feature_names_: list[str] | None = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "RidgeRegressor":
        """
        Fit ridge regression.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            sample_weight: Optional sample weights
            feature_names: Optional feature names for interpretability
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        # Normalize features
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
            
            self.y_mean_ = float(np.mean(y))
            self.y_std_ = float(np.std(y)) + 1e-10
            y = (y - self.y_mean_) / self.y_std_
        
        # Fit sklearn Ridge
        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver="auto",
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Normalize
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        # Predict
        y_pred = self.model_.predict(X)
        
        # Denormalize
        if self.normalize:
            y_pred = y_pred * self.y_std_ + self.y_mean_
        
        return y_pred
    
    @property
    def coef_(self) -> np.ndarray | None:
        """Get coefficients from sklearn model."""
        return self.model_.coef_ if self.model_ is not None else None
    
    @property
    def intercept_(self) -> float:
        """Get intercept from sklearn model."""
        return float(self.model_.intercept_) if self.model_ is not None else 0.0
    
    def save(self, path: Path) -> None:
        """Save model to file."""
        path = Path(path)
        
        state = {
            "model": self.model_,
            "scaler": self.scaler_,
            "y_mean": self.y_mean_,
            "y_std": self.y_std_,
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "feature_names": self.feature_names_,
        }
        
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Path) -> "RidgeRegressor":
        """Load model from file."""
        path = Path(path)
        pkl_path = path.with_suffix(".pkl")
        
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        
        model = cls(
            alpha=state["alpha"],
            fit_intercept=state["fit_intercept"],
            normalize=state["normalize"],
        )
        
        model.model_ = state["model"]
        model.scaler_ = state["scaler"]
        model.y_mean_ = state["y_mean"]
        model.y_std_ = state["y_std"]
        model.feature_names_ = state["feature_names"]
        
        return model
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importances (absolute coefficient values)."""
        if self.model_ is None or self.feature_names_ is None:
            return None
        
        coef = self.model_.coef_.copy()
        
        # Denormalize coefficients for interpretability
        if self.normalize and self.scaler_ is not None:
            coef = coef * self.y_std_ / (self.scaler_.scale_ + 1e-10)
        
        importance = np.abs(coef)
        importance = importance / (importance.sum() + 1e-10)
        
        return dict(zip(self.feature_names_, importance))
    
    def get_params(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
        }


class LogisticClassifier(ProxyModel):
    """
    Binary logistic regression using sklearn.
    
    Wraps sklearn.linear_model.LogisticRegression.
    
    Attributes:
        C: Inverse regularization strength (1/λ)
        max_iter: Maximum iterations
        fit_intercept: Whether to fit intercept term
        normalize: Whether to standardize features before fitting
    """
    
    name = "logistic_classifier"
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        normalize: bool = True,
    ):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        
        # sklearn model
        self.model_: LogisticRegression | None = None
        self.scaler_: StandardScaler | None = None
        
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray = np.array([0, 1])
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "LogisticClassifier":
        """Fit logistic regression."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        # Ensure binary
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Expected binary labels, got {len(self.classes_)} classes")
        
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        # Normalize
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Fit sklearn LogisticRegression
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            solver="lbfgs",
            random_state=42,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict_proba(X)
    
    @property
    def coef_(self) -> np.ndarray | None:
        """Get coefficients from sklearn model."""
        return self.model_.coef_[0] if self.model_ is not None else None
    
    @property
    def intercept_(self) -> float:
        """Get intercept from sklearn model."""
        return float(self.model_.intercept_[0]) if self.model_ is not None else 0.0
    
    def save(self, path: Path) -> None:
        """Save model."""
        path = Path(path)
        
        state = {
            "model": self.model_,
            "scaler": self.scaler_,
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "classes": self.classes_,
            "feature_names": self.feature_names_,
        }
        
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Path) -> "LogisticClassifier":
        """Load model."""
        path = Path(path)
        pkl_path = path.with_suffix(".pkl")
        
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        
        model = cls(
            C=state["C"],
            max_iter=state["max_iter"],
            tol=state["tol"],
            fit_intercept=state["fit_intercept"],
            normalize=state["normalize"],
        )
        
        model.model_ = state["model"]
        model.scaler_ = state["scaler"]
        model.classes_ = state["classes"]
        model.feature_names_ = state["feature_names"]
        
        return model
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importances."""
        if self.model_ is None or self.feature_names_ is None:
            return None
        
        importance = np.abs(self.model_.coef_[0])
        importance = importance / (importance.sum() + 1e-10)
        
        return dict(zip(self.feature_names_, importance))
    
    def get_params(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
        }


class SoftmaxClassifier(ProxyModel):
    """
    Multiclass logistic regression (softmax) using sklearn.
    
    Wraps sklearn.linear_model.LogisticRegression with multi_class='multinomial'.
    """
    
    name = "softmax_classifier"
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        normalize: bool = True,
    ):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        
        # sklearn model
        self.model_: LogisticRegression | None = None
        self.scaler_: StandardScaler | None = None
        
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.feature_names_: list[str] | None = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "SoftmaxClassifier":
        """Fit multiclass logistic regression."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        # Normalize
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Fit sklearn LogisticRegression (multinomial)
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict_proba(X)
    
    @property
    def coef_(self) -> np.ndarray | None:
        """Get coefficients (n_classes, n_features)."""
        return self.model_.coef_ if self.model_ is not None else None
    
    @property
    def intercept_(self) -> np.ndarray | None:
        """Get intercepts (n_classes,)."""
        return self.model_.intercept_ if self.model_ is not None else None
    
    def save(self, path: Path) -> None:
        """Save model."""
        path = Path(path)
        
        state = {
            "model": self.model_,
            "scaler": self.scaler_,
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "normalize": self.normalize,
            "classes": self.classes_,
            "n_classes": self.n_classes_,
            "feature_names": self.feature_names_,
        }
        
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Path) -> "SoftmaxClassifier":
        """Load model."""
        path = Path(path)
        pkl_path = path.with_suffix(".pkl")
        
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        
        model = cls(
            C=state["C"],
            max_iter=state["max_iter"],
            tol=state["tol"],
            normalize=state["normalize"],
        )
        
        model.model_ = state["model"]
        model.scaler_ = state["scaler"]
        model.classes_ = state["classes"]
        model.n_classes_ = state["n_classes"]
        model.feature_names_ = state["feature_names"]
        
        return model
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get average feature importance across classes."""
        if self.model_ is None or self.feature_names_ is None:
            return None
        
        # Average absolute importance across classes
        importance = np.abs(self.model_.coef_).mean(axis=0)
        importance = importance / (importance.sum() + 1e-10)
        
        return dict(zip(self.feature_names_, importance))
    
    def get_params(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "normalize": self.normalize,
        }


class BaselineModel(ProxyModel):
    """
    Baseline models for comparison.
    
    Types:
    - "mean": Predict training mean
    - "persistence": Predict last observed value
    - "ewma": Exponentially weighted moving average
    """
    
    name = "baseline"
    
    def __init__(
        self,
        method: Literal["mean", "persistence", "ewma"] = "mean",
        ewma_span: int = 20,
    ):
        self.method = method
        self.ewma_span = ewma_span
        
        self.mean_: float = 0.0
        self.last_value_: float = 0.0
        self.ewma_alpha_: float = 2 / (ewma_span + 1)
        self.ewma_value_: float = 0.0
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "BaselineModel":
        """Fit baseline (compute statistics)."""
        y = np.asarray(y).ravel()
        
        self.mean_ = float(np.mean(y))
        self.last_value_ = float(y[-1])
        
        # Compute EWMA
        ewma = y[0]
        for val in y[1:]:
            ewma = self.ewma_alpha_ * val + (1 - self.ewma_alpha_) * ewma
        self.ewma_value_ = float(ewma)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate baseline predictions."""
        n_samples = len(X)
        
        if self.method == "mean":
            return np.full(n_samples, self.mean_)
        elif self.method == "persistence":
            return np.full(n_samples, self.last_value_)
        elif self.method == "ewma":
            return np.full(n_samples, self.ewma_value_)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def save(self, path: Path) -> None:
        """Save baseline parameters."""
        path = Path(path)
        
        state = {
            "method": self.method,
            "ewma_span": self.ewma_span,
            "mean": self.mean_,
            "last_value": self.last_value_,
            "ewma_value": self.ewma_value_,
        }
        
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        """Load baseline model."""
        path = Path(path)
        pkl_path = path.with_suffix(".pkl")
        
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        
        model = cls(
            method=state["method"],
            ewma_span=state["ewma_span"],
        )
        model.mean_ = state["mean"]
        model.last_value_ = state["last_value"]
        model.ewma_value_ = state["ewma_value"]
        
        return model
    
    def get_params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "ewma_span": self.ewma_span,
        }
