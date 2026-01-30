"""
Tree-based models for proxy predictions (optional).

Wraps XGBoost/LightGBM with unified interface.
These are optional - the pipeline works with ridge/logreg only.

Usage:
    # Only if xgboost is installed
    from athenai.layers.proxies.models_tree import XGBRegressor, XGBClassifier
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from athenai.layers.proxies.base import ProxyModel
from athenai.core.logging import get_logger


class XGBRegressor(ProxyModel):
    """
    XGBoost regressor wrapper (optional dependency).
    
    Requires: pip install xgboost
    """
    
    name = "xgb_regressor"
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model_ = None
        self.feature_names_: list[str] | None = None
        self.logger = get_logger()
    
    def _check_xgboost(self):
        """Check if XGBoost is available."""
        try:
            import xgboost
            return True
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "XGBRegressor":
        """Fit XGBoost regressor."""
        self._check_xgboost()
        import xgboost as xgb
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )
        
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X)
    
    def save(self, path: Path) -> None:
        """Save model."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        path = Path(path)
        # XGBoost has native save
        self.model_.save_model(str(path.with_suffix(".json")))
        
        # Save metadata
        np.savez_compressed(
            path,
            feature_names=np.array(self.feature_names_ or [], dtype=str),
            params=np.array([
                self.n_estimators,
                self.max_depth,
                self.learning_rate,
                self.reg_alpha,
                self.reg_lambda,
                self.subsample,
                self.colsample_bytree,
                self.random_state,
            ]),
        )
    
    @classmethod
    def load(cls, path: Path) -> "XGBRegressor":
        """Load model."""
        import xgboost as xgb
        
        path = Path(path)
        
        # Load metadata
        data = np.load(path, allow_pickle=True)
        params = data["params"]
        
        model = cls(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            learning_rate=float(params[2]),
            reg_alpha=float(params[3]),
            reg_lambda=float(params[4]),
            subsample=float(params[5]),
            colsample_bytree=float(params[6]),
            random_state=int(params[7]),
        )
        
        # Load XGBoost model
        model.model_ = xgb.XGBRegressor()
        model.model_.load_model(str(path.with_suffix(".json")))
        
        if len(data["feature_names"]) > 0:
            model.feature_names_ = list(data["feature_names"])
        
        return model
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importances from XGBoost."""
        if self.model_ is None or self.feature_names_ is None:
            return None
        
        importance = self.model_.feature_importances_
        importance = importance / importance.sum()
        
        return dict(zip(self.feature_names_, importance))
    
    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }


class XGBClassifier(ProxyModel):
    """
    XGBoost classifier wrapper for binary/multiclass.
    
    Requires: pip install xgboost
    """
    
    name = "xgb_classifier"
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model_ = None
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray | None = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "XGBClassifier":
        """Fit XGBoost classifier."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        
        self.classes_ = np.unique(y)
        
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict_proba(X)
    
    def save(self, path: Path) -> None:
        """Save model."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        path = Path(path)
        self.model_.save_model(str(path.with_suffix(".json")))
        
        np.savez_compressed(
            path,
            feature_names=np.array(self.feature_names_ or [], dtype=str),
            classes=self.classes_,
            params=np.array([
                self.n_estimators,
                self.max_depth,
                self.learning_rate,
                self.reg_alpha,
                self.reg_lambda,
                self.subsample,
                self.colsample_bytree,
                self.random_state,
            ]),
        )
    
    @classmethod
    def load(cls, path: Path) -> "XGBClassifier":
        """Load model."""
        import xgboost as xgb
        
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        params = data["params"]
        
        model = cls(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            learning_rate=float(params[2]),
            reg_alpha=float(params[3]),
            reg_lambda=float(params[4]),
            subsample=float(params[5]),
            colsample_bytree=float(params[6]),
            random_state=int(params[7]),
        )
        
        model.model_ = xgb.XGBClassifier()
        model.model_.load_model(str(path.with_suffix(".json")))
        model.classes_ = data["classes"]
        
        if len(data["feature_names"]) > 0:
            model.feature_names_ = list(data["feature_names"])
        
        return model
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importances."""
        if self.model_ is None or self.feature_names_ is None:
            return None
        
        importance = self.model_.feature_importances_
        importance = importance / importance.sum()
        
        return dict(zip(self.feature_names_, importance))
    
    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }


# LightGBM wrappers (similar structure)
class LGBMRegressor(ProxyModel):
    """
    LightGBM regressor wrapper (optional).
    
    Requires: pip install lightgbm
    """
    
    name = "lgbm_regressor"
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model_ = None
        self.feature_names_: list[str] | None = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "LGBMRegressor":
        """Fit LightGBM regressor."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            )
        
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if feature_names is not None:
            self.feature_names_ = feature_names
        
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=-1,
        )
        
        self.model_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict(X)
    
    def save(self, path: Path) -> None:
        """Save model."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        
        path = Path(path)
        self.model_.booster_.save_model(str(path.with_suffix(".txt")))
        
        np.savez_compressed(
            path,
            feature_names=np.array(self.feature_names_ or [], dtype=str),
            params=np.array([
                self.n_estimators,
                self.max_depth,
                self.learning_rate,
                self.reg_alpha,
                self.reg_lambda,
                self.subsample,
                self.colsample_bytree,
                self.random_state,
            ]),
        )
    
    @classmethod
    def load(cls, path: Path) -> "LGBMRegressor":
        """Load model."""
        import lightgbm as lgb
        
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        params = data["params"]
        
        model = cls(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            learning_rate=float(params[2]),
            reg_alpha=float(params[3]),
            reg_lambda=float(params[4]),
            subsample=float(params[5]),
            colsample_bytree=float(params[6]),
            random_state=int(params[7]),
        )
        
        model.model_ = lgb.Booster(model_file=str(path.with_suffix(".txt")))
        
        if len(data["feature_names"]) > 0:
            model.feature_names_ = list(data["feature_names"])
        
        return model
    
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importances."""
        if self.model_ is None or self.feature_names_ is None:
            return None
        
        importance = self.model_.feature_importances_
        importance = importance / (importance.sum() + 1e-10)
        
        return dict(zip(self.feature_names_, importance))
    
    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }
