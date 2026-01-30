"""
Proxy layer models for enriched RL state.

This module provides trainable proxy models that learn to predict external
market indicators (VIX, rates, recession, factors) from internal cluster features.

During TRAINING (2020-2024):
- External data (FRED, Fama-French) is used as targets (Y)
- Internal cluster features are used as inputs (X)
- Models are trained with walk-forward CV

During INFERENCE (2025+):
- Only internal features (X) are used
- Trained models predict proxy signals
- No external API calls needed

Modules:
- base.py: ProxyTask base class and interfaces
- datasets.py: Dataset builders for X(t), y(t+h)
- models_linear.py: Ridge/LogReg implementations (sklearn-free)
- trainer.py: Walk-forward CV training
- predictor.py: Inference-time predictions
"""

from athenai.layers.proxies.base import (
    ProxyTask,
    ProxyTaskType,
    ProxyModel,
    ProxyPrediction,
)
from athenai.layers.proxies.datasets import (
    ProxyDatasetBuilder,
    build_universe_features,
)
from athenai.layers.proxies.models_linear import (
    RidgeRegressor,
    LogisticClassifier,
    SoftmaxClassifier,
    BaselineModel,
)
from athenai.layers.proxies.trainer import ProxyTrainer
from athenai.layers.proxies.predictor import ProxyPredictor

__all__ = [
    "ProxyTask",
    "ProxyTaskType",
    "ProxyModel",
    "ProxyPrediction",
    "ProxyDatasetBuilder",
    "build_universe_features",
    "RidgeRegressor",
    "LogisticClassifier",
    "SoftmaxClassifier",
    "BaselineModel",
    "ProxyTrainer",
    "ProxyPredictor",
]
