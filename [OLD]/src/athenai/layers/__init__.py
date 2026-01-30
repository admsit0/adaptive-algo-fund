"""
Trainable layer models for enriched RL state (Paso 3).

Proxy Layers train models that predict external market indicators from
internal cluster features. During training, external data (FRED, Fama-French)
is used as targets. During inference (2025+), only internal features are used.

Modules:
- config.py: LayersConfig and proxy-specific configs
- proxies/: Proxy model implementations
  - base.py: ProxyTask, ProxyModel base classes
  - datasets.py: Dataset builders for X(t), y(t+h)
  - models_linear.py: Ridge, LogReg, Softmax (sklearn-free)
  - trainer.py: Walk-forward CV training
  - predictor.py: Inference-time predictions

Proxies:
- Proxy A (VIX): Predicts log(VIX_{t+1}) - market stress
- Proxy B (Rates): Predicts DGS10 change - rate environment
- Proxy C (Recession): Predicts USREC / SP500<MA200 - risk-off
- Proxy D (Factors): Predicts winning factor (SMB/HML/MOM)
"""

from athenai.layers.config import (
    LayersConfig,
    ProxyVIXConfig,
    ProxyRatesConfig,
    ProxyRecessionConfig,
    ProxyFactorsConfig,
    CVConfig,
    UniverseFeaturesConfig,
)

from athenai.layers.proxies import (
    ProxyTask,
    ProxyTaskType,
    ProxyModel,
    ProxyPrediction,
    ProxyDatasetBuilder,
    build_universe_features,
    RidgeRegressor,
    LogisticClassifier,
    SoftmaxClassifier,
    BaselineModel,
    ProxyTrainer,
    ProxyPredictor,
)

__all__ = [
    # Config
    "LayersConfig",
    "ProxyVIXConfig",
    "ProxyRatesConfig",
    "ProxyRecessionConfig",
    "ProxyFactorsConfig",
    "CVConfig",
    "UniverseFeaturesConfig",
    # Core
    "ProxyTask",
    "ProxyTaskType",
    "ProxyModel",
    "ProxyPrediction",
    # Datasets
    "ProxyDatasetBuilder",
    "build_universe_features",
    # Models
    "RidgeRegressor",
    "LogisticClassifier",
    "SoftmaxClassifier",
    "BaselineModel",
    # Training/Inference
    "ProxyTrainer",
    "ProxyPredictor",
]
