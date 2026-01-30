"""Feature engineering modules."""

from athenai.features.personality import (
    PersonalityConfig,
    BuildAlgoPersonalityStaticStep,
)

__all__ = [
    "PersonalityConfig",
    "BuildAlgoPersonalityStaticStep",
]

# Future modules:
# - factor_exposures.py: Beta, PCA loadings
# - regime_signatures.py: Behavior in different market regimes
