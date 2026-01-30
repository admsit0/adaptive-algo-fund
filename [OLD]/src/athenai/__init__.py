"""
AthenAI - Macro Portfolio Management Pipeline
============================================

A modular pipeline for financial data preprocessing, clustering,
macro estimation, and RL-based portfolio management.
"""

__version__ = "0.1.0"

from athenai.core.config import PreprocessConfig
from athenai.core.artifacts import ArtifactStore

__all__ = ["PreprocessConfig", "ArtifactStore", "__version__"]
