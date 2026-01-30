"""
External data fetchers for TRAIN phase only.

These modules download and cache external market data used as targets (Y)
for training proxy models. During INFERENCE (2025+), no external data
is fetched - only internal features are used.

Modules:
- fred.py: FRED API for VIX (VIXCLS), Treasury (DGS10), Recession (USREC), SP500
- famafrench.py: Kenneth French Data Library for factor returns (SMB/HML/MOM)
- schemas.py: Data normalization and alignment utilities
"""

from athenai.external.fred import FREDFetcher
from athenai.external.famafrench import FamaFrenchFetcher
from athenai.external.schemas import (
    align_to_daily,
    align_to_monthly,
    forward_fill_with_lag,
)

__all__ = [
    "FREDFetcher",
    "FamaFrenchFetcher",
    "align_to_daily",
    "align_to_monthly",
    "forward_fill_with_lag",
]
