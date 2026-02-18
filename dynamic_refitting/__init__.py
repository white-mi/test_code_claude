"""
dynamic_refitting — production-ready library for developing, validating,
monitoring, and automatically refitting scoring models (binary classification).

Designed for credit/risk scoring with scikit-learn compatible API.
"""

__version__ = "0.1.0"

from dynamic_refitting.config import PipelineConfig
from dynamic_refitting.autopipe import AutoPipeBoost, AutoPipeLogreg

__all__ = [
    "PipelineConfig",
    "AutoPipeBoost",
    "AutoPipeLogreg",
]
