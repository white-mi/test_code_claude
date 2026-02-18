"""Global configuration and base classes for the dynamic_refitting library."""

from __future__ import annotations

import logging
import json
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("dynamic_refitting")


# ---------------------------------------------------------------------------
# Pipeline configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Centralised configuration for an entire scoring pipeline.

    Parameters
    ----------
    random_state : int
        Global random seed propagated to every stochastic component.
    n_jobs : int
        Default parallelism level (-1 = all cores).
    target_col : str
        Name of the binary target column.
    time_col : str | None
        Name of the datetime / period column used for time-based splitting.
    pos_label : int
        Positive class label (default 1).
    metrics : list[str]
        Metrics to compute during validation.  Supported: ``"auc"``,
        ``"ks"``, ``"brier"``, ``"psi"``, ``"confusion_matrix"``.
    sample_weight_col : str | None
        Column name for sample weights (if any).
    """

    random_state: int = 42
    n_jobs: int = 1
    target_col: str = "target"
    time_col: Optional[str] = None
    pos_label: int = 1
    metrics: List[str] = field(
        default_factory=lambda: ["auc", "ks", "brier", "psi"]
    )
    sample_weight_col: Optional[str] = None

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Abstract base step
# ---------------------------------------------------------------------------

class BaseStep:
    """Base class for every pipeline step.

    Provides a unified interface compatible with scikit-learn:
    ``fit``, ``transform``, ``fit_transform``, ``predict``, ``predict_proba``,
    ``save``, ``load``.

    Sub-classes **must** override at least ``fit`` and ``transform``.
    """

    _fitted: bool = False

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs

    # -- core interface -------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "BaseStep":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.fit(X, y, **kwargs)
        return self.transform(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("This step does not support predict.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("This step does not support predict_proba.")

    # -- serialisation --------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved %s to %s", self.__class__.__name__, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseStep":
        obj = joblib.load(path)
        logger.info("Loaded %s from %s", obj.__class__.__name__, path)
        return obj

    # -- helpers --------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet. "
                "Call .fit() first."
            )

    def get_params(self) -> Dict[str, Any]:
        """Return constructor parameters (sklearn convention)."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def __repr__(self) -> str:  # pragma: no cover
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"


# ---------------------------------------------------------------------------
# Scoring pipeline (ordered sequence of BaseStep)
# ---------------------------------------------------------------------------

class ScoringPipeline:
    """Ordered pipeline of :class:`BaseStep` instances.

    Works similarly to ``sklearn.pipeline.Pipeline`` but supports
    ``predict_proba`` and full serialisation of all steps.
    """

    def __init__(
        self,
        steps: Sequence[BaseStep],
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.steps = list(steps)
        self.config = config or PipelineConfig()
        self._fitted = False

    # -- convenience ----------------------------------------------------------

    @property
    def transformer_steps(self) -> List[BaseStep]:
        return self.steps[:-1]

    @property
    def estimator(self) -> BaseStep:
        return self.steps[-1]

    # -- core interface -------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "ScoringPipeline":
        Xt = X.copy()
        for step in self.transformer_steps:
            Xt = step.fit_transform(Xt, y, **kwargs)
        self.estimator.fit(Xt, y, **kwargs)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = X.copy()
        for step in self.transformer_steps:
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.transform(X)
        return self.estimator.predict(Xt)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.transform(X)
        return self.estimator.predict_proba(Xt)

    # -- serialisation --------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved ScoringPipeline (%d steps) to %s", len(self.steps), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScoringPipeline":
        obj = joblib.load(path)
        logger.info("Loaded ScoringPipeline from %s", path)
        return obj

    def __repr__(self) -> str:  # pragma: no cover
        step_names = [s.__class__.__name__ for s in self.steps]
        return f"ScoringPipeline(steps={step_names})"
