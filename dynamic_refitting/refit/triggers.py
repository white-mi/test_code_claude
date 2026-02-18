"""Refit trigger conditions."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from dynamic_refitting.utils.metrics import calc_auc, calc_psi

logger = logging.getLogger("dynamic_refitting.refit")


class BaseRefitTrigger(ABC):
    """Abstract trigger that decides whether a model refit is needed."""

    @abstractmethod
    def should_refit(self, **kwargs: Any) -> bool:
        """Return True if refit conditions are met."""
        ...

    @abstractmethod
    def get_reason(self) -> str:
        """Human-readable explanation of why refit was triggered."""
        ...


class PerformanceTriggeredRefit(BaseRefitTrigger):
    """Trigger refit when model AUC drops below a threshold.

    Parameters
    ----------
    auc_threshold : float
        Minimum acceptable AUC.
    psi_threshold : float
        Maximum acceptable PSI (prediction-level).
    """

    def __init__(
        self,
        auc_threshold: float = 0.65,
        psi_threshold: float = 0.25,
    ) -> None:
        self.auc_threshold = auc_threshold
        self.psi_threshold = psi_threshold
        self._reason = ""

    def should_refit(
        self,
        y_true: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        reference_scores: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> bool:
        reasons = []

        if y_true is not None and y_score is not None:
            if len(np.unique(y_true)) >= 2:
                current_auc = calc_auc(y_true, y_score)
                if current_auc < self.auc_threshold:
                    reasons.append(
                        f"AUC={current_auc:.4f} < {self.auc_threshold}"
                    )

        if reference_scores is not None and y_score is not None:
            psi = calc_psi(reference_scores, y_score)
            if psi > self.psi_threshold:
                reasons.append(f"PSI={psi:.4f} > {self.psi_threshold}")

        self._reason = "; ".join(reasons) if reasons else ""
        triggered = len(reasons) > 0
        if triggered:
            logger.warning("PerformanceTriggeredRefit: %s", self._reason)
        return triggered

    def get_reason(self) -> str:
        return self._reason


class TimeBasedRefit(BaseRefitTrigger):
    """Trigger refit after a fixed time period has elapsed.

    Parameters
    ----------
    interval_days : int
        Number of days between refits.
    """

    def __init__(self, interval_days: int = 30) -> None:
        self.interval_days = interval_days
        self._last_refit_ts: float = time.time()
        self._reason = ""

    def should_refit(self, **kwargs: Any) -> bool:
        elapsed = time.time() - self._last_refit_ts
        days_elapsed = elapsed / 86400
        if days_elapsed >= self.interval_days:
            self._reason = (
                f"Time elapsed: {days_elapsed:.1f} days >= {self.interval_days}"
            )
            logger.info("TimeBasedRefit: %s", self._reason)
            return True
        self._reason = ""
        return False

    def reset(self) -> None:
        """Mark the current time as the last refit time."""
        self._last_refit_ts = time.time()

    def get_reason(self) -> str:
        return self._reason


class DataVolumeTriggeredRefit(BaseRefitTrigger):
    """Trigger refit when enough new data has accumulated.

    Parameters
    ----------
    min_new_samples : int
        Minimum number of new labelled samples needed.
    """

    def __init__(self, min_new_samples: int = 1000) -> None:
        self.min_new_samples = min_new_samples
        self._accumulated = 0
        self._reason = ""

    def add_samples(self, n: int) -> None:
        """Register that *n* new labelled samples are available."""
        self._accumulated += n

    def should_refit(self, **kwargs: Any) -> bool:
        if self._accumulated >= self.min_new_samples:
            self._reason = (
                f"Accumulated {self._accumulated} samples "
                f">= {self.min_new_samples}"
            )
            logger.info("DataVolumeTriggeredRefit: %s", self._reason)
            return True
        self._reason = ""
        return False

    def reset(self) -> None:
        self._accumulated = 0

    def get_reason(self) -> str:
        return self._reason
