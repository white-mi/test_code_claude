"""Model performance monitoring over time."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep
from dynamic_refitting.utils.metrics import calc_auc, calc_ks, calc_brier

logger = logging.getLogger("dynamic_refitting.monitoring")


class ModelPerformanceMonitor(BaseStep):
    """Track model performance (AUC, KS, Brier) over time windows.

    Parameters
    ----------
    time_col : str
        Column identifying time periods.
    metrics : list[str]
        Which metrics to track.
    auc_threshold : float
        Minimum acceptable AUC before alerting.
    """

    def __init__(
        self,
        time_col: str = "date",
        metrics: Optional[List[str]] = None,
        auc_threshold: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.time_col = time_col
        self.metrics = metrics or ["auc", "ks", "brier"]
        self.auc_threshold = auc_threshold
        self.performance_history_: List[Dict[str, Any]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "ModelPerformanceMonitor":
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def evaluate(
        self,
        y_true: pd.Series,
        y_score: np.ndarray,
        time_values: Optional[pd.Series] = None,
    ) -> List[Dict[str, Any]]:
        """Compute metrics, optionally per time period.

        Parameters
        ----------
        y_true : pd.Series
            True labels.
        y_score : np.ndarray
            Predicted probabilities.
        time_values : pd.Series | None
            Time period for each sample.

        Returns
        -------
        list[dict]
            List of metric dictionaries per period (or a single global entry).
        """
        results = []

        if time_values is not None:
            periods = sorted(time_values.unique())
            for period in periods:
                mask = time_values == period
                yt = y_true[mask]
                ys = y_score[mask]
                if len(yt.unique()) < 2 or len(yt) < 10:
                    continue
                record = self._compute_metrics(yt.values, ys)
                record["period"] = str(period)
                results.append(record)
                if "auc" in record and record["auc"] < self.auc_threshold:
                    logger.warning(
                        "Performance degradation: AUC=%.3f in period %s",
                        record["auc"], period,
                    )
        else:
            record = self._compute_metrics(y_true.values, y_score)
            record["period"] = "global"
            results.append(record)

        self.performance_history_.extend(results)
        return results

    def _compute_metrics(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for m in self.metrics:
            if m == "auc":
                result["auc"] = calc_auc(y_true, y_score)
            elif m == "ks":
                result["ks"] = calc_ks(y_true, y_score)
            elif m == "brier":
                result["brier"] = calc_brier(y_true, y_score)
        return result

    def get_rolling_auc(self, window: int = 3) -> List[float]:
        """Compute rolling AUC from performance history."""
        aucs = [
            r["auc"]
            for r in self.performance_history_
            if "auc" in r
        ]
        if len(aucs) < window:
            return aucs
        return [
            np.mean(aucs[i : i + window])
            for i in range(len(aucs) - window + 1)
        ]
