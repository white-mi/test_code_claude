"""Feature and prediction drift detectors."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from dynamic_refitting.config import BaseStep
from dynamic_refitting.utils.metrics import calc_psi

logger = logging.getLogger("dynamic_refitting.monitoring")


class FeatureDriftDetector(BaseStep):
    """Detect feature-level distribution drift between reference and current data.

    Computes PSI and KS-test for each numeric feature.

    Parameters
    ----------
    psi_threshold : float
        PSI threshold above which drift is flagged.
    ks_threshold : float
        KS p-value threshold below which drift is flagged.
    n_bins : int
        Number of bins for PSI.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        n_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.n_bins = n_bins
        self._reference: Dict[str, np.ndarray] = {}
        self.drift_report_: Dict[str, Dict[str, Any]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeatureDriftDetector":
        """Store reference (training) distributions."""
        for col in X.select_dtypes(include=[np.number]).columns:
            vals = X[col].dropna().values
            if len(vals) > 0:
                self._reference[col] = vals.copy()
        self._fitted = True
        logger.info(
            "FeatureDriftDetector: stored reference for %d features.",
            len(self._reference),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute drift metrics and store them in ``self.drift_report_``."""
        self._check_fitted()
        self.drift_report_ = {}
        for col, ref_vals in self._reference.items():
            if col not in X.columns:
                continue
            cur_vals = X[col].dropna().values
            if len(cur_vals) < 10:
                continue
            psi = calc_psi(ref_vals, cur_vals, n_bins=self.n_bins)
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)

            drifted = psi > self.psi_threshold or ks_pval < self.ks_threshold
            self.drift_report_[col] = {
                "psi": float(psi),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "drifted": drifted,
            }
            if drifted:
                logger.warning(
                    "Drift detected: %s (PSI=%.3f, KS p=%.4f)",
                    col, psi, ks_pval,
                )
        return X

    @property
    def drifted_features(self) -> List[str]:
        """Features flagged as drifted."""
        return [
            col for col, info in self.drift_report_.items() if info["drifted"]
        ]


class PredictionDriftMonitor(BaseStep):
    """Monitor drift in predicted probabilities.

    Parameters
    ----------
    psi_threshold : float
        PSI threshold for prediction distribution shift.
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        n_bins: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self._reference_scores: Optional[np.ndarray] = None
        self.prediction_psi_: float = 0.0
        self.drifted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        reference_scores: Optional[np.ndarray] = None,
        **kw: Any,
    ) -> "PredictionDriftMonitor":
        """Store reference prediction scores."""
        if reference_scores is not None:
            self._reference_scores = np.asarray(reference_scores)
        self._fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        current_scores: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compare current scores to reference."""
        self._check_fitted()
        if self._reference_scores is not None and current_scores is not None:
            self.prediction_psi_ = calc_psi(
                self._reference_scores,
                np.asarray(current_scores),
                n_bins=self.n_bins,
            )
            self.drifted_ = self.prediction_psi_ > self.psi_threshold
            if self.drifted_:
                logger.warning(
                    "Prediction drift: PSI=%.3f (> %.3f)",
                    self.prediction_psi_, self.psi_threshold,
                )
        return X

    def check(self, current_scores: np.ndarray) -> Dict[str, Any]:
        """Convenience method: compute PSI and return a dict."""
        if self._reference_scores is None:
            raise RuntimeError("No reference scores set. Call fit() first.")
        psi = calc_psi(self._reference_scores, current_scores, n_bins=self.n_bins)
        drifted = psi > self.psi_threshold
        return {"psi": psi, "drifted": drifted}
