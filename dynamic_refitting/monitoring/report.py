"""Drift report generator."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep
from dynamic_refitting.monitoring.drift import FeatureDriftDetector, PredictionDriftMonitor
from dynamic_refitting.monitoring.performance import ModelPerformanceMonitor

logger = logging.getLogger("dynamic_refitting.monitoring")


class DriftReportGenerator(BaseStep):
    """Generate comprehensive drift and performance reports.

    Combines outputs from :class:`FeatureDriftDetector`,
    :class:`PredictionDriftMonitor`, and :class:`ModelPerformanceMonitor`
    into a single JSON/dict report.

    Parameters
    ----------
    output_dir : str | Path | None
        Directory to save reports.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir) if output_dir else None
        self.latest_report_: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "DriftReportGenerator":
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def generate(
        self,
        feature_drift: Optional[FeatureDriftDetector] = None,
        prediction_drift: Optional[PredictionDriftMonitor] = None,
        performance_monitor: Optional[ModelPerformanceMonitor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the full report.

        Returns
        -------
        dict
            Structured report with drift, performance, and summary fields.
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        # Feature drift
        if feature_drift is not None:
            report["feature_drift"] = {
                "total_features_checked": len(feature_drift.drift_report_),
                "drifted_features": feature_drift.drifted_features,
                "n_drifted": len(feature_drift.drifted_features),
                "details": feature_drift.drift_report_,
            }

        # Prediction drift
        if prediction_drift is not None:
            report["prediction_drift"] = {
                "psi": prediction_drift.prediction_psi_,
                "drifted": prediction_drift.drifted_,
            }

        # Performance
        if performance_monitor is not None:
            report["performance"] = {
                "history": performance_monitor.performance_history_,
                "rolling_auc": performance_monitor.get_rolling_auc(),
            }

        # Summary
        needs_refit = False
        reasons: List[str] = []
        if feature_drift and len(feature_drift.drifted_features) > 0:
            needs_refit = True
            reasons.append(
                f"{len(feature_drift.drifted_features)} features drifted"
            )
        if prediction_drift and prediction_drift.drifted_:
            needs_refit = True
            reasons.append(
                f"prediction PSI={prediction_drift.prediction_psi_:.3f}"
            )

        report["summary"] = {
            "needs_refit": needs_refit,
            "reasons": reasons,
        }

        self.latest_report_ = report

        if self.output_dir:
            self._save_report(report)

        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"drift_report_{ts}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Saved drift report to %s", path)
