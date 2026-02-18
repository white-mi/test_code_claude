"""Partial Dependence Plot computation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.explainability")


class PartialDependence(BaseStep):
    """Compute partial dependence values for specified features.

    Parameters
    ----------
    features : list[str] | None
        Features to compute PDP for.
    grid_resolution : int
        Number of grid points.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        grid_resolution: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.features = features
        self.grid_resolution = grid_resolution
        self.pdp_results_: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        model: Any = None,
        **kw: Any,
    ) -> "PartialDependence":
        if model is None:
            raise ValueError("PartialDependence requires a fitted model.")

        features = self.features or list(
            X.select_dtypes(include=[np.number]).columns[:10]
        )
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        for feat in features:
            if feat not in X_num.columns:
                continue
            grid = np.linspace(
                X_num[feat].min(), X_num[feat].max(), self.grid_resolution
            )
            avg_predictions = []
            for val in grid:
                X_temp = X_num.copy()
                X_temp[feat] = val
                try:
                    preds = model.predict_proba(X_temp)[:, 1]
                except (AttributeError, IndexError):
                    preds = model.predict(X_temp)
                avg_predictions.append(np.mean(preds))
            self.pdp_results_[feat] = {
                "grid": grid,
                "avg_prediction": np.array(avg_predictions),
            }
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
