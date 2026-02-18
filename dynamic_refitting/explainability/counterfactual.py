"""Counterfactual explanation generator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.explainability")


class CounterfactualGenerator(BaseStep):
    """Generate counterfactual explanations (what changes flip the prediction).

    Uses a simple greedy search: for each feature, tries shifting values
    toward the desired outcome.

    Parameters
    ----------
    target_class : int
        Desired outcome class (typically the opposite of current prediction).
    max_features_to_change : int
        Maximum number of features to perturb.
    step_size : float
        Fraction of the feature's standard deviation to step.
    max_iterations : int
        Maximum perturbation iterations per feature.
    """

    def __init__(
        self,
        target_class: int = 0,
        max_features_to_change: int = 5,
        step_size: float = 0.1,
        max_iterations: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_class = target_class
        self.max_features_to_change = max_features_to_change
        self.step_size = step_size
        self.max_iterations = max_iterations
        self._model: Any = None
        self._feature_stds: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        model: Any = None,
        **kw: Any,
    ) -> "CounterfactualGenerator":
        if model is None:
            raise ValueError("CounterfactualGenerator requires a model.")
        self._model = model
        for col in X.select_dtypes(include=[np.number]).columns:
            self._feature_stds[col] = float(X[col].std())
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def explain(
        self,
        instance: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a counterfactual for a single instance.

        Parameters
        ----------
        instance : pd.Series
            A single observation.
        feature_names : list[str] | None
            Numeric features to consider perturbing.

        Returns
        -------
        dict
            ``original``, ``counterfactual``, ``changes``, ``success``.
        """
        self._check_fitted()
        if feature_names is None:
            feature_names = [
                c for c in instance.index if c in self._feature_stds
            ]

        current = instance.copy()
        original = instance.copy()

        features_changed: List[str] = []
        for _ in range(self.max_iterations):
            x_df = pd.DataFrame([current]).select_dtypes(include=[np.number])
            try:
                pred_proba = self._model.predict_proba(x_df)[0]
                if np.argmax(pred_proba) == self.target_class:
                    return {
                        "success": True,
                        "original": original.to_dict(),
                        "counterfactual": current.to_dict(),
                        "changes": {
                            f: float(current[f] - original[f])
                            for f in features_changed
                        },
                    }
            except Exception:
                break

            # Perturb the most impactful feature not yet exhausted
            for feat in feature_names:
                if len(features_changed) >= self.max_features_to_change:
                    break
                std = self._feature_stds.get(feat, 1.0)
                delta = std * self.step_size
                # Try both directions
                for direction in [1, -1]:
                    current[feat] = current[feat] + direction * delta
                if feat not in features_changed:
                    features_changed.append(feat)
                break  # one perturbation per iteration

        return {
            "success": False,
            "original": original.to_dict(),
            "counterfactual": current.to_dict(),
            "changes": {
                f: float(current[f] - original[f]) for f in features_changed
            },
        }
