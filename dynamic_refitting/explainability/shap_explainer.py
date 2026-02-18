"""SHAP-based model explainability."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.explainability")


class ShapExplainer(BaseStep):
    """Wrapper around SHAP for tree and linear models.

    Parameters
    ----------
    model_type : str
        ``"tree"`` or ``"linear"``.
    max_samples : int
        Maximum background samples for SHAP computation.
    """

    def __init__(
        self,
        model_type: str = "tree",
        max_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_type = model_type
        self.max_samples = max_samples
        self._explainer: Any = None
        self.shap_values_: Optional[np.ndarray] = None
        self.feature_importance_: Optional[pd.Series] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        model: Any = None,
        **kw: Any,
    ) -> "ShapExplainer":
        """Initialize the SHAP explainer.

        Parameters
        ----------
        model : Any
            Fitted model (e.g. LGBMClassifier, LogisticRegression).
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "shap is required for ShapExplainer. "
                "Install it with: pip install shap"
            )
        if model is None:
            raise ValueError("ShapExplainer requires a fitted model.")

        background = X.select_dtypes(include=[np.number]).iloc[: self.max_samples]

        if self.model_type == "tree":
            self._explainer = shap.TreeExplainer(model)
        elif self.model_type == "linear":
            self._explainer = shap.LinearExplainer(model, background)
        else:
            self._explainer = shap.KernelExplainer(
                model.predict_proba, background
            )
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the given data.

        Returns
        -------
        np.ndarray
            SHAP values array (n_samples x n_features).
        """
        self._check_fitted()
        X_num = X.select_dtypes(include=[np.number])
        shap_values = self._explainer.shap_values(X_num)
        # For binary classification, take positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        self.shap_values_ = np.asarray(shap_values)
        self.feature_importance_ = pd.Series(
            np.abs(self.shap_values_).mean(axis=0),
            index=X_num.columns,
        ).sort_values(ascending=False)
        return self.shap_values_

    def get_top_features(self, n: int = 10) -> List[str]:
        """Return top N features by mean |SHAP|."""
        if self.feature_importance_ is None:
            raise RuntimeError("Call .explain() first.")
        return self.feature_importance_.head(n).index.tolist()
