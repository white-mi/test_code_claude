"""Permutation importance for any model."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_perm_importance

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.explainability")


class PermutationImportance(BaseStep):
    """Permutation-based feature importance.

    Parameters
    ----------
    n_repeats : int
        Number of permutation repeats.
    scoring : str
        Scoring metric (e.g. ``"roc_auc"``).
    """

    def __init__(
        self,
        n_repeats: int = 10,
        scoring: str = "roc_auc",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.importances_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        model: Any = None,
        **kw: Any,
    ) -> "PermutationImportance":
        if model is None or y is None:
            raise ValueError("PermutationImportance requires model and y.")
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        result = sklearn_perm_importance(
            model,
            X_num,
            y,
            n_repeats=self.n_repeats,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.importances_ = pd.DataFrame(
            {
                "mean": result.importances_mean,
                "std": result.importances_std,
            },
            index=X_num.columns,
        ).sort_values("mean", ascending=False)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def get_top_features(self, n: int = 10) -> List[str]:
        if self.importances_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.importances_.head(n).index.tolist()
