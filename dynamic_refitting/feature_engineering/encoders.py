"""Categorical encoding steps: target encoding, frequency encoding, embeddings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.feature_engineering")


class TargetEncoderCV(BaseStep):
    """Cross-validated target encoder with smoothing.

    During ``fit``, computes per-fold target-encoded values to avoid
    leakage.  During ``transform`` (on unseen data), applies the global
    encoding learned from the full training set.

    Parameters
    ----------
    cols : list[str] | None
        Columns to encode. If *None*, encodes all object/category columns.
    n_folds : int
        Number of CV folds for encoding.
    smoothing : float
        Smoothing factor (higher = more regularisation toward global mean).
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        n_folds: int = 5,
        smoothing: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self._encoding_map: Dict[str, Dict[Any, float]] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "TargetEncoderCV":
        if y is None:
            raise ValueError("TargetEncoderCV requires y.")
        cols = self.cols or list(X.select_dtypes(include=["object", "category"]).columns)
        self._global_mean = float(y.mean())
        for col in cols:
            counts = X[col].value_counts()
            means = y.groupby(X[col]).mean()
            smooth_map = {}
            for cat in counts.index:
                n = counts[cat]
                cat_mean = means[cat]
                smooth_map[cat] = (
                    (n * cat_mean + self.smoothing * self._global_mean)
                    / (n + self.smoothing)
                )
            self._encoding_map[col] = smooth_map
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for col, mapping in self._encoding_map.items():
            if col in Xt.columns:
                Xt[col] = Xt[col].map(mapping).fillna(self._global_mean).astype(float)
        return Xt

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> pd.DataFrame:
        """CV-based fit_transform to prevent leakage during training."""
        if y is None:
            raise ValueError("TargetEncoderCV requires y.")

        cols = self.cols or list(X.select_dtypes(include=["object", "category"]).columns)
        self._global_mean = float(y.mean())

        Xt = X.copy()
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for col in cols:
            encoded = pd.Series(np.nan, index=X.index, dtype=float)
            for train_idx, val_idx in kf.split(X):
                X_tr = X.iloc[train_idx]
                y_tr = y.iloc[train_idx]
                counts = X_tr[col].value_counts()
                means = y_tr.groupby(X_tr[col]).mean()
                smooth_map = {}
                for cat in counts.index:
                    n = counts[cat]
                    cat_mean = means[cat]
                    smooth_map[cat] = (
                        (n * cat_mean + self.smoothing * self._global_mean)
                        / (n + self.smoothing)
                    )
                encoded.iloc[val_idx] = X.iloc[val_idx][col].map(smooth_map)
            encoded.fillna(self._global_mean, inplace=True)
            Xt[col] = encoded.astype(float)

        # Also fit global mapping for transform on new data
        self.fit(X, y)
        return Xt


class FrequencyEncoder(BaseStep):
    """Encode categorical values by their frequency in the training set.

    Parameters
    ----------
    cols : list[str] | None
        Columns to encode. If *None*, encodes all object/category columns.
    normalize : bool
        If *True*, frequencies are divided by the total count.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self.normalize = normalize
        self._freq_map: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "FrequencyEncoder":
        cols = self.cols or list(X.select_dtypes(include=["object", "category"]).columns)
        for col in cols:
            vc = X[col].value_counts(normalize=self.normalize)
            self._freq_map[col] = vc.to_dict()
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for col, mapping in self._freq_map.items():
            if col in Xt.columns:
                Xt[col] = Xt[col].map(mapping).fillna(0.0).astype(float)
        return Xt


class CategoryEmbedder(BaseStep):
    """Simple ordinal encoding as a lightweight 'embedding'.

    Maps each unique category to an integer. For a real embedding layer,
    integrate with a neural model.

    Parameters
    ----------
    cols : list[str] | None
        Columns to encode.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self._mapping: Dict[str, Dict[Any, int]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "CategoryEmbedder":
        cols = self.cols or list(X.select_dtypes(include=["object", "category"]).columns)
        for col in cols:
            uniq = sorted(X[col].dropna().unique(), key=str)
            self._mapping[col] = {v: i for i, v in enumerate(uniq)}
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for col, mapping in self._mapping.items():
            if col in Xt.columns:
                Xt[col] = Xt[col].map(mapping).fillna(-1).astype(int)
        return Xt
