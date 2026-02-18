"""Validation / data-quality steps for scoring pipelines."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep
from dynamic_refitting.utils.metrics import calc_psi

logger = logging.getLogger("dynamic_refitting.validation")


# ---------------------------------------------------------------------------
# FeatureCleanerConst — drop constant features
# ---------------------------------------------------------------------------

class FeatureCleanerConst(BaseStep):
    """Drop features that are constant (zero variance).

    Parameters
    ----------
    threshold : float
        Number of unique values ratio below which a feature is constant.
    """

    def __init__(self, threshold: float = 1e-10, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.constant_cols_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeatureCleanerConst":
        self.constant_cols_ = []
        for col in X.columns:
            if X[col].nunique(dropna=True) <= 1:
                self.constant_cols_.append(col)
            elif X[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                if X[col].var() < self.threshold:
                    self.constant_cols_.append(col)
        self._fitted = True
        logger.info(
            "FeatureCleanerConst: found %d constant columns.", len(self.constant_cols_)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return X.drop(
            columns=[c for c in self.constant_cols_ if c in X.columns]
        ).copy()


# ---------------------------------------------------------------------------
# FeatureCleanerNan — drop / impute features with excessive NaN
# ---------------------------------------------------------------------------

class FeatureCleanerNan(BaseStep):
    """Handle columns with too many missing values.

    Parameters
    ----------
    nan_threshold : float
        If NaN fraction >= threshold, the column is dropped.
    fill_strategy : str
        How to fill remaining NaNs: ``"median"``, ``"mean"``, ``"zero"``.
    """

    def __init__(
        self,
        nan_threshold: float = 0.9,
        fill_strategy: str = "median",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nan_threshold = nan_threshold
        self.fill_strategy = fill_strategy
        self.drop_cols_: List[str] = []
        self._fill_values: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeatureCleanerNan":
        self.drop_cols_ = []
        self._fill_values = {}
        for col in X.columns:
            nan_frac = X[col].isna().mean()
            if nan_frac >= self.nan_threshold:
                self.drop_cols_.append(col)
            elif X[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                if self.fill_strategy == "median":
                    self._fill_values[col] = float(X[col].median())
                elif self.fill_strategy == "mean":
                    self._fill_values[col] = float(X[col].mean())
                else:
                    self._fill_values[col] = 0.0
        self._fitted = True
        logger.info(
            "FeatureCleanerNan: dropping %d columns, filling %d.",
            len(self.drop_cols_), len(self._fill_values),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.drop(
            columns=[c for c in self.drop_cols_ if c in X.columns]
        ).copy()
        for col, val in self._fill_values.items():
            if col in Xt.columns:
                Xt[col] = Xt[col].fillna(val)
        return Xt


# ---------------------------------------------------------------------------
# FeatureCleanerUnivariate — filter by AUC / IV
# ---------------------------------------------------------------------------

class FeatureCleanerUnivariate(BaseStep):
    """Drop features that have poor univariate discriminative power.

    Parameters
    ----------
    min_auc : float
        Minimum single-feature AUC (or 1-AUC, whichever is higher).
    """

    def __init__(self, min_auc: float = 0.52, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_auc = min_auc
        self.feature_aucs_: Dict[str, float] = {}
        self.selected_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeatureCleanerUnivariate":
        if y is None:
            raise ValueError("FeatureCleanerUnivariate requires y.")
        from sklearn.metrics import roc_auc_score

        self.feature_aucs_ = {}
        self.selected_features_ = []
        for col in X.select_dtypes(include=[np.number]).columns:
            valid = X[col].dropna()
            y_valid = y.loc[valid.index]
            if len(y_valid.unique()) < 2 or len(valid) < 10:
                continue
            try:
                auc = roc_auc_score(y_valid, valid)
                auc = max(auc, 1 - auc)  # handle inverse relationship
            except Exception:
                auc = 0.5
            self.feature_aucs_[col] = auc
            if auc >= self.min_auc:
                self.selected_features_.append(col)

        self._fitted = True
        logger.info(
            "FeatureCleanerUnivariate: kept %d / %d features (min_auc=%.3f).",
            len(self.selected_features_),
            len(self.feature_aucs_),
            self.min_auc,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available + non_numeric].copy()


# ---------------------------------------------------------------------------
# HitrateChecker — ensure minimum event rate per segment
# ---------------------------------------------------------------------------

class HitrateChecker(BaseStep):
    """Check that the positive class rate falls within expected bounds.

    Parameters
    ----------
    min_rate : float
        Minimum expected event rate.
    max_rate : float
        Maximum expected event rate.
    """

    def __init__(
        self,
        min_rate: float = 0.01,
        max_rate: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.actual_rate_: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "HitrateChecker":
        if y is None:
            raise ValueError("HitrateChecker requires y.")
        self.actual_rate_ = float(y.mean())
        if self.actual_rate_ < self.min_rate or self.actual_rate_ > self.max_rate:
            logger.warning(
                "HitrateChecker: actual event rate %.4f outside expected "
                "range [%.4f, %.4f].",
                self.actual_rate_, self.min_rate, self.max_rate,
            )
        else:
            logger.info("HitrateChecker: event rate %.4f OK.", self.actual_rate_)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


# ---------------------------------------------------------------------------
# WoEStabChecker — check WoE stability over time
# ---------------------------------------------------------------------------

class WoEStabChecker(BaseStep):
    """Check Weight of Evidence stability across time periods.

    Parameters
    ----------
    time_col : str
        Column with time period information.
    target_col : str
        Target column name.
    psi_threshold : float
        PSI threshold for instability warning.
    """

    def __init__(
        self,
        time_col: str = "date",
        target_col: str = "target",
        psi_threshold: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.time_col = time_col
        self.target_col = target_col
        self.psi_threshold = psi_threshold
        self.stability_report_: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "WoEStabChecker":
        if self.time_col not in X.columns:
            logger.warning("WoEStabChecker: time_col '%s' not found.", self.time_col)
            self._fitted = True
            return self

        periods = sorted(X[self.time_col].unique())
        if len(periods) < 2:
            self._fitted = True
            return self

        num_cols = X.select_dtypes(include=[np.number]).columns
        ref_period = periods[0]
        ref_data = X[X[self.time_col] == ref_period]

        for col in num_cols:
            if col == self.target_col:
                continue
            psi_values = {}
            ref_vals = ref_data[col].dropna().values
            if len(ref_vals) < 10:
                continue
            for period in periods[1:]:
                period_data = X[X[self.time_col] == period]
                period_vals = period_data[col].dropna().values
                if len(period_vals) < 10:
                    continue
                psi = calc_psi(ref_vals, period_vals)
                psi_values[str(period)] = psi
                if psi > self.psi_threshold:
                    logger.warning(
                        "WoEStabChecker: %s has PSI=%.3f in period %s "
                        "(> %.3f threshold).",
                        col, psi, period, self.psi_threshold,
                    )
            if psi_values:
                self.stability_report_[col] = psi_values

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


# ---------------------------------------------------------------------------
# PopulationStabilityIndex — stand-alone PSI check
# ---------------------------------------------------------------------------

class PopulationStabilityIndex(BaseStep):
    """Compute PSI between training and new data distributions.

    Parameters
    ----------
    n_bins : int
        Number of bins for PSI computation.
    threshold : float
        Alert threshold.
    """

    def __init__(
        self,
        n_bins: int = 10,
        threshold: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.threshold = threshold
        self._reference_distributions: Dict[str, np.ndarray] = {}
        self.psi_values_: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "PopulationStabilityIndex":
        for col in X.select_dtypes(include=[np.number]).columns:
            self._reference_distributions[col] = X[col].dropna().values.copy()
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self.psi_values_ = {}
        for col, ref in self._reference_distributions.items():
            if col not in X.columns:
                continue
            actual = X[col].dropna().values
            if len(ref) < 10 or len(actual) < 10:
                continue
            psi = calc_psi(ref, actual, n_bins=self.n_bins)
            self.psi_values_[col] = psi
            if psi > self.threshold:
                logger.warning(
                    "PSI: %s = %.3f (> %.3f threshold)", col, psi, self.threshold
                )
        return X
