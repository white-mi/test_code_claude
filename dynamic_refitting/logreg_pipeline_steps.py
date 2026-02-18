"""Logistic regression pipeline steps: WoE, binning, stepwise selection."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.logreg")


# ---------------------------------------------------------------------------
# FeatureLinearizer (monotonic binning + WoE transform)
# ---------------------------------------------------------------------------

class FeatureLinearizer(BaseStep):
    """Monotonic binning of continuous features.

    For each numeric feature, finds optimal bins (equal-frequency or
    decision-tree-based) ensuring a monotonic relationship between
    the bin order and the event rate.

    Parameters
    ----------
    n_bins : int
        Maximum number of bins.
    min_bin_size : float
        Minimum fraction of samples per bin.
    monotonic : bool
        Enforce monotonic WoE across bins.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_bin_size: float = 0.05,
        monotonic: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.monotonic = monotonic
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._bin_woe: Dict[str, Dict[int, float]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeatureLinearizer":
        if y is None:
            raise ValueError("FeatureLinearizer requires y.")
        y = y.astype(int)
        for col in X.select_dtypes(include=[np.number]).columns:
            edges, woe_map = self._fit_column(X[col], y)
            self._bin_edges[col] = edges
            self._bin_woe[col] = woe_map
        self._fitted = True
        logger.info("FeatureLinearizer: binned %d features.", len(self._bin_edges))
        return self

    def _fit_column(
        self,
        series: pd.Series,
        y: pd.Series,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """Compute bin edges and WoE for a single column."""
        clean = series.dropna()
        clean_y = y.loc[clean.index]

        # Equal-frequency bins
        try:
            _, edges = pd.qcut(
                clean, q=self.n_bins, retbins=True, duplicates="drop"
            )
        except ValueError:
            edges = np.array([-np.inf, np.inf])

        edges[0] = -np.inf
        edges[-1] = np.inf

        # Merge small bins
        edges = self._merge_small_bins(clean, clean_y, edges)

        # Enforce monotonicity
        if self.monotonic:
            edges = self._enforce_monotonic(clean, clean_y, edges)

        # Compute WoE per bin
        bins = pd.cut(clean, bins=edges, labels=False, include_lowest=True)
        woe_map = {}
        total_events = clean_y.sum()
        total_non_events = len(clean_y) - total_events
        for b in sorted(bins.dropna().unique()):
            mask = bins == b
            events = clean_y[mask].sum()
            non_events = mask.sum() - events
            dist_event = events / max(total_events, 1)
            dist_non_event = non_events / max(total_non_events, 1)
            eps = 1e-6
            woe_map[int(b)] = float(
                np.log((dist_event + eps) / (dist_non_event + eps))
            )
        return edges, woe_map

    def _merge_small_bins(
        self,
        series: pd.Series,
        y: pd.Series,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Merge bins that are too small."""
        min_count = int(self.min_bin_size * len(series))
        while len(edges) > 2:
            bins = pd.cut(series, bins=edges, labels=False, include_lowest=True)
            counts = bins.value_counts()
            small = counts[counts < min_count]
            if len(small) == 0:
                break
            # Merge the smallest bin with its neighbour
            smallest_bin = small.idxmin()
            edge_idx = int(smallest_bin) + 1
            if edge_idx < len(edges) - 1:
                edges = np.delete(edges, edge_idx)
            elif edge_idx - 1 > 0:
                edges = np.delete(edges, edge_idx - 1)
            else:
                break
        return edges

    def _enforce_monotonic(
        self,
        series: pd.Series,
        y: pd.Series,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Merge bins until event rate is monotonic."""
        max_iter = 50
        for _ in range(max_iter):
            bins = pd.cut(series, bins=edges, labels=False, include_lowest=True)
            rates = y.groupby(bins).mean().sort_index()
            if len(rates) <= 2:
                break
            diffs = rates.diff().dropna()
            if (diffs >= 0).all() or (diffs <= 0).all():
                break
            # Find first violation and merge
            violations = diffs[diffs * diffs.iloc[0] < 0]
            if len(violations) == 0:
                break
            merge_idx = int(violations.index[0]) + 1
            if 0 < merge_idx < len(edges) - 1:
                edges = np.delete(edges, merge_idx)
            else:
                break
        return edges

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for col, edges in self._bin_edges.items():
            if col not in Xt.columns:
                continue
            bins = pd.cut(Xt[col], bins=edges, labels=False, include_lowest=True)
            woe_map = self._bin_woe.get(col, {})
            Xt[col] = bins.map(woe_map).fillna(0.0).astype(float)
        return Xt


# ---------------------------------------------------------------------------
# WoEFiller (smoothed WoE for categorical features)
# ---------------------------------------------------------------------------

class WoEFiller(BaseStep):
    """Weight of Evidence encoding with Laplace smoothing.

    Parameters
    ----------
    cols : list[str] | None
        Columns to encode (defaults to object/category columns).
    smoothing : float
        Additive smoothing parameter.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        smoothing: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self.smoothing = smoothing
        self._woe_maps: Dict[str, Dict[Any, float]] = {}
        self._default_woe: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "WoEFiller":
        if y is None:
            raise ValueError("WoEFiller requires y.")
        y = y.astype(int)
        cols = self.cols or list(
            X.select_dtypes(include=["object", "category"]).columns
        )
        total_events = y.sum()
        total_non_events = len(y) - total_events

        for col in cols:
            woe_map: Dict[Any, float] = {}
            for cat, group_y in y.groupby(X[col]):
                events = group_y.sum() + self.smoothing
                non_events = len(group_y) - group_y.sum() + self.smoothing
                dist_e = events / (total_events + self.smoothing * 2)
                dist_ne = non_events / (total_non_events + self.smoothing * 2)
                woe_map[cat] = float(np.log(dist_e / dist_ne))
            self._woe_maps[col] = woe_map
            self._default_woe[col] = 0.0  # fallback for unseen categories
        self._fitted = True
        logger.info("WoEFiller: encoded %d columns.", len(cols))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for col, woe_map in self._woe_maps.items():
            if col in Xt.columns:
                Xt[col] = (
                    Xt[col]
                    .map(woe_map)
                    .fillna(self._default_woe.get(col, 0.0))
                    .astype(float)
                )
        return Xt


# ---------------------------------------------------------------------------
# SelectStepwise (AIC / BIC / p-value based)
# ---------------------------------------------------------------------------

class SelectStepwise(BaseStep):
    """Forward stepwise feature selection based on AIC, BIC, or p-values.

    Parameters
    ----------
    criterion : str
        ``"aic"``, ``"bic"``, or ``"pvalue"``.
    p_threshold : float
        p-value threshold when ``criterion="pvalue"``.
    max_features : int | None
        Maximum features to select.
    """

    def __init__(
        self,
        criterion: str = "aic",
        p_threshold: float = 0.05,
        max_features: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.criterion = criterion.lower()
        self.p_threshold = p_threshold
        self.max_features = max_features
        self.selected_features_: List[str] = []
        self.selection_history_: List[Dict[str, Any]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "SelectStepwise":
        if y is None:
            raise ValueError("SelectStepwise requires y.")
        import statsmodels.api as sm

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[num_cols].fillna(0)

        selected: List[str] = []
        remaining = list(num_cols)

        best_score = np.inf
        max_feat = self.max_features or len(num_cols)

        for step in range(min(max_feat, len(num_cols))):
            best_candidate = None
            best_candidate_score = np.inf

            for candidate in remaining:
                try:
                    trial = selected + [candidate]
                    Xc = sm.add_constant(X_num[trial])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = sm.Logit(y, Xc).fit(disp=0, maxiter=50)

                    if self.criterion == "aic":
                        score = model.aic
                    elif self.criterion == "bic":
                        score = model.bic
                    else:  # pvalue
                        pval = model.pvalues.iloc[-1]
                        score = pval

                    if score < best_candidate_score:
                        best_candidate_score = score
                        best_candidate = candidate
                except Exception:
                    continue

            if best_candidate is None:
                break

            if self.criterion in ("aic", "bic"):
                if best_candidate_score < best_score:
                    best_score = best_candidate_score
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                    self.selection_history_.append({
                        "step": step,
                        "feature": best_candidate,
                        "score": best_candidate_score,
                    })
                else:
                    break
            else:  # pvalue
                if best_candidate_score < self.p_threshold:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                    self.selection_history_.append({
                        "step": step,
                        "feature": best_candidate,
                        "pvalue": best_candidate_score,
                    })
                else:
                    break

        self.selected_features_ = selected
        self._fitted = True
        logger.info(
            "SelectStepwise (%s): selected %d features.",
            self.criterion, len(selected),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        available = [c for c in self.selected_features_ if c in X.columns]
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return X[available + non_numeric].copy()


# ---------------------------------------------------------------------------
# DataScaler
# ---------------------------------------------------------------------------

class DataScaler(BaseStep):
    """Standard scaling wrapper.

    Parameters
    ----------
    cols : list[str] | None
        Columns to scale.  Defaults to all numeric.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self._scaler = StandardScaler()
        self._scale_cols: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "DataScaler":
        self._scale_cols = self.cols or list(
            X.select_dtypes(include=[np.number]).columns
        )
        self._scaler.fit(X[self._scale_cols].fillna(0))
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        avail = [c for c in self._scale_cols if c in Xt.columns]
        Xt[avail] = self._scaler.transform(Xt[avail].fillna(0))
        return Xt


# ---------------------------------------------------------------------------
# LogregFitter
# ---------------------------------------------------------------------------

class LogregFitter(BaseStep):
    """Logistic regression classifier.

    Parameters
    ----------
    C : float
        Regularisation strength (inverse).
    penalty : str
        ``"l1"``, ``"l2"``, ``"elasticnet"``, or ``"none"``.
    l1_ratio : float | None
        ElasticNet mixing parameter.
    max_iter : int
        Maximum iterations.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        l1_ratio: Optional[float] = None,
        max_iter: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.C = C
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.model_: Optional[LogisticRegression] = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kw: Any,
    ) -> "LogregFitter":
        if y is None:
            raise ValueError("LogregFitter requires y.")
        self._feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[self._feature_names].fillna(0)

        solver = "saga" if self.penalty in ("l1", "elasticnet") else "lbfgs"
        params: Dict[str, Any] = dict(
            C=self.C,
            penalty=self.penalty if self.penalty != "none" else None,
            solver=solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        if self.penalty == "elasticnet" and self.l1_ratio is not None:
            params["l1_ratio"] = self.l1_ratio

        self.model_ = LogisticRegression(**params)
        self.model_.fit(X_num, y, sample_weight=sample_weight)
        self._fitted = True
        logger.info(
            "LogregFitter: trained with %d features, C=%.4f, penalty=%s.",
            len(self._feature_names), self.C, self.penalty,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(X[self._feature_names].fillna(0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(X[self._feature_names].fillna(0))

    @property
    def coef_(self) -> np.ndarray:
        self._check_fitted()
        return self.model_.coef_

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


# ---------------------------------------------------------------------------
# CheckFeatureCoeffs
# ---------------------------------------------------------------------------

class CheckFeatureCoeffs(BaseStep):
    """Validate that logistic regression coefficients have expected signs.

    Parameters
    ----------
    expected_signs : dict[str, int] | None
        Mapping of feature → expected coefficient sign (+1 or -1).
        If *None*, simply reports the signs.
    strict : bool
        If *True*, raises an error on sign violations.
    """

    def __init__(
        self,
        expected_signs: Optional[Dict[str, int]] = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.expected_signs = expected_signs or {}
        self.strict = strict
        self.violations_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        logreg_fitter: Optional[LogregFitter] = None,
        **kw: Any,
    ) -> "CheckFeatureCoeffs":
        if logreg_fitter is None:
            logger.warning("CheckFeatureCoeffs: no logreg_fitter provided.")
            self._fitted = True
            return self

        coefs = logreg_fitter.coef_.flatten()
        names = logreg_fitter.feature_names
        self.violations_ = []

        for name, coef in zip(names, coefs):
            expected = self.expected_signs.get(name)
            if expected is not None:
                actual_sign = 1 if coef >= 0 else -1
                if actual_sign != expected:
                    self.violations_.append(name)
                    logger.warning(
                        "Coefficient sign violation: %s has coef=%.4f "
                        "(expected sign=%d).",
                        name, coef, expected,
                    )

        if self.violations_ and self.strict:
            raise ValueError(
                f"Coefficient sign violations in: {self.violations_}"
            )
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
