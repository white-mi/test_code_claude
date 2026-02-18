"""Feature generation steps: aggregates, lags, rolling stats, datetime."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.feature_engineering")


class GroupAggGenerator(BaseStep):
    """Generate group-level aggregate features.

    Parameters
    ----------
    group_cols : list[str]
        Columns to group by.
    agg_cols : list[str]
        Columns to aggregate.
    agg_funcs : list[str]
        Aggregation functions (e.g. ``["mean", "std", "max"]``).
    """

    def __init__(
        self,
        group_cols: List[str] | None = None,
        agg_cols: List[str] | None = None,
        agg_funcs: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.group_cols = group_cols or []
        self.agg_cols = agg_cols or []
        self.agg_funcs = agg_funcs or ["mean", "std"]
        self._agg_map: Dict[str, pd.DataFrame] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "GroupAggGenerator":
        if not self.group_cols or not self.agg_cols:
            self._fitted = True
            return self
        for func in self.agg_funcs:
            self._agg_map[func] = X.groupby(self.group_cols)[self.agg_cols].agg(func)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        if not self.group_cols or not self.agg_cols:
            return X
        Xt = X.copy()
        for func, agg_df in self._agg_map.items():
            suffix = f"_grp_{func}"
            merged = agg_df.add_suffix(suffix)
            Xt = Xt.merge(merged, left_on=self.group_cols, right_index=True, how="left")
        return Xt


class LagFeatureGenerator(BaseStep):
    """Create lag features from numeric columns.

    Parameters
    ----------
    lag_cols : list[str]
        Columns to create lags for.
    lags : list[int]
        Lag periods (e.g. ``[1, 3, 6]``).
    sort_col : str
        Column used to sort before computing lags.
    group_col : str | None
        Optional group column (e.g. customer_id).
    """

    def __init__(
        self,
        lag_cols: List[str] | None = None,
        lags: List[int] | None = None,
        sort_col: str = "date",
        group_col: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lag_cols = lag_cols or []
        self.lags = lags or [1]
        self.sort_col = sort_col
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "LagFeatureGenerator":
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        if not self.lag_cols:
            return X
        Xt = X.sort_values(self.sort_col).copy()
        for col in self.lag_cols:
            for lag in self.lags:
                new_col = f"{col}_lag{lag}"
                if self.group_col:
                    Xt[new_col] = Xt.groupby(self.group_col)[col].shift(lag)
                else:
                    Xt[new_col] = Xt[col].shift(lag)
        return Xt


class RollingStatGenerator(BaseStep):
    """Compute rolling statistics (mean, std, etc.) over a window.

    Parameters
    ----------
    stat_cols : list[str]
        Columns to compute rolling stats for.
    windows : list[int]
        Rolling window sizes.
    funcs : list[str]
        Stats to compute (default ``["mean", "std"]``).
    sort_col : str
        Column to sort by before rolling.
    group_col : str | None
        Optional group column.
    """

    def __init__(
        self,
        stat_cols: List[str] | None = None,
        windows: List[int] | None = None,
        funcs: List[str] | None = None,
        sort_col: str = "date",
        group_col: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.stat_cols = stat_cols or []
        self.windows = windows or [3]
        self.funcs = funcs or ["mean", "std"]
        self.sort_col = sort_col
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "RollingStatGenerator":
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        if not self.stat_cols:
            return X
        Xt = X.sort_values(self.sort_col).copy()
        for col in self.stat_cols:
            for w in self.windows:
                for func in self.funcs:
                    new_col = f"{col}_roll{w}_{func}"
                    if self.group_col:
                        Xt[new_col] = (
                            Xt.groupby(self.group_col)[col]
                            .rolling(w, min_periods=1)
                            .agg(func)
                            .reset_index(level=0, drop=True)
                        )
                    else:
                        Xt[new_col] = Xt[col].rolling(w, min_periods=1).agg(func)
        return Xt


class DatetimeFeatures(BaseStep):
    """Extract date/time components from a datetime column.

    Parameters
    ----------
    datetime_col : str
        Column to extract features from.
    features : list[str]
        Which components to extract (``"month"``, ``"dayofweek"``,
        ``"quarter"``, ``"year"``, ``"dayofyear"``).
    drop_original : bool
        Whether to drop the original datetime column.
    """

    def __init__(
        self,
        datetime_col: str = "date",
        features: List[str] | None = None,
        drop_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.datetime_col = datetime_col
        self.features = features or ["month", "dayofweek", "quarter"]
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kw: Any) -> "DatetimeFeatures":
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        dt = pd.to_datetime(Xt[self.datetime_col])
        for feat in self.features:
            Xt[f"{self.datetime_col}_{feat}"] = getattr(dt.dt, feat)
        if self.drop_original:
            Xt.drop(columns=[self.datetime_col], inplace=True)
        return Xt
