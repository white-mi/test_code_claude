"""Interaction feature generator."""

from __future__ import annotations

import itertools
import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.feature_engineering")


class InteractionGenerator(BaseStep):
    """Generate pairwise interaction features (product and ratio).

    Parameters
    ----------
    cols : list[str] | None
        Numeric columns to combine. If *None*, uses all numeric columns.
    max_pairs : int
        Maximum number of pairs to generate (limits combinatorial explosion).
    interaction_type : str
        ``"multiply"``, ``"ratio"``, or ``"both"``.
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        max_pairs: int = 50,
        interaction_type: str = "multiply",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cols = cols
        self.max_pairs = max_pairs
        self.interaction_type = interaction_type
        self._pairs: List[tuple] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "InteractionGenerator":
        cols = self.cols or list(X.select_dtypes(include=[np.number]).columns)
        all_pairs = list(itertools.combinations(cols, 2))
        self._pairs = all_pairs[: self.max_pairs]
        self._fitted = True
        logger.info(
            "InteractionGenerator: %d pairs from %d columns.",
            len(self._pairs),
            len(cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for c1, c2 in self._pairs:
            if c1 not in Xt.columns or c2 not in Xt.columns:
                continue
            if self.interaction_type in ("multiply", "both"):
                Xt[f"{c1}_x_{c2}"] = Xt[c1] * Xt[c2]
            if self.interaction_type in ("ratio", "both"):
                denom = Xt[c2].replace(0, np.nan)
                Xt[f"{c1}_div_{c2}"] = Xt[c1] / denom
        return Xt
