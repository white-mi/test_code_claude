"""Time-based cross-validation splitter for temporal scoring data."""

from __future__ import annotations

import logging
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("dynamic_refitting.time_split")


class TimeSeriesSplitter:
    """Walk-forward (expanding or sliding window) cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of train/validation folds.
    time_col : str
        Column containing the time information (must be sortable).
    gap : int
        Number of periods to skip between train and validation (avoids leakage).
    expanding : bool
        If *True*, the training window expands; if *False*, uses a fixed
        sliding window whose length equals the first fold's training size.
    """

    def __init__(
        self,
        n_splits: int = 5,
        time_col: str = "date",
        gap: int = 0,
        expanding: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.time_col = time_col
        self.gap = gap
        self.expanding = expanding

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_idx, val_idx) tuples.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``self.time_col``.
        y : ignored

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Train and validation integer-position indices.
        """
        if self.time_col not in X.columns:
            raise ValueError(
                f"Column '{self.time_col}' not found in DataFrame."
            )

        sorted_idx = X[self.time_col].sort_values().index
        unique_times = X[self.time_col].sort_values().unique()
        n_times = len(unique_times)

        if n_times < self.n_splits + 1:
            raise ValueError(
                f"Not enough unique time periods ({n_times}) for "
                f"{self.n_splits} splits."
            )

        # Determine fold boundaries
        fold_size = n_times // (self.n_splits + 1)
        if fold_size < 1:
            fold_size = 1

        for i in range(self.n_splits):
            val_start = (i + 1) * fold_size
            val_end = min(val_start + fold_size, n_times)

            if self.expanding:
                train_end = val_start - self.gap
                train_start = 0
            else:
                train_end = val_start - self.gap
                window = fold_size * 1  # same size as first fold
                train_start = max(0, train_end - window)

            if train_end <= train_start or val_start >= n_times:
                continue

            train_times = set(unique_times[train_start:train_end])
            val_times = set(unique_times[val_start:val_end])

            train_mask = X[self.time_col].isin(train_times)
            val_mask = X[self.time_col].isin(val_times)

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits
