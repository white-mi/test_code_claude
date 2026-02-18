"""Synthetic dataset generation for testing and demos."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def make_scoring_dataset(
    n_samples: int = 10_000,
    n_features: int = 30,
    imbalance_ratio: float = 10.0,
    n_informative: int = 10,
    n_time_periods: int = 12,
    random_state: int = 42,
    time_col: str = "date",
    target_col: str = "target",
) -> pd.DataFrame:
    """Generate a synthetic credit-scoring-style dataset.

    Parameters
    ----------
    n_samples : int
        Total number of rows.
    n_features : int
        Total number of feature columns.
    imbalance_ratio : float
        Ratio of negatives to positives (e.g. 10 means ~10:1).
    n_informative : int
        Number of features that actually carry signal.
    n_time_periods : int
        Number of distinct time periods (months).
    random_state : int
        Random seed.
    time_col : str
        Name for the time column.
    target_col : str
        Name for the target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns, ``target_col``, and ``time_col``.
    """
    rng = np.random.RandomState(random_state)

    # -- features --
    X = rng.randn(n_samples, n_features)
    feature_names = [f"feat_{i:03d}" for i in range(n_features)]

    # -- target (logistic with imbalance) --
    beta = np.zeros(n_features)
    informative_idx = rng.choice(n_features, size=min(n_informative, n_features), replace=False)
    beta[informative_idx] = rng.randn(len(informative_idx)) * 0.8

    # Intercept controls imbalance
    pos_rate = 1.0 / (1.0 + imbalance_ratio)
    intercept = -np.log(1.0 / pos_rate - 1.0)  # logit(pos_rate)

    logits = X @ beta + intercept + rng.randn(n_samples) * 0.3
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.rand(n_samples) < prob).astype(int)

    # -- time column --
    periods = pd.date_range("2023-01-01", periods=n_time_periods, freq="MS")
    time_values = rng.choice(periods, size=n_samples)

    # -- add some categorical / noisy features --
    df = pd.DataFrame(X, columns=feature_names)

    # Add a few categorical features
    n_cat = min(5, n_features)
    for i in range(n_cat):
        col = f"cat_{i:02d}"
        df[col] = rng.choice(["A", "B", "C", "D", "E"], size=n_samples)

    # Add some NaN to mimic real data
    nan_mask = rng.rand(n_samples, n_features) < 0.02
    df.iloc[:, :n_features] = df.iloc[:, :n_features].mask(
        pd.DataFrame(nan_mask, columns=feature_names)
    )

    # Add a constant column (for cleaner tests)
    df["const_col"] = 1.0

    df[target_col] = y
    df[time_col] = time_values

    return df
