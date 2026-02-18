"""Shared fixtures for dynamic_refitting tests."""

from __future__ import annotations

import pytest
import pandas as pd

from dynamic_refitting.utils.data_gen import make_scoring_dataset
from dynamic_refitting.config import PipelineConfig


@pytest.fixture(scope="session")
def scoring_df() -> pd.DataFrame:
    """Synthetic scoring dataset: 10 000 rows, 30 numeric + 5 cat features,
    imbalance ~10:1, 12 monthly time periods."""
    return make_scoring_dataset(
        n_samples=10_000,
        n_features=30,
        imbalance_ratio=10.0,
        n_informative=10,
        n_time_periods=12,
        random_state=42,
    )


@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        random_state=42,
        n_jobs=1,
        target_col="target",
        time_col="date",
        metrics=["auc", "ks", "brier"],
    )


@pytest.fixture(scope="session")
def X_y(scoring_df: pd.DataFrame):
    """Split scoring_df into X and y."""
    y = scoring_df["target"]
    X = scoring_df.drop(columns=["target"])
    return X, y
