"""Tests for feature engineering steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_refitting.feature_engineering.encoders import (
    TargetEncoderCV,
    FrequencyEncoder,
    CategoryEmbedder,
)
from dynamic_refitting.feature_engineering.generators import (
    DatetimeFeatures,
    LagFeatureGenerator,
    RollingStatGenerator,
)
from dynamic_refitting.feature_engineering.interactions import InteractionGenerator


class TestTargetEncoderCV:
    def test_encodes_categoricals(self, X_y):
        X, y = X_y
        encoder = TargetEncoderCV()
        Xt = encoder.fit_transform(X, y)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            assert Xt[col].dtype == float


class TestFrequencyEncoder:
    def test_encodes_frequencies(self, X_y):
        X, y = X_y
        encoder = FrequencyEncoder()
        encoder.fit(X)
        Xt = encoder.transform(X)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            assert Xt[col].dtype == float
            assert Xt[col].max() <= 1.0


class TestCategoryEmbedder:
    def test_ordinal_encoding(self, X_y):
        X, y = X_y
        embedder = CategoryEmbedder()
        embedder.fit(X)
        Xt = embedder.transform(X)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            assert Xt[col].dtype == int


class TestDatetimeFeatures:
    def test_extracts_features(self, X_y):
        X, y = X_y
        dt = DatetimeFeatures(datetime_col="date", features=["month", "dayofweek"])
        dt.fit(X)
        Xt = dt.transform(X)
        assert "date_month" in Xt.columns
        assert "date_dayofweek" in Xt.columns


class TestInteractionGenerator:
    def test_generates_interactions(self, X_y):
        X, y = X_y
        gen = InteractionGenerator(
            cols=["feat_000", "feat_001", "feat_002"],
            interaction_type="multiply",
        )
        gen.fit(X)
        Xt = gen.transform(X)
        assert "feat_000_x_feat_001" in Xt.columns
