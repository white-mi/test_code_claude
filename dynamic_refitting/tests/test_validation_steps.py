"""Tests for validation steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_refitting.validation_steps import (
    FeatureCleanerConst,
    FeatureCleanerNan,
    FeatureCleanerUnivariate,
    HitrateChecker,
    PopulationStabilityIndex,
)


class TestFeatureCleanerConst:
    def test_drops_constant_columns(self, scoring_df):
        cleaner = FeatureCleanerConst()
        y = scoring_df["target"]
        X = scoring_df.drop(columns=["target"])
        cleaner.fit(X, y)
        Xt = cleaner.transform(X)
        assert "const_col" in cleaner.constant_cols_
        assert "const_col" not in Xt.columns


class TestFeatureCleanerNan:
    def test_fills_nans(self, scoring_df):
        cleaner = FeatureCleanerNan(nan_threshold=0.99, fill_strategy="median")
        y = scoring_df["target"]
        X = scoring_df.drop(columns=["target"])
        cleaner.fit(X, y)
        Xt = cleaner.transform(X)
        # Numeric columns should have no NaNs
        num_nans = Xt.select_dtypes(include=[np.number]).isna().sum().sum()
        assert num_nans == 0


class TestFeatureCleanerUnivariate:
    def test_filters_low_auc(self, X_y):
        X, y = X_y
        cleaner = FeatureCleanerUnivariate(min_auc=0.50)
        cleaner.fit(X, y)
        Xt = cleaner.transform(X)
        # Should keep some features
        assert len(cleaner.selected_features_) > 0


class TestHitrateChecker:
    def test_rate_in_range(self, X_y):
        X, y = X_y
        checker = HitrateChecker(min_rate=0.01, max_rate=0.5)
        checker.fit(X, y)
        assert 0.01 <= checker.actual_rate_ <= 0.5


class TestPopulationStabilityIndex:
    def test_psi_same_data(self, X_y):
        X, y = X_y
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        psi_step = PopulationStabilityIndex()
        psi_step.fit(X_num)
        psi_step.transform(X_num)
        # PSI on same data should be very low
        for col, psi_val in psi_step.psi_values_.items():
            assert psi_val < 0.1, f"PSI({col})={psi_val} too high for same data"
