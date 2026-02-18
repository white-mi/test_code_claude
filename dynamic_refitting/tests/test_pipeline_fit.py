"""Tests for basic pipeline fitting and prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dynamic_refitting.config import BaseStep, PipelineConfig, ScoringPipeline
from dynamic_refitting.validation_steps import FeatureCleanerConst, FeatureCleanerNan
from dynamic_refitting.boost_pipeline_steps import (
    FeaturePreSelector,
    ClearCorrelatedFeatures,
    ClearTailFeatures,
    OptunaBoostingFitter,
)
from dynamic_refitting.logreg_pipeline_steps import (
    FeatureLinearizer,
    WoEFiller,
    DataScaler,
    LogregFitter,
)


class TestBoostPipeline:
    """Test the boosting pipeline end-to-end."""

    def test_fit_predict(self, X_y):
        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            FeaturePreSelector(),
            ClearCorrelatedFeatures(),
            ClearTailFeatures(),
            OptunaBoostingFitter(n_trials=3, n_estimators_range=(50, 200)),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        proba = pipe.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2
        assert proba.min() >= 0
        assert proba.max() <= 1

    def test_predict_proba_shape(self, X_y):
        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            FeaturePreSelector(),
            OptunaBoostingFitter(n_trials=2, n_estimators_range=(30, 100)),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        proba = pipe.predict_proba(X.iloc[:100])
        assert proba.shape == (100, 2), f"Expected (100, 2), got {proba.shape}"

    def test_auc_above_threshold(self, X_y):
        """AUC on training data should be at least 0.6."""
        from dynamic_refitting.utils.metrics import calc_auc

        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            FeaturePreSelector(),
            OptunaBoostingFitter(n_trials=5, n_estimators_range=(50, 300)),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        proba = pipe.predict_proba(X)[:, 1]
        auc = calc_auc(y.values, proba)
        assert auc > 0.6, f"AUC={auc} should be > 0.6"


class TestLogregPipeline:
    """Test the logistic regression pipeline."""

    def test_fit_predict(self, X_y):
        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            WoEFiller(),
            FeatureLinearizer(),
            DataScaler(),
            LogregFitter(C=1.0, penalty="l2"),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        preds = pipe.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, X_y):
        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            WoEFiller(),
            DataScaler(),
            LogregFitter(),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        proba = pipe.predict_proba(X.iloc[:50])
        assert proba.shape == (50, 2)


class TestSerialization:
    """Test save/load of pipelines."""

    def test_pipeline_save_load(self, X_y):
        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            FeaturePreSelector(),
            OptunaBoostingFitter(n_trials=2, n_estimators_range=(30, 80)),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_pipe.joblib"
            pipe.save(path)
            loaded = ScoringPipeline.load(path)

            orig_proba = pipe.predict_proba(X.iloc[:10])
            loaded_proba = loaded.predict_proba(X.iloc[:10])
            np.testing.assert_array_almost_equal(orig_proba, loaded_proba)

    def test_config_save_load(self):
        config = PipelineConfig(random_state=123, n_jobs=4, target_col="y")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)
            loaded = PipelineConfig.load(path)
            assert loaded.random_state == 123
            assert loaded.n_jobs == 4
            assert loaded.target_col == "y"
