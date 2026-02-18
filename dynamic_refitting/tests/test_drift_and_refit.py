"""Tests for drift detection and refit triggers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_refitting.monitoring.drift import (
    FeatureDriftDetector,
    PredictionDriftMonitor,
)
from dynamic_refitting.monitoring.performance import ModelPerformanceMonitor
from dynamic_refitting.monitoring.report import DriftReportGenerator
from dynamic_refitting.refit.triggers import (
    PerformanceTriggeredRefit,
    TimeBasedRefit,
    DataVolumeTriggeredRefit,
)
from dynamic_refitting.refit.manager import RefitManager
from dynamic_refitting.config import ScoringPipeline, PipelineConfig
from dynamic_refitting.validation_steps import FeatureCleanerConst, FeatureCleanerNan
from dynamic_refitting.boost_pipeline_steps import (
    FeaturePreSelector,
    OptunaBoostingFitter,
)
from dynamic_refitting.utils.metrics import calc_psi


class TestFeatureDriftDetector:
    def test_no_drift_same_data(self, X_y):
        X, y = X_y
        X_num = X.select_dtypes(include=[np.number])
        detector = FeatureDriftDetector(psi_threshold=0.5)
        detector.fit(X_num)
        detector.transform(X_num)
        # Same data should not show drift
        assert len(detector.drifted_features) == 0

    def test_detects_drift(self, X_y):
        X, y = X_y
        X_num = X.select_dtypes(include=[np.number]).copy()
        detector = FeatureDriftDetector(psi_threshold=0.1)
        detector.fit(X_num)

        # Introduce a strong shift
        X_shifted = X_num.copy()
        for col in X_shifted.columns[:5]:
            X_shifted[col] = X_shifted[col] + 10.0
        detector.transform(X_shifted)
        assert len(detector.drifted_features) > 0


class TestPredictionDriftMonitor:
    def test_no_drift_same_scores(self):
        rng = np.random.RandomState(42)
        ref_scores = rng.rand(1000)
        monitor = PredictionDriftMonitor(psi_threshold=0.2)
        monitor.fit(pd.DataFrame(), reference_scores=ref_scores)
        result = monitor.check(ref_scores)
        assert not result["drifted"]
        assert result["psi"] < 0.1

    def test_detects_drift(self):
        rng = np.random.RandomState(42)
        ref_scores = rng.beta(2, 5, size=1000)
        shifted_scores = rng.beta(5, 2, size=1000)
        monitor = PredictionDriftMonitor(psi_threshold=0.2)
        monitor.fit(pd.DataFrame(), reference_scores=ref_scores)
        result = monitor.check(shifted_scores)
        assert result["drifted"]


class TestPerformanceMonitor:
    def test_evaluate_global(self):
        rng = np.random.RandomState(42)
        y_true = pd.Series(rng.choice([0, 1], size=500, p=[0.9, 0.1]))
        y_score = rng.rand(500)
        monitor = ModelPerformanceMonitor()
        results = monitor.evaluate(y_true, y_score)
        assert len(results) == 1
        assert "auc" in results[0]


class TestRefitTriggers:
    def test_performance_trigger_fires(self):
        trigger = PerformanceTriggeredRefit(auc_threshold=0.9)
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 900 + [1] * 100)
        y_score = rng.rand(1000)  # random scores → bad AUC
        assert trigger.should_refit(y_true=y_true, y_score=y_score)
        assert "AUC" in trigger.get_reason()

    def test_performance_trigger_does_not_fire(self):
        trigger = PerformanceTriggeredRefit(auc_threshold=0.5)
        y_true = np.array([0] * 500 + [1] * 500)
        y_score = np.concatenate([
            np.random.RandomState(42).rand(500) * 0.3,
            np.random.RandomState(42).rand(500) * 0.7 + 0.3,
        ])
        assert not trigger.should_refit(y_true=y_true, y_score=y_score)

    def test_data_volume_trigger(self):
        trigger = DataVolumeTriggeredRefit(min_new_samples=500)
        assert not trigger.should_refit()
        trigger.add_samples(300)
        assert not trigger.should_refit()
        trigger.add_samples(250)
        assert trigger.should_refit()
        trigger.reset()
        assert not trigger.should_refit()


class TestRefitManager:
    def test_refit_on_trigger(self, X_y):
        X, y = X_y
        steps = [
            FeatureCleanerConst(),
            FeatureCleanerNan(),
            FeaturePreSelector(),
            OptunaBoostingFitter(n_trials=2, n_estimators_range=(30, 80)),
        ]
        pipe = ScoringPipeline(steps=steps)
        pipe.fit(X, y)

        # Use a very high AUC threshold so trigger fires
        trigger = PerformanceTriggeredRefit(auc_threshold=0.999)
        manager = RefitManager(pipeline=pipe, triggers=[trigger])
        manager.set_reference_scores(pipe.predict_proba(X)[:, 1])

        result = manager.auto_refit(
            X_monitor=X.iloc[:500],
            y_monitor=y.iloc[:500],
            X_train=X,
            y_train=y,
        )
        assert result is not None  # refit happened
        assert len(manager.refit_history_) == 1


class TestDriftReport:
    def test_generate_report(self, X_y):
        X, y = X_y
        X_num = X.select_dtypes(include=[np.number])
        detector = FeatureDriftDetector()
        detector.fit(X_num)
        detector.transform(X_num)

        reporter = DriftReportGenerator()
        report = reporter.generate(feature_drift=detector)
        assert "feature_drift" in report
        assert "summary" in report
