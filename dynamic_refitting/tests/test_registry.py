"""Tests for model registry and step registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from dynamic_refitting.config import BaseStep
from dynamic_refitting.registry.step_registry import StepRegistry
from dynamic_refitting.registry.model_registry import ModelRegistry


class DummyStep(BaseStep):
    def fit(self, X, y=None, **kw):
        self._fitted = True
        return self

    def transform(self, X):
        return X


class TestStepRegistry:
    def setup_method(self):
        StepRegistry.clear()

    def test_register_and_get(self):
        StepRegistry.register("test", "dummy", DummyStep)
        cls = StepRegistry.get("test", "dummy")
        assert cls is DummyStep

    def test_duplicate_raises(self):
        StepRegistry.register("test", "dummy", DummyStep)
        with pytest.raises(KeyError):
            StepRegistry.register("test", "dummy", DummyStep)

    def test_overwrite(self):
        StepRegistry.register("test", "dummy", DummyStep)
        StepRegistry.register("test", "dummy", DummyStep, overwrite=True)

    def test_list(self):
        StepRegistry.register("cat1", "a", DummyStep)
        StepRegistry.register("cat1", "b", DummyStep)
        assert set(StepRegistry.list_steps("cat1")) == {"a", "b"}


class TestModelRegistry:
    def test_register_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(root_dir=tmpdir)
            reg.register("test_model", {"key": "value"}, metrics={"auc": 0.85})
            versions = reg.list_versions("test_model")
            assert len(versions) == 1
            assert versions[0].metrics["auc"] == 0.85

    def test_promote_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(root_dir=tmpdir)
            reg.register("m", "v1_obj")
            reg.register("m", "v2_obj")
            # Promote v1 first, then v2 (v1 becomes archived)
            reg.promote("m", 1)
            reg.promote("m", 2)
            prod = reg.get_production("m")
            assert prod.version == 2

            reg.rollback("m")
            prod = reg.get_production("m")
            assert prod is not None
            assert prod.version == 1

    def test_load_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(root_dir=tmpdir)
            reg.register("m", [1, 2, 3])
            reg.promote("m", 1)
            obj = reg.load_model("m")
            assert obj == [1, 2, 3]
