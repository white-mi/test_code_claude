# dynamic_refitting

Production-ready Python library for developing, validating, monitoring, and automatically refitting scoring models (binary classification).

Designed for **credit/risk scoring** with a scikit-learn compatible API (`fit` / `transform` / `predict` / `predict_proba`).

## Key Features

- **AutoPipe** — automatic pipeline assembly for LightGBM and Logistic Regression
- **Boost pipeline** — FeaturePreSelector, Boruta selector, Optuna-tuned LightGBM
- **Logreg pipeline** — monotonic binning, WoE encoding, stepwise selection
- **Validation** — constant/NaN cleaning, univariate filtering, PSI, WoE stability
- **Monitoring** — feature drift (PSI + KS), prediction drift, rolling AUC
- **Explainability** — SHAP, permutation importance, PDP, counterfactuals, model cards
- **Model Registry** — versioning, tagging, promotion, rollback
- **Dynamic Refitting** — performance/PSI/time/data-volume triggers, RefitManager
- **Feature Engineering** — target encoding, frequency encoding, lags, rolling stats, interactions
- **CLI** — `dynamic_refitting train|evaluate|monitor|refit`
- **Plugin architecture** — register custom steps via `StepRegistry`

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Boosting Pipeline

```python
from dynamic_refitting.config import PipelineConfig
from dynamic_refitting.autopipe import AutoPipeBoost
from dynamic_refitting.utils.data_gen import make_scoring_dataset

# Generate synthetic data
df = make_scoring_dataset(n_samples=10_000, imbalance_ratio=10.0)
X = df.drop(columns=["target"])
y = df["target"]

# Configure and train
config = PipelineConfig(
    random_state=42,
    target_col="target",
    time_col="date",
    metrics=["auc", "ks", "brier"],
)
pipe = AutoPipeBoost(config=config, optuna_n_trials=20)
pipe.fit(X, y, run_cv=True, n_splits=5)

# Predict
proba = pipe.predict_proba(X)[:, 1]
print(f"Train metrics: {pipe.train_metrics_}")

# Save
pipe.save("model.joblib")
```

### Logistic Regression Pipeline

```python
from dynamic_refitting.autopipe import AutoPipeLogreg

pipe = AutoPipeLogreg(config=config, C=1.0, penalty="l2")
pipe.fit(X, y, run_cv=True)
proba = pipe.predict_proba(X)[:, 1]
```

### Custom Pipeline

```python
from dynamic_refitting.config import ScoringPipeline
from dynamic_refitting.validation_steps import FeatureCleanerConst, FeatureCleanerNan
from dynamic_refitting.boost_pipeline_steps import (
    FeaturePreSelector,
    ClearCorrelatedFeatures,
    FeatureSelector,
    OptunaBoostingFitter,
)

steps = [
    FeatureCleanerConst(),
    FeatureCleanerNan(fill_strategy="median"),
    FeaturePreSelector(nan_threshold=0.8, corr_threshold=0.01),
    ClearCorrelatedFeatures(threshold=0.95),
    FeatureSelector(n_iterations=10),  # Boruta
    OptunaBoostingFitter(n_trials=50, time_col="date"),
]
pipeline = ScoringPipeline(steps=steps)
pipeline.fit(X, y)
```

### Monitoring

```python
from dynamic_refitting.monitoring import (
    FeatureDriftDetector,
    PredictionDriftMonitor,
    ModelPerformanceMonitor,
    DriftReportGenerator,
)

# Feature drift
detector = FeatureDriftDetector(psi_threshold=0.2)
detector.fit(X_train)
detector.transform(X_production)
print(f"Drifted features: {detector.drifted_features}")

# Performance tracking
monitor = ModelPerformanceMonitor(auc_threshold=0.65)
results = monitor.evaluate(y_test, proba_test, time_values=dates_test)

# Drift report
reporter = DriftReportGenerator(output_dir="reports/")
report = reporter.generate(
    feature_drift=detector,
    performance_monitor=monitor,
)
```

### Dynamic Refitting

```python
from dynamic_refitting.refit import (
    RefitManager,
    PerformanceTriggeredRefit,
    TimeBasedRefit,
    DataVolumeTriggeredRefit,
)

triggers = [
    PerformanceTriggeredRefit(auc_threshold=0.65, psi_threshold=0.25),
    TimeBasedRefit(interval_days=30),
    DataVolumeTriggeredRefit(min_new_samples=5000),
]
manager = RefitManager(pipeline=pipeline, triggers=triggers)
manager.set_reference_scores(train_proba)

# Automatic check-and-refit
result = manager.auto_refit(
    X_monitor=X_recent, y_monitor=y_recent,
    X_train=X_full, y_train=y_full,
)
if result is not None:
    print("Model was refitted!")
```

### SHAP Explainability

```python
from dynamic_refitting.explainability import ShapExplainer

explainer = ShapExplainer(model_type="tree")
explainer.fit(X_train, model=lgb_model)
shap_values = explainer.explain(X_test)
print(f"Top features: {explainer.get_top_features(10)}")
```

### Model Registry

```python
from dynamic_refitting.registry import ModelRegistry

registry = ModelRegistry(root_dir=".model_registry")
mv = registry.register(
    "scoring_v2", pipeline,
    metrics={"auc": 0.82, "ks": 0.45},
    tags={"team": "risk"},
)
registry.promote("scoring_v2", version=mv.version)

# Load production model
prod_model = registry.load_model("scoring_v2")
```

### Plugin Architecture

```python
from dynamic_refitting.registry import StepRegistry
from dynamic_refitting.config import BaseStep

class MyCustomStep(BaseStep):
    def fit(self, X, y=None, **kw):
        self._fitted = True
        return self
    def transform(self, X):
        return X

StepRegistry.register("custom", "my_step", MyCustomStep)
cls = StepRegistry.get("custom", "my_step")
```

## CLI

```bash
# Train
dynamic_refitting train --config config.yaml --data train.csv --output model.joblib

# Evaluate
dynamic_refitting evaluate --model model.joblib --data test.csv

# Monitor drift
dynamic_refitting monitor --model model.joblib --train-data train.csv --data prod.csv

# Refit
dynamic_refitting refit --model model.joblib --data new_data.csv --output model_v2.joblib
```

## Package Structure

```
dynamic_refitting/
├── __init__.py
├── autopipe.py              # AutoPipeBoost, AutoPipeLogreg
├── boost_pipeline_steps.py  # Boruta, Optuna LightGBM, correlation cleaning
├── logreg_pipeline_steps.py # WoE, binning, stepwise, logistic regression
├── validation_steps.py      # Data quality, PSI, WoE stability
├── config.py                # BaseStep, ScoringPipeline, PipelineConfig
├── cli.py                   # Command-line interface
├── monitoring/
│   ├── drift.py             # FeatureDriftDetector, PredictionDriftMonitor
│   ├── performance.py       # ModelPerformanceMonitor
│   └── report.py            # DriftReportGenerator
├── explainability/
│   ├── shap_explainer.py    # ShapExplainer
│   ├── importance.py        # PermutationImportance
│   ├── pdp.py               # PartialDependence
│   ├── counterfactual.py    # CounterfactualGenerator
│   └── model_card.py        # ModelCardGenerator
├── registry/
│   ├── step_registry.py     # Plugin architecture
│   ├── model_registry.py    # ModelRegistry (versioning, promotion)
│   └── experiment_tracker.py# ExperimentTracker
├── refit/
│   ├── triggers.py          # Performance/Time/DataVolume triggers
│   ├── manager.py           # RefitManager
│   └── scheduler.py         # RefitScheduler
├── feature_engineering/
│   ├── generators.py        # Lags, rolling stats, datetime, group agg
│   ├── encoders.py          # TargetEncoder, FrequencyEncoder
│   └── interactions.py      # InteractionGenerator
├── utils/
│   ├── metrics.py           # AUC, KS, Brier, PSI
│   ├── time_split.py        # TimeSeriesSplitter
│   └── data_gen.py          # Synthetic dataset generator
└── tests/
    ├── conftest.py           # Shared fixtures
    ├── test_pipeline_fit.py
    ├── test_drift_and_refit.py
    ├── test_validation_steps.py
    ├── test_feature_engineering.py
    └── test_registry.py
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```
