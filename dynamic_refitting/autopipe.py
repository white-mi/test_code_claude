"""AutoPipe — automatic pipeline assembly for boosting and logistic regression."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep, PipelineConfig, ScoringPipeline
from dynamic_refitting.utils.metrics import calc_metrics
from dynamic_refitting.utils.time_split import TimeSeriesSplitter
from dynamic_refitting.validation_steps import (
    FeatureCleanerConst,
    FeatureCleanerNan,
)
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

logger = logging.getLogger("dynamic_refitting.autopipe")


class _AutoPipeBase:
    """Common logic for auto-assembled pipelines."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        custom_steps: Optional[Sequence[BaseStep]] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.custom_steps = list(custom_steps) if custom_steps else []
        self.pipeline_: Optional[ScoringPipeline] = None
        self.cv_results_: List[Dict[str, Any]] = []
        self.train_metrics_: Dict[str, Any] = {}

    def _build_pipeline(self, steps: List[BaseStep]) -> ScoringPipeline:
        all_steps = steps[:]
        # Insert custom steps before the estimator
        if self.custom_steps:
            all_steps = all_steps[:-1] + self.custom_steps + [all_steps[-1]]
        return ScoringPipeline(steps=all_steps, config=self.config)

    def _run_cv(
        self,
        pipeline: ScoringPipeline,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> List[Dict[str, Any]]:
        """Run time-based or standard cross-validation."""
        results: List[Dict[str, Any]] = []

        if self.config.time_col and self.config.time_col in X.columns:
            splitter = TimeSeriesSplitter(
                n_splits=n_splits, time_col=self.config.time_col
            )
            splits = list(splitter.split(X, y))
        else:
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.config.random_state
            )
            splits = list(skf.split(X, y))

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            # Clone pipeline for this fold (create fresh instances)
            fold_pipe = self._build_pipeline(self._default_steps())
            fold_pipe.fit(X_tr, y_tr)
            proba = fold_pipe.predict_proba(X_val)
            scores = proba[:, 1] if proba.ndim == 2 else proba

            metrics = calc_metrics(
                y_val.values, scores, metric_names=self.config.metrics
            )
            metrics["fold"] = fold_idx
            results.append(metrics)
            logger.info("Fold %d: %s", fold_idx, metrics)

        return results

    def _default_steps(self) -> List[BaseStep]:
        raise NotImplementedError

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        run_cv: bool = True,
        n_splits: int = 5,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "_AutoPipeBase":
        """Fit the auto-assembled pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Binary target.
        run_cv : bool
            Whether to run cross-validation before final fit.
        n_splits : int
            Number of CV folds.
        sample_weight : array-like | None
            Optional sample weights.

        Returns
        -------
        self
        """
        logger.info("AutoPipe: starting fit on %d rows.", len(X))

        if run_cv:
            self.cv_results_ = self._run_cv(
                self._build_pipeline(self._default_steps()), X, y, n_splits
            )
            logger.info(
                "AutoPipe: CV complete. Mean AUC=%.4f",
                np.mean([r.get("auc", 0) for r in self.cv_results_]),
            )

        # Final fit on all data
        self.pipeline_ = self._build_pipeline(self._default_steps())
        self.pipeline_.fit(X, y, sample_weight=sample_weight)

        # Compute training metrics
        proba = self.pipeline_.predict_proba(X)
        scores = proba[:, 1] if proba.ndim == 2 else proba
        self.train_metrics_ = calc_metrics(
            y.values, scores, metric_names=self.config.metrics
        )
        logger.info("AutoPipe: train metrics = %s", self.train_metrics_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Pipeline not fitted.")
        return self.pipeline_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("Pipeline not fitted.")
        return self.pipeline_.predict_proba(X)

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("Pipeline not fitted.")
        self.pipeline_.save(path)

    @classmethod
    def load(cls, path: str) -> ScoringPipeline:
        return ScoringPipeline.load(path)


class AutoPipeBoost(_AutoPipeBase):
    """Auto-assembled boosting (LightGBM) pipeline.

    Default steps:
    1. FeatureCleanerConst
    2. FeatureCleanerNan
    3. FeaturePreSelector
    4. ClearCorrelatedFeatures
    5. ClearTailFeatures
    6. OptunaBoostingFitter

    Parameters
    ----------
    config : PipelineConfig | None
    custom_steps : list[BaseStep] | None
        Inserted before the estimator.
    optuna_n_trials : int
        Number of Optuna trials.
    optuna_timeout : int | None
        Optuna timeout.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        custom_steps: Optional[Sequence[BaseStep]] = None,
        optuna_n_trials: int = 30,
        optuna_timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, custom_steps=custom_steps)
        self.optuna_n_trials = optuna_n_trials
        self.optuna_timeout = optuna_timeout

    def _default_steps(self) -> List[BaseStep]:
        cfg = self.config
        return [
            FeatureCleanerConst(random_state=cfg.random_state),
            FeatureCleanerNan(random_state=cfg.random_state),
            FeaturePreSelector(random_state=cfg.random_state),
            ClearCorrelatedFeatures(random_state=cfg.random_state),
            ClearTailFeatures(random_state=cfg.random_state),
            OptunaBoostingFitter(
                n_trials=self.optuna_n_trials,
                timeout=self.optuna_timeout,
                time_col=cfg.time_col,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
            ),
        ]


class AutoPipeLogreg(_AutoPipeBase):
    """Auto-assembled logistic regression pipeline.

    Default steps:
    1. FeatureCleanerConst
    2. FeatureCleanerNan
    3. WoEFiller (categorical)
    4. FeatureLinearizer (numeric)
    5. DataScaler
    6. LogregFitter

    Parameters
    ----------
    config : PipelineConfig | None
    custom_steps : list[BaseStep] | None
    C : float
        Logistic regression regularisation.
    penalty : str
        Penalty type.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        custom_steps: Optional[Sequence[BaseStep]] = None,
        C: float = 1.0,
        penalty: str = "l2",
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, custom_steps=custom_steps)
        self.C = C
        self.penalty = penalty

    def _default_steps(self) -> List[BaseStep]:
        cfg = self.config
        return [
            FeatureCleanerConst(random_state=cfg.random_state),
            FeatureCleanerNan(random_state=cfg.random_state),
            WoEFiller(random_state=cfg.random_state),
            FeatureLinearizer(random_state=cfg.random_state),
            DataScaler(random_state=cfg.random_state),
            LogregFitter(
                C=self.C,
                penalty=self.penalty,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
            ),
        ]
