"""Refit manager — orchestrates the refit lifecycle."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dynamic_refitting.config import BaseStep, ScoringPipeline, PipelineConfig
from dynamic_refitting.refit.triggers import BaseRefitTrigger
from dynamic_refitting.utils.metrics import calc_metrics

logger = logging.getLogger("dynamic_refitting.refit")


class RefitManager:
    """Orchestrate model monitoring, trigger evaluation, and refitting.

    Parameters
    ----------
    pipeline : ScoringPipeline
        The current production pipeline.
    triggers : list[BaseRefitTrigger]
        Conditions that decide when to refit.
    config : PipelineConfig | None
        Pipeline configuration.
    warm_start : bool
        If *True*, attempt warm-start on the estimator.
    """

    def __init__(
        self,
        pipeline: ScoringPipeline,
        triggers: Sequence[BaseRefitTrigger],
        config: Optional[PipelineConfig] = None,
        warm_start: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.triggers = list(triggers)
        self.config = config or PipelineConfig()
        self.warm_start = warm_start
        self.refit_history_: List[Dict[str, Any]] = []
        self._reference_scores: Optional[np.ndarray] = None

    def set_reference_scores(self, scores: np.ndarray) -> None:
        """Store the training-time predicted probabilities for PSI."""
        self._reference_scores = np.asarray(scores)

    def check_triggers(
        self,
        y_true: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
    ) -> List[str]:
        """Evaluate all triggers and return reasons for refit.

        Returns
        -------
        list[str]
            Non-empty if at least one trigger fired.
        """
        reasons: List[str] = []
        for trigger in self.triggers:
            fired = trigger.should_refit(
                y_true=y_true,
                y_score=y_score,
                reference_scores=self._reference_scores,
            )
            if fired:
                reasons.append(trigger.get_reason())
        return reasons

    def refit(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        **fit_kwargs: Any,
    ) -> ScoringPipeline:
        """Refit the pipeline on new data.

        Parameters
        ----------
        X_new : pd.DataFrame
            New training features.
        y_new : pd.Series
            New training labels.

        Returns
        -------
        ScoringPipeline
            The refitted pipeline (same object, mutated in-place).
        """
        logger.info("RefitManager: starting refit on %d samples.", len(X_new))
        start = time.time()

        if self.warm_start:
            # Attempt warm start on the estimator if supported
            estimator = self.pipeline.estimator
            if hasattr(estimator, "model_") and hasattr(estimator.model_, "set_params"):
                try:
                    estimator.model_.set_params(warm_start=True)
                    logger.info("Warm-start enabled on estimator.")
                except Exception:
                    logger.debug("Warm-start not supported by estimator.")

        self.pipeline.fit(X_new, y_new, **fit_kwargs)
        elapsed = time.time() - start

        # Compute post-refit metrics
        try:
            proba = self.pipeline.predict_proba(X_new)
            scores = proba[:, 1] if proba.ndim == 2 else proba
            metrics = calc_metrics(y_new, scores, metric_names=self.config.metrics)
            self._reference_scores = scores
        except Exception:
            metrics = {}

        record = {
            "timestamp": time.time(),
            "n_samples": len(X_new),
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "reasons": [t.get_reason() for t in self.triggers if t.get_reason()],
        }
        self.refit_history_.append(record)
        logger.info(
            "RefitManager: refit complete. Metrics: %s", metrics
        )

        # Reset time/volume triggers
        for trigger in self.triggers:
            if hasattr(trigger, "reset"):
                trigger.reset()

        return self.pipeline

    def auto_refit(
        self,
        X_monitor: pd.DataFrame,
        y_monitor: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **fit_kwargs: Any,
    ) -> Optional[ScoringPipeline]:
        """Check triggers on monitoring data; refit on training data if needed.

        Parameters
        ----------
        X_monitor : pd.DataFrame
            Recent production data with labels.
        y_monitor : pd.Series
            Labels for monitoring data.
        X_train : pd.DataFrame
            Full training data for refitting.
        y_train : pd.Series
            Training labels.

        Returns
        -------
        ScoringPipeline | None
            Refitted pipeline, or None if no trigger fired.
        """
        proba = self.pipeline.predict_proba(X_monitor)
        scores = proba[:, 1] if proba.ndim == 2 else proba

        reasons = self.check_triggers(y_true=y_monitor.values, y_score=scores)
        if reasons:
            logger.warning(
                "RefitManager: triggers fired — %s. Initiating refit.",
                "; ".join(reasons),
            )
            return self.refit(X_train, y_train, **fit_kwargs)
        logger.info("RefitManager: no triggers fired; model is stable.")
        return None
