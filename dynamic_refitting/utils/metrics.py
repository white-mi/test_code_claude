"""Scoring metrics commonly used in credit/risk modelling."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

logger = logging.getLogger("dynamic_refitting.metrics")


def calc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Area Under ROC Curve."""
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def calc_ks(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Kolmogorov-Smirnov statistic.

    Measures the maximum separation between the CDF of scores for
    positive and negative classes.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    all_values = np.sort(np.unique(np.concatenate([pos, neg])))
    ks_vals = []
    for thr in all_values:
        tpr = np.mean(pos <= thr)
        fpr = np.mean(neg <= thr)
        ks_vals.append(abs(tpr - fpr))
    return float(np.max(ks_vals))


def calc_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Brier score (lower is better)."""
    return float(
        brier_score_loss(y_true, y_prob, sample_weight=sample_weight)
    )


def calc_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index.

    Parameters
    ----------
    expected : array-like
        Reference (training) distribution.
    actual : array-like
        New (production/holdout) distribution.
    n_bins : int
        Number of equal-frequency bins.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        PSI value.  <0.1 stable, 0.1-0.25 moderate, >0.25 significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Build bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def calc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_names: Optional[List[str]] = None,
    reference_scores: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute a dictionary of metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities for the positive class.
    metric_names : list[str] | None
        Which metrics to compute.  Default: ``["auc", "ks", "brier"]``.
    reference_scores : array-like | None
        Reference scores for PSI computation (typically training scores).
    sample_weight : array-like | None
        Optional sample weights.

    Returns
    -------
    dict[str, Any]
    """
    if metric_names is None:
        metric_names = ["auc", "ks", "brier"]

    results: Dict[str, Any] = {}
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    for m in metric_names:
        m_lower = m.lower()
        if m_lower == "auc":
            results["auc"] = calc_auc(y_true, y_score, sample_weight)
        elif m_lower == "ks":
            results["ks"] = calc_ks(y_true, y_score)
        elif m_lower == "brier":
            results["brier"] = calc_brier(y_true, y_score, sample_weight)
        elif m_lower == "psi":
            if reference_scores is not None:
                results["psi"] = calc_psi(reference_scores, y_score)
            else:
                logger.warning(
                    "PSI requested but no reference_scores provided; skipping."
                )
        elif m_lower == "confusion_matrix":
            preds = (y_score >= 0.5).astype(int)
            results["confusion_matrix"] = confusion_matrix(
                y_true, preds, sample_weight=sample_weight
            ).tolist()
        else:
            logger.warning("Unknown metric '%s' — skipping.", m)

    return results
