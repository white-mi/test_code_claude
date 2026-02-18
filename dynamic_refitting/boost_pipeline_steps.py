"""Boosting pipeline steps: feature selection, correlation cleaning, Optuna tuning."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from dynamic_refitting.config import BaseStep
from dynamic_refitting.utils.metrics import calc_auc

logger = logging.getLogger("dynamic_refitting.boost")


# ---------------------------------------------------------------------------
# FeaturePreSelector
# ---------------------------------------------------------------------------

class FeaturePreSelector(BaseStep):
    """Pre-select features based on univariate metrics.

    Drops features with:
    - too many missing values (>= ``nan_threshold``),
    - near-zero variance,
    - low absolute correlation with target.

    Parameters
    ----------
    nan_threshold : float
        Maximum fraction of NaN per column (default 0.95).
    var_threshold : float
        Minimum variance (default 1e-8).
    corr_threshold : float
        Minimum absolute correlation with target (default 0.0 = no filter).
    """

    def __init__(
        self,
        nan_threshold: float = 0.95,
        var_threshold: float = 1e-8,
        corr_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nan_threshold = nan_threshold
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.selected_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeaturePreSelector":
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        selected = []
        for col in numeric_cols:
            nan_frac = X[col].isna().mean()
            if nan_frac >= self.nan_threshold:
                logger.debug("Dropping %s: NaN fraction %.2f", col, nan_frac)
                continue
            var = X[col].var()
            if var < self.var_threshold:
                logger.debug("Dropping %s: variance %.2e", col, var)
                continue
            if y is not None and self.corr_threshold > 0:
                corr = abs(X[col].corr(y))
                if corr < self.corr_threshold:
                    logger.debug("Dropping %s: |corr|=%.4f", col, corr)
                    continue
            selected.append(col)
        self.selected_features_ = selected
        self._fitted = True
        logger.info(
            "FeaturePreSelector: kept %d / %d numeric features.",
            len(selected), len(numeric_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return X[self.selected_features_ + non_numeric].copy()


# ---------------------------------------------------------------------------
# ClearCorrelatedFeatures
# ---------------------------------------------------------------------------

class ClearCorrelatedFeatures(BaseStep):
    """Remove highly correlated features (keep the one with higher target corr).

    Parameters
    ----------
    threshold : float
        Correlation threshold above which one of the pair is dropped.
    """

    def __init__(self, threshold: float = 0.95, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.features_to_drop_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "ClearCorrelatedFeatures":
        num_cols = X.select_dtypes(include=[np.number]).columns
        corr_matrix = X[num_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] >= self.threshold].tolist()
            for hc in high_corr:
                if y is not None:
                    corr_col = abs(X[col].corr(y))
                    corr_hc = abs(X[hc].corr(y))
                    drop = hc if corr_col >= corr_hc else col
                else:
                    drop = hc
                to_drop.add(drop)
        self.features_to_drop_ = list(to_drop)
        self._fitted = True
        logger.info(
            "ClearCorrelatedFeatures: dropping %d features.", len(to_drop)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return X.drop(columns=[c for c in self.features_to_drop_ if c in X.columns]).copy()


# ---------------------------------------------------------------------------
# ClearTailFeatures
# ---------------------------------------------------------------------------

class ClearTailFeatures(BaseStep):
    """Clip or remove extreme outlier values.

    Parameters
    ----------
    lower_quantile : float
        Lower clipping quantile.
    upper_quantile : float
        Upper clipping quantile.
    """

    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self._bounds: Dict[str, Tuple[float, float]] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "ClearTailFeatures":
        for col in X.select_dtypes(include=[np.number]).columns:
            lo = float(X[col].quantile(self.lower_quantile))
            hi = float(X[col].quantile(self.upper_quantile))
            self._bounds[col] = (lo, hi)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        Xt = X.copy()
        for col, (lo, hi) in self._bounds.items():
            if col in Xt.columns:
                Xt[col] = Xt[col].clip(lo, hi)
        return Xt


# ---------------------------------------------------------------------------
# FeatureSelector (Boruta-style)
# ---------------------------------------------------------------------------

class FeatureSelector(BaseStep):
    """Boruta-inspired feature selector using a shadow-feature approach.

    Creates random permutations of every feature ('shadows'), trains a
    Random Forest, and keeps only features whose importance exceeds the
    best shadow feature.

    Parameters
    ----------
    n_iterations : int
        Number of Boruta rounds.
    alpha : float
        Significance level.
    max_features_to_select : int | None
        Hard cap on number of features.
    """

    def __init__(
        self,
        n_iterations: int = 20,
        alpha: float = 0.05,
        max_features_to_select: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.max_features_to_select = max_features_to_select
        self.selected_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "FeatureSelector":
        if y is None:
            raise ValueError("FeatureSelector requires y.")

        rng = np.random.RandomState(self.random_state)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_num = X[num_cols].fillna(0)

        hit_counts = {col: 0 for col in num_cols}

        for i in range(self.n_iterations):
            # Create shadow features
            X_shadow = X_num.apply(lambda c: rng.permutation(c.values))
            X_shadow.columns = [f"shadow_{c}" for c in num_cols]
            X_combined = pd.concat([X_num, X_shadow], axis=1)

            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state + i,
                n_jobs=self.n_jobs,
            )
            rf.fit(X_combined, y)
            importances = pd.Series(
                rf.feature_importances_, index=X_combined.columns
            )

            shadow_max = importances[[c for c in importances.index if c.startswith("shadow_")]].max()
            for col in num_cols:
                if importances[col] > shadow_max:
                    hit_counts[col] += 1

        # Select features that hit more than expected by chance
        threshold = self.n_iterations * (1 - self.alpha)
        selected = [col for col, hits in hit_counts.items() if hits >= threshold]

        if self.max_features_to_select is not None:
            importance_order = sorted(
                selected, key=lambda c: hit_counts[c], reverse=True
            )
            selected = importance_order[: self.max_features_to_select]

        self.selected_features_ = selected
        self._fitted = True
        logger.info(
            "FeatureSelector (Boruta): kept %d / %d features.",
            len(selected), len(num_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        available = [c for c in self.selected_features_ if c in X.columns]
        extra = [c for c in non_numeric if c in X.columns]
        return X[available + extra].copy()


# ---------------------------------------------------------------------------
# OptunaBoostingFitter
# ---------------------------------------------------------------------------

class OptunaBoostingFitter(BaseStep):
    """Optuna-tuned LightGBM binary classifier.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials.
    timeout : int | None
        Optuna timeout in seconds.
    eval_metric : str
        Metric for early stopping (``"auc"``).
    n_estimators_range : tuple[int, int]
        Range for ``n_estimators``.
    early_stopping_rounds : int
        LightGBM early stopping patience.
    search_space : dict | None
        Custom Optuna search space override.
    time_col : str | None
        If set, uses time-based split for evaluation.
    """

    def __init__(
        self,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        eval_metric: str = "auc",
        n_estimators_range: Tuple[int, int] = (100, 2000),
        early_stopping_rounds: int = 50,
        search_space: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_trials = n_trials
        self.timeout = timeout
        self.eval_metric = eval_metric
        self.n_estimators_range = n_estimators_range
        self.early_stopping_rounds = early_stopping_rounds
        self.search_space = search_space
        self.time_col = time_col
        self.model_ = None
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = 0.0

    def _get_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return train/val split (time-based or random)."""
        if self.time_col and self.time_col in X.columns:
            sorted_times = sorted(X[self.time_col].unique())
            cutoff = sorted_times[int(len(sorted_times) * 0.8)]
            train_mask = X[self.time_col] <= cutoff
            val_mask = X[self.time_col] > cutoff
        else:
            rng = np.random.RandomState(self.random_state)
            idx = rng.permutation(len(X))
            split = int(len(X) * 0.8)
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[idx[:split]] = True
            val_mask = ~train_mask

        X_train = X.loc[train_mask].select_dtypes(include=[np.number])
        X_val = X.loc[val_mask].select_dtypes(include=[np.number])
        y_train = y.loc[train_mask]
        y_val = y.loc[val_mask]
        return X_train, X_val, y_train, y_val

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kw: Any,
    ) -> "OptunaBoostingFitter":
        if y is None:
            raise ValueError("OptunaBoostingFitter requires y.")

        import optuna
        import lightgbm as lgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_train, X_val, y_train, y_val = self._get_split(X, y)
        self._feature_names = X_train.columns.tolist()

        # Compute imbalance weight
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        def objective(trial: optuna.Trial) -> float:
            ss = self.search_space or {}
            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state,
                "scale_pos_weight": scale_pos_weight,
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    ss.get("n_estimators_low", self.n_estimators_range[0]),
                    ss.get("n_estimators_high", self.n_estimators_range[1]),
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    ss.get("lr_low", 0.005),
                    ss.get("lr_high", 0.3),
                    log=True,
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    ss.get("depth_low", 3),
                    ss.get("depth_high", 10),
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    ss.get("leaves_low", 8),
                    ss.get("leaves_high", 256),
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    ss.get("min_child_low", 5),
                    ss.get("min_child_high", 100),
                ),
                "subsample": trial.suggest_float(
                    "subsample",
                    ss.get("subsample_low", 0.5),
                    ss.get("subsample_high", 1.0),
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    ss.get("colsample_low", 0.5),
                    ss.get("colsample_high", 1.0),
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    ss.get("alpha_low", 1e-8),
                    ss.get("alpha_high", 10.0),
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    ss.get("lambda_low", 1e-8),
                    ss.get("lambda_high", 10.0),
                    log=True,
                ),
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        logger.info(
            "OptunaBoostingFitter: best AUC=%.4f after %d trials.",
            self.best_score_, len(study.trials),
        )

        # Retrain on full data with best params
        best_p = {
            **self.best_params_,
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "scale_pos_weight": scale_pos_weight,
        }
        import lightgbm as lgb

        self.model_ = lgb.LGBMClassifier(**best_p)
        X_all_num = X.select_dtypes(include=[np.number])
        self.model_.fit(X_all_num, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(X[self._feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(X[self._feature_names])


# ---------------------------------------------------------------------------
# AdversarialValidation
# ---------------------------------------------------------------------------

class AdversarialValidation(BaseStep):
    """Detect covariate shift between train and holdout distributions.

    Trains a classifier to distinguish train vs holdout.
    If AUC is high, data distributions differ significantly.

    Parameters
    ----------
    auc_threshold : float
        If adversarial AUC > threshold, raises a warning.
    n_estimators : int
        Number of trees for the adversarial classifier.
    """

    def __init__(
        self,
        auc_threshold: float = 0.7,
        n_estimators: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.auc_threshold = auc_threshold
        self.n_estimators = n_estimators
        self.adversarial_auc_: float = 0.0
        self.feature_importances_: Optional[pd.Series] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        X_holdout: Optional[pd.DataFrame] = None,
        **kw: Any,
    ) -> "AdversarialValidation":
        if X_holdout is None:
            logger.warning("AdversarialValidation: no X_holdout provided; skipping.")
            self._fitted = True
            return self

        num_cols = X.select_dtypes(include=[np.number]).columns
        X_train_num = X[num_cols].fillna(0)
        X_hold_num = X_holdout[num_cols].fillna(0)

        combined_X = pd.concat([X_train_num, X_hold_num], axis=0).reset_index(drop=True)
        combined_y = np.concatenate([
            np.zeros(len(X_train_num)),
            np.ones(len(X_hold_num)),
        ])

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        rf.fit(combined_X, combined_y)
        preds = rf.predict_proba(combined_X)[:, 1]
        self.adversarial_auc_ = roc_auc_score(combined_y, preds)
        self.feature_importances_ = pd.Series(
            rf.feature_importances_, index=num_cols
        ).sort_values(ascending=False)

        if self.adversarial_auc_ > self.auc_threshold:
            logger.warning(
                "AdversarialValidation: AUC=%.3f > %.3f — significant "
                "distribution shift detected!",
                self.adversarial_auc_,
                self.auc_threshold,
            )
        else:
            logger.info(
                "AdversarialValidation: AUC=%.3f — distributions are similar.",
                self.adversarial_auc_,
            )
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
