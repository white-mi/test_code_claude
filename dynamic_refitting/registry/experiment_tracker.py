"""Lightweight experiment tracker (local JSON storage; optional MLflow)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dynamic_refitting.registry")


@dataclass
class ExperimentRun:
    """Record of a single experiment run."""

    run_id: str
    experiment_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "running"  # running | completed | failed


class ExperimentTracker:
    """Local-first experiment tracker.

    Falls back to a simple JSON file store.  When ``mlflow_tracking_uri``
    is provided, additionally logs to MLflow (optional dependency).

    Parameters
    ----------
    storage_dir : str | Path
        Local directory for run records.
    mlflow_tracking_uri : str | None
        Optional MLflow tracking URI.
    """

    def __init__(
        self,
        storage_dir: str | Path = ".experiments",
        mlflow_tracking_uri: Optional[str] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._mlflow_uri = mlflow_tracking_uri
        self._runs: Dict[str, ExperimentRun] = self._load_all()
        self._mlflow = None
        if mlflow_tracking_uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(mlflow_tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                logger.warning(
                    "mlflow not installed; falling back to local tracking."
                )

    # -- public API -----------------------------------------------------------

    def start_run(
        self,
        experiment_name: str,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        run_id = f"{experiment_name}_{int(time.time() * 1000)}"
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            params=params or {},
            tags=tags or {},
        )
        self._runs[run_id] = run
        self._save_run(run)

        if self._mlflow is not None:
            self._mlflow.set_experiment(experiment_name)
            self._mlflow.start_run(run_name=run_id)
            if params:
                self._mlflow.log_params(params)

        logger.info("Started run %s", run_id)
        return run

    def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        run = self._runs[run_id]
        run.metrics.update(metrics)
        self._save_run(run)
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics)

    def log_artifact(self, run_id: str, path: str) -> None:
        run = self._runs[run_id]
        run.artifacts.append(path)
        self._save_run(run)
        if self._mlflow is not None:
            self._mlflow.log_artifact(path)

    def end_run(self, run_id: str, status: str = "completed") -> None:
        run = self._runs[run_id]
        run.status = status
        self._save_run(run)
        if self._mlflow is not None:
            self._mlflow.end_run()
        logger.info("Ended run %s with status=%s", run_id, status)

    def get_run(self, run_id: str) -> ExperimentRun:
        return self._runs[run_id]

    def list_runs(
        self, experiment_name: Optional[str] = None
    ) -> List[ExperimentRun]:
        runs = list(self._runs.values())
        if experiment_name:
            runs = [r for r in runs if r.experiment_name == experiment_name]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    # -- persistence ----------------------------------------------------------

    def _save_run(self, run: ExperimentRun) -> None:
        path = self.storage_dir / f"{run.run_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)

    def _load_all(self) -> Dict[str, ExperimentRun]:
        runs: Dict[str, ExperimentRun] = {}
        for p in self.storage_dir.glob("*.json"):
            with open(p) as f:
                data = json.load(f)
            run = ExperimentRun(**data)
            runs[run.run_id] = run
        return runs
