"""Local model registry — versioning, tagging, promotion, and rollback."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger("dynamic_refitting.registry")


@dataclass
class ModelVersion:
    """Metadata for a single model version."""

    version: int
    path: str
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    stage: str = "staging"  # staging | production | archived
    created_at: float = field(default_factory=time.time)
    description: str = ""


class ModelRegistry:
    """File-system-based model registry with versioning and promotion.

    Parameters
    ----------
    root_dir : str | Path
        Root directory for the registry storage.
    """

    MANIFEST = "manifest.json"

    def __init__(self, root_dir: str | Path = ".model_registry") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root_dir / self.MANIFEST
        self._models: Dict[str, List[ModelVersion]] = self._load_manifest()

    # -- public API -----------------------------------------------------------

    def register(
        self,
        model_name: str,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> ModelVersion:
        """Register a new model version.

        Parameters
        ----------
        model_name : str
            Logical model name (e.g. ``"scoring_v2"``).
        model : Any
            Serialisable model or pipeline object.
        metrics : dict | None
            Evaluation metrics to attach.
        tags : dict | None
            Arbitrary key-value tags.
        description : str
            Human-readable description.

        Returns
        -------
        ModelVersion
        """
        versions = self._models.setdefault(model_name, [])
        next_version = len(versions) + 1

        model_dir = self.root_dir / model_name / f"v{next_version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        mv = ModelVersion(
            version=next_version,
            path=str(model_path),
            tags=tags or {},
            metrics=metrics or {},
            description=description,
        )
        versions.append(mv)
        self._save_manifest()
        logger.info(
            "Registered %s v%d (stage=%s)", model_name, next_version, mv.stage
        )
        return mv

    def get_version(self, model_name: str, version: int) -> ModelVersion:
        versions = self._models.get(model_name, [])
        for mv in versions:
            if mv.version == version:
                return mv
        raise KeyError(f"Version {version} not found for model '{model_name}'.")

    def get_production(self, model_name: str) -> Optional[ModelVersion]:
        """Return the version currently promoted to production."""
        for mv in reversed(self._models.get(model_name, [])):
            if mv.stage == "production":
                return mv
        return None

    def promote(self, model_name: str, version: int) -> None:
        """Promote a specific version to production and demote the current."""
        current_prod = self.get_production(model_name)
        if current_prod is not None:
            current_prod.stage = "archived"
        mv = self.get_version(model_name, version)
        mv.stage = "production"
        self._save_manifest()
        logger.info("Promoted %s v%d to production.", model_name, version)

    def rollback(self, model_name: str) -> Optional[ModelVersion]:
        """Rollback: archive current production, promote previous."""
        versions = self._models.get(model_name, [])
        archived = [v for v in versions if v.stage == "archived"]
        current = self.get_production(model_name)
        if current is not None:
            current.stage = "archived"
        if archived:
            prev = archived[-1]
            prev.stage = "production"
            self._save_manifest()
            logger.info("Rolled back %s to v%d.", model_name, prev.version)
            return prev
        logger.warning("No previous version to rollback for '%s'.", model_name)
        return None

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        return list(self._models.get(model_name, []))

    def load_model(self, model_name: str, version: Optional[int] = None) -> Any:
        """Load a model artefact.  Defaults to production version."""
        if version is not None:
            mv = self.get_version(model_name, version)
        else:
            mv = self.get_production(model_name)
            if mv is None:
                raise KeyError(f"No production version for '{model_name}'.")
        return joblib.load(mv.path)

    def tag(self, model_name: str, version: int, tags: Dict[str, str]) -> None:
        mv = self.get_version(model_name, version)
        mv.tags.update(tags)
        self._save_manifest()

    # -- persistence ----------------------------------------------------------

    def _load_manifest(self) -> Dict[str, List[ModelVersion]]:
        if not self._manifest_path.exists():
            return {}
        with open(self._manifest_path) as f:
            raw = json.load(f)
        result: Dict[str, List[ModelVersion]] = {}
        for name, versions in raw.items():
            result[name] = [ModelVersion(**v) for v in versions]
        return result

    def _save_manifest(self) -> None:
        raw = {
            name: [asdict(v) for v in versions]
            for name, versions in self._models.items()
        }
        with open(self._manifest_path, "w") as f:
            json.dump(raw, f, indent=2, default=str)
