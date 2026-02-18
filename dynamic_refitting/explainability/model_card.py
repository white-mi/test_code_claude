"""Model card generator for documentation and compliance."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.explainability")


class ModelCardGenerator(BaseStep):
    """Generate a model card (structured documentation) for a scoring model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_version : str
        Version string.
    output_dir : str | Path | None
        Directory to save the model card.
    """

    def __init__(
        self,
        model_name: str = "ScoringModel",
        model_version: str = "1.0",
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_version = model_version
        self.output_dir = Path(output_dir) if output_dir else None
        self.card_: Dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kw: Any,
    ) -> "ModelCardGenerator":
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def generate(
        self,
        model_type: str = "unknown",
        features: Optional[List[str]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        dataset_description: str = "",
        intended_use: str = "Binary credit scoring",
        limitations: str = "",
        ethical_considerations: str = "",
    ) -> Dict[str, Any]:
        """Build the model card.

        Returns
        -------
        dict
            Structured model card.
        """
        self.card_ = {
            "model_details": {
                "name": self.model_name,
                "version": self.model_version,
                "type": model_type,
                "created_at": datetime.utcnow().isoformat(),
                "n_features": len(features) if features else 0,
                "features": features or [],
            },
            "intended_use": intended_use,
            "dataset": {
                "description": dataset_description,
            },
            "metrics": {
                "training": training_metrics or {},
                "validation": validation_metrics or {},
            },
            "limitations": limitations,
            "ethical_considerations": ethical_considerations,
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"model_card_{self.model_name}.json"
            with open(path, "w") as f:
                json.dump(self.card_, f, indent=2, default=str)
            logger.info("Model card saved to %s", path)

        return self.card_
