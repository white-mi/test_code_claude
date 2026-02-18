"""Plug-in registry for dynamically registering pipeline step classes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from dynamic_refitting.config import BaseStep

logger = logging.getLogger("dynamic_refitting.registry")

# Global registry: category -> name -> class
_REGISTRY: Dict[str, Dict[str, Type[BaseStep]]] = {}


class StepRegistry:
    """Central registry for pipeline step plug-ins.

    Usage
    -----
    >>> StepRegistry.register("preprocessing", "my_scaler", MyScalerStep)
    >>> cls = StepRegistry.get("preprocessing", "my_scaler")
    >>> step = cls(random_state=42)
    """

    @staticmethod
    def register(
        category: str,
        name: str,
        cls: Type[BaseStep],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a new step class.

        Parameters
        ----------
        category : str
            Grouping key, e.g. ``"boost"``, ``"logreg"``, ``"validation"``.
        name : str
            Unique name within the category.
        cls : type
            A subclass of :class:`BaseStep`.
        overwrite : bool
            If *True*, silently replace an existing registration.
        """
        if not issubclass(cls, BaseStep):
            raise TypeError(f"{cls} must be a subclass of BaseStep.")
        _REGISTRY.setdefault(category, {})
        if name in _REGISTRY[category] and not overwrite:
            raise KeyError(
                f"Step '{name}' is already registered in category "
                f"'{category}'. Use overwrite=True to replace."
            )
        _REGISTRY[category][name] = cls
        logger.debug("Registered step %s/%s -> %s", category, name, cls.__name__)

    @staticmethod
    def get(category: str, name: str) -> Type[BaseStep]:
        """Retrieve a registered step class."""
        try:
            return _REGISTRY[category][name]
        except KeyError:
            raise KeyError(
                f"Step '{name}' not found in category '{category}'. "
                f"Available: {list(_REGISTRY.get(category, {}).keys())}"
            )

    @staticmethod
    def list_categories() -> list[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def list_steps(category: str) -> list[str]:
        return list(_REGISTRY.get(category, {}).keys())

    @staticmethod
    def clear() -> None:
        """Remove all registrations (mainly for testing)."""
        _REGISTRY.clear()
