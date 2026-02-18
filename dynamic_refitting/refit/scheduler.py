"""Refit scheduler — periodic or event-driven scheduling logic."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("dynamic_refitting.refit")


class RefitScheduler:
    """Schedule periodic refit checks.

    This is a lightweight in-process scheduler.  For production use,
    integrate with Airflow, Prefect, or cron.

    Parameters
    ----------
    check_interval_seconds : int
        How often to evaluate triggers (in seconds).
    max_checks : int | None
        Maximum number of checks before stopping (None = unlimited).
    """

    def __init__(
        self,
        check_interval_seconds: int = 3600,
        max_checks: Optional[int] = None,
    ) -> None:
        self.check_interval_seconds = check_interval_seconds
        self.max_checks = max_checks
        self._check_count = 0
        self._callbacks: List[Callable[[], bool]] = []
        self.history_: List[Dict[str, Any]] = []

    def register_callback(self, callback: Callable[[], bool]) -> None:
        """Register a callable that returns True if refit was performed."""
        self._callbacks.append(callback)

    def run_once(self) -> bool:
        """Execute all registered callbacks once.

        Returns
        -------
        bool
            True if any callback triggered a refit.
        """
        self._check_count += 1
        any_refit = False
        for cb in self._callbacks:
            try:
                result = cb()
                if result:
                    any_refit = True
            except Exception as e:
                logger.error("Scheduler callback error: %s", e)

        self.history_.append({
            "check_number": self._check_count,
            "timestamp": time.time(),
            "refit_triggered": any_refit,
        })
        return any_refit

    def run(self) -> None:
        """Run the scheduler loop (blocking).

        Calls :meth:`run_once` at the configured interval until
        ``max_checks`` is reached.
        """
        logger.info(
            "RefitScheduler: starting (interval=%ds, max_checks=%s).",
            self.check_interval_seconds,
            self.max_checks,
        )
        while True:
            self.run_once()
            if self.max_checks and self._check_count >= self.max_checks:
                logger.info("RefitScheduler: max checks reached; stopping.")
                break
            time.sleep(self.check_interval_seconds)
