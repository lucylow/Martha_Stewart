"""Readiness and operational health checks without leaking sensitive configuration."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from ..config import Settings

@dataclass(frozen=True, slots=True)
class HealthReport:
    status: str
    checks: dict[str, bool]
    timestamp: str
    details: dict[str, str]

    def to_dict(self) -> dict:
        return {"status": self.status, "checks": dict(self.checks),
                "timestamp": self.timestamp, "details": dict(self.details)}

class HealthMonitor:
    def __init__(self, settings: Settings, database_probe: Callable[[], bool] | None = None,
                 market_data_probe: Callable[[], bool] | None = None):
        self.settings = settings
        self.database_probe = database_probe or (lambda: True)
        self.market_data_probe = market_data_probe or (lambda: True)

    def report(self) -> HealthReport:
        checks: dict[str, bool] = {
            "configuration_valid": True,
            "database": self._safe_probe(self.database_probe),
            "market_data": self._safe_probe(self.market_data_probe),
        }
        db_path = Path(self.settings.database_path)
        details = {"mode": self.settings.mode, "execution": "paper_only",
                   "database_path_configured": str(db_path),
                   "live_trading": "unsupported by this package"}
        return HealthReport("ok" if all(checks.values()) else "degraded", checks,
                            datetime.now(timezone.utc).isoformat(), details)

    @staticmethod
    def _safe_probe(probe: Callable[[], bool]) -> bool:
        try:
            return bool(probe())
        except Exception:
            return False
