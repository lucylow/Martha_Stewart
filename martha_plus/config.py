"""Environment-backed configuration with conservative defaults.

The module intentionally avoids loading .env files implicitly. Deployment code
should choose and audit its own environment-loading mechanism.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping, Any

from .errors import ConfigurationError


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Expected a boolean value, got {value!r}")


def _float(env: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"{key} must be numeric") from exc


def _int(env: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"{key} must be an integer") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings.

    ``mode`` supports ``research`` and ``paper`` only. There is deliberately no
    live mode in this package; adding one would require a separately reviewed
    broker implementation and an operational risk process.
    """
    mode: str = "research"
    database_path: str = "data/martha_plus.sqlite3"
    log_level: str = "INFO"
    starting_cash: float = 100_000.0
    max_order_notional: float = 5_000.0
    max_position_notional: float = 15_000.0
    max_gross_exposure: float = 0.75
    max_daily_loss: float = 2_000.0
    max_drawdown_fraction: float = 0.12
    max_open_orders: int = 20
    max_quote_age_seconds: float = 5.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    paper_fill_probability: float = 1.0
    allow_short: bool = False
    allow_fractional_shares: bool = False
    kill_switch_file: str = "data/KILL_SWITCH"
    api_token: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, Any] | None = None) -> "Settings":
        env: Mapping[str, Any] = os.environ if environ is None else environ
        settings = cls(
            mode=str(env.get("MARTHA_MODE", "research")).strip().lower(),
            database_path=str(env.get("MARTHA_DATABASE_PATH", "data/martha_plus.sqlite3")),
            log_level=str(env.get("MARTHA_LOG_LEVEL", "INFO")).upper(),
            starting_cash=_float(env, "MARTHA_STARTING_CASH", 100_000.0),
            max_order_notional=_float(env, "MARTHA_MAX_ORDER_NOTIONAL", 5_000.0),
            max_position_notional=_float(env, "MARTHA_MAX_POSITION_NOTIONAL", 15_000.0),
            max_gross_exposure=_float(env, "MARTHA_MAX_GROSS_EXPOSURE", 0.75),
            max_daily_loss=_float(env, "MARTHA_MAX_DAILY_LOSS", 2_000.0),
            max_drawdown_fraction=_float(env, "MARTHA_MAX_DRAWDOWN_FRACTION", 0.12),
            max_open_orders=_int(env, "MARTHA_MAX_OPEN_ORDERS", 20),
            max_quote_age_seconds=_float(env, "MARTHA_MAX_QUOTE_AGE_SECONDS", 5.0),
            commission_bps=_float(env, "MARTHA_COMMISSION_BPS", 1.0),
            slippage_bps=_float(env, "MARTHA_SLIPPAGE_BPS", 2.0),
            paper_fill_probability=_float(env, "MARTHA_PAPER_FILL_PROBABILITY", 1.0),
            allow_short=_bool(env.get("MARTHA_ALLOW_SHORT"), False),
            allow_fractional_shares=_bool(env.get("MARTHA_ALLOW_FRACTIONAL_SHARES"), False),
            kill_switch_file=str(env.get("MARTHA_KILL_SWITCH_FILE", "data/KILL_SWITCH")),
            api_token=str(env.get("MARTHA_API_TOKEN", "")),
        )
        settings.validate()
        return settings

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Settings":
        """Load from config-style mappings, accepting field names rather than env names."""
        allowed = {f.name for f in fields(cls)}
        unknown = set(mapping) - allowed
        if unknown:
            raise ConfigurationError(f"Unknown settings: {', '.join(sorted(unknown))}")
        try:
            settings = cls(**dict(mapping))
        except TypeError as exc:
            raise ConfigurationError(str(exc)) from exc
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.mode not in {"research", "paper"}:
            raise ConfigurationError("mode must be 'research' or 'paper'; live execution is unsupported")
        if self.starting_cash <= 0:
            raise ConfigurationError("starting_cash must be positive")
        for name in ("max_order_notional", "max_position_notional", "max_daily_loss"):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be positive")
        if not 0 < self.max_gross_exposure <= 1.0:
            raise ConfigurationError("max_gross_exposure must be in (0, 1]")
        if not 0 < self.max_drawdown_fraction < 1:
            raise ConfigurationError("max_drawdown_fraction must be in (0, 1)")
        if self.max_open_orders < 1:
            raise ConfigurationError("max_open_orders must be at least 1")
        if self.max_quote_age_seconds <= 0:
            raise ConfigurationError("max_quote_age_seconds must be positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ConfigurationError("fees and slippage must not be negative")
        if not 0 <= self.paper_fill_probability <= 1:
            raise ConfigurationError("paper_fill_probability must be between 0 and 1")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError("log_level is not a recognized Python logging level")

    def as_safe_dict(self) -> dict[str, Any]:
        """Return a representation safe for diagnostics; never expose the API token."""
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        result["api_token"] = "[configured]" if self.api_token else "[not configured]"
        return result
