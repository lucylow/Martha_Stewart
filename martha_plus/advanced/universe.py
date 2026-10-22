"""Point-in-time universe construction with explicit exclusion reasons."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SecurityRecord:
    symbol: str
    as_of: date
    active: bool = True
    exchange: str = "UNKNOWN"
    asset_class: str = "equity"
    sector: str = "unknown"
    market_cap: float | None = None
    average_dollar_volume: float | None = None
    price: float | None = None
    ipo_date: date | None = None
    delisting_date: date | None = None
    flags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        for key in ("market_cap", "average_dollar_volume", "price"):
            value = getattr(self, key)
            if value is not None and (not isfinite(float(value)) or value < 0):
                raise ValueError(f"{key} must be finite and non-negative")
        if self.ipo_date and self.ipo_date > self.as_of:
            raise ValueError("security cannot be in the universe before IPO date")


@dataclass(frozen=True, slots=True)
class UniversePolicy:
    exchanges: frozenset[str] = frozenset()
    asset_classes: frozenset[str] = frozenset({"equity"})
    excluded_symbols: frozenset[str] = frozenset()
    excluded_flags: frozenset[str] = frozenset({"halted", "delisted", "restricted"})
    minimum_market_cap: float = 0.0
    minimum_average_dollar_volume: float = 0.0
    minimum_price: float = 0.0
    minimum_history_days: int = 0
    require_active: bool = True

    def __post_init__(self) -> None:
        if any(not isfinite(v) or v < 0 for v in (self.minimum_market_cap,
               self.minimum_average_dollar_volume, self.minimum_price)):
            raise ValueError("universe thresholds must be finite and non-negative")
        if self.minimum_history_days < 0:
            raise ValueError("minimum_history_days must be non-negative")


@dataclass(frozen=True, slots=True)
class UniverseDecision:
    symbol: str
    included: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    as_of: date
    symbols: tuple[str, ...]
    decisions: tuple[UniverseDecision, ...]
    policy_fingerprint: str

    @property
    def excluded(self) -> dict[str, tuple[str, ...]]:
        return {item.symbol: item.reasons for item in self.decisions if not item.included}


class UniverseBuilder:
    """Build a historical universe; caller must provide survivorship-safe data."""
    def __init__(self, policy: UniversePolicy | None = None) -> None:
        self.policy = policy or UniversePolicy()

    def decide(self, security: SecurityRecord, as_of: date) -> UniverseDecision:
        p = self.policy
        symbol = security.symbol.strip().upper()
        reasons: list[str] = []
        if security.as_of > as_of:
            reasons.append("record_not_available_as_of_date")
        # A currently delisted security may still belong in a historical universe
        # when its delisting date lies after the requested point in time.
        if p.require_active and not security.active and not (security.delisting_date and security.delisting_date > as_of):
            reasons.append("inactive_security")
        if security.ipo_date and security.ipo_date > as_of:
            reasons.append("not_yet_ipo")
        if security.delisting_date and security.delisting_date <= as_of:
            reasons.append("already_delisted")
        if p.exchanges and security.exchange.upper() not in {x.upper() for x in p.exchanges}:
            reasons.append("exchange_not_allowed")
        if p.asset_classes and security.asset_class.lower() not in {x.lower() for x in p.asset_classes}:
            reasons.append("asset_class_not_allowed")
        if symbol in {x.upper() for x in p.excluded_symbols}:
            reasons.append("symbol_explicitly_excluded")
        if security.flags & p.excluded_flags:
            reasons.extend(f"flag:{flag}" for flag in sorted(security.flags & p.excluded_flags))
        if security.market_cap is None:
            if p.minimum_market_cap > 0:
                reasons.append("market_cap_missing")
        elif security.market_cap < p.minimum_market_cap:
            reasons.append("market_cap_below_minimum")
        if security.average_dollar_volume is None:
            if p.minimum_average_dollar_volume > 0:
                reasons.append("average_dollar_volume_missing")
        elif security.average_dollar_volume < p.minimum_average_dollar_volume:
            reasons.append("average_dollar_volume_below_minimum")
        if security.price is None:
            if p.minimum_price > 0:
                reasons.append("price_missing")
        elif security.price < p.minimum_price:
            reasons.append("price_below_minimum")
        if p.minimum_history_days and security.ipo_date:
            if (as_of - security.ipo_date).days < p.minimum_history_days:
                reasons.append("insufficient_listed_history")
        return UniverseDecision(symbol, not reasons, tuple(dict.fromkeys(reasons)))

    def build(self, records: Iterable[SecurityRecord], as_of: date) -> UniverseSnapshot:
        decisions: list[UniverseDecision] = []
        seen: set[str] = set()
        for record in records:
            symbol = record.symbol.strip().upper()
            if symbol in seen:
                raise ValueError(f"duplicate security record for {symbol}")
            seen.add(symbol)
            decisions.append(self.decide(record, as_of))
        decisions.sort(key=lambda d: d.symbol)
        included = tuple(item.symbol for item in decisions if item.included)
        from dataclasses import fields
        policy_dict = {item.name: (sorted(getattr(self.policy, item.name))
                                   if isinstance(getattr(self.policy, item.name), (set, frozenset))
                                   else getattr(self.policy, item.name))
                       for item in fields(self.policy)}
        import hashlib, json
        digest = hashlib.sha256(json.dumps(policy_dict, sort_keys=True, default=str).encode()).hexdigest()
        return UniverseSnapshot(as_of, included, tuple(decisions), digest)

    @staticmethod
    def stable_intersection(*universes: Sequence[str]) -> tuple[str, ...]:
        if not universes:
            return ()
        shared = set(str(x).strip().upper() for x in universes[0])
        for universe in universes[1:]:
            shared.intersection_update(str(x).strip().upper() for x in universe)
        return tuple(sorted(s for s in shared if s))

    @staticmethod
    def minimum_history_filter(symbol_to_first_date: Mapping[str, date], as_of: date,
                               minimum_days: int) -> tuple[str, ...]:
        if minimum_days < 0:
            raise ValueError("minimum_days must be non-negative")
        return tuple(sorted(symbol.strip().upper() for symbol, first in symbol_to_first_date.items()
                            if (as_of - first).days >= minimum_days))
