"""Small role-based access-control engine with explicit-deny and audit trails.

This is an application-layer authorization helper, not a replacement for
network authentication, database permissions, or infrastructure security.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
from typing import Iterable, Mapping, Sequence


class Action(str, Enum):
    READ = "read"
    WRITE = "write"
    TRADE_PAPER = "trade_paper"
    TRADE_LIVE = "trade_live"
    APPROVE_STRATEGY = "approve_strategy"
    CHANGE_RISK_LIMITS = "change_risk_limits"
    ADMINISTER = "administer"
    EXPORT = "export"


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    roles: frozenset[str] = frozenset()
    active: bool = True
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("principal subject cannot be empty")
        if any(not r.strip() for r in self.roles):
            raise ValueError("role names cannot be empty")


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    role: str
    action: Action
    resource_prefix: str = "*"
    effect: Effect = Effect.ALLOW
    conditions: Mapping[str, str] = field(default_factory=dict)
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.role.strip():
            raise ValueError("policy name and role are required")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("policy expiry must be timezone-aware")

    def matches(self, principal: Principal, action: Action, resource: str,
                context: Mapping[str, str], now: datetime) -> bool:
        if self.expires_at and now >= self.expires_at:
            return False
        if self.role not in principal.roles or self.action is not action:
            return False
        if self.resource_prefix != "*" and not resource.startswith(self.resource_prefix):
            return False
        return all(context.get(k) == v for k, v in self.conditions.items())


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    reason: str
    principal: str
    action: Action
    resource: str
    matched_policies: tuple[str, ...] = ()
    decided_at: str = ""


@dataclass(frozen=True, slots=True)
class AccessEvent:
    principal: str
    action: str
    resource: str
    allowed: bool
    reason: str
    occurred_at: str
    correlation_id: str | None = None


class AccessDenied(PermissionError):
    """Raised when an access check is enforced and not permitted."""


class AccessController:
    """Thread-safe policy evaluator where any matching explicit deny wins."""
    def __init__(self, policies: Iterable[Policy] = (), *,
                 clock=lambda: datetime.now(timezone.utc),
                 default_allow: bool = False,
                 forbid_live_trading: bool = True) -> None:
        self._policies = list(policies)
        self._clock = clock
        self._default_allow = default_allow
        self._forbid_live_trading = forbid_live_trading
        self._events: list[AccessEvent] = []
        self._lock = RLock()

    @property
    def policies(self) -> tuple[Policy, ...]:
        with self._lock:
            return tuple(self._policies)

    @property
    def events(self) -> tuple[AccessEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def add_policy(self, policy: Policy, *, replace: bool = False) -> None:
        with self._lock:
            for index, existing in enumerate(self._policies):
                if existing.name == policy.name:
                    if not replace:
                        raise ValueError(f"policy already exists: {policy.name}")
                    self._policies[index] = policy
                    return
            self._policies.append(policy)

    def remove_policy(self, name: str) -> bool:
        with self._lock:
            before = len(self._policies)
            self._policies = [p for p in self._policies if p.name != name]
            return len(self._policies) != before

    def evaluate(self, principal: Principal, action: Action, resource: str, *,
                 context: Mapping[str, str] | None = None,
                 correlation_id: str | None = None) -> Decision:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("authorization clock must return timezone-aware datetime")
        normalized_resource = resource.strip()
        normalized_context = dict(context or {})
        with self._lock:
            policies = tuple(self._policies)
        matches = tuple(p for p in policies if p.matches(
            principal, action, normalized_resource, normalized_context, now))
        denies = tuple(p for p in matches if p.effect is Effect.DENY)
        allows = tuple(p for p in matches if p.effect is Effect.ALLOW)
        if not principal.active:
            allowed, reason = False, "principal_inactive"
        elif action is Action.TRADE_LIVE and self._forbid_live_trading:
            allowed, reason = False, "live_trading_disabled_by_policy"
        elif denies:
            allowed, reason = False, "explicit_deny"
        elif allows:
            allowed, reason = True, "matched_allow_policy"
        else:
            allowed, reason = self._default_allow, "default_allow" if self._default_allow else "no_matching_policy"
        decision = Decision(allowed, reason, principal.subject, action,
                            normalized_resource, tuple(p.name for p in matches), now.isoformat())
        event = AccessEvent(principal.subject, action.value, normalized_resource,
                            allowed, reason, now.isoformat(), correlation_id)
        with self._lock:
            self._events.append(event)
        return decision

    def require(self, principal: Principal, action: Action, resource: str, **kwargs) -> Decision:
        decision = self.evaluate(principal, action, resource, **kwargs)
        if not decision.allowed:
            raise AccessDenied(f"{action.value} denied for {principal.subject}: {decision.reason}")
        return decision

    def prune_events(self, *, before: datetime) -> int:
        if before.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        with self._lock:
            retained = [e for e in self._events if datetime.fromisoformat(e.occurred_at) >= before]
            removed = len(self._events) - len(retained)
            self._events = retained
            return removed

    def summary(self) -> dict[str, int]:
        with self._lock:
            events = tuple(self._events)
        return {
            "total": len(events),
            "allowed": sum(e.allowed for e in events),
            "denied": sum(not e.allowed for e in events),
            "distinct_principals": len({e.principal for e in events}),
            "distinct_resources": len({e.resource for e in events}),
        }


def standard_paper_policies() -> tuple[Policy, ...]:
    """Return a conservative baseline; administrators must assign roles explicitly."""
    return (
        Policy("reader-read", "reader", Action.READ),
        Policy("research-read", "researcher", Action.READ),
        Policy("research-write", "researcher", Action.WRITE, "research/"),
        Policy("paper-operator", "operator", Action.TRADE_PAPER, "orders/paper/"),
        Policy("risk-officer", "risk_officer", Action.CHANGE_RISK_LIMITS, "risk/"),
        Policy("strategy-reviewer", "reviewer", Action.APPROVE_STRATEGY, "strategy/"),
        Policy("admin-read", "admin", Action.READ),
        Policy("admin-write", "admin", Action.WRITE),
        Policy("admin-export", "admin", Action.EXPORT),
    )
