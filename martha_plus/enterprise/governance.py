"""Strategy registry, approval workflow, immutability checks, and shadow-mode evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from threading import RLock
from typing import Any, Callable, Iterable, Mapping
import json


class StrategyStage(str, Enum):
    DRAFT = "draft"
    RESEARCH = "research"
    BACKTESTED = "backtested"
    SHADOW = "shadow"
    PAPER_APPROVED = "paper_approved"
    PAPER_ACTIVE = "paper_active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


ALLOWED_STAGE_TRANSITIONS = {
    StrategyStage.DRAFT: {StrategyStage.RESEARCH, StrategyStage.RETIRED},
    StrategyStage.RESEARCH: {StrategyStage.BACKTESTED, StrategyStage.SUSPENDED, StrategyStage.RETIRED},
    StrategyStage.BACKTESTED: {StrategyStage.SHADOW, StrategyStage.SUSPENDED, StrategyStage.RETIRED},
    StrategyStage.SHADOW: {StrategyStage.PAPER_APPROVED, StrategyStage.SUSPENDED, StrategyStage.RETIRED},
    StrategyStage.PAPER_APPROVED: {StrategyStage.PAPER_ACTIVE, StrategyStage.SUSPENDED, StrategyStage.RETIRED},
    StrategyStage.PAPER_ACTIVE: {StrategyStage.SUSPENDED, StrategyStage.RETIRED},
    StrategyStage.SUSPENDED: {StrategyStage.RESEARCH, StrategyStage.BACKTESTED, StrategyStage.SHADOW, StrategyStage.RETIRED},
    StrategyStage.RETIRED: set(),
}


@dataclass(frozen=True, slots=True)
class StrategyManifest:
    strategy_id: str
    version: str
    code_digest: str
    config: Mapping[str, Any]
    owner: str
    created_at: str
    dataset_digest: str | None = None
    notes: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.strategy_id, self.version, self.code_digest, self.owner)):
            raise ValueError("strategy_id, version, code_digest, and owner are required")
        if len(self.code_digest) < 16:
            raise ValueError("code_digest must be a stable fingerprint, not an empty placeholder")
        parsed = datetime.fromisoformat(self.created_at)
        if parsed.tzinfo is None:
            raise ValueError("created_at must be timezone-aware ISO timestamp")

    @property
    def config_digest(self) -> str:
        return digest_json(self.config)

    @property
    def manifest_digest(self) -> str:
        return digest_json({"strategy_id": self.strategy_id, "version": self.version,
            "code_digest": self.code_digest, "config_digest": self.config_digest,
            "owner": self.owner, "created_at": self.created_at, "dataset_digest": self.dataset_digest,
            "notes": self.notes, "tags": self.tags})


@dataclass(frozen=True, slots=True)
class StrategyTransition:
    strategy_id: str
    version: str
    from_stage: StrategyStage
    to_stage: StrategyStage
    actor: str
    reason: str
    at: str
    evidence: Mapping[str, str] = field(default_factory=dict)


class GovernanceError(RuntimeError):
    pass


def digest_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
    return sha256(data).hexdigest()


class StrategyRegistry:
    """Immutable manifest versions plus an append-only stage transition log."""
    def __init__(self, *, clock=lambda: datetime.now(timezone.utc)) -> None:
        self._clock = clock
        self._manifests: dict[tuple[str, str], StrategyManifest] = {}
        self._stages: dict[tuple[str, str], StrategyStage] = {}
        self._transitions: list[StrategyTransition] = []
        self._approvals: dict[tuple[str, str], set[str]] = {}
        self._approval_evidence: dict[tuple[str, str], dict[str, str]] = {}
        self._lock = RLock()

    @property
    def transitions(self) -> tuple[StrategyTransition, ...]:
        with self._lock:
            return tuple(self._transitions)

    def register(self, manifest: StrategyManifest) -> None:
        key = manifest.strategy_id, manifest.version
        with self._lock:
            if key in self._manifests:
                existing = self._manifests[key]
                if existing.manifest_digest != manifest.manifest_digest:
                    raise GovernanceError("strategy version is immutable; choose a new version")
                return
            if any(sid == manifest.strategy_id and version == manifest.version for sid, version in self._manifests):
                raise GovernanceError("duplicate strategy version")
            self._manifests[key] = manifest
            self._stages[key] = StrategyStage.DRAFT
            self._approvals[key] = set()
            self._approval_evidence[key] = {}

    def get(self, strategy_id: str, version: str) -> StrategyManifest:
        with self._lock:
            return self._manifests[(strategy_id, version)]

    def stage(self, strategy_id: str, version: str) -> StrategyStage:
        with self._lock:
            return self._stages[(strategy_id, version)]

    def transition(self, strategy_id: str, version: str, target: StrategyStage, *,
                   actor: str, reason: str, evidence: Mapping[str, str] | None = None) -> StrategyTransition:
        if not actor.strip() or not reason.strip():
            raise ValueError("actor and reason are required for governance transitions")
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("clock must be timezone-aware")
        key = (strategy_id, version)
        with self._lock:
            current = self._stages[key]
            if target not in ALLOWED_STAGE_TRANSITIONS[current]:
                raise GovernanceError(f"transition {current.value} -> {target.value} is not allowed")
            manifest = self._manifests[key]
            if target is StrategyStage.PAPER_APPROVED:
                approvers = self._approvals[key]
                if len(approvers) < 2:
                    raise GovernanceError("paper approval requires two distinct reviewers")
                required = {"backtest_report", "risk_review", "dataset_digest"}
                if not required.issubset(set((evidence or {}).keys())):
                    raise GovernanceError(f"approval evidence must include {sorted(required)}")
                if manifest.dataset_digest and (evidence or {}).get("dataset_digest") != manifest.dataset_digest:
                    raise GovernanceError("approval dataset digest differs from manifest")
            if target is StrategyStage.PAPER_ACTIVE and current is not StrategyStage.PAPER_APPROVED:
                raise GovernanceError("paper activation requires prior approval")
            record = StrategyTransition(strategy_id, version, current, target, actor, reason,
                                        now.isoformat(), dict(evidence or {}))
            self._stages[key] = target
            self._transitions.append(record)
            return record

    def approve(self, strategy_id: str, version: str, reviewer: str, *,
                evidence_digest: str) -> None:
        if not reviewer.strip() or len(evidence_digest) < 16:
            raise ValueError("reviewer and evidence digest are required")
        key = (strategy_id, version)
        with self._lock:
            if self._stages[key] not in (StrategyStage.SHADOW, StrategyStage.PAPER_APPROVED):
                raise GovernanceError("strategy must be in shadow stage before approval")
            manifest = self._manifests[key]
            if reviewer == manifest.owner:
                raise GovernanceError("strategy owner cannot approve their own strategy")
            self._approvals[key].add(reviewer)
            self._approval_evidence[key][reviewer] = evidence_digest

    def approval_evidence(self, strategy_id: str, version: str) -> Mapping[str, str]:
        with self._lock:
            return dict(self._approval_evidence[(strategy_id, version)])

    def list_versions(self, strategy_id: str | None = None) -> tuple[tuple[StrategyManifest, StrategyStage], ...]:
        with self._lock:
            keys = sorted(self._manifests)
            return tuple((self._manifests[k], self._stages[k]) for k in keys
                         if strategy_id is None or k[0] == strategy_id)

    def active_paper_strategies(self) -> tuple[StrategyManifest, ...]:
        with self._lock:
            return tuple(self._manifests[key] for key, stage in self._stages.items()
                         if stage is StrategyStage.PAPER_ACTIVE)

    def audit(self) -> tuple[str, ...]:
        errors = []
        with self._lock:
            for key, manifest in self._manifests.items():
                if key != (manifest.strategy_id, manifest.version):
                    errors.append(f"manifest key mismatch: {key}")
            for transition in self._transitions:
                key = transition.strategy_id, transition.version
                if key not in self._manifests:
                    errors.append(f"transition without manifest: {key}")
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    strategy_id: str
    timestamp: str
    shadow_signal: float
    reference_signal: float
    absolute_difference: float
    threshold_breached: bool


class ShadowComparator:
    """Compare live paper candidates with a stable reference strategy."""
    def __init__(self, *, max_abs_difference: float = 0.25) -> None:
        if max_abs_difference < 0:
            raise ValueError("threshold cannot be negative")
        self.threshold = max_abs_difference
        self._observations: list[ShadowObservation] = []
        self._lock = RLock()

    def compare(self, strategy_id: str, shadow_signal: float, reference_signal: float,
                timestamp: datetime) -> ShadowObservation:
        from math import isfinite
        if not isfinite(shadow_signal) or not isfinite(reference_signal):
            raise ValueError("signals must be finite")
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        diff = abs(shadow_signal - reference_signal)
        record = ShadowObservation(strategy_id, timestamp.isoformat(), shadow_signal,
                                   reference_signal, diff, diff > self.threshold)
        with self._lock:
            self._observations.append(record)
        return record

    def observations(self, strategy_id: str | None = None) -> tuple[ShadowObservation, ...]:
        with self._lock:
            rows = tuple(self._observations)
        return tuple(row for row in rows if strategy_id is None or row.strategy_id == strategy_id)

    def summary(self, strategy_id: str) -> dict[str, float | int]:
        rows = self.observations(strategy_id)
        diffs = [row.absolute_difference for row in rows]
        return {"observations": len(rows), "breaches": sum(row.threshold_breached for row in rows),
                "mean_abs_difference": sum(diffs) / len(diffs) if diffs else 0.0,
                "max_abs_difference": max(diffs, default=0.0)}
