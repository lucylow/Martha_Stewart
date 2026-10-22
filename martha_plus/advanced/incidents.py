"""Incident lifecycle tracking for risk, data-quality, and service alerts."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4


class IncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    REOPENED = "reopened"


@dataclass(frozen=True, slots=True)
class IncidentEvent:
    incident_id: str
    kind: str
    occurred_at: str
    actor: str
    note: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    key: str
    title: str
    severity: str
    status: IncidentStatus
    opened_at: str
    updated_at: str
    acknowledged_by: str | None = None
    resolved_by: str | None = None
    resolution_note: str | None = None
    occurrence_count: int = 1
    events: tuple[IncidentEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class IncidentManager:
    """In-memory incident registry; replace with durable storage for multi-process use."""
    def __init__(self, *, max_events_per_incident: int = 500,
                 resolved_key_ttl_seconds: float = 0.0) -> None:
        if max_events_per_incident < 1 or resolved_key_ttl_seconds < 0:
            raise ValueError("invalid incident retention settings")
        self.max_events_per_incident = max_events_per_incident
        self.resolved_key_ttl_seconds = resolved_key_ttl_seconds
        self._incidents: dict[str, Incident] = {}
        self._open_by_key: dict[str, str] = {}
        self._lock = RLock()

    def open(self, key: str, title: str, severity: str, *, actor: str = "system",
             note: str = "", metadata: Mapping[str, Any] | None = None) -> Incident:
        key, title, severity = key.strip(), title.strip(), severity.strip().lower()
        if not key or not title or severity not in {"info", "warning", "error", "critical"}:
            raise ValueError("key/title and a known severity are required")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing_id = self._open_by_key.get(key)
            if existing_id:
                existing = self._incidents[existing_id]
                event = IncidentEvent(existing_id, "occurrence", now, actor, note, dict(metadata or {}))
                incident = Incident(existing.incident_id, existing.key, existing.title,
                                    max((existing.severity, severity), key=lambda x: {"info": 0, "warning": 1, "error": 2, "critical": 3}[x]),
                                    existing.status, existing.opened_at, now, existing.acknowledged_by,
                                    existing.resolved_by, existing.resolution_note, existing.occurrence_count + 1,
                                    (existing.events + (event,))[-self.max_events_per_incident:],
                                    {**existing.metadata, **dict(metadata or {})})
                self._incidents[existing_id] = incident
                return incident
            incident_id = uuid4().hex
            event = IncidentEvent(incident_id, "opened", now, actor, note, dict(metadata or {}))
            incident = Incident(incident_id, key, title, severity, IncidentStatus.OPEN,
                                now, now, occurrence_count=1, events=(event,), metadata=dict(metadata or {}))
            self._incidents[incident_id] = incident
            self._open_by_key[key] = incident_id
            return incident

    def acknowledge(self, incident_id: str, actor: str, note: str = "") -> Incident:
        return self._update(incident_id, "acknowledged", actor, note)

    def resolve(self, incident_id: str, actor: str, note: str) -> Incident:
        if not note.strip():
            raise ValueError("resolution note is required")
        return self._update(incident_id, "resolved", actor, note)

    def reopen(self, incident_id: str, actor: str, note: str = "") -> Incident:
        return self._update(incident_id, "reopened", actor, note)

    def _update(self, incident_id: str, action: str, actor: str, note: str) -> Incident:
        if not actor.strip():
            raise ValueError("actor is required")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if incident_id not in self._incidents:
                raise KeyError(incident_id)
            old = self._incidents[incident_id]
            transitions = {"acknowledged": {IncidentStatus.OPEN, IncidentStatus.REOPENED},
                           "resolved": {IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED, IncidentStatus.REOPENED},
                           "reopened": {IncidentStatus.RESOLVED}}
            if old.status not in transitions[action]:
                raise ValueError(f"cannot {action} incident in state {old.status.value}")
            status = {"acknowledged": IncidentStatus.ACKNOWLEDGED,
                      "resolved": IncidentStatus.RESOLVED,
                      "reopened": IncidentStatus.REOPENED}[action]
            event = IncidentEvent(incident_id, action, now, actor, note)
            new = Incident(old.incident_id, old.key, old.title, old.severity, status,
                           old.opened_at, now,
                           actor if action == "acknowledged" else old.acknowledged_by,
                           actor if action == "resolved" else (None if action == "reopened" else old.resolved_by),
                           note if action == "resolved" else (None if action == "reopened" else old.resolution_note),
                           old.occurrence_count, (old.events + (event,))[-self.max_events_per_incident:], old.metadata)
            self._incidents[incident_id] = new
            if status is IncidentStatus.RESOLVED:
                self._open_by_key.pop(old.key, None)
            else:
                self._open_by_key[old.key] = incident_id
            return new

    def get(self, incident_id: str) -> Incident:
        with self._lock:
            if incident_id not in self._incidents:
                raise KeyError(incident_id)
            return self._incidents[incident_id]

    def list(self, *, status: IncidentStatus | None = None, severity: str | None = None,
             limit: int = 100) -> list[Incident]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._lock:
            items = list(self._incidents.values())
        if status is not None:
            items = [item for item in items if item.status is status]
        if severity is not None:
            items = [item for item in items if item.severity == severity]
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items[:limit]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._incidents.values())
        by_status = {state.value: sum(item.status is state for item in items) for state in IncidentStatus}
        by_severity = {level: sum(item.severity == level and item.status is not IncidentStatus.RESOLVED for item in items)
                       for level in ("info", "warning", "error", "critical")}
        return {"total": len(items), "by_status": by_status, "unresolved_by_severity": by_severity,
                "open_count": sum(item.status is not IncidentStatus.RESOLVED for item in items)}
