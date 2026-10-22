"""Deterministic event replay and bounded batch iteration for debugging pipelines."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from .eventbus import EventEnvelope


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    sequence: int
    event: EventEnvelope
    payload_hash: str


@dataclass(frozen=True, slots=True)
class ReplayCheckpoint:
    last_sequence: int
    last_event_id: str | None
    state_hash: str
    created_at: str


class EventJournal:
    """Append-only in-memory journal with explicit monotonic sequence numbers."""
    def __init__(self, max_records: int = 100_000) -> None:
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        self.max_records = max_records
        self._records: list[ReplayRecord] = []
        self._next_sequence = 1

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)

    def append(self, event: EventEnvelope) -> ReplayRecord:
        payload_hash = sha256(self._canonical(dict(event.payload)).encode()).hexdigest()
        record = ReplayRecord(self._next_sequence, event, payload_hash)
        self._next_sequence += 1
        self._records.append(record)
        self._records = self._records[-self.max_records:]
        return record

    def range(self, *, after_sequence: int = 0, before_sequence: int | None = None,
              event_type: str | None = None, limit: int = 10_000) -> list[ReplayRecord]:
        if after_sequence < 0 or (before_sequence is not None and before_sequence <= after_sequence):
            raise ValueError("invalid sequence range")
        if not 1 <= limit <= 100_000:
            raise ValueError("limit must be between 1 and 100000")
        result = [record for record in self._records if record.sequence > after_sequence
                  and (before_sequence is None or record.sequence < before_sequence)
                  and (event_type is None or record.event.event_type == event_type)]
        return result[:limit]

    def verify(self) -> dict[str, Any]:
        errors: list[str] = []
        previous = 0
        for record in self._records:
            if record.sequence <= previous:
                errors.append(f"sequence not increasing at {record.sequence}")
            previous = record.sequence
            digest = sha256(self._canonical(dict(record.event.payload)).encode()).hexdigest()
            if digest != record.payload_hash:
                errors.append(f"payload mismatch at {record.sequence}")
        return {"ok": not errors, "count": len(self._records), "errors": errors,
                "first_sequence": self._records[0].sequence if self._records else None,
                "last_sequence": self._records[-1].sequence if self._records else None}

    def __len__(self) -> int:
        return len(self._records)


class ReplayEngine:
    """Replay events through a handler; supports filtering and resumable checkpoints."""
    def __init__(self, journal: EventJournal) -> None:
        self.journal = journal

    def replay(self, handler: Callable[[EventEnvelope], Any], *, after_sequence: int = 0,
               before_sequence: int | None = None, event_types: set[str] | None = None,
               batch_size: int = 1000, fail_fast: bool = True) -> dict[str, Any]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        processed = errors = 0
        error_items: list[dict[str, Any]] = []
        cursor = after_sequence
        while True:
            batch = self.journal.range(after_sequence=cursor, before_sequence=before_sequence,
                                       limit=batch_size)
            if not batch:
                break
            for record in batch:
                cursor = record.sequence
                if event_types is not None and record.event.event_type not in event_types:
                    continue
                try:
                    handler(record.event)
                    processed += 1
                except Exception as exc:
                    errors += 1
                    error_items.append({"sequence": record.sequence, "event_id": record.event.event_id,
                                        "error": f"{type(exc).__name__}: {exc}"})
                    if fail_fast:
                        return {"processed": processed, "errors": errors, "last_sequence": cursor,
                                "failed": True, "error_items": error_items}
            if len(batch) < batch_size:
                break
        return {"processed": processed, "errors": errors,
                "last_sequence": cursor, "failed": False, "error_items": error_items}

    def checkpoint(self, last_sequence: int, state: Mapping[str, Any]) -> ReplayCheckpoint:
        if last_sequence < 0:
            raise ValueError("last_sequence must be non-negative")
        record = next((r for r in self.journal._records if r.sequence == last_sequence), None)
        return ReplayCheckpoint(last_sequence, record.event.event_id if record else None,
                                sha256(EventJournal._canonical(dict(state)).encode()).hexdigest(),
                                datetime.now(timezone.utc).isoformat())

    def verify_checkpoint(self, checkpoint: ReplayCheckpoint, state: Mapping[str, Any]) -> bool:
        digest = sha256(EventJournal._canonical(dict(state)).encode()).hexdigest()
        if digest != checkpoint.state_hash:
            return False
        if checkpoint.last_sequence == 0:
            return checkpoint.last_event_id is None
        record = next((r for r in self.journal._records if r.sequence == checkpoint.last_sequence), None)
        return record is not None and record.event.event_id == checkpoint.last_event_id


def batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield bounded lists from an iterable without materializing the whole source."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def deterministic_event_id(source: str, offset: int, payload: Mapping[str, Any]) -> str:
    if not source.strip() or offset < 0:
        raise ValueError("source and non-negative offset are required")
    raw = EventJournal._canonical({"source": source, "offset": offset, "payload": payload})
    return sha256(raw.encode()).hexdigest()[:32]
