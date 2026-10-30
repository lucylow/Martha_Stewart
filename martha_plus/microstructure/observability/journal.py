"""Tamper-evident append-only journal for local simulation evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


GENESIS = "0" * 64


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize JSON-compatible evidence deterministically."""
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"journal payload is not canonical JSON: {exc}") from exc


@dataclass(frozen=True, slots=True)
class JournalEntry:
    sequence: int
    timestamp_ns: int
    category: str
    payload: dict[str, Any]
    previous_hash: str
    entry_hash: str


@dataclass(frozen=True, slots=True)
class JournalVerification:
    valid: bool
    checked_entries: int
    failed_sequence: int | None
    reason: str = ""


class HashChainJournal:
    """Hash-chain event journal with monotonic sequence and event timestamps.

    A hash chain makes accidental edits detectable when the trusted final hash
    is retained separately.  It is not a digital signature and does not prove
    who created an entry.  Persisted journal bytes should be backed up and
    access-controlled by the caller.
    """

    def __init__(self, *, run_id: str, created_at_ns: int = 0) -> None:
        if not run_id.strip() or created_at_ns < 0:
            raise ValueError("invalid journal identity")
        self.run_id = run_id
        self.created_at_ns = created_at_ns
        self._entries: list[JournalEntry] = []
        self._last_timestamp_ns = created_at_ns
        self._last_hash = GENESIS

    @property
    def entries(self) -> tuple[JournalEntry, ...]:
        return tuple(self._entries)

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def append(self, timestamp_ns: int, category: str,
               payload: Mapping[str, Any]) -> JournalEntry:
        if timestamp_ns < self._last_timestamp_ns:
            raise ValueError("journal timestamps must be non-decreasing")
        if not category.strip():
            raise ValueError("category must not be blank")
        # Serialize and parse once to detach nested caller-owned containers and
        # reject unsupported objects before they enter the audit chain.
        body = json.loads(canonical_json(dict(payload)))
        envelope = {
            "run_id": self.run_id,
            "sequence": len(self._entries),
            "timestamp_ns": timestamp_ns,
            "category": category,
            "payload": body,
            "previous_hash": self._last_hash,
        }
        digest = hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
        row = JournalEntry(len(self._entries), timestamp_ns, category, body,
                           self._last_hash, digest)
        self._entries.append(row)
        self._last_timestamp_ns = timestamp_ns
        self._last_hash = digest
        return row

    def append_system_event(self, category: str, payload: Mapping[str, Any]) -> JournalEntry:
        """Convenience timestamp based on wall time (avoid in deterministic tests)."""
        timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        return self.append(max(timestamp_ns, self._last_timestamp_ns), category, payload)

    def verify(self, *, expected_final_hash: str | None = None) -> JournalVerification:
        previous = GENESIS
        previous_ts = self.created_at_ns
        for index, row in enumerate(self._entries):
            if row.sequence != index:
                return JournalVerification(False, index, index, "sequence discontinuity")
            if row.timestamp_ns < previous_ts:
                return JournalVerification(False, index, index, "timestamp regression")
            envelope = {
                "run_id": self.run_id,
                "sequence": row.sequence,
                "timestamp_ns": row.timestamp_ns,
                "category": row.category,
                "payload": row.payload,
                "previous_hash": previous,
            }
            try:
                expected = hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
            except ValueError as exc:
                return JournalVerification(False, index, index, str(exc))
            if row.previous_hash != previous:
                return JournalVerification(False, index, index, "previous hash mismatch")
            if row.entry_hash != expected:
                return JournalVerification(False, index, index, "entry hash mismatch")
            previous = row.entry_hash
            previous_ts = row.timestamp_ns
        if expected_final_hash is not None and previous != expected_final_hash:
            return JournalVerification(False, len(self._entries), None, "final hash mismatch")
        return JournalVerification(True, len(self._entries), None)

    def to_jsonl(self) -> str:
        header = {"type": "journal", "version": 1, "run_id": self.run_id,
                  "created_at_ns": self.created_at_ns}
        lines = [canonical_json(header)]
        for row in self._entries:
            lines.append(canonical_json({
                "type": "entry", "sequence": row.sequence,
                "timestamp_ns": row.timestamp_ns, "category": row.category,
                "payload": row.payload, "previous_hash": row.previous_hash,
                "entry_hash": row.entry_hash,
            }))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_jsonl(cls, text: str) -> "HashChainJournal":
        lines = text.splitlines()
        if not lines:
            raise ValueError("journal is empty")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid journal header") from exc
        if header.get("type") != "journal" or header.get("version") != 1:
            raise ValueError("unsupported journal format")
        journal = cls(run_id=str(header["run_id"]),
                      created_at_ns=int(header["created_at_ns"]))
        for line_number, line in enumerate(lines[1:], start=2):
            try:
                payload = json.loads(line)
                entry = JournalEntry(int(payload["sequence"]), int(payload["timestamp_ns"]),
                                     str(payload["category"]), dict(payload["payload"]),
                                     str(payload["previous_hash"]), str(payload["entry_hash"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid journal row at line {line_number}") from exc
            journal._entries.append(entry)
            journal._last_timestamp_ns = max(journal._last_timestamp_ns, entry.timestamp_ns)
            journal._last_hash = entry.entry_hash
        verification = journal.verify()
        if not verification.valid:
            raise ValueError(f"journal integrity check failed: {verification.reason}")
        return journal


class AuditTrail:
    """Small semantic facade that standardizes common operational categories."""

    ALLOWED = {"market_event", "order_request", "order_ack", "execution",
               "risk_decision", "configuration", "incident", "checkpoint"}

    def __init__(self, journal: HashChainJournal) -> None:
        self.journal = journal

    def record(self, timestamp_ns: int, category: str, payload: Mapping[str, Any]) -> JournalEntry:
        if category not in self.ALLOWED:
            raise ValueError(f"unsupported audit category: {category}")
        return self.journal.append(timestamp_ns, category, payload)

    def record_order(self, timestamp_ns: int, order_id: str, action: str,
                     quantity: int, *, symbol: str, strategy_id: str) -> JournalEntry:
        if quantity < 0 or not order_id.strip() or not symbol.strip():
            raise ValueError("invalid order audit fields")
        return self.record(timestamp_ns, "order_request", {
            "order_id": order_id, "action": action, "quantity": quantity,
            "symbol": symbol.upper(), "strategy_id": strategy_id,
        })

    def record_risk(self, timestamp_ns: int, order_id: str, allowed: bool,
                    codes: Iterable[str]) -> JournalEntry:
        return self.record(timestamp_ns, "risk_decision", {
            "order_id": order_id, "allowed": bool(allowed), "codes": list(codes),
        })
