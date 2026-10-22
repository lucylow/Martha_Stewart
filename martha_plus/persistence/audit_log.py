"""Append-only SHA-256 hash-chained audit file.

The chain makes accidental or unsophisticated modification detectable. It is not
proof against an attacker who can rewrite the entire file and this code; export
and periodically anchor the final hash in separately controlled storage.
"""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

GENESIS_HASH = "0" * 64


class AuditLogError(Exception):
    """Raised when an audit log cannot be written or is malformed."""

class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @staticmethod
    def _canonical(payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          default=lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value))

    @classmethod
    def _hash_entry(cls, body: Mapping[str, Any]) -> str:
        return hashlib.sha256(cls._canonical(body).encode("utf-8")).hexdigest()

    def _last_hash_and_sequence(self) -> tuple[str, int]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS_HASH, 0
        verification = self.verify()
        if not verification["valid"]:
            raise AuditLogError(f"cannot append to invalid audit log at line {verification['first_bad_line']}")
        with self.path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        last = json.loads(lines[-1])
        return last["hash"], int(last["sequence"])

    def append(self, event_type: str, payload: Mapping[str, Any],
               timestamp: datetime | None = None) -> dict[str, Any]:
        if not event_type or not event_type.strip():
            raise ValueError("event_type is required")
        stamp = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._lock:
            previous_hash, sequence = self._last_hash_and_sequence()
            body = {"sequence": sequence + 1, "timestamp": stamp.isoformat(),
                    "event_type": event_type.strip(), "payload": dict(payload),
                    "previous_hash": previous_hash}
            entry = dict(body)
            entry["hash"] = self._hash_entry(body)
            line = self._canonical(entry) + "\n"
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise AuditLogError(f"unable to append audit entry: {exc}") from exc
            return entry

    def verify(self) -> dict[str, Any]:
        previous_hash = GENESIS_HASH
        expected_sequence = 1
        count = 0
        if not self.path.exists():
            return {"valid": True, "entries": 0, "final_hash": GENESIS_HASH,
                    "first_bad_line": None, "reason": "empty"}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.strip():
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        return {"valid": False, "entries": count, "final_hash": previous_hash,
                                "first_bad_line": line_number, "reason": "invalid_json"}
                    expected_hash = entry.get("hash")
                    body = {key: value for key, value in entry.items() if key != "hash"}
                    if entry.get("sequence") != expected_sequence:
                        return {"valid": False, "entries": count, "final_hash": previous_hash,
                                "first_bad_line": line_number, "reason": "sequence_mismatch"}
                    if entry.get("previous_hash") != previous_hash:
                        return {"valid": False, "entries": count, "final_hash": previous_hash,
                                "first_bad_line": line_number, "reason": "previous_hash_mismatch"}
                    actual_hash = self._hash_entry(body)
                    if actual_hash != expected_hash:
                        return {"valid": False, "entries": count, "final_hash": previous_hash,
                                "first_bad_line": line_number, "reason": "hash_mismatch"}
                    previous_hash = actual_hash
                    expected_sequence += 1
                    count += 1
        except OSError as exc:
            raise AuditLogError(f"unable to read audit log: {exc}") from exc
        return {"valid": True, "entries": count, "final_hash": previous_hash,
                "first_bad_line": None, "reason": "ok"}
