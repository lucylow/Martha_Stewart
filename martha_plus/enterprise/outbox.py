"""SQLite transactional outbox with leases, retries, and dead-letter handling.

Use the same SQLite connection/transaction as domain changes where possible.
The outbox does not itself deliver to a remote service; a worker polls leased rows.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class OutboxEvent:
    event_id: str
    topic: str
    payload: Any
    created_at: str
    attempts: int
    status: str
    available_at: float
    lease_owner: str | None = None
    lease_until: float | None = None
    last_error: str | None = None


class SqliteOutbox:
    def __init__(self, connection: sqlite3.Connection, *, clock: Callable[[], float],
                 max_attempts: int = 8) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.connection = connection
        self.clock = clock
        self.max_attempts = max_attempts
        self._lock = RLock()
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS outbox_events (
            event_id TEXT PRIMARY KEY, topic TEXT NOT NULL, payload TEXT NOT NULL,
            created_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL, available_at REAL NOT NULL,
            lease_owner TEXT, lease_until REAL, last_error TEXT
        )""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_outbox_ready ON outbox_events(status, available_at)")
        self.connection.commit()

    def enqueue(self, topic: str, payload: Any, *, event_id: str | None = None,
                available_at: float | None = None) -> str:
        if not topic.strip():
            raise ValueError("topic is required")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ident = event_id or str(uuid.uuid4())
        now = self.clock()
        with self._lock:
            self.connection.execute("""INSERT INTO outbox_events
                (event_id, topic, payload, created_at, status, available_at)
                VALUES (?, ?, ?, ?, 'pending', ?)""",
                (ident, topic, encoded, datetime.now(timezone.utc).isoformat(),
                 now if available_at is None else available_at))
            self.connection.commit()
        return ident

    def lease(self, owner: str, *, limit: int = 50, lease_seconds: float = 30.0) -> tuple[OutboxEvent, ...]:
        if not owner.strip() or limit < 1 or lease_seconds <= 0:
            raise ValueError("owner, positive limit, and lease duration are required")
        now = self.clock()
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute("""UPDATE outbox_events SET status='pending', lease_owner=NULL,
                    lease_until=NULL WHERE status='leased' AND lease_until <= ?""", (now,))
                rows = self.connection.execute("""SELECT * FROM outbox_events
                    WHERE status='pending' AND available_at <= ? AND attempts < ?
                    ORDER BY created_at, event_id LIMIT ?""", (now, self.max_attempts, limit)).fetchall()
                ids = [row["event_id"] for row in rows]
                for ident in ids:
                    self.connection.execute("""UPDATE outbox_events SET status='leased', lease_owner=?,
                        lease_until=? WHERE event_id=? AND status='pending'""",
                        (owner, now + lease_seconds, ident))
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        with self._lock:
            return tuple(self._decode(self._row(ident)) for ident in ids)

    def _row(self, event_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM outbox_events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return row

    @staticmethod
    def _decode(row: sqlite3.Row) -> OutboxEvent:
        return OutboxEvent(row["event_id"], row["topic"], json.loads(row["payload"]),
                           row["created_at"], row["attempts"], row["status"],
                           row["available_at"], row["lease_owner"], row["lease_until"],
                           row["last_error"])

    def acknowledge(self, event_id: str, owner: str) -> None:
        with self._lock:
            cur = self.connection.execute("""UPDATE outbox_events SET status='sent', lease_owner=NULL,
                lease_until=NULL, last_error=NULL WHERE event_id=? AND status='leased' AND lease_owner=?""",
                (event_id, owner))
            if cur.rowcount != 1:
                self.connection.rollback()
                raise RuntimeError("event is not leased by this owner")
            self.connection.commit()

    def fail(self, event_id: str, owner: str, error: str, *, retry_delay: float = 5.0) -> str:
        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")
        with self._lock:
            row = self._row(event_id)
            if row["status"] != "leased" or row["lease_owner"] != owner:
                raise RuntimeError("event is not leased by this owner")
            attempts = row["attempts"] + 1
            dead = attempts >= self.max_attempts
            status = "dead" if dead else "pending"
            self.connection.execute("""UPDATE outbox_events SET attempts=?, status=?, available_at=?,
                lease_owner=NULL, lease_until=NULL, last_error=? WHERE event_id=?""",
                (attempts, status, self.clock() + retry_delay, error[:1000], event_id))
            self.connection.commit()
            return status

    def get(self, event_id: str) -> OutboxEvent:
        with self._lock:
            return self._decode(self._row(event_id))

    def list_events(self, *, status: str | None = None, limit: int = 100) -> tuple[OutboxEvent, ...]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        query = "SELECT * FROM outbox_events"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at, event_id LIMIT ?"
        params += (limit,)
        with self._lock:
            rows = self.connection.execute(query, params).fetchall()
            return tuple(self._decode(row) for row in rows)

    def purge_sent(self, *, before_iso: str) -> int:
        with self._lock:
            cur = self.connection.execute("DELETE FROM outbox_events WHERE status='sent' AND created_at < ?",
                                          (before_iso,))
            self.connection.commit()
            return cur.rowcount

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.connection.execute("SELECT status, COUNT(*) AS n FROM outbox_events GROUP BY status").fetchall()
            return {row["status"]: row["n"] for row in rows}
