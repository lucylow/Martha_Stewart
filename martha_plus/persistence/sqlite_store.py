"""Dependency-free SQLite persistence with schema versioning and safe serialization."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from uuid import uuid4
from typing import Any, Iterator, Mapping
from ..errors import StorageError
from ..types import Fill, Order, OrderStatus

SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")

class SQLiteStore:
    """Small transactional store. Use a single instance per process where possible."""
    def __init__(self, path: str | Path, timeout_seconds: float = 5.0):
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self._lock = RLock()
        self._memory = str(path) == ":memory:"
        self._dsn = f"file:martha_plus_{uuid4().hex}?mode=memory&cache=shared" if self._memory else str(self.path)
        self._anchor: sqlite3.Connection | None = None
        if self._memory:
            # Keep one shared-memory connection alive for the store lifetime.
            self._anchor = self._connect()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._dsn, timeout=self.timeout_seconds,
                                     isolation_level=None, check_same_thread=False, uri=self._memory)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.execute("COMMIT")
            except Exception as exc:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if isinstance(exc, (ValueError, TypeError, KeyError)):
                    raise
                raise StorageError(f"database transaction failed: {exc}") from exc
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.executescript("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO schema_meta(singleton, version) VALUES(1, 1);
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_orders_symbol_created ON orders(symbol, created_at);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL CHECK(quantity > 0),
                    price REAL NOT NULL CHECK(price > 0),
                    commission REAL NOT NULL CHECK(commission >= 0),
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES orders(order_id) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_fills_symbol_timestamp ON fills(symbol, timestamp);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, timestamp);
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    gross_exposure REAL NOT NULL,
                    net_exposure REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """)
                version = connection.execute("SELECT version FROM schema_meta WHERE singleton=1").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise StorageError(f"database schema {version} is newer than supported {SCHEMA_VERSION}")
            finally:
                connection.close()

    def save_order(self, order: Order) -> None:
        request = order.request
        payload = json.dumps(order, default=_json_default, sort_keys=True)
        created = request.created_at.isoformat()
        updated = order.updated_at.isoformat()
        with self.transaction() as conn:
            row = conn.execute("SELECT order_id, client_order_id FROM orders WHERE client_order_id=?",
                               (request.client_order_id,)).fetchone()
            if row and row["order_id"] != order.order_id:
                # This signals an idempotency conflict; it should not be silently replaced.
                raise StorageError("client_order_id already maps to a different order_id")
            conn.execute("""INSERT INTO orders(order_id,client_order_id,status,symbol,payload_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(order_id) DO UPDATE SET
                status=excluded.status,payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
                (order.order_id, request.client_order_id, order.status.value, request.symbol,
                 payload, created, updated))

    def save_fill(self, fill: Fill) -> None:
        with self.transaction() as conn:
            conn.execute("""INSERT OR IGNORE INTO fills(fill_id,order_id,symbol,side,quantity,price,commission,timestamp,payload_json)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (fill.fill_id, fill.order_id, fill.symbol, fill.side.value, fill.quantity,
                 fill.price, fill.commission, fill.timestamp.isoformat(),
                 json.dumps(fill, default=_json_default, sort_keys=True)))

    def append_event(self, event_type: str, payload: Mapping[str, Any],
                     timestamp: datetime | None = None) -> None:
        if not event_type.strip():
            raise ValueError("event_type is required")
        stamp = (timestamp or datetime.now(timezone.utc)).isoformat()
        serialized = json.dumps(payload, default=_json_default, sort_keys=True)
        with self.transaction() as conn:
            conn.execute("INSERT INTO events(event_type,timestamp,payload_json) VALUES(?,?,?)",
                         (event_type, stamp, serialized))

    def save_snapshot(self, snapshot: Any) -> None:
        payload = json.dumps(snapshot, default=_json_default, sort_keys=True)
        with self.transaction() as conn:
            conn.execute("""INSERT INTO equity_snapshots(timestamp,cash,equity,gross_exposure,net_exposure,daily_pnl,payload_json)
                VALUES(?,?,?,?,?,?,?)""",
                (snapshot.timestamp.isoformat(), snapshot.cash, snapshot.equity,
                 snapshot.gross_exposure, snapshot.net_exposure, snapshot.daily_pnl, payload))

    def order_count(self, status: str | None = None) -> int:
        connection = self._connect()
        try:
            if status is None:
                row = connection.execute("SELECT COUNT(*) AS n FROM orders").fetchone()
            else:
                row = connection.execute("SELECT COUNT(*) AS n FROM orders WHERE status=?", (status,)).fetchone()
            return int(row["n"])
        finally:
            connection.close()

    def list_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        connection = self._connect()
        try:
            if event_type:
                rows = connection.execute("SELECT * FROM events WHERE event_type=? ORDER BY event_id DESC LIMIT ?",
                                          (event_type, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (limit,)).fetchall()
        finally:
            connection.close()
        return [{"event_id": r["event_id"], "event_type": r["event_type"],
                 "timestamp": r["timestamp"], "payload": json.loads(r["payload_json"])} for r in rows]

    def list_recent_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        finally:
            connection.close()
        return [{"order_id": r["order_id"], "client_order_id": r["client_order_id"],
                 "status": r["status"], "symbol": r["symbol"],
                 "created_at": r["created_at"], "updated_at": r["updated_at"],
                 "payload": json.loads(r["payload_json"])} for r in rows]

    def count_fills(self) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0])
        finally:
            connection.close()

    def checkpoint(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
        finally:
            connection.close()

    def close(self) -> None:
        """Release the anchor connection used for shared in-memory databases."""
        with self._lock:
            if self._anchor is not None:
                self._anchor.close()
                self._anchor = None
