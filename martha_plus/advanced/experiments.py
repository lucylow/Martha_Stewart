"""SQLite-backed experiment registry with immutable parameter fingerprints."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def fingerprint(parameters: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(parameters).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    run_id: str
    name: str
    status: str
    parameters: Mapping[str, Any]
    metrics: Mapping[str, float]
    code_version: str
    data_fingerprint: str
    created_at: str
    completed_at: str | None
    error: str | None = None


class ExperimentRegistry:
    """Persist experiment runs atomically; never silently overwrite an existing run."""
    STATUSES = {"created", "running", "completed", "failed", "cancelled"}
    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.database, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        if self.database != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                parameters_json TEXT NOT NULL, parameter_hash TEXT NOT NULL,
                metrics_json TEXT NOT NULL, code_version TEXT NOT NULL,
                data_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL,
                completed_at TEXT, error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_experiment_name_created
                ON experiment_runs(name, created_at);
            CREATE INDEX IF NOT EXISTS idx_experiment_param_hash
                ON experiment_runs(name, parameter_hash);
            CREATE TABLE IF NOT EXISTS experiment_tags (
                run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                tag TEXT NOT NULL, PRIMARY KEY(run_id, tag)
            );
        """)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, name: str, parameters: Mapping[str, Any], *, code_version: str = "unknown",
               data_fingerprint: str = "unknown", tags: Sequence[str] = (),
               run_id: str | None = None) -> ExperimentRun:
        if not name.strip():
            raise ValueError("experiment name is required")
        params_json = canonical_json(parameters)
        run_id = run_id or uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("""INSERT INTO experiment_runs
                    (run_id,name,status,parameters_json,parameter_hash,metrics_json,code_version,
                     data_fingerprint,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (run_id, name, "created", params_json, fingerprint(parameters), "{}",
                     code_version, data_fingerprint, timestamp))
                self._conn.executemany("INSERT INTO experiment_tags(run_id,tag) VALUES (?,?)",
                                       [(run_id, str(tag)) for tag in sorted(set(tags))])
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        result = self.get(run_id)
        assert result is not None
        return result

    def start(self, run_id: str) -> ExperimentRun:
        return self._transition(run_id, {"created"}, "running")

    def complete(self, run_id: str, metrics: Mapping[str, float]) -> ExperimentRun:
        normalized = {str(k): float(v) for k, v in metrics.items()}
        if any(not isfinite(v) for v in normalized.values()):
            raise ValueError("metrics must be finite; encode missing metric values as omitted keys")
        return self._finish(run_id, {"running", "created"}, "completed", canonical_json(normalized), None)

    def fail(self, run_id: str, error: str) -> ExperimentRun:
        if not error.strip():
            raise ValueError("error description is required")
        return self._finish(run_id, {"running", "created"}, "failed", "{}", error[:4000])

    def cancel(self, run_id: str, reason: str = "cancelled by user") -> ExperimentRun:
        return self._finish(run_id, {"created", "running"}, "cancelled", "{}", reason[:4000])

    def _transition(self, run_id: str, allowed: set[str], new_status: str) -> ExperimentRun:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT status FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if row["status"] not in allowed:
                    raise ValueError(f"cannot transition {row['status']} to {new_status}")
                self._conn.execute("UPDATE experiment_runs SET status=? WHERE run_id=?", (new_status, run_id))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        result = self.get(run_id)
        assert result is not None
        return result

    def _finish(self, run_id: str, allowed: set[str], new_status: str, metrics_json: str,
                error: str | None) -> ExperimentRun:
        completed_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT status FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if row["status"] not in allowed:
                    raise ValueError(f"cannot transition {row['status']} to {new_status}")
                self._conn.execute("UPDATE experiment_runs SET status=?,metrics_json=?,completed_at=?,error=? WHERE run_id=?",
                                    (new_status, metrics_json, completed_at, error, run_id))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        result = self.get(run_id)
        assert result is not None
        return result

    def get(self, run_id: str) -> ExperimentRun | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            return self._decode(row)

    def list(self, *, name: str | None = None, status: str | None = None,
             tag: str | None = None, limit: int = 100, offset: int = 0) -> list[ExperimentRun]:
        if limit < 1 or limit > 10_000 or offset < 0:
            raise ValueError("invalid pagination")
        if status is not None and status not in self.STATUSES:
            raise ValueError("unknown status")
        clauses: list[str] = []
        params: list[Any] = []
        if name:
            clauses.append("r.name=?"); params.append(name)
        if status:
            clauses.append("r.status=?"); params.append(status)
        if tag:
            clauses.append("EXISTS (SELECT 1 FROM experiment_tags t WHERE t.run_id=r.run_id AND t.tag=?)"); params.append(tag)
        sql = "SELECT r.* FROM experiment_runs r" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        sql += " ORDER BY r.created_at DESC, r.run_id LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        with self._lock:
            return [self._decode(row) for row in self._conn.execute(sql, params).fetchall()]

    def compare(self, run_ids: Sequence[str], metric: str, *, descending: bool = True) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for run_id in run_ids:
            run = self.get(run_id)
            if run is None:
                raise KeyError(run_id)
            value = run.metrics.get(metric)
            result.append({"run_id": run_id, "name": run.name, "status": run.status,
                           "metric": metric, "value": value})
        return sorted(result, key=lambda row: (row["value"] is not None,
                                               row["value"] if row["value"] is not None else 0),
                      reverse=descending)

    def tags(self, run_id: str) -> tuple[str, ...]:
        with self._lock:
            row = self._conn.execute("SELECT run_id FROM experiment_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return tuple(r[0] for r in self._conn.execute(
                "SELECT tag FROM experiment_tags WHERE run_id=? ORDER BY tag", (run_id,)).fetchall())

    def _decode(self, row: sqlite3.Row) -> ExperimentRun:
        return ExperimentRun(row["run_id"], row["name"], row["status"],
                             json.loads(row["parameters_json"]), json.loads(row["metrics_json"]),
                             row["code_version"], row["data_fingerprint"], row["created_at"],
                             row["completed_at"], row["error"])
