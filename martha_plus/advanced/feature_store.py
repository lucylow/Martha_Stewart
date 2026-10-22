"""Local, immutable feature partitions with schema and lineage metadata."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .features import FeatureRow


@dataclass(frozen=True, slots=True)
class FeaturePartition:
    name: str
    row_count: int
    schema: tuple[str, ...]
    checksum: str
    created_at: str
    source_fingerprint: str
    schema_version: int


class FeatureStoreError(RuntimeError):
    pass


class LocalFeatureStore:
    """Store JSONL feature rows in immutable partition files.

    A final metadata file is the commit marker. Data is first written to a temp
    file and atomically renamed. A partition name may never be overwritten unless
    `replace=True` is explicitly requested by research code.
    """
    def __init__(self, root: str | Path, *, schema_version: int = 1) -> None:
        if schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        self.root = Path(root)
        self.schema_version = schema_version
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
            raise ValueError("partition name may only use letters, digits, hyphen, underscore")
        return name

    def write(self, name: str, rows: Sequence[FeatureRow | Mapping[str, Any]], *,
              source_fingerprint: str, replace: bool = False) -> FeaturePartition:
        partition = self._safe_name(name)
        if not source_fingerprint.strip():
            raise ValueError("source_fingerprint is required")
        data_path = self.root / f"{partition}.jsonl"
        meta_path = self.root / f"{partition}.manifest.json"
        if not replace and (data_path.exists() or meta_path.exists()):
            raise FileExistsError(f"feature partition already exists: {partition}")
        records: list[dict[str, Any]] = []
        columns: set[str] = set()
        for row in rows:
            if isinstance(row, FeatureRow):
                record = row.as_dict()
            elif isinstance(row, Mapping):
                record = dict(row)
            else:
                raise TypeError("feature rows must be FeatureRow or mapping values")
            columns.update(record)
            records.append(record)
        schema = tuple(sorted(columns))
        canonical_lines = [json.dumps(self._clean(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
                           for row in records]
        payload = ("\n".join(canonical_lines) + ("\n" if canonical_lines else "")).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        info = FeaturePartition(partition, len(records), schema, checksum, created_at,
                                source_fingerprint, self.schema_version)
        meta = {"name": info.name, "row_count": info.row_count, "schema": list(info.schema),
                "checksum": info.checksum, "created_at": info.created_at,
                "source_fingerprint": info.source_fingerprint, "schema_version": info.schema_version}
        self._atomic_bytes(data_path, payload)
        try:
            self._atomic_bytes(meta_path, (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode())
        except Exception:
            # A missing manifest means the partition is not committed; remove the data file.
            try: data_path.unlink()
            except OSError: pass
            raise
        return info

    def read(self, name: str, *, expected_schema_version: int | None = None) -> tuple[list[dict[str, Any]], FeaturePartition]:
        partition = self._safe_name(name)
        data_path = self.root / f"{partition}.jsonl"
        meta_path = self.root / f"{partition}.manifest.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            payload = data_path.read_bytes()
        except FileNotFoundError as exc:
            raise FeatureStoreError(f"feature partition not found or uncommitted: {partition}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise FeatureStoreError(f"cannot read feature partition: {partition}") from exc
        version = meta.get("schema_version")
        expected = self.schema_version if expected_schema_version is None else expected_schema_version
        if version != expected:
            raise FeatureStoreError(f"schema version mismatch: expected {expected}, got {version}")
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != meta.get("checksum"):
            raise FeatureStoreError("feature partition checksum mismatch")
        try:
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FeatureStoreError("feature partition contains invalid JSONL") from exc
        if len(rows) != meta.get("row_count"):
            raise FeatureStoreError("feature row count does not match manifest")
        schema = tuple(meta.get("schema", []))
        if tuple(sorted({k for row in rows for k in row})) != schema:
            raise FeatureStoreError("feature schema does not match manifest")
        info = FeaturePartition(partition, len(rows), schema, checksum, meta["created_at"],
                                meta["source_fingerprint"], version)
        return rows, info

    def list(self) -> list[FeaturePartition | dict[str, str]]:
        result: list[FeaturePartition | dict[str, str]] = []
        for meta_path in sorted(self.root.glob("*.manifest.json")):
            name = meta_path.name[:-len(".manifest.json")]
            try:
                rows, info = self.read(name)
                result.append(info)
            except Exception as exc:
                result.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        return result

    @staticmethod
    def _clean(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if not __import__("math").isfinite(value):
                raise ValueError("feature values must be finite")
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("feature timestamps must be timezone-aware")
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(k): LocalFeatureStore._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LocalFeatureStore._clean(v) for v in value]
        if hasattr(value, "value"):
            return LocalFeatureStore._clean(value.value)
        return str(value)

    @staticmethod
    def _atomic_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            try: os.unlink(temp_name)
            except OSError: pass
            raise
