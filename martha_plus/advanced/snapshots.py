"""Atomic JSON checkpoint persistence with integrity verification."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SnapshotInfo:
    path: str
    created_at: str
    version: int
    checksum: str
    size_bytes: int


class SnapshotError(RuntimeError):
    pass


class SnapshotStore:
    """Write/read JSON snapshots atomically and reject corrupt or wrong-version files."""
    def __init__(self, directory: str | Path, *, schema_version: int = 1) -> None:
        if schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        self.directory = Path(directory)
        self.schema_version = schema_version
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, payload: Mapping[str, Any]) -> SnapshotInfo:
        safe_name = self._safe_name(name)
        created_at = datetime.now(timezone.utc).isoformat()
        body = {"schema_version": self.schema_version, "created_at": created_at, "payload": payload}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
        checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        envelope = {**body, "checksum": checksum}
        serialized = json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n"
        target = self.directory / f"{safe_name}.json"
        fd, temp_name = tempfile.mkstemp(prefix=f".{safe_name}-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
            try:
                dir_fd = os.open(self.directory, os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except (AttributeError, OSError):
                pass
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return SnapshotInfo(str(target), created_at, self.schema_version, checksum, len(serialized.encode()))

    def load(self, name: str, *, expected_version: int | None = None) -> tuple[dict[str, Any], SnapshotInfo]:
        path = self.directory / f"{self._safe_name(name)}.json"
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SnapshotError(f"snapshot not found: {name}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"snapshot cannot be read: {name}") from exc
        version = envelope.get("schema_version")
        required = self.schema_version if expected_version is None else expected_version
        if version != required:
            raise SnapshotError(f"snapshot version mismatch: expected {required}, got {version}")
        checksum = envelope.get("checksum")
        body = {key: value for key, value in envelope.items() if key != "checksum"}
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if not isinstance(checksum, str) or checksum != actual:
            raise SnapshotError("snapshot checksum mismatch")
        if not isinstance(body.get("payload"), dict):
            raise SnapshotError("snapshot payload must be an object")
        info = SnapshotInfo(str(path), body["created_at"], version, actual, path.stat().st_size)
        return body["payload"], info

    @staticmethod
    def _safe_name(name: str) -> str:
        value = name.strip()
        if not value or value in {".", ".."} or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in value):
            raise ValueError("snapshot name may contain only letters, digits, hyphens, and underscores")
        return value

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                items.append({"name": path.stem, "schema_version": envelope.get("schema_version"),
                              "created_at": envelope.get("created_at"), "size_bytes": path.stat().st_size})
            except (OSError, json.JSONDecodeError):
                items.append({"name": path.stem, "corrupt": True, "size_bytes": path.stat().st_size})
        return items
