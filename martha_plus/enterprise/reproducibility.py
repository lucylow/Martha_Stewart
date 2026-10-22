"""Reproducibility manifests for research runs, datasets, parameters, and outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import os
import platform
import sys


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    name: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    source_uri: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchManifest:
    run_id: str
    created_at: str
    git_revision: str | None
    python_version: str
    platform: str
    strategy_id: str
    strategy_version: str
    config_digest: str
    dataset_digests: tuple[ArtifactDigest, ...]
    code_digests: tuple[ArtifactDigest, ...]
    result_digests: tuple[ArtifactDigest, ...] = ()
    random_seed: int | None = None
    dependency_lock_digest: str | None = None
    notes: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.run_id, self.strategy_id, self.strategy_version, self.config_digest)):
            raise ValueError("run and strategy identity fields are required")
        if datetime.fromisoformat(self.created_at).tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    def as_dict(self) -> dict[str, Any]:
        def dump(items: Iterable[ArtifactDigest]) -> list[dict[str, Any]]:
            return [{"name": a.name, "sha256": a.sha256, "size_bytes": a.size_bytes,
                     "media_type": a.media_type, "source_uri": a.source_uri} for a in items]
        return {"run_id": self.run_id, "created_at": self.created_at,
            "git_revision": self.git_revision, "python_version": self.python_version,
            "platform": self.platform, "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version, "config_digest": self.config_digest,
            "dataset_digests": dump(self.dataset_digests), "code_digests": dump(self.code_digests),
            "result_digests": dump(self.result_digests), "random_seed": self.random_seed,
            "dependency_lock_digest": self.dependency_lock_digest,
            "notes": self.notes, "metadata": dict(self.metadata)}

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return sha256(encoded).hexdigest()

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict() | {"manifest_digest": self.digest}
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return target


def digest_file(path: str | Path, *, chunk_size: int = 1024 * 1024,
                media_type: str = "application/octet-stream", source_uri: str | None = None) -> ArtifactDigest:
    target = Path(path)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    hasher = sha256()
    size = 0
    with target.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return ArtifactDigest(target.name, hasher.hexdigest(), size, media_type, source_uri)


def digest_value(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_manifest(*, run_id: str, strategy_id: str, strategy_version: str,
                   config: Mapping[str, Any], datasets: Iterable[str | Path] = (),
                   code_files: Iterable[str | Path] = (), result_files: Iterable[str | Path] = (),
                   git_revision: str | None = None, random_seed: int | None = None,
                   dependency_lock: str | Path | None = None,
                   notes: str = "", metadata: Mapping[str, str] | None = None,
                   now: datetime | None = None) -> ResearchManifest:
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    dataset_rows = tuple(digest_file(p, media_type="text/csv" if str(p).endswith(".csv") else "application/octet-stream") for p in datasets)
    code_rows = tuple(digest_file(p, media_type="text/x-python" if str(p).endswith(".py") else "text/plain") for p in code_files)
    result_rows = tuple(digest_file(p) for p in result_files)
    lock_digest = digest_file(dependency_lock).sha256 if dependency_lock else None
    return ResearchManifest(run_id, timestamp.isoformat(), git_revision, sys.version,
        platform.platform(), strategy_id, strategy_version, digest_value(config), dataset_rows,
        code_rows, result_rows, random_seed, lock_digest, notes, dict(metadata or {}))


def verify_artifact(path: str | Path, expected_sha256: str) -> bool:
    actual = digest_file(path).sha256
    return actual.lower() == expected_sha256.lower()


def verify_manifest_file(path: str | Path) -> tuple[bool, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = data.pop("manifest_digest", None)
    actual = sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return claimed == actual, actual
