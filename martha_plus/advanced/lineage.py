"""Artifact lineage and deterministic manifest generation."""
from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Any


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    path: str
    size_bytes: int
    sha256: str
    modified_at: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    manifest_version: int
    created_at: str
    artifacts: tuple[ArtifactEntry, ...]
    metadata: Mapping[str, Any]
    manifest_hash: str


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1024:
        raise ValueError("chunk_size must be >= 1024")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(paths: Iterable[str | Path], *, base_dir: str | Path | None = None,
                   metadata: Mapping[str, Any] | None = None) -> ArtifactManifest:
    root = Path(base_dir).resolve() if base_dir is not None else None
    entries: list[ArtifactEntry] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path}")
        if root is not None:
            try:
                display = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"artifact {path} is outside base_dir {root}") from exc
        else:
            display = path.name
        if display in seen:
            raise ValueError(f"duplicate artifact path: {display}")
        seen.add(display)
        stat = path.stat()
        entries.append(ArtifactEntry(display, stat.st_size, hash_file(path),
                                     datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()))
    entries.sort(key=lambda item: item.path)
    metadata_dict = dict(metadata or {})
    payload = {"manifest_version": 1, "artifacts": [e.__dict__ if hasattr(e, "__dict__") else
               {"path": e.path, "size_bytes": e.size_bytes, "sha256": e.sha256, "modified_at": e.modified_at}
               for e in entries], "metadata": metadata_dict}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return ArtifactManifest(1, datetime.now(timezone.utc).isoformat(), tuple(entries), metadata_dict, digest)


def manifest_to_dict(manifest: ArtifactManifest) -> dict[str, Any]:
    return {"manifest_version": manifest.manifest_version, "created_at": manifest.created_at,
            "artifacts": [{"path": x.path, "size_bytes": x.size_bytes, "sha256": x.sha256,
                           "modified_at": x.modified_at} for x in manifest.artifacts],
            "metadata": dict(manifest.metadata), "manifest_hash": manifest.manifest_hash}


def verify_manifest(manifest: Mapping[str, Any], base_dir: str | Path) -> dict[str, Any]:
    root = Path(base_dir).resolve()
    checks: list[dict[str, Any]] = []
    for entry in manifest.get("artifacts", []):
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            checks.append({"path": str(relative), "ok": False, "reason": "unsafe path"})
            continue
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            checks.append({"path": str(relative), "ok": False, "reason": "path escapes root"})
            continue
        if not target.is_file():
            checks.append({"path": str(relative), "ok": False, "reason": "missing file"})
            continue
        actual_hash = hash_file(target)
        actual_size = target.stat().st_size
        ok = actual_hash == entry.get("sha256") and actual_size == entry.get("size_bytes")
        checks.append({"path": str(relative), "ok": ok, "sha256_matches": actual_hash == entry.get("sha256"),
                       "size_matches": actual_size == entry.get("size_bytes")})
    return {"ok": all(item["ok"] for item in checks), "checked": len(checks), "checks": checks}


def write_manifest(manifest: ArtifactManifest, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    content = json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)
    return target
