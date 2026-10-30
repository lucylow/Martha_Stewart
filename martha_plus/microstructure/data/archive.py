"""Append-only event archive with checksummed records and atomic finalization."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from ..types import MarketEvent
from .normalizer import event_from_dict, event_to_dict


@dataclass(frozen=True, slots=True)
class ArchiveStats:
    path: str
    records: int
    bytes_written: int
    sha256: str
    first_ts_ns: int | None
    last_ts_ns: int | None


@dataclass(frozen=True, slots=True)
class ArchiveRecord:
    line_number: int
    event: MarketEvent
    digest: str


class EventArchiveWriter:
    """Write one JSON object per line to a temporary file then atomically rename.

    This writer does not fsync directories or claim durable distributed
    persistence.  It is intended for local research data and reproducible runs.
    """

    def __init__(self, path: str | Path, *, metadata: Mapping[str, str] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self._handle = None
        self._temp_path: Path | None = None
        self._hasher = hashlib.sha256()
        self._records = 0
        self._bytes = 0
        self._first: int | None = None
        self._last: int | None = None

    def __enter__(self) -> "EventArchiveWriter":
        if self._handle is not None:
            raise RuntimeError("archive writer is already open")
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        self._temp_path = Path(name)
        self._handle = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        self._write_payload({"type": "header", "version": 1, "metadata": self.metadata})
        return self

    def _write_payload(self, payload: Mapping[str, object]) -> None:
        if self._handle is None:
            raise RuntimeError("archive writer is not open")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        line = raw + "\n"
        self._handle.write(line)
        encoded = line.encode("utf-8")
        self._hasher.update(encoded)
        self._bytes += len(encoded)

    def write(self, event: MarketEvent) -> None:
        if self._handle is None:
            raise RuntimeError("archive writer is not open")
        if self._last is not None and event.ts_ns < self._last:
            raise ValueError("archive events must be timestamp ordered")
        event_payload = event_to_dict(event)
        canonical = json.dumps(event_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload = {"type": "event", "event": event_payload,
                   "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
        self._write_payload(payload)
        self._records += 1
        self._first = event.ts_ns if self._first is None else self._first
        self._last = event.ts_ns

    def write_many(self, events: Iterable[MarketEvent]) -> int:
        count = 0
        for event in events:
            self.write(event)
            count += 1
        return count

    def close(self, *, commit: bool = True) -> ArchiveStats | None:
        if self._handle is None:
            return None
        handle, self._handle = self._handle, None
        try:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            if commit:
                assert self._temp_path is not None
                os.replace(self._temp_path, self.path)
            elif self._temp_path is not None:
                self._temp_path.unlink(missing_ok=True)
        except Exception:
            if self._temp_path is not None:
                self._temp_path.unlink(missing_ok=True)
            raise
        if not commit:
            return None
        return ArchiveStats(str(self.path), self._records, self._bytes,
                            self._hasher.hexdigest(), self._first, self._last)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(commit=exc_type is None)


class EventArchiveReader:
    """Iterate event archive rows and fail closed on malformed/checksum data."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata: dict[str, str] = {}
        self.digest: str | None = None

    def __iter__(self) -> Iterator[ArchiveRecord]:
        hasher = hashlib.sha256()
        with self.path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if not first:
                raise ValueError("archive is empty")
            hasher.update(first.encode("utf-8"))
            try:
                header = json.loads(first)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid archive header") from exc
            if header.get("type") != "header" or header.get("version") != 1:
                raise ValueError("unsupported archive format")
            self.metadata = dict(header.get("metadata", {}))
            line_number = 1
            for line in handle:
                line_number += 1
                hasher.update(line.encode("utf-8"))
                try:
                    payload = json.loads(line)
                    if payload.get("type") != "event":
                        raise ValueError("unknown record type")
                    event = event_from_dict(payload["event"])
                    canonical = json.dumps(event_to_dict(event), sort_keys=True,
                                           separators=(",", ":"), allow_nan=False)
                    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    if payload.get("sha256") != digest:
                        raise ValueError("event checksum mismatch")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid archive record at line {line_number}: {exc}") from exc
                yield ArchiveRecord(line_number, event, digest)
        self.digest = hasher.hexdigest()

    def read_all(self) -> tuple[MarketEvent, ...]:
        return tuple(row.event for row in self)

    def verify_ordering(self) -> bool:
        previous: tuple[int, int] | None = None
        for row in self:
            key = (row.event.ts_ns, row.event.sequence)
            if previous is not None and key < previous:
                return False
            previous = key
        return True


def archive_events(path: str | Path, events: Iterable[MarketEvent], *,
                   metadata: Mapping[str, str] | None = None) -> ArchiveStats:
    writer = EventArchiveWriter(path, metadata=metadata)
    writer.__enter__()
    try:
        writer.write_many(events)
        result = writer.close(commit=True)
        assert result is not None
        return result
    except Exception:
        writer.close(commit=False)
        raise
