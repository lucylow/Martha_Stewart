"""Safe, deterministic export helpers for operational and research reports."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import hashlib
import html
import json
import os
import tempfile


class ExportError(RuntimeError):
    pass


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime values must be timezone-aware")
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _normalize(value.value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _normalize(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {name: _normalize(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize(v) for v in value]
    raise TypeError(f"unsupported export type: {type(value).__name__}")


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    normalized = _normalize(value)
    if pretty:
        text = json.dumps(normalized, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        text = json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def csv_bytes(rows: Iterable[Mapping[str, Any]], *, columns: Sequence[str] | None = None,
              spreadsheet_safe: bool = True) -> bytes:
    materialized = [_normalize(row) for row in rows]
    if any(not isinstance(row, dict) for row in materialized):
        raise TypeError("CSV rows must be mappings")
    if columns is None:
        keys = set()
        for row in materialized:
            keys.update(row)
        columns = tuple(sorted(keys))
    else:
        columns = tuple(columns)
        if len(columns) != len(set(columns)):
            raise ValueError("CSV columns must be unique")
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in materialized:
        normalized = {}
        for col in columns:
            value = row.get(col)
            if value is None:
                text = ""
            elif isinstance(value, (dict, list)):
                text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            else:
                text = str(value)
            if spreadsheet_safe and text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
                text = "'" + text
            normalized[col] = text
        writer.writerow(normalized)
    return buffer.getvalue().encode("utf-8")


def atomic_write(path: str | Path, content: bytes, *, mode: int = 0o600) -> Path:
    """Atomically replace a file in its destination directory and set restrictive mode."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            dir_fd = os.open(target.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        return target
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json(path: str | Path, value: Any, *, pretty: bool = True) -> tuple[Path, str]:
    content = json_bytes(value, pretty=pretty)
    target = atomic_write(path, content)
    return target, hashlib.sha256(content).hexdigest()


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], *,
              columns: Sequence[str] | None = None, spreadsheet_safe: bool = True) -> tuple[Path, str]:
    content = csv_bytes(rows, columns=columns, spreadsheet_safe=spreadsheet_safe)
    target = atomic_write(path, content)
    return target, hashlib.sha256(content).hexdigest()


def html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *,
               title: str = "Report") -> str:
    """Render a simple escaped HTML table; never trust values as HTML."""
    if not headers:
        raise ValueError("headers are required")
    header_html = "".join(f"<th scope=\"col\">{html.escape(str(value))}</th>" for value in headers)
    body = []
    for row in rows:
        values = tuple(row)
        if len(values) != len(headers):
            raise ValueError("each report row must match header length")
        cells = "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in values)
        body.append(f"<tr>{cells}</tr>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(title)}</title></head><body><main><h1>{html.escape(title)}</h1>"
            f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            "</main></body></html>")


def rounded(value: float, *, places: int = 2) -> str:
    if places < 0 or places > 12:
        raise ValueError("places must be between 0 and 12")
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN))


def redact_mapping(data: Mapping[str, Any], *, sensitive_keys: Iterable[str] =
                   ("password", "secret", "token", "api_key", "authorization", "credential"),
                   replacement: str = "[REDACTED]") -> dict[str, Any]:
    keywords = tuple(word.casefold() for word in sensitive_keys)
    result = {}
    for key, value in data.items():
        if any(keyword in str(key).casefold() for keyword in keywords):
            result[str(key)] = replacement
        elif isinstance(value, Mapping):
            result[str(key)] = redact_mapping(value, sensitive_keys=sensitive_keys, replacement=replacement)
        elif isinstance(value, (list, tuple)):
            result[str(key)] = [_redact_nested(item, sensitive_keys, replacement) for item in value]
        else:
            result[str(key)] = value
    return result


def manifest_for_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        entries.append({"path": str(path), "size_bytes": size, "sha256": digest.hexdigest()})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "files": entries,
            "manifest_digest": hashlib.sha256(json_bytes(entries)).hexdigest()}


def _redact_nested(value: Any, sensitive_keys: Iterable[str], replacement: str) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value, sensitive_keys=sensitive_keys, replacement=replacement)
    if isinstance(value, (list, tuple)):
        return [_redact_nested(item, sensitive_keys, replacement) for item in value]
    return value
