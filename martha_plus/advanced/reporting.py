"""Safe export of research metrics to JSON, CSV, and self-contained HTML."""
from __future__ import annotations
import csv
import html
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ReportSection:
    title: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    rows: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchReport:
    title: str
    generated_at: str
    run_id: str
    sections: tuple[ReportSection, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, title: str, run_id: str, sections: Sequence[ReportSection],
               metadata: Mapping[str, Any] | None = None) -> "ResearchReport":
        if not title.strip() or not run_id.strip():
            raise ValueError("title and run_id are required")
        return cls(title.strip(), datetime.now(timezone.utc).isoformat(), run_id,
                   tuple(sections), dict(metadata or {}))


def _clean(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("report datetime values must be timezone-aware")
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if hasattr(value, "value"):
        return _clean(value.value)
    return str(value)


def report_to_dict(report: ResearchReport) -> dict[str, Any]:
    return {"title": report.title, "generated_at": report.generated_at, "run_id": report.run_id,
            "metadata": _clean(report.metadata),
            "sections": [{"title": section.title, "metrics": _clean(section.metrics),
                          "notes": list(section.notes), "rows": _clean(section.rows)}
                         for section in report.sections]}


def render_html(report: ResearchReport) -> str:
    """Render an escaped, self-contained report; all user-controlled strings are escaped."""
    payload = report_to_dict(report)
    parts = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>{html.escape(report.title)}</title>",
             "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124}"
             "h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;margin:1rem 0}"
             "th,td{border:1px solid #ddd;padding:.45rem;text-align:left;overflow-wrap:anywhere}"
             "th{background:#f4f4f4}.muted{color:#666;font-size:.9rem}.note{padding:.5rem 0}</style></head><body>",
             f"<h1>{html.escape(report.title)}</h1>",
             f"<p class='muted'>Run {html.escape(report.run_id)} · {html.escape(report.generated_at)}</p>"]
    if report.metadata:
        parts.append("<h2>Metadata</h2><dl>")
        for key, value in sorted(report.metadata.items()):
            parts.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(_clean(value)))}</dd>")
        parts.append("</dl>")
    for section in report.sections:
        parts.append(f"<section><h2>{html.escape(section.title)}</h2>")
        if section.metrics:
            parts.append("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>")
            for key, value in sorted(section.metrics.items()):
                parts.append(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(_clean(value)))}</td></tr>")
            parts.append("</tbody></table>")
        for note in section.notes:
            parts.append(f"<p class='note'>{html.escape(note)}</p>")
        if section.rows:
            columns = sorted({key for row in section.rows for key in row})
            parts.append("<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in columns) + "</tr></thead><tbody>")
            for row in section.rows:
                parts.append("<tr>" + "".join(f"<td>{html.escape(str(_clean(row.get(c))))}</td>" for c in columns) + "</tr>")
            parts.append("</tbody></table>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_markdown(report: ResearchReport) -> str:
    lines = [f"# {report.title}", "", f"Run: `{report.run_id}`  ", f"Generated: {report.generated_at}", ""]
    for section in report.sections:
        lines.extend([f"## {section.title}", ""])
        if section.metrics:
            lines.extend(["| Metric | Value |", "|---|---:|"])
            for key, value in sorted(section.metrics.items()):
                lines.append(f"| {str(key).replace('|', '\\|')} | {str(_clean(value)).replace('|', '\\|')} |")
            lines.append("")
        lines.extend(f"- {note}" for note in section.notes)
        if section.notes:
            lines.append("")
        if section.rows:
            columns = sorted({key for row in section.rows for key in row})
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join("---" for _ in columns) + " |")
            for row in section.rows:
                lines.append("| " + " | ".join(str(_clean(row.get(c))).replace("|", "\\|") for c in columns) + " |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_json(report: ResearchReport, path: str | Path) -> Path:
    return _atomic_write(path, json.dumps(report_to_dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_html(report: ResearchReport, path: str | Path) -> Path:
    return _atomic_write(path, render_html(report))


def write_markdown(report: ResearchReport, path: str | Path) -> Path:
    return _atomic_write(path, render_markdown(report))


def write_metrics_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    columns = sorted({str(key) for row in rows for key in row})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            if columns:
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: _clean(row.get(k)) for k in columns})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try: os.unlink(tmp_name)
        except OSError: pass
        raise
    return target


def _atomic_write(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, target)
    except Exception:
        try: os.unlink(tmp_name)
        except OSError: pass
        raise
    return target
