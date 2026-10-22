"""Structured JSON logging with conservative secret redaction."""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

_BEARER_PATTERN = re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+")
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(\s*[:=]\s*)([^\s,;]+)")

def redact_secrets(value: str) -> str:
    result = str(value)
    # Process bearer credentials first, so an Authorization field can never leave
    # the token as a second whitespace-separated word.
    result = _BEARER_PATTERN.sub(r"\1[REDACTED]", result)
    result = _SECRET_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", result)
    return result

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }
        for name in ("event", "symbol", "strategy_id", "order_id", "latency_ms", "status_code"):
            if hasattr(record, name):
                payload[name] = redact_secrets(getattr(record, name))
        if record.exc_info:
            payload["exception"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)

def configure_logging(level: str = "INFO", logger_name: str = "martha_plus") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(getattr(handler, "_martha_plus", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._martha_plus = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger
