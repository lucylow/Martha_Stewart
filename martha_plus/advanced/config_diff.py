"""Safe configuration fingerprinting and deployment drift reports."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping, Sequence

SECRET_TOKENS = ("token", "secret", "password", "credential", "private_key", "api_key", "dsn")


def is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(token in normalized for token in SECRET_TOKENS)


def safe_configuration(config: Mapping[str, Any], *, reveal_secret_presence: bool = True) -> dict[str, Any]:
    """Return a JSON-compatible configuration view without disclosing secret values."""
    result: dict[str, Any] = {}
    for key, value in sorted(config.items(), key=lambda pair: str(pair[0])):
        name = str(key)
        if is_sensitive_key(name):
            result[name] = "[configured]" if reveal_secret_presence and value else "[not configured]"
        elif isinstance(value, Mapping):
            result[name] = safe_configuration(value, reveal_secret_presence=reveal_secret_presence)
        elif isinstance(value, (list, tuple)):
            result[name] = [safe_configuration(item, reveal_secret_presence=reveal_secret_presence)
                            if isinstance(item, Mapping) else item for item in value]
        else:
            result[name] = value
    return result


def configuration_fingerprint(config: Mapping[str, Any], *, include_secret_values: bool = False) -> str:
    """Hash normalized config; by default secrets affect only presence, not raw value."""
    payload = dict(config) if include_secret_values else safe_configuration(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compare_configurations(expected: Mapping[str, Any], actual: Mapping[str, Any], *,
                           ignored_paths: Sequence[str] = ()) -> dict[str, Any]:
    """Compare nested mappings with dot-path ignores and redact values at secret keys."""
    ignored = set(ignored_paths)
    changes: list[dict[str, Any]] = []
    def visit(left: Any, right: Any, path: str) -> None:
        if path in ignored:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right), key=str):
                child = f"{path}.{key}" if path else str(key)
                if key not in left:
                    changes.append({"path": child, "kind": "added", "expected": None,
                                    "actual": _value_for_path(str(key), right[key])})
                elif key not in right:
                    changes.append({"path": child, "kind": "removed", "expected": _value_for_path(str(key), left[key]),
                                    "actual": None})
                else:
                    visit(left[key], right[key], child)
        elif left != right:
            key = path.rsplit(".", 1)[-1]
            changes.append({"path": path, "kind": "changed",
                            "expected": _value_for_path(key, left), "actual": _value_for_path(key, right)})
    visit(expected, actual, "")
    return {"matches": not changes, "change_count": len(changes), "changes": changes,
            "expected_fingerprint": configuration_fingerprint(expected),
            "actual_fingerprint": configuration_fingerprint(actual)}


def _value_for_path(key: str, value: Any) -> Any:
    return ("[configured]" if value else "[not configured]") if is_sensitive_key(key) else value


def require_config_match(expected: Mapping[str, Any], actual: Mapping[str, Any], *,
                         ignored_paths: Sequence[str] = ()) -> None:
    report = compare_configurations(expected, actual, ignored_paths=ignored_paths)
    if not report["matches"]:
        paths = ", ".join(item["path"] for item in report["changes"][:20])
        raise RuntimeError(f"configuration drift detected ({report['change_count']} changes): {paths}")
