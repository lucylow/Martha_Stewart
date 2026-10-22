"""Canonical HMAC request signing with bounded timestamp skew and nonce replay defense.

The caller must protect its key material with a dedicated secret manager. Never
hard-code signing keys in source control or log the canonical string.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from threading import RLock
from time import time
from typing import Any, Callable, Mapping
from urllib.parse import urlencode


class SignatureError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class SignedPayload:
    method: str
    path: str
    timestamp: int
    nonce: str
    body_digest: str
    signature: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    accepted: bool
    reason: str
    key_id: str
    timestamp: int
    nonce: str


def canonical_json(body: Any) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_request(method: str, path: str, timestamp: int, nonce: str,
                      body_digest: str) -> bytes:
    normalized_method = method.upper().strip()
    normalized_path = path.strip()
    if not normalized_method or not normalized_path.startswith("/"):
        raise ValueError("method and absolute path are required")
    if "\n" in normalized_path or "\r" in normalized_path:
        raise ValueError("path cannot contain newlines")
    if not nonce or "\n" in nonce or "\r" in nonce:
        raise ValueError("nonce is required and cannot contain newlines")
    return "\n".join((normalized_method, normalized_path, str(int(timestamp)), nonce, body_digest)).encode("utf-8")


def sign_request(secret: bytes, method: str, path: str, timestamp: int, nonce: str, body: Any) -> SignedPayload:
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise ValueError("secret must contain at least 16 bytes")
    body_hash = hashlib.sha256(canonical_json(body)).hexdigest()
    message = canonical_request(method, path, timestamp, nonce, body_hash)
    signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
    return SignedPayload(method.upper(), path, int(timestamp), nonce, body_hash, signature)


class NonceStore:
    def __init__(self, *, ttl_seconds: float = 300.0, clock: Callable[[], float] = time) -> None:
        if ttl_seconds <= 0:
            raise ValueError("nonce TTL must be positive")
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._nonces: dict[tuple[str, str], float] = {}
        self._lock = RLock()

    def consume(self, key_id: str, nonce: str) -> bool:
        if not key_id or not nonce:
            return False
        now = self.clock()
        with self._lock:
            for item in [key for key, expiry in self._nonces.items() if expiry <= now]:
                del self._nonces[item]
            marker = (key_id, nonce)
            if marker in self._nonces:
                return False
            self._nonces[marker] = now + self.ttl_seconds
            return True

    def size(self) -> int:
        with self._lock:
            return len(self._nonces)


class HMACRequestVerifier:
    def __init__(self, secrets_by_key_id: Mapping[str, bytes], *,
                 max_clock_skew_seconds: int = 60,
                 clock: Callable[[], float] = time,
                 nonce_store: NonceStore | None = None) -> None:
        if max_clock_skew_seconds < 0:
            raise ValueError("clock skew cannot be negative")
        self.secrets = dict(secrets_by_key_id)
        if any(len(key) < 16 for key in self.secrets.values()):
            raise ValueError("all secrets must contain at least 16 bytes")
        self.max_skew = max_clock_skew_seconds
        self.clock = clock
        self.nonces = nonce_store or NonceStore(clock=clock)

    def verify(self, key_id: str, signed: SignedPayload, body: Any, *,
               consume_nonce: bool = True) -> VerificationResult:
        now = int(self.clock())
        secret = self.secrets.get(key_id)
        if secret is None:
            return VerificationResult(False, "unknown_key", key_id, signed.timestamp, signed.nonce)
        if abs(now - signed.timestamp) > self.max_skew:
            return VerificationResult(False, "timestamp_out_of_window", key_id, signed.timestamp, signed.nonce)
        digest = hashlib.sha256(canonical_json(body)).hexdigest()
        if not hmac.compare_digest(digest, signed.body_digest):
            return VerificationResult(False, "body_digest_mismatch", key_id, signed.timestamp, signed.nonce)
        expected = hmac.new(secret, canonical_request(signed.method, signed.path,
            signed.timestamp, signed.nonce, signed.body_digest), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signed.signature):
            return VerificationResult(False, "bad_signature", key_id, signed.timestamp, signed.nonce)
        if consume_nonce and not self.nonces.consume(key_id, signed.nonce):
            return VerificationResult(False, "nonce_replayed", key_id, signed.timestamp, signed.nonce)
        return VerificationResult(True, "verified", key_id, signed.timestamp, signed.nonce)


def canonical_query(parameters: Mapping[str, str | int | float]) -> str:
    """Sort query parameters to avoid signature disagreement between clients."""
    pairs = sorted((str(k), str(v)) for k, v in parameters.items())
    return urlencode(pairs, doseq=True)
