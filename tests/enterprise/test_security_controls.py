from datetime import datetime, timedelta, timezone

import pytest

from martha_plus.enterprise.authz import Action, AccessController, AccessDenied, Effect, Policy, Principal
from martha_plus.enterprise.signed_requests import HMACRequestVerifier, NonceStore, sign_request
from martha_plus.enterprise.api_guard import (CircuitGuard, RequestGuard, RequestMetadata,
    RequestPolicy, RequestRejected, TokenBucketLimiter, sanitize_log_value)


def test_authz_deny_overrides_allow_and_logs():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    controller = AccessController([
        Policy("allow", "ops", Action.WRITE, "risk/"),
        Policy("deny", "ops", Action.WRITE, "risk/secret", Effect.DENY),
    ], clock=lambda: now)
    principal = Principal("alice", frozenset({"ops"}))
    assert controller.evaluate(principal, Action.WRITE, "risk/public").allowed
    denied = controller.evaluate(principal, Action.WRITE, "risk/secret/limits")
    assert not denied.allowed and denied.reason == "explicit_deny"
    assert controller.summary() == {"total": 2, "allowed": 1, "denied": 1,
        "distinct_principals": 1, "distinct_resources": 2}
    with pytest.raises(AccessDenied):
        controller.require(principal, Action.WRITE, "unknown")


def test_authz_live_orders_disabled_even_when_policy_allows():
    policy = Policy("live", "admin", Action.TRADE_LIVE, "orders/")
    controller = AccessController([policy])
    result = controller.evaluate(Principal("admin", frozenset({"admin"})),
                                 Action.TRADE_LIVE, "orders/live/abc")
    assert not result.allowed
    assert result.reason == "live_trading_disabled_by_policy"


def test_hmac_signatures_verify_and_replay_is_rejected():
    secret = b"very-secure-test-key-0123456789"
    clock = lambda: 1_800_000_000.0
    signed = sign_request(secret, "post", "/orders", 1_800_000_000, "nonce-1", {"qty": 3})
    verifier = HMACRequestVerifier({"key-1": secret}, clock=clock,
                                   nonce_store=NonceStore(clock=clock))
    assert verifier.verify("key-1", signed, {"qty": 3}).accepted
    replay = verifier.verify("key-1", signed, {"qty": 3})
    assert not replay.accepted and replay.reason == "nonce_replayed"
    tampered = sign_request(secret, "POST", "/orders", 1_800_000_000, "nonce-2", {"qty": 3})
    result = verifier.verify("key-1", tampered, {"qty": 4})
    assert result.reason == "body_digest_mismatch"


def test_hmac_rejects_unknown_keys_and_stale_timestamps():
    secret = b"very-secure-test-key-0123456789"
    signed = sign_request(secret, "GET", "/health", 10, "n1", {})
    verifier = HMACRequestVerifier({"known": secret}, clock=lambda: 1000, max_clock_skew_seconds=5)
    assert verifier.verify("missing", signed, {}).reason == "unknown_key"
    assert verifier.verify("known", signed, {}).reason == "timestamp_out_of_window"


def test_token_bucket_refills_and_enforces_capacity():
    now = [0.0]
    limiter = TokenBucketLimiter(capacity=2, refill_per_second=1, clock=lambda: now[0])
    assert limiter.consume("client").allowed
    assert limiter.consume("client").allowed
    denied = limiter.consume("client")
    assert not denied.allowed and denied.retry_after_seconds == 1
    now[0] += 1.0
    assert limiter.consume("client").allowed


def test_request_guard_and_query_allowlist():
    guard = RequestGuard(RequestPolicy(max_body_bytes=100, allowed_methods=("POST",)))
    guard.validate(RequestMetadata("POST", "/orders", 10, 1, 5, "application/json"))
    with pytest.raises(RequestRejected):
        guard.validate(RequestMetadata("GET", "/orders", 0, 0, 1, None))
    guard.validate_query({"limit": 10}, allowed={"limit"}, required={"limit"})
    with pytest.raises(RequestRejected, match="unknown_query"):
        guard.validate_query({"token": "abc"}, allowed={"limit"})
    assert guard.validate_json_object({"symbol": "ABC"}, allowed_fields={"symbol"}) == {"symbol": "ABC"}


def test_circuit_guard_opens_then_recovers():
    now = [0.0]
    breaker = CircuitGuard(failure_threshold=2, reset_timeout=10, clock=lambda: now[0])
    assert breaker.allow_call()
    breaker.failure(); breaker.failure()
    assert not breaker.allow_call()
    now[0] = 10
    assert breaker.allow_call()
    breaker.success()
    assert breaker.snapshot().state == "closed"


def test_log_sanitizer_flattens_controls():
    assert sanitize_log_value("hi\nthere\x00x") == "hi therex"
    assert len(sanitize_log_value("x" * 100, max_length=10)) == 10
