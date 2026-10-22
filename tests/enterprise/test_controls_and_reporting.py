from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from martha_plus.enterprise.outbox import SqliteOutbox
from martha_plus.enterprise.restricted_list import (RestrictedInstrumentRegistry, Restriction,
    RestrictionLevel, Side)
from martha_plus.enterprise.surveillance import (OrderObservation, SurveillanceEngine, AlertSeverity)
from martha_plus.enterprise.scenario import Position, Scenario, ScenarioEngine, expected_shortfall, scenario_breach
from martha_plus.enterprise.signal_calibration import (classification_metrics, reliability_bins,
    expected_calibration_error, search_threshold)
from martha_plus.enterprise.exports import csv_bytes, html_table, redact_mapping, write_json
from martha_plus.enterprise.reproducibility import verify_manifest_file
from martha_plus.enterprise.reproducibility import build_manifest

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_sqlite_outbox_lease_ack_and_retry():
    now = [100.0]
    db = sqlite3.connect(":memory:")
    outbox = SqliteOutbox(db, clock=lambda: now[0], max_attempts=2)
    event_id = outbox.enqueue("orders.paper", {"symbol": "ABC"})
    leased = outbox.lease("worker-a", limit=10)
    assert len(leased) == 1 and leased[0].event_id == event_id
    assert outbox.fail(event_id, "worker-a", "temporary", retry_delay=1) == "pending"
    now[0] += 1
    leased = outbox.lease("worker-b")
    assert leased[0].attempts == 1
    outbox.acknowledge(event_id, "worker-b")
    assert outbox.get(event_id).status == "sent"
    assert outbox.counts()["sent"] == 1


def test_outbox_dead_letters_after_max_attempts_and_requires_lease_owner():
    now = [0.0]
    outbox = SqliteOutbox(sqlite3.connect(":memory:"), clock=lambda: now[0], max_attempts=1)
    event_id = outbox.enqueue("audit", {})
    outbox.lease("worker")
    with pytest.raises(RuntimeError):
        outbox.acknowledge(event_id, "other-worker")
    assert outbox.fail(event_id, "worker", "bad payload") == "dead"


def test_restricted_list_blocks_and_reduce_only_rules():
    registry = RestrictedInstrumentRegistry(clock=lambda: NOW)
    registry.upsert("r1", Restriction("ABC", RestrictionLevel.BLOCK, "legal hold", NOW), actor="compliance")
    blocked = registry.check_order("abc", Side.BUY, 10)
    assert not blocked.allowed and blocked.restriction_ids == ("r1",)
    registry.remove("r1", actor="compliance", reason="released")
    registry.upsert("r2", Restriction("XYZ", RestrictionLevel.REDUCE_ONLY, "limit", NOW), actor="compliance")
    assert not registry.check_order("XYZ", Side.BUY, 3, current_position=0).allowed
    assert registry.check_order("XYZ", Side.SELL, 3, current_position=5).allowed


def test_surveillance_flags_large_orders_and_repeated_cancels():
    engine = SurveillanceEngine(clock=lambda: NOW, repeated_cancel_threshold=3,
        repeated_cancel_window_seconds=30, max_single_order_notional=1000)
    large = OrderObservation("o0", "a", "ABC", "buy", 100, 20, "submitted", NOW)
    flags = engine.ingest(large)
    assert flags[0].rule == "large_order_notional"
    for idx in range(3):
        engine.ingest(OrderObservation(f"c{idx}", "a", "XYZ", "sell", 1, 10,
                                      "cancelled", NOW + timedelta(seconds=idx)))
    assert any(f.rule == "repeated_cancellations" for f in engine.flags)
    flag = next(f for f in engine.flags if f.rule == "large_order_notional")
    engine.review(flag.flag_id, reviewer="compliance", disposition="cleared", note="reviewed")
    assert engine.summary()["open_flags"] >= 1


def test_scenario_engine_and_expected_shortfall():
    positions = [Position("A", 10, 100, beta=1.2, sector="tech"),
                 Position("B", 5, 200, beta=0.8, sector="finance")]
    result = ScenarioEngine().evaluate(positions, Scenario("crash", shocks={"A": -0.2}, default_shock=-0.1))
    assert result.base_value == 2000
    assert result.pnl == pytest.approx(-300)
    assert scenario_breach(result, max_loss_fraction=0.1)
    assert expected_shortfall([-10, -5, 1, 2], confidence=0.75) == pytest.approx(10)


def test_signal_calibration_and_threshold_search():
    labels = [0, 0, 1, 1, 1]
    probs = [0.1, 0.4, 0.5, 0.7, 0.9]
    metrics = classification_metrics(labels, probs)
    assert metrics.observations == 5 and metrics.brier_score >= 0
    bins = reliability_bins(labels, probs, bins=5)
    assert sum(b.count for b in bins) == 5
    assert 0 <= expected_calibration_error(labels, probs, bins=5) <= 1
    best = search_threshold(labels, probs, objective="f1")
    assert best.f1 > 0.5


def test_exports_escape_csv_formulas_and_html_injection(tmp_path):
    data = csv_bytes([{"name": "=SUM(A1:A2)", "n": 2}], columns=["name", "n"])
    assert b"'=SUM" in data
    html = html_table(["name"], [["<script>alert(1)</script>"]], title="<secret>")
    assert "&lt;script&gt;" in html and "<script>alert" not in html
    assert redact_mapping({"api_key": "abc", "nested": {"password": "x"}}) == {
        "api_key": "[REDACTED]", "nested": {"password": "[REDACTED]"}}
    path, digest = write_json(tmp_path / "report.json", {"ok": True})
    assert path.exists() and len(digest) == 64


def test_reproducibility_manifest_verifies(tmp_path):
    code_file = tmp_path / "strategy.py"
    dataset = tmp_path / "prices.csv"
    code_file.write_text("print('hello')\n")
    dataset.write_text("close\n100\n")
    manifest = build_manifest(run_id="r1", strategy_id="s", strategy_version="1",
        config={"lookback": 10}, datasets=[dataset], code_files=[code_file],
        now=NOW, random_seed=3)
    path = manifest.write_json(tmp_path / "manifest.json")
    ok, digest = verify_manifest_file(path)
    assert ok and digest == manifest.digest
