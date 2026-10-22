from datetime import datetime, timedelta, timezone

import pytest

from martha_plus.enterprise.streaming import (BarAggregator, EventTimeBuffer, LateEventPolicy, StreamEvent, StreamRouter)
from martha_plus.enterprise.point_in_time import PointInTimeIndex, VersionedValue, asof_join, assert_no_future_features
from martha_plus.enterprise.fx import FXBook, FXQuote, FXRateUnavailable
from martha_plus.enterprise.quality_rules import (QualityRuleBook, RuleSeverity, required_fields,
    numeric_range, uniqueness, monotonic_datetime, quarantine_indices)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def event(ident, second, price=10.0, size=1.0):
    ts = NOW + timedelta(seconds=second)
    return StreamEvent(ident, "trade", "ABC", ts, ts, {"price": price, "size": size})


def test_event_time_buffer_orders_events_and_deduplicates():
    buffer = EventTimeBuffer(allowed_lateness=timedelta(seconds=1))
    assert buffer.push(event("e2", 2)) == ()
    output = buffer.push(event("e1", 1))
    assert [item.event_id for item in output] == ["e1"]
    assert buffer.push(event("e2", 2)) == ()
    assert buffer.stats().duplicates == 1
    assert [item.event_id for item in buffer.flush()] == ["e2"]


def test_event_time_buffer_handles_late_policy_and_id_conflicts():
    buffer = EventTimeBuffer(allowed_lateness=timedelta(seconds=1), late_policy=LateEventPolicy.SIDE_OUTPUT)
    buffer.push(event("a", 10)); buffer.push(event("b", 12))
    buffer.push(event("late", 2))
    assert [row.event_id for row in buffer.late_events] == ["late"]
    with pytest.raises(ValueError):
        buffer.push(StreamEvent("a", "trade", "ABC", NOW + timedelta(seconds=11),
                                NOW + timedelta(seconds=11), {"price": 99}))


def test_bar_aggregator_builds_ohlcv_and_closes_by_watermark():
    agg = BarAggregator(timedelta(seconds=60))
    agg.add(event("a", 10, 10, 2))
    agg.add(event("b", 20, 12, 3))
    agg.add(event("c", 30, 9, 1))
    bars = agg.close_before(NOW + timedelta(seconds=60))
    assert len(bars) == 1
    bar = bars[0]
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.trade_count) == (10, 12, 9, 9, 6, 3)


def test_stream_router_isolates_failed_handlers():
    router = StreamRouter()
    seen = []
    router.subscribe("trade", lambda ev: (_ for _ in ()).throw(RuntimeError("fail")))
    router.subscribe("trade", lambda ev: seen.append(ev.event_id))
    errors = router.dispatch(event("a", 1))
    assert len(errors) == 1 and seen == ["a"]


def test_point_in_time_lookup_never_uses_unpublished_revision():
    index = PointInTimeIndex()
    index.add(VersionedValue("EPS", 1.0, NOW - timedelta(days=60), NOW - timedelta(days=30), source="file1"))
    index.add(VersionedValue("EPS", 2.0, NOW - timedelta(days=20), NOW + timedelta(days=5), source="file2"))
    result = index.lookup("EPS", NOW)
    assert result.matched and result.value == 1.0
    stale = index.lookup("EPS", NOW, max_age=timedelta(days=10))
    assert not stale.matched and stale.reason == "stale_value"


def test_asof_join_backward_only_and_future_feature_guard():
    left = [{"symbol": "A", "when": NOW + timedelta(days=1)}]
    right = [{"symbol": "A", "time": NOW, "value": 10},
             {"symbol": "A", "time": NOW + timedelta(days=2), "value": 20}]
    joined = asof_join(left, right, left_time="when", right_time="time", by="symbol", fields=("value",))
    assert joined[0]["value_pit"] == 10 and joined[0]["_pit_matched"]
    with pytest.raises(ValueError, match="future leakage"):
        assert_no_future_features([NOW + timedelta(days=2)], [NOW])


def test_fx_book_converts_direct_and_inverse_cross_rates():
    book = FXBook(now=lambda: NOW, max_age=timedelta(days=1))
    book.update(FXQuote("USD", "CAD", 1.35, NOW - timedelta(minutes=1)))
    book.update(FXQuote("CAD", "EUR", 0.67, NOW - timedelta(minutes=1)))
    direct = book.convert(100, "USD", "CAD")
    assert direct.converted_amount == pytest.approx(135)
    cross = book.convert(100, "USD", "EUR")
    assert cross.converted_amount == pytest.approx(90.45)
    inverse = book.convert(135, "CAD", "USD")
    assert inverse.converted_amount == pytest.approx(100)


def test_fx_book_rejects_stale_or_future_quotes():
    book = FXBook(now=lambda: NOW, max_age=timedelta(seconds=30))
    book.update(FXQuote("USD", "CAD", 1.35, NOW - timedelta(hours=1)))
    with pytest.raises(FXRateUnavailable):
        book.convert(100, "USD", "CAD")
    book.update(FXQuote("USD", "EUR", 0.9, NOW + timedelta(days=1)))
    with pytest.raises(FXRateUnavailable):
        book.convert(100, "USD", "EUR", as_of=NOW)


def test_quality_rules_collect_results_and_quarantine():
    rules = QualityRuleBook([
        required_fields(["symbol", "close"], rule_id="required"),
        numeric_range("close", minimum=0, rule_id="close_nonnegative"),
        uniqueness(["symbol"], rule_id="unique_symbol"),
        monotonic_datetime("time", rule_id="time_order"),
    ], clock=lambda: NOW)
    rows = [{"symbol": "A", "close": 10, "time": NOW},
            {"symbol": "A", "close": -1, "time": NOW - timedelta(days=1)},
            {"close": 2, "time": NOW}]
    report = rules.run("sample", rows)
    assert not report.passed
    assert report.summary()["failed"] >= 3
    assert set(quarantine_indices(report)) >= {0, 1, 2}
