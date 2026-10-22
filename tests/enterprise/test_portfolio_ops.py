from datetime import datetime, timedelta, timezone

import pytest

from martha_plus.enterprise.portfolio_replay import (PortfolioEventStore, PortfolioEventType,
    PortfolioProjector, PortfolioReplayError, create_event)
from martha_plus.enterprise.ledger_reconciliation import (ExecutionRecord, ExecutionReconciler,
    DifferenceKind, account_cash_control, compare_cash_controls)
from martha_plus.enterprise.liquidity import (LiquiditySnapshot, LiquidityPolicy, LiquidityGuard, vwap_schedule)
from martha_plus.enterprise.risk_limits import (RiskLimitEngine, standard_risk_limits, LimitMetric,
    RiskLimit, LimitComparison, BreachSeverity)
from martha_plus.enterprise.cashflows import CashFlow

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_portfolio_event_store_replay_positions_and_cash():
    events = [
        create_event("a", 1, PortfolioEventType.OPENING_POSITION, symbol="ABC", quantity_delta=10,
                     price=10, cash_delta=0, occurred_at=NOW, event_id="e1"),
        create_event("a", 2, PortfolioEventType.MARK, symbol="ABC", price=12,
                     occurred_at=NOW + timedelta(seconds=1), event_id="e2"),
        create_event("a", 3, PortfolioEventType.CASH_ADJUSTMENT, cash_delta=500,
                     occurred_at=NOW + timedelta(seconds=2), event_id="e3"),
    ]
    store = PortfolioEventStore()
    store.append_many(events)
    assert store.verify("a") == ()
    projection = PortfolioProjector().project(store.events("a"), account="a")
    assert projection.positions[0].quantity == 10
    assert projection.positions[0].market_value == 120
    assert projection.cash_by_currency["USD"] == 500
    assert PortfolioProjector().verify_replay_determinism(events, account="a")


def test_portfolio_event_store_rejects_gaps_and_conflicting_duplicate_ids():
    store = PortfolioEventStore()
    store.append(create_event("a", 1, PortfolioEventType.CASH_ADJUSTMENT, cash_delta=1,
                              occurred_at=NOW, event_id="same"))
    with pytest.raises(PortfolioReplayError):
        store.append(create_event("a", 3, PortfolioEventType.CASH_ADJUSTMENT, cash_delta=1,
                                  occurred_at=NOW, event_id="e3"))
    with pytest.raises(PortfolioReplayError):
        store.append(create_event("a", 1, PortfolioEventType.CASH_ADJUSTMENT, cash_delta=2,
                                  occurred_at=NOW, event_id="same"))


def exec_record(ident, qty, price, source):
    return ExecutionRecord(ident, "o1", "acct", "ABC", "buy", qty, price, 1.0,
                           NOW, source=source)


def test_execution_reconciliation_detects_breaks_and_reviews():
    recon = ExecutionReconciler(clock=lambda: NOW)
    summary = recon.reconcile([exec_record("e1", 10, 100, "internal"),
                               exec_record("e2", 2, 10, "internal")],
                              [exec_record("e1", 9, 100, "broker"),
                               exec_record("e3", 1, 20, "broker")])
    assert summary.break_count >= 2
    qty_break = next(row for row in summary.breaks if row.kind is DifferenceKind.QUANTITY_MISMATCH)
    recon.review(qty_break.break_id, reviewer="ops", note="verified", resolved=True)
    assert recon.summary()["resolved"] == 1


def test_cash_control_breaks():
    internal = [exec_record("e1", 10, 100, "internal")]
    external = [exec_record("e1", 10, 100, "broker")]
    left, right = account_cash_control(internal), account_cash_control(external)
    assert compare_cash_controls(left, right) == ()
    assert compare_cash_controls({("a", "USD"): 10}, {("a", "USD"): 9})[0]["difference"] == 1


def test_liquidity_guard_and_vwap_caps():
    snapshot = LiquiditySnapshot("ABC", NOW.isoformat(), 99, 101, 500, 450,
                                 10000, spread_bps=20)
    guard = LiquidityGuard(LiquidityPolicy(max_participation=0.1, max_spread_bps=50))
    assert guard.check(snapshot, requested_quantity=500).allowed
    blocked = guard.check(snapshot, requested_quantity=2000)
    assert not blocked.allowed and blocked.recommended_max_quantity == pytest.approx(1000)
    schedule = vwap_schedule(100, [100, 200, 300], max_participation=0.5)
    assert sum(schedule) == pytest.approx(100)
    with pytest.raises(ValueError):
        vwap_schedule(1000, [100, 100], max_participation=0.1)


def test_risk_limit_engine_blocks_and_halts():
    engine = RiskLimitEngine(standard_risk_limits())
    result = engine.evaluate({"gross_exposure": 2.5, "drawdown": 0.25,
                              "daily_loss": -0.01, "data_age_seconds": 2})
    assert not result.permitted and result.halted
    assert any(row.metric is LimitMetric.GROSS_EXPOSURE for row in result.breaches)
    first = result.breaches[0]
    engine.acknowledge(first.breach_id, actor="risk", note="reviewed")
    assert engine.summary()["unacknowledged"] == len(result.breaches) - 1


def test_risk_limits_ignore_missing_values_and_can_disable():
    engine = RiskLimitEngine([RiskLimit("open", LimitMetric.OPEN_ORDERS, LimitComparison.MAX, 5,
                                        BreachSeverity.WARNING)])
    assert not engine.evaluate({}).breaches
    assert engine.evaluate({"open_orders": 10}).breaches
    assert engine.disable("open")
    assert not engine.evaluate({"open_orders": 20}).breaches


def test_portfolio_replay_accounts_for_short_cover_basis():
    events = [
        create_event("short", 1, PortfolioEventType.OPENING_POSITION, symbol="XYZ",
                     quantity_delta=-10, price=10, occurred_at=NOW, event_id="s1"),
        create_event("short", 2, PortfolioEventType.FILL, symbol="XYZ",
                     quantity_delta=3, price=8, cash_delta=-24, occurred_at=NOW + timedelta(seconds=1), event_id="s2"),
        create_event("short", 3, PortfolioEventType.MARK, symbol="XYZ", price=8,
                     occurred_at=NOW + timedelta(seconds=2), event_id="s3"),
    ]
    projection = PortfolioProjector().project(events, account="short")
    position = projection.positions[0]
    assert position.quantity == -7
    assert position.market_value == -56
    assert position.unrealized_pnl == pytest.approx(14)
