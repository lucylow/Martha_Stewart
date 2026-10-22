from datetime import date, datetime, timedelta, timezone

import pytest

from martha_plus.enterprise.taxlots import TaxLotBook, LotMethod, LotError
from martha_plus.enterprise.cashflows import CashFlow, xnpv, xirr, time_weighted_return, drawdown_series
from martha_plus.enterprise.cash_management import (CashBook, CashFlowItem, CashFlowKind,
    InsufficientCash, ReservationStatus)
from martha_plus.enterprise.benchmark import benchmark_metrics, rolling_beta
from martha_plus.enterprise.allocation import project_to_simplex, diagnose_allocation, inverse_volatility_weights

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_tax_lots_fifo_and_hifo_realized_pnl():
    book = TaxLotBook(id_factory=iter([f"id{i}" for i in range(100)]).__next__)
    book.add_lot("a", "abc", 10, 10, date(2024, 1, 1), lot_id="low")
    book.add_lot("a", "abc", 10, 20, date(2024, 2, 1), lot_id="high")
    fifo = book.dispose("a", "abc", 5, 30, date(2025, 1, 1), method=LotMethod.FIFO)
    assert fifo[0].lot_id == "low" and fifo[0].realized_pnl == 100
    hifo = book.dispose("a", "abc", 5, 30, date(2025, 1, 1), method=LotMethod.HIFO)
    assert hifo[0].lot_id == "high" and hifo[0].realized_pnl == 50
    assert book.integrity_check() == ()


def test_tax_lot_sale_is_atomic_when_insufficient():
    book = TaxLotBook()
    book.add_lot("a", "X", 3, 10, date(2024, 1, 1))
    with pytest.raises(LotError):
        book.dispose("a", "X", 4, 20, date(2025, 1, 1))
    assert book.position("a", "X").quantity == 3


def test_tax_lot_split_preserves_basis():
    book = TaxLotBook()
    book.add_lot("a", "X", 10, 100, date(2024, 1, 1), lot_id="l1")
    assert book.apply_split("a", "X", 2, date(2025, 1, 1)) == 1
    pos = book.position("a", "X")
    assert pos.quantity == 20 and pos.cost_basis == 1000 and pos.average_cost == 50


def test_xirr_and_twr_known_examples():
    flows = [CashFlow(date(2024, 1, 1), -1000), CashFlow(date(2024, 12, 31), 1100)]
    assert xnpv(0.1, flows) == pytest.approx(0, abs=0.01)
    assert xirr(flows) == pytest.approx(0.1, abs=1e-5)
    assert time_weighted_return([0.1, -0.05]) == pytest.approx(0.045)
    assert drawdown_series([100, 120, 90]) == pytest.approx([0, 0, -0.25])


def test_xirr_rejects_unbracketed_series():
    with pytest.raises(ValueError):
        xirr([CashFlow(date(2024, 1, 1), -100), CashFlow(date(2025, 1, 1), -100)])


def test_cash_book_reservation_and_forecast():
    book = CashBook(clock=lambda: NOW, id_factory=lambda: "res-1")
    book.set_balance("a", "USD", 1000, actor="ops", reason="opening")
    book.post_flow(CashFlowItem("f1", "a", "USD", date(2026, 1, 2), -100,
                                CashFlowKind.TRADE_SETTLEMENT))
    reservation = book.reserve("a", "USD", 800, purpose="order", expires_at=NOW + timedelta(hours=1))
    assert book.balance("a", "USD").available == pytest.approx(200)
    with pytest.raises(InsufficientCash):
        book.reserve("a", "USD", 300, purpose="too large")
    forecast = book.forecast("a", "USD", start=date(2026, 1, 1), end=date(2026, 1, 2), minimum_reserve=950)
    assert forecast[-1].closing_balance == 900 and forecast[-1].reserve_breach
    book.update_reservation(reservation.reservation_id, ReservationStatus.RELEASED,
                            actor="ops", reason="cancelled")
    assert book.balance("a", "USD").available == 1000
    assert book.audit() == ()


def test_benchmark_metrics_and_rolling_beta():
    portfolio = [0.02, -0.01, 0.03, 0.01]
    benchmark = [0.01, -0.02, 0.02, 0.0]
    metrics = benchmark_metrics(portfolio, benchmark)
    assert metrics.observations == 4
    assert metrics.tracking_error > 0
    assert metrics.beta > 0
    assert len(rolling_beta(portfolio, benchmark, window=2)) == 3


def test_simplex_projection_and_inverse_vol_weights():
    weights = project_to_simplex({"A": 0.8, "B": 0.8, "C": -0.2}, total=1,
                                 lower={"A": 0, "B": 0, "C": 0},
                                 upper={"A": 0.6, "B": 0.6, "C": 0.6})
    assert sum(weights.values()) == pytest.approx(1)
    assert max(weights.values()) <= 0.6
    inv = inverse_volatility_weights({"A": 0.1, "B": 0.2})
    assert inv["A"] == pytest.approx(2 * inv["B"])
    diagnostic = diagnose_allocation(weights, target_total=1)
    assert diagnostic.feasible
