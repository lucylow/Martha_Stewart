import unittest
from decimal import Decimal

from martha_plus.microstructure.risk import (
    DrawdownTracker, Mark, MarketShock, PositionBook, PortfolioRiskMonitor,
    RiskLimit, RiskSeverity, ScenarioLibrary, ShockType, StressEngine,
    StressScenario, concentration_report,
)
from martha_plus.microstructure.types import Side, SymbolSpec


class PortfolioBookTests(unittest.TestCase):
    def setUp(self):
        self.book = PositionBook({"ABC": SymbolSpec("ABC", tick_size=Decimal("0.01"))})

    def test_partial_close_realizes_profit_and_revalues_residual(self):
        self.book.apply_fill("ABC", Side.BUY, 10, 100)
        self.book.mark(Mark("ABC", Decimal(110), 1))
        self.book.apply_fill("ABC", Side.SELL, 4, 120)
        row = self.book.valuation("ABC")
        self.assertEqual(row.realized_pnl, Decimal("0.80"))
        self.assertEqual(row.unrealized_pnl, Decimal("0.60"))
        self.assertEqual(row.total_pnl, Decimal("1.40"))

    def test_short_position_has_negative_market_value(self):
        self.book.apply_fill("ABC", Side.SELL, 2, 100)
        self.book.mark(Mark("ABC", Decimal(90), 1))
        self.assertLess(self.book.valuation("ABC").market_value, 0)
        self.assertGreater(self.book.valuation("ABC").unrealized_pnl, 0)

    def test_drawdown_tracker(self):
        tracker = DrawdownTracker(Decimal(100))
        tracker.update(1, 110)
        row = tracker.update(2, 88)
        self.assertEqual(row.drawdown, Decimal(22))
        self.assertEqual(tracker.maximum_fraction, Decimal("0.2"))

    def test_risk_monitor_and_concentration(self):
        monitor = PortfolioRiskMonitor((RiskLimit("gross", Decimal(100)),))
        breaches = monitor.evaluate({"gross": Decimal(120)})
        self.assertEqual(len(breaches), 1)
        report = concentration_report({"A": Decimal(90), "B": Decimal(10)})
        self.assertEqual(report.largest_weight, Decimal("0.9"))


class StressTests(unittest.TestCase):
    def test_price_shock_changes_marked_value(self):
        scenario = StressScenario("shock", "test", (
            MarketShock("ABC", ShockType.PRICE_RETURN, Decimal("-0.1")),
        ))
        result = StressEngine().run({"ABC": 10}, {"ABC": Decimal(100)}, scenario, cash=Decimal(50))
        self.assertEqual(result.base_value, Decimal(1050))
        self.assertEqual(result.stressed_value, Decimal(950))
        self.assertEqual(result.pnl_change, Decimal(-100))

    def test_library_scenarios_are_explicit(self):
        scenario = ScenarioLibrary.liquidity_withdrawal(("ABC",))
        self.assertEqual(len(scenario.shocks), 2)


if __name__ == "__main__":
    unittest.main()
