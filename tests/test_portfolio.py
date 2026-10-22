import unittest
from datetime import datetime, timezone, timedelta
from martha_plus.portfolio import (PortfolioLedger, equal_weight_targets,
                                   inverse_volatility_targets, normalize_weights)
from martha_plus.portfolio.analytics import drawdown_episodes, performance_summary, snapshot_returns
from martha_plus.risk.exposure import covariance_portfolio_volatility, exposure_report
from martha_plus.risk.stress import run_stress_scenario, run_stress_suite, standard_scenarios
from martha_plus.types import Fill, Side

class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.ledger = PortfolioLedger(10_000)
        t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.ledger.apply_fill(Fill("F1", "O1", "ABC", Side.BUY, 10, 100, 1, t))
        self.ledger.mark({"ABC": 110})

    def test_buy_fill_updates_cash_position_and_equity(self):
        position = self.ledger.positions()["ABC"]
        self.assertEqual(position.quantity, 10)
        self.assertEqual(position.average_cost, 100)
        self.assertEqual(position.market_value, 1100)
        self.assertAlmostEqual(self.ledger.cash, 8999)
        self.assertAlmostEqual(self.ledger.equity(), 10099)

    def test_duplicate_fill_is_idempotent(self):
        fill = self.ledger.fills[0]
        self.ledger.apply_fill(fill)
        self.assertEqual(len(self.ledger.fills), 1)
        self.assertEqual(self.ledger.positions()["ABC"].quantity, 10)

    def test_sell_realizes_profit(self):
        t = datetime(2024, 1, 2, tzinfo=timezone.utc)
        self.ledger.apply_fill(Fill("F2", "O2", "ABC", Side.SELL, 4, 110, 0.5, t))
        self.assertEqual(self.ledger.positions()["ABC"].quantity, 6)
        self.assertAlmostEqual(self.ledger.realized_pnl(t.date()), 39.5)

    def test_short_position_can_be_accounted_if_risk_layer_allows(self):
        t = datetime(2024, 1, 2, tzinfo=timezone.utc)
        book = PortfolioLedger(1000)
        book.apply_fill(Fill("SHORT", "O", "ABC", Side.SELL, 2, 50, 0, t))
        book.mark({"ABC": 40})
        self.assertEqual(book.positions()["ABC"].quantity, -2)
        self.assertEqual(book.positions()["ABC"].unrealized_pnl, 20)

    def test_snapshot_exposure_is_fractional(self):
        snapshot = self.ledger.snapshot()
        self.assertGreater(snapshot.gross_exposure, 0)
        self.assertAlmostEqual(snapshot.gross_exposure, abs(1100) / snapshot.equity)

    def test_target_allocators(self):
        self.assertEqual(equal_weight_targets(["A", "B"], 0.6), {"A": 0.3, "B": 0.3})
        self.assertAlmostEqual(sum(inverse_volatility_targets({"A": 0.1, "B": 0.2}).values()), 1)
        normalized = normalize_weights({"A": 0.8, "B": 0.8}, 1.0)
        self.assertAlmostEqual(sum(abs(v) for v in normalized.values()), 1)

    def test_exposure_report(self):
        report = exposure_report(self.ledger.snapshot())
        self.assertEqual(report["position_count"], 1)
        self.assertEqual(report["effective_position_count"], 1)

    def test_covariance_portfolio_volatility(self):
        matrix = {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.01, "B": 0.09}}
        result = covariance_portfolio_volatility({"A": 0.5, "B": 0.5}, matrix)
        self.assertGreater(result, 0)

    def test_covariance_missing_pair_raises(self):
        with self.assertRaises(ValueError):
            covariance_portfolio_volatility({"A": 1, "B": 1}, {"A": {"A": 0.1}})

    def test_stress_scenario(self):
        result = run_stress_scenario(self.ledger.snapshot(), {"ABC": -0.1}, "selloff")
        self.assertAlmostEqual(result.pnl, -110)
        self.assertLess(result.pnl_fraction_of_equity, 0)

    def test_stress_suite_standard_scenarios(self):
        scenarios = standard_scenarios(["ABC", "XYZ"])
        result = run_stress_suite(self.ledger.snapshot(), scenarios)
        self.assertEqual(len(result), 4)

    def test_performance_summary_and_drawdown_episodes(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        snapshots = []
        for i, equity in enumerate([10000, 10500, 9000, 9800, 11000]):
            snapshots.append(type("Snap", (), {"timestamp": base + timedelta(days=i),
                "equity": equity, "gross_exposure": 0.1, "net_exposure": 0.05})())
        summary = performance_summary(snapshots)
        self.assertAlmostEqual(summary["total_return"], 0.1)
        episodes = drawdown_episodes(snapshots)
        self.assertEqual(len(episodes), 1)
        self.assertIsNotNone(episodes[0].recovery_time)
        self.assertEqual(len(snapshot_returns(snapshots)), 4)

if __name__ == "__main__":
    unittest.main()
