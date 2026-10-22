import unittest
from martha_plus.backtest.validation import (block_bootstrap_returns, parameter_sweep,
                                              transaction_cost_sensitivity)
from martha_plus.risk import (PortfolioRiskMonitor, downside_deviation, expected_shortfall,
                              gain_to_pain_ratio, historical_var, omega_ratio, tail_report)
from martha_plus.execution.costs import estimate_costs
from martha_plus.portfolio import PortfolioLedger
from martha_plus.risk import KillSwitch
from martha_plus.strategies import MomentumStrategy
from tests.helpers import bars, settings

class ValidationTests(unittest.TestCase):
    def test_tail_risk_helpers(self):
        sample = [-0.1, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04]
        self.assertGreater(historical_var(sample, 0.9), 0)
        self.assertGreater(expected_shortfall(sample, 0.9), 0)
        self.assertGreater(downside_deviation(sample), 0)
        self.assertGreater(omega_ratio(sample), 0)
        self.assertGreater(gain_to_pain_ratio(sample), 0)
        self.assertEqual(tail_report(sample)["observations"], len(sample))

    def test_tail_risk_validates_input(self):
        with self.assertRaises(ValueError):
            historical_var([], 0.95)
        with self.assertRaises(ValueError):
            expected_shortfall([0.01], 1.0)

    def test_portfolio_risk_monitor_halts_at_drawdown(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            config = settings(max_drawdown_fraction=0.1, kill_switch_file=temp + "/HALT")
            kill = KillSwitch(config.kill_switch_file)
            monitor = PortfolioRiskMonitor(config, kill)
            state = monitor.update(100_000)
            self.assertFalse(state.halted)
            state = monitor.update(88_000)
            self.assertTrue(state.halted)
            self.assertIn("drawdown_limit", state.reasons)

    def test_portfolio_risk_monitor_requires_ack_for_reset(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            monitor = PortfolioRiskMonitor(settings(kill_switch_file=temp + "/HALT"), KillSwitch(temp + "/HALT"))
            with self.assertRaises(PermissionError):
                monitor.reset_high_water_mark(90_000)
            monitor.reset_high_water_mark(90_000, operator_acknowledged=True)

    def test_nonpositive_equity_triggers_halt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            kill = KillSwitch(temp + "/HALT")
            monitor = PortfolioRiskMonitor(settings(kill_switch_file=temp + "/HALT"), kill)
            with self.assertRaises(ValueError):
                monitor.update(0)
            self.assertTrue(kill.active)

    def test_cost_estimator_with_impact(self):
        result = estimate_costs(10_000, commission_bps=1, spread_bps=2,
                                slippage_bps=3, average_daily_dollar_volume=1_000_000)
        self.assertGreater(result.total_cost, 0)
        self.assertGreater(result.effective_cost_bps, 0)
        self.assertGreater(result.slippage_cost, 3)

    def test_cost_estimator_zero_notional(self):
        result = estimate_costs(0)
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(result.effective_cost_bps, 0)

    def test_cost_estimator_rejects_invalid_volume(self):
        with self.assertRaises(ValueError):
            estimate_costs(1000, average_daily_dollar_volume=0)

    def test_bootstrap_is_seeded(self):
        returns_ = [0.01, -0.02, 0.03, -0.01]
        first = block_bootstrap_returns(returns_, simulations=100, seed=7)
        second = block_bootstrap_returns(returns_, simulations=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["simulations"], 100)

    def test_bootstrap_rejects_invalid_returns(self):
        with self.assertRaises(ValueError):
            block_bootstrap_returns([], simulations=10)
        with self.assertRaises(ValueError):
            block_bootstrap_returns([-1.1])

    def test_parameter_sweep_orders_best_score_first(self):
        data = bars(count=100)
        results = parameter_sweep(data, [(3, 15), (5, 20)],
                                 lambda params: MomentumStrategy(params[0], params[1]),
                                 metric="total_return", settings=settings(), minimum_history=30)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_cost_sensitivity_returns_grid(self):
        data = bars(count=100)
        results = transaction_cost_sensitivity(data, lambda: MomentumStrategy(5, 20),
                    [0, 1], [0, 3], base_settings=settings(), minimum_history=30)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].commission_bps, 0)

if __name__ == "__main__":
    unittest.main()
