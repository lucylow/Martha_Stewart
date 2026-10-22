import unittest
from martha_plus.backtest import BacktestEngine, calculate_metrics, trade_statistics
from martha_plus.backtest.walk_forward import evaluate_walk_forward, walk_forward_folds
from martha_plus.strategies import MomentumStrategy, MeanReversionStrategy
from tests.helpers import bars, settings

class BacktestTests(unittest.TestCase):
    def test_backtest_is_repeatable(self):
        data = bars(count=180, seed=22)
        engine = BacktestEngine(settings(), spread_bps=2)
        first = engine.run(data, MomentumStrategy(5, 20), minimum_history=30)
        second = engine.run(data, MomentumStrategy(5, 20), minimum_history=30)
        self.assertEqual(len(first.equity_curve), len(second.equity_curve))
        self.assertAlmostEqual(first.equity_curve[-1].equity, second.equity_curve[-1].equity)
        self.assertGreater(first.metrics["observations"], 0)

    def test_backtest_rejects_empty_bars(self):
        with self.assertRaises(ValueError):
            BacktestEngine().run([], MomentumStrategy())

    def test_backtest_requires_enough_bars(self):
        with self.assertRaises(ValueError):
            BacktestEngine().run(bars(count=20), MomentumStrategy(), minimum_history=35)

    def test_csv_backtest_api_shape(self):
        data = bars(count=100)
        result = BacktestEngine(settings()).run(data, MeanReversionStrategy(window=20), minimum_history=30)
        payload = result.to_dict()
        self.assertIn("metrics", payload)
        self.assertIn("orders", payload)
        self.assertIn("max_drawdown", payload["metrics"])

    def test_performance_metrics_empty_sample(self):
        result = calculate_metrics([])
        self.assertEqual(result["observations"], 0)
        self.assertEqual(result["sharpe"], 0)

    def test_metrics_positive_return(self):
        result = calculate_metrics([100, 102, 101, 110])
        self.assertGreater(result["total_return"], 0)
        self.assertGreaterEqual(result["max_drawdown"], 0)

    def test_trade_statistics(self):
        result = trade_statistics([10, -5, 15, 0])
        self.assertEqual(result["trade_count"], 4)
        self.assertEqual(result["win_count"], 2)
        self.assertAlmostEqual(result["win_rate"], 0.5)
        self.assertEqual(result["profit_factor"], 5)

    def test_walk_forward_folds_are_chronological(self):
        values = list(range(30))
        folds = walk_forward_folds(values, train_size=10, test_size=5)
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0].train[-1], 9)
        self.assertEqual(folds[0].test[0], 10)
        self.assertEqual(folds[1].test[0], 15)

    def test_rolling_window_walk_forward(self):
        folds = walk_forward_folds(list(range(30)), 10, 5, step_size=5, expanding=False)
        self.assertEqual(len(folds[1].train), 10)
        self.assertEqual(folds[1].train[0], 5)

    def test_walk_forward_parameter_selection(self):
        values = list(range(24))
        def score(sample, parameter):
            return sum(sample) * parameter
        results = evaluate_walk_forward(values, [1, 2, 3], score, 8, 4)
        self.assertTrue(results)
        self.assertTrue(all(result.parameters == 3 for result in results))
        self.assertEqual(results[0].test_count, 4)

    def test_walk_forward_rejects_empty_parameter_grid(self):
        with self.assertRaises(ValueError):
            evaluate_walk_forward(list(range(10)), [], lambda a, b: 0, 5, 2)

if __name__ == "__main__":
    unittest.main()
