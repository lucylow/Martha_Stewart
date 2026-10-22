import unittest
from martha_plus.backtest import MultiAssetBacktestEngine
from martha_plus.strategies import MomentumStrategy
from tests.helpers import bars, settings

class MultiAssetBacktestTests(unittest.TestCase):
    def test_shared_capital_result_contains_both_symbols(self):
        first = bars("AAA", 120, seed=11)
        second = bars("BBB", 120, seed=12)
        result = MultiAssetBacktestEngine(settings()).run(
            {"AAA": first, "BBB": second},
            {"AAA": MomentumStrategy(5, 20), "BBB": MomentumStrategy(5, 20)},
            minimum_history=30)
        self.assertEqual(result.symbols, ("AAA", "BBB"))
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIn("max_drawdown", result.metrics)
        self.assertEqual(result.to_dict()["symbols"], ["AAA", "BBB"])

    def test_multi_asset_rejects_missing_strategy(self):
        with self.assertRaises(ValueError):
            MultiAssetBacktestEngine().run({"AAA": bars("AAA", 80)}, {}, minimum_history=30)

    def test_multi_asset_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            MultiAssetBacktestEngine().run({"AAA": bars("AAA", 20)},
                                           {"AAA": MomentumStrategy()}, minimum_history=30)

    def test_multi_asset_rejects_empty_universe(self):
        with self.assertRaises(ValueError):
            MultiAssetBacktestEngine().run({}, {})

if __name__ == "__main__":
    unittest.main()
