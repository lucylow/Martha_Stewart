import unittest
from martha_plus.strategies import (BreakoutStrategy, MeanReversionStrategy,
                                    MomentumStrategy, PairsZScoreStrategy,
                                    VolatilityTargetStrategy, WeightedEnsemble)
from tests.helpers import bars

class StrategyTests(unittest.TestCase):
    def test_momentum_returns_typed_signal(self):
        data = bars(count=100)
        signal = MomentumStrategy(fast_window=5, slow_window=20).generate("TEST", data)
        self.assertEqual(signal.symbol, "TEST")
        self.assertGreaterEqual(signal.strength, 0)
        self.assertLessEqual(signal.strength, 1)
        self.assertIn("edge_bps", signal.metadata)

    def test_momentum_rejects_short_history(self):
        with self.assertRaises(ValueError):
            MomentumStrategy(5, 20).generate("TEST", bars(count=10))

    def test_mean_reversion_long_only(self):
        signal = MeanReversionStrategy(window=20, allow_short=False).generate("TEST", bars(count=60))
        self.assertGreaterEqual(signal.target_weight, 0)

    def test_mean_reversion_invalid_threshold(self):
        with self.assertRaises(ValueError):
            MeanReversionStrategy(window=20, entry_z=0.5, exit_z=1.0)

    def test_breakout_signal_bounds(self):
        signal = BreakoutStrategy(lookback=15, atr_window=10).generate("TEST", bars(count=60))
        self.assertLessEqual(abs(signal.target_weight), 0.15)
        self.assertIn("atr_fraction", signal.metadata)

    def test_volatility_target_signal_bounds(self):
        signal = VolatilityTargetStrategy(momentum_window=10, vol_window=10).generate("TEST", bars(count=60))
        self.assertLessEqual(abs(signal.target_weight), 0.25)
        self.assertIn("ewma_volatility", signal.metadata)

    def test_ensemble_clips_combined_weight(self):
        data = bars(count=100)
        ensemble = WeightedEnsemble([(MomentumStrategy(5, 20, max_weight=0.7), 1)], max_abs_weight=0.1)
        signal = ensemble.generate("TEST", data)
        self.assertLessEqual(abs(signal.target_weight), 0.1)

    def test_ensemble_requires_positive_weights(self):
        with self.assertRaises(ValueError):
            WeightedEnsemble([(MomentumStrategy(), 0)])

    def test_pairs_requires_aligned_timestamps(self):
        left = bars("LEFT", 80)
        right = bars("RIGHT", 80, seed=6)
        strategy = PairsZScoreStrategy(lookback=30)
        signal = strategy.generate("LEFT", left, right)
        self.assertEqual(signal.metadata["paired_symbol"], "RIGHT")
        changed = list(right)
        from dataclasses import replace
        changed[-1] = replace(changed[-1], timestamp=changed[-1].timestamp.replace(day=2))
        with self.assertRaises(ValueError):
            strategy.generate("LEFT", left, changed)

if __name__ == "__main__":
    unittest.main()
