import math
import unittest
from martha_plus.indicators import (atr, bollinger_bands, ema, ewma_volatility, returns,
                                    rolling_beta, rolling_std, rsi, sma, zscore_series)
from martha_plus.math_utils import (annualized_volatility, compound_returns, correlation,
                                    max_drawdown, percentile, sharpe_ratio, stddev)

class IndicatorTests(unittest.TestCase):
    def test_returns(self):
        self.assertAlmostEqual(returns([100, 110, 99])[0], 0.1)
        with self.assertRaises(ValueError):
            returns([1, 0, 2])

    def test_sma_window_alignment(self):
        self.assertEqual(sma([1, 2, 3, 4], 3), [None, None, 2, 3])

    def test_ema_seeds_with_initial_mean(self):
        values = ema([1, 2, 3, 4, 5], 3)
        self.assertEqual(values[:2], [None, None])
        self.assertAlmostEqual(values[2], 2)
        self.assertAlmostEqual(values[-1], 4)

    def test_rolling_std_short_window(self):
        result = rolling_std([2, 2, 2, 2], 2)
        self.assertEqual(result[0], None)
        self.assertEqual(result[-1], 0)

    def test_zscore_series_constant(self):
        self.assertEqual(zscore_series([5, 5, 5], 3), [None, None, 0.0])

    def test_rsi_monotone_up(self):
        result = rsi(list(range(1, 20)), 5)
        self.assertEqual(result[-1], 100)

    def test_atr(self):
        high = [11, 12, 13, 14]
        low = [9, 10, 11, 12]
        close = [10, 11, 12, 13]
        output = atr(high, low, close, 3)
        self.assertIsNone(output[0])
        self.assertAlmostEqual(output[-1], 2.0)

    def test_bollinger_bands_output_lengths(self):
        lower, middle, upper = bollinger_bands([1, 2, 3, 4, 5], 3)
        self.assertEqual(len(lower), 5)
        self.assertLess(lower[-1], middle[-1])
        self.assertGreater(upper[-1], middle[-1])

    def test_ewma_volatility(self):
        vol = ewma_volatility([0.01, 0.02, -0.01])
        self.assertEqual(len(vol), 3)
        self.assertTrue(all(v is not None and v > 0 for v in vol))

    def test_rolling_beta(self):
        benchmark = [0.01, 0.02, -0.01, 0.03]
        asset = [2*x for x in benchmark]
        beta = rolling_beta(asset, benchmark, 3)
        self.assertIsNone(beta[0])
        self.assertAlmostEqual(beta[-1], 2)

    def test_invalid_windows_fail(self):
        for func in (sma, ema, rolling_std, zscore_series):
            with self.assertRaises(ValueError):
                func([1, 2], 0)

    def test_statistics_helpers(self):
        self.assertAlmostEqual(stddev([1, 3]), math.sqrt(2))
        self.assertAlmostEqual(correlation([1, 2, 3], [2, 4, 6]), 1)
        self.assertEqual(percentile([1, 2, 3], 50), 2)
        self.assertAlmostEqual(compound_returns([0.1, -0.1]), -0.01)
        self.assertGreater(annualized_volatility([0.01, -0.01]), 0)
        self.assertGreater(sharpe_ratio([0.01, 0.02, 0.0]), 0)

    def test_max_drawdown(self):
        value, peak, trough = max_drawdown([100, 120, 90, 110])
        self.assertAlmostEqual(value, 0.25)
        self.assertEqual((peak, trough), (1, 2))

if __name__ == "__main__":
    unittest.main()
