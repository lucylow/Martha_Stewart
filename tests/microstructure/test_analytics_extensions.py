from decimal import Decimal
import unittest

from martha_plus.microstructure.analytics.indicators import (
    AdverseSelectionTracker, BookFeatureExtractor, KyleLambdaEstimator,
    TradePrint, VolumeImbalanceTracker, microprice, top_imbalance,
)
from martha_plus.microstructure.analytics.rolling import (
    ExponentialMovingAverage, ExponentiallyWeightedVariance, ReturnSeries,
    RollingMoments, RollingQuantile, VolumeBarBuilder, WelfordAccumulator,
)
from martha_plus.microstructure.types import BookLevelView, BookTop, Side


class RollingStatisticsTests(unittest.TestCase):
    def test_rolling_moments_evict_old_values(self):
        stat = RollingMoments(3)
        stat.extend([1, 2, 3, 4])
        self.assertEqual(stat.count, 3)
        self.assertEqual(stat.mean, 3.0)
        self.assertAlmostEqual(stat.variance, 1.0)

    def test_welford_merge_matches_single_pass(self):
        left, right, full = WelfordAccumulator(), WelfordAccumulator(), WelfordAccumulator()
        for value in (1, 2, 3):
            left.update(value)
            full.update(value)
        for value in (4, 5, 6):
            right.update(value)
            full.update(value)
        left.merge(right)
        self.assertAlmostEqual(left.mean, full.mean)
        self.assertAlmostEqual(left.variance, full.variance)

    def test_ema_warmup_and_reset(self):
        ema = ExponentialMovingAverage(4, min_periods=2)
        self.assertIsNone(ema.update(10))
        self.assertIsNotNone(ema.update(12))
        ema.reset()
        self.assertIsNone(ema.value)

    def test_ew_variance_and_return_series(self):
        ew = ExponentiallyWeightedVariance(0.9)
        ew.update(1)
        self.assertIsNotNone(ew.update(2))
        returns = ReturnSeries(log_returns=True)
        self.assertIsNone(returns.update(100))
        self.assertGreater(returns.update(101), 0)
        self.assertGreater(returns.update(102), 0)
        self.assertIsNotNone(returns.annualized_volatility(252))

    def test_quantile_is_interpolated(self):
        q = RollingQuantile(5, 0.5)
        for value in [1, 3, 2]:
            q.update(value)
        self.assertEqual(q.value, 2)

    def test_volume_bars_split_large_trade_and_flush(self):
        builder = VolumeBarBuilder(10)
        bars = builder.update(1, 100, 25)
        self.assertEqual(len(bars), 2)
        self.assertEqual([bar.volume for bar in bars], [10, 10])
        tail = builder.flush()
        self.assertEqual(tail.volume, 5)


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.top = BookTop("ABC", 1, BookLevelView(99, 8, 2), BookLevelView(101, 2, 1))

    def test_microprice_and_imbalance(self):
        self.assertEqual(microprice(self.top), Decimal("100.6"))
        self.assertEqual(top_imbalance(self.top), Decimal("0.6"))

    def test_feature_extractor_updates_history(self):
        extractor = BookFeatureExtractor(history=5)
        row = extractor.update(self.top)
        self.assertEqual(row.spread_ticks, 2)
        self.assertEqual(len(extractor.history), 1)
        self.assertIsNotNone(extractor.mean_spread())

    def test_toxicity_tracker_uses_fixed_event_buckets(self):
        tracker = VolumeImbalanceTracker(bucket_trades=2)
        tracker.update(TradePrint(1, 100, 3, Side.BUY))
        result = tracker.update(TradePrint(2, 101, 1, Side.SELL))
        self.assertEqual(result.buckets, 1)
        self.assertEqual(result.total_volume, 4)
        self.assertEqual(result.normalized_imbalance, Decimal("0.5"))

    def test_kyle_lambda_requires_variation(self):
        estimator = KyleLambdaEstimator(10)
        estimator.update(100, 0)
        estimator.update(101, 5)
        estimator.update(102, 10)
        estimator.update(103, 15)
        self.assertIsNotNone(estimator.value)

    def test_adverse_selection_tracks_multiple_horizons(self):
        tracker = AdverseSelectionTracker((10, 20))
        tracker.record_fill(0, 100, Side.BUY, 2)
        tracker.observe_mid(10, 101)
        tracker.observe_mid(20, 99)
        summary = tracker.summary()
        self.assertEqual(summary[10]["count"], 1)
        self.assertEqual(summary[20]["count"], 1)
        self.assertGreater(Decimal(summary[10]["mean_markout_bps"]), 0)
        self.assertLess(Decimal(summary[20]["mean_markout_bps"]), 0)


if __name__ == "__main__":
    unittest.main()
