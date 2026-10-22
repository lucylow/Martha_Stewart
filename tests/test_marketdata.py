import csv
import tempfile
import unittest
from pathlib import Path
from datetime import timedelta
from martha_plus.errors import DataValidationError
from martha_plus.marketdata import (CSVBarProvider, QuoteCache, SyntheticMarketData,
                                   inspect_bars, resample_bars)
from tests.helpers import bars, quote

class MarketDataTests(unittest.TestCase):
    def test_seeded_synthetic_is_reproducible(self):
        first = SyntheticMarketData(seed=42).generate("abc", 12)
        second = SyntheticMarketData(seed=42).generate("ABC", 12)
        self.assertEqual([b.close for b in first], [b.close for b in second])

    def test_csv_provider_loads_grouped_bars(self):
        sample = Path(__file__).parent.parent / "examples" / "sample_bars.csv"
        provider = CSVBarProvider(sample)
        self.assertIn("DEMO", provider.symbols())
        self.assertEqual(len(provider.bars("DEMO")), 180)
        self.assertEqual(provider.latest_quote("DEMO").venue, "CSV_CLOSE_PROXY")

    def test_csv_provider_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            path.write_text("timestamp,symbol,close\n2024-01-01T00:00:00+00:00,X,2\n", encoding="utf-8")
            with self.assertRaises(DataValidationError):
                CSVBarProvider(path)

    def test_quote_cache_rejects_out_of_order_quote(self):
        now = quote().timestamp
        cache = QuoteCache(max_age_seconds=30)
        cache.update(quote(stamp=now))
        with self.assertRaises(DataValidationError):
            cache.update(quote(stamp=now - timedelta(seconds=1)))

    def test_quote_cache_rejects_stale_quote(self):
        now = quote().timestamp
        cache = QuoteCache(max_age_seconds=2)
        cache.update(quote(stamp=now - timedelta(seconds=5)))
        with self.assertRaises(DataValidationError):
            cache.get("TEST", now=now)

    def test_quote_cache_rejects_excessive_spread(self):
        from martha_plus.types import Quote
        q = Quote("WIDE", 90, 110)
        cache = QuoteCache(max_spread_bps=100)
        with self.assertRaises(DataValidationError):
            cache.update(q)

    def test_quality_report_for_clean_bars(self):
        values = bars(count=10)
        report = inspect_bars(values, expected_interval_seconds=86400)
        self.assertTrue(report.ok, report.to_dict())

    def test_quality_detects_interval_gap(self):
        values = bars(count=5)
        bad = [values[0], values[1], values[3]]
        report = inspect_bars(bad, expected_interval_seconds=86400, max_absolute_return=None)
        self.assertFalse(report.ok)
        self.assertIn("interval_gap", {issue.code for issue in report.issues})

    def test_resample_aggregates_ohlcv(self):
        values = bars(count=6)
        result = resample_bars(values, 2 * 86400)
        self.assertEqual(len(result), 4)  # fixed UTC epoch buckets can split the first interval
        self.assertEqual(result[0].open, values[0].open)
        self.assertEqual(sum(item.volume for item in result), sum(item.volume for item in values))
        self.assertTrue(all(item.high >= max(item.open, item.close) for item in result))

    def test_resample_rejects_unsorted(self):
        values = bars(count=4)
        with self.assertRaises(ValueError):
            resample_bars([values[1], values[0]], 86400)

if __name__ == "__main__":
    unittest.main()
