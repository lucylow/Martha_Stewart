import unittest
from datetime import date, datetime, time, timedelta, timezone
from martha_plus.advanced.data_sources import (StaticMarketDataProvider, FailoverMarketDataProvider,
    DataQualityPolicy, MarketDataUnavailable, CachedMarketDataProvider)
from martha_plus.advanced.data_quality_monitor import DataQualityMonitor, QualityPolicy
from martha_plus.advanced.universe import SecurityRecord, UniversePolicy, UniverseBuilder
from martha_plus.advanced.corporate_actions import SplitEvent, DividendEvent, adjust_for_splits, total_return_series
from martha_plus.advanced.session_calendar import SessionCalendar
from martha_plus.advanced.feature_store import LocalFeatureStore, FeatureStoreError
from .helpers import NOW, make_bars, make_quote

class DataOpsTests(unittest.TestCase):
    def test_failover_skips_stale_quote(self):
        stale = StaticMarketDataProvider(quotes=[make_quote(at=NOW-timedelta(days=1))], name="stale")
        good = StaticMarketDataProvider(quotes=[make_quote(at=NOW)], name="good")
        provider = FailoverMarketDataProvider([stale, good], policy=DataQualityPolicy(max_quote_age=timedelta(seconds=30)), now=lambda: NOW)
        self.assertEqual(provider.get_quote("AAA").symbol, "AAA")
        self.assertFalse(provider.history[0].ok)
        self.assertTrue(provider.history[1].ok)

    def test_all_data_sources_fail_closed(self):
        provider = FailoverMarketDataProvider([StaticMarketDataProvider()], now=lambda: NOW)
        with self.assertRaises(MarketDataUnavailable): provider.get_quote("MISSING")

    def test_cache_hits(self):
        provider = StaticMarketDataProvider(quotes=[make_quote(at=NOW)])
        tick = {"t":0.0}
        cache = CachedMarketDataProvider(provider, quote_ttl=3, monotonic_clock=lambda: tick["t"])
        cache.get_quote("AAA"); cache.get_quote("AAA")
        self.assertEqual(cache.cache_info()["hits"], 1)

    def test_quality_monitor_detects_large_return(self):
        bars = make_bars(closes=[100]*20 + [200] + [201]*5, start=NOW-timedelta(days=25))
        result = DataQualityMonitor(QualityPolicy(max_abs_return=.2, stale_after_seconds=99999999)).analyze_bars(
            bars, now=NOW, expected_interval_seconds=86400)
        self.assertTrue(any(item.code == "extreme_return" for item in result.findings))

    def test_universe_builder_and_slots_fingerprint(self):
        builder = UniverseBuilder(UniversePolicy(minimum_price=5, excluded_symbols=frozenset({"BAD"})))
        snapshot = builder.build([
            SecurityRecord("good", date(2025,1,1), price=10, market_cap=100),
            SecurityRecord("BAD", date(2025,1,1), price=20),
            SecurityRecord("cheap", date(2025,1,1), price=2)], date(2025,1,2))
        self.assertEqual(snapshot.symbols, ("GOOD",))
        self.assertTrue(snapshot.policy_fingerprint)
        self.assertIn("BAD", snapshot.excluded)

    def test_split_adjustment_and_dividends(self):
        bars = make_bars(closes=[100,102,51,53], start=datetime(2025,1,1,tzinfo=timezone.utc))
        split = SplitEvent("AAA", date(2025,1,3), 2.0)
        adjusted = adjust_for_splits(bars, [split])
        self.assertAlmostEqual(adjusted.bars[0].close, 50)
        dividend = DividendEvent("AAA", date(2025,1,3), .5)
        returns = total_return_series(bars, [dividend])
        self.assertAlmostEqual(returns[2], (51 + .5) / 102 - 1)

    def test_session_calendar_holiday_early_close(self):
        calendar = SessionCalendar(holidays={date(2025,1,1)}, early_closes={date(2025,7,3):time(13,0)})
        self.assertIsNone(calendar.session(date(2025,1,1)))
        session = calendar.session(date(2025,7,3))
        self.assertTrue(session.early_close)
        self.assertEqual(session.close_at.hour, 13)
        self.assertFalse(calendar.is_open(datetime(2025,7,3,18,0,tzinfo=timezone.utc)))

    def test_feature_store_integrity(self):
        import tempfile
        from martha_plus.advanced.features import FeaturePipeline
        with tempfile.TemporaryDirectory() as d:
            store = LocalFeatureStore(d)
            rows = FeaturePipeline().transform(make_bars(closes=[100,101,102]))
            info = store.write("sample", rows, source_fingerprint="bars-hash")
            loaded, loaded_info = store.read("sample")
            self.assertEqual(info.checksum, loaded_info.checksum)
            self.assertEqual(len(loaded), 3)
            with self.assertRaises(FileExistsError): store.write("sample", rows, source_fingerprint="x")
            data_path = __import__('pathlib').Path(d)/"sample.jsonl"
            data_path.write_text("tampered")
            with self.assertRaises(FeatureStoreError): store.read("sample")

if __name__ == "__main__": unittest.main()
