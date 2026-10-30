import tempfile
import unittest
from pathlib import Path

from martha_plus.microstructure.data import (
    EventArchiveReader, EventArchiveWriter, FeedEnvelope, FeedSequenceMonitor,
    MarketDataNormalizer, MarketEventValidator, PriceConverter, QualityCode,
    TimestampConverter,
)
from martha_plus.microstructure.types import EventKind, MarketEvent, Side, SymbolSpec


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = SymbolSpec("ABC")

    def test_price_and_timestamp_conversion(self):
        converter = PriceConverter(self.spec)
        self.assertEqual(converter.to_ticks("12.34"), 1234)
        self.assertEqual(str(converter.from_ticks(1234)), "12.34")
        self.assertEqual(TimestampConverter.to_ns(2.5, "ms"), 2_500_000)
        with self.assertRaises(ValueError):
            converter.to_ticks("12.345")

    def test_normalizer_sorts_deduplicates_and_audits(self):
        normalizer = MarketDataNormalizer(self.spec)
        rows = [
            {"timestamp": 2, "timestamp_unit": "ms", "sequence": 2,
             "kind": "add", "side": "buy", "price": "10.00", "quantity": 2,
             "order_id": "b2"},
            {"timestamp": 1, "timestamp_unit": "ms", "sequence": 1,
             "kind": "add", "side": "sell", "price": "10.02", "quantity": 3,
             "order_id": "a1"},
            {"timestamp": 2, "timestamp_unit": "ms", "sequence": 2,
             "kind": "add", "side": "buy", "price": "10.00", "quantity": 2,
             "order_id": "b2"},
        ]
        result = normalizer.normalize(rows)
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.events[0].ts_ns, 1_000_000)
        self.assertEqual(result.audit.duplicate_count, 1)

    def test_strict_normalizer_rejects_bad_batch(self):
        normalizer = MarketDataNormalizer(self.spec, strict=True)
        with self.assertRaises(ValueError):
            normalizer.normalize([{"kind": "not-real"}])

    def test_event_validator_rejects_symbol_mismatch(self):
        validator = MarketEventValidator("ABC")
        result = validator.validate_event(MarketEvent(1, 1, "XYZ", EventKind.HALT))
        self.assertFalse(result.accepted)
        self.assertEqual(result.errors[0].code, QualityCode.SYMBOL_MISMATCH)

    def test_sequence_monitor_detects_gap(self):
        monitor = FeedSequenceMonitor()
        first = MarketEvent(1, 0, "ABC", EventKind.HALT)
        second = MarketEvent(2, 1, "ABC", EventKind.HALT)
        monitor.observe(FeedEnvelope("feed", "book", 1, first, provider_sequence=10))
        issues = monitor.observe(FeedEnvelope("feed", "book", 2, second, provider_sequence=12))
        self.assertEqual(issues[0].code, QualityCode.SEQUENCE_GAP)
        self.assertEqual(monitor.gaps, 1)

    def test_archive_detects_record_tampering(self):
        event = MarketEvent(1, 1, "ABC", EventKind.ADD, Side.BUY, 100, 3, "b1")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            with EventArchiveWriter(path) as writer:
                writer.write(event)
            content = path.read_text(encoding="utf-8")
            content = content.replace('"price_ticks":100', '"price_ticks":101')
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                EventArchiveReader(path).read_all()

    def test_archive_round_trip_and_atomic_commit(self):
        events = (
            MarketEvent(1, 1, "ABC", EventKind.ADD, Side.BUY, 100, 3, "b1"),
            MarketEvent(2, 2, "ABC", EventKind.CANCEL, order_id="b1"),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            with EventArchiveWriter(path, metadata={"run": "unit-test"}) as writer:
                writer.write_many(events)
            reader = EventArchiveReader(path)
            self.assertEqual(reader.read_all(), events)
            self.assertTrue(reader.verify_ordering())
            self.assertEqual(reader.metadata["run"], "unit-test")


if __name__ == "__main__":
    unittest.main()
