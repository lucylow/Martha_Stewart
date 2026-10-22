import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from martha_plus.execution.rate_limiter import TokenBucket
from martha_plus.marketdata.microstructure import compute_features, order_flow_imbalance, summarize_quote_sequence
from martha_plus.persistence.audit_log import AuditLog, AuditLogError
from martha_plus.services.batch import BatchProcessor
from martha_plus.types import Quote
from tests.helpers import bars, settings
from martha_plus.clock import FrozenClock
from martha_plus.marketdata.synthetic import SyntheticMarketData
from martha_plus.risk import KillSwitch
from martha_plus.services import TradingResearchEngine
from martha_plus.strategies import MomentumStrategy

class OperationalSafeguardTests(unittest.TestCase):
    def test_token_bucket_capacity_and_refill(self):
        class FakeTime:
            def __init__(self): self.value = 100.0
            def __call__(self): return self.value
        clock = FakeTime()
        bucket = TokenBucket(2, 1, clock=clock)
        self.assertTrue(bucket.try_acquire())
        self.assertTrue(bucket.try_acquire())
        self.assertFalse(bucket.try_acquire())
        clock.value += 1.5
        self.assertTrue(bucket.try_acquire())

    def test_token_bucket_rejects_invalid_request(self):
        bucket = TokenBucket(2, 1)
        with self.assertRaises(ValueError): bucket.try_acquire(0)
        with self.assertRaises(ValueError): bucket.try_acquire(3)

    def test_microprice_moves_toward_larger_bid_queue(self):
        now = datetime.now(timezone.utc)
        quote = Quote("ABC", 99, 101, now, bid_size=90, ask_size=10)
        features = compute_features(quote)
        self.assertGreater(features.microprice, features.midpoint)
        self.assertGreater(features.depth_imbalance, 0)

    def test_microstructure_summary_validates_time_order(self):
        now = datetime.now(timezone.utc)
        q1 = Quote("ABC", 99, 101, now, 10, 20)
        q2 = Quote("ABC", 100, 102, now + timedelta(seconds=1), 20, 10)
        result = summarize_quote_sequence([q1, q2])
        self.assertEqual(result["count"], 2)
        self.assertNotEqual(result["cumulative_ofi"], 0)
        with self.assertRaises(ValueError): summarize_quote_sequence([q2, q1])

    def test_ofi_requires_same_symbol_and_increasing_time(self):
        now = datetime.now(timezone.utc)
        q1 = Quote("ABC", 99, 101, now, 10, 10)
        q2 = Quote("XYZ", 99, 101, now + timedelta(seconds=1), 10, 10)
        with self.assertRaises(ValueError): order_flow_imbalance(q1, q2)
        with self.assertRaises(ValueError): order_flow_imbalance(q1, q1)

    def test_audit_log_chain_verifies_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.jsonl"
            audit = AuditLog(path)
            audit.append("order_requested", {"symbol": "ABC", "quantity": 2})
            audit.append("order_filled", {"symbol": "ABC", "price": 100})
            report = audit.verify()
            self.assertTrue(report["valid"])
            self.assertEqual(report["entries"], 2)
            content = path.read_text(encoding="utf-8").replace('"price":100', '"price":101')
            path.write_text(content, encoding="utf-8")
            self.assertFalse(audit.verify()["valid"])
            with self.assertRaises(AuditLogError): audit.append("later", {})

    def test_batch_processor_isolates_symbol_failure(self):
        source = SyntheticMarketData(seed=123)
        source.generate("GOOD", 100)
        with tempfile.TemporaryDirectory() as temp:
            config = settings(kill_switch_file=temp + "/HALT")
            engine = TradingResearchEngine(config, source, MomentumStrategy(5, 20),
                       clock=FrozenClock(source.latest_quote("GOOD").timestamp),
                       kill_switch=KillSwitch(config.kill_switch_file))
            result = BatchProcessor(engine).run(["GOOD", "MISSING"])
            self.assertEqual(result.requested_count, 2)
            self.assertEqual(result.succeeded_count, 1)
            self.assertEqual(result.failed_count, 1)

    def test_batch_rejects_duplicate_symbols(self):
        source = SyntheticMarketData(seed=123)
        source.generate("GOOD", 100)
        with tempfile.TemporaryDirectory() as temp:
            config = settings(kill_switch_file=temp + "/HALT")
            engine = TradingResearchEngine(config, source, MomentumStrategy(5, 20),
                       clock=FrozenClock(source.latest_quote("GOOD").timestamp),
                       kill_switch=KillSwitch(config.kill_switch_file))
            with self.assertRaises(ValueError): BatchProcessor(engine).run(["GOOD", "good"])

if __name__ == "__main__":
    unittest.main()
