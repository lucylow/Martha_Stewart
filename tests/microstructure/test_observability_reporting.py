import unittest
from decimal import Decimal
from dataclasses import replace

from martha_plus.microstructure.clock import LatencyKind, LatencyModel, LatencyProfile
from martha_plus.microstructure.matching.engine import MatchingEngine
from martha_plus.microstructure.observability import (
    AlertEvaluator, AlertRule, AuditTrail, FeedHealthEvaluator,
    HashChainJournal, HealthState, HealthThresholds, SimulationCircuitBreaker,
)
from martha_plus.microstructure.reporting import ReportExporter, RunSummaryBuilder
from martha_plus.microstructure.replay.runner import MarketReplay
from martha_plus.microstructure.data.contracts import FeedHealth
from martha_plus.microstructure.types import (
    EventKind, MarketEvent, OrderRequest, OrderType, Side, SymbolSpec,
)


class BidOnce:
    strategy_id = "report-strategy"

    def __init__(self):
        self.sent = False

    def on_market_data(self, event, context):
        if self.sent or context.top.bid is None or context.top.ask is None:
            return ()
        self.sent = True
        return (OrderRequest("report-child", context.symbol, Side.BUY,
                             OrderType.LIMIT, 1,
                             limit_price_ticks=context.top.bid.price_ticks,
                             strategy_id=self.strategy_id),)


class ObservabilityTests(unittest.TestCase):
    def test_journal_hash_chain_round_trip(self):
        journal = HashChainJournal(run_id="test-run", created_at_ns=1)
        audit = AuditTrail(journal)
        audit.record_order(2, "o-1", "new", 4, symbol="abc", strategy_id="s1")
        audit.record_risk(3, "o-1", True, ())
        self.assertTrue(journal.verify(expected_final_hash=journal.last_hash).valid)
        restored = HashChainJournal.from_jsonl(journal.to_jsonl())
        self.assertTrue(restored.verify().valid)
        self.assertEqual(restored.last_hash, journal.last_hash)

    def test_journal_detects_mutated_entry(self):
        journal = HashChainJournal(run_id="tamper", created_at_ns=0)
        journal.append(1, "configuration", {"mode": "paper"})
        original = journal.entries[0]
        journal._entries[0] = replace(original, payload={"mode": "live"})
        self.assertFalse(journal.verify().valid)

    def test_feed_health_reports_gap_as_unhealthy(self):
        feed = FeedHealth("provider", "book", 100, 95, 5, 0, 1, 0,
                          100, 10_000, 10_100, 50.0)
        evaluator = FeedHealthEvaluator(HealthThresholds(max_feed_age_ns=1_000,
                                                          max_gap_count=0))
        report = evaluator.report(feed, 10_500)
        self.assertEqual(report.overall, HealthState.UNHEALTHY)
        self.assertFalse(report.healthy)

    def test_alert_evaluator_and_simulation_breaker(self):
        rules = (AlertRule("rejects-high", "reject_rate", "gt", Decimal("0.1")),)
        alerts = AlertEvaluator(rules).evaluate({"reject_rate": Decimal("0.2")})
        self.assertEqual(len(alerts), 1)
        breaker = SimulationCircuitBreaker(max_consecutive_rejections=1, max_anomalies=0)
        self.assertTrue(breaker.record_order_result(False))
        self.assertFalse(breaker.record_order_result(False))
        self.assertEqual(breaker.reason, "consecutive_rejections")
        breaker.reset()
        self.assertFalse(breaker.tripped)


class ReportingTests(unittest.TestCase):
    def make_result(self):
        zero = LatencyProfile(0, 0, 0, 0)
        latency = LatencyModel({kind: zero for kind in LatencyKind})
        events = (
            MarketEvent(1, 1, "ABC", EventKind.ADD, Side.BUY, 99, 2, "ext-bid"),
            MarketEvent(2, 2, "ABC", EventKind.ADD, Side.SELL, 101, 2, "ext-ask"),
            MarketEvent(3, 3, "ABC", EventKind.TRADE, Side.SELL, 99, 3),
        )
        return MarketReplay(MatchingEngine(SymbolSpec("ABC")), BidOnce(),
                            latency=latency).run(events)

    def test_run_summary_uses_strategy_liquidity_role_and_owner_ledger(self):
        result = self.make_result()
        summary = RunSummaryBuilder().from_run_result(result)
        self.assertEqual(summary.symbol, "ABC")
        self.assertEqual(summary.maker_quantity, 1)
        self.assertEqual(summary.taker_quantity, 0)
        self.assertEqual(summary.maker_fraction, Decimal(1))
        self.assertIsNotNone(summary.final_cash)
        self.assertIsNotNone(summary.final_equity)

    def test_json_csv_html_exports(self):
        summary = RunSummaryBuilder().from_run_result(self.make_result())
        self.assertIn('"symbol": "ABC"', ReportExporter.to_json(summary))
        self.assertIn("metric,value", ReportExporter.to_csv(summary))
        page = ReportExporter.to_html(summary, title="Test <report>")
        self.assertIn("Test &lt;report&gt;", page)
        self.assertNotIn("Test <report>", page)


if __name__ == "__main__":
    unittest.main()
