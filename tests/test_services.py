import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from martha_plus.clock import FrozenClock
from martha_plus.marketdata.base import InMemoryMarketData
from martha_plus.marketdata.synthetic import SyntheticMarketData
from martha_plus.persistence import SQLiteStore
from martha_plus.services import HealthMonitor, TradingResearchEngine
from martha_plus.risk import KillSwitch
from martha_plus.strategies import MomentumStrategy
from martha_plus.types import Signal
from tests.helpers import settings

class ServiceTests(unittest.TestCase):
    def setUp(self):
        source = SyntheticMarketData(seed=23)
        source.generate("TEST", 100, drift=0.001, volatility=0.004)
        self.data = source
        last_time = source.latest_quote("TEST").timestamp
        self.clock = FrozenClock(last_time)
        self.temp = tempfile.TemporaryDirectory()
        self.config = settings(kill_switch_file=str(Path(self.temp.name) / "HALT"))
        self.engine = TradingResearchEngine(self.config, self.data, MomentumStrategy(5, 20),
                                            clock=self.clock, kill_switch=KillSwitch(self.config.kill_switch_file))

    def tearDown(self):
        self.temp.cleanup()

    def test_status_is_paper_only(self):
        status = self.engine.status()
        self.assertTrue(status["paper_only"])
        self.assertEqual(status["mode"], "paper")

    def test_process_returns_tick_result(self):
        result = self.engine.process("TEST")
        self.assertEqual(result.symbol, "TEST")
        self.assertIn(result.decision, {"no_order", "filled", "rejected", "accepted", "expired"})

    def test_kill_switch_blocks_new_order(self):
        class AlwaysLong:
            strategy_id = "test_long"
            def generate(self, symbol, history):
                return Signal(symbol, 0.2, 1.0, "test target", history[-1].timestamp)
        self.engine.strategy = AlwaysLong()
        self.engine.kill_switch.activate("test")
        result = self.engine.process("TEST")
        self.assertEqual(result.decision, "rejected")
        self.assertIn("trading_halted", result.reason)

    def test_store_records_ticks(self):
        store = SQLiteStore(Path(self.temp.name) / "engine.sqlite3")
        engine = TradingResearchEngine(self.config, self.data, MomentumStrategy(5, 20),
                                       store=store, clock=self.clock,
                                       kill_switch=KillSwitch(self.config.kill_switch_file))
        engine.process("TEST")
        events = store.list_events("strategy_tick")
        self.assertEqual(len(events), 1)
        store.close()

    def test_health_monitor_degraded_on_failed_probe(self):
        report = HealthMonitor(self.config, database_probe=lambda: False).report()
        self.assertEqual(report.status, "degraded")
        self.assertFalse(report.checks["database"])

    def test_health_probe_exception_is_safe(self):
        def broken():
            raise RuntimeError("private error")
        report = HealthMonitor(self.config, market_data_probe=broken).report()
        self.assertEqual(report.status, "degraded")
        self.assertNotIn("private error", str(report.to_dict()))

if __name__ == "__main__":
    unittest.main()
