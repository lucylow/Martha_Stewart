import unittest
from datetime import timedelta
from martha_plus.risk import (CircuitBreaker, KillSwitch, RiskContext, RiskLimits,
                             cap_quantity_by_limits, fixed_fractional_quantity,
                             target_weight_quantity, volatility_scaled_quantity)
from martha_plus.types import OrderRequest, Side
from tests.helpers import ledger, quote, settings

class RiskTests(unittest.TestCase):
    def setUp(self):
        self.config = settings()
        self.book = ledger()
        self.now = quote().timestamp
        self.q = quote(stamp=self.now)
        self.risk = RiskLimits(self.config)
        self.snapshot = self.book.snapshot(self.now)
        self.context = RiskContext(self.snapshot, self.q, now=self.now)

    def test_normal_order_approved(self):
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 10), self.context)
        self.assertTrue(decision.approved, decision.reason)

    def test_oversized_order_rejected(self):
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 100), self.context)
        self.assertFalse(decision.approved)
        self.assertIn("max_order_notional", decision.reason)

    def test_stale_quote_rejected(self):
        old = quote(stamp=self.now - timedelta(seconds=30))
        context = RiskContext(self.snapshot, old, now=self.now)
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 1), context)
        self.assertIn("stale_quote", decision.reason)

    def test_symbol_mismatch_rejected(self):
        decision = self.risk.evaluate(OrderRequest("OTHER", Side.BUY, 1), self.context)
        self.assertIn("symbol_mismatch", decision.reason)

    def test_short_sell_rejected(self):
        decision = self.risk.evaluate(OrderRequest("TEST", Side.SELL, 1), self.context)
        self.assertIn("short_selling_disabled", decision.reason)

    def test_halt_rejected(self):
        context = RiskContext(self.snapshot, self.q, trading_halted=True, now=self.now)
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 1), context)
        self.assertIn("trading_halted", decision.reason)

    def test_daily_loss_limit_rejected(self):
        context = RiskContext(self.snapshot, self.q, daily_realized_pnl=-3000, now=self.now)
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 1), context)
        self.assertIn("daily_loss_limit", decision.reason)

    def test_drawdown_limit_rejected(self):
        context = RiskContext(self.snapshot, self.q, current_drawdown_fraction=0.2, now=self.now)
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 1), context)
        self.assertIn("drawdown_limit", decision.reason)

    def test_fractional_shares_rejected_by_default(self):
        decision = self.risk.evaluate(OrderRequest("TEST", Side.BUY, 0.5), self.context)
        self.assertIn("fractional_quantity_disabled", decision.reason)

    def test_sizing_functions(self):
        self.assertEqual(fixed_fractional_quantity(10000, 0.01, 100, 95), 20)
        qty = volatility_scaled_quantity(10000, 0.01, 0.02, 100, fractional=True)
        self.assertAlmostEqual(qty, 20)
        self.assertEqual(target_weight_quantity(10000, 1000, 0.2, 100), 10)
        self.assertEqual(cap_quantity_by_limits(100, 100, 2000, 5000, 0), 20)

    def test_fixed_fractional_rejects_identical_stop(self):
        with self.assertRaises(ValueError):
            fixed_fractional_quantity(1000, 0.01, 100, 100)

    def test_kill_switch_lifecycle(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            switch = KillSwitch(Path(temp) / "HALT")
            self.assertFalse(switch.active)
            switch.activate("test reason")
            self.assertTrue(switch.active)
            self.assertIn("test reason", switch.reason())
            switch.deactivate()
            self.assertFalse(switch.active)

    def test_circuit_breaker_opens_after_failures(self):
        breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0.01)
        breaker.record_failure()
        self.assertTrue(breaker.allow_request())
        breaker.record_failure()
        self.assertFalse(breaker.allow_request())
        breaker.reset()
        self.assertTrue(breaker.allow_request())

if __name__ == "__main__":
    unittest.main()
