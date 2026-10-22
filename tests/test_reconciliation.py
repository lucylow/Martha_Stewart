import unittest
from datetime import datetime, timezone
from martha_plus.execution.reconciliation import reconcile
from martha_plus.portfolio import PortfolioLedger
from martha_plus.types import Fill, Order, OrderRequest, OrderStatus, Side

class ReconciliationTests(unittest.TestCase):
    def test_healthy_filled_order_reconciles(self):
        req = OrderRequest("ABC", Side.BUY, 2, client_order_id="C1")
        order = Order("O1", req, OrderStatus.FILLED, 2, 100)
        fill = Fill("F1", "O1", "ABC", Side.BUY, 2, 100, 0, datetime.now(timezone.utc))
        report = reconcile([order], [fill], {})
        self.assertTrue(report.ok, report.issues)

    def test_orphan_fill_is_reported(self):
        fill = Fill("F1", "missing", "ABC", Side.BUY, 2, 100, 0, datetime.now(timezone.utc))
        report = reconcile([], [fill], {})
        self.assertFalse(report.ok)
        self.assertEqual(report.issues[0].code, "orphan_fill")

    def test_filled_quantity_mismatch_is_reported(self):
        req = OrderRequest("ABC", Side.BUY, 2)
        order = Order("O1", req, OrderStatus.FILLED, 2, 100)
        fill = Fill("F1", "O1", "ABC", Side.BUY, 1, 100, 0, datetime.now(timezone.utc))
        report = reconcile([order], [fill], {})
        self.assertFalse(report.ok)
        self.assertIn("filled_quantity_mismatch", {issue.code for issue in report.issues})

if __name__ == "__main__":
    unittest.main()
