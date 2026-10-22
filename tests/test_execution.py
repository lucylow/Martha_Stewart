import unittest
from datetime import datetime, timezone
from martha_plus.errors import DuplicateOrder
from martha_plus.execution import OrderManager, PaperBroker
from martha_plus.persistence import SQLiteStore
from martha_plus.portfolio import PortfolioLedger
from martha_plus.risk import RiskContext, RiskLimits
from martha_plus.types import OrderRequest, OrderStatus, OrderType, Side
from tests.helpers import quote, settings

class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = settings()
        self.now = datetime.now(timezone.utc)
        self.q = quote(stamp=self.now)
        self.book = PortfolioLedger(self.config.starting_cash)
        self.risk = RiskLimits(self.config)
        self.context = RiskContext(self.book.snapshot(self.now), self.q, now=self.now)

    def test_paper_market_order_fills(self):
        result = PaperBroker(self.config, seed=1).execute(OrderRequest("TEST", Side.BUY, 10), self.q)
        self.assertEqual(result.order.status, OrderStatus.FILLED)
        self.assertEqual(len(result.fills), 1)
        self.assertGreater(result.fills[0].price, self.q.ask)

    def test_unmarketable_limit_stays_open(self):
        req = OrderRequest("TEST", Side.BUY, 1, OrderType.LIMIT, limit_price=90)
        result = PaperBroker(self.config, seed=1).execute(req, self.q)
        self.assertEqual(result.order.status, OrderStatus.ACCEPTED)
        self.assertEqual(len(result.fills), 0)

    def test_paper_fill_probability_can_skip_fill(self):
        config = settings(paper_fill_probability=0.0)
        result = PaperBroker(config, seed=1).execute(OrderRequest("TEST", Side.BUY, 1), self.q)
        self.assertEqual(result.order.status, OrderStatus.EXPIRED)

    def test_order_manager_persists_parent_before_fill(self):
        store = SQLiteStore(":memory:")
        manager = OrderManager(PaperBroker(self.config), self.risk, self.book,
                               persist_order=store.save_order, persist_fill=store.save_fill)
        result = manager.submit(OrderRequest("TEST", Side.BUY, 10), self.q, self.context)
        self.assertEqual(result.order.status, OrderStatus.FILLED)
        self.assertEqual(store.order_count(), 1)
        self.assertEqual(store.count_fills(), 1)
        store.close()

    def test_order_manager_idempotent_client_id(self):
        manager = OrderManager(PaperBroker(self.config), self.risk, self.book)
        request = OrderRequest("TEST", Side.BUY, 10, client_order_id="same-id")
        first = manager.submit(request, self.q, self.context)
        second = manager.submit(request, self.q, self.context)
        self.assertEqual(first.order.order_id, second.order.order_id)

    def test_order_manager_rejects_idempotency_conflict(self):
        manager = OrderManager(PaperBroker(self.config), self.risk, self.book)
        first = OrderRequest("TEST", Side.BUY, 10, client_order_id="same-id")
        manager.submit(first, self.q, self.context)
        second = OrderRequest("TEST", Side.BUY, 11, client_order_id="same-id")
        with self.assertRaises(DuplicateOrder):
            manager.submit(second, self.q, self.context)

    def test_risk_rejected_order_not_executed(self):
        manager = OrderManager(PaperBroker(self.config), self.risk, self.book)
        request = OrderRequest("TEST", Side.BUY, 1000)
        result = manager.submit(request, self.q, self.context)
        self.assertEqual(result.order.status, OrderStatus.REJECTED)
        self.assertEqual(len(self.book.fills), 0)

    def test_cancel_open_order(self):
        req = OrderRequest("TEST", Side.BUY, 1, OrderType.LIMIT, limit_price=90,
                           client_order_id="limit-id")
        manager = OrderManager(PaperBroker(self.config), self.risk, self.book)
        result = manager.submit(req, self.q, self.context)
        self.assertEqual(len(manager.open_orders), 1)
        cancelled = manager.cancel("limit-id")
        self.assertEqual(cancelled.status, OrderStatus.CANCELLED)
        self.assertEqual(len(manager.open_orders), 0)

if __name__ == "__main__":
    unittest.main()
