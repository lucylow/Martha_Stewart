import unittest
from datetime import datetime
from martha_plus.errors import DataValidationError
from martha_plus.types import Bar, Quote, OrderRequest, OrderType, Side, OrderStatus
from tests.helpers import quote

class TypeTests(unittest.TestCase):
    def test_quote_normalizes_symbol_and_calculates_spread(self):
        q = quote(" test ", 50)
        self.assertEqual(q.symbol, "TEST")
        self.assertAlmostEqual(q.midpoint, 50)
        self.assertAlmostEqual(q.spread, 0.02)
        self.assertGreater(q.spread_bps, 0)

    def test_quote_rejects_crossed_market(self):
        with self.assertRaises(DataValidationError):
            Quote("TEST", 101, 100)

    def test_quote_requires_timezone(self):
        with self.assertRaises(DataValidationError):
            Quote("TEST", 99, 100, datetime(2024, 1, 1))

    def test_bar_checks_ohlc(self):
        with self.assertRaises(DataValidationError):
            Bar("TEST", quote().timestamp, 100, 99, 98, 100)

    def test_order_requires_limit_price(self):
        with self.assertRaises(DataValidationError):
            OrderRequest("TEST", Side.BUY, 1, OrderType.LIMIT)

    def test_order_rejects_limit_price_on_market_order(self):
        with self.assertRaises(DataValidationError):
            OrderRequest("TEST", Side.BUY, 1, OrderType.MARKET, limit_price=100)

    def test_integer_quantity_is_valid(self):
        req = OrderRequest("TEST", Side.BUY, 10)
        self.assertEqual(req.quantity, 10)

    def test_terminal_statuses(self):
        for status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            self.assertTrue(status.terminal)
        self.assertFalse(OrderStatus.ACCEPTED.terminal)

if __name__ == "__main__":
    unittest.main()
