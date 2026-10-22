import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from martha_plus.persistence import SQLiteStore
from martha_plus.types import OrderRequest, Order, OrderStatus, Side, Fill

class PersistenceTests(unittest.TestCase):
    def test_memory_store_records_events_and_orders(self):
        store = SQLiteStore(":memory:")
        store.append_event("test", {"value": 1})
        events = store.list_events("test")
        self.assertEqual(events[0]["payload"]["value"], 1)
        request = OrderRequest("ABC", Side.BUY, 1)
        order = Order("O1", request, OrderStatus.FILLED, 1, 100)
        store.save_order(order)
        self.assertEqual(store.order_count(), 1)
        store.close()

    def test_disk_store_reopens(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.sqlite3"
            store = SQLiteStore(path)
            store.append_event("ready", {"ok": True})
            store.close()
            reopened = SQLiteStore(path)
            self.assertEqual(len(reopened.list_events()), 1)
            reopened.close()

    def test_duplicate_fill_is_ignored(self):
        store = SQLiteStore(":memory:")
        request = OrderRequest("ABC", Side.BUY, 1)
        store.save_order(Order("O1", request, OrderStatus.FILLED, 1, 100))
        fill = Fill("F1", "O1", "ABC", Side.BUY, 1, 100, 0, datetime.now(timezone.utc))
        store.save_fill(fill)
        store.save_fill(fill)
        self.assertEqual(store.count_fills(), 1)
        store.close()

    def test_order_client_id_conflict_is_rejected(self):
        store = SQLiteStore(":memory:")
        req = OrderRequest("ABC", Side.BUY, 1, client_order_id="same")
        store.save_order(Order("O1", req, OrderStatus.NEW))
        with self.assertRaises(Exception):
            store.save_order(Order("O2", req, OrderStatus.NEW))
        store.close()

    def test_event_limit_validated(self):
        store = SQLiteStore(":memory:")
        with self.assertRaises(ValueError):
            store.list_events(limit=0)
        store.close()

if __name__ == "__main__":
    unittest.main()
