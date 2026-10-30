import unittest
from decimal import Decimal

from martha_plus.microstructure.clock import LatencyKind, LatencyModel, LatencyProfile
from martha_plus.microstructure.matching.engine import MatchingEngine
from martha_plus.microstructure.replay.runner import MarketReplay
from martha_plus.microstructure.strategies.base import StrategyContext
from martha_plus.microstructure.types import (
    EventKind, MarketEvent, OrderRequest, OrderStatus, OrderType, Side,
    SymbolSpec, TimeInForce,
)


class OneShotBidder:
    strategy_id = "test-bidder"

    def __init__(self):
        self.sent = False

    def on_market_data(self, event, context: StrategyContext):
        if self.sent or context.top.bid is None or context.top.ask is None:
            return ()
        self.sent = True
        return (OrderRequest("child-1", context.symbol, Side.BUY, OrderType.LIMIT,
                             2, limit_price_ticks=context.top.bid.price_ticks,
                             time_in_force=TimeInForce.GTC,
                             strategy_id=self.strategy_id),)

class OneShotTaker:
    strategy_id = "test-taker"

    def __init__(self):
        self.sent = False

    def on_market_data(self, event, context: StrategyContext):
        if self.sent or context.top.ask is None:
            return ()
        self.sent = True
        return (OrderRequest("take-ask", context.symbol, Side.BUY, OrderType.LIMIT,
                             1, limit_price_ticks=context.top.ask.price_ticks,
                             strategy_id=self.strategy_id),)


class MatchingEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = MatchingEngine(SymbolSpec("ABC"))

    def test_limit_order_sweeps_price_then_time(self):
        self.engine.add_external_liquidity("ask-1", Side.SELL, 101, 2, 1)
        self.engine.add_external_liquidity("ask-2", Side.SELL, 101, 3, 1)
        request = OrderRequest("buy-1", "ABC", Side.BUY, OrderType.LIMIT, 4,
                               limit_price_ticks=101, strategy_id="test")
        result = self.engine.submit(request, 2)
        self.assertTrue(result.ack.accepted)
        self.assertEqual([x.maker_order_id for x in result.executions], ["ask-1", "ask-2"])
        self.assertEqual([x.quantity for x in result.executions], [2, 2])
        self.assertEqual(result.record.status, OrderStatus.FILLED)
        self.engine.validate()

    def test_fok_rejects_without_partial_execution(self):
        self.engine.add_external_liquidity("ask-1", Side.SELL, 101, 2, 1)
        request = OrderRequest("buy-fok", "ABC", Side.BUY, OrderType.LIMIT, 3,
                               limit_price_ticks=101, time_in_force=TimeInForce.FOK)
        result = self.engine.submit(request, 2)
        self.assertFalse(result.ack.accepted)
        self.assertEqual(result.ack.reject_code.value, "fok_not_fillable")
        self.assertEqual(len(self.engine.executions), 0)
        self.assertIsNone(self.engine.order_status("buy-fok"))

    def test_post_only_rejects_marketable_order(self):
        self.engine.add_external_liquidity("ask-1", Side.SELL, 101, 2, 1)
        request = OrderRequest("post-only", "ABC", Side.BUY, OrderType.LIMIT, 1,
                               limit_price_ticks=101, time_in_force=TimeInForce.POST_ONLY)
        result = self.engine.submit(request, 2)
        self.assertFalse(result.ack.accepted)
        self.assertEqual(result.ack.reject_code.value, "post_only_would_take")

    def test_resting_order_cancel_conserves_quantity(self):
        request = OrderRequest("resting", "ABC", Side.BUY, OrderType.LIMIT, 5,
                               limit_price_ticks=99, strategy_id="test")
        result = self.engine.submit(request, 1)
        self.assertTrue(result.ack.accepted)
        cancel = self.engine.cancel("resting", 2)
        self.assertTrue(cancel.accepted)
        self.assertEqual(cancel.cancelled_quantity, 5)
        self.engine.validate()


class ReplayTests(unittest.TestCase):
    def test_delayed_strategy_order_is_filled_by_later_external_flow(self):
        zero = LatencyProfile(0, 0, 0, 0)
        latency = LatencyModel({kind: zero for kind in LatencyKind}, seed=1)
        events = (
            MarketEvent(1, 1, "ABC", EventKind.ADD, Side.BUY, 99, 10, "ext-bid"),
            MarketEvent(2, 2, "ABC", EventKind.ADD, Side.SELL, 101, 10, "ext-ask"),
            MarketEvent(3, 3, "ABC", EventKind.TRADE, Side.SELL, 99, 12),
        )
        engine = MatchingEngine(SymbolSpec("ABC"))
        result = MarketReplay(engine, OneShotBidder(), latency=latency).run(events)
        self.assertEqual(result.accepted_orders, 1)
        self.assertEqual(result.filled_quantity, 12)
        self.assertEqual(result.rejected_orders, 0)
        self.assertEqual(len(result.anomalies), 0)
        self.assertEqual(result.venue_acks[0].status, OrderStatus.ACCEPTED)
        self.assertEqual(engine.order_status("child-1"), OrderStatus.FILLED)

    def test_replay_rejects_empty_stream(self):
        replay = MarketReplay(MatchingEngine(SymbolSpec("ABC")), OneShotBidder())
        with self.assertRaises(ValueError):
            replay.run(())

    def test_client_fill_reconciles_later_synthetic_cancel(self):
        zero = LatencyProfile(0, 0, 0, 0)
        latency = LatencyModel({kind: zero for kind in LatencyKind}, seed=1)
        events = (
            MarketEvent(1, 1, "ABC", EventKind.ADD, Side.BUY, 99, 5, "ext-bid"),
            MarketEvent(2, 2, "ABC", EventKind.ADD, Side.SELL, 101, 1, "ext-ask"),
            MarketEvent(3, 3, "ABC", EventKind.CANCEL, order_id="ext-ask"),
        )
        result = MarketReplay(MatchingEngine(SymbolSpec("ABC")), OneShotTaker(),
                              latency=latency).run(events)
        self.assertEqual(result.filled_quantity, 1)
        self.assertEqual(result.anomalies, ())


if __name__ == "__main__":
    unittest.main()
