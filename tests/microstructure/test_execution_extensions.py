import unittest
from decimal import Decimal

from martha_plus.microstructure.execution import (
    ExecutionCostEstimator, ExecutionCostInputs, ImpactParameters,
    ParentOrder, ProbabilisticQueueModel, QueueEvent, QueueEventType,
    QueuePosition, ScheduleBuilder, ScheduleKind, SlippageAccumulator,
    SlippageSample, SquareRootImpactModel, evenly_spaced_buckets,
)
from martha_plus.microstructure.types import Side, SymbolSpec


class QueueModelTests(unittest.TestCase):
    def test_trade_consumes_ahead_then_own_order(self):
        queue = QueuePosition("ours", 10, 5, 10)
        self.assertEqual(queue.apply(QueueEvent(1, QueueEventType.TRADE, 12)), 2)
        self.assertEqual(queue.quantity_ahead, 0)
        self.assertEqual(queue.own_quantity, 3)
        self.assertEqual(queue.apply(QueueEvent(2, QueueEventType.TRADE, 3)), 3)
        self.assertTrue(queue.is_filled)

    def test_cancel_ahead_improves_rank_not_own_conservation(self):
        queue = QueuePosition("ours", 10, 5)
        queue.apply(QueueEvent(1, QueueEventType.CANCEL_AHEAD, 4))
        self.assertEqual(queue.quantity_ahead, 6)
        self.assertEqual(queue.own_quantity, 5)
        self.assertEqual(queue.filled_quantity + queue.cancelled_quantity, 0)

    def test_cancel_own_order_preserves_conservation(self):
        queue = QueuePosition("ours", 0, 5)
        self.assertEqual(queue.cancel(2, 1), 2)
        self.assertEqual(queue.own_quantity, 3)
        self.assertEqual(queue.cancelled_quantity, 2)

    def test_probabilistic_queue_estimate_is_bounded(self):
        queue = QueuePosition("ours", 10, 5)
        model = ProbabilisticQueueModel(cancellation_ahead_fraction=Decimal("0.5"))
        estimate = model.expected_fill(queue, traded_quantity=20, cancelled_quantity=4)
        self.assertEqual(estimate, Decimal(5))


class ImpactAndScheduleTests(unittest.TestCase):
    def test_square_root_impact_monotonic(self):
        model = SquareRootImpactModel(ImpactParameters(eta=Decimal("1")))
        small = model.temporary_bps(quantity=100, average_daily_volume=10_000, volatility=Decimal("0.02"))
        large = model.temporary_bps(quantity=400, average_daily_volume=10_000, volatility=Decimal("0.02"))
        self.assertGreater(large, small)

    def test_cost_estimator_zero_and_nonzero_order(self):
        estimator = ExecutionCostEstimator(SymbolSpec("ABC"))
        zero = estimator.estimate(ExecutionCostInputs(Side.BUY, 0, 100, Decimal(100),
                                                       Decimal(2), 1_000, Decimal("0.02")))
        self.assertEqual(zero.total_cost, 0)
        result = estimator.estimate(ExecutionCostInputs(Side.BUY, 100, 100, Decimal(100),
                                                         Decimal(2), 1_000, Decimal("0.02"),
                                                         Decimal("0.001")))
        self.assertGreater(result.total_bps, 0)

    def test_twap_conserves_parent_quantity(self):
        parent = ParentOrder("p1", "ABC", Side.BUY, 101, 0, 100, ScheduleKind.TWAP)
        buckets = evenly_spaced_buckets(0, 100, 4, 400)
        plans = ScheduleBuilder().build(parent, buckets)
        self.assertEqual(sum(x.quantity for x in plans), 101)
        self.assertEqual(plans[-1].remaining_quantity, 0)

    def test_pov_schedule_never_exceeds_parent(self):
        parent = ParentOrder("p2", "ABC", Side.SELL, 1000, 0, 100,
                             ScheduleKind.POV, Decimal("0.1"))
        buckets = evenly_spaced_buckets(0, 100, 5, 1000)
        plans = ScheduleBuilder().build(parent, buckets)
        self.assertLessEqual(sum(x.quantity for x in plans), 1000)

    def test_slippage_weighting(self):
        acc = SlippageAccumulator()
        acc.add(SlippageSample(1, Side.BUY, Decimal(100), Decimal(101), 1))
        acc.add(SlippageSample(2, Side.BUY, Decimal(100), Decimal(102), 3))
        summary = acc.summary()
        self.assertEqual(summary.samples, 2)
        self.assertGreater(summary.weighted_bps, summary.mean_bps)


if __name__ == "__main__":
    unittest.main()
