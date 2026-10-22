import unittest
from datetime import timedelta
from martha_plus.types import Side
from martha_plus.advanced.order_slicing import build_twap, build_vwap, build_participation, round_plan_to_lot
from martha_plus.advanced.execution_quality import ExecutionObservation, analyze_execution, slippage_bps
from martha_plus.advanced.walkforward import expanding_window_folds, purged_kfold_indices, validate_fold_separation
from martha_plus.advanced.sensitivity import expand_parameter_grid, pareto_frontier, run_parameter_trials, summarize_sensitivity
from martha_plus.advanced.rebalancing import build_rebalance_plan
from martha_plus.advanced.liquidity import estimate_transaction_cost, capacity_curve, quote_imbalance
from .helpers import NOW, make_quote

class ExecutionResearchTests(unittest.TestCase):
    def test_twap_vwap_participation(self):
        twap = build_twap("aaa", Side.BUY, 100, NOW, NOW+timedelta(minutes=9), 10)
        self.assertAlmostEqual(sum(x.quantity for x in twap.slices), 100)
        vwap = build_vwap("AAA", Side.SELL, 100, NOW, NOW+timedelta(minutes=2), [1,2,1])
        self.assertAlmostEqual(vwap.slices[1].quantity, 50)
        part = build_participation("AAA", Side.BUY, 10, NOW, [100,100,100], .05)
        self.assertLessEqual(max(s.quantity for s in part.slices), 5.00001)

    def test_lot_rounding(self):
        plan = build_twap("AAA", Side.BUY, 13, NOW, NOW+timedelta(minutes=2), 3)
        rounded = round_plan_to_lot(plan, lot_size=5)
        self.assertAlmostEqual(rounded.total_quantity, 10)
        self.assertAlmostEqual(sum(s.quantity for s in rounded.slices), 10)

    def test_execution_quality(self):
        observation = ExecutionObservation("o1", "AAA", Side.BUY, 10, 10, 100, 101, 102, fees=1)
        self.assertGreater(slippage_bps(observation), 0)
        report = analyze_execution([observation])
        self.assertAlmostEqual(report.quantity_fill_rate, 1)
        self.assertGreater(report.implementation_shortfall, 0)

    def test_purged_chronological_folds(self):
        folds = expanding_window_folds(100, min_train_size=30, test_size=10, gap=2, embargo=3)
        self.assertTrue(validate_fold_separation(folds)["valid"])
        train, test = purged_kfold_indices(100, 5, purge=2, embargo=2)[2]
        self.assertFalse(set(train) & set(test))

    def test_grid_frontier_and_sensitivity(self):
        grid = expand_parameter_grid({"fast":[2,3], "slow":[8,10]})
        self.assertEqual(len(grid), 4)
        self.assertEqual(pareto_frontier([{"return":1,"risk":2},{"return":2,"risk":1},{"return":.5,"risk":3}], maximize=["return"], minimize=["risk"]), [1])
        trials = run_parameter_trials({"p":[1,2,3]}, lambda p: {"score":float(p["p"])})
        report = summarize_sensitivity(trials, "score", parameter_name="p")
        self.assertTrue(report.boundary_optimum)

    def test_rebalance_and_cost_model(self):
        plan = build_rebalance_plan(equity=10000, current_quantities={"AAA":10}, prices={"AAA":100,"BBB":50},
                                    target_weights={"AAA":.2,"BBB":.3}, minimum_trade_notional=10)
        self.assertEqual(plan.order_count, 2)
        cost = estimate_transaction_cost(side=Side.BUY, quantity=10, price=100, spread_bps=5,
                                         commission_bps=1, average_daily_dollar_volume=1000000, daily_volatility=.02)
        self.assertGreater(cost.total_cost, 0)
        self.assertEqual(len(capacity_curve(1e6,.02,[.01,.05])), 2)
        self.assertAlmostEqual(quote_imbalance(make_quote()).get("spread_bps"), make_quote().spread_bps)

if __name__ == "__main__": unittest.main()
