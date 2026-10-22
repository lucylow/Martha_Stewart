import unittest
from datetime import timedelta
from martha_plus.advanced.data_contracts import DataContract, FieldRule, ohlcv_contract, validate_time_series
from martha_plus.advanced.features import FeatureConfig, FeaturePipeline
from martha_plus.advanced.portfolio import PortfolioOptimizer, covariance_from_returns, shrink_covariance
from martha_plus.advanced.attribution import relative_performance, brinson_fachler, factor_regression
from martha_plus.advanced.statistical_tests import bootstrap_mean, benjamini_hochberg, permutation_p_value
from martha_plus.advanced.regimes import RegimeDetector, MarketRegime
from martha_plus.advanced.risk_budget import risk_parity_weights, liquidity_capped_quantity
from martha_plus.advanced.constraints import AllocationConstraints, ConstraintProjector
from martha_plus.advanced.portfolio_diagnostics import concentration_report, weight_drift
from .helpers import make_bars

class AnalyticsTests(unittest.TestCase):
    def test_contract_ohlcv_and_boolean_numeric_rejected(self):
        rows = [{"symbol":"AAA", "timestamp":"2025-01-01", "open":1, "high":2, "low":0.5, "close":1.5, "volume":3}]
        self.assertTrue(ohlcv_contract().validate(rows).valid)
        bad = [dict(rows[0], close=True)]
        self.assertFalse(ohlcv_contract().validate(bad).valid)

    def test_time_series_detects_duplicates(self):
        report = validate_time_series([{"timestamp": 1, "id":"a"}, {"timestamp": 1, "id":"b"}])
        self.assertFalse(report.valid)

    def test_feature_pipeline_is_causal_and_forward_labels_separate(self):
        bars = make_bars(closes=[100 + i for i in range(25)])
        rows = FeaturePipeline(FeatureConfig(return_windows=(1, 5), moving_average_windows=(5,),
                                              volatility_windows=(5,), volume_windows=(3,))).transform(bars)
        self.assertIsNone(rows[0].get("return_1"))
        self.assertAlmostEqual(rows[5].get("return_5"), 105 / 100 - 1)
        labels = FeaturePipeline.forward_returns(bars, 2)
        self.assertEqual(labels[-2:], [None, None])
        self.assertAlmostEqual(labels[0], 102 / 100 - 1)

    def test_cross_sectional_rank_handles_ties_and_missing(self):
        ranks = FeaturePipeline.cross_sectional_rank({"a": 1.0, "b": 1.0, "c": None})
        self.assertEqual(ranks["a"], ranks["b"])
        self.assertIsNone(ranks["c"])

    def test_covariance_and_shrinkage(self):
        cov = covariance_from_returns({"A":[.01,.02,.03], "B":[.02,.01,.04]})
        shrunk = shrink_covariance(cov, .5)
        self.assertEqual(shrunk["A"]["A"], cov["A"]["A"])
        self.assertAlmostEqual(shrunk["A"]["B"], cov["A"]["B"] * .5)

    def test_optimizer_respects_simplex_and_caps(self):
        cov = {"A":{"A":.04,"B":.01}, "B":{"A":.01,"B":.09}}
        result = PortfolioOptimizer(max_weight=.7, risk_aversion=2).optimize({"A":.1,"B":.08}, cov)
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, places=6)
        self.assertLessEqual(max(result.weights.values()), .700001)
        self.assertGreaterEqual(min(result.weights.values()), 0)

    def test_relative_performance_and_factor_regression(self):
        relative = relative_performance([.02,.01,-.01,.03], [.01,.01,0,.02])
        self.assertEqual(relative.observations, 4)
        self.assertGreater(relative.active_return, 0)
        y = [1 + 2*x for x in [1,2,3,4,5,6]]
        fit = factor_regression(y, {"factor":[1,2,3,4,5,6]}, ridge=0.01)
        self.assertAlmostEqual(fit["betas"]["factor"], 2, places=2)

    def test_brinson_additive_effects(self):
        r = brinson_fachler({"tech":.6,"cash":.4}, {"tech":.1,"cash":0},
                            {"tech":.5,"cash":.5}, {"tech":.08,"cash":0})
        self.assertEqual(set(r), {"tech", "cash"})
        self.assertIn("interaction", r["tech"])

    def test_bootstrap_is_deterministic_and_bh(self):
        self.assertEqual(bootstrap_mean([1,2,3], repetitions=200, seed=99), bootstrap_mean([1,2,3], repetitions=200, seed=99))
        bh = benjamini_hochberg([.001,.02,.8], q=.05)
        self.assertEqual(bh["discoveries"], [True, True, False])
        self.assertGreater(permutation_p_value([5,6,7], [1,2,3], repetitions=500, seed=5), 0)

    def test_regime_and_risk_parity(self):
        prices = [100 * (1.001 ** i) for i in range(80)]
        current = RegimeDetector().current(prices)
        self.assertIsInstance(current.regime, MarketRegime)
        cov = {"A":{"A":.04,"B":.0}, "B":{"A":.0,"B":.09}}
        result = risk_parity_weights(cov, max_iterations=5000)
        self.assertAlmostEqual(sum(result.weights.values()), 1)
        self.assertGreater(result.portfolio_volatility, 0)
        self.assertGreater(liquidity_capped_quantity(equity=100000, risk_fraction=.01,
                            stop_distance=2, price=100, average_daily_volume=10000), 0)

    def test_constraint_projector_and_diagnostics(self):
        projector = ConstraintProjector(AllocationConstraints(total_weight=1, max_single_name=.6,
            group_by_symbol={"A":"tech","B":"tech","C":"health"}, max_group_weight={"tech":.7}))
        weights = projector.project({"A":.7,"B":.2,"C":.1})
        self.assertTrue(projector.validate(weights).feasible)
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertLessEqual(concentration_report(weights)["hhi"], 1)
        self.assertGreater(weight_drift({"A":1},{"A":.5})["l1_drift"], 0)

if __name__ == "__main__": unittest.main()
