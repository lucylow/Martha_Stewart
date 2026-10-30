import unittest

from martha_plus.systematic import (
    DataIntegrityError,
    PageHinkley,
    UniverseInterval,
    assert_available_by,
    conformal_interval,
    delisting_return,
    lagged_signal_backtest,
    members_at,
    point_in_time_join,
)


class PointInTimeJoinTests(unittest.TestCase):
    def setUp(self):
        self.decisions = [{"symbol": "ABC", "decision_ts_ns": 20}]
        self.observations = [
            {"symbol": "ABC", "available_ts_ns": 10, "value": 1},
            {"symbol": "ABC", "available_ts_ns": 25, "value": 2},
        ]

    def test_future_observation_never_matches(self):
        match = point_in_time_join(self.decisions, self.observations)[0]
        self.assertEqual(match.observation["value"], 1)
        self.assertEqual(match.age_ns, 10)

    def test_same_timestamp_is_available(self):
        row = [{"symbol": "ABC", "decision_ts_ns": 10}]
        match = point_in_time_join(row, self.observations)[0]
        self.assertEqual(match.observation["value"], 1)
        self.assertEqual(match.age_ns, 0)

    def test_missing_match_fails_closed(self):
        with self.assertRaises(DataIntegrityError):
            point_in_time_join(self.decisions, [{"symbol": "XYZ", "available_ts_ns": 5}])

    def test_optional_missing_match_is_explicit(self):
        match = point_in_time_join(self.decisions, [], require_match=False)[0]
        self.assertIsNone(match.observation)
        self.assertIsNone(match.age_ns)

    def test_stale_observation_fails_closed(self):
        with self.assertRaises(DataIntegrityError):
            point_in_time_join(self.decisions, self.observations, max_age_ns=9)
        with self.assertRaises(DataIntegrityError):
            point_in_time_join(self.decisions, self.observations, max_age_ns=1.5)

    def test_stale_observation_can_be_returned_as_missing(self):
        match = point_in_time_join(self.decisions, self.observations,
                                   max_age_ns=9, require_match=False)[0]
        self.assertIsNone(match.observation)

    def test_same_time_correction_uses_last_input_record(self):
        rows = [
            {"symbol": "ABC", "available_ts_ns": 10, "value": 1},
            {"symbol": "ABC", "available_ts_ns": 10, "value": 3},
        ]
        match = point_in_time_join([{"symbol": "ABC", "decision_ts_ns": 10}], rows)[0]
        self.assertEqual(match.observation["value"], 3)

    def test_rejects_invalid_timestamps_and_keys(self):
        with self.assertRaises(DataIntegrityError):
            point_in_time_join([{"symbol": "ABC", "decision_ts_ns": True}], [])
        with self.assertRaises(DataIntegrityError):
            point_in_time_join([{"decision_ts_ns": 1}], [])
        with self.assertRaises(DataIntegrityError):
            point_in_time_join(self.decisions, self.observations, key_fields="symbol")
        with self.assertRaises(DataIntegrityError):
            point_in_time_join(self.decisions, [{"symbol": "ABC", "available_ts_ns": -1}])

    def test_available_by_guards_future_information(self):
        assert_available_by(10, 10)
        with self.assertRaises(DataIntegrityError):
            assert_available_by(11, 10)


class UniverseTests(unittest.TestCase):
    def test_membership_requires_known_and_valid_at_decision(self):
        intervals = [
            UniverseInterval("A", 10, 40, 8),
            UniverseInterval("B", 10, None, 21),
            UniverseInterval("C", 10, None, 8),
        ]
        self.assertEqual(members_at(intervals, 20), ("A", "C"))
        self.assertEqual(members_at(intervals, 21), ("A", "B", "C"))
        self.assertEqual(members_at(intervals, 40), ("B", "C"))

    def test_universe_input_validation(self):
        with self.assertRaises(DataIntegrityError):
            UniverseInterval("", 1, None, 1)
        with self.assertRaises(DataIntegrityError):
            UniverseInterval(12, 1, None, 1)
        with self.assertRaises(DataIntegrityError):
            UniverseInterval("A", 2, 2, 1)

    def test_delisting_return_requires_explicit_recovery(self):
        self.assertAlmostEqual(delisting_return(10.0, 0.0), -1.0)
        self.assertAlmostEqual(delisting_return(10.0, 5.0), -0.5)
        with self.assertRaises(DataIntegrityError):
            delisting_return(0.0, 1.0)
        with self.assertRaises(DataIntegrityError):
            delisting_return(True, 1.0)


class LaggedBacktestTests(unittest.TestCase):
    def test_signal_is_delayed_one_period(self):
        result = lagged_signal_backtest([1.0, 1.0, -1.0], [0.10, 0.20, -0.10])
        self.assertEqual(result.positions, (0.0, 1.0, 1.0))
        self.assertEqual(result.gross_returns, (0.0, 0.20, -0.10))

    def test_turnover_costs_are_charged_on_position_changes(self):
        result = lagged_signal_backtest([1.0, -1.0, 0.0], [0.1, 0.1, 0.1], cost_bps=10)
        self.assertEqual(result.turnover, (0.0, 1.0, 2.0))
        self.assertAlmostEqual(result.total_cost, 0.003)
        self.assertAlmostEqual(result.net_returns[2], -0.102)

    def test_exposure_is_bounded_and_short_allowed(self):
        result = lagged_signal_backtest([4.0, -3.0], [0.1, 0.1], exposure_limit=1.5)
        self.assertEqual(result.positions, (0.0, 1.5))

    def test_empty_inputs_are_valid(self):
        result = lagged_signal_backtest([], [])
        self.assertEqual(result.net_returns, ())
        self.assertEqual(result.total_turnover, 0.0)

    def test_one_period_has_defined_sharpe_placeholder(self):
        result = lagged_signal_backtest([1.0], [0.05])
        self.assertEqual(result.sharpe, 0.0)

    def test_leveraged_loss_cannot_break_equity_curve(self):
        with self.assertRaisesRegex(ValueError, "insolvency"):
            lagged_signal_backtest([3.0, 3.0], [0.0, -0.5], exposure_limit=3.0)

    def test_invalid_alignment_cost_and_returns_rejected(self):
        with self.assertRaises(ValueError):
            lagged_signal_backtest([1.0], [])
        with self.assertRaises(ValueError):
            lagged_signal_backtest([1.0], [0.0], cost_bps=-1)
        with self.assertRaises(ValueError):
            lagged_signal_backtest([1.0], [-1.0])
        with self.assertRaises(ValueError):
            lagged_signal_backtest([float("nan")], [0.0])


class AdaptiveTests(unittest.TestCase):
    def test_conformal_finite_sample_interval(self):
        self.assertEqual(conformal_interval(5.0, [-1.0, 2.0, 1.0], coverage=0.5), (4.0, 6.0))

    def test_conformal_returns_unbounded_interval_when_sample_is_too_small(self):
        lower, upper = conformal_interval(0.0, [1.0], coverage=0.95)
        self.assertEqual(lower, float("-inf"))
        self.assertEqual(upper, float("inf"))

    def test_conformal_rejects_missing_or_invalid_calibration(self):
        with self.assertRaises(ValueError):
            conformal_interval(0.0, [])
        with self.assertRaises(ValueError):
            conformal_interval(0.0, [float("nan")])
        with self.assertRaises(ValueError):
            conformal_interval(0.0, [1.0], coverage=1.0)

    def test_page_hinkley_detects_shift_and_can_reset(self):
        detector = PageHinkley(threshold=1.0, delta=0.01)
        self.assertFalse(any(detector.update(0.0) for _ in range(20)))
        self.assertTrue(any(detector.update(3.0) for _ in range(10)))
        detector.reset()
        self.assertEqual(detector.count, 0)

    def test_page_hinkley_validates_parameters_and_observations(self):
        with self.assertRaises(ValueError):
            PageHinkley(threshold=0.0)
        detector = PageHinkley()
        with self.assertRaises(ValueError):
            detector.update(float("inf"))


if __name__ == "__main__":
    unittest.main()
