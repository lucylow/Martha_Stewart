import unittest
from datetime import date, datetime, timedelta, timezone
from martha_plus.advanced.eventbus import EventBus, EventEnvelope
from martha_plus.advanced.scheduler import ManualScheduler, JobSpec, JobStatus
from martha_plus.advanced.universe import SecurityRecord, UniverseBuilder, UniversePolicy
from martha_plus.advanced.feature_diagnostics import feature_profile, missingness_report, detect_constant_features, compare_profiles
from martha_plus.advanced.config_diff import require_config_match
from martha_plus.advanced.operations import HealthRegistry, HealthStatus
from martha_plus.advanced.data_contracts import DataContract, FieldRule, ContractViolation
from martha_plus.advanced.replay import EventJournal
from martha_plus.advanced.order_slicing import build_twap
from martha_plus.types import Side
from .helpers import NOW

class HardeningTests(unittest.TestCase):
    def test_dead_letter_replay_only_retries_failed_handler(self):
        bus = EventBus()
        calls = {"healthy":0,"flaky":0}
        bus.subscribe("tick", lambda event: calls.update(healthy=calls["healthy"]+1), name="healthy")
        def flaky(event):
            calls["flaky"] += 1
            if calls["flaky"] == 1:
                raise RuntimeError("first call fails")
        bus.subscribe("tick", flaky, name="flaky")
        event = EventEnvelope("tick", {"value":1}, event_id="first")
        result = bus.publish(event)
        self.assertEqual(result.failed, 1)
        replay = bus.replay_dead_letters(handler_names={"flaky"})
        self.assertEqual(replay[0].delivered, 1)
        self.assertEqual(calls["healthy"], 1)
        self.assertEqual(calls["flaky"], 2)

    def test_republish_retries_failed_handler_without_repeating_successful_side_effects(self):
        bus = EventBus()
        calls = {"healthy": 0, "flaky": 0}
        bus.subscribe("tick", lambda event: calls.update(healthy=calls["healthy"] + 1), name="healthy")

        def flaky(event):
            calls["flaky"] += 1
            if calls["flaky"] == 1:
                raise RuntimeError("temporary failure")

        bus.subscribe("tick", flaky, name="flaky")
        event = EventEnvelope("tick", {"value": 1}, event_id="republish-check")
        first = bus.publish(event)
        self.assertEqual(first.failed, 1)
        second = bus.publish(event)
        self.assertEqual(second.failed, 0)
        self.assertEqual(calls, {"healthy": 1, "flaky": 2})
        self.assertTrue(bus.publish(event).skipped_duplicate)

    def test_scheduler_max_jobs_does_not_drop_other_due_jobs(self):
        now = datetime(2025,1,1,tzinfo=timezone.utc)
        ran = []
        scheduler = ManualScheduler()
        for name in ("a", "b", "c"):
            scheduler.register(JobSpec(name, lambda name=name: ran.append(name), interval=timedelta(minutes=1)), now=now-timedelta(minutes=2))
        one = scheduler.tick(now, max_jobs=1)
        two = scheduler.tick(now, max_jobs=1)
        three = scheduler.tick(now, max_jobs=1)
        self.assertEqual(len(one + two + three), 3)
        self.assertEqual(set(ran), {"a","b","c"})

    def test_point_in_time_universe_keeps_security_not_yet_delisted(self):
        builder = UniverseBuilder(UniversePolicy())
        record = SecurityRecord("OLDCO", date(2020,1,1), active=False, ipo_date=date(2010,1,1), delisting_date=date(2022,1,1))
        decision = builder.decide(record, date(2021,1,1))
        self.assertTrue(decision.included)
        after_delist = builder.decide(record, date(2023,1,1))
        self.assertFalse(after_delist.included)

    def test_feature_profile_missingness_and_constant_detection(self):
        rows = [{"x":1.0,"y":2.0},{"x":1.0,"y":None},{"x":1.0}]
        profile = feature_profile(rows)
        self.assertEqual(profile["x"]["missing_fraction"], 0)
        self.assertGreater(profile["y"]["missing_fraction"], 0)
        self.assertIn("x", detect_constant_features({"x":[1,1,1], "z":[1,2,3]}))
        self.assertIn("y", missingness_report(rows, threshold=.3)["flagged"])
        drift = compare_profiles({"x":{"missing_fraction":0}}, {"x":{"missing_fraction":.5}})
        self.assertFalse(drift["matches"])

    def test_config_match_error_does_not_echo_secret(self):
        with self.assertRaises(RuntimeError) as context:
            require_config_match({"api_token":"private-value"}, {"api_token":"different-private-value"})
        self.assertNotIn("private-value", str(context.exception))

    def test_health_critical_failure_makes_unhealthy(self):
        registry = HealthRegistry()
        registry.register("database", lambda: False, critical=True)
        snapshot = registry.run()
        self.assertEqual(snapshot.status, HealthStatus.UNHEALTHY)
        self.assertEqual(snapshot.critical_failures, 1)

    def test_field_rule_contract_and_max_error_stop(self):
        contract = DataContract("small", 1, (FieldRule("x", kind=int, minimum=0),))
        report = contract.validate([{}, {"x":-1}, {"x":-2}], max_errors=1)
        self.assertFalse(report.valid)
        with self.assertRaises(ContractViolation): report.raise_if_invalid()

    def test_journal_retention_is_bounded(self):
        journal = EventJournal(max_records=2)
        for i in range(4): journal.append(EventEnvelope("x", {"i":i}))
        self.assertEqual(len(journal), 2)
        self.assertEqual(journal.range()[0].event.payload["i"], 2)

    def test_twap_rejects_naive_datetimes(self):
        with self.assertRaises(ValueError): build_twap("AAA", Side.BUY, 10, datetime(2025,1,1), NOW, 2)

if __name__ == "__main__": unittest.main()
