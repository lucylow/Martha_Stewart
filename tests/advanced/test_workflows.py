import unittest
from datetime import timedelta
from martha_plus.advanced.eventbus import EventBus, EventEnvelope
from martha_plus.advanced.alerts import AlertManager, Alert, AlertPolicy, Severity, InMemoryNotifier
from martha_plus.advanced.experiments import ExperimentRegistry
from martha_plus.advanced.replay import EventJournal, ReplayEngine, batched
from martha_plus.advanced.lineage import build_manifest, verify_manifest, manifest_to_dict
from martha_plus.advanced.snapshots import SnapshotStore, SnapshotError
from martha_plus.advanced.incidents import IncidentManager, IncidentStatus

class WorkflowTests(unittest.TestCase):
    def test_event_bus_retries_and_dead_letters(self):
        bus = EventBus()
        count = {"n": 0}
        def handler(event):
            count["n"] += 1
            if count["n"] < 2: raise RuntimeError("transient")
        bus.subscribe("tick", handler)
        result = bus.publish(EventEnvelope("tick", {"x":1}), retries=1)
        self.assertEqual(result.delivered, 1)
        self.assertEqual(result.failed, 0)
        self.assertTrue(bus.publish(EventEnvelope("tick", {}, event_id="same"), idempotent=True).event_id)

    def test_event_bus_dead_letter_can_be_retried_after_failure(self):
        bus = EventBus()
        should_fail = {"yes": True}
        bus.subscribe("event", lambda e: (_ for _ in ()).throw(RuntimeError()) if should_fail["yes"] else None)
        event = EventEnvelope("event", {}, event_id="retryable")
        self.assertEqual(bus.publish(event).failed, 1)
        should_fail["yes"] = False
        self.assertEqual(bus.publish(event).delivered, 1)

    def test_alert_cooldown_and_resolution(self):
        clock = {"t":0.0}
        notifier = InMemoryNotifier()
        manager = AlertManager([notifier], AlertPolicy(cooldown_seconds=5, repeat_after_seconds=20),
                               monotonic_clock=lambda: clock["t"])
        alert = Alert("title", "msg", Severity.ERROR, "service:api")
        self.assertEqual(len(manager.emit(alert)), 1)
        self.assertEqual(manager.emit(alert), ())
        self.assertTrue(manager.resolve("service:api"))
        clock["t"] = 6
        self.assertEqual(len(manager.emit(alert)), 1)

    def test_experiment_state_transitions_and_filtering(self):
        registry = ExperimentRegistry()
        run = registry.create("momentum", {"fast":5}, tags=["daily"])
        registry.start(run.run_id)
        final = registry.complete(run.run_id, {"sharpe":1.2})
        self.assertEqual(final.status, "completed")
        self.assertEqual(registry.list(tag="daily")[0].run_id, run.run_id)
        with self.assertRaises(ValueError): registry.start(run.run_id)
        registry.close()

    def test_journal_replay_and_checkpoint(self):
        journal = EventJournal()
        event = EventEnvelope("fill", {"qty":2})
        journal.append(event)
        journal.append(EventEnvelope("order", {"qty":1}))
        seen = []
        result = ReplayEngine(journal).replay(lambda e: seen.append(e.event_type), event_types={"fill"})
        self.assertEqual(seen, ["fill"])
        cp = ReplayEngine(journal).checkpoint(1, {"cash":100})
        self.assertTrue(ReplayEngine(journal).verify_checkpoint(cp, {"cash":100}))
        self.assertEqual(list(batched(range(5), 2)), [[0,1],[2,3],[4]])

    def test_snapshot_checksum_and_safe_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            info = store.save("latest", {"cash":100})
            data, read_info = store.load("latest")
            self.assertEqual(data, {"cash":100})
            with self.assertRaises(ValueError): store.save("../unsafe", {})
            path = info.path
            with open(path, "w", encoding="utf8") as file: file.write('{"x":1}')
            with self.assertRaises(SnapshotError): store.load("latest")

    def test_manifest_detects_tampering(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as directory:
            p = pathlib.Path(directory) / "a.txt"
            p.write_text("abc")
            manifest = build_manifest([p], base_dir=directory)
            self.assertTrue(verify_manifest(manifest_to_dict(manifest), directory)["ok"])
            p.write_text("changed")
            self.assertFalse(verify_manifest(manifest_to_dict(manifest), directory)["ok"])

    def test_incident_lifecycle(self):
        manager = IncidentManager()
        first = manager.open("db-down", "DB down", "critical")
        second = manager.open("db-down", "DB still down", "warning")
        self.assertEqual(first.incident_id, second.incident_id)
        self.assertEqual(second.occurrence_count, 2)
        acknowledged = manager.acknowledge(first.incident_id, "operator")
        self.assertEqual(acknowledged.status, IncidentStatus.ACKNOWLEDGED)
        resolved = manager.resolve(first.incident_id, "operator", "recovered")
        self.assertEqual(resolved.status, IncidentStatus.RESOLVED)

if __name__ == "__main__": unittest.main()
