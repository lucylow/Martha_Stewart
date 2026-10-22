from datetime import datetime, timezone

import pytest

from martha_plus.enterprise.order_lifecycle import (OrderEvent, OrderState, OrderStateMachine,
    InvalidTransition)
from martha_plus.enterprise.idempotency import IdempotencyConflict, IdempotencyRegistry
from martha_plus.enterprise.governance import (StrategyManifest, StrategyRegistry, StrategyStage,
    GovernanceError, ShadowComparator)
from martha_plus.enterprise.job_runner import JobRunner, JobSpec, JobStatus, JobContext


def test_order_lifecycle_enforces_transition_graph_and_versions():
    machine = OrderStateMachine()
    machine.create("o1", requested_quantity=5)
    created = machine.get("o1")
    validated = machine.transition("o1", OrderEvent.VALIDATE, actor="risk", expected_version=0)
    assert validated.state is OrderState.VALIDATED and validated.version == 1
    with pytest.raises(RuntimeError):
        machine.transition("o1", OrderEvent.ACCEPT, actor="broker", expected_version=0)
    accepted = machine.transition("o1", OrderEvent.ACCEPT, actor="broker", expected_version=1)
    partial = machine.transition("o1", OrderEvent.PARTIAL_FILL, actor="paper", fill_quantity=2)
    filled = machine.transition("o1", OrderEvent.FILL, actor="paper", fill_quantity=3)
    assert filled.state is OrderState.FILLED and filled.filled_quantity == 5
    with pytest.raises(InvalidTransition):
        machine.transition("o1", OrderEvent.CANCEL, actor="operator")
    assert machine.replay("o1").state is OrderState.FILLED
    assert machine.journal[-1].metadata["fill_quantity"] == "3.0"


def test_order_lifecycle_refuses_overfill():
    machine = OrderStateMachine()
    machine.create("o2", requested_quantity=10)
    machine.transition("o2", OrderEvent.VALIDATE, actor="risk")
    machine.transition("o2", OrderEvent.ACCEPT, actor="broker")
    with pytest.raises(ValueError):
        machine.transition("o2", OrderEvent.FILL, actor="paper", fill_quantity=12)


def test_idempotency_returns_prior_response_and_detects_conflict():
    clock = [10.0]
    registry = IdempotencyRegistry(ttl_seconds=5, clock=lambda: clock[0])
    calls = []
    first = registry.execute("key", {"symbol": "ABC", "qty": 2}, lambda: calls.append(1) or {"ok": True})
    second = registry.execute("key", {"qty": 2, "symbol": "ABC"}, lambda: calls.append(2))
    assert not first.replayed and second.replayed
    assert len(calls) == 1
    with pytest.raises(IdempotencyConflict):
        registry.execute("key", {"symbol": "ABC", "qty": 3}, lambda: None)
    clock[0] = 16
    assert registry.inspect("key") is None


def test_strategy_governance_requires_dual_review_and_evidence():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    registry = StrategyRegistry(clock=lambda: now)
    manifest = StrategyManifest("momentum", "1.0", "a" * 64, {"window": 20}, "owner",
                                now.isoformat(), dataset_digest="d" * 64)
    registry.register(manifest)
    registry.transition("momentum", "1.0", StrategyStage.RESEARCH, actor="owner", reason="start research")
    registry.transition("momentum", "1.0", StrategyStage.BACKTESTED, actor="reviewer", reason="backtest done")
    registry.transition("momentum", "1.0", StrategyStage.SHADOW, actor="reviewer", reason="start shadow")
    registry.approve("momentum", "1.0", "reviewer-a", evidence_digest="e" * 64)
    with pytest.raises(GovernanceError):
        registry.transition("momentum", "1.0", StrategyStage.PAPER_APPROVED,
                            actor="reviewer-a", reason="promote", evidence={})
    registry.approve("momentum", "1.0", "reviewer-b", evidence_digest="f" * 64)
    evidence = {"backtest_report": "r1", "risk_review": "r2", "dataset_digest": "d" * 64}
    registry.transition("momentum", "1.0", StrategyStage.PAPER_APPROVED,
                        actor="reviewer-b", reason="approved", evidence=evidence)
    registry.transition("momentum", "1.0", StrategyStage.PAPER_ACTIVE,
                        actor="operator", reason="paper activation")
    assert len(registry.active_paper_strategies()) == 1


def test_strategy_manifest_version_is_immutable():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    registry = StrategyRegistry(clock=lambda: now)
    first = StrategyManifest("s", "1", "a" * 64, {"n": 1}, "owner", now.isoformat())
    changed = StrategyManifest("s", "1", "a" * 64, {"n": 2}, "owner", now.isoformat())
    registry.register(first)
    with pytest.raises(GovernanceError):
        registry.register(changed)


def test_shadow_comparator_flags_threshold_breaches():
    comparator = ShadowComparator(max_abs_difference=0.1)
    row = comparator.compare("shadow", 0.8, 0.2, datetime.now(timezone.utc))
    assert row.threshold_breached
    assert comparator.summary("shadow")["breaches"] == 1


def test_job_runner_runs_dependencies_and_retries():
    attempts = {"a": 0}
    runner = JobRunner()
    runner.add(JobSpec("a", lambda ctx: attempts.__setitem__("a", attempts["a"] + 1) or
                       (_ for _ in ()).throw(ValueError("transient")) if attempts["a"] == 0 else "ok",
                       max_attempts=2))
    runner.add(JobSpec("b", lambda ctx: "downstream", dependencies=("a",)))
    a = runner.run("a")
    assert a.status is JobStatus.SUCCEEDED and a.attempt == 2
    b = runner.run("b")
    assert b.status is JobStatus.SUCCEEDED


def test_job_runner_skips_failed_dependency_and_cooperative_cancel():
    runner = JobRunner()
    runner.add(JobSpec("fail", lambda ctx: 1 / 0))
    runner.add(JobSpec("dependent", lambda ctx: "never", dependencies=("fail",)))
    assert runner.run("fail").status is JobStatus.FAILED
    assert runner.run("dependent").status is JobStatus.SKIPPED
    assert runner.cancel("unknown") is False
