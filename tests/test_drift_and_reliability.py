"""Tests for workload drift, burst estimator and reliability metrics — Review 2 extensions.

This module enforces the observable/evaluation boundary fixed in
healing/burst_estimator.py: controller observations must not depend on hidden
ground-truth burst state when legitimate observable history is held constant.
"""

import copy

import numpy as np
import pytest

from config import build_default_config, SchedulerConfig
from errors import ValidationError
from evaluation.reliability import compute_reliability_metrics, jains_fairness_index, reliability_summary_for_arms
from healing.burst_estimator import BurstEstimator, perturb_observation
from healing.config import HealingConfig
from scheduler.runtime import ObservedProcess, RuntimeObservation
from workload.drift import DRIFT_KINDS, DriftSpec, apply_drift, injection_time_for_workload
from workload.generator import WorkloadGenerator
from workload.models import Workload, Process


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_drift_kinds_are_declared():
    assert "burst_shock" in DRIFT_KINDS
    assert "arrival_storm" in DRIFT_KINDS
    assert "phase_transition" in DRIFT_KINDS
    assert "priority_inversion" in DRIFT_KINDS


def test_drift_preserves_pre_injection_prefix():
    gen = WorkloadGenerator(build_default_config().families)
    base = gen.generate("mixed", 42)
    for kind in ("burst_shock", "arrival_storm", "priority_inversion"):
        drifted = gen.generate_with_drift("mixed", 42, kind)
        boundary = injection_time_for_workload(base)
        pre_base = [p for p in base if p.arrival_time < boundary]
        pre_drift = [p for p in drifted if p.arrival_time < boundary]
        assert len(pre_base) == len(pre_drift)
        for a, b in zip(pre_base, pre_drift):
            assert a.pid == b.pid
            assert a.arrival_time == b.arrival_time
            # Pre-injection burst/priority must be bit-identical.
            assert a.burst_time == b.burst_time
            assert a.priority == b.priority


def test_drift_burst_shock_increases_burst():
    gen = WorkloadGenerator(build_default_config().families)
    base = gen.generate("mixed", 7)
    drifted = gen.generate_with_drift("mixed", 7, "burst_shock")
    boundary = injection_time_for_workload(base)
    for orig, new in zip(base.processes, drifted.processes):
        if orig.arrival_time >= boundary:
            assert new.burst_time >= orig.burst_time


def test_drift_arrival_storm_compresses_arrivals():
    gen = WorkloadGenerator(build_default_config().families)
    base = gen.generate("mixed", 9)
    drifted = gen.generate_with_drift("mixed", 9, "arrival_storm")
    assert max(p.arrival_time for p in drifted) <= max(p.arrival_time for p in base)


def test_drift_phase_transition_requires_target():
    gen = WorkloadGenerator(build_default_config().families)
    base = gen.generate("mixed", 5)
    spec = DriftSpec(kind="phase_transition", phase_target_family="short_jobs", seed=5)
    with pytest.raises(ValidationError):
        apply_drift(base, spec, target_workload=None)
    target = gen.generate("short_jobs", 99)
    result = apply_drift(base, spec, target_workload=target)
    assert len(result) == len(base)
    assert result.name.endswith("phase_transition")


def test_drift_spec_validation():
    with pytest.raises(ValidationError):
        DriftSpec(kind="unknown_kind")
    with pytest.raises(ValidationError):
        DriftSpec(kind="burst_shock", injection_fraction=0)
    with pytest.raises(ValidationError):
        DriftSpec(kind="phase_transition")


def test_drift_injection_preserves_pid_count_and_order():
    """Drift must not change process count and must keep pid-arrival pairing."""
    gen = WorkloadGenerator(build_default_config().families)
    base = gen.generate("mixed", 123)
    for kind in DRIFT_KINDS:
        if kind == "phase_transition":
            target = gen.generate("short_jobs", 999)
            drifted = apply_drift(base, DriftSpec(kind=kind, phase_target_family="short_jobs", seed=123), target_workload=target)
        else:
            drifted = gen.generate_with_drift("mixed", 123, kind)
        assert len(drifted) == len(base)
        assert sorted(p.pid for p in drifted) == sorted(p.pid for p in base)


# ---------------------------------------------------------------------------
# Burst estimator — observable behaviour
# ---------------------------------------------------------------------------

def _make_obs(pid_bursts, current_time=10):
    """Helper: ground-truth observation with given (pid, burst, remaining)."""
    procs = []
    for pid, burst, rem in pid_bursts:
        procs.append(ObservedProcess(pid=pid, arrival_time=0, burst_time=burst, remaining_burst=rem, priority=1))
    return RuntimeObservation(
        current_time=current_time,
        ready_processes=tuple(procs),
        completed_count=0,
        arrived_count=len(procs),
        observed_mean_burst=float(sum(b for _, b, _ in pid_bursts) / len(pid_bursts)) if pid_bursts else 0,
        observed_priority_spread=0,
        mean_ready_arrival_age=1.0,
        previous_policy=None,
    )


def test_burst_estimator_causal():
    est = BurstEstimator(alpha=0.5, error_cv=0.0, seed=0)
    assert est.predict("global") == 10  # initial_guess
    est.observe_completed(20, bucket="short_jobs")
    est.observe_completed(20, bucket="short_jobs")
    assert est.predict("short_jobs") == 20
    assert est.global_mean == 20.0
    est.reset()
    assert est.predict("short_jobs") == 10
    # Service tracking also cleared.
    assert est.service_done(1) == 0


def test_burst_estimator_records_service_and_completion():
    est = BurstEstimator(alpha=0.5, error_cv=0.0, seed=0)
    est.record_service(pid=1, service=4)
    est.record_service(pid=1, service=3)
    assert est.service_done(1) == 7
    # Completion uses observable total.
    est.record_completion(pid=1, bucket="global")
    assert est.predict("global") == 7
    assert est.service_done(1) == 0  # cleared after completion
    with pytest.raises(ValidationError):
        est.record_completion(pid=999)


def test_burst_estimator_with_noise_produces_error():
    est = BurstEstimator(alpha=0.5, error_cv=0.5, error_bias=0.0, seed=123)
    est.observe_completed(15, bucket="global")
    pred, err = est.predict_with_noise(true_burst=15, bucket="global")
    assert pred >= 1
    assert err is not None
    assert err.true_burst == 15
    assert isinstance(err.relative_error, float)
    # Error is evaluation-only: prediction does not use true_burst.
    est2 = BurstEstimator(alpha=0.5, error_cv=0.5, error_bias=0.0, seed=123)
    est2.observe_completed(15, bucket="global")
    pred2, _ = est2.predict_with_noise(true_burst=None, bucket="global")
    # Same seed, same history → same prediction regardless of true_burst value.
    assert pred == pred2


def test_burst_estimator_zero_noise_is_deterministic_not_ground_truth():
    """Zero-noise means no added log-normal noise; prediction equals EWMA mean,
    not ground truth. Documented as 'no added noise', not 'unbiased/accurate'."""
    est = BurstEstimator(alpha=1.0, error_cv=0.0, seed=0, initial_guess=10)
    # Seed EWMA with two samples mean=20.
    est.observe_completed(10, bucket="global")
    est.observe_completed(30, bucket="global")
    # EWMA with alpha=1.0 tracks last sample: 30
    assert est.predict("global") == 30
    # Even though a future job's true burst is 100, zero-noise prediction stays 30.
    pred, err = est.predict_with_noise(true_burst=100, bucket="global")
    assert pred == 30
    # Repeated calls are deterministic.
    pred2, _ = est.predict_with_noise(true_burst=1, bucket="global")
    assert pred2 == 30
    assert err is not None
    assert err.relative_error != 0.0  # error shows inaccuracy, not zero


def test_perturb_observation_observable_only_remaining():
    """Remaining estimate must be max(1, predicted_total - service_done),
    not derived from hidden proc.remaining_burst proc.burst_time."""
    obs = RuntimeObservation(
        current_time=10,
        ready_processes=(
            ObservedProcess(pid=1, arrival_time=0, burst_time=100, remaining_burst=90, priority=1),
            ObservedProcess(pid=2, arrival_time=0, burst_time=5, remaining_burst=1, priority=1),
        ),
        completed_count=0,
        arrived_count=2,
        observed_mean_burst=52.5,
        observed_priority_spread=0,
        mean_ready_arrival_age=1.0,
        previous_policy=None,
    )
    est = BurstEstimator(alpha=0.5, error_cv=0.0, seed=0, initial_guess=10)
    # No history → predicted_total=10 for both.
    # Service for pid 1 is 0 (never dispatched), so remaining should be 10-0=10,
    # NOT 100* (1 - progress) which would be 100 * (remaining/burst) = 90 etc.
    # The old leaky code would have produced 9 (10 * 90/100) and 2 etc.
    # Fixed code produces observable-based remaining.
    est.record_service(pid=1, service=4)  # pid1 received 4 units of service
    # pid2 never dispatched → service 0
    noisy = perturb_observation(obs, est)
    p1 = next(p for p in noisy.observation.ready_processes if p.pid == 1)
    p2 = next(p for p in noisy.observation.ready_processes if p.pid == 2)
    # predicted_total=10 for both, service p1=4 ⇒ remaining 6; service p2=0 ⇒ remaining 10
    assert p1.burst_time == 10
    assert p1.remaining_burst == 6  # 10-4
    assert p2.burst_time == 10
    assert p2.remaining_burst == 10  # 10-0
    # Crucially, these values are independent of hidden 100/90 and 5/1.
    # Change hidden bursts but keep same estimator history/service → same observation.


def test_observation_independence_from_hidden_ground_truth():
    """Regression: identical observable history + config must yield identical
    controller observation even if hidden true burst differs arbitrarily."""
    # Same legitimate observable history: estimator seeded with same completions
    # and same per-pid service.
    def make_estimator():
        e = BurstEstimator(alpha=0.5, error_cv=0.3, seed=42, initial_guess=12)
        e.observe_completed(20, bucket="global")
        e.observe_completed(15, bucket="global")
        e.record_service(pid=1, service=5)
        # pid 2 never dispatched → service 0 (do not record, defaults to 0)
        return e

    # Two ground-truth observations with wildly different hidden bursts/remaining,
    # but same pids/arrivals/priority/current_time. arrived_count must be >= completed+ready.
    true_a = RuntimeObservation(
        current_time=20,
        ready_processes=(
            ObservedProcess(pid=1, arrival_time=0, burst_time=8, remaining_burst=2, priority=1),
            ObservedProcess(pid=2, arrival_time=1, burst_time=200, remaining_burst=199, priority=2),
        ),
        completed_count=2,
        arrived_count=4,
        observed_mean_burst=104.0,
        observed_priority_spread=1,
        mean_ready_arrival_age=5.0,
        previous_policy=None,
    )
    true_b = RuntimeObservation(
        current_time=20,
        ready_processes=(
            ObservedProcess(pid=1, arrival_time=0, burst_time=50, remaining_burst=45, priority=1),
            ObservedProcess(pid=2, arrival_time=1, burst_time=3, remaining_burst=1, priority=2),
        ),
        completed_count=2,
        arrived_count=4,
        observed_mean_burst=26.5,
        observed_priority_spread=1,
        mean_ready_arrival_age=5.0,
        previous_policy=None,
    )
    est_a = make_estimator()
    est_b = make_estimator()
    out_a = perturb_observation(true_a, est_a)
    out_b = perturb_observation(true_b, est_b)
    # Controller-visible observations must be identical (same estimator state).
    assert out_a.observation.ready_processes == out_b.observation.ready_processes
    assert out_a.observation.observed_mean_burst == out_b.observation.observed_mean_burst
    # Evaluation-only errors will differ (they contain true_burst), but those
    # are not observable by controller.
    assert out_a.errors[0].true_burst != out_b.errors[0].true_burst


def test_causal_estimation_future_does_not_affect_past():
    """Future completions must not influence past predictions."""
    est_past_before = BurstEstimator(alpha=0.5, error_cv=0.0, seed=0, initial_guess=10)
    # Shared history up to t: two completions.
    for est in (est_past_before,):
        est.observe_completed(10, bucket="global")
        est.observe_completed(20, bucket="global")
    pred_at_t_before_future = est_past_before.predict("global")  # after 2 samples: (10+20)/2? Actually EWMA alpha0.5
    # Compute EWMA: first 10, then 0.5*20+0.5*10=15 → global_mean = (10+20)/2=15, but EWMA bucket =15, global_mean=15 → predict 15
    assert pred_at_t_before_future == 15

    # Now advance one branch with a future completion, other without.
    est_with_future = copy.deepcopy(est_past_before)
    est_with_future.observe_completed(100, bucket="global")
    # At time t (before future), prediction was 15; after future, prediction is different.
    # But if we snapshot the estimator at time t, its prediction must not have anticipated future.
    # To enforce causality, we check: cloning before future preserves snapshot.
    est_snapshot = copy.deepcopy(est_past_before)  # snapshot at t
    assert est_snapshot.predict("global") == 15
    assert est_with_future.predict("global") != 15  # future does affect later, as expected
    # The key: prediction at t is identical whether or not we later observe future.
    # This holds because estimator only appends; earlier predictions are not retroactively changed.


def test_minimum_remaining_bounds_and_edge_cases():
    """Remaining must be >=1, <=predicted_total, handle zero/negative over-predict."""
    est = BurstEstimator(alpha=1.0, error_cv=0.0, seed=0, initial_guess=5)
    # Predicted total =5
    assert est.predict("global") == 5

    # Case: service 0 → remaining = predicted
    obs0 = _make_obs([(1, 100, 100)], current_time=0)
    out0 = perturb_observation(obs0, est)
    assert out0.observation.ready_processes[0].remaining_burst == 5
    assert out0.observation.ready_processes[0].burst_time == 5

    # Case: service exceeds prediction → clamp to 1 (minimum), not 0/negative
    est2 = BurstEstimator(alpha=1.0, error_cv=0.0, seed=0, initial_guess=5)
    est2.record_service(pid=1, service=10)  # more than predicted 5
    out1 = perturb_observation(obs0, est2)
    assert out1.observation.ready_processes[0].remaining_burst == 1

    # Case: service exactly equals prediction → remaining 0 would be invalid → clamp to 1
    est3 = BurstEstimator(alpha=1.0, error_cv=0.0, seed=0, initial_guess=5)
    est3.record_service(pid=1, service=5)
    out2 = perturb_observation(obs0, est3)
    assert out2.observation.ready_processes[0].remaining_burst == 1

    # Case: multiple pids, each bounded independently.
    est4 = BurstEstimator(alpha=1.0, error_cv=0.0, seed=0, initial_guess=10)
    est4.record_service(pid=1, service=9)  # remaining 1
    # pid 2 never dispatched → service 0, remaining 10
    obs2 = _make_obs([(1, 999, 999), (2, 999, 1)], current_time=5)
    out3 = perturb_observation(obs2, est4)
    r1 = next(p for p in out3.observation.ready_processes if p.pid == 1)
    r2 = next(p for p in out3.observation.ready_processes if p.pid == 2)
    assert r1.remaining_burst == 1
    assert r2.remaining_burst == 10
    assert r1.remaining_burst <= r1.burst_time
    assert r2.remaining_burst <= r2.burst_time


def test_perturb_hides_burst_and_produces_valid_remaining():
    """With noisy estimator, perturbed bursts vary but remaining stays valid."""
    obs = RuntimeObservation(
        current_time=10,
        ready_processes=(
            ObservedProcess(pid=1, arrival_time=0, burst_time=10, remaining_burst=10, priority=1),
            ObservedProcess(pid=2, arrival_time=2, burst_time=40, remaining_burst=30, priority=2),
        ),
        completed_count=0,
        arrived_count=2,
        observed_mean_burst=25.0,
        observed_priority_spread=1,
        mean_ready_arrival_age=5.0,
        previous_policy=None,
    )
    est_exact = BurstEstimator(error_cv=0.0, seed=0, initial_guess=12)
    noisy = perturb_observation(obs, est_exact)
    assert len(noisy.observation.ready_processes) == 2
    for p in noisy.observation.ready_processes:
        assert 1 <= p.remaining_burst <= p.burst_time
        assert p.burst_time >= 1
    # Noisy vs exact should differ in distribution, not just structure.
    est_noisy = BurstEstimator(error_cv=0.8, seed=42, initial_guess=12)
    # Seed history so predictions have variance: observe a few totals.
    for v in [10, 30, 5, 50]:
        est_noisy.observe_completed(v, bucket="global")
    # Over multiple draws with same history, noise produces different totals
    # because draw_counter advances. Test genuineness: collect 5 predictions.
    est_clone = copy.deepcopy(est_noisy)
    preds = set()
    for _ in range(5):
        pred, _ = est_clone.predict_with_noise(bucket="global")
        preds.add(pred)
    # With cv=0.8, noise should produce at least 2 distinct predictions in 5 draws.
    assert len(preds) >= 2, f"expected varied noisy predictions, got {preds}"
    # All predictions still positive.
    assert all(p >= 1 for p in preds)


def test_burst_estimator_validation():
    with pytest.raises(ValidationError):
        BurstEstimator(alpha=0)
    with pytest.raises(ValidationError):
        BurstEstimator(error_cv=-1)
    with pytest.raises(ValidationError):
        BurstEstimator(initial_guess=0)
    est = BurstEstimator()
    with pytest.raises(ValidationError):
        est.record_service(pid=1, service=0)
    with pytest.raises(ValidationError):
        est.record_service(pid=-1, service=5)
    with pytest.raises(ValidationError):
        est.observe_completed(0, bucket="global")


def test_healing_config_with_new_flags():
    cfg = HealingConfig()
    assert cfg.drift_enabled is False
    assert cfg.burst_estimator_enabled is False
    cfg2 = HealingConfig(drift_enabled=True, drift_kind="arrival_storm", burst_estimator_enabled=True, burst_estimator_error_cv=0.5)
    assert cfg2.drift_kind == "arrival_storm"
    assert cfg2.burst_estimator_error_cv == 0.5
    with pytest.raises(ValidationError):
        HealingConfig(drift_kind="invalid")
    with pytest.raises(ValidationError):
        HealingConfig(burst_estimator_error_cv=-1)


def test_reliability_jains_fairness():
    assert jains_fairness_index([10, 10, 10]) == pytest.approx(1.0)
    assert 0 < jains_fairness_index([1, 100]) < 1.0
    assert jains_fairness_index([]) == 1.0
    assert jains_fairness_index([5]) == 1.0


def test_reliability_metrics_from_schedule():
    from evaluation.metrics import compute_metrics
    from scheduler.runtime import RuntimeSimulator
    from workload.models import Workload, Process

    wl = Workload("test", [
        Process(pid=1, arrival_time=0, burst_time=5, priority=1),
        Process(pid=2, arrival_time=1, burst_time=3, priority=2),
        Process(pid=3, arrival_time=2, burst_time=2, priority=1),
    ])
    from scheduler.fcfs import FCFS
    from workload.models import build_schedule_result
    from config import SchedulerConfig as SCfg
    result = FCFS(SCfg(round_robin_quantum=4, switching_cost=0)).run(wl)
    metrics = compute_reliability_metrics(result, incidents=[], audit_gains=[])
    assert metrics.mean_waiting >= 0
    assert 0 <= metrics.fairness_index <= 1.0
    assert metrics.p95_waiting >= metrics.mean_waiting
    assert metrics.rollback_rate is None  # no interventions
    assert metrics.fallback_triggered is False
    # P95 and max are well-defined on non-empty schedule.
    assert metrics.max_waiting >= metrics.p95_waiting

    class FakeIncident:
        action_changed = True
        verdict = "rejected"
        verification_decisions = 5
        simulated_time = 10
        window_end_time = 20
        fallback_triggered = False

    class FakeIncident2(FakeIncident):
        verdict = "accepted"

    m2 = compute_reliability_metrics(result, incidents=[FakeIncident(), FakeIncident2()], audit_gains=[10.0, -5.0])
    assert m2.interventions == 2
    assert m2.rollback_rate == pytest.approx(0.5)
    assert m2.false_intervention_rate == pytest.approx(0.5)
    assert m2.recovery_delay_decisions == 5.0


def test_reliability_summary_for_arms():
    from evaluation.reliability import compute_reliability_metrics
    from workload.models import Workload, Process, ScheduleResult
    from scheduler.fcfs import FCFS
    from config import SchedulerConfig

    wl = Workload("test", [
        Process(pid=1, arrival_time=0, burst_time=5, priority=1),
        Process(pid=2, arrival_time=1, burst_time=3, priority=2),
    ])
    result = FCFS(SchedulerConfig(switching_cost=0)).run(wl)
    m_fcfs = compute_reliability_metrics(result, incidents=[], audit_gains=[])
    m_other = compute_reliability_metrics(result, incidents=[], audit_gains=[])
    import dataclasses
    m_worse = dataclasses.replace(m_other, mean_waiting=m_other.mean_waiting + 10.0)

    summary = reliability_summary_for_arms({
        "fixed_fcfs": [m_fcfs],
        "rule_wait": [m_worse],
    }, baseline_arm="fixed_fcfs")
    assert len(summary) == 2
    rule_row = next(r for r in summary if r["arm"] == "rule_wait")
    assert rule_row["regression_rate"] == 1.0
    assert rule_row["mean_waiting"] == m_worse.mean_waiting


def test_workload_generator_with_drift_api():
    gen = WorkloadGenerator(build_default_config().families)
    w = gen.generate_with_drift("short_jobs", 42, "burst_shock")
    assert w.name.endswith("burst_shock")
    assert len(w) == 15
    with pytest.raises(ValidationError):
        gen.generate_with_drift("short_jobs", 42, "unknown_drift")


def test_rollback_and_fallback_metrics():
    """Verify rollback/fallback counting matches incident ledger semantics."""
    from scheduler.fcfs import FCFS

    wl = Workload("test", [
        Process(pid=1, arrival_time=0, burst_time=4, priority=1),
        Process(pid=2, arrival_time=0, burst_time=4, priority=1),
    ])
    result = FCFS(SchedulerConfig(switching_cost=0)).run(wl)

    class Inc:
        def __init__(self, verdict, fallback=False):
            self.action_changed = True
            self.verdict = verdict
            self.verification_decisions = 5
            self.simulated_time = 0
            self.window_end_time = 5
            self.fallback_triggered = fallback

    # 2 interventions: 1 accepted (kept), 1 inconclusive (rolled back) → rollback 0.5
    incs = [Inc("accepted"), Inc("inconclusive")]
    m = compute_reliability_metrics(result, incidents=incs, audit_gains=[])
    assert m.rollback_rate == pytest.approx(0.5)
    assert m.fallback_triggered is False
    # Fallback triggered on any incident with fallback flag.
    incs2 = [Inc("rejected", fallback=True)]
    m2 = compute_reliability_metrics(result, incidents=incs2, audit_gains=[])
    assert m2.fallback_triggered is True
