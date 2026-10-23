"""Small, inspectable teacher scenarios against real public project APIs.

Run from repository root: PYTHONPATH=. .venv/bin/python docs/manual_evidence/scenarios.py
These checks do not invoke pytest, tune thresholds, or overwrite research evidence.
"""
from config import SchedulerConfig, build_default_config
from evaluation.metrics import compute_metrics
from healing import HealingConfig, SelfHealingController
from healing.actions import SchedulerSettings, apply_healing_action
from healing.policies import RuleWaitingTimePolicy
from scheduler import POLICY_CLASSES, RuntimeSimulator
from scheduler.runtime import RuntimeActionDecision
from workload.models import Process, Workload
from workload.generator import WorkloadGenerator
from workload.drift import injection_time_for_workload


def show(result):
    print('  service:', [(s.pid, s.start_time, s.end_time) for s in result.trace])
    print('  pid arrival burst first completion wait turnaround response')
    for o in result.outcomes:
        print(' ', o.pid, o.arrival_time, o.burst_time, o.start_time,
              o.completion_time, o.waiting_time, o.turnaround_time, o.response_time)
    m = compute_metrics(result)
    print(f'  mean wait={m.avg_waiting_time:.6f}, turnaround={m.avg_turnaround_time:.6f}')


def rejected(label, operation):
    try:
        operation()
    except (ValueError, TypeError) as exc:
        print(f'EXPECTED REJECTION {label}: {type(exc).__name__}: {exc}')
    else:
        raise AssertionError(f'{label} unexpectedly accepted')


def main():
    cfg = SchedulerConfig(round_robin_quantum=2, switching_cost=0)
    workload = Workload('teacher-hand-example', [Process(1, 0, 5, 2),
                                                Process(2, 0, 3, 1), Process(3, 0, 1, 3)])
    print('HAND EXAMPLE: all arrivals 0; P1 burst 5 priority 2; P2 3/1; P3 1/3; q=2 switch cost=0')
    expected = {'FCFS': ([0, 5, 8], [5, 8, 9]),
                'SJF': ([4, 1, 0], [9, 4, 1]),
                'Priority': ([3, 0, 8], [8, 3, 9]),
                'Round Robin': ([4, 5, 4], [9, 8, 5])}
    for cls in POLICY_CLASSES:
        result = cls(cfg).run(workload)
        print(cls.name)
        show(result)
        waits, ends = expected[cls.name]
        assert [o.waiting_time for o in result.outcomes] == waits
        assert [o.completion_time for o in result.outcomes] == ends
    print('HAND CALCULATIONS MATCH all four algorithms')
    for cls in POLICY_CLASSES:
        single = cls(cfg).run(Workload('one-late-job', [Process(1, 5, 3, 1)]))
        assert single.outcomes[0].waiting_time == 0 and single.outcomes[0].completion_time == 8
        empty = cls(cfg).run(Workload('empty-explicit', [], allow_empty=True))
        assert not empty.outcomes and empty.total_elapsed_time == 0
        print(f'EDGE {cls.name}: late one-job completion=8 wait=0 idle=5; explicit empty elapsed=0')
    boundary = Workload('slice-boundary', [Process(1, 0, 4, 1), Process(2, 2, 1, 1)])
    rr = POLICY_CLASSES[2](cfg).run(boundary)
    print('RR arrival exactly at quantum end:'); show(rr)
    assert [s.pid for s in rr.trace] == [1, 2, 1]
    defaults = build_default_config()
    stable = Workload('stable-spaced', [Process(i+1, i*10, 2, 1) for i in range(15)])
    ctrl = SelfHealingController(HealingConfig(), defaults.scheduler, RuleWaitingTimePolicy(HealingConfig()))
    ctrl.begin_episode(0)
    run = RuntimeSimulator(defaults.scheduler).run(stable, ctrl, training=False)
    episode = ctrl.end_episode()
    print('NORMAL HEALING LOOP: decisions=', len(run.decisions), 'incidents=', len(episode.incidents),
          'mean waiting=', compute_metrics(run.result).avg_waiting_time)
    assert not episode.incidents
    gen = WorkloadGenerator(defaults.families)
    base = gen.generate('mixed', 42)
    phase = gen.generate_with_drift('mixed', 42, 'phase_transition')
    boundary_time = injection_time_for_workload(base)
    print('PHASE TRANSITION mixed seed=42 -> short_jobs; boundary=', boundary_time)
    print('  pid arrival base_burst phase_burst base_priority phase_priority')
    for p, d in zip(base, phase):
        print(' ', p.pid, p.arrival_time, p.burst_time, d.burst_time, p.priority, d.priority)
        if p.arrival_time < boundary_time:
            assert p == d
        assert p.arrival_time == d.arrival_time
    assert phase == gen.generate_with_drift('mixed', 42, 'phase_transition')
    rejected('empty default workload', lambda: Workload('empty', []))
    rejected('malformed process', lambda: Workload('bad', ['not a Process']))
    rejected('duplicate pid', lambda: Workload('bad', [Process(1, 0, 2, 1), Process(1, 1, 2, 1)]))
    rejected('negative arrival', lambda: Process(1, -1, 2, 1))
    rejected('zero burst', lambda: Process(1, 0, 0, 1))
    rejected('zero quantum', lambda: SchedulerConfig(round_robin_quantum=0))
    rejected('runtime action 99', lambda: RuntimeActionDecision(99))
    rejected('healing action 99', lambda: apply_healing_action(99, current=SchedulerSettings(0, 4),
             last_accepted=None, min_quantum=1, max_quantum=16))
    rejected('unknown family', lambda: gen.generate('unknown', 42))
    rejected('unknown phase target', lambda: gen.generate_with_drift('mixed', 42, 'phase_transition', phase_target_family='unknown'))
    print('ALL direct API scenarios completed')


if __name__ == '__main__':
    main()
