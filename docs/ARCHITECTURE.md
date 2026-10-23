# Architecture

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

The project system is the **self-healing CPU scheduler**. This document describes its
modules, its data flow and — most importantly — the boundary that makes its claims causal.

## 1. Layers

```
workload/         reproducible workloads and their models
scheduler/        the event-driven engine (RuntimeSimulator) and the four fixed policies
healing/          the self-healing controller: monitor, detector, diagnosis, actions,
                  recovery policies, verification, rollback, ledger
rl/               Q-learning agent, runtime controllers, offline selector and reward models
experiments/      the studies that run the above and write artifacts
evaluation/       metrics, comparison statistics and paired bootstrap
visualization/    figures rendered from the study artifacts
```

### `workload/`

* `models.py` — `Process`, `Workload` (canonically ordered, fingerprinted),
  `ProcessOutcome`, `ScheduleResult`.
* `generator.py` — `WorkloadGenerator.generate(family, seed)` draws a workload from a
  declared family (arrival pattern × burst distribution × priority pattern) with
  `numpy.random.default_rng(seed)`, so a `(family, seed)` pair is a reproducible instance;
  `generate_with_drift(family, seed, drift_kind)` injects a deterministic mid-trace phase shift
  while preserving the pre-injection prefix.
* `fingerprint.py` — content fingerprint used to prove that two arms were evaluated on the
  same instance.
* `drift.py` — **workload drift**: `DriftSpec` + `apply_drift` inject `burst_shock`,
  `arrival_storm`, `phase_transition` (batch→interactive) and `priority_inversion` at a
  declared fraction of the arrival span (default 0.35).  Pre-injection work is untouched,
  so the stable and drifted traces share a prefix — the controller cannot distinguish them
  until the drift arrives.

### `scheduler/`

* `runtime.py` — the event engine. One evolving FIFO ready queue; FCFS/RR dispatch the head,
  SJF picks the shortest remaining burst, Priority picks the highest priority. Non-RR service
  is non-preemptive; RR service is `min(quantum, remaining_burst)`. Switching cost is charged
  when the running process changes. The engine
  * exposes only `RuntimeObservation` (ready set, counters, observed mean burst, previous
    policy) to a controller and never the workload, the future event list or the schedule;
  * records per-decision `RuntimeDecisionRecord`s, including the processes that completed
    during the interval and their realized waiting times;
  * accepts an optional per-decision `quantum` override, so a controller can change the
    Round-Robin quantum without touching the reward normalization (documented in the class
    docstring).
* `fcfs.py`, `sjf.py`, `round_robin.py`, `priority.py` — standalone reference schedulers used
  as untouched baselines; `POLICY_CLASSES` maps names to classes. The test suite pins the
  engine to these implementations for constant configurations.

### `healing/`

| Module | Responsibility |
|---|---|
| `config.py` | every declared threshold, gate, cost and bound, validated in `__post_init__` and serialized into the summary artifact; **reliability extensions** `burst_estimator_*` and `drift_*` are additive flags (default off) so the frozen evaluation stays byte-identical |
| `monitor.py` | `CausalMonitor`: rolling completion window, EWMA reference and dispersion band with floors, trailing-window load level and its own reference/band; `MonitorSnapshot` is the immutable evidence object |
| `detector.py` | `DegradationDetector.evaluate(...)` → `DetectionOutcome`. Suspect when the window mean exceeds `reference + k·band`; declared only after hysteresis, outside cooldown, inside the intervention budget, past the ready/work/congestion gates. Every suppression path has its own reason, so "no action" is always explained in the ledger |
| `diagnosis.py` | `build_ready_evidence` + `diagnose`: a fixed-precedence mapping from *visible* ready-set evidence to one primary category (head-of-line blocking, fairness starvation, priority-ordering anomaly, tail dispersion, queue congestion) or `unexplained`, with the matched evidence attached |
| `actions.py` | the 8-action recovery space, `SchedulerSettings`, `apply_healing_action`, and the no-op rule: an action that cannot change the live configuration reports `changed=False` and is never counted as an intervention |
| `policies.py` | `HealingContext` + `HealingContextEncoder` (648 states), `MappedRulePolicy` (predeclared diagnosis→action rules), `HoldOnlyPolicy` (the detection control), `QLearningHealingPolicy` (epsilon-greedy with forced first visits, masked exploration and masked bootstrapping; greedy and read-only during evaluation) |
| `verifier.py` | `verify(VerificationEvidence, config) -> VerificationVerdict`: the tri-state verdict. It subtracts the declared **phase baseline** — a least-squares prediction of the *no-action* post-window queue level from quantities observed at the trigger (`trigger_load`, `ready_count`, `log1p(visible_work)`, fitted on the training control) — converted into the statistic's own units through its denominator, and compares the residual with the declared accept / reject boundaries (+0.30 / −0.30, the 95th and 5th percentiles of the control residual). It also refuses to judge a window with too little telemetry (`insufficient_evidence_thin_telemetry`, `insufficient_evidence_window_truncated`). `rollback_required` folds in the declared `inconclusive_rollback` posture |
| `burst_estimator.py` | **Uncertainty boundary**: `BurstEstimator` (causal modelled EWMA per-bucket predictor plus log-normal multiplicative noise, `EWMA × exp(N(bias, sigma²))` where `sigma = error_cv` is the log-space standard deviation (name retained for backward compatibility; not the multiplier's CV ≈ 0.307 for sigma=0.30), **not** calibrated or validated against real traces) and `perturb_observation` which hides exact `remaining_burst` behind noisy estimates.  The controller sees only the estimate; evaluation still measures against ground truth.  Fixed SJF still receives exact burst lengths in the same experiment and is therefore an oracle baseline, not an equal-information comparison.  Default `error_cv=0` degenerates to exact information (backwards compatible) |
| `controller.py` | `SelfHealingController`: the state machine (idle → verification window open → verdict), the incident ledger, the reward assignment, the rollback at the next epoch, the safe fallback, and the counterfactual replay helpers (`forced_holds`) |

### `experiments/` and `evaluation/`

* `healing_config.py` — frozen design: `SPLIT_TAGS`, seeds, repetitions, arms, the six
  training families (all but `poisson_arrivals`).
* `healing_experiment.py` — training, per-split evaluation of every arm, detection summary,
  the 10 predeclared paired comparisons, the counterfactual audit, the action-effect probe,
  Q-table coverage, demo selection and the generated report.
* `healing_thresholds.py` — derives the declared boundaries from the **no-action control** on
  the training split: the congestion gate's effect, the null distribution of the closure
  statistic (p50/p90/p95), the phase model (`post_load_baseline`) and its residual percentiles,
  the comparison of candidate phase-model specifications (`healing_verification_models.csv`),
  plus the measured false-accept / false-reject rates of the declared rule on that control and
  out of sample on the validation control.
* `healing_detection.py` — controlled-degradation scenarios (`burst_shock`, `arrival_storm`)
  injected into real validation traces, run on the hold-only control, and summarised into
  recall, a precision proxy, detection delay and false triggers per stable trace.
* `evaluation/paired.py` — family-stratified, crossed-bootstrap paired differences shared
  with the runtime appendix.
* `evaluation/metrics.py` — `compute_metrics(ScheduleResult)` → waiting/turnaround/response
  times, p95/max waiting, CPU utilization, throughput, context switches, switching overhead.
* `evaluation/reliability.py` — **reliability beyond mean waiting time**:
  `compute_reliability_metrics` (P95, starvation count, Jain fairness index, recovery delay
  in decisions and time, false-intervention rate from counterfactual audit, rollback and
  regression rates, fallback rate) and `reliability_summary_for_arms` (per-arm aggregation
  against a baseline).  All definitions are on `ScheduleResult` + incident ledger and have
  no hidden parameters.

## 2. Data flow of one trace

1. `WorkloadGenerator` produces a `Workload` from `(split seed, family, repetition)`.
2. `RuntimeSimulator.run(workload, controller)` starts an event loop. At every decision epoch
   it builds a `RuntimeObservation` and calls `controller.select_action(...)`; it then runs
   the chosen slice, charges switching cost if the running process changed, collects
   completions, and calls `controller.observe_decision(record)` (when the controller defines
   it) followed by `observe_transition(...)`.
3. `SelfHealingController` maintains the monitor and detector. On a detected event it
   diagnoses, builds the context, asks the policy (with a masked action set), applies the
   action, and opens a verification window.
4. When the window closes, `healing/verifier.py` turns the evidence into one of four
   outcomes — `accepted`, `rejected` (a confirmed failure), `inconclusive` (insufficient
   evidence) or `held` (nothing was changed) — the controller applies the declared posture
   (`accepted` keeps the change, `rejected` and `inconclusive` roll back to the last accepted
   configuration), updates the learned policy from the reward, and records the incident.
5. `end_episode()` closes the ledger. A window still open at the trace end is concluded as
   `inconclusive` with the explicit reason `insufficient_evidence_window_truncated` and
   follows the same posture as any other inconclusive window (rolled back unless the
   configuration was declared to keep it), so no incident is ever silently discarded.
6. `experiments/healing_experiment.py` aggregates the ledger, writes CSV/JSON artifacts,
   audits interventions by counterfactual replay, and renders figures and the report.

## 3. The causal boundary (why the claims are about scheduling, not about cheating)

* The controller's only inputs are `RuntimeObservation` (the arrived, unfinished ready set at
  the current epoch) and completion notices of processes that have already finished. The
  workload object, its future arrivals, its fingerprints and the final schedule are never
  passed to it; this is enforced by tests (`tests/test_runtime_scheduler.py` asserts that the
  controller interfaces accept observations, not workloads).
* Detection thresholds are computed from *observed* completions: to trigger, the trace must
  first run badly, and the reference must have seen enough normal completions.
* Verification compares quantities that are all observed before the verdict.
* The reward the learner maximizes is the verified window statistic minus a declared
  intervention cost, so "act and fail" is worse than "hold".

## 4. Invariants the test suite pins

* With a control policy that never intervenes, the healing loop's schedule equals the
  corresponding standalone policy *exactly* (`detector_only == fixed_fcfs` in the study
  tables; asserted in `tests/test_healing_controller.py`).
* Masked-out actions are never requested, HOLD is always available, and an action that is
  already in force is reported as `changed=False`, never as a recovery.
* A rolled-back configuration is restored at the first epoch at or after the recorded end of
  the verification window; after the safe fallback the trace uses exactly one pinned
  configuration.
* Identical seeds reproduce identical schedules, identical Q tables and identical artifacts.
* Splits share no workload instance (fingerprints), and the derivation of the declared
  thresholds rejects the final-test split by construction.
