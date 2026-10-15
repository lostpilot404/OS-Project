# Methodology

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

This document states exactly what is measured, on what data, with which statistics, and
which parts of the pipeline are causal. Every number quoted here is reproduced in
`docs/EVALUATION_RESULTS.md` and traceable to a file under `results/healing/`.

## 1. Object of study

The **self-healing CPU scheduler**: a scheduling loop that (i) monitors realized performance
from completed work only, (ii) declares a degradation event against a declared reference with
threshold, hysteresis and cooldown, (iii) diagnoses a cause from the visible ready set,
(iv) selects one of eight recovery actions with a predeclared rule or a tabular Q-learning
controller, (v) applies it, (vi) verifies the outcome over a declared window, and
(vii) accepts it, rolls it back, or falls back to a pinned safe configuration — recording the
decision, its evidence, its cost and its outcome.

Two questions are being asked:

1. **Does the loop help relative to doing nothing?** (`rule_wait`/`q_recovery` vs
   `detector_only`, which runs the full monitoring/detection/diagnosis machinery but never
   intervenes.)
2. **Does the loop beat a fixed scheduler?** (`rule_wait`/`q_recovery` vs `fixed_sjf`,
   `fixed_fcfs`, `fixed_round_robin`, `fixed_priority`.)

Both are answered on a held-out seed, with paired statistics over identical workloads, and
with a counterfactual audit of every intervention.

## 2. Workloads and splits

Families (`config.py`, seven in total): `short_jobs`, `long_jobs`, `mixed`, `cpu_bursty`,
`priority_skewed`, `staggered_interactive`, `poisson_arrivals`. Each family fixes the arrival
pattern, the burst distribution (uniform, bimodal or staggered-interactive) and the priority
pattern. A workload instance is `WorkloadGenerator.generate(family, seed)`; the resulting
`Workload` carries a content fingerprint.

Splits (`experiments/healing_config.py`, `SPLIT_TAGS = {training: 51, validation: 67,
final_test: 83}`):

| Split | Master seed | Traces | Used for |
|---|---:|---|---|
| training | `7101…7105` | 1200 per seed, cycling the six non-`poisson_arrivals` families | training the learned controllers; deriving the declared thresholds from the no-action control |
| validation | `8201` | 20 per family (140) | development decisions, the worked demonstration trace |
| final test | `20719` | 30 per family (210) | the reported comparison; touched once, after the design was frozen |

Workload seeds come from `derive_seed(master_seed, SPLIT_TAGS[split], family_index,
repetition)`, so the three splits are disjoint **by construction** (verified by a test that
compares fingerprints) and every arm is evaluated on exactly the same instances.

`poisson_arrivals` is deliberately excluded from training: the learned controllers are
evaluated on an arrival process they never saw, so the study measures distribution shift
instead of fitting it.

**Seed hygiene.** `20501` was used by an early end-to-end smoke run before the detector's
congestion gate and the verification rule were finalised, and the `9401` stream was used
while the harness was built; both are retired as evidence. The held-out seed `20719` was
declared in `experiments/healing_config.py` before the frozen run and was not evaluated with
the final system before it. The offline policy-selection study and the runtime appendix have
their own, separate seed streams.

## 3. Arms

| Arm | What it is |
|---|---|
| `fixed_fcfs`, `fixed_sjf`, `fixed_round_robin`, `fixed_priority` | the untouched standalone schedulers (`scheduler/*.py`), no adaptation |
| `detector_only` | the full monitoring/detection/diagnosis loop with a policy that never intervenes — the control that separates *detecting* from *acting* |
| `rule_fairness` | predeclared fairness-oriented diagnosis→action rule |
| `rule_wait` | predeclared waiting-time-oriented diagnosis→action rule |
| `q_recovery` | tabular Q-learning recovery, five independent training seeds (one arm each, reported together) |
| `rule_wait_from_sjf`, `q_recovery_from_sjf` | the same two recovery methods started from SJF instead of FCFS, to test whether the loop damages an already good configuration |

The rule arms and the learned arms all start from the same naive configuration (FCFS), except
the `*_from_sjf` arms, so the comparison measures the *recovery* and not the starting point.

## 4. Metrics

All metrics come from `evaluation/metrics.py::compute_metrics(ScheduleResult)` and are
computed from the finished schedule only:

* `avg_waiting_time`, `median_waiting_time`, `p95_waiting_time`, `max_waiting_time` — a
  process's waiting time is `start_time − arrival_time` over the *whole* run, including every
  slice it was preempted (Round Robin therefore shows low response time and high waiting
  time);
* `avg_turnaround_time` = `completion_time − arrival_time`;
* `avg_response_time` = time to the first slice;
* `cpu_utilization` (percent) and `throughput` (completions per time unit);
* `context_switches` and `switching_overhead`;
* healing-specific: `incidents`, `interventions`, `accepted`, `rolled_back`, `held`,
  `censored`, `fallback_triggered`, `final_policy`, `final_quantum`, plus per-incident
  evidence in the incident ledger.

The reported headline metric is **mean waiting time**; p95 and the audit gains are reported
alongside it because a mean alone hides exactly the starvation behaviour the loop claims to
fix.

## 5. Statistics

* **Paired comparisons.** Every arm is evaluated on the *same* workload instances, so
  differences are computed per workload and resampled: `evaluation/paired.py` implements a
  family-stratified, crossed bootstrap (model seeds × workloads) with 1000 replicates at seed
  `104729`, producing a mean difference and a 95 % percentile interval. Intervals are
  conditional on the declared design (five training seeds, seven families, fixed
  repetitions); they are not a claim about all workloads.
* **Ten predeclared comparisons** are computed (`healing_comparisons.csv`):
  `rule_wait` vs `detector_only`/`fixed_fcfs`/`fixed_sjf`, `rule_fairness` vs `fixed_fcfs`,
  `q_recovery` vs `detector_only`/`fixed_fcfs`/`fixed_sjf`/`rule_wait`,
  `q_recovery_from_sjf` vs `fixed_sjf`, `rule_wait_from_sjf` vs `fixed_sjf`.
* **Counterfactual audit** (the causal evidence). For every intervention the study replays
  the same trace with the recorded decision prefix and then *holds* the pre-incident
  configuration, so the state at the incident is identical and the only difference is the
  adaptation. `intervention_gain = hold_total_waiting − actual_total_waiting`; a negative gain
  means the intervention made the trace worse. An intervention is a **false recovery** if its
  gain is ≤ 0. This is the only measurement in the repository that can attribute an effect to
  a recovery action, and it is why the study can report that the in-run verdict and the truth
  disagree.
* **Detection statistics.** The control's incident ledger gives the detector's false-trigger
  rate (events declared while nothing changed), reported per trace.

## 6. Derived thresholds (not tuned by hand)

`experiments/healing_thresholds.py` runs the no-action control over the training split and
derives:

* `min_load_factor`: the congestion gate. The sweep in
  `healing_threshold_derivation.csv` shows control detections per trace falling from 1.52
  (gate off) to 1.05 (`0.8`), 0.86 (`1.0`) and 0.04 (`1.25`); `1.0` is the strictest value
  that still leaves the detector active, and it is the declared value.
* `accept_closure_factor`: the verification threshold, declared at the 95th percentile of
  the *no-action* closure statistic (measured 2.38 → declared 2.4). Its measured false-accept
  rate on the control is 4.5 % (training) and 3.1 % (validation): the rule certifies
  "success" on a window in which nothing changed at most 5 % of the time.

The verification rule is therefore a *conservative safety gate*, not a claim of benefit; the
audit supplies the benefit measurement.

## 7. Protocol actually followed

1. Build the healing package against the existing, tested event engine; pin its semantics
   with tests (identical schedules under a no-op control, rollback timing, fallback,
   counterfactual prefix equality).
2. Develop thresholds, gates and the reward shape on the **training** and **validation**
   splits only. Every rejected alternative is recorded in `docs/DESIGN_AND_CHOICES.md`
   together with the measurement that rejected it.
3. Freeze `experiments/healing_config.py`, declare a new held-out seed (`20719`), and run
   `python main.py healing-study` once.
4. Report every arm — including the arms that lose — and both the favourable and the
   unfavourable audit outcomes.

## 8. What is *not* claimed

* No real-OS integration: the system drives a simulator.
* No guarantee that recovery beats a fixed policy; the measured answer is that it does not, on
  these families.
* No claim that the in-run verdicts are causal evidence; they are a runtime decision rule,
  and the audit shows they can be wrong in both directions.
* Intervals are conditional on the declared design (five training seeds, seven families,
  fixed repetitions); they are not a claim about all workloads.
