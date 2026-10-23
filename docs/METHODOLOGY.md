# Methodology

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

This document states exactly what is measured, on what data, with which statistics, and
which parts of the pipeline are causal. Every number quoted here is reproduced in
`docs/EVALUATION_RESULTS.md` and traceable to a file under `results/healing/`.

## 1. Object of study

The **self-healing CPU scheduler — reliable recovery under workload drift and uncertainty**:
a scheduling loop that (i) monitors realized performance from completed work only, (ii)
declares a degradation event against a declared reference with threshold, hysteresis and
cooldown, (iii) diagnoses a cause from the visible ready set, (iv) selects one of eight
recovery actions with a predeclared rule or a tabular Q-learning controller, (v) applies it,
(vi) verifies the outcome over a declared window, and (vii) accepts it, rolls it back, or
falls back to a pinned safe configuration — recording the decision, its evidence, its cost
and its outcome.

The research shift (Review 2) is from *can RL find a better policy?* to *can an adaptive
scheduler recognize when it is degrading, recover safely from workload changes, and avoid
harmful interventions under uncertainty?*  The existing architecture already supports that
question: the change is to measure reliability, not just mean waiting time, and to test the
loop on workload shifts and burst-prediction errors it was never trained on.

Three questions are being asked:

1. **Does the loop help relative to doing nothing?** (`rule_wait`/`q_recovery` vs
   `detector_only`, which runs the full monitoring/detection/diagnosis machinery but never
   intervenes.)
2. **Does the loop beat a fixed scheduler?** (`rule_wait`/`q_recovery` vs `fixed_sjf`,
   `fixed_fcfs`, `fixed_round_robin`, `fixed_priority`.)
3. **Does it recover reliably under drift and uncertainty?**  Measured by `evaluation/reliability.py`:
   recovery delay (decisions and time), false-intervention / rollback / regression rates,
   P95 waiting time, starvation count, Jain fairness index, context-switch overhead and
   CPU utilization — computed on workload shifts (`workload/drift.py`) and burst-estimator
   noise (`healing/burst_estimator.py`) that were held out of training, with paired
   bootstrap and counterfactual audit as the ground truth.

(1) and (2) are answered on the frozen held-out seed; (3) is additive — the same pipeline
with `HealingConfig(drift_enabled=True)` or `HealingConfig(burst_estimator_enabled=True)`
produces the reliability numbers without changing the committed artifacts.

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
| final test | `23819` | 30 per family (210) | the reported comparison; touched once, after the design was frozen |

Workload seeds come from `derive_seed(master_seed, SPLIT_TAGS[split], family_index,
repetition)`, so the three splits are disjoint **by construction** (verified by a test that
compares fingerprints) and every arm is evaluated on exactly the same instances.

`poisson_arrivals` is deliberately excluded from training: the learned controllers are
evaluated on an arrival process they never saw, so the study measures distribution shift
instead of fitting it.

**Seed hygiene.** `20501` was used by an early end-to-end smoke run before the detector's
congestion gate and the verification rule were finalised, and the `9401` stream was used
while the harness was built. `20719`, `21617` and `21977` were each declared as a held-out
seed, evaluated **once**, and then *examined*: after their runs the verification rule, the
reward and the audit were reworked (tri-state verification with a phase-aware residual, the
truncated-window rollback rule, the audit denominator fix, and the replacement of the free
closure fit by a physical observation row for the phase reference). Because substantive
changes were made after looking at them, all of them are retired as evidence and cannot be
used to report results again; the same is true of `72601`. The seed used for the reported
study, **`23819`**, was declared in `experiments/healing_config.py` *before* the final
evaluation of the frozen system and is refused as a development seed by the configuration's
own validation. The offline policy-selection study and the runtime appendix have their own,
separate seed streams.

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
computed from the finished schedule only, plus reliability overlays from
`evaluation/reliability.py` (same inputs, no new definitions of waiting time):

* `avg_waiting_time`, `median_waiting_time`, `p95_waiting_time`, `max_waiting_time` — a
  process's waiting time is `start_time − arrival_time` over the *whole* run, including every
  slice it was preempted (Round Robin therefore shows low response time and high waiting
  time);
* `avg_turnaround_time` = `completion_time − arrival_time`;
* `avg_response_time` = time to the first slice;
* `cpu_utilization` (percent) and `throughput` (completions per time unit);
* `context_switches` and `switching_overhead`;
* healing-specific: `incidents`, `interventions`, `accepted`, `rejected`, `inconclusive`,
  `held`, `confirmed_failures`, `fallback_triggered`, `final_policy`, `final_quantum`, plus
  per-incident evidence in the incident ledger;
* **reliability** (additive, from `evaluation/reliability.py` on `ScheduleResult` + ledger + audit):
  `p95_waiting_time`, `max_waiting_time`, starvation count (> 3× median wait),
  Jain's fairness index over waiting times, recovery delay (median decisions / time between
  detection and verification close), `false_intervention_rate` (audited interventions with
  `matched_gain ≤ 0`), `rollback_rate` (`(rejected+inconclusive)/interventions`),
  `regression_rate` (share of traces worse than `fixed_fcfs`), `fallback_rate`,
  `cpu_utilization` and `switching_overhead` — reported alongside mean waiting time because
  a mean alone hides tail and safety behaviour, and because the reliability question is
  evaluated on workload shifts the learner never saw.

The three verdicts are distinct and must never be collapsed into "not accepted":

| Verdict | Meaning | Effect on the configuration | Effect on learning |
|---|---|---|---|
| `accepted` | the window cleared the declared accept boundary: the change is certified as an improvement | kept, and becomes the new last-accepted configuration | reward on the same statistic; failure counter reset |
| `rejected` | the window fell below the declared reject boundary: the change is a **confirmed failure** | rolled back to the last accepted configuration | reward minus the declared rollback penalty; failure counter incremented (two confirmed failures in a row disable adaptation and take the safe fallback) |
| `inconclusive` | neither boundary was met: the evidence is too weak, or the window was too short (thin telemetry or a trace-end truncation) | rolled back as well (`inconclusive_rollback = true`: the loop never keeps a configuration it could not verify) | **no** rollback penalty and **no** failure count — ambiguity must not teach the controller that an action was harmful, and must not be able to trigger the fallback |
| `held` | the controller chose not to change anything (`changed = false`) | nothing to roll back | scored on the same statistic, so that holding competes with acting |

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
* **Counterfactual audit** (the causal evidence). Two counterfactuals are reported, with
  different denominators, and no rate ever mixes them:
  * **matched replay** (`matched_gain`, one row per *intervention*): the same controller is
    re-run on the same workload with the same decision prefix, forcing exactly that incident
    to hold, while every later decision is taken as usual. The prefix identity is verified by
    `divergence_decision_index`, which must equal the audited incident's decision index; the
    gain is `matched_hold_total_waiting − actual_total_waiting`. The **false-recovery rate** is
    the share of *audited interventions* whose gain is ≤ 0 (zero-effect interventions count as
    not helping). This is the denominator used by every rate in `healing_audit.csv` and in the
    per-arm summary.
  * **never-heal replay** (`hold_forever_gain` / `verdict = episode_level`, one row per
    *episode*): the recorded configuration prefix with no further healing after that episode's
    first incident. It answers a different question ("what if the loop had stopped here?") and
    is therefore reported in its own table.
  * Interventions are never exposed to the online controller: both replays run *after* the
    study, from recorded decisions.
* **Detection statistics.** The control's incident ledger gives the detector's false-trigger
  rate (events declared while nothing changed), reported per trace. On top of it,
  `experiments/healing_detection.py` injects two controlled step changes (a burst shock that
  multiplies the CPU time of every process arriving after the injection instant, and an
  arrival storm that compresses the remaining arrivals) into real validation traces, keeping
  the pids, priorities and the pre-injection prefix intact, and measures recall and detection
  delay against the injection instant on the hold-only control arm. The share of detections
  that fall on an injected trace is reported as a **precision proxy** (not a precision: the
  injection is a workload property, and a stable trace may legitimately degrade).

## 6. Derived thresholds (not tuned by hand)

`experiments/healing_thresholds.py` runs the no-action control over the training split and
derives:

* `min_load_factor`: the congestion gate. The sweep in
  `healing_threshold_derivation.csv` shows control detections per trace falling from 1.52
  (gate off) to 1.05 (`0.8`), 0.86 (`1.0`) and 0.04 (`1.25`); `1.0` is the strictest value
  that still leaves the detector active, and it is the declared value.
* `accept_closure_factor`: the *unconditional* share of the closure gap the window must close
  (declared 2.4, i.e. more than the whole gap, as the null distribution sits above 1). It is
  reported for continuity and for the derivation table, not used for the verdict.
* `post_load_baseline`: the phase model. A least-squares fit on the training control predicts
  the **no-action post-window load** from three physically observed quantities that are known
  at the trigger (the trigger's own queue level, the ready-set size, `log1p` of the visible
  remaining work), and the expectation of the closure statistic is then formed with the
  statistic's own denominator. It explains 96.1 % of the post-load variance and removes 80.2 %
  of the null closure's spread (residual sd 0.160 vs 0.806 raw), which is what makes the
  residual comparable across traces. The comparison of candidate specifications — including the
  free closure fit this replaced — is written to `results/healing/healing_verification_models.csv`
  by the run itself, together with the *paired* uncertainty of the decision-relevant metric
  (2000-draw cluster bootstrap over workload traces, fixed seed, reference = the declared
  specification). That comparison is what makes "no richer reference" a measurement rather than
  an assertion: on the 700 judged validation interventions the two-load-geometry reference and
  the earlier free closure fit certify **fewer** helpful interventions (−0.084 [−0.169, −0.005]
  and −0.084 [−0.168, −0.004]), a ready-count-only reference (+0.021 [−0.021, +0.065]) and
  clock-augmented references (+0.000) include zero, and a monitor-window-augmented reference
  certifies more (+0.043 [+0.010, +0.087]) but is **not** adopted: it conditions the expectation
  on the monitor's own window maturity instead of the workload phase, and the declared reference
  already certifies 15.0 % of helpful validation windows while certifying none of the 117
  harmful ones.
* `accept_residual_factor` / `reject_residual_factor`: the two operative boundaries, declared
  at the **95th and 5th percentiles of the same control residual** (measured +0.263 / −0.257 →
  declared +0.30 / −0.30, i.e. rounded outward to the declared step of 0.05). The measured
  false-accept / false-reject rates are 3.4 % / 3.1 % on the training control and 2.1 % / 1.1 %
  out of sample on the 95 judgeable control windows of the validation split (of its 98 declared
  events) — the split that took no part in the fit.

The verification rule is therefore a *conservative safety gate*, not a claim of benefit; the
audit supplies the benefit measurement, and `healing_summary.json` reports the measured
discrimination of the statistic itself (`verification_discrimination`).

## 7. Protocol actually followed

1. Build the healing package against the existing, tested event engine; pin its semantics
   with tests (identical schedules under a no-op control, rollback timing, fallback,
   counterfactual prefix equality).
2. Develop thresholds, gates and the reward shape on the **training** and **validation**
   splits only. Every rejected alternative is recorded in `docs/DESIGN_AND_CHOICES.md`
   together with the measurement that rejected it.
3. Freeze `experiments/healing_config.py`, declare a new held-out seed (`23819`, with
   `20501`, `20719`, `21617`, `72601` and `21977` recorded as retired), and run
   `python main.py healing-study` once.
4. Report every arm — including the arms that lose — and both the favourable and the
   unfavourable audit outcomes.

## 8. What is *not* claimed

* No real-OS integration: the system drives a simulator.
* No guarantee that recovery beats a fixed policy; the measured answer is that it does not, on
  these families.
* No claim that the in-run verdicts are causal evidence; they are a runtime decision rule,
  and the audit measures where it stands: on the reported split **65 of 65** accepted
  interventions were genuinely helpful and 26 of 34 rejected ones were harmful, but the audit
  also records 8 interventions that were rejected although they helped.
* No claim that an inconclusive verdict means "no effect": it means the declared evidence was
  insufficient. Measured on the held-out split, 91.0–92.1 % of the genuinely helpful
  interventions end inconclusive.
* Intervals are conditional on the declared design (five training seeds, seven families,
  fixed repetitions); they are not a claim about all workloads.
