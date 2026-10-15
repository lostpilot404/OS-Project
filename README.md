# Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads

**Project system: a self-healing CPU scheduler.** A discrete-event CPU simulator runs a
reproducible workload, monitors what the scheduler actually achieves, declares a degradation
event when the realized waiting time exceeds the trace's own stable behaviour, diagnoses the
likely cause from observable evidence, asks a recovery controller (a predeclared rule or a
tabular Q-learning agent) for a recovery action, applies it, verifies the outcome over a
declared observation window, and then **accepts, rolls back, or falls back** to a safe
configuration — recording every decision, its evidence, its cost and its outcome.

Everything in this repository is a *simulation*. Nothing here claims to be a real operating
system kernel change; see [Limitations](#limitations-and-honest-findings).

---

## 1. The problem

A fixed scheduling policy is a guess made before the workload is known: FCFS is fair but
slow for short jobs, SJF is fast but needs burst knowledge, Round Robin protects
responsiveness but pays for every switch. A *dynamic* workload changes character while it
runs, so the interesting question is not "which policy is best on average" but **"can a
scheduler notice that it is doing badly right now, decide what to change, and prove that the
change helped?"**

This repository answers that question in two layers:

| Layer | What it does | Status |
|---|---|---|
| **Self-healing scheduler** (`healing/`, `experiments/healing_*`, `main.py healing-study`) | monitor → detect → diagnose → decide → apply → verify → accept/rollback/fallback, with a causal observation boundary and a counterfactual recovery audit | **primary system** |
| **Offline policy-selection study** (`rl/adaptive.py`, `experiments/study.py`, `main.py study`) | chooses one policy per complete workload from workload features before execution | preserved comparison/historical workflow |
| **Runtime simulator appendix** (`scheduler/runtime.py`, `main.py runtime-experiment`) | the event-driven engine and causal Q-learning controller without the healing loop | preserved historical appendix; the engine is reused by the healing system |

## 2. Self-healing scheduler vs. offline policy selector

* The **offline selector** sees a whole workload description (arrival pattern, burst
  distribution, priorities) up front, picks one of four policies *before* execution, and is
  evaluated on end-of-workload cost and regret. It never reacts, never verifies and cannot
  undo a bad choice.
* The **self-healing scheduler** sees only the *current* ready set of arrived processes and
  the *past* outcomes of completed work. It may change the configuration mid-trace, it must
  verify the change with evidence it can observe, and it must undo the change when the
  evidence does not support keeping it. It is allowed to be wrong, but it is not allowed to
  keep a change it cannot justify.

## 3. Architecture and workflow

```
                    ┌──────────────────────────── one trace (one workload) ───────────────────────────┐
 arrivals ──► ready queue ──► dispatch decision ──► CPU executes slice ──► completion events
                    ▲                    ▲                                        │
                    │                    │                                        ▼
                    │            ┌───────┴────────┐                     ┌──────────────────────┐
                    │            │  self-healing  │  observation        │  causal monitor       │
                    │            │  controller    │◄────────────────────┤  - rolling wait stats │
                    │            │  (healing/)    │  (arrived work only)│  - stable reference   │
                    │            └───────┬────────┘                     │  - load accrual rate  │
                    │                    │                              └──────────┬───────────┘
                    │            ┌───────▼────────┐                                │
                    │            │ detector       │◄───────────────────────────────┘
                    │            │ threshold + hysteresis + cooldown + gates
                    │            └───────┬────────┘
                    │            ┌───────▼────────┐   evidence-backed cause
                    │            │ diagnoser      │   (head-of-line blocking, fairness starvation,
                    │            └───────┬────────┘    priority anomaly, tail dispersion, congestion)
                    │            ┌───────▼────────┐
                    └────────────┤ recovery policy│   rule (predeclared) | tabular Q-learning | hold-only
                                 └───────┬────────┘
                                         │ one of 8 actions (hold, set FCFS/SJF/RR/Priority,
                                         │ shrink/grow quantum, revert to last accepted)
                                 ┌───────▼────────┐   verification window
                                 │ verifier       │   normalised load closure vs. declared threshold
                                 └───────┬────────┘
                                         ▼
                          accept │ roll back │ safe fallback (after repeated failures)
```

All ten stages are implemented and exercised end to end in `healing/controller.py`:

1. **generate** a reproducible workload (`workload/generator.py`, seeds derived per split);
2. **simulate** the schedule (`scheduler/runtime.py`, event-driven, one evolving FIFO queue);
3. **monitor** realized performance from completed work only (`healing/monitor.py`);
4. **detect** degradation against a declared reference with a threshold, hysteresis,
   cooldown and gates (`healing/detector.py`);
5. **diagnose** the cause from the visible ready set (`healing/diagnosis.py`);
6. **decide** with the recovery policy (`healing/policies.py`);
7. **apply** the action — a configuration change or an explicit hold (`healing/actions.py`);
8. **verify** over a declared observation window (`healing/controller.py`);
9. **accept / roll back / fall back**, with a hard budget on interventions per trace;
10. **record** decision, evidence, cost and outcome in an incident ledger, and audit it with
    a counterfactual replay (`experiments/healing_experiment.py`).

### What the controller observes and what it controls

*It observes* (never the workload, never future arrivals, never the finished schedule):
the ready set with arrival times, remaining bursts and priorities; the completion notices of
processes finished during the last decision interval; its own past decisions and the
configuration in force.

*It controls* one configuration: the dispatch policy (FCFS, SJF, Round Robin, Priority) and
the Round-Robin quantum.

## 4. Degradation, recovery, verification, rollback

| Stage | Declared mechanism | Where |
|---|---|---|
| Degradation event | rolling mean waiting time over the last 8 completions exceeds `reference + 1.25 × band` for 2 consecutive evaluations, outside a 3-decision cooldown, with ≥2 ready processes, enough visible work, and the queue at least as congested as the trace's normal level | `healing/detector.py`, `healing/config.py` |
| Reference / band | EWMA of normal windows, with floors so the band cannot collapse | `healing/monitor.py` |
| Diagnosis | observable ready-set evidence with a fixed precedence order | `healing/diagnosis.py` |
| Recovery actions | `hold`, `set_fcfs`, `set_sjf`, `set_round_robin`, `set_priority`, `shrink_quantum`, `grow_quantum`, `revert_to_last_accepted`; actions that cannot change anything are masked out before the policy is asked | `healing/actions.py`, `healing/controller.py` |
| Verification | the window closes after 5 decisions **and** 3 completions, or after 14 decisions; the verdict is taken on the **phase-adjusted residual** of the normalised load closure `(trigger_load − post_load) / max(trigger_load − reference_load, load_band)`, i.e. the closure minus the value the declared phase model predicts with no intervention; that prediction is a linear function of the trigger's own observables (queue level, ready-set size, `log1p(visible work)`) | `healing/verifier.py`, `healing/controller.py`, `experiments/healing_thresholds.py` |
| Verdicts | **`accepted`** (cleared the 95th percentile of the control residual, +0.30), **`rejected`** (below the 5th percentile, −0.30: a *confirmed* failure), **`inconclusive`** (neither, or too little telemetry: rolled back like a failure but never counted as one), **`held`** (nothing was changed) | `healing/verifier.py`, `healing/config.py` |
| Reward (learning) | the *same* phase-adjusted residual minus the declared accept boundary, minus the intervention cost (1 % of the reference waiting time per change), minus the rollback penalty only on a **confirmed** failure — inconclusive windows cost the action and nothing else | `healing/policies.py`, `healing/controller.py` |
| Safety | after 2 consecutive **confirmed** failures the loop stops adapting, pins the last accepted configuration and records the fallback | `healing/controller.py` |

Thresholds are not hand-waved: `experiments/healing_thresholds.py` measures the no-action
control on the **training** split and derives the constants (see
`results/healing/healing_threshold_derivation.csv`).

## 5. Install and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock      # numpy, pandas, matplotlib, pytest

# primary system: full self-healing study (training, held-out evaluation, audit, figures)
.venv/bin/python main.py healing-study

# preserved historical workflows
.venv/bin/python main.py study               # offline policy-selection study
.venv/bin/python main.py runtime-experiment  # runtime simulator appendix (no healing)
.venv/bin/python main.py experiment          # legacy offline pipeline
.venv/bin/python main.py train               # legacy offline training only
```

Useful flags: `--results-dir DIR`, `--figures-dir DIR`, `--no-figures`, `--no-probe`.
Everything is written under `results/healing/` and `figures/healing/` by default.

Tests:

```bash
.venv/bin/python -m pytest -q          # 472 passed
```

## 6. Reproduction

```bash
# full study into an isolated directory (creates its own tables, figures and report)
.venv/bin/python main.py healing-study --results-dir /tmp/healing-results --figures-dir /tmp/healing-figures

# the threshold derivation alone (prints the derivation, uses training/validation only)
.venv/bin/python -m experiments.healing_thresholds
```

Design is frozen in `experiments/healing_config.py`: training seeds `7101–7105`
(1200 traces each, six families, `poisson_arrivals` held out of training), validation seed
`8201` (20 traces per family), **held-out final-test seed `23819`** (30 traces per family),
1000 bootstrap replicates with seed `104729`. Splits are disjoint by construction
(`derive_seed(master_seed, SPLIT_TAG, family_index, repetition)`), so no arm can ever see a
trace another arm was evaluated on.

## 7. Key results (held-out final-test seed 23819, 210 traces per arm)

Mean waiting time (simulated time units); lower is better. Learned arms are averaged over
their 5 independent training seeds (1050 traces).

| Arm | Mean waiting | P95 waiting | Mean response | Incidents / trace | Interventions / trace |
|---|---:|---:|---:|---:|---:|
| `fixed_round_robin` | 217.05 | 310.41 | 26.50 | – | – |
| `fixed_fcfs` | 130.84 | 258.42 | 130.84 | – | – |
| `fixed_priority` | 128.32 | 262.75 | 128.32 | – | – |
| `detector_only` (control) | 130.84 | 258.42 | 130.84 | 0.65 | 0.00 |
| `rule_fairness` | 136.78 | 263.72 | 110.40 | 0.71 | 0.71 |
| `q_recovery` (5 seeds) | 122.56 | 251.98 | 122.54 | 0.67 | 0.65 |
| `rule_wait` | **122.39** | 251.88 | 122.39 | 0.67 | 0.67 |
| `rule_wait_from_sjf` | 94.87 | 249.54 | 94.87 | 0.82 | 0.02 |
| `q_recovery_from_sjf` (5 seeds) | 94.66 | 249.20 | 94.66 | 0.82 | 0.00 |
| `fixed_sjf` | **94.66** | **249.20** | 94.66 | – | – |

Paired, family-stratified bootstrap over identical workloads (1000 replicates):

* `rule_wait` − `detector_only` = **−8.45** wait units, 95 % CI [−9.44, −7.35] → the
  recovery loop's changes do help relative to doing nothing.
* `q_recovery` − `fixed_fcfs` = **−8.27**, CI [−9.31, −7.20] → the learned controller also
  improves on its own starting configuration.
* `rule_wait` − `fixed_sjf` = **+27.73**, CI [+26.05, +29.72]; `q_recovery` − `fixed_sjf` =
  **+27.91**, CI [+26.11, +29.87] → **neither recovery arm reaches the fixed SJF baseline.**
* `q_recovery` − `rule_wait` = **+0.18**, CI [−0.04, +0.48] → the learned controller is
  statistically indistinguishable from the predeclared rule.
* `q_recovery_from_sjf` − `fixed_sjf` = **0.0000** (CI [0, 0]): started from the strongest
  fixed configuration the loop intervenes **zero** times and reproduces SJF exactly.

Counterfactual replay audit of every intervention (the honest causal check: replay the same
trace with the recorded decisions but *hold* at the incident, so the difference is the
marginal value of adapting):

| Arm | Audited interventions | Mean gain (wait units) | Helped | False recoveries |
|---|---:|---:|---:|---:|
| `rule_fairness` | 150 | −130.84 | 23.3 % | 76.7 % |
| `rule_wait` | 141 | +188.79 | 94.3 % | 5.7 % |
| `q_recovery` | 678 | +192.07 | 95.0 % | 5.0 % |
| `rule_wait_from_sjf` | 4 | −169.25 | 0.0 % | 100.0 % |

Verification on the same split (65 accepted, 34 rejected interventions): **every accepted
intervention was genuinely helpful** (65 of 65, mean gain +184.9 wait units), 76.5 % of the
rejected ones were harmful against a base rate of 14.0 %, and the phase-adjusted residual
ranks helpful above harmful with AUC **0.951** (raw closure 0.613). What it still cannot do is
*certify*: 91.0 % (`rule_wait`) and 92.1 % (`q_recovery`) of the genuinely helpful
interventions end `inconclusive` and are rolled back after the window. The full evidence,
including the comparison of phase-model specifications, is in
`docs/EVALUATION_RESULTS.md`.

## 8. Limitations and honest findings

* **The self-healing loop does not beat fixed SJF on these workloads.** It beats its own
  naive starting point (FCFS) and it does not damage SJF when started from it, but the
  ordering `SJF < rule_wait ≈ q_recovery < FCFS < rule_fairness` holds on the held-out split.
* **The detector's false-trigger rate is measurable and not zero.** With no intervention at
  all the control declares 0.62 events per trace on the training split and 0.66 on the
  held-out split; with a controlled burst shock injected into validation traces, recall is
  75.0 % at a median delay of 6 decision intervals, while 0.70 events per stable trace are
  still declared. A burst-backlog trace is at its worst at the beginning, and the monitor
  cannot tell "this trace is hard" from "this configuration is bad".
* **Verification is honest, and honesty has a price.** The verdict is taken on the
  phase-adjusted residual of the closure statistic: `accepted` above +0.30, `rejected` below
  −0.30, `inconclusive` in between (both boundaries are percentiles of the no-action control,
  false-accept 3.4 %, false-reject 3.1 %). Only **6.7 %** of the genuinely helpful
  interventions clear the declared null boundary at all, because the best action moves the
  statistic by about 0.95 of the residual spread that one window shows; 91.0–92.1 % of helpful
  interventions therefore end `inconclusive` and are rolled back after the window. The filter
  works in the tails: **65 of 65 accepted interventions helped** (base rate 83.9 %) and 76.5 %
  of the rejected ones were harmful (base rate 14.0 %).
* **The learned controller is barely trained relative to its state space**: 29–33 of 648
  states are visited in 1200 training traces per seed — 3.8–7.2 % of the pairs that are
  reachable at all from those contexts and 1.1–1.3 % of the declared table, and the coverage
  curve is flat after ~900 episodes (`healing_coverage.csv`).
* **The reward is a weak proxy for the size of an effect and a decent one for its sign**:
  pooled Pearson correlation with the counterfactually measured per-process effect is 0.10,
  but the within-incident rank correlation is 0.622 (median 0.875) and its argmax is the truly
  best action in 15 of 21 probed incidents.
* Everything is measured in a **synthetic discrete-event simulator** with switch costs and
  observable burst times; no real kernel, no real hardware, no real workload traces.
  The simulator grants the scheduler burst knowledge, which a real OS does not have.
* Full detail, including the metric definitions and every caveat, is in
  [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) and
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## 9. Documentation map

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | modules, data flow, causal boundary |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | design, splits, seeds, metrics, statistics, protocol |
| [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) | every number quoted above, traceable to a file |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | simulator scope, assumptions, unverified ideas |
| [`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md) | decisions, rejected alternatives, derivations |
| [`docs/TRANSFORMATION_AUDIT.md`](docs/TRANSFORMATION_AUDIT.md) | what changed in this finalization and what was claimed before |
| [`docs/FINALIZATION_CHECKLIST.md`](docs/FINALIZATION_CHECKLIST.md) | Complete / Needs Fixing / Blocked / Not Verified |
| [`docs/IMPLEMENTATION_PROGRESS.md`](docs/IMPLEMENTATION_PROGRESS.md) | module-by-module status |
| [`docs/APPENDIX_RUNTIME_SIMULATOR.md`](docs/APPENDIX_RUNTIME_SIMULATOR.md) | the preserved runtime appendix |
| [`docs/AUDIT_AND_PLAN.md`](docs/AUDIT_AND_PLAN.md) | historical plan (superseded) |

## 10. Artifacts

```
results/healing/    healing_metrics_{validation,final_test}.csv   per-trace metrics of every arm
                    healing_incidents_{validation,final_test}.csv per-incident ledger (evidence, action, verdict)
                    healing_episodes_{validation,final_test}.csv  per-trace healing summary
                    healing_audit.csv                             counterfactual replay audit of interventions
                    healing_comparisons.csv                       paired bootstrap comparisons
                    healing_detection_summary.csv                 detections, interventions, verdicts per arm
                    healing_threshold_derivation.csv              derived vs declared thresholds
                    healing_control_incidents.csv                 the no-action control's incident ledger
                    healing_action_probe.csv                      recorded reward vs measured true effect
                    healing_coverage.csv                          Q-table coverage per training seed
                    healing_coverage_curve.csv                    coverage against training length (does more training still buy new states?)
                    healing_q_seed_variability.csv                per-seed spread of the learned arms
                    healing_detection_scenarios.csv               recall/delay/false triggers on controlled injected degradations
                    healing_training_episodes.csv                 training episodes per seed
                    healing_demo.json                             one worked detection→recovery trace
                    healing_report.md, healing_summary.json       generated report and machine-readable summary
figures/healing/    healing_waiting_time.png                      mean/p95 waiting by arm
                    healing_recovery_outcomes.png                 verdicts and audit outcomes per arm
                    healing_action_effects.png                    recorded reward vs measured effect per action
```

20 machine-readable artifacts and 3 figures are written by the default run; an independent
re-run into a separate directory reproduces all 23 byte for byte.
