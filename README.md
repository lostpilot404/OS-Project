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
| Verification | the window closes after 5 decisions **and** 3 completions, or after 14 decisions; the verdict uses the **normalised load closure**, `(trigger_load − post_load) / max(trigger_load − reference_load, load_band)`, against a threshold derived from the no-action control | `healing/controller.py`, `experiments/healing_thresholds.py` |
| Reward (learning) | the same closure statistic minus the declared intervention cost (1 % of the reference waiting time per change, 2 % extra on rollback) — so holding beats a change that does not help | `healing/policies.py`, `healing/controller.py` |
| Safety | after 2 consecutive failed verifications the loop stops adapting, pins the last accepted configuration and records the fallback | `healing/controller.py` |

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
.venv/bin/python -m pytest -q          # 443 passed
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
`8201` (20 traces per family), **held-out final-test seed `20719`** (30 traces per family),
1000 bootstrap replicates with seed `104729`. Splits are disjoint by construction
(`derive_seed(master_seed, SPLIT_TAG, family_index, repetition)`), so no arm can ever see a
trace another arm was evaluated on.

## 7. Key results (held-out final-test seed 20719, 210 traces per arm)

Mean waiting time (simulated time units); lower is better. Learned arms are averaged over
their 5 independent training seeds (1050 traces).

| Arm | Mean waiting | P95 waiting | Mean response | Incidents / trace | Interventions / trace |
|---|---:|---:|---:|---:|---:|
| `fixed_round_robin` | 218.92 | 310.04 | 26.25 | – | – |
| `fixed_fcfs` | 128.88 | 256.01 | 128.88 | – | – |
| `fixed_priority` | 128.56 | 261.02 | 128.56 | – | – |
| `detector_only` (control) | 128.88 | 256.01 | 128.88 | 0.69 | 0.00 |
| `rule_fairness` | 135.60 | 263.22 | 107.29 | 0.76 | 0.76 |
| `q_recovery` (5 seeds) | 122.26 | 252.33 | 121.69 | 0.71 | 0.67 |
| `rule_wait` | **121.18** | 251.35 | 121.18 | 0.70 | 0.70 |
| `rule_wait_from_sjf` | 95.77 | 249.42 | 95.77 | 0.84 | 0.03 |
| `q_recovery_from_sjf` (5 seeds) | 95.53 | 248.96 | 95.53 | 0.84 | 0.00 |
| `fixed_sjf` | **95.53** | **248.96** | 95.53 | – | – |

Paired, family-stratified bootstrap over identical workloads (1000 replicates):

* `rule_wait` − `detector_only` = **−7.70** wait units, 95 % CI [−8.68, −6.68] → the
  recovery loop's changes do help relative to doing nothing.
* `q_recovery` − `fixed_fcfs` = **−6.62**, CI [−8.03, −5.12] → the learned controller also
  improves on its own starting configuration.
* `rule_wait` − `fixed_sjf` = **+25.65**, CI [+23.67, +27.71]; `q_recovery` − `fixed_sjf` =
  **+26.74**, CI [+24.46, +28.92] → **neither recovery arm reaches the fixed SJF baseline.**
* `q_recovery` − `rule_wait` = **+1.09**, CI [+0.12, +2.51] → the learned controller is
  marginally *worse* than the predeclared rule.
* `q_recovery_from_sjf` − `fixed_sjf` = **0.0000** (CI [0, 0]): started from the strongest
  fixed configuration the loop intervenes **zero** times and reproduces SJF exactly.

Counterfactual replay audit of every intervention (the honest causal check: replay the same
trace with the recorded decisions but *hold* at the incident, so the difference is the
marginal value of adapting):

| Arm | Audited interventions | Mean gain (wait units) | Helped | False recoveries |
|---|---:|---:|---:|---:|
| `rule_fairness` | 297 | −146.90 | 19.9 % | 80.1 % |
| `rule_wait` | 285 | +170.38 | 95.1 % | 4.9 % |
| `q_recovery` | 1354 | +153.88 | 88.2 % | 11.8 % |
| `rule_wait_from_sjf` | 12 | −126.83 | 0.0 % | 100.0 % |

## 8. Limitations and honest findings

* **The self-healing loop does not beat fixed SJF on these workloads.** It beats its own
  naive starting point (FCFS) and it does not damage SJF when started from it, but the
  ordering `SJF < rule_wait ≈ q_recovery < FCFS < rule_fairness` holds on the held-out split.
* **The detector's false-trigger rate is measurable and not zero.** With no intervention at
  all the control declares 0.62 events per trace on the training split and 0.69 on the
  held-out split: a burst-backlog trace is at its worst at the beginning, and the monitor
  cannot tell "this trace is hard" from "this configuration is bad".
* **The in-run verification is deliberately strict, and that costs real recoveries.**
  Its threshold is derived from the no-action control (95th percentile of the closure
  statistic) and keeps the false-accept rate at 4.5 % on the control, but it then rejects
  89.7 % (`q_recovery`) to 97.6 % (`rule_wait`) of interventions that the counterfactual
  audit shows *did* help. A trigger is a local maximum of the monitored statistic, so the
  statistic decays afterwards whatever the scheduler does; single-trace verification cannot
  separate the two, which is exactly why the audit exists.
* **The learned controller is barely trained relative to its state space**: 30–33 of 648
  states and 1.1 % of state-action pairs are visited in 1200 training traces per seed.
* **The reward is a weak proxy for the true effect**: Pearson correlation with the
  counterfactually measured per-process effect is 0.10 (argmax agreement 15/21 incidents),
  although it does rank the genuinely best action (`set_sjf`) above the worst
  (`set_round_robin`).
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
                    healing_training_episodes.csv                 training episodes per seed
                    healing_demo.json                             one worked detection→recovery trace
                    healing_report.md, healing_summary.json       generated report and machine-readable summary
figures/healing/    healing_waiting_time.png                      mean/p95 waiting by arm
                    healing_recovery_outcomes.png                 verdicts and audit outcomes per arm
                    healing_action_effects.png                    recorded reward vs measured effect per action
```
