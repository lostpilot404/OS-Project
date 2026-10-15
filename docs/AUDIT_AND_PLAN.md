# Repository audit and implementation plan (historical)

> **Historical document, superseded.** It records the state of the repository at commit
> `cfdafa2` and the plan that produced the *offline policy-selection* system. The project
> system is now the **self-healing scheduler**; see `README.md`,
> `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md` and `docs/EVALUATION_RESULTS.md`.
> Nothing below describes the current primary architecture.

Official project title (preserved throughout the rework):

> **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

This document is the audit of the repository state at commit `cfdafa2` and the plan for
reworking it into an **offline, one-policy-per-workload policy selection and evaluation
system** (FCFS / SJF / Round Robin / Priority, tabular Q-learning). It is a historical
record of the plan; implemented behaviour is documented in `docs/METHODOLOGY.md`,
`docs/ARCHITECTURE.md`, `docs/EVALUATION_RESULTS.md` and `docs/LIMITATIONS.md`.

---

## 1. Audit — what exists and works

Test baseline before any change: **273 passed** (`python -m pytest -q`).

### 1.1 Preserved, working functionality

| Area | Modules | Status |
|---|---|---|
| Four fixed policies | `scheduler/{fcfs,sjf,round_robin,priority}.py`, `scheduler/base.py` | Working, well tested (hand-built + seeded traces) |
| Process / workload / schedule model | `workload/models.py` | Working; canonical `(arrival_time, pid)` order, SHA-256 workload fingerprint, trace validation |
| Seeded workload generator | `workload/generator.py`, `config.WorkloadFamilyConfig` | Working; 7 families (6 training + 1 held-out `poisson_arrivals`) |
| Six standard metrics | `evaluation/metrics.py` | Working; waiting / turnaround / response / utilisation / throughput / context switches |
| Tabular Q-learning agent | `rl/q_learning.py` | Working; epsilon schedule, visit counts, terminal updates |
| **Original offline selector** | `rl/adaptive.py`, `rl/state.py`, `rl/reward.py`, `experiments/{train,evaluate,report,run_experiment}.py` | Working; one decision per workload from 4 pre-execution features, reward = 6-metric ratio vs four-policy mean |
| Event-driven runtime simulator | `scheduler/runtime.py`, `rl/runtime_*`, `experiments/runtime_*` | Working (own test suite) but **out of scope**: this project is not a runtime scheduler |
| CLI | `main.py` | Working (`runtime-experiment`, `experiment`, `train`) |
| Artifacts | `results/`, `figures/` | Present from earlier runs; **stale relative to this rework** and must be regenerated, not reused |

### 1.2 Gaps against the requested research system

1. **Workload fingerprinting** — `rl/state.py` has only 4 features (median burst, burst
   CV, offered load, priority std). Missing: process count, arrival-distribution
   descriptors, burst skew, explicit priority-distribution descriptors. No standalone
   documented information boundary.
2. **Risk-aware objective** — only the original 6-metric reward exists. No tail/risk
   criterion, no documented weight rationale, no head-to-head comparison of objectives.
3. **Decision explanations** — none. The legacy `PolicySelectionDecision` records data
   but produces no structured, evidence-backed explanation.
4. **Counterfactual evaluation** — the legacy selector computes all four reference runs
   inside `_run` and folds them into the decision object; there is no post-decision
   counterfactual phase, no regret calculation, and no enforced isolation between
   selection and counterfactual outcomes.
5. **Robustness analysis** — 5 training seeds exist, but there is no variability /
   poor-selection / regret reporting, no distribution-shift families beyond one held-out
   family, and no explicit preserved held-out test split separate from the ad-hoc
   evaluation set.
6. **Baselines / ablations** — the four fixed policies are compared, but there are no
   non-learning selection baselines and no controlled ablations isolating features vs
   objective.
7. **Verification** — no tests for information leakage of fingerprint features,
   objective arithmetic, explanation correctness, counterfactual isolation, regret,
   end-to-end reproducibility, or artifact schema validation.
8. **Documentation** — README presents the project as a runtime-adaptive scheduler; the
   requested system is an offline policy-selection and evaluation system. The official
   title above must be preserved.

---

## 2. Design decisions

### 2.1 System boundary

* **This is an offline policy-selection and evaluation system, not a runtime OS
  scheduler.** One decision per complete workload; the selected policy runs the whole
  workload; no switching at runtime.
* The event-driven runtime simulator stays in the repository as a **preserved appendix**
  with its tests; it is not part of the research system and no result in the main
  documentation comes from it. The CLI keeps `runtime-experiment` working.

### 2.2 Information boundary (selector input)

The offline selector may use **only the declared process table** of the batch workload:
`(pid, arrival_time, burst_time, priority)` per process. It must never use schedule
outcomes, metrics, counterfactual results, rewards, or anything derived from executing a
policy. Every fingerprint feature is a pure function of this table. The boundary is
documented in `docs/METHODOLOGY.md` and enforced by tests.

### 2.3 Workload fingerprinting (`workload/fingerprint.py`)

Interpretable features, all computed from the process table only:

| Feature | Definition | Motivation |
|---|---|---|
| `process_count` | number of processes | workload size |
| `arrival_span` | `max(arrival) - min(arrival)` | arrival spread |
| `arrival_rate` | `n / max(1, arrival_span)` | offered arrival intensity |
| `arrival_gap_cv` | std/mean of successive sorted-arrival gaps (0 if all gaps are 0) | arrival-distribution shape |
| `burst_mean` | mean burst | scale |
| `burst_cv` | std/mean of bursts | burst variability |
| `burst_skew` | Bowley (quantile) skew of bursts: `(p90 + p10 - 2*p50)/(p90 - p10)`, 0 if degenerate | burst skew / tail asymmetry |
| `priority_std` | population std of priorities | priority spread |
| `priority_unique_fraction` | distinct priorities / n | priority-distribution granularity |
| `priority_burst_correlation` | Pearson correlation of (priority, burst); 0 if degenerate | whether priority order tracks short-job order |

The tabular state discretises a declared subset of these into bins (mixed-radix index);
all features remain available for explanations.

### 2.4 Objectives (`rl/objective.py`)

Both objectives are computed **after** an action is fixed (training reward, or post-hoc
evaluation). Both are reported side by side everywhere — the new objective never
silently replaces the original.

* **Original objective** — preserved unchanged: 6-metric ratio vs the four-policy
  arithmetic mean (`rl/reward.py`); cost = `-reward`.
* **Risk-aware objective `risk_aware_v1`** — lower-is-better cost
  `J = w_wait·n(mean waiting) + w_resp·n(mean response) + w_switch·n(context switches per
  process) + w_tail·n(p95 per-process waiting)`,
  where `n(x) = clip(x / ref_x, 0, 2)` and `ref_x` is the four-policy arithmetic mean on
  the same workload (zero reference ⇒ `n = 1`).
  Default weights `w_wait=0.35, w_resp=0.25, w_switch=0.15, w_tail=0.25` (sum to 1).
  **Risk criterion:** an explicit tail term — the 95th percentile of per-process waiting
  time (nearest-rank) — with weight `w_tail`; the objective is therefore mean–tail and
  risk-averse with respect to worst-case waiting. Normalisation makes J scale-free
  across families; the four policies serve only as a normalisation reference and are
  never a selection input.

### 2.5 Selection, counterfactuals, regret, explanations

* `rl/selector.py` — `WorkloadPolicySelector.select(workload)` computes fingerprint →
  state → greedy/epsilon action from Q-values **only**; it cannot execute policies.
* `evaluation/counterfactual.py` — `run_counterfactuals(...)` runs all four fixed
  policies on the same workload **after** the decision, records metrics + both
  objectives per policy, and computes hindsight regret
  `regret = cost(selected) - min_a cost(a)` under the declared objective.
* `rl/explain.py` — structured explanation per evaluation selection: workload
  characteristics (values + bins), selected policy, decision-time Q-value evidence,
  post-hoc objective values for all four policies, regret, and a list of **reasons whose
  evidence fields are recomputable** from these numbers (no free-form invented
  justifications).

### 2.6 Study design (`experiments/study.py`)

* **Arms (Q-learning):** `ql_original` (preserved legacy selector: original 4-feature
  state + original objective), `ql_risk` (fingerprint state + risk objective),
  plus 2×2 ablations `ql_risk_origstate`, `ql_orig_fingerprint`.
* **Non-learning baselines:** `best_fixed` (best single policy chosen on **training**
  workloads only), `fingerprint_rule` (predeclared deterministic decision list on
  fingerprint features), `random` (seeded uniform). The hindsight oracle is reported
  only as a labelled upper bound for regret, never as a selector.
* **Splits:** training families (6) for Q-training; a **preserved held-out test set**
  (fixed master seed, 8 families × repetitions) evaluated only at the end; held-out
  families `poisson_arrivals` and `heavy_tail_stress` (new, out-of-distribution) never
  appear in training. Train/validation/test fingerprints are checked disjoint.
* **Robustness reporting:** per family / split / training seed — mean and sd of J and
  regret, poor-selection frequency (regret > 0 and regret > 0.1), standard scheduling
  metrics for every arm, both objectives for every arm.

### 2.7 Verification plan

New test modules: `test_fingerprint.py` (feature values, purity, information boundary),
`test_objective.py` (hand-computed J, normalisation/clipping, original preserved),
`test_explanations.py` (evidence recomputation, no fabricated claims),
`test_counterfactual.py` (selection isolation, regret), `test_reproducibility.py`
(identical seeds ⇒ identical outputs), `test_output_validation.py` (artifact schema),
`test_study_design.py` (split disjointness, baseline provenance, ablation matrix).
All 273 pre-existing tests must keep passing.

---

## 3. Implementation sequence (reviewable commits)

1. **Audit & plan** — this document.
2. **Fingerprinting** — `workload/fingerprint.py` + tests.
3. **Objectives** — `rl/objective.py` + `RiskObjectiveConfig` + tests (original reward
   preserved byte-for-byte).
4. **Selector / explanations / counterfactuals** — `rl/selector.py`, `rl/explain.py`,
   `evaluation/counterfactual.py` + tests.
5. **Study pipeline** — `experiments/study_config.py`, `experiments/baselines.py`,
   `experiments/study.py`, `main.py study` + tests.
6. **Verification suites** — leakage, reproducibility, artifact validation.
7. **Regenerate artifacts** — full test suite, then re-run the study and the preserved
   legacy pipeline; fresh numbers only.
8. **Documentation** — README (official title; explicit “offline policy-selection and
  evaluation system, not a runtime OS scheduler”), `docs/ARCHITECTURE.md`,
   `docs/METHODOLOGY.md`, `docs/EVALUATION_RESULTS.md`,
   `docs/IMPLEMENTATION_PROGRESS.md`, `docs/LIMITATIONS.md`, runtime simulator moved to
   `docs/APPENDIX_RUNTIME_SIMULATOR.md`.

Honesty rules for step 7–8: no invented gains, no stale metrics presented as new,
completion claimed only for what the test suite and regenerated artifacts verify.
