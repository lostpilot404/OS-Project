# Methodology

Official project title (unchanged):

> **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

## 0. What the system is

An **offline, one-policy-per-workload policy selection and evaluation system**.  For a
complete batch workload the system selects exactly one of the four conventional
policies — **FCFS, SJF (non-preemptive), Round Robin, Priority (non-preemptive)** —
using **tabular Q-learning** over interpretable workload characteristics.  The selected
policy then runs the entire workload.  **This is not a runtime OS scheduler**: there is
no within-schedule switching, no event loop, no kernel integration.  The preserved
event-driven simulator in `docs/APPENDIX_RUNTIME_SIMULATOR.md` is not part of the
system and contributes no reported result.

The system is *explainable* (every selection carries a structured, evidence-backed
explanation) and *risk-aware* (the declared training/evaluation objective penalises
tail waiting explicitly).  It is evaluated with post-decision counterfactuals,
hindsight regret, controlled ablations, non-learning baselines and a preserved
held-out test split.

## 1. Information boundary

The selector's decision function may use **only the declared process table** of the
batch workload: `(pid, arrival_time, burst_time, priority)` per process.  This is the
complete information set of the offline batch setting: the whole job list is known
before execution starts (the same assumption the textbook SJF/Priority policies make).

Explicitly **forbidden** as selector inputs:

* schedule outcomes, execution traces, waiting/response/turnaround times;
* metrics of any policy run, including the four fixed policies;
* counterfactual results ("what would have happened under another policy");
* rewards, objective values and regret of any kind;
* anything derived from the above.

The module `workload/fingerprint.py` implements the boundary: it imports nothing from
the schedulers, metrics, RL stack or experiment layer, and every feature is a pure
function of the process table.  Tests enforce both the import boundary and behavioural
purity (`tests/test_fingerprint.py`).  The decision function `WorkloadPolicySelector.
select` receives no policies and cannot execute any schedule
(`tests/test_counterfactual.py::TestSelectionIsolation`).

Counterfactual outcomes are computed **only after an action is fixed** — as the
training reward, or as post-hoc evaluation evidence.  They never re-enter the decision
function.  This asymmetry is the system's core causal structure.

## 2. Workload fingerprinting

`workload/fingerprint.py` extracts eleven interpretable characteristics from the
process table (exact definitions in the module docstring):

| Feature | Definition | Role |
|---|---|---|
| `process_count` | number of processes | workload size |
| `arrival_span` | max − min arrival time | arrival spread |
| `arrival_rate` | `n / max(1, span)` | arrival intensity |
| `arrival_gap_cv` | pop-std/mean of successive sorted-arrival gaps | arrival-distribution shape |
| `burst_mean` | mean burst | scale |
| `burst_cv` | pop-std/mean of bursts | burst variability |
| `burst_skew` | Bowley quantile skew of bursts | burst asymmetry |
| `first_arrival_burst_ratio` | mean burst of earliest arrivals / median burst | head-of-line blocking risk |
| `priority_std` | pop-std of priorities | priority spread |
| `priority_unique_fraction` | distinct priorities / n | priority granularity |
| `priority_burst_correlation` | Pearson(priority, burst) | whether priority order tracks short-job order |

All features are deterministic, order-invariant and reproducible; the workload digest
(SHA-256 of the process table) proves pairing across policies and splits.

**Tabular state.** The risk-aware selector discretises a declared subset of four
features × 3 bins = **81 states**: `burst_cv`, `first_arrival_burst_ratio`,
`arrival_gap_cv`, `priority_burst_correlation`.  Each maps to one selection mechanism:
short-job ordering (SJF), head-of-line blocking (Round Robin) and priority ordering
(Priority).  The remaining features stay in the fingerprint for explanations.  Four
features keep the tabular space small enough for the declared training budget to cover
(a 243-state pilot was visibly under-trained; see `docs/LIMITATIONS.md`).

## 3. Objectives

Both objectives are implemented side by side (`rl/objective.py`) and every arm is
scored under **both** in every artifact.  The new objective never silently replaces
the original.

### 3.1 Original objective (preserved)

The six-metric reward of `rl/reward.py`, unchanged: per metric, the chosen-policy value
over the arithmetic mean of the four fixed policies on the same workload (clipped
ratios), combined with weights 0.30 waiting / 0.25 turnaround / 0.20 response / 0.10
switches-per-process / 0.075 utilisation / 0.075 throughput.  Higher reward is better;
equivalent cost = −reward.  Tests pin byte-equality with `rl.reward.compute_reward`.

### 3.2 Risk-aware objective `risk_aware_v1`

Lower-is-better cost

```
J = 0.35·n(mean waiting) + 0.25·n(mean response)
  + 0.15·n(context switches per process) + 0.25·n(p95 per-process waiting)
```

with normalisation `n(x) = clip(x / reference, 0, 2)`, `reference` = the four-policy
arithmetic mean on the same workload (zero reference ⇒ `n = 1`).  The Q-learning
training signal is `−J`.

* **Normalisation rationale.** Per-workload ratios make J scale-free across families
  with different time scales; the clip bounds single-metric blow-ups; a policy that
  matches the four-policy mean on every term scores exactly `J = 1`.
* **Risk criterion.** The fourth term is an explicit **tail-risk penalty**: the
  nearest-rank 95th percentile of per-process waiting times (effectively the maximum
  at n≈15).  Weight `w_tail = 0.25` makes the objective mean–tail and risk-averse
  toward waiting-time outliers; `w_switch = 0.15` prices context-switch overhead at a
  declared unit cost per switch.
* **Weights** are declared in `config.RiskObjectiveConfig`, sum to one, and are not
  tuned on evaluation data.  `tests/test_objective.py` pins the arithmetic
  hand-computed on synthetic outcomes, including a case where raising `w_tail` flips a
  ranking.

## 4. Selection, counterfactuals, regret, explanations

* **Decision** (`rl/selector.py`): fingerprint → state → Q-values → one action.
  Greedy evaluation compares only actions with training visits
  (`QLearningAgent.greedy_visited_action`); zero-initialised unvisited entries are
  never treated as estimates — necessary because the `−J` signal is non-positive and
  would otherwise lose every comparison against the 0.0 initialisation.  Wholly unseen
  states use a documented lowest-index fallback, labelled `unseen_state_fallback` so
  explanations never present it as learned evidence.
* **Counterfactual evaluation** (`evaluation/counterfactual.py`): strictly after the
  decision, all four fixed policies run the identical workload (fingerprints checked),
  both objectives are scored for each, and **hindsight selection regret** =
  `cost(selected) − min_a cost(a)` is recorded under both objectives.
* **Explanations** (`rl/explain.py`): one structured record per selection with the
  workload characteristics (values + state bins), the selection basis (Q-values and
  visits; or the baseline rule evidence), the objective values of all four policies,
  regret, and a list of reasons whose `evidence` fields are the exact computed numbers.
  Statements are deterministic templates over the evidence; a reason is emitted only
  when its claim is true (e.g. "lowest normalised term" appears only for strictly
  lowest terms).  `tests/test_explanations.py` re-derives the evidence independently
  and checks the verdicts, including the case where the selector did *not* pick the
  hindsight best.

## 5. Study design

Frozen in `experiments/study_config.py`; the study name is
`risk_aware_offline_policy_selection`.

* **Training workloads**: on the fly from the six training families, seed stream
  `derive_seed(seed, 0, episode)` (the preserved legacy stream).
* **Q arms** (equal budget: 5 seeds × 3600 one-step episodes each):

  | Arm | State | Objective | Selection rule |
  |---|---|---|---|
  | `ql_orig` | original 4-feature | original | preserved full-space argmax |
  | `ql_risk` | fingerprint 4-feature | `risk_aware_v1` | visited-action masking |
  | `ql_risk_origstate` | original 4-feature | `risk_aware_v1` | visited-action masking |
  | `ql_orig_fingerprint` | fingerprint 4-feature | original | visited-action masking |

  `ql_orig` reproduces the preserved legacy selector exactly (pinned by
  `tests/test_study_design.py::TestLegacyPinning`).  The three new arms form the
  controlled 2×2 ablation of (features × objective); they share the selection rule so
  only the two factors vary.
* **Training budget.** Episodes are one-step terminal updates (one sample per episode).
  The legacy default of 1200 episodes leaves the 324 state-action cells with under four
  samples each and the per-state greedy policy is then estimation noise; every Q arm
  therefore receives the same 3600-episode budget (≈11 samples/cell).  The legacy
  pipeline keeps its historical 1200-episode default.
* **Non-learning baselines** (`experiments/baselines.py`, all predeclared):
  `best_fixed` (lowest mean training cost — training data only),
  `fingerprint_rule` (deterministic decision list over fingerprint features, fixed in
  code), `random` (seeded uniform).  The hindsight oracle is used only for regret and
  is never presented as a selector.
* **Splits.** Validation (seed 5501) is for examples/reporting only.  The **held-out
  test split** (seed 19501, 8 families × 30 workloads) is evaluated once, at the end.
  Two families never appear in training: `poisson_arrivals` (preserved) and
  `heavy_tail_stress` (25 processes, 90% short bursts, slow Poisson arrivals,
  skewed priorities) — the distribution-shift conditions.  Train/validation/test
  fingerprints are checked disjoint and duplicates are rejected.
* **Design-freeze protocol.** Two implementation bugs (a zero-initialisation/reward-sign
  artifact and a missing head-of-line state feature) and the underpowered legacy
  training budget were identified and fixed on mechanism grounds while developing
  against test-seed 9501.  At design freeze the test seed was rotated to **19501**,
  which no development run had ever generated; every number in
  `docs/EVALUATION_RESULTS.md` comes from that untouched split.  Development runs
  under 9501 are not reported as results.

## 6. Robustness and comparison protocol

`results/study/robustness.csv` and `summary.json` report, per arm × split × family ×
training seed: mean/sd declared-objective cost, mean/sd regret, poor-selection rate
(regret > 0), severe rate (regret > 0.1), hindsight-best rate, mean original reward and
all six standard scheduling metrics of the selected policies.  Distribution shift is
reported as in-distribution vs out-of-distribution means and poor rates.  Paired
arm-vs-arm comparisons use one difference per unique workload (model seeds averaged
first) with a percentile bootstrap 95% CI (2000 resamples, fixed seed 424242).
Performance variability across training seeds is reported as mean ± sd of per-seed
mean cost.

## 7. Reproducibility and verification

Every random draw (workloads, agents, baselines, bootstrap) is seeded through
`config.derive_seed`; gzip artifacts carry fixed headers.  Two runs with the same
configuration are byte-identical (`tests/test_reproducibility.py`).  Output validation
checks schemas, row accounting, regret consistency, recomputes the objective cost from
the recorded terms, matches selected-policy metrics against the counterfactual table
and verifies explanation coverage for every selection
(`tests/test_output_validation.py`).  Information leakage, counterfactual isolation and
explanation correctness have their own suites; the full suite is 386 tests.
