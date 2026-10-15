# Design and implementation choices

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic
Workloads**.  This document records the implementation choices of the **offline
policy-selection and evaluation system** and why they were made.  The methodology
(weights, splits, protocols) is in `docs/METHODOLOGY.md`; the preserved event-driven
simulator is in `docs/APPENDIX_RUNTIME_SIMULATOR.md` and is not part of this system.

## System boundary

* One decision per complete workload: choose FCFS, SJF, Round Robin or Priority for
  the whole batch before execution.  No runtime switching — the project is explicitly
  **not** a runtime OS scheduler.
* Single CPU, integer time, no I/O, no blocking, no deadlines (`workload/models.py`).
  Context switches are counted; the configured switch cost is zero, so switch overhead
  enters the risk objective as a declared per-switch unit cost, not as simulated time.

## Information set

* The selector sees the declared process table only (see `docs/METHODOLOGY.md`,
  "Information boundary").  Exact burst lengths are assumed known at scheduling time —
  the same idealised assumption textbook SJF makes; this is a simulator convention, not
  a claim about burst prediction in real kernels.
* Counterfactual outcomes are computed after the action is fixed, as training reward or
  post-hoc evidence.  `select()` cannot run policies by construction.

## Fingerprinting

* Features are quantile/mean/std based, chosen to be hand-checkable and to map onto
  selection mechanisms (`workload/fingerprint.py`).  Quantiles use nearest-rank (the
  workload model's convention) so features stay on the integer burst grid.
* `first_arrival_burst_ratio` (mean burst of the earliest arrivals over the median
  burst) was added specifically because quantile skew cannot see a single extreme
  head-of-line job — the regime where Round Robin wins.

## Tabular state

* Four features × 3 bins = 81 states.  A 243-state pilot (five features) was visibly
  under-trained at equal episode budget and was replaced by the mechanism-complete
  four-feature subset.  The state is a declared subset; every fingerprint feature
  remains available to explanations.
* Linear binners with declared value ranges (`config.FingerprintStateConfig`); the
  middle bin of `first_arrival_burst_ratio` starts at 4.0, matching the predeclared
  rule's head-of-line threshold.

## Learning

* One-step terminal Q-learning per workload (the original design, preserved): reward
  is computed from the four-policy counterfactuals after the action is fixed; no
  discounting.
* Training signal for the risk arm is `−J` (higher is better).
* **Visited-action masking** for the new arms: greedy comparisons use only actions with
  training visits.  With non-positive `−J` rewards the zero initialisation of unvisited
  entries would otherwise win every comparison — an initialisation artifact, not
  learning.  Wholly unseen states fall back to the lowest action index and are labelled
  `unseen_state_fallback`.  The preserved `ql_orig` arm keeps the legacy full-space
  argmax (ties to FCFS) and is decision-pinned to `rl/adaptive.py` by test.
* Training budget 3600 episodes/arm/seed (≈11 samples per state-action cell); all Q
  arms equal.  The legacy pipeline keeps 1200.

## Objectives

* Original six-metric reward preserved byte-for-byte (`rl/reward.py`) and reported
  everywhere alongside `risk_aware_v1`.
* `risk_aware_v1` uses per-workload ratio normalisation (four-policy mean reference,
  clip 2.0) and an explicit p95-waiting tail term with weight 0.25.  The weights are
  declared, sum to one, and were not tuned on evaluation data.

## Evaluation

* Counterfactual evaluation runs all four fixed policies after every decision and
  computes hindsight regret under **both** objectives.
* Explanations are generated from computed evidence only; statement templates cannot
  emit a claim whose evidence is false (term-comparison reasons exist only for strict
  minima; fallbacks are labelled).
* The hindsight oracle appears only as regret's reference, never as a selector.

## Baselines and ablations

* Non-learning baselines are predeclared in code (rule thresholds fixed before any
  evaluation run); `best_fixed` reads training data only.
* The 2×2 ablation varies exactly two factors — state (original vs fingerprint) and
  objective (original vs risk) — under one shared selection rule; `ql_orig` is kept
  outside the 2×2 because preserving the original selector bit-for-bit is the point of
  that arm.

## Reproducibility

* Every draw is seeded via `config.derive_seed` (SHA-256 based; not Python `hash`).
* Explanation logs are gzip streams with fixed headers; two runs are byte-identical.
* Large explanation logs are gitignored (regenerable from seeds + CLI), consistent with
  the repository's convention for large decision logs; compact tables, reports and
  worked examples are tracked.

## Design-freeze protocol

Implementation bugs and the training budget were settled on mechanism grounds before
the final test seed (19501) was ever generated; see `docs/METHODOLOGY.md`,
"Design-freeze protocol".  Reported numbers come only from that untouched split.
