# Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads

An **offline, one-policy-per-workload CPU scheduling policy selection and evaluation
system**.  For a complete batch workload the system extracts interpretable workload
characteristics and uses **tabular Q-learning** to select one of **FCFS, SJF, Round
Robin, or Priority** for the whole workload before execution.  Every selection is
accompanied by a structured, evidence-backed explanation, evaluated with post-decision
counterfactuals and hindsight regret, and compared against the original Q-learning
objective, controlled ablations, and predeclared non-learning baselines on a preserved
held-out test split.

> **This is not a runtime OS scheduler.**  One decision is made per workload; the
> selected policy runs the entire schedule.  There is no within-schedule switching and
> no kernel integration.  (The repository also preserves an event-driven runtime
> simulator as an appendix — see below — which is not the project system and
> contributes no result on this page.)

## How it works

```text
declared process table (pid, arrival, burst, priority)     <- information boundary
        │
        ▼
workload fingerprint (11 interpretable features: process count, arrival
distribution, burst variability/skew, head-of-line ratio, priority distribution)
        │
        ▼
tabular state (4 mechanism features × 3 bins = 81 states)
        │
        ▼
Q-learning selector ──► ONE policy for the whole workload   (decision-time evidence:
        │                 FCFS / SJF / Round Robin / Priority   features + Q-values)
        │
        ▼  (strictly after the decision)
counterfactual evaluation of all four policies + regret + structured explanation
```

The selector cannot execute policies or read outcomes (`select()` is a pure decision
function); counterfactual results enter only afterwards, as the training reward or as
post-hoc evidence.  The declared risk-aware objective is

```
J = 0.35·n(mean waiting) + 0.25·n(mean response)
  + 0.15·n(context switches/process) + 0.25·n(p95 waiting)     (lower is better)
```

where `n(x)` is the clipped ratio to the four-policy mean on the same workload; the
p95 term is the explicit risk criterion.  The **original** six-metric objective is
preserved unchanged and scored alongside `J` everywhere — see
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Held-out test results (240 workloads, seed 19501, evaluated once)

Full analysis: [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) · generated
report: [`results/study/report.md`](results/study/report.md).

| Arm | Mean cost J | Mean regret | Hindsight-best % | Poor % | Mean wait | Mean response |
|---|---:|---:|---:|---:|---:|---:|
| Risk-aware Q selector (fingerprint state + `risk_aware_v1`) | 0.7953 | 0.0574 | 82.5 | 17.5 | 81.79 | **76.32** |
| Original Q selector (preserved) | 0.7963 | 0.0524 | **85.6** | **14.4** | **78.20** | 77.83 |
| Ablation: original objective + fingerprint state | **0.7838** | **0.0405** | 84.5 | 15.5 | 79.36 | 78.39 |
| Ablation: risk objective + original state | 0.8253 | 0.0874 | 76.4 | 23.6 | 80.94 | 77.90 |
| Predeclared fingerprint rule | 0.7752 | 0.0372 | 83.7 | 16.3 | 79.19 | 78.47 |
| Best fixed policy (SJF, training-only) | 0.8419 | 0.1040 | 77.7 | 22.3 | 78.17 | 78.17 |
| Random (seeded) | 0.9778 | 0.2398 | 26.4 | 73.6 | 110.39 | 75.66 |

Measured, honest findings (details in `docs/EVALUATION_RESULTS.md`):

* **Fingerprint features clearly help**: −0.0293 cost over the original state under the
  risk objective (95% CI [−0.0467, −0.0120]), and far better behaviour out of
  distribution.
* **The risk-aware vs original objective comparison is a tie with nuance** at the
  frozen budget (Δ = +0.0000 vs the preserved selector; +0.0120 vs the original
  objective within the same state) while buying the best response time and explicit
  tail protection.  Both objectives are scored for every arm in every artifact.
* **The predeclared rule wins on these families** — a genuine result about how
  structured the synthetic families are, not a performance claim for learning.
* Two families are never trained on (`poisson_arrivals`, `heavy_tail_stress`);
  distribution-shift numbers are reported separately and the test seed was rotated at
  design freeze so the final split is genuinely held out.

## Decision explanations and counterfactuals

Every selection in the study carries a structured explanation: workload
characteristics with state bins, the selection basis (Q-values and visits, or the rule
branch), both objective values for all four policies, hindsight regret, and only those
comparison claims the computed evidence supports — no invented justifications.  Three
fully worked examples are checked in at
[`results/study/explanations_examples.json`](results/study/explanations_examples.json);
the full per-selection logs are regenerable from the recorded seeds.

## Installation and commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q            # 386 tests
python main.py study           # full study: train, held-out test, artifacts, report
```

Other commands (all preserved and tested):

```bash
python main.py experiment          # preserved legacy offline pipeline
python main.py train               # preserved legacy offline training only
python main.py runtime-experiment  # preserved event-driven simulator appendix
python main.py study --no-figures --results-dir /tmp/study   # isolated regeneration
```

Every run is seeded and deterministic: two runs with the same configuration are
byte-identical (verified by tests).  Artifacts land in `results/study/` (decisions,
counterfactuals, robustness tables, explanations, `summary.json`, generated
`report.md`) and `figures/study/`.

## Documentation

| Document | Contents |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | information boundary, fingerprint definitions, objectives and weights, risk criterion, splits, design-freeze protocol |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module map, data flow diagram, tested invariants |
| [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md) | all measured results and their interpretation |
| [`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md) | implementation choices and rationale |
| [`docs/IMPLEMENTATION_PROGRESS.md`](docs/IMPLEMENTATION_PROGRESS.md) | what is implemented and how it is verified |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | what the system does not show |
| [`docs/AUDIT_AND_PLAN.md`](docs/AUDIT_AND_PLAN.md) | pre-rework repository audit and the plan this work follows |
| [`docs/APPENDIX_RUNTIME_SIMULATOR.md`](docs/APPENDIX_RUNTIME_SIMULATOR.md) | preserved event-driven simulator (appendix, not the project system) |

## Preserved components

The four fixed policies, the workload model/generator, the metrics module, the original
offline selector (`rl/adaptive.py`, `rl/state.py`, `rl/reward.py`) with its legacy
pipeline (`main.py experiment`), and the event-driven runtime simulator with its own
test suite are all preserved and runnable.  The legacy artifacts in `results/` and the
simulator artifacts in `results/runtime/` were regenerated after the rework and belong
to those preserved components; the project's research system is the offline
policy-selection study above.
