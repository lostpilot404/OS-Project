# Architecture

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic
Workloads** — implemented as an **offline, one-policy-per-workload policy selection and
evaluation system** (not a runtime OS scheduler).

## Module map

```text
workload/models.py            Process, Workload (canonical order, SHA-256 digest),
                              ScheduleResult + trace validation
workload/generator.py         Seeded synthetic families (8: 6 training + 2 held-out OOD)
workload/fingerprint.py       11 interpretable process-table-only features
                              (the selector's complete information set)

scheduler/{fcfs,sjf,round_robin,priority}.py, scheduler/base.py
                              The four fixed policies and shared simulation engine
evaluation/metrics.py         The six standard metrics (waiting, turnaround, response,
                              utilisation, throughput, context switches)
evaluation/counterfactual.py  POST-decision four-policy evaluation, both objectives,
                              hindsight selection regret

rl/q_learning.py              Tabular Q-learning agent (visits, masking, epsilon)
rl/state.py                   Original 4-feature state encoder (ablation + ql_orig)
rl/fingerprint_state.py       Fingerprint-subset state encoder (81 states)
rl/reward.py                  Preserved original six-metric reward
rl/objective.py               PolicyOutcome; original + risk_aware_v1 objectives,
                              normalisation and weights; both scored everywhere
rl/selector.py                WorkloadPolicySelector: select() is a pure decision
                              function; train_episode()/evaluate() compute
                              counterfactuals only after the action is fixed
rl/explain.py                 Structured evidence-backed explanations

experiments/study_config.py   Frozen design: seeds, splits, arms, OOD families
experiments/baselines.py      Predeclared rule, training-only best-fixed, random
experiments/study.py          Study runner: training, held-out evaluation, robustness
                              tables, generated report, artifacts
experiments/{train,evaluate,report,run_experiment}.py
                              Preserved legacy offline pipeline (main.py experiment)
experiments/runtime_*.py      Preserved simulator appendix (main.py runtime-experiment)

visualization/study_plots.py  Study figures, drawn from generated tables only
results/study/                Study artifacts (decisions, counterfactuals, robustness,
                              explanations, summary, report)
```

## Data flow

```text
                    ┌────────────────────────────────────────────────┐
                    │ declared process table (pid, arrival, burst,    │
                    │ priority) — the INFORMATION BOUNDARY            │
                    └───────────────┬────────────────────────────────┘
                                    │ pure functions only
                    ┌───────────────▼───────────────┐
                    │ workload.fingerprint (11)     │
                    │ + state encoder (4 → 81 bins) │
                    └───────────────┬───────────────┘
                                    │
   Q-table ──────────────►  rl.selector.select()  ──► SelectionDecision
   (learned values)            (no schedulers,           (policy, rule,
                                no metrics)               Q-values, visits)
                                    │
                        action fixed; only now:
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │ evaluation.counterfactual.run_counterfactuals │
                    │  FCFS │ SJF │ Round Robin │ Priority (same    │
                    │  workload; digests checked)                   │
                    └───────────────┬───────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
   rl.objective (both)     regret = cost − min cost   rl.explain
   = training reward       (both objectives)          (evidence-backed
   (after the action)                                reasons per selection)
```

## Invariants (enforced by tests)

1. `select()` cannot execute a policy or read counterfactual outcomes
   (`tests/test_counterfactual.py::TestSelectionIsolation`).
2. Fingerprint features import nothing from schedulers/metrics/RL and are pure
   functions of the process table (`tests/test_fingerprint.py`).
3. Counterfactuals are computed strictly after the action is fixed (same test module;
   training order pinned by a call-order test).
4. The original objective reproduces `rl.reward.compute_reward` exactly
   (`tests/test_objective.py`).
5. Every explanation claim re-derives from recorded evidence
   (`tests/test_explanations.py`).
6. Regret is `cost(selected) − min_a cost(a)` with non-negative values
   (`tests/test_counterfactual.py`).
7. Same seeds ⇒ byte-identical artifacts (`tests/test_reproducibility.py`).
8. Artifacts satisfy schema/consistency contracts
   (`tests/test_output_validation.py`).
9. `ql_orig` is decision-equivalent to the preserved legacy selector
   (`tests/test_study_design.py::TestLegacyPinning`); held-out families never enter
   training and splits are fingerprint-disjoint (same module).

## Preserved components

The legacy offline pipeline (`main.py experiment`, `main.py train`) and the
event-driven simulator (`main.py runtime-experiment`) remain runnable and tested.  The
simulator is documented in `docs/APPENDIX_RUNTIME_SIMULATOR.md`; it is not the project
system and no reported result comes from it.
