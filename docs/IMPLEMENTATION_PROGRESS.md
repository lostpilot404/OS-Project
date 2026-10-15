# Implementation progress

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic
Workloads**.  Status of every requested component, with the verification that backs it.
Full suite: **386 tests, all passing** (`python -m pytest -q`); 273 of those are the
preserved pre-rework tests.

| # | Requested component | Status | Where | Verification |
|---|---|---|---|---|
| 1 | Workload fingerprinting (process count, arrival distribution, burst variability, burst skew, priority distribution) | **Implemented** | `workload/fingerprint.py` (11 features), `rl/fingerprint_state.py` | `tests/test_fingerprint.py` (20): hand-computed values, determinism, order invariance |
| 1 | Information boundary documented and enforced | **Implemented** | `docs/METHODOLOGY.md` §1 | import-boundary + behavioural purity tests (same module) |
| 2 | Risk-aware objective (waiting, response, switch overhead) with normalisation and documented weights/risk criterion | **Implemented** | `rl/objective.py`, `config.RiskObjectiveConfig` | `tests/test_objective.py` (20): hand-computed J, clipping, zero-reference, tail-weight ranking flip |
| 2 | Comparison against the original objective (not a silent replacement) | **Implemented** | both objectives scored for every arm/workload in all artifacts | original preserved byte-equality test; ablation `ql_orig_fingerprint` vs `ql_risk`; `orig_*`/`risk_*` columns |
| 3 | Structured decision explanations with computed evidence | **Implemented** | `rl/explain.py`, `results/study/explanations/`, `explanations_examples.json` | `tests/test_explanations.py` (14): evidence recomputation, verdict agreement, no fabricated claims, fallback labelling |
| 4 | Counterfactual evaluation after the decision | **Implemented** | `evaluation/counterfactual.py` | `tests/test_counterfactual.py`: isolation spies, call-order pin, direct-computation match |
| 4 | Hindsight selection regret | **Implemented** | `RegretRecord`, both objectives | regret arithmetic + non-negativity tests; artifact consistency tests |
| 5 | Robustness: workload families, held-out workloads, multiple seeds, distribution shift, variability, poor-selection frequency, regret, standard metrics | **Implemented** | `results/study/robustness.csv`, `summary.json`, `docs/EVALUATION_RESULTS.md` §4 | split-disjointness tests; output-validation of rates and row accounting |
| 5 | Genuinely held-out test set | **Implemented** | test seed 19501, untouched at design freeze (`docs/METHODOLOGY.md` §5) | `tests/test_study_design.py` (families + fingerprints disjoint) |
| 6 | Baselines: original Q selector, risk-aware selector, non-learning selection baselines | **Implemented** | 4 Q arms + `best_fixed` / `fingerprint_rule` / `random` | legacy-equivalence pin; baseline provenance + rule determinism tests |
| 6 | Ablations isolating features and the risk objective | **Implemented** | `ql_risk` / `ql_risk_origstate` / `ql_orig_fingerprint` (2×2 minus pinned legacy arm) | ablation-matrix completeness test; reported paired CIs |
| 7 | Tests: leakage, objective, explanation, counterfactual isolation, regret, reproducibility, output validation | **Implemented** | `tests/test_fingerprint.py`, `test_objective.py`, `test_explanations.py`, `test_counterfactual.py`, `test_selector.py`, `test_study_design.py`, `test_reproducibility.py`, `test_output_validation.py` | full suite green; byte-identical reruns |
| 7 | Regenerate all experiments and artifacts after implementation | **Done** | `results/study/` (study), `results/` + `figures/` (legacy pipeline), `results/runtime/` (preserved simulator) | manifests and generated reports carry only post-rework values |
| 8 | Documentation: architecture, methodology, diagrams, progress, evaluation, limitations; title preserved; offline-not-runtime stated | **Done** | `README.md`, `docs/{ARCHITECTURE,METHODOLOGY,EVALUATION_RESULTS,DESIGN_AND_CHOICES,LIMITATIONS,AUDIT_AND_PLAN}.md`, `docs/APPENDIX_RUNTIME_SIMULATOR.md` | values on doc pages traced to `results/study/summary.json` |

## Sequence of reviewable commits

1. `docs` repository audit and implementation plan (`docs/AUDIT_AND_PLAN.md`).
2. `feat` workload fingerprinting + tested information boundary.
3. `feat` `risk_aware_v1` objective alongside the preserved original reward.
4. `feat` selector decision boundary, counterfactuals, regret, explanations.
5. `feat` study pipeline (arms, baselines, robustness, artifacts, CLI).
6. `test` study-design / reproducibility / output-validation suites.
7. `fix` visited-action masking and the 81-state mechanism subset (reward-scale
   artifact and state-sparsity fixes, documented in `docs/DESIGN_AND_CHOICES.md`).
8. `chore` regenerated study, legacy and runtime artifacts; documentation rewrite.

## Known deviations from the initial plan (all documented)

* The tabular state uses 4 of the 11 fingerprint features (the 5-feature/243-state
  pilot was under-trained at equal budget); all 11 remain in explanations.
* The new arms use visited-action masking; `ql_orig` preserves the legacy full-space
  argmax and is test-pinned.  This difference is declared in
  `docs/LIMITATIONS.md`.
* Training budget is 3600 episodes/arm/seed for every Q arm (legacy pipeline keeps
  1200); rationale in `docs/METHODOLOGY.md` §5.
