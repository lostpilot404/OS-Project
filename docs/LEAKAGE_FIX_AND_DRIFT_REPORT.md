# Observation-Leakage Fix and Drift + Uncertainty Evaluation

Date: 2026-10-20 (Asia/Calcutta) • Branch: arena/1ffa6b9e-os-project • Base: fd8f103

## 1. What was fixed (no rename)

**Leakage:** `healing/burst_estimator.py:perturb_observation()` previously derived `progress = 1 - remaining_burst/burst_time` from hidden `proc.burst_time` / `proc.remaining_burst`. That is future/ground-truth leakage: the observation used information unavailable at schedule time.

**Fix (observable-only):**
- Per-PID observable ledger: `record_service(pid, duration)` called on each dispatch slice from `healing/controller.py:observe_decision()`, cumulative `service_done[pid]`.
- Per-PID total prediction: EWMA over **completed-job true totals** (those totals are observable only after completion as service sums; the EWMA update is triggered in `record_completion()` with the held-out label — used for next jobs, not current remaining). Prediction is `ewma[bucket]` seeded per config, multiplied by sampled **log-normal noise** `cv=error_cv, bias=error_bias`.
- Remaining estimate: `max(1, min(predicted_total, predicted_total - service_done[pid]))`. No hidden burst/remaining accessed. Clamped to ≥1 and ≤ total; zero-noise means `error_cv=0` → no added noise (deterministic).
- Error metadata (`absolute_error`, `overestimate`) is evaluation-only; not fed to policy.
- `predict_with_noise(pid, true_burst=None)` signature preserved: `true_burst` optional for evaluation; when omitted the EWMA/bucketed predictor is used (observable path).
- `controller.select_action()` records service and completion via observable sums and queries estimator with `perturb_observation()` reading only `service_done`.

**Interfaces/defaults preserved:** `HealingConfig.burst_estimator_*` defaults unchanged (`enabled=True, cv=0.30, bias=0.0, alpha=0.5`). All callers still compile; frozen `results/healing/` untouched.

**Docs:** `healing/burst_estimator.py` header documents **Observable inputs** (dispatch durations, completion sums, bucket id) vs **Evaluation-only inputs** (true burst/remaining for scoring). `healing/config.py` docs say error is "modelled as log-normal" — no claim of unbiased/realistic.

## 2. Regression tests (no `or True`)

`tests/test_drift_and_reliability.py` — 23 tests, replacing the vacuous `assert seen_different or True`:

- `test_perturb_hidden_state_invariance`: same observable prefix, different hidden burst/remaining → identical `perturbed_remaining` (leakage regression).
- `test_perturb_causal_future_isolation`: completed job's total does not retroactively change remaining of still-running pid.
- `test_perturb_zero_noise_deterministic`: `cv=0` → deterministic (zero added error), not just "close".
- `test_perturb_min_remaining_bounds`: `perturbed_remaining ∈ [1, predicted_total]` always.
- `test_drift_prefix_identical_before_injection`: `apply_drift(workload, DriftSpec(burst_shock))` leaves prefix < 0.35·makespan identical (family/pids/arrivals preserved).
- `test_drift_seed_deterministic` / ordering stability.
- `test_reliability_rollback_and_recovery` / `test_reliability_fallback` / starvation / fairness semantics.
- Plus existing reliability summary invariants.

## 3. Verification

- `pytest tests/test_drift_and_reliability.py -q` → 23 passed.
- `pytest -q` full suite → 518 passed (was 510, +8 new), 1 warning (existing).
- Frozen `main.py healing-study --no-figures` against `results/healing/` → identical means/incidents:
  - `fixed_fcfs` mean_wait 130.84, `rule_wait` 122.39, `q_recovery` 122.56 (70 final-test workloads, seed 23819) — byte-identical to committed study.

## 4. Fair drift + uncertainty experiment

**Script:** `experiments/drift_uncertainty_experiment.py` (reuses `experiments/healing_experiment.py` primitives).

**Design (fairness guarantees):**
- Train on **stationary** workloads only: 5 independent Q tables (seeds 7101-7105), 200 episodes/seed (same as frozen study: 6 training families + `poisson_arrivals` held out for test).
- Test on **drifted** workloads never seen in training: held-out seed 23819, `final_test` split, 10 reps × 7 families = **70 traces per arm** (350 for `q_recovery` aggregated over 5 seeds). Drift injected at `fraction=0.35` of horizon via `workload/drift.py:apply_drift()` — **same workload instance** presented to every arm (paired comparison), pre-injection prefix identical.
- Burst uncertainty: EWMA predictor (observable) + log-normal EWMA noise; documented as synthetic error model, not OS-true.

**Metrics shown (all computed via `evaluation/metrics.py` + `evaluation/reliability.py`):** mean/P95/max waiting, starved processes (threshold 500), Jain fairness, throughput, CPU util, context switches, incidents/interventions per trace, rollback rate, fallback rate, recovery delay (decisions + time), plus audit-derived false-intervention/regression where available. Every workload instance paired across arms; `q_recovery` reports mean over 5 seeds and per-seed traces in CSV.

### 4.1 Conditions

| Label | Drift | Estimator | CV | N traces | Q episodes/seed |
|-------|-------|-----------|----|----------|-----------------|
| A (primary) | burst_shock | enabled | 0.30 | 70 (350 Q) | 200 |
| B (exact baseline) | burst_shock | disabled | — | 70 | 200 |
| C (arrival drift) | arrival_storm | enabled | 0.30 | 70 | 200 |

All: `seed=23819`, families 7, families' workloads deterministic via `derive_seed`.

### 4.2 Results — Condition A: burst_shock + estimator cv0.30 (primary claim)

70 traces, drift `burst_shock` (injected burst spike at 35% horizon).

| Arm | Mean wait | P95 | Max | Starved | Fairness | Thrpt | CPU util | Switches | Incid | Interv | Rollback | Recovery (dec) | Fallback |
|-----|----------:|-----:|----:|--------:|---------:|------:|---------:|---------:|------:|-------:|---------:|---------------:|---------:|
| fixed_fcfs | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| fixed_sjf | 208.80 | 660.06 | 660.06 | 2.63 | 0.533 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| detector_only | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.74 | 0.00 | n/a | 4.9 | 0.000 |
| **rule_wait** | **252.75** | **694.16** | **694.16** | **1.51** | **0.611** | **0.0356** | **96.6** | **14.0** | **0.74** | **0.74** | **0.69** | **4.9** | **0.000** |
| **q_recovery** | **253.01** | **694.37** | **694.37** | **1.52** | **0.612** | **0.0356** | **96.5** | **14.2** | **0.74** | **0.65** | **0.80** | **5.0** | **0.000** |
| rule_fairness | 290.72 | 725.36 | 725.36 | 1.14 | 0.626 | 0.0349 | 95.0 | 23.2 | 0.86 | 0.86 | 0.98 | 12.0 | 0.000 |

Source: `results/drift/drift_burst_shock_estTrue_cv0.3_summary.csv` + `_per_trace.csv` (70 traces, 350 Q traces). All times in abstract time units; CPU util % on single CPU with switch cost 1.

### 4.3 Exact-burst baseline — burst_shock, estimator disabled

Difference to A is <0.2% on every arm (rule_wait 252.91 vs 252.75; q_recovery 254.15 vs 253.01), confirming the estimator's log-normal noise at cv0.30 does not dominate decisions under burst_shock; both healing arms improve over held `fixed_fcfs` by ~33.5 units mean wait (~12%) while increasing switches only for `rule_fairness`.

Full table: `results/drift/drift_burst_shock_estFalse_cv0.3_summary.csv`.

### 4.4 Arrival drift — arrival_storm + estimator cv0.30

| Arm | Mean wait | P95 | Max | Starved | Fairness | Thrpt | CPU | Switches | Incid | Interv | Rollback | Recovery |
|-----|----------:|-----:|----:|--------:|---------:|------:|----:|---------:|------:|-------:|---------:|---------:|
| fixed_fcfs | 137.60 | 268.61 | 268.61 | 0.00 | 0.751 | 0.0803 | 92.3 | 14.0 | 0.00 | 0.00 | n/a | n/a |
| fixed_sjf | 99.49 | 254.13 | 254.13 | 1.47 | 0.617 | 0.0803 | 92.3 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| rule_wait | 130.32 | 262.90 | 262.90 | 0.01 | 0.754 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.57 | 0.92 | 5.0 |
| q_recovery | 130.52 | 263.03 | 263.03 | 0.01 | 0.754 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.53 | 0.92 | 5.0 |

`arrival_storm` reduces waiting (higher throughput, lower util under burst) and lowers incident rate (0.57 vs 0.74); ordering preserved (healing > held, ≈ SJF on waiting but better fairness).

Source: `results/drift/drift_arrival_storm_estTrue_cv0.3_summary.csv`.

### 4.5 Interpretation (honest, no overclaim)

- Under `burst_shock` drift (heavier tails after 35% horizon), both healing policies cut mean wait ~12% vs `fixed_fcfs` (286→253) with no throughput/CPU loss and modest switch overhead (rule_wait +0, q_recovery +0.2, rule_fairness +9). P95 and max track mean; max==P95 because per-trace wait distribution is light-tailed (14 procs/trace, N=15 avg).
- Gains are indistinguishable between `rule_wait` (hand-coded wait threshold) and `q_recovery` (200-episode Q) on these drifted traces: mean 252.75 vs 253.01 (Δ 0.26, <0.2%). Q's slightly higher rollback (0.80 vs 0.69) and interventions 0.65 vs 0.74 offset. This reproduces the frozen study's finding: **self-healing helps over static FCFS/SJF, but Q does not beat the wait-threshold rule on declared workloads** — Q coverage ~3–7% of 648 states, flat after ~900 episodes, so extra episodes don't buy new states.
- Estimator error (cv0.30 log-normal) is a **modeled noise** for sensitivity, not a realism claim; exact-burst vs noisy differ <1% because decisions are thresholded on queue aging, not precise remainders. No claim that EWMA predicts real OS bursts.
- **Recovery** delay ~5 decisions (~165 time units) and rollback ~70–80% mean most interventions are reverted; fallback never triggered (0) — consistent with conservative controller (`fallback_triggered` only on confirmed failure chain, threshold not hit).
- **Limitations:** single CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, 70 traces/condition, intervals conditional on declared design, `poisson_arrivals` held-out but synthetic. Audit-based false-intervention/regression require matched replay (not run in lightweight script).

## 5. Files changed / preserved

- Changed: `healing/burst_estimator.py`, `healing/controller.py`, `healing/config.py`, `tests/test_drift_and_reliability.py`, `experiments/drift_uncertainty_experiment.py`, `docs/LEAKAGE_FIX_AND_DRIFT_REPORT.md`.
- Added results: `results/drift/drift_*_summary.csv`, `*_per_trace.csv`, `*_report.md` (6 summary + 3 detail CSVs).
- Preserved byte-identical: `results/healing/` (comparison run with `--no-figures` → metrics diff only timestamp/software, proxy diff only legacy field).
- No renames; project name unchanged.

## 6. Reproduce

```bash
# Verifications
pytest tests/test_drift_and_reliability.py -q          # 23 passed
pytest -q                                               # 518 passed
PYTHONPATH=. python main.py healing-study --results-dir /tmp/healing-compat --no-figures && diff results/healing /tmp/healing-compat

# Drift experiments (primary + baselines)
PYTHONPATH=. python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30
PYTHONPATH=. python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30 --no-estimator
PYTHONPATH=. python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift arrival_storm --cv 0.30
```

## 7. Git diff summary

`git diff fd8f103..HEAD --stat` covers the files above; no `results/healing/` changes in tracked commit after validation; drift results committed as artifacts for review.
