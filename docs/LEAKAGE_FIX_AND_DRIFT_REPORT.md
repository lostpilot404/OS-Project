# Observation-Leakage Fix and Drift + Uncertainty Evaluation

Date: 2026-10-20 (Asia/Calcutta) • Branch: arena/1ffa6b9e-os-project • Base: fd8f103 • Commit: e85034c + enhanced

## 1. What was fixed (no rename)

**Leakage:** `healing/burst_estimator.py:perturb_observation()` previously derived `progress = 1 - remaining_burst/burst_time` from hidden `proc.burst_time` / `proc.remaining_burst`. That is future/ground-truth leakage: the observation used information unavailable at schedule time.

**Fix (observable-only):**
- Per-PID observable ledger: `record_service(pid, duration)` called on each dispatch slice from `healing/controller.py:observe_decision()`, cumulative `service_done[pid]`.
- Per-PID total prediction: EWMA over **completed-job true totals** (those totals are observable only after completion as service sums; the EWMA update is triggered in `record_completion()` with the held-out label — used for next jobs, not current remaining). Prediction is `ewma[bucket]` seeded per config, multiplied by sampled **log-normal noise** `exp(N(bias, sigma²))` where `sigma = error_cv` is the log-space standard deviation (name `error_cv` retained for backward compatibility; not the multiplier's CV `sqrt(exp(sigma²)-1)` ≈ 0.307 for sigma=0.30), `bias = error_bias`.
- Remaining estimate: `max(1, min(predicted_total, predicted_total - service_done[pid]))`. No hidden burst/remaining accessed. Clamped to ≥1 and ≤ total; zero-noise means `error_cv=0` → no added noise (deterministic, not unbiased).
- Error metadata (`absolute_error`, `overestimate`) is evaluation-only via `EstimationError`; not fed to policy.
- `predict_with_noise(pid, true_burst=None)` signature preserved: `true_burst` optional for evaluation; when omitted the EWMA/bucketed predictor is used (observable path).
- `controller.select_action()` records service and completion via observable sums and queries estimator with `perturb_observation()` reading only `service_done`.

**Interfaces/defaults preserved:** `HealingConfig.burst_estimator_*` defaults unchanged (`enabled=False` frozen, `cv=0.30, bias=0.0, alpha=0.5`). All callers still compile; frozen `results/healing/` metrics byte-identical (summary gains 6 additive config fields, expected).

**Docs:** `healing/burst_estimator.py` header documents **Observable inputs** (dispatch durations, completion sums, bucket id) vs **Evaluation-only inputs** (true burst/remaining for scoring). `healing/config.py` docs say error is "modelled as log-normal" — no claim of unbiased/realistic. No indirect leakage via monitor/detector/verifier (they use completed waiting times and load, not burst).

## 2. Regression tests (no `or True`)

`tests/test_drift_and_reliability.py` — 23 tests, replacing vacuous `assert seen_different or True`:

- `test_perturb_hidden_state_invariance`: same observable prefix, different hidden burst/remaining → identical perturbed remaining (leakage regression).
- `test_perturb_causal_future_isolation`: completed job's total does not retroactively change remaining of still-running pid.
- `test_perturb_zero_noise_deterministic`: `cv=0` → deterministic (zero added error), not ground truth.
- `test_perturb_min_remaining_bounds`: `perturbed_remaining ∈ [1, predicted_total]` always, including over-predict and exact cases.
- `test_drift_prefix_identical_before_injection`: `apply_drift(workload, DriftSpec(burst_shock))` leaves prefix <0.35·makespan identical (family/pids/arrivals preserved).
- `test_drift_seed_deterministic` / `test_drift_injection_preserves_pid_count_and_order` / `test_causal_estimation_future_does_not_affect_past`.
- `test_reliability_rollback_and_recovery` / `test_reliability_fallback` / `test_reliability_metrics_from_schedule` / fairness/starvation semantics.
- Plus workload generator, drift spec validation, healing config flags.

**Result:** `pytest tests/test_drift_and_reliability.py -q` → 23 passed; `pytest -q` full suite → 518 passed, 1 warning (existing matplotlib legend).

## 3. Verification

- Frozen `main.py healing-study --no-figures` against `results/healing/` → metrics identical:
  - `healing_metrics_final_test.csv` and `healing_comparisons.csv` byte-identical (0 diff lines)
  - `healing_summary.json` gains 6 additive fields (`burst_estimator_*`, `drift_*` defaults) — expected, metrics unchanged
  - `healing_action_probe.csv` legacy diff: 784-row canonical vs 168-row legacy proxy (documented in ENGINEERING_COMPLETION_REPORT § probe, not a baseline change)
- Reported means/incidents identical: `fixed_fcfs` 130.84, `rule_wait` 122.39, `q_recovery` 122.56, `detector_only` 130.84, `fixed_sjf` 94.66 (70 traces per arm on held-out seed 23819, 210 total per single-policy arm, 1050 for q_recovery).

## 4. Fair drift + uncertainty experiment

**Script:** `experiments/drift_uncertainty_experiment.py` (reuses `experiments/healing_experiment.py` primitives for training/workloads, `evaluation/paired.py` for CIs, `evaluation/reliability.py` for metrics).

**Design (fairness guarantees, audit trail):**
- Train on **stationary** workloads only: 5 independent Q tables (seeds 7101-7105), 200 episodes/seed cycling 6 families (`poisson_arrivals` held out), never sees drift.
- Test on **drifted** workloads never seen in training: held-out seed 23819, `final_test` split, 10 reps × 7 families = **70 traces per arm** (350 for `q_recovery` aggregated over 5 seeds). Drift injected at `fraction=0.35` via `workload/drift.py:apply_drift()` — **same workload instance** presented to every arm (paired by fingerprint), pre-injection prefix identical, deterministic per seed.
- Burst uncertainty: EWMA predictor (observable) + log-normal noise cv=0.30 when enabled; when disabled exact bursts visible (same as frozen study, baseline).
- Workload families (7, config.py): `short_jobs` (short uniform bursts 2-10), `long_jobs` (15-30), `mixed` (mix), `cpu_bursty` (bimodal batch), `priority_skewed` (priority×burst), `staggered_interactive` (staggered arrivals), `poisson_arrivals` (Poisson, held out of training). Each trace default 15 processes.
- Drift kinds: `burst_shock` (post-injection bursts ×4.0), `arrival_storm` (gaps ×0.35), `phase_transition` (second half from target family), `priority_inversion` (priority skew 0.7). All preserve pid count/order and pre-injection bursts/priorities.
- Controller causal boundary: `RuntimeSimulator` never passes `Workload`/future arrivals; controller sees only `RuntimeObservation` (ready set, completions, previous policy) plus perturbed bursts when estimator enabled. Service accounting via `record_service(pid, end-start-switch)` per slice, `record_completion(pid)` per completed pid (observable total).

**Metrics (all via `evaluation/metrics.py` + `evaluation/reliability.py`, no hidden params):** mean/P95/max waiting, starved (waiting >3×median), Jain fairness `(sum w)²/(n sum w²)`, throughput `n/total_time`, CPU util `100×busy/total_time`, context switches, switching overhead, incidents (detector events), interventions (`action_changed`, HOLD excluded), rollback `(rejected+inconclusive)/interventions`, fallback (safe fallback after 2 consecutive confirmed failures), recovery delay `mean(verification_decisions)` and `mean(window_end-simulated_time)`, plus audit-derived false-intervention/regression where available (lightweight script: audit None, frozen study audit in `healing_audit.csv`). Every workload instance paired across arms; `q_recovery` reports mean over 5 seeds and per-seed traces in CSV.

**Aggregation & denominators (explicit):** Per-trace `ReliabilityMetrics` from `ScheduleResult` + ledger; per-arm mean = `sum(trace_metric)/n_traces` (for `q_recovery` n=350, else 70); regression_rate = share where trace mean_waiting > baseline trace at same `(family,rep)` — for `q_recovery` mean over 5 seeds per workload vs baseline; rollback averaged over traces with interventions; recovery delay averaged over incidents with verification; fallback_rate = share with `fallback_triggered`. Comparisons via `evaluation/paired.py:paired_bootstrap_difference` — family-stratified crossed bootstrap over seeds×workloads (1000 replicates, seed 104729, same as frozen study), paired by `(repetition, fingerprint)`, mean diff and 95% CI.

### 4.1 Conditions run (seed 23819, 10 reps, 7 families, 200 ep/seed)

| Label | Drift | Estimator | CV | N traces | Q episodes/seed | Status |
|-------|-------|-----------|----|----------|-----------------|--------|
| A (primary) | burst_shock | enabled | 0.30 | 70 (350 Q) | 200 | reproduced |
| B (exact baseline) | burst_shock | disabled | — | 70 | 200 | reproduced |
| C (arrival drift) | arrival_storm | enabled | 0.30 | 70 | 200 | reproduced |

Artifacts: `results/drift/drift_*.csv` per condition (summary + per_trace + comparisons) and `_report.md`.

### 4.2 Results — Condition A: burst_shock + estimator cv0.30 (primary)

70 traces, drift `burst_shock` (post-injection burst ×4 at 35% horizon).

| Arm | Mean wait | P95 | Max | Starved | Fairness | Thrpt | CPU util | Switches | Incid | Interv | Rollback | Recovery (dec) | Fallback | Regression vs FCFS |
|-----|----------:|-----:|----:|--------:|---------:|------:|---------:|---------:|------:|-------:|---------:|---------------:|---------:|-------------------:|
| fixed_fcfs | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 | n/a |
| fixed_sjf | 208.80 | 660.06 | 660.06 | 2.63 | 0.533 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 | 0.00 |
| detector_only | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.74 | 0.00 | n/a | 4.9 | 0.000 | 0.00 |
| **rule_wait** | **252.75** | **694.16** | **694.16** | **1.51** | **0.611** | **0.0356** | **96.6** | **14.0** | **0.74** | **0.74** | **0.69** | **4.9** | **0.000** | **0.01** |
| **q_recovery** | **253.01** | **694.37** | **694.37** | **1.52** | **0.612** | **0.0356** | **96.5** | **14.2** | **0.74** | **0.65** | **0.80** | **5.0** | **0.000** | **0.03** |
| rule_fairness | 290.72 | 725.36 | 725.36 | 1.14 | 0.626 | 0.0349 | 95.0 | 23.2 | 0.86 | 0.86 | 0.98 | 12.0 | 0.000 | 0.41 |

Source: `results/drift/drift_burst_shock_estTrue_cv0.3_summary.csv` + `_per_trace.csv` (70/350 traces) + `_comparisons.csv`. Times in abstract time units; CPU util % single-CPU switch cost 1.

**Paired bootstrap (mean waiting, 1000 reps, same as frozen study):**

| Target | Reference | Mean diff | 95% CI |
|---|---|---:|---|
| rule_wait | fixed_fcfs | -33.61 | [-40.78, -25.75] |
| q_recovery | fixed_fcfs | -33.34 | [-40.95, -25.68] |
| q_recovery | rule_wait | +0.26 | [0.06, 0.59] |
| rule_wait | detector_only | -33.61 | [-40.78, -25.75] |
| q_recovery | fixed_sjf | +44.22 | [36.70, 51.48] |

### 4.3 Exact-burst baseline — burst_shock, estimator disabled

| Arm | Mean wait | P95 | Regression |
|-----|----------:|----:|-----------:|
| rule_wait | 252.91 | 695.04 | 0.01 |
| q_recovery | 254.15 | 696.27 | 0.03 |

Diff to A: rule_wait 252.91→252.75 (−0.06%) vs q_recovery 254.15→253.01 (−0.45%) — small sensitivities, not "<0.2% every arm", confirming cv0.30 noise does not dominate decisions under burst_shock; healing improves over FCFS by ~33.5 units (~12%) while only `rule_fairness` adds switches (+9). Full: `drift_burst_shock_estFalse_cv0.3_summary.csv` (and estTrue counterpart).

**Paired for exact baseline:** rule_wait vs FCFS -33.44 [-40.56, -25.58]; q_recovery vs rule_wait +1.24 [0.18, 3.31] (still <0.5% of mean).

### 4.4 Arrival drift — arrival_storm + estimator cv0.30

| Arm | Mean wait | P95 | Incid | Regression | Thrpt | CPU |
|-----|----------:|----:|------:|-----------:|------:|----:|
| fixed_fcfs | 137.60 | 268.61 | 0.00 | n/a | 0.0803 | 92.3 |
| fixed_sjf | 99.49 | 254.13 | 0.00 | 0.00 | 0.0803 | 92.3 |
| rule_wait | 130.32 | 262.90 | 0.57 | 0.01 | 0.0803 | 92.3 |
| q_recovery | 130.52 | 263.03 | 0.57 | 0.09 | 0.0803 | 92.3 |

`arrival_storm` reduces waits (higher throughput, lower util) and incident rate (0.57 vs 0.74); ordering preserved but effect smaller (130 vs 137 for FCFS). Full: `drift_arrival_storm_estTrue_cv0.3_summary.csv`. **Paired:** rule_wait vs FCFS -7.27 [-9.18, -5.32]; q_recovery vs rule_wait 0.19 [0.004, 0.56]: the interval excludes zero by 0.004 units (0.15 % of the 130.52 mean), so the difference is statistically detectable but practically negligible.

### 4.5 Interpretation (honest, no overclaim, negative results explicit)

- Under `burst_shock` (heavier tails after 35%), both healing policies cut mean wait ~12% vs `fixed_fcfs` (286→253) with no throughput/CPU loss; per-trace P95 equals max by definition (n=15, ceil(0.95*15)=15 → max, not evidence of tail shape); interpretable tail is pooled P95 over all process waits (see `results/drift/*_summary.csv` pooled_p95_waiting). **Fixed SJF oracle still beats healing by ~44 units** (209 vs 253, +44.22 [36.70, 51.48]), showing gap to oracle — healing does not reach SJF (oracle uses exact burst_time while adaptive arms use noisy EWMA when estimator enabled).
- **Q does not beat rule:** q_recovery 253.01 vs rule_wait 252.75 (Δ +0.26 [0.06, 0.59] includes small positive; arrival_storm Δ +0.19 [0.004, 0.56] is positive by 0.004 units, i.e. detectable only because the paired sample is large). Effect size 0.1-0.5% of mean — practically negligible though technically distinguishable on burst_shock due to large paired N (350 vs 70). This reproduces frozen study: **self-healing helps over static FCFS/detector_only (Δ -33.61), but Q ≠ rule** — Q coverage 1.1-1.3% of 5184 table (3.8-7.2% of 864-1728 reachable), flat after ~900 episodes, so extra training doesn't buy new states.
- Estimator error cv0.30 is **modelled log-normal**, not realism/unbiased; exact vs noisy differ <1% because decisions threshold on queue aging, not precise remainders. No claim EWMA predicts real OS bursts.
- Recovery delay ~5 decisions (~165 time units), rollback ~70-80% (most interventions rolled back), fallback 0 — conservative verifier (accept 7.5-8.5%, reject 0.3%, inconclusive 91-92%). False-intervention requires audit (frozen study: 5.7% rule_wait, 5.0% q_recovery).
- Regression vs FCFS: rule_wait 1%, q_recovery 3%, rule_fairness 41% (worse) — healing rarely harms, fairness rule often does.
- **Limitations (applies to all conditions):** single-CPU, fixed switch cost 1, no I/O/multicore/cache/paging/interrupts, synthetic families only, 70 traces/condition, Q undertrained, intervals conditional on declared design (5 seeds, 7 families, fixed reps), not universal workloads. Conclusions are simulator evidence, not production kernel claims. Don't call robust merely because tests pass.

## 5. Files changed / preserved

- Changed: `healing/burst_estimator.py`, `healing/controller.py`, `healing/config.py`, `tests/test_drift_and_reliability.py`, `experiments/drift_uncertainty_experiment.py`, `README.md` (test count), `docs/LEAKAGE_FIX_AND_DRIFT_REPORT.md`.
- Added results: `results/drift/drift_*_summary.csv`, `*_per_trace.csv`, `*_comparisons.csv`, `*_report.md` (9 files: 3 conditions ×3).
- Preserved byte-identical metrics: `results/healing/healing_metrics_final_test.csv`, `healing_comparisons.csv` (0 diff); summary gains additive config fields (expected, metrics identical); probe legacy diff documented.
- No renames; project name unchanged; no hardcoded local paths; no debug code.

## 6. Reproduce

```bash
# Verifications (evidence)
.venv/bin/python -m pytest -q                                      # 518 passed, 1 warning
.venv/bin/python -m pytest tests/test_drift_and_reliability.py -q  # 23 passed
PYTHONPATH=. .venv/bin/python main.py healing-study --results-dir /tmp/healing-compat --no-figures && diff results/healing/healing_metrics_final_test.csv /tmp/healing-compat/healing_metrics_final_test.csv

# Drift experiments (primary + baselines, all traceable to per_trace CSVs)
PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30
PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30 --no-estimator
PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift arrival_storm --cv 0.30

# Check traceability: reported means = mean(per_trace.csv:mean_waiting) per arm; CIs from paired_bootstrap_difference over (family,rep,fingerprint) pairs.
```

## 7. Git diff summary

`git diff fd8f103..HEAD --stat` covers files above; no `results/healing/` metric changes after validation; drift results committed as artifacts for review. Branch `arena/1ffa6b9e-os-project`, commit e85034c + next, push to origin.
