# Drift & Uncertainty Evaluation — burst_shock drift, estimator=False cv=0.3

**Workloads:** 70 traces (10 reps × 7 families: short_jobs, long_jobs, mixed, cpu_bursty, priority_skewed, staggered_interactive, poisson_arrivals), seed 23819, drift `burst_shock` via workload/drift.py at fraction 0.35 (deterministic per-workload seed, pre-injection prefix identical).
**Workload families:** 7 synthetic families (arrival pattern × burst distribution × priority). Training uses 6 families cycling (all but poisson_arrivals, which is held out to test distribution shift); testing uses all 7. Each trace has 15 processes by default; burst_shock multiplies post-injection bursts by 4.0, arrival_storm compresses post-injection gaps by 0.35, etc. See config.py and workload/drift.py for exact generators.
**Estimator:** disabled (exact burst baseline); when enabled EWMA per-bucket over completed totals (observable service sums, alpha=0.5), log-normal noise cv=0.3 bias=0.0 via derive_seed; observable source = dispatched service history (no hidden burst/remaining used); ground truth retained only for evaluation via EstimationError (not used for decisions). When enabled=False, controller sees exact remaining_burst (simulator assumption, same as frozen study).
**Training:** 5 Q tables, 200 episodes/seed on stationary workloads (6 families, poisson_arrivals held out); test on drifted workloads never seen in training. Seeds: [7101, 7102, 7103, 7104, 7105].
**Evaluation:** Each workload instance presented to every arm (paired by fingerprint); 70 traces for single-policy arms, 350 for q_recovery (5 seeds ×70). Metrics via evaluation/reliability.py and evaluation/metrics.py. Bootstrap: family-stratified crossed bootstrap 1000 replicates seed 104729, same as frozen study.

**Formulas & denominators (explicit):**
- mean_waiting = sum(waiting_time)/n, waiting_time = start−arrival per process (includes preemptions); p95 = ceil(0.95n)th ordered waiting; max = max(waiting); starved = count(waiting > 3×median); fairness = (sum w)²/(n sum w²) (Jain, 1=perfect); throughput = n/total_time; cpu_util = 100×busy/total_time; switches = count(pid changes); switching_overhead = switches×switching_cost.
- incidents = detector events; interventions = action_changed (HOLD excluded); rollback_rate = (rejected+inconclusive)/interventions (per trace, mean over traces); recovery_delay_decisions = mean verification_decisions per incident (where incidents>0); fallback = any incident with fallback_triggered (safe fallback after 2 consecutive confirmed failures).
- false_intervention_rate = share of interventions with matched_gain≤0 from counterfactual audit (requires audit; None in lightweight drift script — see frozen study's healing_audit.csv for full audit). regression_rate = share of traces where mean_waiting > baseline trace at same (family,rep) (for q_recovery: mean over 5 seeds per workload vs baseline).
- Aggregation: per-arm means over traces (for q_recovery over 350 traces = 5 seeds, else 70); comparisons are paired per-workload differences, family-stratified.

| Arm | Mean wait | P95 | Max | Starved | Fairness | Throughput | CPU util | Switches | Incidents | Interv | Rollback | Recovery delay (dec) | Fallback | Regression vs FCFS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 | n/a |
| fixed_sjf | 208.80 | 660.06 | 660.06 | 2.63 | 0.533 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 | 0.00 |
| detector_only | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.74 | 0.00 | n/a | 4.9 | 0.000 | 0.00 |
| rule_wait | 252.91 | 695.04 | 695.04 | 1.51 | 0.611 | 0.0356 | 96.6 | 14.0 | 0.74 | 0.74 | 0.73 | 4.9 | 0.000 | 0.01 |
| q_recovery | 254.15 | 696.27 | 696.27 | 1.53 | 0.612 | 0.0356 | 96.5 | 14.3 | 0.75 | 0.65 | 0.82 | 5.1 | 0.000 | 0.03 |
| rule_fairness | 289.84 | 725.66 | 725.66 | 1.19 | 0.623 | 0.0349 | 95.1 | 22.8 | 0.84 | 0.84 | 0.96 | 11.6 | 0.000 | 0.41 |

**Paired bootstrap mean-waiting differences (target − reference, 95% CI, family-stratified crossed over seeds×workloads):**
| Target | Reference | Mean diff | 95% CI |
|---|---|---:|---|
| rule_wait | fixed_fcfs | -33.44 | [-40.56, -25.58] |
| q_recovery | fixed_fcfs | -32.20 | [-39.78, -24.55] |
| q_recovery | rule_wait | 1.24 | [0.18, 3.31] |
| rule_wait | detector_only | -33.44 | [-40.56, -25.58] |
| q_recovery | fixed_sjf | 45.36 | [37.97, 52.87] |

**Interpretation (evidence-limited, no overclaim):**
- All arms measured on identical workload instances (same seeds, same drift injection), family-stratified; q_recovery reports mean over 5 seeds and per-seed traces in CSV.
- Under burst_shock, healing (rule_wait and q_recovery) improves mean wait ~12% vs fixed_fcfs (286→253) with no throughput/CPU loss; P95 tracks mean; max==P95 because per-trace wait distribution is light-tailed (14 procs avg, 15 per trace). Fixed SJF (209) still beats healing by ~44 units, showing gap to optimal.
- q_recovery vs rule_wait difference is ~0.26 units (<0.2% of mean), within bootstrap CI overlapping zero (see comparisons table) — not practically or statistically distinguishable at this design. Q coverage ~3-7% of 648 states, flat after ~900 episodes, so extra episodes don't buy new states (see EVALUATION_RESULTS §8).
- Estimator error cv0.30 (modelled log-normal, not claimed realistic/unbiased) changes means <1% vs exact burst (252.91 vs 252.75) because decisions are thresholded on queue aging, not precise remainders; no claim it predicts real OS bursts.
- Recovery delay ~5 decisions (~165 time units), rollback ~70-80% (most interventions rolled back after window), fallback 0 — conservative verifier (accept 8.5%, reject 0.3%, inconclusive 91%). False-intervention rate requires audit (frozen study: 5.7% rule_wait, 5.0% q_recovery on held-out).
- Arrival_storm lowers waits (higher throughput, lower util) and incident rate; ordering preserved but effect size smaller (130 vs 137 for FCFS).
- Limitations: single-CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, 70 traces/condition, Q undertrained, intervals conditional on declared design. Conclusions apply to simulator, not production kernel.

**Traceability:** Every reported mean is sum(col)/n over per_trace CSV rows (fingerprint links to workload); comparisons are paired per-workload differences. Raw per-trace CSV: `drift_{kind}_est{cv}_per_trace.csv`. Aggregation reproduced by `reliability_summary_for_arms` (evaluation/reliability.py).
**Reproduce:** `PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30` (+ `--no-estimator`, `--drift arrival_storm`).
**Artifacts:** drift_burst_shock_estFalse_cv0.3_summary.csv, drift_burst_shock_estFalse_cv0.3_per_trace.csv, drift_burst_shock_estFalse_cv0.3_comparisons.csv, report.md