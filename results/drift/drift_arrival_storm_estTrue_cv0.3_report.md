# Drift & Uncertainty Evaluation — arrival_storm drift, estimator=True cv=0.3

**Workloads:** 70 traces (10 reps × 7 families: short_jobs, long_jobs, mixed, cpu_bursty, priority_skewed, staggered_interactive, poisson_arrivals), seed 23819, drift `arrival_storm` via workload/drift.py at fraction 0.35 (deterministic per-workload seed, pre-injection prefix identical).
**Workload families:** 7 synthetic families (arrival pattern × burst distribution × priority). Training uses 6 families cycling (all but poisson_arrivals, which is held out to test distribution shift); testing uses all 7. Each trace has 15 processes by default; burst_shock multiplies post-injection bursts by 4.0, arrival_storm compresses post-injection gaps by 0.35, etc. See config.py and workload/drift.py for exact generators.
**Estimator:** enabled; when enabled EWMA per-bucket over completed totals (observable service sums, alpha=0.5), log-normal noise cv=0.3 bias=0.0 via derive_seed; observable source = dispatched service history (no hidden burst/remaining used); ground truth retained only for evaluation via EstimationError (not used for decisions). When enabled=False, controller sees exact remaining_burst (simulator assumption, same as frozen study). SJF baseline `fixed_sjf` always uses exact burst_time (oracle, not equal-information) even when estimator is enabled for adaptive arms; no estimated-burst SJF variant is included in this run.
**Training:** 5 Q tables, 200 episodes/seed on stationary workloads (6 families, poisson_arrivals held out); test on drifted workloads never seen in training. Seeds: [7101, 7102, 7103, 7104, 7105].
**Evaluation:** Each workload instance presented to every arm (paired by fingerprint); 70 traces for single-policy arms, 350 for q_recovery (5 seeds ×70). Metrics via evaluation/reliability.py and evaluation/metrics.py. Bootstrap: family-stratified crossed bootstrap 1000 replicates seed 104729, same as frozen study.

**Formulas & denominators (explicit):**
- mean_waiting = sum(waiting_time)/n, waiting_time = start−arrival per process (includes preemptions); per-trace p95 = ceil(0.95n)th ordered waiting — with n=15 per trace, ceil(0.95*15)=15 so per-trace p95 is definitionally the max (not evidence about tail shape); max = max(waiting); pooled_p95_waiting = ceil(0.95*N)th over all process waits pooled across traces (N≈1050 for 70-trace arms, 5250 for q_recovery), interpretable tail; starved = count(waiting > 3×median); fairness = (sum w)²/(n sum w²) (Jain, 1=perfect); throughput = n/total_time; cpu_util = 100×busy/total_time; switches = count(pid changes); switching_overhead = switches×switching_cost.
- incidents = detector events; interventions = action_changed (HOLD excluded); rollback_rate = (rejected+inconclusive)/interventions (per trace, mean over traces); recovery_delay_decisions = mean verification_decisions per incident (where incidents>0); fallback = any incident with fallback_triggered (safe fallback after 2 consecutive confirmed failures).
- false_intervention_rate = share of interventions with matched_gain≤0 from counterfactual audit (requires audit; None in lightweight drift script — see frozen study's healing_audit.csv for full audit). regression_rate = share of traces where mean_waiting > baseline trace at same (family,rep) (for q_recovery: mean over 5 seeds per workload vs baseline).
- Aggregation: per-arm means over traces (for q_recovery over 350 traces = 5 seeds, else 70); comparisons are paired per-workload differences, family-stratified.

| Arm | Mean wait | P95 (per-trace mean of max) | Pooled P95 (all waits) | Pooled max | Starved | Fairness | Throughput | CPU util | Switches | Incidents | Interv | Rollback | Recovery delay (dec) | Fallback | Regression vs FCFS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 137.60 | 268.61 (=max, n=15) | 396.00 (N=1050) | 563.00 | 0.00 | 0.751 | 0.0803 | 92.3 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 | n/a |
| fixed_sjf | 99.49 | 254.13 (=max, n=15) | 358.00 (N=1050) | 543.00 | 1.47 | 0.617 | 0.0803 | 92.3 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 | 0.00 |
| detector_only | 137.60 | 268.61 (=max, n=15) | 396.00 (N=1050) | 563.00 | 0.00 | 0.751 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.00 | n/a | 5.0 | 0.000 | 0.00 |
| rule_wait | 130.32 | 262.90 (=max, n=15) | 388.00 (N=1050) | 563.00 | 0.01 | 0.754 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.57 | 0.92 | 5.0 | 0.000 | 0.01 |
| q_recovery | 130.52 | 263.03 (=max, n=15) | 388.00 (N=5250) | 563.00 | 0.01 | 0.754 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.53 | 0.92 | 5.0 | 0.000 | 0.09 |
| rule_fairness | 140.56 | 269.93 (=max, n=15) | 402.00 (N=1050) | 563.00 | 0.00 | 0.755 | 0.0792 | 90.8 | 18.2 | 0.60 | 0.60 | 1.00 | 9.8 | 0.000 | 0.36 |

**Paired bootstrap mean-waiting differences (target − reference, 95% CI, family-stratified crossed over seeds×workloads; Q's 350 records = 5 seeds×70 shared workload instances, not 350 independent workloads; crossed bootstrap in evaluation/paired.py::paired_bootstrap_difference resamples seeds and workloads independently retaining pairing and stratification):**
| Target | Reference | Mean diff | 95% CI |
|---|---|---:|---|
| rule_wait | fixed_fcfs | -7.27 | [-9.18, -5.32] |
| q_recovery | fixed_fcfs | -7.08 | [-9.09, -5.08] |
| q_recovery | rule_wait | 0.19 | [0.00, 0.56] |
| rule_wait | detector_only | -7.27 | [-9.18, -5.32] |
| q_recovery | fixed_sjf | 31.02 | [28.13, 33.90] |

**Interpretation (evidence-limited, no overclaim):**
- All arms measured on identical workload instances (same seeds, same drift injection), family-stratified; q_recovery reports mean over 5 seeds and per-seed traces in CSV.
- Under arrival_storm, healing improves mean wait ~5% vs fixed_fcfs (138→130) with no throughput/CPU loss; per-trace P95 equals max by definition (n=15, ceil(0.95*15)=15), so the interpretable tail is pooled P95 over all process waits. Fixed SJF oracle (~99, exact burst knowledge, not equal-information) still beats healing by ~31 units (see q_recovery−fixed_sjf CI), showing gap to oracle ordering; adaptive arms used noisy EWMA estimates when estimator enabled.
- q_recovery vs rule_wait (arrival_storm, estimator=True): 0.19 [0.00, 0.56] (CI includes 0, not statistically distinguishable; effect 0.1-0.5% of mean); n=5 training seeds×70 workloads=350 paired records, crossed family-stratified bootstrap retains pairing. Q coverage ~3-7% of 648 states, flat after ~900 episodes, so extra episodes don't buy new states (see EVALUATION_RESULTS §8). For cross-drift reference: burst_shock +0.26 [0.06,0.59] excludes 0 (~0.1% mean), arrival_storm +0.19 [0.00,0.56] includes 0.
- Estimator error cv0.30 (modelled log-normal, not claimed realistic/unbiased): rule_wait 252.91→252.75 (−0.06%) vs Q 254.15→253.01 (−0.45%) — small sensitivities, not '<0.2% every arm'; decisions are thresholded on queue aging, not precise remainders; no claim it predicts real OS bursts.
- Recovery delay ~5 decisions (~165 time units), rollback ~70-80% (most interventions rolled back after window), fallback 0 — conservative verifier (accept 8.5%, reject 0.3%, inconclusive 91%). False-intervention rate requires audit (frozen study: 5.7% rule_wait, 5.0% q_recovery on held-out).
- Arrival_storm lowers waits (higher throughput, lower util) and incident rate; ordering preserved but effect size smaller (130 vs 137 for FCFS).
- Limitations: single-CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, 70 traces/condition, Q undertrained, intervals conditional on declared design. Conclusions apply to simulator, not production kernel.

**Traceability:** Every reported mean is sum(col)/n over per_trace CSV rows (fingerprint links to workload); comparisons are paired per-workload differences. Raw per-trace CSV: `drift_{kind}_est{cv}_per_trace.csv`. Aggregation reproduced by `reliability_summary_for_arms` (evaluation/reliability.py).
**Reproduce:** `PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 10 --train-episodes 200 --drift burst_shock --cv 0.30` (+ `--no-estimator`, `--drift arrival_storm`).
**Artifacts:** drift_arrival_storm_estTrue_cv0.3_summary.csv, drift_arrival_storm_estTrue_cv0.3_per_trace.csv, drift_arrival_storm_estTrue_cv0.3_comparisons.csv, report.md