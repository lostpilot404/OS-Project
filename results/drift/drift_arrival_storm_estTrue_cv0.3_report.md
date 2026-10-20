# Drift & Uncertainty Evaluation — arrival_storm drift, estimator=True cv=0.3

**Workloads:** 70 traces (10 reps × 7 families), seed 23819, drift `arrival_storm` via workload/drift.py at fraction 0.35 (deterministic, pre-injection prefix identical).
**Estimator:** enabled; when enabled EWMA per-bucket over completed totals (observable service sums), log-normal noise cv=0.3 bias=0.0 alpha=0.5; observable source = dispatched service history (no hidden burst/remaining used); ground truth retained only for evaluation.
**Training:** 5 Q tables, 200 episodes/seed on stationary workloads (6 families, poisson_arrivals held out); test on drifted workloads never seen in training.
**Q seeds:** [7101, 7102, 7103, 7104, 7105]

| Arm | Mean wait | P95 | Max | Starved | Fairness | Throughput | CPU util | Switches | Incidents | Interv | Rollback | Recovery delay | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 137.60 | 268.61 | 268.61 | 0.00 | 0.751 | 0.0803 | 92.3 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| fixed_sjf | 99.49 | 254.13 | 254.13 | 1.47 | 0.617 | 0.0803 | 92.3 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| detector_only | 137.60 | 268.61 | 268.61 | 0.00 | 0.751 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.00 | n/a | 5.0 | 0.000 |
| rule_wait | 130.32 | 262.90 | 262.90 | 0.01 | 0.754 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.57 | 0.92 | 5.0 | 0.000 |
| q_recovery | 130.52 | 263.03 | 263.03 | 0.01 | 0.754 | 0.0803 | 92.3 | 14.0 | 0.57 | 0.53 | 0.92 | 5.0 | 0.000 |
| rule_fairness | 140.56 | 269.93 | 269.93 | 0.00 | 0.755 | 0.0792 | 90.8 | 18.2 | 0.60 | 0.60 | 1.00 | 9.8 | 0.000 |

**Interpretation notes:**
- All arms measured on identical workload instances (same seeds, same drift injection), family-stratified.
- False-intervention / regression from audit require matched replay (not run here for lightweight script); rollback/recovery/fallback from incident ledger.
- Estimator documented as 'error modelled as log-normal', not 'unbiased' or 'realistic'; no claim about real OS burst prediction.
- Limitations: single-CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, Q coverage ~3-7% of 648 states (flat after ~900 episodes), intervals conditional on declared design.