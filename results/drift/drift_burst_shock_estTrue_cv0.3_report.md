# Drift & Uncertainty Evaluation — burst_shock drift, estimator=True cv=0.3

**Workloads:** 70 traces (10 reps × 7 families), seed 23819, drift `burst_shock` via workload/drift.py at fraction 0.35 (deterministic, pre-injection prefix identical).
**Estimator:** enabled; when enabled EWMA per-bucket over completed totals (observable service sums), log-normal noise cv=0.3 bias=0.0 alpha=0.5; observable source = dispatched service history (no hidden burst/remaining used); ground truth retained only for evaluation.
**Training:** 5 Q tables, 200 episodes/seed on stationary workloads (6 families, poisson_arrivals held out); test on drifted workloads never seen in training.
**Q seeds:** [7101, 7102, 7103, 7104, 7105]

| Arm | Mean wait | P95 | Max | Starved | Fairness | Throughput | CPU util | Switches | Incidents | Interv | Rollback | Recovery delay | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| fixed_sjf | 208.80 | 660.06 | 660.06 | 2.63 | 0.533 | 0.0356 | 96.6 | 14.0 | 0.00 | 0.00 | n/a | n/a | 0.000 |
| detector_only | 286.36 | 725.26 | 725.26 | 1.51 | 0.622 | 0.0356 | 96.6 | 14.0 | 0.74 | 0.00 | n/a | 4.9 | 0.000 |
| rule_wait | 252.75 | 694.16 | 694.16 | 1.51 | 0.611 | 0.0356 | 96.6 | 14.0 | 0.74 | 0.74 | 0.69 | 4.9 | 0.000 |
| q_recovery | 253.01 | 694.37 | 694.37 | 1.52 | 0.612 | 0.0356 | 96.5 | 14.2 | 0.74 | 0.65 | 0.80 | 5.0 | 0.000 |
| rule_fairness | 290.72 | 725.36 | 725.36 | 1.14 | 0.626 | 0.0349 | 95.0 | 23.2 | 0.86 | 0.86 | 0.98 | 12.0 | 0.000 |

**Interpretation notes:**
- All arms measured on identical workload instances (same seeds, same drift injection), family-stratified.
- False-intervention / regression from audit require matched replay (not run here for lightweight script); rollback/recovery/fallback from incident ledger.
- Estimator documented as 'error modelled as log-normal', not 'unbiased' or 'realistic'; no claim about real OS burst prediction.
- Limitations: single-CPU, fixed switch cost 1, no I/O/multicore/cache, synthetic families only, Q coverage ~3-7% of 648 states (flat after ~900 episodes), intervals conditional on declared design.