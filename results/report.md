# Generated experiment results

> This file is generated from `summary.json` by `experiments.report.render_report`; do not edit result values by hand.

## Scope and run design

The evaluated system is an **offline workload-aware policy selector**. Each model makes one batch decision from a complete synthetic workload description and applies one policy to the entire workload. It is not an OS scheduler and does not switch policies during execution.

- Training: 5 independent seeds (42, 43, 44, 45, 46), 1200 terminal episodes per seed.
- Evaluation: 210 unique workloads (30 per condition across 7 conditions).
- Selector decisions: 1050 model-decision rows (5 independently trained models per unique workload). These repeated rows are not additional unique workloads.
- Metric records: 5250 rows, comprising 4200 paired baseline records and 1050 learned-selector records.
- Training conditions: short_jobs, long_jobs, mixed, cpu_bursty, priority_skewed, staggered_interactive.
- Held-out condition(s): poisson_arrivals.
- All four baselines receive the same workload as each learned decision; evaluation fingerprints are disjoint from the training workloads.

## Results across evaluation workloads

Entries are arithmetic mean ± sample standard deviation. Baseline variability is across unique workload instances; selector variability is across model-seed × workload decisions. Context-switch count is per workload. The reward uses the separate derived context-switches-per-process metric.

| Policy | Waiting | Turnaround | Response | CPU util. (%) | Throughput | Context switches | n |
|---|---|---|---|---|---|---|---|
| FCFS | 123.39 ± 80.71 | 142.34 ± 91.49 | 123.39 ± 80.71 | 97.84 ± 6.19 | 0.08 ± 0.07 | 14.00 ± 0.00 | 210 |
| SJF | 88.10 ± 67.11 | 107.06 ± 77.80 | 88.10 ± 67.11 | 97.84 ± 6.19 | 0.08 ± 0.07 | 14.00 ± 0.00 | 210 |
| Round Robin | 167.57 ± 135.06 | 186.52 ± 145.75 | 19.38 ± 8.22 | 97.84 ± 6.19 | 0.08 ± 0.07 | 75.40 ± 40.91 | 210 |
| Priority | 122.22 ± 81.44 | 141.17 ± 92.25 | 122.22 ± 81.44 | 97.84 ± 6.19 | 0.08 ± 0.07 | 14.00 ± 0.00 | 210 |
| Offline Policy Selector (Q-Learning) | 88.66 ± 67.89 | 107.61 ± 78.54 | 88.11 ± 68.47 | 97.84 ± 6.17 | 0.08 ± 0.07 | 15.53 ± 4.04 | 1050 |

## Learned policy choices by condition

Counts below aggregate 5 independently trained selectors over the same 30 unique workload instances per condition. `unseen-state` counts identify evaluation states with no training update; their equal zero Q-values fall back to the deterministic lowest-index action (FCFS) and are not evidence of a learned choice.

| Condition | Unique workloads | Model decisions | FCFS n | SJF n | Round Robin n | Priority n | Unseen-state decisions |
|---|---|---|---|---|---|---|---|
| short_jobs | 30 | 150 | 3 | 117 | 30 | 0 | 2 |
| long_jobs | 30 | 150 | 5 | 145 | 0 | 0 | 4 |
| mixed | 30 | 150 | 0 | 150 | 0 | 0 | 0 |
| cpu_bursty | 30 | 150 | 0 | 150 | 0 | 0 | 0 |
| priority_skewed | 30 | 150 | 4 | 144 | 0 | 2 | 0 |
| staggered_interactive | 30 | 150 | 0 | 0 | 150 | 0 | 0 |
| poisson_arrivals | 30 | 150 | 46 | 104 | 0 | 0 | 32 |

### Per-seed selection stability

The per-seed table in `summary.json` is generated from `decisions.csv` and should be consulted alongside the pooled counts; pooled counts alone do not establish that every independent training run learned the same condition-specific action.

| Seed | Condition | Unique workloads | FCFS | SJF | Round Robin | Priority | Unseen-state decisions |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | short_jobs | 30 | 0 | 21 | 9 | 0 | 0 |
| 42 | long_jobs | 30 | 1 | 29 | 0 | 0 | 0 |
| 42 | mixed | 30 | 0 | 30 | 0 | 0 | 0 |
| 42 | cpu_bursty | 30 | 0 | 30 | 0 | 0 | 0 |
| 42 | priority_skewed | 30 | 2 | 28 | 0 | 0 | 0 |
| 42 | staggered_interactive | 30 | 0 | 0 | 30 | 0 | 0 |
| 42 | poisson_arrivals | 30 | 9 | 21 | 0 | 0 | 6 |
| 43 | short_jobs | 30 | 1 | 20 | 9 | 0 | 1 |
| 43 | long_jobs | 30 | 1 | 29 | 0 | 0 | 1 |
| 43 | mixed | 30 | 0 | 30 | 0 | 0 | 0 |
| 43 | cpu_bursty | 30 | 0 | 30 | 0 | 0 | 0 |
| 43 | priority_skewed | 30 | 2 | 28 | 0 | 0 | 0 |
| 43 | staggered_interactive | 30 | 0 | 0 | 30 | 0 | 0 |
| 43 | poisson_arrivals | 30 | 19 | 11 | 0 | 0 | 15 |
| 44 | short_jobs | 30 | 1 | 29 | 0 | 0 | 1 |
| 44 | long_jobs | 30 | 1 | 29 | 0 | 0 | 1 |
| 44 | mixed | 30 | 0 | 30 | 0 | 0 | 0 |
| 44 | cpu_bursty | 30 | 0 | 30 | 0 | 0 | 0 |
| 44 | priority_skewed | 30 | 0 | 30 | 0 | 0 | 0 |
| 44 | staggered_interactive | 30 | 0 | 0 | 30 | 0 | 0 |
| 44 | poisson_arrivals | 30 | 9 | 21 | 0 | 0 | 6 |
| 45 | short_jobs | 30 | 1 | 17 | 12 | 0 | 0 |
| 45 | long_jobs | 30 | 1 | 29 | 0 | 0 | 1 |
| 45 | mixed | 30 | 0 | 30 | 0 | 0 | 0 |
| 45 | cpu_bursty | 30 | 0 | 30 | 0 | 0 | 0 |
| 45 | priority_skewed | 30 | 0 | 28 | 0 | 2 | 0 |
| 45 | staggered_interactive | 30 | 0 | 0 | 30 | 0 | 0 |
| 45 | poisson_arrivals | 30 | 6 | 24 | 0 | 0 | 2 |
| 46 | short_jobs | 30 | 0 | 30 | 0 | 0 | 0 |
| 46 | long_jobs | 30 | 1 | 29 | 0 | 0 | 1 |
| 46 | mixed | 30 | 0 | 30 | 0 | 0 | 0 |
| 46 | cpu_bursty | 30 | 0 | 30 | 0 | 0 | 0 |
| 46 | priority_skewed | 30 | 0 | 30 | 0 | 0 | 0 |
| 46 | staggered_interactive | 30 | 0 | 0 | 30 | 0 | 0 |
| 46 | poisson_arrivals | 30 | 3 | 27 | 0 | 0 | 3 |

## Paired waiting-time and turnaround comparisons

Aggregate reductions are computed from the reported means as `100 × (baseline mean − selector mean) / baseline mean`. They are not averages of per-workload percentage ratios.

| Baseline | Waiting-time reduction | Turnaround-time reduction | Paired model-workload observations | Unique workloads |
|---|---:|---:|---:|---:|
| FCFS | 28.1% | 24.4% | 1050 | 210 |
| SJF | -0.6% | -0.5% | 1050 | 210 |
| Round Robin | 47.1% | 42.3% | 1050 | 210 |
| Priority | 27.5% | 23.8% | 1050 | 210 |

## State coverage and limitations

The tabular state has 4 variables, 3 bins per variable, and 81 possible states; coverage differs by training seed. Held-out Poisson arrivals are excluded from training by design. State overlap with training conditions is not guaranteed; unseen held-out states use the documented zero-initialization tie fallback. Accordingly, held-out-family performance is a limited generalization check, not evidence of broad distributional generalization.

| Condition | Unique workloads | Distinct states in evaluation set | Model decisions | State-seen rate | Unseen-state decisions |
|---|---:|---:|---:|---:|---:|
| cpu_bursty | 30 | 2 | 150 | 100.0% | 0 |
| long_jobs | 30 | 5 | 150 | 97.3% | 4 |
| mixed | 30 | 4 | 150 | 100.0% | 0 |
| poisson_arrivals | 30 | 7 | 150 | 78.7% | 32 |
| priority_skewed | 30 | 6 | 150 | 100.0% | 0 |
| short_jobs | 30 | 7 | 150 | 98.7% | 2 |
| staggered_interactive | 30 | 1 | 150 | 100.0% | 0 |

### Training-seed variability

| Training seed | Mean episodic reward | States visited / possible | Random-action exploration rate |
|---:|---:|---:|---:|
| 42 | 0.21 | 27 / 81 | 19.1% |
| 43 | 0.21 | 28 / 81 | 19.4% |
| 44 | 0.22 | 24 / 81 | 17.2% |
| 45 | 0.21 | 28 / 81 | 17.6% |
| 46 | 0.22 | 26 / 81 | 18.8% |

The exploration rate is the observed fraction of episode decisions that took the epsilon-greedy random-action branch; it is distinct from the epsilon probability and from the fraction of actions differing from greedy.

## Reproducibility

- Runtime: Python 3.11.2 (CPython).
- Dependency versions: numpy 2.4.6, pandas 3.0.6, matplotlib 3.11.2, pytest 9.1.1.
- Locked dependency manifest SHA-256: `9b059d7a455a846c437068a7b0a018579cee02be4dac6778addb8cf543572772`.
- Rerun with `python main.py experiment`; omit plotting with `python main.py experiment --no-figures`.
- Machine-readable inputs/results are `config.json`, `workloads.csv`, `training_history.json`, `q_table.json`, `metrics.csv`, `decisions.csv`, and `summary.json` in this directory.
