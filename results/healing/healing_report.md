# Self-healing CPU scheduling: generated report

This file is generated from the machine-readable artifacts in this directory (`healing_metrics_*.csv`, `healing_incidents_*.csv`, `healing_audit.csv`, `healing_comparisons.csv`, `healing_summary.json`). Every number below appears in those files.

* split: **final_test**
* workloads per arm: 210
* training seeds: [7101, 7102, 7103, 7104, 7105]
* final-test seed: 23819
* bootstrap replicates: 1000

## Mean metrics by arm

| Arm | Mean waiting | P95 waiting | Mean response | Context switches | Incidents | Interventions |
|---|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 130.838 | 258.424 | 130.838 | 14.000 | 0.000 | 0.000 |
| fixed_sjf | 94.658 | 249.205 | 94.658 | 14.000 | 0.000 | 0.000 |
| fixed_round_robin | 217.054 | 310.414 | 26.497 | 75.262 | 0.000 | 0.000 |
| fixed_priority | 128.317 | 262.748 | 128.317 | 14.000 | 0.000 | 0.000 |
| detector_only | 130.838 | 258.424 | 130.838 | 14.000 | 0.652 | 0.000 |
| rule_fairness | 136.780 | 263.719 | 110.403 | 20.019 | 0.714 | 0.714 |
| rule_wait | 122.388 | 251.876 | 122.388 | 14.000 | 0.671 | 0.671 |
| rule_wait_from_sjf | 94.873 | 249.543 | 94.873 | 14.000 | 0.824 | 0.019 |
| q_recovery | 122.564 | 251.976 | 122.540 | 14.034 | 0.671 | 0.646 |
| q_recovery_from_sjf | 94.658 | 249.205 | 94.658 | 14.000 | 0.824 | 0.000 |

## Predeclared paired comparisons (family-stratified bootstrap)

| Target | Reference | Metric | Difference | 95% CI |
|---|---|---|---:|---|
| rule_wait | fixed_fcfs | avg_waiting_time | -8.4495 | [-9.5366, -7.4297] |
| rule_wait | fixed_fcfs | p95_waiting_time | -6.5476 | [-8.1146, -5.0523] |
| rule_wait | fixed_fcfs | max_waiting_time | -6.5476 | [-8.1527, -5.0944] |
| rule_wait | fixed_fcfs | avg_turnaround_time | -8.4495 | [-9.5302, -7.4650] |
| rule_wait | fixed_fcfs | avg_response_time | -8.4495 | [-9.4800, -7.4890] |
| rule_wait | fixed_fcfs | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_fcfs | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_fcfs | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | avg_waiting_time | +27.7305 | [+26.0515, +29.7235] |
| rule_wait | fixed_sjf | p95_waiting_time | +2.6714 | [+1.2136, +4.2190] |
| rule_wait | fixed_sjf | max_waiting_time | +2.6714 | [+1.1995, +4.1813] |
| rule_wait | fixed_sjf | avg_turnaround_time | +27.7305 | [+25.8391, +29.6616] |
| rule_wait | fixed_sjf | avg_response_time | +27.7305 | [+25.9877, +29.4625] |
| rule_wait | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | avg_waiting_time | -8.4495 | [-9.4370, -7.3504] |
| rule_wait | detector_only | p95_waiting_time | -6.5476 | [-8.1619, -5.0520] |
| rule_wait | detector_only | max_waiting_time | -6.5476 | [-8.1144, -5.0093] |
| rule_wait | detector_only | avg_turnaround_time | -8.4495 | [-9.4219, -7.3788] |
| rule_wait | detector_only | avg_response_time | -8.4495 | [-9.4978, -7.4349] |
| rule_wait | detector_only | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_fairness | fixed_fcfs | avg_waiting_time | +5.9422 | [+4.3813, +7.4543] |
| rule_fairness | fixed_fcfs | p95_waiting_time | +5.2952 | [+3.4848, +7.1238] |
| rule_fairness | fixed_fcfs | max_waiting_time | +5.2952 | [+3.3761, +7.0675] |
| rule_fairness | fixed_fcfs | avg_turnaround_time | +5.9422 | [+4.5586, +7.3716] |
| rule_fairness | fixed_fcfs | avg_response_time | -20.4343 | [-23.2087, -17.8486] |
| rule_fairness | fixed_fcfs | cpu_utilization | -1.8880 | [-2.0806, -1.6800] |
| rule_fairness | fixed_fcfs | throughput | -0.0013 | [-0.0014, -0.0011] |
| rule_fairness | fixed_fcfs | context_switches | +6.0190 | [+5.3238, +6.7715] |
| q_recovery | fixed_fcfs | avg_waiting_time | -8.2737 | [-9.3094, -7.1997] |
| q_recovery | fixed_fcfs | p95_waiting_time | -6.4476 | [-7.9601, -4.8586] |
| q_recovery | fixed_fcfs | max_waiting_time | -6.4476 | [-8.0691, -4.9389] |
| q_recovery | fixed_fcfs | avg_turnaround_time | -8.2737 | [-9.3640, -7.2494] |
| q_recovery | fixed_fcfs | avg_response_time | -8.2976 | [-9.3132, -7.2211] |
| q_recovery | fixed_fcfs | cpu_utilization | -0.0196 | [-0.0481, +0.0000] |
| q_recovery | fixed_fcfs | throughput | -0.0000 | [-0.0001, +0.0000] |
| q_recovery | fixed_fcfs | context_switches | +0.0343 | [+0.0000, +0.0991] |
| q_recovery | fixed_sjf | avg_waiting_time | +27.9063 | [+26.1082, +29.8708] |
| q_recovery | fixed_sjf | p95_waiting_time | +2.7714 | [+1.1345, +4.2107] |
| q_recovery | fixed_sjf | max_waiting_time | +2.7714 | [+1.3809, +4.2651] |
| q_recovery | fixed_sjf | avg_turnaround_time | +27.9063 | [+26.0213, +29.8194] |
| q_recovery | fixed_sjf | avg_response_time | +27.8824 | [+26.0341, +29.7919] |
| q_recovery | fixed_sjf | cpu_utilization | -0.0196 | [-0.0534, +0.0000] |
| q_recovery | fixed_sjf | throughput | -0.0000 | [-0.0001, +0.0000] |
| q_recovery | fixed_sjf | context_switches | +0.0343 | [+0.0000, +0.1153] |
| q_recovery | rule_wait | avg_waiting_time | +0.1758 | [-0.0443, +0.4827] |
| q_recovery | rule_wait | p95_waiting_time | +0.1000 | [-0.0867, +0.3544] |
| q_recovery | rule_wait | max_waiting_time | +0.1000 | [-0.0886, +0.3401] |
| q_recovery | rule_wait | avg_turnaround_time | +0.1758 | [-0.0528, +0.4494] |
| q_recovery | rule_wait | avg_response_time | +0.1519 | [-0.0756, +0.4200] |
| q_recovery | rule_wait | cpu_utilization | -0.0196 | [-0.0513, +0.0000] |
| q_recovery | rule_wait | throughput | -0.0000 | [-0.0001, +0.0000] |
| q_recovery | rule_wait | context_switches | +0.0343 | [+0.0000, +0.1067] |
| q_recovery | detector_only | avg_waiting_time | -8.2737 | [-9.3284, -7.2302] |
| q_recovery | detector_only | p95_waiting_time | -6.4476 | [-7.9963, -4.9932] |
| q_recovery | detector_only | max_waiting_time | -6.4476 | [-8.1175, -4.9717] |
| q_recovery | detector_only | avg_turnaround_time | -8.2737 | [-9.3665, -7.1917] |
| q_recovery | detector_only | avg_response_time | -8.2976 | [-9.3250, -7.2517] |
| q_recovery | detector_only | cpu_utilization | -0.0196 | [-0.0528, +0.0000] |
| q_recovery | detector_only | throughput | -0.0000 | [-0.0001, +0.0000] |
| q_recovery | detector_only | context_switches | +0.0343 | [+0.0000, +0.0972] |
| q_recovery_from_sjf | fixed_sjf | avg_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | p95_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | max_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | avg_turnaround_time | -0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | avg_response_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | throughput | -0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | avg_waiting_time | +0.2149 | [+0.0444, +0.4299] |
| rule_wait_from_sjf | fixed_sjf | p95_waiting_time | +0.3381 | [-0.0048, +0.7858] |
| rule_wait_from_sjf | fixed_sjf | max_waiting_time | +0.3381 | [-0.0048, +0.8192] |
| rule_wait_from_sjf | fixed_sjf | avg_turnaround_time | +0.2149 | [+0.0444, +0.4594] |
| rule_wait_from_sjf | fixed_sjf | avg_response_time | +0.2149 | [+0.0444, +0.4372] |
| rule_wait_from_sjf | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |

## Detection and recovery summary

Verdicts apply to interventions only: *accepted* is a verified change, *rejected* is a
change whose window closed no part of the trigger gap (a confirmed failure),
*inconclusive* is a change the window could neither verify nor condemn, and *held*
counts incidents that changed nothing.

| Arm | Episodes | Incidents | Interventions | Accepted | Rejected | Inconclusive | Unverified share | Inconclusive share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| fixed_sjf | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| fixed_round_robin | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| fixed_priority | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| detector_only | 210 | 137 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| rule_fairness | 210 | 150 | 150 | 2 | 31 | 117 | 0.987 | 0.780 |
| rule_wait | 210 | 141 | 141 | 12 | 0 | 129 | 0.915 | 0.915 |
| rule_wait_from_sjf | 210 | 173 | 4 | 0 | 1 | 3 | 1.000 | 0.750 |
| q_recovery | 1050 | 705 | 678 | 51 | 2 | 625 | 0.925 | 0.922 |
| q_recovery_from_sjf | 1050 | 865 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |

## Declared thresholds and what they were derived from

Both constants are derived from the *no-action control* on the training split (`healing_threshold_derivation.csv`, `healing_control_incidents.csv`); the final-test split never informed them.

| Quantity | Value |
|---|---:|
| control traces (training) | 6000 |
| degradation events declared with no intervention | 3737 |
| events per trace (false-trigger rate of the detector) | 0.623 |
| raw load closure p50 / p95 | 1.401 / 2.379 |
| unconditional `accept_closure_factor` (declared) | 2.4 |
| false-accept rate of the unconditional rule on the control | 0.046 |
| phase baseline R^2 / residual sd | 0.961 / 0.160 |
| residual p05 / p95 | -0.257 / +0.263 |
| derived accept / reject residual factor | +0.30 / -0.30 |
| declared accept / reject residual factor | +0.30 / -0.30 |
| false-accept / false-reject rate of the declared rule on the control | 0.034 / 0.031 |
| out-of-sample (validation control) false-accept / false-reject | 0.021 / 0.011 |

Congestion-gate sweep (what each `min_load_factor` removes from the control):

| min_load_factor | incidents per trace |
|---|---:|
| 0.00 | 1.517 |
| 0.50 | 1.475 |
| 0.80 | 1.052 |
| 1.00 | 0.858 |
| 1.25 | 0.042 |

## Counterfactual audit of interventions (matched replay)

Every audited intervention is re-run with the *same controller and the same workload*,
with that one incident forced to hold (`healing.controller.SelfHealingController`,
``forced_holds``). `matched_gain = hold_total_waiting - actual_total_waiting`: the total
waiting time the matched counterfactual accumulated minus the realized run's. Because the
counterfactual keeps healing after the forced hold, the difference is the marginal value of
that single decision, not of shutting the loop down (the older `hold_forever_gain` column
answers that second question and is kept for continuity).

| Arm | Audited interventions | Mean matched gain | Helped | Harmful | False recoveries | Verified and helped | Rejected but helpful | Verified but harmful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rule_fairness | 150 | -130.840 | 35 | 112 | 115 | 2 | 8 | 0 |
| rule_wait | 141 | +188.794 | 133 | 3 | 8 | 12 | 0 | 0 |
| rule_wait_from_sjf | 4 | -169.250 | 0 | 4 | 4 | 0 | 0 | 0 |
| q_recovery | 678 | +192.069 | 644 | 17 | 34 | 51 | 0 | 0 |

The same comparison for the second counterfactual, one row per *episode*: the recorded
configuration prefix with no further healing after that episode's first incident
(`verdict = episode_level`, `divergence_decision_index = -1`). Its denominator is episodes,
not interventions, which is why it is kept out of the table above.

| Arm | Episodes | Mean matched gain | Helped | Harmful |
|---|---:|---:|---:|---:|
| rule_fairness | 135 | -138.652 | 32 | 101 |
| rule_wait | 135 | +197.156 | 132 | 3 |
| rule_wait_from_sjf | 4 | -169.250 | 0 | 4 |
| q_recovery | 655 | +198.948 | 642 | 12 |

## Detector behaviour on controlled degradation scenarios (validation split)

Injected scenarios change the workload, never the configuration, and the arm is the
hold-only control, so every detection below is caused by the workload itself. `stable`
traces are the false-trigger probe; `burst_shock` multiplies the CPU time of every
process arriving after the injection instant, `arrival_storm` compresses the remaining
arrivals towards it. `detection_share` is the recall against the injected step change,
`incidents_per_trace` on stable traces is the false-trigger frequency, and the delay is
measured from the injection instant. A detection is only counted when it happens at or
after that instant.

| Scenario / family | Traces | Detected | Detection share | Incidents per stable trace | Median delay (time units) |
|---|---:|---:|---:|---:|---:|
| stable | 140 | 96 | 0.686 | 0.700 | 111.5 |
| burst_shock | 140 | 105 | 0.750 | 0.771 | 134.0 |
| arrival_storm | 140 | 79 | 0.564 | 0.571 | 81.0 |
| all | 420 | 280 | 0.667 | 0.681 | 113.5 |

## Learned-controller coverage

Encoder: 648 states x 8 actions = 5184 state-action pairs.
Visits are concentrated because the event supply is small: recovery events are rare by
construction, so coverage is reported as a distribution rather than a percentage.

| Seed | Visited states | Visited pairs | Pairs coverage | Incidents | Decisions in already-sampled states | Median visits/state | Max visits/pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7101 | 32 | 58 | 0.0112 | 772 | 0.890 | 4.5 | 240 |
| 7102 | 29 | 57 | 0.0110 | 753 | 0.884 | 4.0 | 251 |
| 7103 | 30 | 62 | 0.0120 | 769 | 0.909 | 6.0 | 240 |
| 7104 | 33 | 66 | 0.0127 | 771 | 0.900 | 6.0 | 262 |
| 7105 | 31 | 56 | 0.0108 | 745 | 0.862 | 5.0 | 258 |

Coverage against training length (does more training still buy new states?):

| Seed | Episodes | Visited states | Visited pairs | Incidents |
|---|---:|---:|---:|---:|
| 7101 | 300 | 23 | 41 | 193 |
| 7101 | 600 | 30 | 55 | 390 |
| 7101 | 900 | 32 | 58 | 576 |
| 7101 | 1200 | 32 | 58 | 772 |
| 7102 | 300 | 22 | 36 | 191 |
| 7102 | 600 | 26 | 50 | 381 |
| 7102 | 900 | 27 | 53 | 561 |
| 7102 | 1200 | 29 | 57 | 753 |
| 7103 | 300 | 20 | 44 | 190 |
| 7103 | 600 | 24 | 55 | 383 |
| 7103 | 900 | 29 | 61 | 574 |
| 7103 | 1200 | 30 | 62 | 769 |
| 7104 | 300 | 25 | 47 | 189 |
| 7104 | 600 | 31 | 62 | 384 |
| 7104 | 900 | 32 | 63 | 579 |
| 7104 | 1200 | 33 | 66 | 771 |
| 7105 | 300 | 24 | 38 | 188 |
| 7105 | 600 | 29 | 53 | 367 |
| 7105 | 900 | 30 | 55 | 556 |
| 7105 | 1200 | 31 | 56 | 745 |

## Training variability of the learned arms

| Arm | Training seed | Traces | Mean waiting | P95 waiting | Interventions | Accepted | Rejected | Inconclusive | Seed sd (mean waiting) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_recovery | 7101 | 210 | 122.425 | 251.871 | 0.662 | 0.052 | 0.000 | 0.610 | 0.146 |
| q_recovery | 7102 | 210 | 122.800 | 252.014 | 0.600 | 0.048 | 0.000 | 0.552 | 0.146 |
| q_recovery | 7103 | 210 | 122.506 | 251.957 | 0.648 | 0.048 | 0.000 | 0.600 | 0.146 |
| q_recovery | 7104 | 210 | 122.490 | 252.086 | 0.662 | 0.048 | 0.000 | 0.614 | 0.146 |
| q_recovery | 7105 | 210 | 122.599 | 251.952 | 0.657 | 0.048 | 0.010 | 0.600 | 0.146 |
| q_recovery_from_sjf | 7101 | 210 | 94.658 | 249.205 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7102 | 210 | 94.658 | 249.205 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7103 | 210 | 94.658 | 249.205 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7104 | 210 | 94.658 | 249.205 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7105 | 210 | 94.658 | 249.205 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Generated by `python main.py healing-study`.
