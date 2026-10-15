# Self-healing CPU scheduling: generated report

This file is generated from the machine-readable artifacts in this directory (`healing_metrics_*.csv`, `healing_incidents_*.csv`, `healing_audit.csv`, `healing_comparisons.csv`, `healing_summary.json`). Every number below appears in those files.

* split: **final_test**
* workloads per arm: 210
* training seeds: [7101, 7102, 7103, 7104, 7105]
* final-test seed: 21977
* bootstrap replicates: 1000

## Mean metrics by arm

| Arm | Mean waiting | P95 waiting | Mean response | Context switches | Incidents | Interventions |
|---|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 128.925 | 258.224 | 128.925 | 14.000 | 0.000 | 0.000 |
| fixed_sjf | 94.555 | 248.095 | 94.555 | 14.000 | 0.000 | 0.000 |
| fixed_round_robin | 217.143 | 309.838 | 26.393 | 75.062 | 0.000 | 0.000 |
| fixed_priority | 128.102 | 262.310 | 128.102 | 14.000 | 0.000 | 0.000 |
| detector_only | 128.925 | 258.224 | 128.925 | 14.000 | 0.657 | 0.000 |
| rule_fairness | 134.907 | 263.452 | 107.132 | 19.971 | 0.724 | 0.724 |
| rule_wait | 120.772 | 252.662 | 120.772 | 14.000 | 0.686 | 0.686 |
| rule_wait_from_sjf | 94.831 | 248.552 | 94.831 | 14.000 | 0.810 | 0.029 |
| q_recovery | 121.015 | 253.060 | 120.794 | 14.098 | 0.682 | 0.641 |
| q_recovery_from_sjf | 94.555 | 248.095 | 94.555 | 14.000 | 0.810 | 0.000 |

## Predeclared paired comparisons (family-stratified bootstrap)

| Target | Reference | Metric | Difference | 95% CI |
|---|---|---|---:|---|
| rule_wait | fixed_fcfs | avg_waiting_time | -8.1537 | [-9.2675, -6.9969] |
| rule_wait | fixed_fcfs | p95_waiting_time | -5.5619 | [-7.0857, -4.1429] |
| rule_wait | fixed_fcfs | max_waiting_time | -5.5619 | [-7.1002, -4.1045] |
| rule_wait | fixed_fcfs | avg_turnaround_time | -8.1537 | [-9.3042, -6.9134] |
| rule_wait | fixed_fcfs | avg_response_time | -8.1537 | [-9.2748, -7.0005] |
| rule_wait | fixed_fcfs | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_fcfs | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_fcfs | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | avg_waiting_time | +26.2168 | [+24.2789, +28.2194] |
| rule_wait | fixed_sjf | p95_waiting_time | +4.5667 | [+3.0713, +6.0198] |
| rule_wait | fixed_sjf | max_waiting_time | +4.5667 | [+2.9999, +6.1432] |
| rule_wait | fixed_sjf | avg_turnaround_time | +26.2168 | [+24.3514, +28.0480] |
| rule_wait | fixed_sjf | avg_response_time | +26.2168 | [+24.3187, +28.0696] |
| rule_wait | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | avg_waiting_time | -8.1537 | [-9.3115, -6.9648] |
| rule_wait | detector_only | p95_waiting_time | -5.5619 | [-6.9762, -4.1664] |
| rule_wait | detector_only | max_waiting_time | -5.5619 | [-7.0574, -4.2423] |
| rule_wait | detector_only | avg_turnaround_time | -8.1537 | [-9.2811, -7.0278] |
| rule_wait | detector_only | avg_response_time | -8.1537 | [-9.2307, -6.8890] |
| rule_wait | detector_only | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_fairness | fixed_fcfs | avg_waiting_time | +5.9813 | [+4.5340, +7.4869] |
| rule_fairness | fixed_fcfs | p95_waiting_time | +5.2286 | [+3.4088, +7.0198] |
| rule_fairness | fixed_fcfs | max_waiting_time | +5.2286 | [+3.4852, +7.0382] |
| rule_fairness | fixed_fcfs | avg_turnaround_time | +5.9813 | [+4.5628, +7.3322] |
| rule_fairness | fixed_fcfs | avg_response_time | -21.7933 | [-24.7027, -18.8068] |
| rule_fairness | fixed_fcfs | cpu_utilization | -1.8433 | [-2.0506, -1.6488] |
| rule_fairness | fixed_fcfs | throughput | -0.0013 | [-0.0015, -0.0011] |
| rule_fairness | fixed_fcfs | context_switches | +5.9714 | [+5.2475, +6.6668] |
| q_recovery | fixed_fcfs | avg_waiting_time | -7.9107 | [-9.1901, -6.8647] |
| q_recovery | fixed_fcfs | p95_waiting_time | -5.1638 | [-6.5261, -3.8270] |
| q_recovery | fixed_fcfs | max_waiting_time | -5.1638 | [-6.5888, -3.9417] |
| q_recovery | fixed_fcfs | avg_turnaround_time | -7.9107 | [-9.0468, -6.6465] |
| q_recovery | fixed_fcfs | avg_response_time | -8.1310 | [-9.2567, -7.0404] |
| q_recovery | fixed_fcfs | cpu_utilization | -0.0511 | [-0.1425, +0.0000] |
| q_recovery | fixed_fcfs | throughput | -0.0001 | [-0.0002, +0.0000] |
| q_recovery | fixed_fcfs | context_switches | +0.0981 | [+0.0000, +0.2934] |
| q_recovery | fixed_sjf | avg_waiting_time | +26.4597 | [+24.5222, +28.3970] |
| q_recovery | fixed_sjf | p95_waiting_time | +4.9648 | [+3.4258, +6.4505] |
| q_recovery | fixed_sjf | max_waiting_time | +4.9648 | [+3.4950, +6.5842] |
| q_recovery | fixed_sjf | avg_turnaround_time | +26.4597 | [+24.5261, +28.4697] |
| q_recovery | fixed_sjf | avg_response_time | +26.2395 | [+24.2712, +28.2653] |
| q_recovery | fixed_sjf | cpu_utilization | -0.0511 | [-0.1484, +0.0000] |
| q_recovery | fixed_sjf | throughput | -0.0001 | [-0.0002, +0.0000] |
| q_recovery | fixed_sjf | context_switches | +0.0981 | [+0.0000, +0.3000] |
| q_recovery | rule_wait | avg_waiting_time | +0.2429 | [-0.0447, +0.5977] |
| q_recovery | rule_wait | p95_waiting_time | +0.3981 | [-0.0614, +1.0343] |
| q_recovery | rule_wait | max_waiting_time | +0.3981 | [-0.0782, +1.0811] |
| q_recovery | rule_wait | avg_turnaround_time | +0.2429 | [-0.0664, +0.5799] |
| q_recovery | rule_wait | avg_response_time | +0.0227 | [-0.4160, +0.3870] |
| q_recovery | rule_wait | cpu_utilization | -0.0511 | [-0.1445, +0.0000] |
| q_recovery | rule_wait | throughput | -0.0001 | [-0.0002, +0.0000] |
| q_recovery | rule_wait | context_switches | +0.0981 | [+0.0000, +0.3001] |
| q_recovery | detector_only | avg_waiting_time | -7.9107 | [-9.0334, -6.8270] |
| q_recovery | detector_only | p95_waiting_time | -5.1638 | [-6.5906, -3.9313] |
| q_recovery | detector_only | max_waiting_time | -5.1638 | [-6.5899, -3.7837] |
| q_recovery | detector_only | avg_turnaround_time | -7.9107 | [-9.0779, -6.7278] |
| q_recovery | detector_only | avg_response_time | -8.1310 | [-9.3041, -6.9305] |
| q_recovery | detector_only | cpu_utilization | -0.0511 | [-0.1408, +0.0000] |
| q_recovery | detector_only | throughput | -0.0001 | [-0.0002, +0.0000] |
| q_recovery | detector_only | context_switches | +0.0981 | [+0.0000, +0.2895] |
| q_recovery_from_sjf | fixed_sjf | avg_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | p95_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | max_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | avg_turnaround_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | avg_response_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | throughput | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | avg_waiting_time | +0.2762 | [+0.0619, +0.5709] |
| rule_wait_from_sjf | fixed_sjf | p95_waiting_time | +0.4571 | [+0.0476, +1.1151] |
| rule_wait_from_sjf | fixed_sjf | max_waiting_time | +0.4571 | [+0.0000, +1.0480] |
| rule_wait_from_sjf | fixed_sjf | avg_turnaround_time | +0.2762 | [+0.0618, +0.5709] |
| rule_wait_from_sjf | fixed_sjf | avg_response_time | +0.2762 | [+0.0695, +0.5587] |
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
| detector_only | 210 | 138 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |
| rule_fairness | 210 | 152 | 152 | 3 | 16 | 133 | 0.980 | 0.875 |
| rule_wait | 210 | 144 | 144 | 6 | 3 | 135 | 0.958 | 0.938 |
| rule_wait_from_sjf | 210 | 170 | 6 | 3 | 0 | 3 | 0.500 | 0.500 |
| q_recovery | 1050 | 716 | 673 | 22 | 13 | 638 | 0.967 | 0.948 |
| q_recovery_from_sjf | 1050 | 850 | 0 | 0 | 0 | 0 | 0.000 | 0.000 |

## Declared thresholds and what they were derived from

Both constants are derived from the *no-action control* on the training split (`healing_threshold_derivation.csv`, `healing_control_incidents.csv`); the final-test split never informed them.

| Quantity | Value |
|---|---:|
| control traces (training) | 6000 |
| degradation events declared with no intervention | 3737 |
| events per trace (false-trigger rate of the detector) | 0.623 |
| raw load closure p50 / p95 | 1.406 / 2.375 |
| unconditional `accept_closure_factor` (declared) | 2.4 |
| false-accept rate of the unconditional rule on the control | 0.045 |
| phase baseline R^2 / residual sd | 0.819 / 0.339 |
| residual p05 / p95 | -0.408 / +0.621 |
| derived accept / reject residual factor | +0.65 / -0.45 |
| declared accept / reject residual factor | +0.65 / -0.45 |
| false-accept / false-reject rate of the declared rule on the control | 0.047 / 0.039 |
| out-of-sample (validation control) false-accept / false-reject | 0.031 / 0.061 |

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
| rule_fairness | 152 | -126.191 | 29 | 117 | 123 | 0 | 6 | 1 |
| rule_wait | 144 | +178.778 | 136 | 5 | 8 | 6 | 2 | 0 |
| rule_wait_from_sjf | 6 | -145.000 | 0 | 6 | 6 | 0 | 0 | 3 |
| q_recovery | 673 | +185.143 | 637 | 27 | 36 | 18 | 9 | 4 |

The same comparison for the second counterfactual, one row per *episode*: the recorded
configuration prefix with no further healing after that episode's first incident
(`verdict = episode_level`, `divergence_decision_index = -1`). Its denominator is episodes,
not interventions, which is why it is kept out of the table above.

| Arm | Episodes | Mean matched gain | Helped | Harmful |
|---|---:|---:|---:|---:|
| rule_fairness | 136 | -138.537 | 28 | 106 |
| rule_wait | 136 | +188.853 | 131 | 5 |
| rule_wait_from_sjf | 6 | -145.000 | 0 | 6 |
| q_recovery | 644 | +193.469 | 623 | 21 |

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
| 7101 | 32 | 60 | 0.0116 | 773 | 0.890 | 5.5 | 235 |
| 7102 | 29 | 57 | 0.0110 | 754 | 0.882 | 4.0 | 250 |
| 7103 | 30 | 64 | 0.0123 | 772 | 0.912 | 6.5 | 237 |
| 7104 | 32 | 68 | 0.0131 | 771 | 0.901 | 6.0 | 252 |
| 7105 | 30 | 60 | 0.0116 | 745 | 0.875 | 5.5 | 256 |

Coverage against training length (does more training still buy new states?):

| Seed | Episodes | Visited states | Visited pairs | Incidents |
|---|---:|---:|---:|---:|
| 7101 | 300 | 23 | 41 | 193 |
| 7101 | 600 | 30 | 55 | 390 |
| 7101 | 900 | 32 | 59 | 576 |
| 7101 | 1200 | 32 | 60 | 773 |
| 7102 | 300 | 22 | 36 | 191 |
| 7102 | 600 | 26 | 48 | 381 |
| 7102 | 900 | 27 | 53 | 562 |
| 7102 | 1200 | 29 | 57 | 754 |
| 7103 | 300 | 20 | 43 | 190 |
| 7103 | 600 | 24 | 52 | 384 |
| 7103 | 900 | 29 | 61 | 578 |
| 7103 | 1200 | 30 | 64 | 772 |
| 7104 | 300 | 24 | 47 | 189 |
| 7104 | 600 | 30 | 62 | 385 |
| 7104 | 900 | 31 | 65 | 579 |
| 7104 | 1200 | 32 | 68 | 771 |
| 7105 | 300 | 24 | 38 | 188 |
| 7105 | 600 | 29 | 54 | 368 |
| 7105 | 900 | 29 | 56 | 556 |
| 7105 | 1200 | 30 | 60 | 745 |

## Training variability of the learned arms

| Arm | Training seed | Traces | Mean waiting | P95 waiting | Interventions | Accepted | Rejected | Inconclusive | Seed sd (mean waiting) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_recovery | 7101 | 210 | 120.985 | 252.771 | 0.619 | 0.024 | 0.005 | 0.590 | 0.254 |
| q_recovery | 7102 | 210 | 121.164 | 253.257 | 0.610 | 0.019 | 0.010 | 0.581 | 0.254 |
| q_recovery | 7103 | 210 | 120.755 | 252.657 | 0.676 | 0.024 | 0.014 | 0.638 | 0.254 |
| q_recovery | 7104 | 210 | 120.803 | 253.352 | 0.652 | 0.019 | 0.014 | 0.619 | 0.254 |
| q_recovery | 7105 | 210 | 121.365 | 253.262 | 0.648 | 0.019 | 0.019 | 0.610 | 0.254 |
| q_recovery_from_sjf | 7101 | 210 | 94.555 | 248.095 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7102 | 210 | 94.555 | 248.095 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7103 | 210 | 94.555 | 248.095 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7104 | 210 | 94.555 | 248.095 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| q_recovery_from_sjf | 7105 | 210 | 94.555 | 248.095 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Generated by `python main.py healing-study`.
