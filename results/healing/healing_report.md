# Self-healing CPU scheduling: generated report

This file is generated from the machine-readable artifacts in this directory (`healing_metrics_*.csv`, `healing_incidents_*.csv`, `healing_audit.csv`, `healing_comparisons.csv`, `healing_summary.json`). Every number below appears in those files.

* split: **final_test**
* workloads per arm: 210
* training seeds: [7101, 7102, 7103, 7104, 7105]
* final-test seed: 20719
* bootstrap replicates: 1000

## Mean metrics by arm

| Arm | Mean waiting | P95 waiting | Mean response | Context switches | Incidents | Interventions |
|---|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 128.878 | 256.014 | 128.878 | 14.000 | 0.000 | 0.000 |
| fixed_sjf | 95.525 | 248.957 | 95.525 | 14.000 | 0.000 | 0.000 |
| fixed_round_robin | 218.920 | 310.043 | 26.251 | 75.186 | 0.000 | 0.000 |
| fixed_priority | 128.559 | 261.024 | 128.559 | 14.000 | 0.000 | 0.000 |
| detector_only | 128.878 | 256.014 | 128.878 | 14.000 | 0.690 | 0.000 |
| rule_fairness | 135.597 | 263.224 | 107.292 | 20.471 | 0.762 | 0.762 |
| rule_wait | 121.176 | 251.348 | 121.176 | 14.000 | 0.705 | 0.705 |
| rule_wait_from_sjf | 95.767 | 249.424 | 95.767 | 14.000 | 0.838 | 0.029 |
| q_recovery | 122.262 | 252.326 | 121.690 | 14.234 | 0.706 | 0.666 |
| q_recovery_from_sjf | 95.525 | 248.957 | 95.525 | 14.000 | 0.838 | 0.000 |

## Predeclared paired comparisons (family-stratified bootstrap)

| Target | Reference | Metric | Difference | 95% CI |
|---|---|---|---:|---|
| rule_wait | fixed_fcfs | avg_waiting_time | -7.7022 | [-8.7120, -6.6656] |
| rule_wait | fixed_fcfs | p95_waiting_time | -4.6667 | [-6.0239, -3.2667] |
| rule_wait | fixed_fcfs | max_waiting_time | -4.6667 | [-6.1627, -3.3570] |
| rule_wait | fixed_fcfs | avg_turnaround_time | -7.7022 | [-8.6465, -6.6992] |
| rule_wait | fixed_fcfs | avg_response_time | -7.7022 | [-8.6972, -6.7699] |
| rule_wait | fixed_fcfs | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_fcfs | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_fcfs | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | avg_waiting_time | +25.6508 | [+23.6715, +27.7082] |
| rule_wait | fixed_sjf | p95_waiting_time | +2.3905 | [+1.0186, +3.7668] |
| rule_wait | fixed_sjf | max_waiting_time | +2.3905 | [+0.9140, +3.8386] |
| rule_wait | fixed_sjf | avg_turnaround_time | +25.6508 | [+23.7375, +27.6311] |
| rule_wait | fixed_sjf | avg_response_time | +25.6508 | [+23.7685, +27.7215] |
| rule_wait | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | avg_waiting_time | -7.7022 | [-8.6823, -6.6812] |
| rule_wait | detector_only | p95_waiting_time | -4.6667 | [-6.0193, -3.3332] |
| rule_wait | detector_only | max_waiting_time | -4.6667 | [-6.1952, -3.3469] |
| rule_wait | detector_only | avg_turnaround_time | -7.7022 | [-8.6920, -6.6959] |
| rule_wait | detector_only | avg_response_time | -7.7022 | [-8.6858, -6.7145] |
| rule_wait | detector_only | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait | detector_only | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_fairness | fixed_fcfs | avg_waiting_time | +6.7190 | [+5.4167, +8.1071] |
| rule_fairness | fixed_fcfs | p95_waiting_time | +7.2095 | [+5.3807, +8.9480] |
| rule_fairness | fixed_fcfs | max_waiting_time | +7.2095 | [+5.4045, +8.9196] |
| rule_fairness | fixed_fcfs | avg_turnaround_time | +6.7190 | [+5.3622, +8.1291] |
| rule_fairness | fixed_fcfs | avg_response_time | -21.5857 | [-24.3246, -18.6631] |
| rule_fairness | fixed_fcfs | cpu_utilization | -2.0057 | [-2.2248, -1.7650] |
| rule_fairness | fixed_fcfs | throughput | -0.0014 | [-0.0016, -0.0012] |
| rule_fairness | fixed_fcfs | context_switches | +6.4714 | [+5.7330, +7.2430] |
| q_recovery | fixed_fcfs | avg_waiting_time | -6.6159 | [-8.0274, -5.1157] |
| q_recovery | fixed_fcfs | p95_waiting_time | -3.6886 | [-5.1291, -2.1700] |
| q_recovery | fixed_fcfs | max_waiting_time | -3.6886 | [-5.2106, -2.1314] |
| q_recovery | fixed_fcfs | avg_turnaround_time | -6.6159 | [-8.0027, -5.0649] |
| q_recovery | fixed_fcfs | avg_response_time | -7.1877 | [-8.3229, -5.8998] |
| q_recovery | fixed_fcfs | cpu_utilization | -0.1120 | [-0.2026, -0.0422] |
| q_recovery | fixed_fcfs | throughput | -0.0001 | [-0.0002, -0.0000] |
| q_recovery | fixed_fcfs | context_switches | +0.2343 | [+0.0867, +0.4391] |
| q_recovery | fixed_sjf | avg_waiting_time | +26.7371 | [+24.4593, +28.9181] |
| q_recovery | fixed_sjf | p95_waiting_time | +3.3686 | [+1.7998, +5.0421] |
| q_recovery | fixed_sjf | max_waiting_time | +3.3686 | [+1.7111, +5.1382] |
| q_recovery | fixed_sjf | avg_turnaround_time | +26.7371 | [+24.6203, +28.9978] |
| q_recovery | fixed_sjf | avg_response_time | +26.1653 | [+24.1049, +28.2978] |
| q_recovery | fixed_sjf | cpu_utilization | -0.1120 | [-0.2089, -0.0411] |
| q_recovery | fixed_sjf | throughput | -0.0001 | [-0.0002, -0.0000] |
| q_recovery | fixed_sjf | context_switches | +0.2343 | [+0.0895, +0.4343] |
| q_recovery | rule_wait | avg_waiting_time | +1.0863 | [+0.1177, +2.5119] |
| q_recovery | rule_wait | p95_waiting_time | +0.9781 | [-0.0638, +2.2677] |
| q_recovery | rule_wait | max_waiting_time | +0.9781 | [-0.0756, +2.3610] |
| q_recovery | rule_wait | avg_turnaround_time | +1.0863 | [+0.0757, +2.4801] |
| q_recovery | rule_wait | avg_response_time | +0.5145 | [-0.2979, +1.5960] |
| q_recovery | rule_wait | cpu_utilization | -0.1120 | [-0.2048, -0.0407] |
| q_recovery | rule_wait | throughput | -0.0001 | [-0.0002, -0.0000] |
| q_recovery | rule_wait | context_switches | +0.2343 | [+0.0847, +0.4257] |
| q_recovery | detector_only | avg_waiting_time | -6.6159 | [-7.9916, -5.0638] |
| q_recovery | detector_only | p95_waiting_time | -3.6886 | [-5.0783, -2.1818] |
| q_recovery | detector_only | max_waiting_time | -3.6886 | [-5.1403, -2.1808] |
| q_recovery | detector_only | avg_turnaround_time | -6.6159 | [-7.9977, -4.9690] |
| q_recovery | detector_only | avg_response_time | -7.1877 | [-8.2814, -5.8699] |
| q_recovery | detector_only | cpu_utilization | -0.1120 | [-0.2100, -0.0427] |
| q_recovery | detector_only | throughput | -0.0001 | [-0.0002, -0.0000] |
| q_recovery | detector_only | context_switches | +0.2343 | [+0.0714, +0.4381] |
| q_recovery_from_sjf | fixed_sjf | avg_waiting_time | -0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | p95_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | max_waiting_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | avg_turnaround_time | +0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | avg_response_time | -0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | cpu_utilization | -0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | throughput | -0.0000 | [+0.0000, +0.0000] |
| q_recovery_from_sjf | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | avg_waiting_time | +0.2416 | [+0.0447, +0.5407] |
| rule_wait_from_sjf | fixed_sjf | p95_waiting_time | +0.4667 | [+0.0998, +0.9289] |
| rule_wait_from_sjf | fixed_sjf | max_waiting_time | +0.4667 | [+0.0857, +0.9714] |
| rule_wait_from_sjf | fixed_sjf | avg_turnaround_time | +0.2416 | [+0.0431, +0.5239] |
| rule_wait_from_sjf | fixed_sjf | avg_response_time | +0.2416 | [+0.0495, +0.5040] |
| rule_wait_from_sjf | fixed_sjf | cpu_utilization | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | throughput | +0.0000 | [+0.0000, +0.0000] |
| rule_wait_from_sjf | fixed_sjf | context_switches | +0.0000 | [+0.0000, +0.0000] |

## Detection and recovery summary

| Arm | Episodes | Incidents | Interventions | Accepted | Rolled back | Censored | In-run false-recovery rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_fcfs | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| fixed_sjf | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| fixed_round_robin | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| fixed_priority | 210 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| detector_only | 210 | 145 | 0 | 0 | 0 | 0 | 0.000 |
| rule_fairness | 210 | 160 | 160 | 3 | 154 | 3 | 0.963 |
| rule_wait | 210 | 148 | 148 | 10 | 126 | 12 | 0.851 |
| rule_wait_from_sjf | 210 | 176 | 6 | 3 | 3 | 0 | 0.500 |
| q_recovery | 1050 | 741 | 699 | 32 | 623 | 44 | 0.891 |
| q_recovery_from_sjf | 1050 | 880 | 0 | 0 | 0 | 0 | 0.000 |

## Declared thresholds and what they were derived from

Both constants are derived from the *no-action control* on the training split (`healing_threshold_derivation.csv`, `healing_control_incidents.csv`); the final-test split never informed them.

| Quantity | Value |
|---|---:|
| control traces (training) | 6000 |
| degradation events declared with no intervention | 3737 |
| events per trace (false-trigger rate of the detector) | 0.623 |
| null load closure p50 | 1.406 |
| null load closure p95 | 2.375 |
| derived `accept_closure_factor` | 2.4 |
| declared `accept_closure_factor` | 2.4 |
| false-accept rate of the declared rule on the control | 0.045 |

Congestion-gate sweep (what each `min_load_factor` removes from the control):

| min_load_factor | incidents per trace |
|---|---:|
| 0.00 | 1.517 |
| 0.50 | 1.475 |
| 0.80 | 1.052 |
| 1.00 | 0.858 |
| 1.25 | 0.042 |

## Counterfactual replay audit of interventions

An intervention's gain is `hold_total_waiting - actual_total_waiting`: the total waiting time the same trace would have accumulated if the controller had held its pre-incident configuration from that point on, minus what the realized run accumulated.

| Arm | Audited interventions | Mean gain | Helped | False recoveries | Verified and helped | Rejected but helpful |
|---|---:|---:|---:|---:|---:|---:|
| rule_fairness | 297 | -146.896 | 59 | 238 | 1 | 30 |
| rule_wait | 285 | +170.375 | 271 | 14 | 9 | 123 |
| rule_wait_from_sjf | 12 | -126.833 | 0 | 12 | 0 | 0 |
| q_recovery | 1354 | +153.880 | 1194 | 160 | 27 | 559 |

Generated by `python main.py healing-study`.
