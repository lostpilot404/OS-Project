# Evaluation results

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

Everything below is produced by one command,

```bash
.venv/bin/python main.py healing-study
```

and every number is traceable to a file under `results/healing/` (named after each table).
The run reported here used the frozen design: training seeds `7101–7105` (1200 traces each),
validation seed `8201` (140 traces), **held-out final-test seed `20719`** (210 traces), 1000
bootstrap replicates at seed `104729`. The study's own training/derivation phase reported
`results/healing/healing_summary.json` and, in total, evaluated 550 hold-out traces for the
fixed/rule arms and 1050 for each learned arm (five seeds).

Hardware/software: Python 3.11.2, NumPy 2.4.6, pandas 3.0.6, Matplotlib 3.11.2, pytest 9.1.1
(recorded in `healing_summary.json` under `software`). Total wall clock of the frozen run:
28 s.

---

## 1. Held-out comparison (final-test seed 20719)

Mean of 210 traces per arm (learned arms: 1050 traces, five training seeds). Waiting,
response and turnaround times are in simulated time units; utilization in percent.

| Arm | Mean waiting | P95 waiting | Max waiting | Mean response | Mean turnaround | CPU util. | Throughput | Context switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_round_robin` | 218.92 | 310.04 | 310.04 | 26.25 | 237.87 | 78.318 | 0.0685 | 75.19 |
| `rule_fairness` | 135.60 | 263.22 | 263.22 | 107.29 | 154.54 | 90.279 | 0.0762 | 20.47 |
| `fixed_fcfs` | 128.88 | 256.01 | 256.01 | 128.88 | 147.82 | 92.285 | 0.0776 | 14.00 |
| `detector_only` (control) | 128.88 | 256.01 | 256.01 | 128.88 | 147.82 | 92.285 | 0.0776 | 14.00 |
| `fixed_priority` | 128.56 | 261.02 | 261.02 | 128.56 | 147.51 | 92.285 | 0.0776 | 14.00 |
| `q_recovery` (5 seeds) | 122.26 | 252.33 | 252.33 | 121.69 | 141.21 | 92.173 | 0.0774 | 14.23 |
| `rule_wait` | 121.18 | 251.35 | 251.35 | 121.18 | 140.12 | 92.285 | 0.0776 | 14.00 |
| `rule_wait_from_sjf` | 95.77 | 249.42 | 249.42 | 95.77 | 114.71 | 92.285 | 0.0776 | 14.00 |
| `q_recovery_from_sjf` (5 seeds) | 95.53 | 248.96 | 248.96 | 95.53 | 114.47 | 92.285 | 0.0776 | 14.00 |
| `fixed_sjf` | **95.53** | **248.96** | **248.96** | **95.53** | **114.47** | 92.285 | 0.0776 | 14.00 |

Reading of the table:

* the **control is exactly FCFS** (128.88 / 256.01 / 14.00 switches, to the last digit): the
  monitoring, detection and diagnosis stages change nothing when the policy never intervenes;
* the **waiting-time rule and the learned controller both beat the control** (121.18 and
  122.26 vs 128.88) and **both lose to fixed SJF**;
* the **fairness rule is worse than the control** (135.60): its starvation→Round-Robin
  mapping buys responsiveness (response 107.29 vs 128.88) at the cost of waiting time and 6.5
  extra switches per trace;
* started from SJF, the loop is **inert and harmless**: `q_recovery_from_sjf` reproduces SJF
  exactly (0 interventions), `rule_wait_from_sjf` intervenes 0.03 times per trace and costs
  0.24 wait units.

## 2. Paired comparisons with uncertainty

Family-stratified crossed bootstrap over identical workloads, 1000 replicates, seed 104729
(`healing_comparisons.csv`; lower is better for waiting time, so a negative difference favours
the target).

| Target | Reference | Metric | Difference | 95 % CI |
|---|---|---|---:|---|
| `rule_wait` | `detector_only` | mean waiting | **−7.70** | [−8.69, −6.71] |
| `rule_wait` | `fixed_fcfs` | mean waiting | **−7.70** | [−8.71, −6.67] |
| `rule_wait` | `fixed_sjf` | mean waiting | +25.65 | [+23.67, +27.71] |
| `q_recovery` | `fixed_fcfs` | mean waiting | **−6.62** | [−8.03, −5.12] |
| `q_recovery` | `fixed_sjf` | mean waiting | +26.74 | [+24.46, +28.92] |
| `q_recovery` | `rule_wait` | mean waiting | +1.09 | [+0.12, +2.51] |
| `rule_fairness` | `fixed_fcfs` | mean waiting | +6.72 | [+5.42, +8.11] |
| `q_recovery_from_sjf` | `fixed_sjf` | mean waiting | 0.0000 | [0.00, 0.00] |
| `rule_wait_from_sjf` | `fixed_sjf` | mean waiting | +0.24 | [+0.04, +0.54] |

Every interval excludes zero except the deliberately identical pair, so the ordering
`fixed_sjf < rule_wait < q_recovery ≈ rule_wait` and `rule_fairness > fixed_fcfs` are all
resolved at this design. The remaining tolerance in the learned arm (five seeds) is visible in
the `q_recovery − rule_wait` interval, which just excludes zero in favour of the *rule*.

## 3. Counterfactual replay audit (the causal measurement)

Each intervention is replayed in the same trace with the recorded prefix and then held. The
gain is `hold_total_waiting − actual_total_waiting` in wait units; a negative gain means the
adaptation made the trace worse. A **false recovery** is an intervention with gain ≤ 0
(`healing_audit.csv`).

| Arm | Audited interventions | Mean gain | Median gain | Helped | False recoveries | False-recovery rate |
|---|---:|---:|---:|---:|---:|---:|
| `rule_fairness` | 297 | −146.90 | −94.00 | 59 | 238 | **80.1 %** |
| `rule_wait` | 285 | +170.38 | +142.00 | 271 | 14 | **4.9 %** |
| `q_recovery` | 1354 | +153.88 | +139.00 | 1194 | 160 | **11.8 %** |
| `rule_wait_from_sjf` | 12 | −126.83 | −103.00 | 0 | 12 | **100 %** |

The same table shows what the *in-run* verifier did with those interventions:

| Arm | Verified and helped | Verified but harmful | Rejected but helpful | Rejected and harmful | In-run rejection rate | False-reject rate |
|---|---:|---:|---:|---:|---:|---:|
| `rule_fairness` | 1 | 2 | 30 | 124 | 51.9 % | 19.5 % |
| `rule_wait` | 9 | 1 | 123 | 3 | 44.2 % | **97.6 %** |
| `q_recovery` | 27 | 5 | 559 | 64 | 46.0 % | **89.7 %** |
| `rule_wait_from_sjf` | 0 | 3 | 0 | 3 | 25.0 % | 0.0 % |

**This is the central honest finding of the study.** The recovery *actions* mostly help: the
counterfactual audit says 95.1 % of the waiting-time rule's interventions and 88.2 % of the
learned controller's interventions reduced total waiting time. But the **in-run verification
rule rejects 97.6 % (rule) and 89.7 % (learned) of those helpful interventions**, because a
trigger is a local maximum of the monitored statistic and the statistic decays afterwards
whether or not anything was changed. The loop therefore undoes most of the improvements it
finds. It is only because the changed configuration is in force for the whole verification
window (5–14 decisions) before the rollback that the arms still finish ahead of FCFS.

Correspondingly, the fairness rule's damage is under-detected: its Round-Robin interventions
cost a mean 147 wait units each, and 80.1 % of them are false recoveries.

## 4. Detection behaviour and false triggers

The no-action control (`detector_only`) declares events without ever acting, so every event it
declares is either the workload's own behaviour or noise. Held-out split:

| Arm | Episodes | Incidents | Incidents / trace | Interventions | Accepted | Rolled back | Held (no change) | Censored | Episodes hitting the safe fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `detector_only` | 210 | 145 | 0.690 | 0 | 0 | 0 | 145 | 0 | 0 |
| `rule_fairness` | 210 | 160 | 0.762 | 160 | 3 | 154 | 0 | 3 | 20 |
| `rule_wait` | 210 | 148 | 0.705 | 148 | 10 | 126 | 0 | 12 | 0 |
| `rule_wait_from_sjf` | 210 | 176 | 0.838 | 6 | 3 | 3 | 170 | 0 | 0 |
| `q_recovery` | 1050 | 741 | 0.706 | 699 | 32 | 623 | 42 | 44 | 4 |
| `q_recovery_from_sjf` | 1050 | 880 | 0.838 | 0 | 0 | 0 | 880 | 0 | 0 |

Diagnoses on the control: fairness starvation 125, head-of-line blocking 8, priority-ordering
anomaly 8, queue congestion 4 (tail dispersion and `unexplained` never fired). Actions chosen
on the held-out split: the fairness rule chose `set_round_robin` 144 times, `set_sjf` 8,
`set_priority` 8; the waiting rule chose `set_sjf` 140, `set_priority` 8; the learned
controller chose `set_sjf` 563, `set_priority` 97, `hold` 42, `set_round_robin` 39.

Two further observations:

* **The learned controller learned to hold from SJF**: `q_recovery_from_sjf` opened 880
  incidents and intervened zero times — the learned values prefer holding when the
  configuration is already the strongest one available, which is the intended behaviour of a
  cost-aware reward.
* **The detector fires at all on traces that are simply hard**: 0.69 events per trace with no
  intervention, most of them (`fairness_starvation`) early, while the initial burst backlog is
  at its worst. The congestion gate (`min_load_factor = 1.0`) already removes 45 % of those
  events (see §5); what remains is the price of a reference-based detector on traces that
  begin with a backlog.

## 5. Derived thresholds

`healing_threshold_derivation.csv` (from the **training** control only):

| Quantity | Value |
|---|---:|
| control traces | 6000 |
| degradation events declared (no intervention) | 3737 |
| events per trace | 0.623 |
| null closure p50 | 1.406 |
| null closure p95 | 2.375 |
| derived `accept_closure_factor` | 2.4 |
| declared `accept_closure_factor` | 2.4 |
| false-accept rate of the declared rule on the control | 0.045 |

Congestion-gate sweep (600 control traces each): `min_load_factor` 0.00 → 1.517 events/trace,
0.50 → 1.475, 0.80 → 1.052, 1.00 → 0.858, 1.25 → 0.042. The declared value is 1.0.

The declared verification threshold is therefore calibrated to *reject* rather than to
*accept*: under the control it certifies at most 5 % of no-change windows as successes. The
cost of that conservatism is measured in §3 as a 90–98 % false-reject rate.

## 6. Reward versus measured truth

`healing_action_probe.csv` replays, on the validation split, **every** action at **every**
detected incident and compares what the controller would have recorded as reward with the
measured effect on total waiting time (per process). 21 incidents × 8 actions = 168 rows.

| Diagnosis | Action | Recorded reward (band units) | Measured true effect (wait units/process) |
|---|---|---:|---:|
| fairness starvation | `set_sjf` | 1.007 | **+3.85** |
| fairness starvation | `set_priority` | 0.865 | +0.46 |
| fairness starvation | `hold`, `set_fcfs`, `shrink_quantum`, `grow_quantum`, `revert` | 0.884 | 0.00 |
| fairness starvation | `set_round_robin` | 0.683 | **−3.04** |
| head-of-line blocking | `set_round_robin` | 0.872 | +5.53 |
| head-of-line blocking | `set_sjf` | 1.068 | +1.67 |

Across all probed rows, Pearson correlation between the recorded reward and the measured true
effect is **0.10**, and the reward's argmax agrees with the truth's argmax in **15 of 21**
incidents. The reward does capture the main contrast — it ranks `set_sjf` above every no-op
and `set_round_robin` below them — but it is a weak proxy for the magnitude of the effect, and
it is why the learned improvements are small.

## 7. What the learned controller saw (coverage)

| Training seed | Encoder states | Visited states | Visited state-action pairs | Coverage | Training episodes | Training incidents | Training interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7101 | 648 | 33 | 57 | 1.10 % | 1200 | 774 | 697 |
| 7102 | 648 | 30 | 55 | 1.06 % | 1200 | 758 | 669 |
| 7103 | 648 | 30 | 63 | 1.22 % | 1200 | 774 | 679 |
| 7104 | 648 | 32 | 63 | 1.22 % | 1200 | 769 | 645 |
| 7105 | 648 | 31 | 57 | 1.10 % | 1200 | 746 | 653 |

The learned controller is trained on ~750 incidents per seed but touches about 1 % of its
table. Any claim about the RL arm has to be read with that in mind: the state space is larger
than the trace budget of this study.

## 8. Worked demonstration trace

`healing_demo.json` (chosen by a rule declared before the run: learned arm on the validation
split, preferring an episode with both an accepted and a rolled-back verdict; the episode is
identified by arm **and** training seed, and `_demo_payload` refuses to write the artifact if
the replay does not reproduce the recorded incident count).

| Step | Observed |
|---|---|
| workload | `long_jobs`, validation repetition 4, fingerprint `1787…`, arm `q_recovery` seed 7103 |
| detection | incident at decision 6, `degradation_detected`, window mean above `reference + 1.25·band`, evidence retained |
| diagnosis | `head_of_line_blocking` |
| decision | `set_priority` — FCFS → Priority (subject to the mask; `hold` was also offered) |
| verification | window closes at t = 235 with `closure_fraction = 2.60` against the declared `2.40` |
| verdict | **accepted**, reward +0.136, configuration kept to the end of the trace |
| result | mean waiting 152.53 over 15 processes, 14 context switches |

The same file records the full decision log (`decisions`), so the trace can be re-derived
decision by decision.

## 9. Preserved historical results (not the project's headline results)

These come from the preserved offline study and the non-healing runtime appendix; they are
reproducible with `python main.py study` and `python main.py runtime-experiment` and live in
`results/study/` and `results/runtime/`. They are reported here only so the two systems can be
told apart.

* Offline policy selection (seed 19501): risk-aware Q mean cost 0.7953 / regret 0.0574,
  original Q 0.7963 / 0.0524, fingerprint rule 0.7752 / 0.0372, SJF 0.8419 / 0.1040,
  random 0.9778 / 0.2398.
* Runtime appendix (seed 19301, 210 traces): mean waiting — FCFS 127.797, SJF 93.345,
  Round Robin 215.729, Priority 129.207, causal heuristic 203.404, runtime Q 104.789;
  runtime Q − SJF = +11.443, CI [+5.022, +19.297].

The same conclusion appears in all three systems: **a learned adaptation does not beat plain
SJF on these workloads**, and here the self-healing loop is the narrowest of the three gaps.
