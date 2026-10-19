# Evaluation results

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

Everything below is produced by one command,

```bash
.venv/bin/python main.py healing-study
```

and every number is traceable to a file under `results/healing/` (named after each table).
The run reported here used the frozen design: training seeds `7101–7105` (1200 traces each),
validation seed `8201` (140 traces), **held-out final-test seed `23819`** (210 traces), 1000
bootstrap replicates at seed `104729`. The seed was declared in
`experiments/healing_config.py` **before** the frozen run and is never used for any
development decision; the six streams already consumed (`9401`, `20501`, `20719`, `21617`,
`72601` and `21977`) are retired in the same file and cannot come back.

Hardware/software: Python 3.11.2, NumPy 2.4.6, pandas 3.0.6, Matplotlib 3.11.2, pytest 9.1.1
(recorded in `healing_summary.json` under `software`). Wall clock of the frozen run: ≈ 47 s.
Every table in this file was re-read from the artifacts after the run; the whole pipeline is
deterministic (two independent runs into separate directories produced 21/21 machine-readable
artifacts and 3/3 figures byte-identical).

The verification rule was changed after seed `21977` had been evaluated: the phase reference
is now a *physical* observation row (`trigger_load`, ready-set size, visible work) instead of a
free fit of the closure statistic, and the declared boundaries are ±0.30 instead of
+0.65/−0.45. The `21977` tables are therefore historical and are kept only in
§10; everything in §1–§9 below is the seed-`23819` run.

---

## 1. Held-out comparison (final-test seed 23819)

Mean of 210 traces per arm (learned arms: 1050 traces, five training seeds). Waiting,
response and turnaround times are in simulated time units; utilization and throughput in
percent / processes per time unit.

| Arm | Mean waiting | P95 waiting | Max waiting | Mean response | Mean turnaround | CPU util. | Throughput | Context switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_sjf` | **94.66** | **249.20** | **249.20** | **94.66** | **113.59** | 92.268 | 0.0778 | 14.00 |
| `q_recovery_from_sjf` (5 seeds) | 94.66 | 249.20 | 249.20 | 94.66 | 113.59 | 92.268 | 0.0778 | 14.00 |
| `rule_wait_from_sjf` | 94.87 | 249.54 | 249.54 | 94.87 | 113.81 | 92.268 | 0.0778 | 14.00 |
| `rule_wait` | 122.39 | 251.88 | 251.88 | 122.39 | 141.33 | 92.268 | 0.0778 | 14.00 |
| `q_recovery` (5 seeds) | 122.56 | 251.98 | 251.98 | 122.54 | 141.50 | 92.249 | 0.0778 | 14.03 |
| `fixed_priority` | 128.32 | 262.75 | 262.75 | 128.32 | 147.25 | 92.268 | 0.0778 | 14.00 |
| `detector_only` (control) | 130.84 | 258.42 | 258.42 | 130.84 | 149.77 | 92.268 | 0.0778 | 14.00 |
| `fixed_fcfs` | 130.84 | 258.42 | 258.42 | 130.84 | 149.77 | 92.268 | 0.0778 | 14.00 |
| `rule_fairness` | 136.78 | 263.72 | 263.72 | 110.40 | 155.72 | 90.380 | 0.0766 | 20.02 |
| `fixed_round_robin` | 217.05 | 310.41 | 310.41 | 26.50 | 235.99 | 78.253 | 0.0687 | 75.26 |

Reading of the table:

* the **control is exactly FCFS** (130.838 / 258.424 / 14.00 switches, to the last digit): the
  monitoring, detection and diagnosis stages change nothing when the policy never intervenes;
* the **waiting-time rule and the learned controller both beat the control** (122.39 and
  122.56 vs 130.84) and **both lose to fixed SJF**;
* the **fairness rule is worse than the control** (136.78): its starvation→Round-Robin mapping
  buys responsiveness (response 110.40 vs 130.84, −20.43 [−23.21, −17.85]) at the cost of
  waiting time and six extra switches per trace;
* started from SJF, the loop is **inert and harmless**: `q_recovery_from_sjf` reproduces SJF
  exactly (0 interventions), `rule_wait_from_sjf` intervenes 0.02 times per trace and costs
  0.21 wait units;
* throughput and utilization are essentially identical across the FCFS/SJF/priority arms
  (0.0778 / 92.3 %) because a single-CPU simulator with a fixed workload completes the same
  total work; the differences show up in *who* waits, not in *how much* work is done.

## 2. Paired comparisons with uncertainty

Family-stratified crossed bootstrap over identical workloads, 1000 replicates, seed 104729
(`healing_comparisons.csv`; lower is better for waiting time, so a negative difference favours
the target).

| Target | Reference | Metric | Difference | 95 % CI |
|---|---|---|---:|---|
| `rule_wait` | `detector_only` | mean waiting | **−8.45** | [−9.44, −7.35] |
| `rule_wait` | `fixed_fcfs` | mean waiting | **−8.45** | [−9.54, −7.43] |
| `rule_wait` | `fixed_sjf` | mean waiting | +27.73 | [+26.05, +29.72] |
| `q_recovery` | `fixed_fcfs` | mean waiting | **−8.27** | [−9.31, −7.20] |
| `q_recovery` | `fixed_sjf` | mean waiting | +27.91 | [+26.11, +29.87] |
| `q_recovery` | `rule_wait` | mean waiting | +0.18 | [−0.04, +0.48] |
| `rule_fairness` | `fixed_fcfs` | mean waiting | +5.94 | [+4.38, +7.45] |
| `q_recovery_from_sjf` | `fixed_sjf` | mean waiting | 0.00 | [0.00, 0.00] |
| `rule_wait_from_sjf` | `fixed_sjf` | mean waiting | +0.21 | [+0.04, +0.43] |

The ordering `fixed_sjf < rule_wait ≈ q_recovery < fixed_fcfs < rule_fairness` is resolved at
this design; the learned arm is **not** distinguishable from the waiting-time rule (the
interval is [−0.04, +0.48]). Training variability across the five learned seeds is small: the
seed spread of the mean waiting time is 0.146 wait units (`healing_q_seed_variability.csv`),
against a paired difference of −8.27 for the learned arm.

## 3. Counterfactual replay audit (the causal measurement)

**Matched replay.** For every intervention, the *same controller* is re-run on the *same
workload* with the *same recorded prefix* and exactly that one incident forced to hold
(`forced_holds`), while every later decision is still taken by the controller as usual. The
recorded identity check (`divergence_decision_index`) confirms that the prefix is identical up
to the incident decision and diverges at it. The gain is
`matched_hold_total_waiting − actual_total_waiting`; a non-positive gain is what this study
calls a **false recovery** (its denominator is *all* audited interventions). A second,
separately reported counterfactual (`hold_forever_*`) replays the prefix and then never heals
again, which answers the different question of what shutting the loop down at that point would
have been worth (`healing_audit.csv`).

| Arm | Audited | Mean matched gain | Median | Helped | False recoveries | False-recovery rate |
|---|---:|---:|---:|---:|---:|---:|
| `rule_fairness` | 150 | −130.84 | −110.5 | 35 | 115 | **76.7 %** |
| `rule_wait` | 141 | +188.79 | +172.0 | 133 | 8 | **5.7 %** |
| `q_recovery` | 678 | +192.07 | +173.0 | 644 | 34 | **5.0 %** |
| `rule_wait_from_sjf` | 4 | −169.25 | −158.5 | 0 | 4 | **100 %** |

The action effect is large and consistent: the waiting-time rule's interventions reduce total
waiting by ≈ 189 units on average and help 94.3 % of the time; the learned controller's are
indistinguishable from that (+192.07 / 95.0 %). The fairness rule is genuinely harmful and the
SJF-start arm has almost nothing to fix (four interventions, all harmful).

Every rate in this table has an *intervention* denominator. The second counterfactual — the
recorded prefix with no further healing after the episode's first incident — has an *episode*
denominator and is therefore reported in its own table in `healing_report.md`: 135 episodes for
each rule arm and 655 for the learned arm, with mean gains of −138.65 (`rule_fairness`),
+197.16 (`rule_wait`) and +198.95 (`q_recovery`). The two views agree closely (for `rule_wait`
+188.79 per intervention vs +197.16 per episode), which is the check that neither is an
artefact of its denominator. In the other direction the audit makes the failure visible in
both signs: 8 rejected interventions were actually helpful and **0 accepted interventions were
harmful** on this split (65 accepted, all helpful), while `rule_fairness` accumulates 115
false recoveries across 150 interventions.

## 4. Verification behaviour, measured against that audit

The in-run rule compares the window's *phase-adjusted* closure with two boundaries derived
from the no-action control of the **training** split (§5). The three outcomes are
`accepted` / `rejected` / `inconclusive`; only `accepted` certifies the change, only
`rejected` is a confirmed failure, and an `inconclusive` change follows the declared safety
posture (rolled back, but not counted as a failure — `healing.config.inconclusive_rollback`).

| Arm | Interventions | Accepted | Rejected | Inconclusive | Confirmed failures | Episodes hitting the fallback |
|---|---:|---:|---:|---:|---:|---:|
| `rule_fairness` | 150 | 2 (1.3 %) | 31 (20.7 %) | 117 (78.0 %) | 31 | 1 |
| `rule_wait` | 141 | 12 (8.5 %) | 0 | 129 (91.5 %) | 0 | 0 |
| `q_recovery` | 678 | 51 (7.5 %) | 2 (0.3 %) | 625 (92.2 %) | 2 | 0 |
| `rule_wait_from_sjf` | 4 | 0 | 1 (25 %) | 3 (75 %) | 1 | 0 |

The single fallback episode is `rule_fairness` on repetition 21 (`poisson_arrivals`): its
second consecutive confirmed failure in that trace occurs at decision 20, so the arm pins the
last accepted configuration and records `fallback_triggered = true` in the incident ledger
(`healing_incidents_final_test.csv`). The same arm records one fallback episode on the
validation split.

Against the matched counterfactual:

| Arm | Accepted & helpful | Accepted but harmful | Rejected & harmful | Rejected but helpful | Not certified, of the helpful ones |
|---|---:|---:|---:|---:|---:|
| `rule_fairness` | 2 | 0 | 23 | 8 | 94.3 % |
| `rule_wait` | 12 | 0 | 0 | 0 | 91.0 % |
| `q_recovery` | 51 | 0 | 2 | 0 | 92.1 % |
| `rule_wait_from_sjf` | 0 | 0 | 1 | 0 | n/a (nothing was helpful) |

**This is the honest core of the study, and it is a two-sided result.**

* The verifier *accepts* a population that is purely beneficial on this split: **65 of 65
  accepted interventions were helpful** (mean matched gain +184.9), where only 83.5 % of all
  audited interventions helped and 14.0 % harmed. No harmful change was ever certified.
* The verifier *rejects* a population that is strongly enriched in harm: of the 34 rejected
  interventions, 26 were harmful (76.5 %) against a base rate of 14.0 %, i.e. **5.5×
  enrichment**. Rejection therefore carries information, and it is what the safety posture acts
  on.
* What it cannot do is *certify*: 91.0 % (`rule_wait`) and 92.1 % (`q_recovery`) of the
  genuinely helpful interventions end inconclusive. They are still executed — the change is
  kept for the whole verification window before the declared rollback — which is why the arms
  finish ahead of FCFS at all, but they are never recorded as verified.
* The statistic separates the two populations well on this split: rank AUC of the
  phase-adjusted residual for "this intervention helped" is **0.951** (the raw closure reaches
  **0.613**; chance is 0.5). `healing_summary.json` (`verification_discrimination`) reports this
  together with every count and denominator used above, so the section can be recomputed from
  the artifacts alone. The residual's advantage over the raw statistic is the largest single
  improvement of this round (0.720 → 0.951 against the retired `21977` stream, §10).

### 4.1 Why the verifier cannot certify more (measured power, not a threshold choice)

`healing_action_probe.csv` and `healing_audit.csv` together bound what any single-window rule
could do here:

| Quantity | Value | Source |
|---|---:|---|
| paired change of the closure statistic caused by `set_sjf` (best action, canonical) | **+0.116** units | `healing_action_probe.csv`, `paired_closure_change`, **98** incidents (784 rows, canonical `controller_reward` probe) |
| paired change caused by `set_round_robin` (worst offered action, canonical) | **−0.137** units | same (canonical) |
| paired change caused by `set_priority` / five inert actions (canonical) | **−0.007** / 0.000 | same (canonical) — legacy `healing_action_probe.legacy_raw_proxy.csv` (21 incidents) gave **+0.152 / −0.159 / +0.018** respectively |
| spread of the statistic across control windows | 0.806 (sd) | training control, 3737 windows |
| spread after subtracting the declared phase model | **0.160 (sd)** | training control, R² = 0.961 (80.2 % of the variance removed) |
| paired spread of the same statistic against its own no-action replay | **0.090** (sd) | `paired_closure_change`, canonical validation probe (legacy 0.101 sd on 21 incidents) |
| effect of the best action in units of the *online* residual | **0.73 sd** | derived (0.116/0.160); legacy 0.95 sd (0.152/0.160) |
| windows of the same action needed for 80 % power at α = 0.05 (two-sided, paired against a known baseline) | **≈ 15** | derived (canonical); legacy ≈ 39 for 0.95 sd |
| genuinely helpful interventions that clear the unconditional null 95th percentile | **6.7 %** (54 of 812, mean gain +184.9) | `verification_discrimination` |
| genuinely helpful interventions that clear the operative acceptance boundary | **8.0 %** (65 of 812) | `verification_discrimination`, `healing_audit.csv` |

So the recovery actions are *not* invisible to the monitored statistic — they are simply
smaller than the noise a single window carries online. The declared phase model removes 80.2 %
of the statistic's no-action variance, which is what lifts the rank AUC to 0.951 and makes
every accepted intervention on this split genuinely helpful; what it cannot do is shrink the
*residual* to the size of a 0.12-unit canonical action effect (legacy **0.15** on 21 incidents), because the residual sd is 0.160 and the best action's paired change is **0.116** in the canonical validation probe (98 incidents; legacy 0.152). With the model as declared, one window cannot turn a **0.73-sd** (legacy 0.95-sd) signal into a reliable certificate at a fixed 5 % false-accept budget, and the honest
engineering answer is the three-outcome rule: accept the extreme tail, reject the clearly bad
tail, and call everything else what it is — inconclusive. §10 records the model comparison that
chose the declared specification, including the two candidates that were measured and set
aside.

### 4.2 The declared posture is a choice, and it was measured

Rolling an unverified change back is a *posture*, not a mathematical consequence. It was
compared on the **validation split only** with the identical rule and thresholds, flipping the
single declared flag `inconclusive_rollback`:

| Posture (validation, 140 traces) | `rule_wait` mean waiting | `q_recovery` (5 seeds) | `rule_fairness` | Policy switches (rule_wait) |
|---|---:|---:|---:|---:|
| rollback (declared) | 118.18 | 118.70 ± 0.21 | **133.40** | 196 |
| keep | 117.13 | 117.87 ± 0.26 | 144.77 | 102 |

Rolling back costs about one wait unit on the arms whose actions are usually useful — inside
the paired noise of that split — but it is clearly better for the arm whose actions are usually
harmful (−11 wait units on `rule_fairness`), and it halves the number of configuration
changes. The conservative posture is kept: **the loop never keeps a configuration it could not
verify**, the price is one extra switch per unverified change, and the cost is bounded because
no failure counter and no rollback penalty are attached to an inconclusive window.

## 5. Derived thresholds

`healing_threshold_derivation.csv` (from the **training** control only, 6000 traces):

| Quantity | Value |
|---|---:|
| control traces / degradation events with no intervention | 6000 / 3737 |
| events per trace | 0.623 |
| windows the verifier can judge (complete, ≥ 3 completions, positive duration) | 3633 (104 truncated windows excluded, 0 thin) |
| null closure p50 / p90 / p95 | 1.401 / 2.183 / 2.379 |
| declared `accept_closure_factor` (reporting, unconditional) | 2.4 |
| false-accept rate of that unconditional rule on the control | 0.046 |
| declared phase baseline: `intercept, trigger_load, ready_count, log1p(visible_work)` | −2.422506, −0.008145, 0.998700, −0.082333 |
| phase-baseline R² (variance of the null closure explained) | 0.961 |
| residual sd after the baseline (raw sd 0.806) | 0.160 (−80.2 %) |
| control residual p05 / p95 | −0.2565 / +0.2631 |
| derived / declared `accept_residual_factor` | 0.30 / 0.30 |
| derived / declared `reject_residual_factor` | −0.30 / −0.30 |
| measured false-accept / false-reject of the declared rule on the control | 0.034 / 0.031 |
| out-of-sample (validation control: 95 judgeable windows of 98 declared events) false-accept / false-reject | 0.021 / 0.011 |

Both operative boundaries are therefore percentile points of the *same* control distribution
(95th and 5th, rounded outward to the declared step of 0.05), and the out-of-sample check stays
below the 5 % design target on a split that took no part in fitting. Congestion-gate sweep (600
control traces each): `min_load_factor` 0.00 → 1.517 events/trace, 0.50 → 1.475, 0.80 → 1.052,
1.00 → 0.858, 1.25 → 0.042; the declared value is 1.0.

## 6. Detector quality: recall, delay and false triggers

The no-action control declares events without ever acting, so every event it declares comes
from the workload. `healing_detection_scenarios.csv` (validation split, 140 traces per
scenario) injects a *controlled* step change into two thirds of the traces:

| Scenario | Traces | Detected | Detection share (recall) | Incidents / trace | Median delay (time units) | Median delay (decisions) |
|---|---:|---:|---:|---:|---:|---:|
| `stable` (false-trigger probe) | 140 | 96 | 0.686 | 0.700 | 111.5 | 6.0 |
| `burst_shock` (CPU time ×4 after t\*) | 140 | 105 | 0.750 | 0.771 | 134.0 | 6.0 |
| `arrival_storm` (arrivals compressed ×0.35) | 140 | 79 | 0.564 | 0.571 | 81.0 | 6.0 |
| all | 420 | 280 | 0.667 | 0.681 | 113.5 | **6.0** |

* **Recall is the measure that matters here**: 75.0 % for a burst shock and 56.4 % for an
  arrival storm, at a detection delay of six decision intervals (≈ 80–135 simulated time
  units) after the injection.
* **Precision proxy 0.657**: two thirds of all declared detections fall on an injected trace.
  It is a proxy, not a precision: the injection is a workload property, not a label of the
  event the detector is looking for, and a stable trace may legitimately degrade.
* **The false-trigger frequency is the honest weak point**: 0.70 incidents per *stable* trace,
  and a trace-length effect is visible in the fact that 68.6 % of stable traces produce one.
  These traces begin with an arrival backlog, so the reference itself is established from a
  busy start. The hit rate falls to 0.042 events/trace at `min_load_factor = 1.25`, but at that
  setting the detector also misses the two injections almost completely: the trade-off between
  delayed detection, unnecessary recovery and missed degradation is explicit in the sweep, and
  the declared setting (1.0) was chosen on the **training** control, never on these
  validation numbers.
* **The loop does not trigger itself.** No incident ever opens while its own verification
  window is open (the controller records the explicit suppression reason
  `verification-window-open`), consecutive incidents are separated by at least the declared
  cooldown of three decisions measured from the end of the previous window, the number of
  interventions per trace is capped at six, and a sustained degradation keeps failing the
  reference instead of being absorbed into it (tests
  `test_no_incident_opens_while_a_verification_window_is_open`,
  `test_sustained_degradation_is_not_absorbed_into_the_reference`).

Diagnoses on the held-out control (`detector_only`, 137 incidents): fairness starvation 124,
priority-ordering anomaly 8, head-of-line blocking 3, queue congestion 2; tail dispersion and
`unexplained` never fired. On the validation control (98 incidents) the same four categories
appear (84 / 5 / 4 / 5), so the diagnosis mix is stable across splits.

## 7. Reward versus measured truth

`healing_action_probe.csv` (canonical, **98** validation incidents × 8 actions = **784** rows) replays **every** action at **every** detected incident and compares the **actual controller reward** — phase-adjusted residual minus the declared accept boundary minus the band-scaled intervention cost and, on `rejected`, the rollback penalty (`healing/reward.py:controller_reward`) — with the measured effect on total waiting time (per process). The raw closure statistic (`(trigger−post)/max(trigger−reference,band)`), the cost-including raw proxy (`raw_load_improvement_band_units`), the phase-adjusted residual (`closure_residual = closure−expected_closure`), and the final controller reward are distinct columns in the canonical file; the legacy `healing_action_probe.legacy_raw_proxy.csv` (21 incidents × 8 = 168 rows) recorded only the raw proxy as `reward_band_units`.

| Action | Controller reward (band units, canonical) | Raw proxy `raw_load_improvement_band_units` (canonical) | Measured true effect (wait units / process, canonical) | Recorded raw-proxy reward (band units, legacy 21-incident) | Legacy true effect |
|---|---:|---:|---:|---:|---:|
| `set_sjf` | **−0.250** | 1.036 | **+3.478** | 0.978 | **+3.625** |
| `set_priority` | −0.375 | 0.914 | +0.061 | 0.844 | +0.483 |
| `set_round_robin` | **−0.518** | 0.784 | **−1.168** | 0.666 | **−2.476** |
| five inert (`hold`, `set_fcfs`, `shrink_quantum`, `grow_quantum`, `revert`) | −0.304 | 0.949 | 0.000 | 0.853 | 0.000 |

Two findings matter more than the pooled correlation:

* **the reward ranks actions correctly inside an incident**, which is the quantity learning
  actually needs — measured on the canonical probe (`controller_reward` vs `true_effect_per_process`): mean within-incident **Spearman** rank correlation **0.449** (median **0.531**), `argmax(controller_reward)` is the truly best action in **64 of 98** incidents (65.3 %), pooled Pearson **0.197** (pooled Spearman **0.223**). The historical 21-incident legacy raw-proxy file (`reward_band_units` as raw proxy) gave mean 0.622 (median 0.875), 15 of 21 (71.4 %), pooled Pearson 0.104 — quoted here only as a labeled legacy comparison. Different traces have different scales; the pooled number alone understates the within-incident signal.
* **five of the eight actions are provably inert from the declared starting policy**: on an
  FCFS start `hold`, `set_fcfs`, `shrink_quantum`, `grow_quantum` and
  `revert_to_last_accepted` all have exactly zero effect. Only `set_sjf`, `set_priority` and
  `set_round_robin` can change anything — in the canonical 98-incident probe `set_sjf` is the truly best action in **69 of 98** incidents, `set_round_robin` in **12**, `set_priority` in **6** (legacy 21-incident probe: `set_sjf` 15 of 21). The learning problem is therefore much smaller than "8 actions" suggests.

## 8. What the learned controller saw (coverage)

| Training seed | Encoder states | Visited states | Visited pairs | Pairs of the declared table | Pairs of the reachable table | Incidents | Decisions with ≥ 2 known actions | Defaulted decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7101 | 648 | 32 | 58 | 1.12 % | 6.71 % | 772 | 89.0 % | 11.0 % |
| 7102 | 648 | 29 | 57 | 1.10 % | 6.60 % | 753 | 88.4 % | 11.6 % |
| 7103 | 648 | 30 | 62 | 1.20 % | 7.18 % | 769 | 90.9 % | 9.1 % |
| 7104 | 648 | 33 | 66 | 1.27 % | 3.82 % | 771 | 90.0 % | 10.0 % |
| 7105 | 648 | 31 | 56 | 1.08 % | 6.48 % | 745 | 86.2 % | 13.8 % |

* **The declared table is 648 states × 8 actions = 5184 pairs, but only a fraction of them
  are reachable from the contexts the training data actually produces**: 108 × 8 = 864 pairs
  for seeds 7101, 7102, 7103 and 7105 (the `current_policy` dimension is always the starting
  FCFS because a change only survives if it is verified) and 216 × 8 = 1728 for seed 7104,
  whose traces expose a second `current_policy` bin. Against that per-seed, data-driven
  denominator the visited pairs are 3.8–7.2 % of what is reachable, versus 1.1–1.3 % of the
  declared table; the artifact reports both numerators against both denominators
  (`healing_coverage.csv`).
* **Visits are concentrated**: the median visited state is seen 4–6 times and the most visited
  pair 240–262 times, i.e. a handful of cells carry most of the evidence while 14–18 visited
  states ever saw a single action.
* **More training does not fix it**: the coverage curve (`healing_coverage_curve.csv`) grows
  41 → 55 → 58 → 58 visited pairs for seed 7101 as episodes go 300 → 600 → 900 → 1200, and the
  last 300 episodes add **no** new pair (seeds 7102 and 7105 add one). The event supply, not the
  budget, is the binding constraint: a trace produces 0.62 incidents on average, so 1200 traces
  buy ≈ 750 decisions in a 648-cell space.
* **Only four of the eight actions are ever offered** from the FCFS start; the mutation mask
  removes the rest (`offered_actions_exercised = 4`, `chosen_actions_exercised = 4`).
* Consequence, stated as a limitation rather than fixed by a bigger algorithm: the learned arm
  is statistically indistinguishable from the hand-written waiting-time rule (§2) and its
  advantage over FCFS (−8.27) is the same advantage the rule gets (−8.45).

## 9. Worked demonstration trace

`healing_demo.json` (selected by a rule declared before the run: learned arm on the validation
split, preferring an episode with both an accepted and a non-accepted verdict; the episode is
identified by arm **and** training seed, and `_demo_payload` refuses to write the artifact if
the replay does not reproduce the recorded incident count).

| Step | Observed |
|---|---|
| workload | `long_jobs`, validation repetition 0 (ledger episode 20), fingerprint `2cc74cad24d7c29d`, arm `q_recovery` seed 7105 |
| detection | incident at decision 6 (t = 142), `fairness_starvation`, window mean above `reference + 1.25·band` |
| decision | `set_sjf` — FCFS → SJF (subject to the mask) |
| verification | window closes with `closure_fraction = 2.324` against an expected closure of `2.010` for this trigger phase (trigger load 10.77, reference 10.60, band 2.36, 9 ready processes holding 234 work units) |
| verdict | **accepted** (`verified_phase_residual`), residual +0.314 against the declared accept boundary +0.30 |
| reward | −0.053: the reward is the residual *minus the accept boundary*, so an accepted action sits just above zero |
| result | 14 context switches, configuration kept to the end of the trace |

The same file records the full decision log (`decisions`), so the trace can be re-derived
decision by decision.

## 10. Preserved historical results (not the project's headline results)

These come from the preserved offline study and the non-healing runtime appendix; they are
reproducible with `python main.py study` and `python main.py runtime-experiment` and live in
`results/study/` and `results/runtime/`. They are reported here only so the two systems can be
told apart.

* Offline policy selection (seed 19501): risk-aware Q mean cost 0.7953 / regret 0.0574,
  fingerprint rule 0.7752 / 0.0372, SJF 0.8419 / 0.1040.
* Runtime appendix (seed 19301, 210 traces): mean waiting — FCFS 127.797, SJF 93.345,
  Round Robin 215.729, Priority 129.207, causal heuristic 203.404, runtime Q 104.789;
  runtime Q − SJF = +11.443, CI [+5.022, +19.297].

The same conclusion appears in all three systems: **a learned adaptation does not beat plain
SJF on these workloads**, and here the self-healing loop is the narrowest of the three gaps.

### 10.1 Retired held-out stream `21977` (historical, superseded)

The tables above replace an earlier frozen run on the held-out stream `21977`. That stream was
evaluated once with the *previous* phase reference (a free fit of the closure statistic with
boundaries +0.65/−0.45) and was then examined while the physical reference was chosen, so its
numbers are development evidence, not held-out results, and are not reproduced by the current
code. They are kept here only to make the size of the change inspectable:

| Quantity | `21977` (previous rule) | `23819` (reported rule) |
|---|---:|---:|
| `rule_wait` mean waiting | 120.77 | 122.39 |
| `q_recovery` mean waiting | 121.01 | 122.56 |
| accepted interventions | 34 | 65 |
| accepted that were genuinely helpful | 70.6 % | **100 %** (65 of 65) |
| rejected interventions that were harmful | 46.9 % (15 of 32) | 76.5 % (26 of 34) |
| rank AUC of the residual for "helped" | 0.720 | **0.951** |
| certifyable share of genuinely helpful interventions | 6.1 % | 6.7 % at the null p95 / 8.0 % at the declared boundary |

The mean waiting times are within the paired noise of the two streams (the arms, workloads and
policies are unchanged); what moved is the verifier: the same screening budget now rejects more
of the harmful population and never certifies a harmful change on the reported stream.
