# Evaluation results

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

Everything below is produced by one command,

```bash
.venv/bin/python main.py healing-study
```

and every number is traceable to a file under `results/healing/` (named after each table).
The run reported here used the frozen design: training seeds `7101–7105` (1200 traces each),
validation seed `8201` (140 traces), **held-out final-test seed `21977`** (210 traces), 1000
bootstrap replicates at seed `104729`. The seed was declared in
`experiments/healing_config.py` **before** the frozen run and is never used for any
development decision; the five streams already consumed (`9401`, `20501`, `20719`, `21617` and
`72601`) are retired in the same file and cannot come back.

Hardware/software: Python 3.11.2, NumPy 2.4.6, pandas 3.0.6, Matplotlib 3.11.2, pytest 9.1.1
(recorded in `healing_summary.json` under `software`). Wall clock of the frozen run: ≈ 31 s.
Every table in this file was re-read from the artifacts after the run; the whole pipeline is
deterministic (two independent runs into separate directories produced 23/23 byte-identical
artifacts).

---

## 1. Held-out comparison (final-test seed 21977)

Mean of 210 traces per arm (learned arms: 1050 traces, five training seeds). Waiting,
response and turnaround times are in simulated time units; utilization and throughput in
percent / processes per time unit.

| Arm | Mean waiting | P95 waiting | Max waiting | Mean response | Mean turnaround | CPU util. | Throughput | Context switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed_round_robin` | 217.14 | 309.84 | 309.84 | 26.39 | 236.04 | 78.329 | 0.0705 | 75.06 |
| `rule_fairness` | 134.91 | 263.45 | 263.45 | 107.13 | 153.81 | 90.441 | 0.0776 | 19.97 |
| `fixed_fcfs` | 128.93 | 258.22 | 258.22 | 128.93 | 147.82 | 92.285 | 0.0800 | 14.00 |
| `detector_only` (control) | 128.93 | 258.22 | 258.22 | 128.93 | 147.82 | 92.285 | 0.0800 | 14.00 |
| `fixed_priority` | 128.10 | 262.31 | 262.31 | 128.10 | 147.00 | 92.285 | 0.0800 | 14.00 |
| `q_recovery` (5 seeds) | 121.01 | 253.06 | 253.06 | 120.79 | 139.91 | 92.234 | 0.0796 | 14.10 |
| `rule_wait` | 120.77 | 252.66 | 252.66 | 120.77 | 139.67 | 92.285 | 0.0800 | 14.00 |
| `rule_wait_from_sjf` | 94.83 | 248.55 | 248.55 | 94.83 | 113.73 | 92.285 | 0.0800 | 14.00 |
| `q_recovery_from_sjf` (5 seeds) | 94.56 | 248.10 | 248.10 | 94.56 | 113.45 | 92.285 | 0.0800 | 14.00 |
| `fixed_sjf` | **94.56** | **248.10** | **248.10** | **94.56** | **113.45** | 92.285 | 0.0800 | 14.00 |

Reading of the table:

* the **control is exactly FCFS** (128.925 / 258.224 / 14.00 switches, to the last digit): the
  monitoring, detection and diagnosis stages change nothing when the policy never intervenes;
* the **waiting-time rule and the learned controller both beat the control** (120.77 and
  121.01 vs 128.93) and **both lose to fixed SJF**;
* the **fairness rule is worse than the control** (134.91): its starvation→Round-Robin mapping
  buys responsiveness (response 107.13 vs 128.93) at the cost of waiting time and six extra
  switches per trace;
* started from SJF, the loop is **inert and harmless**: `q_recovery_from_sjf` reproduces SJF
  exactly (0 interventions), `rule_wait_from_sjf` intervenes 0.03 times per trace and costs
  0.28 wait units;
* throughput and utilization are essentially identical across the FCFS/SJF/priority arms
  (0.0800 / 92.3 %) because a single-CPU simulator with a fixed workload completes the same
  total work; the differences show up in *who* waits, not in *how much* work is done.

## 2. Paired comparisons with uncertainty

Family-stratified crossed bootstrap over identical workloads, 1000 replicates, seed 104729
(`healing_comparisons.csv`; lower is better for waiting time, so a negative difference favours
the target).

| Target | Reference | Metric | Difference | 95 % CI |
|---|---|---|---:|---|
| `rule_wait` | `detector_only` | mean waiting | **−8.15** | [−9.31, −6.96] |
| `rule_wait` | `fixed_fcfs` | mean waiting | **−8.15** | [−9.27, −7.00] |
| `rule_wait` | `fixed_sjf` | mean waiting | +26.22 | [+24.28, +28.22] |
| `q_recovery` | `fixed_fcfs` | mean waiting | **−7.91** | [−9.19, −6.86] |
| `q_recovery` | `fixed_sjf` | mean waiting | +26.46 | [+24.52, +28.40] |
| `q_recovery` | `rule_wait` | mean waiting | +0.24 | [−0.04, +0.60] |
| `rule_fairness` | `fixed_fcfs` | mean waiting | +5.98 | [+4.53, +7.49] |
| `q_recovery_from_sjf` | `fixed_sjf` | mean waiting | 0.0000 | [0.0000, 0.0000] |
| `rule_wait_from_sjf` | `fixed_sjf` | mean waiting | +0.28 | [+0.06, +0.57] |

The ordering `fixed_sjf < rule_wait < q_recovery ≈ rule_wait` and
`rule_fairness > fixed_fcfs` is resolved at this design; the learned arm is **not**
distinguishable from the waiting-time rule (the interval is [−0.04, +0.60], i.e. it may be a
hair better or a hair worse). Training variability across the five learned seeds is small:
seed spread of the mean waiting time is 0.254 wait units (`healing_q_seed_variability.csv`),
against a paired difference of −8.15 for the rule.

## 3. Counterfactual replay audit (the causal measurement)

**Matched replay.** For every intervention, the *same controller* is re-run on the *same
workload* with the *same recorded prefix* and exactly that one incident forced to hold
(`forced_holds`), while every later decision is still taken by the controller as usual. The
recorded identity check (`divergence_decision_index`) confirms that the prefix is identical
up to the incident decision and diverges at it. The gain is
`matched_hold_total_waiting − actual_total_waiting`; a non-positive gain is what this study
calls a **false recovery** (its denominator is *all* audited interventions). A second,
separately reported counterfactual (`hold_forever_*`) replays the prefix and then never heals
again, which answers the different question of what shutting the loop down at that point would
have been worth (`healing_audit.csv`).

| Arm | Audited | Mean matched gain | Median | Helped | False recoveries | False-recovery rate |
|---|---:|---:|---:|---:|---:|---:|
| `rule_fairness` | 152 | −126.19 | −108.0 | 29 | 123 | **80.9 %** |
| `rule_wait` | 144 | +178.78 | +159.5 | 136 | 8 | **5.6 %** |
| `q_recovery` | 673 | +185.14 | +160.0 | 637 | 36 | **5.3 %** |
| `rule_wait_from_sjf` | 6 | −145.00 | −129.0 | 0 | 6 | **100 %** |

The action effect is therefore large and consistent: the waiting-time rule's interventions
reduce total waiting by ≈ 179 units on average and help 94.4 % of the time; the learned
controller's are indistinguishable from that (+185.14 / 94.7 %). The fairness rule is
genuinely harmful and the SJF-start arm has nothing to fix.

Every rate in this table has an *intervention* denominator. The second counterfactual — the
recorded prefix with no further healing after the episode's first incident — has an *episode*
denominator and is therefore reported in its own table in `healing_report.md`: 136 episodes for
each rule arm and 644 for the learned arm, with mean gains of −138.54 (`rule_fairness`),
+188.85 (`rule_wait`) and +193.47 (`q_recovery`). The two views agree closely (for `rule_wait`
+178.78 per intervention vs +188.85 per episode), which is the check that neither is an
artefact of its denominator.

## 4. Verification behaviour, measured against that audit

The in-run rule compares the window's *phase-adjusted* closure with two boundaries derived
from the no-action control of the **training** split (§5). The three outcomes are
`accepted` / `rejected` / `inconclusive`; only `accepted` certifies the change, only
`rejected` is a confirmed failure, and an `inconclusive` change follows the declared safety
posture (rolled back, but not counted as a failure — `healing.config.inconclusive_rollback`).

| Arm | Interventions | Accepted | Rejected | Inconclusive | Confirmed failures | Episodes hitting the fallback |
|---|---:|---:|---:|---:|---:|---:|
| `rule_fairness` | 152 | 3 (2.0 %) | 16 (10.5 %) | 133 (87.5 %) | 16 | 0 |
| `rule_wait` | 144 | 6 (4.2 %) | 3 (2.1 %) | 135 (93.8 %) | 3 | 0 |
| `q_recovery` | 673 | 22 (3.3 %) | 13 (1.9 %) | 638 (94.8 %) | 13 | 0 |
| `rule_wait_from_sjf` | 6 | 3 (50 %) | 0 | 3 (50 %) | 0 | 0 |

Against the matched counterfactual:

| Arm | Accepted & helpful | Accepted but harmful | Rejected & harmful | Rejected but helpful | Not certified, of the helpful ones |
|---|---:|---:|---:|---:|---:|
| `rule_fairness` | 0 | 1 | 10 | 6 | 100 % |
| `rule_wait` | 6 | 0 | 1 | 2 | 95.6 % |
| `q_recovery` | 18 | 4 | 4 | 9 | 97.2 % |
| `rule_wait_from_sjf` | 0 | 3 | 0 | 0 | n/a (nothing was helpful) |

**This is the honest core of the study, and it is a two-sided result.**

* The verifier *accepts* a population that is enriched in benefit: 24 of the 34 accepted
  interventions helped the trace (70.6 %) against a base rate of 82.3 % — and, seen from the
  other side, 70.6 % of the accepted changes were right while only 15.9 % of all audited
  changes were harmful. It is still a *filter*, not a benefit estimator: at a fixed false-accept
  budget of 5 % on the control it cannot certify the ordinary, modest improvement — §4.1 shows
  why.
* The verifier *rejects* a population that is strongly enriched in harm: of the 32 rejected
  interventions, 15 were harmful (46.9 %) against the same base rate of 15.9 %, i.e. 2.9×
  enrichment. Rejection therefore carries information, and it is what the safety posture acts
  on.
* What it cannot do is *certify*: 95.6 % (`rule_wait`) and 97.2 % (`q_recovery`) of the
  genuinely helpful interventions end inconclusive. They are still executed — the change is
  kept for the whole verification window before the declared rollback — which is why the arms
  finish ahead of FCFS at all, but they are never recorded as verified.
* The statistic itself separates the two populations imperfectly but measurably: rank AUC of
  the phase-adjusted residual for "this intervention helped" is **0.720** on the held-out split
  (the raw closure reaches **0.610**; chance is 0.5). `healing_summary.json`
  (`verification_discrimination`) reports this together with every count and denominator used
  above, so the section can be recomputed from the artifacts alone.

### 4.1 Why the verifier cannot certify more (measured power, not a threshold choice)

`healing_action_probe.csv` and `healing_audit.csv` together bound what any single-window rule
could do here:

| Quantity | Value | Source |
|---|---:|---|
| paired change of the closure statistic caused by `set_sjf` (the best action) | **+0.152** units | `healing_action_probe.csv`, `paired_closure_change`, 21 incidents |
| paired change caused by `set_round_robin` (the worst offered action) | −0.159 units | same |
| paired change caused by `set_priority` and by the five inert actions | +0.018 / 0.000 | same |
| spread of the statistic across control windows | 0.797 (sd) | training control, 3737 windows |
| spread after subtracting the phase model | **0.339 (sd)** | training control, R² = 0.819 |
| paired spread of the same statistic against its own no-action replay | 0.103 (sd) | `paired_closure_change`, validation probe |
| effect of the best action in units of the *online* residual | **0.45 sd** | derived from the two rows above |
| windows of the same action needed for 80 % power at α = 0.05 (two-sided, paired against a known baseline) | **≈ 39** | derived |
| genuinely helpful interventions that clear the declared null boundary | **6.1 %** (49 of 802, mean gain +244) | `verification_discrimination` |

So the recovery actions are *not* invisible to the monitored statistic — they are simply
smaller than the noise a single window carries online. The gap between the two spread rows is
the interesting number: the same statistic, compared against its own no-action replay, has a
spread of only 0.103, while the phase model still leaves 0.339 of online spread. Most of what
looks like noise is therefore workload and phase variation that the declared model does not
capture; a better phase model is the route to more certified recoveries, not a looser
boundary. With the model as declared, one window cannot turn a 0.45-sd signal into a reliable
certificate, and the honest engineering answer is the three-outcome rule: accept the extreme
tail, reject the clearly bad tail, and call everything else what it is — inconclusive.

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
| null closure p50 / p90 / p95 | 1.406 / 2.179 / 2.375 |
| declared `accept_closure_factor` (reporting, unconditional) | 2.4 |
| false-accept rate of that unconditional rule on the control | 0.045 |
| phase-baseline R² (variance of the null closure explained) | 0.819 |
| residual sd after the baseline (raw sd 0.797) | 0.339 (−57.5 %) |
| control residual p05 / p95 | −0.408 / +0.621 |
| derived / declared `accept_residual_factor` | 0.65 / 0.65 |
| derived / declared `reject_residual_factor` | −0.45 / −0.45 |
| measured false-accept / false-reject of the declared rule on the control | 0.047 / 0.039 |
| out-of-sample (validation control, 98 windows) false-accept / false-reject | 0.031 / 0.061 |

Both operative boundaries are therefore percentile points of the *same* control distribution
(95th and 5th, rounded outward), and the out-of-sample check stays within ±1.5 points of the
5 % design target on a split that took no part in fitting. Congestion-gate sweep (600 control
traces each): `min_load_factor` 0.00 → 1.517 events/trace, 0.50 → 1.475, 0.80 → 1.052,
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

Diagnoses on the held-out control: fairness starvation 132, priority-ordering anomaly 12,
head-of-line blocking 5, queue congestion 3; tail dispersion and `unexplained` never fired.

## 7. Reward versus measured truth

`healing_action_probe.csv` replays, on the validation split, **every** action at **every**
detected incident and compares what the controller would have recorded as reward with the
measured effect on total waiting time (per process): 21 incidents × 8 actions = 168 rows.

| Action | Recorded reward (band units) | Measured true effect (wait units / process) |
|---|---:|---:|
| `set_sjf` | 0.978 | **+3.625** |
| `set_priority` | 0.844 | +0.483 |
| `set_round_robin` | 0.666 | **−2.476** |
| `hold`, `set_fcfs`, `shrink_quantum`, `grow_quantum`, `revert` | 0.853 | 0.000 |

Two findings matter more than the pooled correlation:

* **the reward ranks actions correctly inside an incident**, which is the quantity learning
  actually needs: mean within-incident rank correlation between the recorded reward and the
  measured effect is 0.622 (median 0.875), and `argmax(reward)` is the truly best action in
  15 of 21 incidents. The pooled Pearson correlation is only 0.104 because different traces
  have wildly different scales; quoting it alone would understate the signal.
* **five of the eight actions are provably inert from the declared starting policy**: on an
  FCFS start `hold`, `set_fcfs`, `shrink_quantum`, `grow_quantum` and
  `revert_to_last_accepted` all have exactly zero effect. Only `set_sjf`, `set_priority` and
  `set_round_robin` can change anything, and `set_sjf` is the best action in 15 of the 21
  probed incidents. The learning problem is therefore much smaller than "8 actions" suggests.

## 8. What the learned controller saw (coverage)

| Training seed | Encoder states | Visited states | Visited pairs | Pairs of the declared table | Pairs of the reachable table | Incidents | Decisions with ≥ 2 known actions | Defaulted decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7101 | 648 | 32 | 60 | 1.16 % | 6.94 % | 773 | 89.0 % | 11.0 % |
| 7102 | 648 | 29 | 57 | 1.10 % | 6.60 % | 754 | 88.2 % | 11.8 % |
| 7103 | 648 | 30 | 64 | 1.23 % | 7.41 % | 772 | 91.2 % | 8.8 % |
| 7104 | 648 | 32 | 68 | 1.31 % | 7.87 % | 771 | 90.1 % | 9.9 % |
| 7105 | 648 | 30 | 60 | 1.16 % | 6.94 % | 745 | 87.5 % | 12.5 % |

* **The declared table is 648 states × 8 actions = 5184 pairs, but only 108 × 8 = 864 of them
  are reachable from the contexts the training data actually produces** (the `current_policy`
  dimension is always the starting FCFS because a change only survives if it is verified).
  Against that data-driven denominator, coverage is 6.6–7.9 % of reachable pairs instead of
  1.1–1.3 % of the declared table; both are reported.
* **Visits are concentrated**: the median visited state is seen 4–6.5 times and the most
  visited pair 235–256 times, i.e. a handful of cells carry most of the evidence while 11–16
  visited states ever saw a single action.
* **More training does not fix it**: the coverage curve (`healing_coverage_curve.csv`) grows
  41 → 55 → 59 → 60 visited pairs for seed 7101 as episodes go 300 → 600 → 900 → 1200, and the
  last 300 episodes add **one** new pair. The event supply, not the budget, is the binding
  constraint: a trace produces 0.62 incidents on average, so 1200 traces buy ≈ 750 decisions
  in a 648-cell space.
* **Only four of the eight actions are ever offered** from the FCFS start; the mutation mask
  removes the rest (`offered_actions_exercised = 4`, `chosen_actions_exercised = 4`).
* Consequence, stated as a limitation rather than fixed by a bigger algorithm: the learned arm
  is statistically indistinguishable from the hand-written waiting-time rule (§2) and its
  advantage over FCFS (−7.91) is the same advantage the rule gets (−8.15).

## 9. Worked demonstration trace

`healing_demo.json` (selected by a rule declared before the run: learned arm on the validation
split, preferring an episode with both an accepted and a non-accepted verdict; the episode is
identified by arm **and** training seed, and `_demo_payload` refuses to write the artifact if
the replay does not reproduce the recorded incident count).

| Step | Observed |
|---|---|
| workload | `long_jobs`, validation repetition 6, fingerprint `d1f128605bac62b4`, arm `q_recovery` seed 7101 |
| detection | incident at decision 7 (t = 173), `fairness_starvation`, window mean above `reference + 1.25·band` |
| decision | `set_sjf` — FCFS → SJF (subject to the mask) |
| verification | window closes with `closure_fraction = 2.665` against an expected closure of `2.008` for this trigger phase |
| verdict | **accepted** (`verified_phase_residual`), residual +0.658 vs the declared accept boundary +0.65 |
| reward | −0.059: the reward is the residual *minus the accept boundary*, so an accepted action normally sits just above zero |
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
