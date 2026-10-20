# Limitations

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

This file separates three things that are easy to confuse: what the code *measures*, what the
workload model *assumes*, and what is *unverified*. Nothing in this repository is a claim
about a real operating system.

## 1. The system does not touch a real operating system

* The scheduler drives `scheduler/runtime.py`, a **discrete-event simulator**. There is no
  kernel module, no syscall, no CFS/EEVDF integration, no hardware measurement. "CPU
  utilization", "context switch" and "waiting time" are simulator quantities.
* The simulator models one CPU, a single FIFO ready queue, non-preemptive FCFS/SJF/Priority,
  quantum-based Round Robin, a fixed switching cost and no I/O, no page faults, no interrupts,
  no multicore effects, no cache or NUMA effects.
* No experiment here was run on real workload traces. Every workload instance is generated
  from a family definition plus a seed.

## 2. Model assumptions that a real scheduler does not enjoy

* **Burst times are visible by default.** `RuntimeObservation` exposes each ready process's burst and
  remaining burst, so SJF and the diagnosis features are implementable. A real scheduler does
  not know these numbers; it would need predictors, and predictor error would change every
  number in `docs/EVALUATION_RESULTS.md`.  This is now **measurable**: set
  `HealingConfig(burst_estimator_enabled=True, burst_estimator_error_cv=0.30)` and the loop
  sees `healing/burst_estimator.py`'s modelled online EWMA per-bucket predictor plus log-normal multiplicative noise (`EWMA + exp(N(bias, cv²))`, **not** calibrated or validated against real traces and not claimed to be realistic)
  instead of ground truth (`perturb_observation`); evaluation still measures against truth so
  the cost of uncertainty can be quantified. Fixed SJF in the same experiment still receives exact burst lengths and is therefore an oracle baseline, not an equal-information comparison.  The frozen study leaves it off so committed
  artifacts stay byte-identical.
* **Two recovery actions are only meaningful under Round Robin.** `shrink_quantum` and
  `grow_quantum` are masked out (and report `changed=False`) whenever the live policy is not
  Round Robin, because they cannot change the schedule there.
* **The arrival process is stationary within a trace by default.** The traces used here have a burst
  backlog at the start and then drain; the detector's reference is a slow EWMA of that
  behaviour. Neither the model nor the detector claims to handle non-stationary real traffic.
  This too is now **injectable**: `workload/drift.py` (and `WorkloadGenerator.generate_with_drift`)
  inserts `burst_shock`, `arrival_storm`, `phase_transition` and `priority_inversion` at a
  declared fraction of the arrival span, preserving the pre-injection prefix so stable and
  drifted traces can be compared causally.  `experiments/healing_detection.py` already did
  this for measurement; `drift.py` promotes it to a first-class workload feature.
* **Switch cost is constant** (`switching_cost = 1` time unit) and there are no other
  adaptation costs (no migration, no cache invalidation, no re-tuning latency). Real recovery
  costs are action-specific and larger.
* **Reliability is no longer hidden in the mean.** Even with drift and burst error turned off,
  `evaluation/reliability.py` computes P95 waiting time, starvation count, Jain fairness index,
  recovery delay (decisions and time), false-intervention / rollback / regression rates,
  fallback rate, throughput and utilization from `ScheduleResult` + incident ledger + audit — so
  every future experiment that enables drift or the estimator can report them without changing
  the engine.

## 3. Measured weaknesses of the healing loop (with numbers)

* **The loop does not beat fixed SJF.** Final-test means: SJF 94.66 vs `rule_wait` 122.39 vs
  `q_recovery` 122.56 vs FCFS 130.84. The paired intervals exclude zero
  (see `docs/EVALUATION_RESULTS.md` §2), and the learned arm is statistically
  indistinguishable from the hand-written rule.
* **Verification can certify only an extreme tail of the genuinely helpful interventions.**
  At the declared 5 % false-accept budget on the control, *6.7 %* of the helpful interventions
  clear the declared null boundary at all (54 of 812 on the held-out split; their mean gain is
  +184.9 wait units), and 8.0 % clear the declared accept boundary. The measured reason: the
  best action moves the window statistic by **+0.116** in the canonical 98-incident probe (legacy 21-incident probe **+0.152**), which is **0.73** of the residual spread
  (0.160) the phase model leaves (legacy 0.95), so a single 5-decision window cannot certify an ordinary
  improvement — the same *≈ 15 windows* per action would be needed for 80 % power at α = 0.05 (legacy ≈ 39 for 0.95 sd).
  What one window *can* do is filter the tails: **65 of 65 accepted interventions helped** the
  trace (base rate 83.5 %) and 76.5 % of the rejected ones were harmful (base rate 14.0 %, a
  5.5× enrichment). Rank AUC of the residual for "this intervention helped" is 0.951; the raw
  closure statistic reaches 0.613.
* **`inconclusive` swallows most of the helpful interventions** (91.0 % for `rule_wait`, 92.1 %
  for `q_recovery`). The declared posture still rolls them back after the verification window,
  so the loop executes a change, benefits from it for 5–14 decisions, and then undoes it. The
  benefit that remains visible in the arm means (≈8.5 wait units over FCFS) is the residual of
  that behaviour, not the value of the actions themselves (the audit measures ≈+189 to +192 per
  intervention).
* **The verifier's accepted set is not guaranteed clean — it is a calibrated rate.** The accept
  boundary is the 95th percentile of the *no-action* residual, so any intervention whose window
  statistic happens to look like a no-action window clears it, whatever its end-of-trace effect
  was: on the retired `21977` stream 10 of the 34 accepted interventions were harmful (29.4 %),
  and the expected count for a harmful population that the window cannot separate is exactly
  the 5 % design budget. On the reported stream the same rule accepted 65 interventions and
  **none** of them was harmful, but that is one split's draw, not an invariant — the audit, not
  the verdict, is what establishes benefit. The `rule_wait_from_sjf` arm is the case where the
  safety posture is the only protection: every audited intervention was harmful (mean −169) and
  one window was rejected, which is what the failure counter saw. That is why every
  accept/reject number in this documentation is reported as a count *with* its denominator
  rather than as a rate.
* **The detector has a measurable false-trigger rate.** On the no-action control it declares
  0.62 events per trace on the training split and 0.65 on the held-out split. With a controlled
  step change injected into validation traces, recall is 75.0 % (`burst_shock`) and 56.4 %
  (`arrival_storm`) at a median delay of 6 decision intervals, *while 0.70 events per stable
  trace are still declared* — the false-trigger rate is the honest weak point of a
  reference-based detector. The reported "precision proxy" (0.657) is not a precision: the
  injection is a workload property, not a label of the event being detected. Raising the
  congestion gate to 1.25 removes the false triggers and with them almost all true detections
  (0.042 events/trace), so the trade-off is documented rather than optimised away.
* **The learned controller is undertrained, and more training does not fix it.** 1200 training
  traces per seed visit 29–33 of 648 states — 1.1–1.3 % of the declared 5184-pair table, and
  3.8–7.2 % of the pairs *reachable at all* from the training contexts (864 of them for four
  seeds, 1728 for seed 7104, whose traces expose a second starting-policy bin). The coverage
  curve is flat after ~900 episodes: over the last 300 episodes seed 7101 gains no new pair
  and the other seeds gain 1–4 (`healing_coverage_curve.csv`). 13–17 visited states ever saw a
  single action, and 9.1–13.8 % of decisions could only fall back to the initial value. The
  state space is larger than the event supply of this study; that is a limitation of the
  design, and it is reported rather than papered over with a more complex algorithm.
* **The reward is a weak proxy for the size of an effect, and a decent one for its sign.**
  Pooled Pearson correlation of the canonical `controller_reward` with the counterfactually measured per-process effect is **0.197** (pooled Spearman **0.223**; legacy raw proxy 0.104).
  Different traces have different scales. Within an incident, mean Spearman rank correlation of `controller_reward` is **0.449** (median **0.531**) and its argmax is the truly best action in **64 of 98** probed incidents (65.3 %; legacy raw proxy: 0.622 mean, 0.875 median, 15 of 21). Five of the eight actions have exactly zero effect from the declared start, so the
  learning problem is a three-way choice — canonical `set_sjf` best in 69, `set_round_robin` in 12, `set_priority` in 6, `hold` in 11 of 98 (legacy 15 of 21 for `set_sjf`).
* **The fairness-oriented rule is empirically wrong** on these families: mapping fairness
  starvation to Round Robin costs a mean 130.8 wait units per intervention on the held-out
  split (76.7 % false recoveries; 129.1 and 83.0 % on validation). It is kept in the study as a
  predeclared, honest negative result, not as a recommended method.
* **Higher p95 barely improves under recovery.** P95 waiting is 258.42 (FCFS/control) → 251.88
  (`rule_wait`) → 249.20 (SJF). Recovery moves the mean (≈8 units) slightly more than the tail
  (≈6.5 units, `rule_wait` − `detector_only`).
* **The safe fallback fired exactly once, and only in the arm that deserves it.** At most 6
  interventions per trace; adaptation is disabled after 2 consecutive *confirmed* failures, and
  only a `rejected` window counts as a confirmed failure — an `inconclusive` window never does,
  so ambiguity cannot disable the loop. On the held-out split that combination occurred once,
  in `rule_fairness` (repetition 21, second confirmed failure at decision 20 of episode 201;
  one fallback episode of 210 traces), and once on the validation split in the same arm — the
  arm whose interventions are mostly harmful (false-recovery rate 0.767). No other arm reached
  it, because none of them accumulated two confirmed failures in a trace. The mechanism is also
  exercised by unit tests (`tests/test_healing_controller.py`); the effect of one fallback on a
  whole trace is a single observation, not a measured distribution — a limitation, stated
  rather than implied.

## 4. Statistical scope

* Intervals are **conditional on the declared design**: five training seeds, seven synthetic
  families, 20 (validation) or 30 (final-test) repetitions per family. They describe
  variability over those seeds and traces, not over "all workloads".
* The bootstrap resamples model seeds and workloads in a family-stratified way
  (`evaluation/paired.py`); comparisons between arms that share training seeds are therefore
  paired, but the learned arms' five seeds are only five independent draws.
* The action-effect probe is a diagnostic on the validation split — canonical **98** probed incidents (784 rows); the legacy preserved file has 21 incidents (168 rows) — its correlations are reported as observations, not as estimates of a population quantity.
* `results/study/`, `results/runtime/` and `figures/study/` contain the preserved
  offline-study and runtime-appendix artifacts. They use their own seed streams and their own
  protocols; their numbers must not be mixed with the healing numbers. The legacy pipeline
  (`main.py experiment`) generates its output on demand and none of it is committed.

## 5. Unverified ideas (explicitly not claims)

* That the same loop would behave well on real workload traces, or on a kernel scheduler with
  burst prediction error.
* That a counterfactual-matched verifier would fix the certification problem online: the
  *offline* matched replay is implemented and is the audit's ground truth, but reproducing it
  at run time requires the no-action counterfactual of the same trace, which a live scheduler
  does not have. The measured gap between the phase model's residual spread (0.160) and the
  paired spread of the same statistic against its own hold run (0.103) is the size of the
  remaining headroom; closing it (for example with a per-trace drain model or several windows
  per action) is **not** implemented here, and the specification search that was run on
  train/validation (`healing_verification_models.csv`) is recorded, not hidden.  However,
  `evaluation/reliability.py` now makes the *consequence* of that headroom measurable
  (rollback rate, recovery delay, false-intervention rate) and `workload/drift.py` lets
  future work test whether richer references actually shrink it under workload shift.
* That the tri-state verdict generalises beyond the family mix tested: the accept/reject
  boundaries are percentiles of a control distribution measured on six training families, and
  `poisson_arrivals` (never seen in training) is the only distribution shift in the evaluation.
* That the safe fallback limits damage in practice: it triggered in this study, but its effect
  was not isolated in a dedicated ablation.
* That any of the diagnosis categories correspond to real scheduling pathology names: they are
  observable-evidence labels defined in `healing/diagnosis.py`, not a theory of kernel
  behaviour.

## 6. Reproducibility caveats

* Wall-clock timings in `RuntimeRun` (`observation_ns`, `selection_ns`, `wall_ns`) are measured
  but are **not** used as results; they vary between machines and are excluded from every
  table.
* Matplotlib figure rendering is optional (`--no-figures`); the tables and report are generated
  regardless, and the figures are rendered *from* the tables.
* `.venv/` is not part of the repository; recreate it from `requirements.lock` as described in
  `README.md`.
* The artifacts under `results/healing/` and `figures/healing/` belong to the frozen run on the
  held-out seed `23819`. They are evidence, not build output: regenerating them with a different
  configuration or seed replaces the reported results, so run experiments into a separate
  `--results-dir` unless the study is being re-frozen deliberately.

## 7. Release status

Four states are used: **Complete**, **Needs Fixing**, **Blocked**, **Not Verified**.

| Item | State | Basis |
|---|---|---|
| Full healing lifecycle (generate → simulate → monitor → detect → diagnose → decide → apply → verify → accept/rollback/fallback → record) | **Complete** | `healing/controller.py`, exercised end to end by `tests/test_healing_controller.py` and the study run |
| Tri-state verification (accepted / rejected / inconclusive) with declared, control-derived boundaries | **Complete** (implemented and tested); its *certification rate* is a research limitation | `healing/verifier.py`, `tests/test_healing_verification.py`, `healing_verification_models.csv` |
| Rollback, safe fallback and repeated-detection behaviour | **Complete** | unit-tested; the fallback fired on real traces (`rule_fairness`, held-out repetition 21, episode 201, decision 20, two consecutive confirmed failures; and once on validation) |
| Counterfactual audit (matched replay, prefix identity, separated denominators) | **Complete** | `healing_audit.csv` (`divergence_decision_index` equals the audited decision for all 1673 audited interventions), `tests/test_counterfactual.py` |
| Causal isolation of online decisions from future/counterfactual information | **Complete** | `tests/test_healing_verification.py` (a later arrival cannot change a closed incident); the audit is computed after the run only |
| Detector quality on controlled degradations | **Complete** (measured); the false-trigger rate itself is a limitation | `tests/test_healing_detection.py`, `healing_detection_scenarios.csv` |
| Q-learning coverage analysis | **Complete** (analysis) | `healing_coverage.csv`, `healing_coverage_curve.csv` |
| Reproducibility of the committed evidence | **Complete** | independent run into a separate directory reproduces 21/21 artifacts and 3/3 figures byte for byte |
| Documentation consistent with the generated artifacts | **Complete** | every quoted number re-read from `results/healing/` |
| The loop outperforming fixed SJF, or RL outperforming the predeclared rule | **Needs Fixing** (research level: action set, workload families and reward informativeness — no code defect identified) | `healing_comparisons.csv` |
| Certification of ordinary improvements (only 6.7 % of helpful interventions clear the accept boundary) | **Needs Fixing** (needs several windows per action or a per-trace drain model; measured, not implemented) | `healing_summary.json` (`verification_discrimination`) |
| Detector false triggers under stable conditions | **Needs Fixing** (documented trade-off against the congestion-gate sweep) | `healing_threshold_derivation.csv` (gate sweep), `healing_detection_scenarios.csv` |
| Tabular-Q coverage of the reachable state-action space | **Needs Fixing** (event supply, not training budget; flat after ~900 episodes) | `healing_coverage_curve.csv` |
| Real-OS integration, burst prediction, non-stationary arrivals, hardware measurement | **Not Verified** (out of scope; no claim made) | — |
| Generalisation beyond the seven synthetic families and the declared seed streams | **Not Verified** | — |
| Anything identified as blocked by compute or data | **Blocked** — nothing was blocked for the reported results | — |
