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

* **Burst times are visible.** `RuntimeObservation` exposes each ready process's burst and
  remaining burst, so SJF and the diagnosis features are implementable. A real scheduler does
  not know these numbers; it would need predictors, and predictor error would change every
  number in `docs/EVALUATION_RESULTS.md`.
* **Two recovery actions are only meaningful under Round Robin.** `shrink_quantum` and
  `grow_quantum` are masked out (and report `changed=False`) whenever the live policy is not
  Round Robin, because they cannot change the schedule there.
* **The arrival process is stationary within a trace.** The traces used here have a burst
  backlog at the start and then drain; the detector's reference is a slow EWMA of that
  behaviour. Neither the model nor the detector claims to handle non-stationary real traffic.
* **Switch cost is constant** (`switching_cost = 1` time unit) and there are no other
  adaptation costs (no migration, no cache invalidation, no re-tuning latency). Real recovery
  costs are action-specific and larger.

## 3. Measured weaknesses of the healing loop (with numbers)

* **The loop does not beat fixed SJF.** Final-test means: SJF 94.56 vs `rule_wait` 120.77 vs
  `q_recovery` 121.01 vs FCFS 128.93. The paired intervals exclude zero
  (see `docs/EVALUATION_RESULTS.md` §2), and the learned arm is statistically
  indistinguishable from the hand-written rule.
* **Verification can certify only an extreme tail of the genuinely helpful interventions.**
  At the declared 5 % false-accept budget on the control, *6.1 %* of the helpful interventions
  clear the null boundary at all (49 of 802 on the held-out split; their mean gain is +244 wait
  units). The measured reason: the best action moves the window statistic by +0.152, which is
  0.45 of the residual spread (0.339) that the phase model leaves, so a single 5-decision window
  cannot certify an ordinary improvement — roughly 39 windows of the same action would be
  needed for 80 % power at α = 0.05. What one window *can* do is filter the tails: 24 of 34
  accepted interventions helped the trace (base rate 82.3 %) and 46.9 % of the rejected ones
  were harmful (base rate 15.9 %). Rank AUC of the residual for "this intervention helped" is
  0.720; the raw closure statistic reaches 0.610.
* **`inconclusive` swallows most of the helpful interventions** (95.6 % for `rule_wait`, 97.2 %
  for `q_recovery`). The declared posture still rolls them back after the verification window,
  so the loop executes a change, benefits from it for 5–14 decisions, and then undoes it. The
  benefit that remains visible in the arm means (≈8 wait units over FCFS) is the residual of
  that behaviour, not the value of the actions themselves (the audit measures ≈+179 to +185 per
  intervention).
* **The verifier also accepted harmful changes**: 3 of the 6 `rule_wait_from_sjf` interventions
  (an arm whose every audited intervention was harmful, mean −145) were accepted, and 1
  `rule_fairness` and 4 `q_recovery` accepted interventions were harmful. With 34 accepted
  interventions in total, 8 were wrong — small numbers, and the reason they are reported as
  counts with their denominator rather than as rates.
* **The detector has a measurable false-trigger rate.** On the no-action control it declares
  0.62 events per trace on the training split and 0.66 on the held-out split. With a controlled
  step change injected into validation traces, recall is 75.0 % (`burst_shock`) and 56.4 %
  (`arrival_storm`) at a median delay of 6 decision intervals, *while 0.70 events per stable
  trace are still declared* — the false-trigger rate is the honest weak point of a
  reference-based detector. The reported "precision proxy" (0.657) is not a precision: the
  injection is a workload property, not a label of the event being detected. Raising the
  congestion gate to 1.25 removes the false triggers and with them almost all true detections
  (0.042 events/trace), so the trade-off is documented rather than optimised away.
* **The learned controller is undertrained, and more training does not fix it.** 1200 training
  traces per seed visit 29–32 of 648 states — 6.6–7.9 % of the state-action pairs that are
  *reachable at all* from the training contexts (864 of the declared 5184, because the
  `current_policy` dimension is always the starting FCFS). The coverage curve is flat after
  ~900 episodes: seed 7101 gains one new state-action pair between episodes 900 and 1200
  (`healing_coverage_curve.csv`). 11–16 visited states ever saw a single action, and 8.8–12.5 %
  of decisions could only fall back to the initial value. The state space is larger than the
  event supply of this study; that is a limitation of the design, and it is reported rather
  than papered over with a more complex algorithm.
* **The reward is a weak proxy for the size of an effect, and a decent one for its sign.**
  Pooled Pearson correlation with the counterfactually measured per-process effect is 0.10
  (different traces have different scales); within an incident, the mean rank correlation is
  0.622 (median 0.875) and the reward's argmax is the truly best action in 15 of 21 probed
  incidents. Five of the eight actions have exactly zero effect from the declared start, so the
  learning problem is a three-way choice.
* **The fairness-oriented rule is empirically wrong** on these families: mapping fairness
  starvation to Round Robin costs a mean 126 wait units per intervention (80.9 % false
  recoveries). It is kept in the study as a predeclared, honest negative result, not as a
  recommended method.
* **Higher p95 barely improves under recovery.** P95 waiting is 258.22 (FCFS/control) → 252.66
  (`rule_wait`) → 248.10 (SJF). Recovery moves the mean (≈8 units) more than the tail (≈5.5).
* **The safe fallback never fired in the frozen study.** At most 6 interventions per trace;
  adaptation is disabled after 2 consecutive *confirmed* failures, and only a `rejected`
  window counts as a confirmed failure — an `inconclusive` window never does, so ambiguity
  cannot disable the loop. Reaching the fallback therefore requires two consecutive confirmed
  failures in one trace, which the 210-trace held-out split never produced in any arm (0 of
  210 traces; `healing_detection_summary.csv`). The mechanism is exercised by unit tests
  (`tests/test_healing_controller.py`), but its effect on a whole trace is not measured — a
  limitation, stated rather than implied.

## 4. Statistical scope

* Intervals are **conditional on the declared design**: five training seeds, seven synthetic
  families, 20 (validation) or 30 (final-test) repetitions per family. They describe
  variability over those seeds and traces, not over "all workloads".
* The bootstrap resamples model seeds and workloads in a family-stratified way
  (`evaluation/paired.py`); comparisons between arms that share training seeds are therefore
  paired, but the learned arms' five seeds are only five independent draws.
* The action-effect probe is a diagnostic on the validation split with 21 probed incidents; its
  correlations are reported as observations, not as estimates of a population quantity.
* `results/study/`, `results/runtime/` and `figures/study/`, `figures/` contain the preserved
  offline-study and runtime-appendix artifacts. They use their own seed streams and their own
  protocols; their numbers must not be mixed with the healing numbers.

## 5. Unverified ideas (explicitly not claims)

* That the same loop would behave well on real workload traces, or on a kernel scheduler with
  burst prediction error.
* That a counterfactual-matched verifier would fix the certification problem online: the
  *offline* matched replay is implemented and is the audit's ground truth, but reproducing it
  at run time requires the no-action counterfactual of the same trace, which a live scheduler
  does not have. The measured gap between the phase model's residual spread (0.339) and the
  paired spread of the same statistic against its own hold run (0.103) is the size of the
  remaining headroom; closing it (for example with a better phase model or several windows per
  action) is **not** implemented here.
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
