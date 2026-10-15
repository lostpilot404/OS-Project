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

* **The loop does not beat fixed SJF.** Final-test means: SJF 95.53 vs
  `rule_wait` 121.18 vs `q_recovery` 122.26 vs FCFS 128.88. The paired intervals exclude zero
  (see `docs/EVALUATION_RESULTS.md` §2).
* **The detector has a measurable false-trigger rate.** On the no-action control it declares
  0.62 events per trace on the training split and 0.69 on the held-out split. Most of them are
  diagnosed as fairness starvation while the initial backlog is at its worst; a reference-based
  detector cannot separate "this trace is hard" from "this configuration is wrong".
* **The in-run verification is conservative and mis-certifies both ways.** Its threshold is
  derived so that the false-accept rate under the control is 4.5 % (training) / 3.1 %
  (validation). The price is measured: it rejects 97.6 % (`rule_wait`) and 89.7 %
  (`q_recovery`) of the interventions that the counterfactual audit shows were helpful, and it
  accepted 1 harmful intervention in each of those arms. Single-trace verification cannot
  attribute causality when the monitored statistic decays after its own local maximum.
* **The learned controller is undertrained.** 1200 training traces per seed visit 30–33 of 648
  states (≈1.1 % of state-action pairs). The state space is larger than the trace budget of
  this study, so the RL arm's result is a statement about *this budget*, not about tabular
  Q-learning in general.
* **The reward is a weak proxy.** Pearson correlation with the counterfactually measured
  per-process effect is 0.10; the reward's argmax matches the truth's argmax in 15 of 21
  probed incidents. It ranks the best action (`set_sjf`) above the no-ops and the worst
  (`set_round_robin`) below them, and that is the extent of what can be claimed.
* **The fairness-oriented rule is empirically wrong** on these families: mapping
  fairness starvation to Round Robin costs a mean 147 wait units per intervention (80.1 %
  false recoveries). It is kept in the study as a predeclared, honest negative result, not as a
  recommended method.
* **Higher p95 does not improve under recovery.** P95 waiting is 256.01 (FCFS/control) →
  251.35 (`rule_wait`) → 248.96 (SJF). Recovery barely moves the tail; the mean moves more.
* **Intervention budget and fallback bound the experiment.** At most 6 interventions per trace
  and adaptation is disabled after 2 consecutive failed verifications. 20 of 210
  `rule_fairness` traces ended in the safe-fallback state; the learned arms ended in it 4
  times out of 1050. Enabling more adaptation is a design choice that was not tested.

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
* That a counterfactual-matched verifier (comparing the post-window statistic against a matched
  no-action baseline) would fix the false-reject problem: it is the natural next step, it is
  motivated by the measurements in `docs/EVALUATION_RESULTS.md` §3, and it has **not** been
  implemented or tested here.
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
