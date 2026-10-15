# Limitations

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic
Workloads**.  This page records what the system does **not** show and where its results
stop being valid.  It should be read together with `docs/EVALUATION_RESULTS.md`.

## Scope

* **Offline selection only.**  One policy per complete batch workload.  This is not a
  runtime OS scheduler: no within-schedule switching, no event loop, no kernel
  integration, no preemption decisions at runtime.  The preserved event-driven
  simulator (`docs/APPENDIX_RUNTIME_SIMULATOR.md`) is a separate appendix and none of
  its numbers appear in the project's results.
* **Synthetic, single-CPU, integer-time workloads** with known burst lengths at
  scheduling time (the textbook-SJF assumption).  No I/O, blocking, deadlines,
  multicore interference, priority aging, burst-prediction error or physical switch
  latency is modelled.  Context switches are counted, not timed (configured switch cost
  is zero); the risk objective prices them at a declared unit cost.

## Method

* **The risk-aware objective does not uniformly beat the original objective.**  At the
  frozen budget the two are statistically tied when the state is the original
  (ql_risk vs ql_orig: Δ ≈ 0.000 [−0.014, +0.014]) and the original-reward ablation is
  slightly ahead within the fingerprint state (Δ = +0.012 [0.007, 0.018]).  The
  finite-sample reason (the p95 tail term makes `−J` the noisier one-step signal) and
  the asymptotic argument (the `−J`-greedy policy is the J-optimal policy within the
  shared state class) are documented in `docs/EVALUATION_RESULTS.md` §3; we claim a
  measured comparison, not a win.
* **A predeclared rule beats all Q arms on mean cost** on these workload families
  (0.775 vs 0.784–0.825).  The families map almost 1:1 onto the optimal policy, which
  is exactly what a small decision list over two mechanism features can exploit.  The
  learned arms are competitive on hindsight-best rate and poor rate, but this result
  cautions against expecting tabular Q-learning to dominate cheap rules when the
  workload families are this structured.
* **Selection-rule difference between arms.**  `ql_orig` preserves the original
  selector's full-space argmax (zero-initialised ties fall to FCFS); the three new arms
  compare only visited actions (unvisited zero values would dominate non-positive `−J`
  rewards).  The 2×2 ablation among the new arms is clean (one shared rule), but
  ql_orig-vs-new comparisons carry this third factor.  It is declared here and in
  `docs/DESIGN_AND_CHOICES.md`.
* **State aliasing.**  The 81-state tabular abstraction is coarse: workloads in the
  same bin can want different policies (e.g. some `short_jobs` draws have head-of-line
  long jobs).  Irreducible per-state mixing limits every tabular arm; richer function
  approximation is out of scope.
* **Budget sensitivity.**  At 1200 episodes the risk-trained arms were clearly
  noisier; at 3600 the gaps largely closed.  Only these two budgets were observed
  during development; a systematic learning-curve study was not run.

## Evaluation

* **The held-out split is held out from tuning, not from the world.**  The final test
  seed (19501) was never generated before design freeze, but the workload *families*
  are the same synthetic generator families used in development; OOD claims are
  limited to the two held-out families (`poisson_arrivals`, `heavy_tail_stress`) and
  the stated distribution shifts.
* **Bootstrap CIs quantify workload sampling noise only**, not family-design
  uncertainty; results are conditional on the eight declared families and the declared
  objective weights.
* **Poor-selection rates depend on the declared objective.**  Regret is measured under
  `risk_aware_v1` (and reported under the original objective as well); a different
  weight vector would change every ranking.
* **Explanations are evidence transcripts, not causal proofs.**  They report what the
  computed numbers say (Q-value margins, term comparisons, post-hoc objectives); the
  "reasons" do not establish why the world behaves as it does.

## Engineering

* Tabular Q-values use Python/NumPy floats; episodes are terminal one-step updates, so
  no convergence guarantee is claimed.
* Per-selection explanation logs are large and gitignored (regenerable from the
  recorded seeds and CLI); the tracked examples cover three documented criteria.
* Controller-side wall time is not measured; this is an offline simulator study, so no
  runtime overhead claims are made.
