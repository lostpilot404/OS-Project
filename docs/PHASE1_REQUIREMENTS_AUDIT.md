# Phase 1 — Requirements Audit

**Project:** Reinforcement Learning–Based Workload-Aware CPU Scheduling with Dynamic Policy Selection
**Repository:** `lostpilot404/OS-Project`
**Branch:** `arena/9d830321-os-project` (baseline commit `3f9775e`)
**Audit status:** COMPLETE — with **16 blocking gaps** identified (see §4). No code has been written.
**Date:** 2026-10-08

---

## 1. Purpose and method

This audit extracts, from the supplied materials only, every explicit requirement of the project; it
then identifies missing, ambiguous, contradictory, and technically underspecified items. Per the
project's own rule ("NO ASSUMPTIONS"), nothing below is inferred or filled in by the auditor.

Every requirement is tagged `R-nn` and every gap `M-nn` so that later phases (design, tests,
implementation, validation) can be traced back to a line item here.

**Nothing in this document is invented, and no numeric design value has been chosen.**

---

## 2. Sources of truth — availability status

| Priority | Source | Status | Consequence |
|---|---|---|---|
| 1 | Explicit requirements in the current task prompt | **AVAILABLE** (full text) | Audited in full in §3 |
| 2 | Supplied project **presentation** | **MISSING** | Cannot be audited; likely holds scope detail, state variables, metrics |
| 2 | Supplied **Review 1** document | **MISSING** | Cannot be audited; likely holds review feedback / agreed parameters |
| 2 | Supplied **project review guideline** | **MISSING** | Cannot be audited; likely holds deliverable format and review criteria |
| 3 | Clarification from the project owner | Not yet obtained | Requested in §6 |
| 4 | General engineering judgment | Permitted **only** for implementation details that do not change project requirements | Not yet exercised |

### 2.1 Evidence of absence

The repository at the session baseline contains exactly one file:

```
$ git log --all --oneline --stat
3f9775e Initial commit
 README.md | 1 +
 1 file changed, 1 insertion(+)

$ cat README.md
# OS-Project
```

Additional checks performed:

| Check | Result |
|---|---|
| `find` for `.pdf`, `.docx`, `.ppt/.pptx`, `.md`, `.txt`, `.csv` anywhere under the workspace | Only the stub `README.md` |
| `git ls-remote --heads origin` | Single branch `main` at `3f9775e` — no other branches carrying materials |
| `git ls-remote --tags origin` | No tags |
| `git log --all` | One commit only |
| GitHub issues / PRs on `lostpilot404/OS-Project` | None |
| Files attached to this conversation | None |

**Conclusion:** all three "source of truth" documents (priority level 2) are unavailable. The
prompt repeatedly defers concrete parameters to them — e.g. *"The project materials identify
workload characteristics such as: CPU utilization, Queue length, Burst time, Waiting time as
candidate state information"* and *"The project materials identify the desired directions: lower
waiting time, …"* — so their absence removes the designated home of the quantitative
specification. This is gap **M-01** and it is blocking.

---

## 3. Explicit requirements extracted from the prompt

### 3.1 Scope and system (R-01 – R-07)

| ID | Requirement (as stated) |
|---|---|
| R-01 | The deliverable is a **Python-based CPU scheduling simulator**. |
| R-02 | A **Q-Learning agent** observes workload/system state and **dynamically selects** one of four conventional CPU scheduling policies. |
| R-03 | Project is an **academic Operating Systems + Reinforcement Learning** project. |
| R-04 | Scope is exactly: single-CPU scheduling simulation; synthetic process workloads; FCFS; SJF; Round Robin; Priority Scheduling; Q-Learning adaptive decision layer; dynamic policy selection based on workload/system state; performance comparison between the adaptive approach and conventional policies; Python implementation; NumPy, Pandas and Matplotlib *may* be used where appropriate. |
| R-05 | Explicit **non-goals**: Linux-kernel replacement, multi-core, real-time, energy-aware, cloud, edge, production OS scheduler, DQN/Deep RL, distributed scheduler. |
| R-06 | Scope must **not be expanded** without explicit approval. |
| R-07 | Conceptual flow is fixed as: Workload → Observe State → Q-Learning Agent → Select Scheduling Policy → Execute Scheduling Simulation → Measure Performance → Calculate Reward → Update Q-Table → Observe next state. |

### 3.2 Reinforcement learning (R-08 – R-11)

| ID | Requirement |
|---|---|
| R-08 | The RL agent is a **policy-selection layer only**; the four conventional algorithms must not be replaced by a fully learned scheduler unless explicitly instructed. |
| R-09 | Use **tabular Q-Learning** unless explicitly changed. |
| R-10 | The implementation must clearly define: state representation; state encoding/discretization; action space; reward function; Q-table; learning rate; discount factor; exploration strategy; training procedure; evaluation procedure. |
| R-11 | **Actions are exactly**: FCFS, SJF, Round Robin, Priority. |

### 3.3 Traditional schedulers (R-12 – R-14)

| ID | Requirement |
|---|---|
| R-12 | Implement the four algorithms **correctly and independently before** connecting them to RL. |
| R-13 | Each scheduler must: accept a well-defined process/workload representation; produce a **deterministic** result for deterministic input; calculate the required scheduling metrics; be **independently testable**; and contain **no RL-specific logic**. |
| R-14 | No duplicated scheduling logic anywhere in the codebase. |

### 3.4 Workload model (R-15 – R-17)

| ID | Requirement |
|---|---|
| R-15 | Process attributes are **Process ID, Arrival time, Burst time, Priority**; no extra process attributes unless necessary **and explicitly justified**. |
| R-16 | **Multiple workload conditions** must be supported (this is the premise of workload-aware policy selection). |
| R-17 | Candidate state information named by the materials: **CPU utilization, queue length, burst time, waiting time**. State variables must not be added to or removed from this set without **documenting the decision**. |

### 3.5 Metrics, comparison, experiment design (R-18 – R-22)

| ID | Requirement |
|---|---|
| R-18 | Evaluate: **waiting time, turnaround time, response time, CPU utilization, throughput, context switches** — using **correct standard definitions**, which must not be altered to improve appearances. |
| R-19 | The adaptive scheduler must be compared against **FCFS, SJF, Round Robin, Priority**. |
| R-20 | Baselines must receive **the same workload conditions**; the adaptive scheduler must not gain an advantage via different workloads, different inputs, or hidden preprocessing. |
| R-21 | Before implementing the final experiment pipeline, these ten items must be **explicitly defined**: (1) workload categories; (2) process-generation rules; (3) number of processes; (4) parameter ranges; (5) scheduling configuration; (6) RL hyperparameters; (7) training procedure; (8) evaluation procedure; (9) metrics; (10) repetitions/seeds if required. |
| R-22 | Missing values must **not** be invented — ask first. |

### 3.6 Experimental integrity (R-23)

| ID | Requirement |
|---|---|
| R-23 | Never fabricate results. Before implementation: no claims that one algorithm is better; no invented performance percentages, benchmark tables, or claims that Q-Learning improves performance. After implementation: results only from actual execution; configuration and random seed recorded; experiments reproducible; **expected** outcomes clearly distinguished from **measured** results. |

### 3.7 Architecture and code quality (R-24 – R-26)

| ID | Requirement |
|---|---|
| R-24 | Modular separation of: workload generation, scheduler implementations, RL logic, metrics, experiments, visualization, tests. The directory layout shown in the prompt is an **engineering proposal only**; unnecessary modules must not be created to mimic it. |
| R-25 | Production-quality academic code: clear names; type hints where useful; docstrings for public functions/classes; small focused functions; no duplicated scheduling logic; no hidden global state; **no hard-coded experimental values scattered through the code**; **centralized configuration**; deterministic execution when a seed is provided; useful exceptions and validation; **no dead code**; **no placeholder implementations disguised as completed functionality**; do not over-engineer. |
| R-26 | Observed runtime environment (fact, not a requirement): Python 3.11.2; `numpy`, `pandas`, `matplotlib`, `pytest` are **not installed**; `pip` can reach PyPI. Installing third-party packages therefore requires authorization. |

### 3.8 Testing and validation (R-27 – R-30)

| ID | Requirement |
|---|---|
| R-27 | Tests must cover: (1) process/workload validation; (2) FCFS correctness; (3) SJF correctness; (4) Round Robin correctness; (5) Priority correctness; (6) metric calculations; (7) state construction; (8) reward calculation; (9) Q-table updates; (10) action selection; (11) training behaviour; (12) reproducibility; (13) end-to-end execution. |
| R-28 | Edge cases to include: empty workload; single process; simultaneous arrivals; equal burst times; equal priorities; large burst times; processes arriving after CPU idle periods; Round Robin boundary conditions; zero/invalid values where applicable. |
| R-29 | Tests must not merely mirror the implementation incorrectly. |
| R-30 | Validation before completion: schedulers verified against **manually calculable cases**; all metric formulas verified; Q-Learning update logic verified **mathematically**; state/action dimensions verified; reward behaviour verified; reproducibility verified; evaluation verified to **not continue training**; verified that **no future information leaks into the decision state**; verified that the adaptive scheduler **actually uses the Q-Learning policy**; verified that **baselines receive equivalent workloads**. |

### 3.9 Visualization and documentation (R-31 – R-32)

| ID | Requirement |
|---|---|
| R-31 | Matplotlib used only where useful; comparison plots for waiting time, turnaround time, response time, CPU utilization, throughput, context switches; plus policy-selection behaviour across workload conditions. No decorative charts; **no plots containing fabricated values**. |
| R-32 | Professional README covering: problem; objective; scope; architecture; scheduling algorithms; RL approach; state; actions; reward; metrics; install; run; test; reproduce experiments; **actual results only after experiments are run**; limitations; future scope. No claim of real OS/kernel integration unless it exists. |

### 3.10 Prohibited additions without approval (R-33)

DQN; PPO; Actor-Critic; neural networks; GPU requirements; Docker; Kubernetes; cloud deployment;
web application; database; REST API; frontend; Linux kernel modifications; multi-core scheduling;
real-time scheduling; energy optimization; federated learning; distributed RL; additional
scheduling algorithms. (`R-05` restates several of these as scope exclusions.)

### 3.11 Process rules (R-34 – R-37)

| ID | Requirement |
|---|---|
| R-34 | Development must follow phases 1→6: requirements audit → design → implementation (schedulers → metrics → workload generation → RL state/action/reward → Q-Learning → adaptive selection → experiment pipeline → visualization) → testing → experimentation → final audit. |
| R-35 | **Stop and ask** when: a requirement is missing; two supplied documents conflict; a choice changes scope; a metric is ambiguous; an RL hyperparameter is unspecified; a workload-generation rule is unspecified; a scheduler variant is unspecified; a research claim cannot be verified; an external dependency is required but not authorized; a feature would materially change the project. |
| R-36 | The **final output** must contain: (1) project structure; (2) files created/modified; (3) what each major module does; (4) test results; (5) experiment configuration; (6) actual experimental results **if experiments were actually run**; (7) known limitations; (8) unresolved decisions awaiting approval. |
| R-37 | Optimise for **correctness, traceability, reproducibility, and strict adherence to supplied requirements** — not implementation speed. Never claim something is implemented, tested, or experimentally demonstrated when it is not; never fabricate citations, datasets, results, benchmarks, or research findings; never silently fill missing requirements with assumptions. |

---

## 4. Missing / ambiguous / underspecified items

Each item states the decision required, why it matters, and whether it blocks Phases 2–5.
"Blocking" means: implementation of the affected part cannot start without an answer.

| ID | Missing decision | Why it matters | Blocking? | Affects |
|---|---|---|---|---|
| **M-01** | **The three source-of-truth documents are absent** (presentation, Review 1, review guideline). | They are priority-2 requirements and the designated location of the quantitative spec. Prompt–document conflicts cannot even be checked. | **YES — highest** | Everything |
| **M-02** | **Round Robin time quantum.** | Changes every RR metric; forbidden assumption (§3 prompt list). | **YES** | RR, metrics, all comparisons |
| **M-03** | **Preemptive vs non-preemptive variant** for SJF (plain SJF vs SRTF) and for Priority (non-preemptive vs preemptive). | Different algorithms ⇒ different baselines ⇒ different conclusions. | **YES** | SJF, Priority, comparisons |
| **M-04** | **Decision-epoch model for "dynamic" policy selection**: is one policy chosen per workload before execution, or is the policy re-selected *during* execution? If during execution: at which epochs (arrival events / completions / quantum expiry / fixed intervals), and what happens to a running process when the policy changes (especially for non-preemptive policies)? | Defines the RL problem itself: what an episode is, how a state is observed mid-run, and whether policy switches can preempt. | **YES** | RL design, simulator core, experiments |
| **M-05** | **State representation details**: which of {CPU utilisation, queue length, burst time, waiting time} are used; are they *instantaneous* values sampled at a decision epoch or *workload-level aggregate* statistics; how many discretisation bins and what are the bin edges. | Determines Q-table size and whether future information leaks into the state (R-30). | **YES** | `state.py`, Q-table, RL validity |
| **M-06** | **RL hyperparameters**: learning rate α; discount factor γ; exploration strategy (ε-greedy? initial ε, decay schedule, floor) or alternative; number of training episodes; Q-table initialisation; update timing (per decision vs per episode). | Explicitly forbidden to invent. | **YES** | `q_learning.py`, training |
| **M-07** | **Reward function and weights**: how the six metrics/directions are combined into a scalar; weights for each; absolute vs relative-to-baseline normalisation; scale normalisation across heterogeneous metrics; per-decision vs per-episode reward. | The prompt states: *"Do not invent reward weights. If weighting is required and not specified, stop and ask."* | **YES** | `reward.py`, learning signal |
| **M-08** | **Workload generation rules**: number of workload categories and their definitions; number of processes per workload; arrival-time distribution and range (or all-at-zero); burst-time distribution and range; priority range and direction (is 1 the highest priority?); whether arrivals are guaranteed ≥ 0; seeds. | Explicitly forbidden to invent; determines the entire experiment. | **YES** | `generator.py`, experiments |
| **M-09** | **Training/evaluation protocol**: are training and evaluation workloads disjoint? How many evaluation workloads per condition? Number of random seeds / repetitions? Are results reported as mean (and spread)? Is the agent evaluated greedily? | Required by R-21 and by reproducibility (R-23). | **YES** | experiments, results |
| **M-10** | **Context-switch accounting convention**: is a context switch costless (counted only) or does it consume time (and if so, how much)? Does a switch count when the same process continues after its quantum? Does preemption count? Is the first dispatch counted? | Directly changes the "context switches" metric and the timing of every scheduler. | **YES** | all schedulers, metrics |
| **M-11** | **Tie-breaking rules**: SJF with equal burst times; Priority with equal priorities; simultaneous arrivals; whether ties break by arrival order, then PID. | Determinism (R-13) and reproducibility (R-23). | **YES** | all schedulers, tests |
| **M-12** | **Metric formula conventions**: acceptance of the standard textbook definitions — waiting = turnaround − burst (− switching overhead if modelled); turnaround = completion − arrival; response = first-dispatch − arrival; CPU utilisation = busy time / makespan; throughput = completed processes / makespan; averages over completed processes. | R-18 requires "correct standard definitions"; the auditor proposes the standard set for explicit confirmation rather than assuming. | Partially — confirm in Phase 2 sign-off | `metrics.py`, tests |
| **M-13** | **Dependency authorization**: may NumPy / Pandas / Matplotlib / pytest be installed from PyPI (currently absent)? Or must the project be standard-library-only? | External dependency requires authorization (R-35). Also determines the test runner. | **YES** | whole project |
| **M-14** | **Configuration mechanism**: format and location of the centralised configuration (e.g. dataclasses in `config.py`, JSON, TOML, YAML — the last would add a dependency) and whether experiment configurations are versioned alongside results. | R-25 requires centralized configuration; R-21 requires the experiment spec to be recorded. | No — propose in Phase 2 for sign-off | repo layout |
| **M-15** | **Reproducibility mechanism**: single global seed vs per-component seeds derived deterministically (e.g. `numpy.random.SeedSequence`); whether stochastic policy-selection randomness in training is seeded; whether generated workloads are stored as artefacts. | R-23/R-30 require reproducibility. Proposed in Phase 2. | No — propose in Phase 2 | experiments, tests |
| **M-16** | **Scope question — oracle/upper-bound baseline**: is a non-RL "oracle" (best fixed policy per workload) to be reported alongside the four conventional baselines? The prompt's scope lists only comparison against the four conventional policies, so adding an oracle would be a scope addition requiring approval. | Avoids adding unrequested scope; also frames how "adaptive" results are interpreted. | Requires a yes/no (can be answered any time before Phase 5) | experiments, README |

### 4.1 Items checked and found NOT to be conflicts

- Within the prompt itself, **no internal contradiction was found**. The stated non-goals (R-05),
  the prohibited additions (R-33), and the stated scope (R-04) are mutually consistent.
- The prompt's directory layout is explicitly labelled "a proposed engineering organization only",
  so it does not conflict with R-24's instruction not to create unnecessary modules.
- **Prompt-vs-document conflicts cannot be evaluated** because the documents are missing (M-01).
  Per the prompt's conflict rule, this must be reported rather than silently resolved — it is
  reported here and in §6.

---

## 5. What Phase 1 deliberately did NOT do

- No scheduler, RL, metric, workload, experiment, test, or visualisation code has been written.
  The repository still contains only `README.md`; this audit file is the single addition.
- No numeric value (quantum, α, γ, ε, episodes, process counts, ranges, seeds, weights, thresholds)
  has been chosen, proposed as a default, or embedded anywhere.
- No expected outcome, performance claim, or benchmark has been stated.

---

## 6. Blocking questions put to the project owner

| Question | Gap(s) resolved |
|---|---|
| Q1 — Supply the three project documents, or authorise an explicit prompt-only path? | M-01 |
| Q2 — What does "dynamic" mean: one selection per workload, or re-selection at defined decision epochs during execution? | M-04 |
| Q3 — Which scheduler variants: non-preemptive SJF and Priority, or SRTF / preemptive priority? | M-03 |
| Q4 — Who supplies the numeric specification (quantum, workload parameters, RL hyperparameters, episodes, seeds)? | M-02, M-06, M-08, M-09 |
| Q5 — Context-switch accounting: costless counter, or modelled time overhead; and counting convention? | M-10 |
| Q6 — Dependency authorization for NumPy / Pandas / Matplotlib / pytest (or standard-library-only)? | M-13, R-26 |

Still outstanding after that round and tracked for later sign-off: **M-05** (state/discretisation),
**M-07** (reward weights), **M-11** (tie-breaking), **M-12** (metric formulas), **M-14**
(configuration format), **M-15** (seed strategy), **M-16** (oracle baseline yes/no).

---

## 7. Resolution of this audit

**The project owner was asked about every blocking gap and then issued a follow-up
instruction authorising the documented-defaults path**: where an implementation detail was
genuinely unspecified, the simplest standard academic choice was to be used, kept
configurable, documented, and reported.  The instruction also fixed the decisions this audit
had flagged as blocking:

| Gap | Resolution |
|---|---|
| M-01 (documents missing) | Neither the presentation, Review 1 document nor review guideline was supplied; the prompt plus the authorising instruction is the requirements baseline, and every choice made in their absence is documented in `docs/DESIGN_AND_CHOICES.md`. |
| M-02 (quantum) | Fixed quantum of 4 time units, configurable (`SchedulerConfig.round_robin_quantum`). |
| M-03 (variants) | Non-preemptive SJF and non-preemptive Priority, as the follow-up instruction states. |
| M-04 (decision model) | One policy per workload, selected before execution (`docs/DESIGN_AND_CHOICES.md` §1). |
| M-05 (state) | Four pre-execution workload characteristics, 3 bins each, documented substitution for the measured candidates (§5). |
| M-06/M-07 (RL hyperparameters, reward weights) | Declared values in `config.py`, documented and untuned (§6–§7). |
| M-08/M-09 (workload rules, protocol) | Six declared conditions, 600 training episodes, 10 evaluation repetitions per condition, disjoint seeds (§9–§10). |
| M-10 (switching convention) | Costless by default, counted as changes of the running process (§3). |
| M-11/M-12 (tie-breaking, metric definitions) | Explicit total orders and textbook metric definitions (§2, §4). |
| M-13 (dependencies) | NumPy, pandas, Matplotlib and pytest approved; pinned in `requirements.txt`. |
| M-14/M-15 (config format, seeds) | Frozen dataclasses in `config.py`; SHA-256 child-seed derivation from master seeds. |
| M-16 (oracle baseline) | **Not implemented** — it would have been a scope addition; listed under future scope in the README. |

**The gate below is therefore satisfied by explicit authorisation rather than by supplying the
missing documents, and the audit is retained as the historical record of what was missing.**

## 8. Gate condition for Phase 2 (Design)

Phase 2 may begin when:

1. M-01 is resolved (documents supplied, or an authorised prompt-only path agreed);
2. M-02, M-03, M-04 are answered (they determine the simulator's core semantics);
3. M-06, M-07, M-08, M-09, M-10 are answered or explicitly delegated with approval;
4. M-13 is answered (otherwise no test runner and no plotting capability exist);
5. The remaining items (M-05, M-11, M-12, M-14, M-15, M-16) are either answered or scheduled for
   Phase-2 sign-off before the code they affect is written.

Until then, this audit is the only project artefact, and the codebase remains untouched.
