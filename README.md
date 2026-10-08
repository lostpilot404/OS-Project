# Reinforcement Learning–Based Workload-Aware CPU Scheduling with Dynamic Policy Selection

A Python simulator of a **single CPU** in which a **tabular Q-learning** agent observes the
profile of a workload and **selects one of four conventional CPU scheduling policies** —
First-Come First-Served, Shortest Job First, Round Robin or Priority — for that workload.
The four policies are implemented independently and correctly; the agent is a
policy-selection layer on top of them, never a replacement for them.

The core contribution is:

```
Workload → State → Q-Learning → Select FCFS / SJF / Round Robin / Priority
        → Evaluate → Reward → Learn
```

Everything reported under [Results](#17-results) was produced by running this code
(`python3 main.py experiment`, ~7 s).  No value in this repository was entered by hand.

---

## 1. Problem statement

Conventional CPU scheduling policies are each optimal in a different regime.  Non-preemptive
Shortest Job First minimises mean waiting time when jobs can be compared, but it gives no
responsiveness to short jobs that arrive behind a long one.  Round Robin bounds response time
but pays for it in context switches and in a higher mean waiting time.  Priority scheduling
respects the importance of jobs but ignores their length, and FCFS is order-fair and otherwise
weak.

A real workload mix contains all of these regimes.  The question this project asks is
therefore: *given a workload, can a reinforcement-learning agent learn which conventional
policy serves it best, using only information available before the workload runs?*

## 2. Motivation

* Scheduling is the classical OS problem where a small policy difference changes every
  performance metric, which makes it a good testbed for learning-based control.
* Reinforcement learning suits the problem because the best policy depends on the workload
  distribution, which is not known analytically and varies over time.
* Keeping the conventional algorithms intact and learning only *which* one to apply produces
  a system whose behaviour stays explainable: every schedule is a textbook schedule.

## 3. Objectives

1. Implement FCFS, SJF, Round Robin (configurable quantum) and non-preemptive Priority
   scheduling correctly, deterministically and independently.
2. Compute the six required metrics with standard definitions.
3. Generate reproducible synthetic workloads from several workload conditions that span the
   regimes in which the four policies trade off.
4. Build a tabular Q-learning agent that selects a policy from a discretised workload state.
5. Verify, **before training**, which policy each workload class actually favours.
6. Compare the adaptive scheduler with each conventional policy **on identical, held-out
   workloads**.
7. Report only measured results, with the configuration and seeds recorded.

## 4. Scope

**In scope:** single-CPU simulation; abstract integer time units; synthetic workloads;
the four policies above; tabular Q-learning over the four actions; metric computation,
comparison tables and Matplotlib figures.

**Out of scope (not implemented, not claimed):** Linux kernel integration, multi-core or
distributed scheduling, real-time scheduling, energy-aware scheduling, cloud/edge
scheduling, deep RL (DQN, PPO, actor-critic) or any other learning algorithm, real process
traces, databases, REST APIs, web front ends, Docker or Kubernetes, and any use as a
production scheduler.

An earlier version of this project also contained a *Round-Robin quantum controller*
(a second learning agent that multiplied the Round-Robin quantum).  That extension was
**removed**: it was not part of the original four-policy adaptive scheduler.  The
Round-Robin **quantum parameter** (4 time units) stays — it is part of Round Robin.

## 5. System architecture

```
OS-Project/
├── main.py                     command-line entry point (experiment | train)
├── config.py                   every tunable value, in frozen dataclasses
├── errors.py                   ValidationError / ConfigurationError
├── workload/
│   ├── models.py               Process, Workload, ExecutionSlice, ProcessOutcome, ScheduleResult
│   └── generator.py            WorkloadGenerator: nine workload conditions
├── scheduler/
│   ├── base.py                 Timeline, SchedulingPolicy, shared non-preemptive engine
│   ├── fcfs.py  sjf.py  round_robin.py  priority.py
├── evaluation/
│   ├── metrics.py              the six metrics, defined once
│   └── comparison.py           summary and comparison tables for the experiment
├── rl/
│   ├── state.py                StateSnapshot, binners, StateEncoder, observe_workload_state
│   ├── reward.py               reward equation and its breakdown
│   ├── q_learning.py           QLearningAgent + epsilon schedule
│   └── adaptive.py             AdaptiveScheduler: the learning loop
├── experiments/
│   ├── verify_classes.py       pre-training verification of the workload classes
│   ├── train.py                training loop and history
│   ├── evaluate.py             evaluation protocol and integrity checks
│   └── run_experiment.py       the full study, artefact writing
├── visualization/plots.py      Matplotlib figures (Agg backend)
├── tests/                      296 tests
├── docs/
│   ├── PHASE1_REQUIREMENTS_AUDIT.md   requirements audit performed before coding
│   └── DESIGN_AND_CHOICES.md          every open decision, documented
├── results/                    measured results of the committed run
├── figures/                    figures of the committed run
└── requirements.txt
```

Data flows one way: a workload is generated, its profile is encoded into a state, the agent
selects an action, the corresponding policy produces a `ScheduleResult`, metrics are computed
from that result, the reward is computed against the four conventional policies on the same
workload, and the Q-table is updated.  No module writes to another module's state, and no
value is shared through globals.

## 6. Scheduling algorithms

| Policy | Variant | Selection rule (ties broken as shown) |
|---|---|---|
| FCFS | non-preemptive | smallest `(arrival_time, pid)` |
| SJF | non-preemptive (no SRTF) | smallest `(burst_time, arrival_time, pid)` |
| Round Robin | preemptive, quantum = 4 | FIFO ready queue, `min(quantum, remaining)` per turn |
| Priority | non-preemptive, 1 = highest | smallest `(priority, arrival_time, pid)` |

A context switch is a change of the running process; the first dispatch is not counted, and a
process that keeps the CPU across a quantum boundary is not switched out.  Switching is
costless by default (`switching_cost = 0`) and merely counted; a positive cost is supported
and tested.  Full semantics — including the arrival convention at quantum boundaries — are in
[`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md) §2.

## 7. Q-Learning approach

Tabular Q-learning, one decision per workload (see
[`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md) §1).  One episode is one workload:

```
workload -> observe state -> ε-greedy action -> run that policy -> metrics -> reward vs
four conventional policies -> Q(s,a) += α·(reward − Q(s,a)) -> next episode
```

Training (`run_training_episode`) explores with the ε schedule and updates the Q-table;
evaluation (`run_evaluation`) is greedy and provably never updates it (the Q-table and the
visit counts are compared before and after, and the run fails if they changed).

## 8. Workload conditions

Nine synthetic classes, generated by declared rules (never hand-tuned per result).  The
classes were rebuilt so that the regimes in which the four policies trade off are actually
present; the pre-training verification (§16.1) measures which policy each class favours
**before** the agent is trained.

| Class | Jobs | Bursts | Arrivals | Priorities | Reward-optimal policy (measured) |
|---|---|---|---|---|---|
| `short_batch` | 15 | uniform 1–4 | batch (all at t = 0) | uniform 1–5 | **SJF** (100% of probes) |
| `short_stream` | 15 | uniform 1–4 | uniform over [0, 30] | uniform 1–5 | **SJF** (100%) |
| `long_batch` | 15 | uniform 20–50 | batch (all at t = 0) | uniform 1–5 | **SJF** (100%) |
| `interactive` | 20 | 90% short 1–2, 10% long 60–100 | long jobs in [0, 2], short jobs over [0, 50] | uniform 1–5 | **Round Robin** (68%) |
| `interactive_sparse` | 20 | 90% short 1–2, 10% long 60–100 | long jobs in [0, 2], short jobs over [0, 110] | uniform 1–5 | **Round Robin** (60%) |
| `priority_aligned` | 15 | uniform 1–50 | uniform over [0, 20] | by burst rank: shortest = highest priority | **SJF** (100%) |
| `priority_skewed` | 15 | uniform 1–50 | uniform over [0, 20] | 70% in the band 1–2 | **SJF** (100%) |
| `quantum_sensitive` | 20 | uniform 2–12 (straddles the quantum 4) | uniform over [0, 30] | uniform 1–5 | **SJF** (100%) |
| `mixed` | 15 | uniform 1–50 | uniform over [0, 20] | uniform 1–5 | **SJF** (100%) |

The mechanism that separates SJF from Round Robin is **whether short jobs arrive while a
long job is running**: in the interactive classes a few long background jobs are released at
the head of the window while very short jobs keep streaming in, so the non-preemptive
policies block behind a long job and Round Robin's preemption wins waiting time as well as
response time.  The control classes (`short_stream`, `quantum_sensitive`) have staggered
arrivals but no long jobs: Round Robin degenerates to FCFS or pays heavily in context
switches, and SJF wins.  The priority classes test that the agent does not learn the naive
"priorities present → pick Priority" rule.

## 9. State representation

Seven **pre-execution** workload characteristics, discretised into 3 bins each →
**2187 states**.  CPU utilisation, queue length and waiting time cannot form the state
because they are *measured* quantities (the decision precedes execution), so their
pre-execution analogues are used; this substitution is documented in
[`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md) §4.

| Variable | Meaning | Bins (low → high) |
|---|---|---|
| `burst_profile` | median burst time (log bins) | < 4.64, 4.64–21.5, ≥ 21.5 |
| `burst_dispersion` | coefficient of variation of bursts | < 1.0, 1.0–2.0, ≥ 2.0 |
| `long_job_share` | share of total burst time from jobs ≥ 16 (= 4 × quantum) | < ⅓, ⅓–⅔, ≥ ⅔ |
| `arrival_concentration` | fraction of jobs at the modal arrival time | < ⅓, ⅓–⅔, ≥ ⅔ |
| `offered_load` | total burst time ÷ `max(1, arrival span)` (log bins) | < 2.46, 2.46–12.1, ≥ 12.1 |
| `priority_spread` | σ of priorities | < 0.667, 0.667–1.333, ≥ 1.333 |
| `priority_burst_alignment` | Pearson correlation of priority number and burst time | < −⅓, −⅓–⅓, ≥ ⅓ |

Encoded as a mixed-radix index with `burst_profile` most significant.  The state contains no
scheduling result and no metric, so it cannot leak the outcome of a decision.

## 10. Action space

| Action | Policy |
|---|---|
| 0 | FCFS |
| 1 | SJF |
| 2 | Round Robin |
| 3 | Priority |

## 11. Reward function

```
term_m = min(2.0, metric_m / reference_m)      reference_m = mean of metric m over the four
                                               conventional policies on the same workload
reward = Σ_benefits w_m (term_m − 1) + Σ_costs w_m (1 − term_m)

benefits : CPU utilisation 0.075, throughput 0.075
costs    : waiting 0.30, turnaround 0.25, response 0.20, context switches per process 0.10
```

`reward = 0` means "exactly as good as the average conventional policy on this workload";
positive means better.  Weights were declared before the first experiment and **not**
changed when the workload classes and the state were redesigned — the redesign changed
*where* the policies differ, not what "better" means
([`docs/DESIGN_AND_CHOICES.md`](docs/DESIGN_AND_CHOICES.md) §5).  A structural
consequence, measured and reported rather than engineered around: because the reward uses
unweighted means, **Priority can at best tie SJF** and **FCFS is essentially never** the
reward-argmax (§17.6).

## 12. Metrics

| Metric | Definition |
|---|---|
| Waiting time | `turnaround − burst`, mean over processes |
| Turnaround time | `completion − arrival`, mean over processes |
| Response time | `first execution − arrival`, mean over processes |
| CPU utilisation | `100 × busy time / total elapsed time` |
| Throughput | `processes / total elapsed time` |
| Context switches | changes of the running process; also reported per process |

## 13. Installation

Python 3.10+ is required (3.11 was used).  Dependencies: NumPy, pandas, Matplotlib, pytest.

```bash
python3 -m venv .venv && . .venv/bin/activate   # recommended, keeps the system clean
pip install -r requirements.txt
```

On a Debian/Ubuntu system with an externally managed Python, `pip install --break-system-
packages -r requirements.txt` also works, at the cost of installing into the system Python.

## 14. Usage

```bash
python3 main.py experiment                 # verify → train → evaluate → tabulate → plot
python3 main.py train                      # train only, print the training history
python3 main.py experiment --no-figures    # skip the figures
python3 main.py experiment --results-dir my_results --figures-dir my_figures
```

`experiment` runs the whole study in a fixed order and writes every artefact:

```
results/config.json              exact configuration of the run (all hyperparameters, seeds)
results/class_verification.json  pre-training verification of the workload classes
results/training_history.json    per-episode training record
results/q_table.json             the learned Q-table with state bins and visit counts
results/workloads.csv            every evaluated workload and its fingerprint
results/metrics.csv              one row per (workload, scheduler, regime)
results/decisions.csv            one row per adaptive decision (incl. reward-argmax comparison)
results/summary.json             all tables and headline numbers
figures/*.png                    six figures
```

## 15. Testing

```bash
python3 -m pytest -q          # 296 tests, ~17 s
```

The suite covers: the four schedulers' correctness (hand-computed traces, invariants,
edge cases — unchanged since the first version); metric definitions; workload generation
rules (batch-head arrivals, long-mode bound, burst-aligned priorities); state construction
and discretisation (including that the interactive classes occupy states disjoint from the
no-long-job control classes); the reward arithmetic and validation; the Q-learning agent;
the adaptive scheduler (learning, greedy evaluation, reproducibility); the pre-training
verification (structure, stream disjointness, and the design property that the default
classes have at least two distinct modal winners); the experiment design (quantum
controller removed, reward weights unchanged, workload-aware selection end-to-end,
held-out evaluation, identical workloads for all baselines, state coverage); the
comparison tables; and the full pipeline (artefacts, byte-identical re-runs, CLI).

## 16. Experiment protocol

```
1. VERIFY (pre-training)   225 probe workloads (25 per class, seed 777): run all four
                           policies on each, compute every action's reward, record the
                           reward-argmax per class, and measure whether the reward-argmax
                           is a consistent function of the encoded state.
2. TRAIN                   5400 episodes (600 per class, seed 42), one workload per episode.
3. EVALUATE                90 held-out workloads (10 per class, seed 2024): all four
                           baselines + the greedy adaptive scheduler on identical workloads.
```

Three master seeds, all distinct (enforced by configuration validation): training 42,
evaluation 2024, verification 777.  Runtime integrity checks (the run fails loudly if any is
violated): no evaluation workload was used during training; the Q-table and visit counts
are unchanged by evaluation; every evaluation decision is greedy; all schedulers of one
workload carry the same fingerprint; the adaptive scheduler's metrics equal the chosen
baseline's metrics on the same workload.

### 16.1 Pre-training verification — do the classes favour different policies?

Measured on 225 probe workloads **before** any training (winner = reward-argmax):

| Class | FCFS | SJF | Round Robin | Priority | Modal winner |
|---|---|---|---|---|---|
| `short_batch` | 0% | **100%** | 0% | 0% | SJF |
| `short_stream` | 0% | **100%** | 0% | 0% | SJF |
| `long_batch` | 0% | **100%** | 0% | 0% | SJF |
| `interactive` | 0% | 32% | **68%** | 0% | **Round Robin** |
| `interactive_sparse` | 16% | 24% | **60%** | 0% | **Round Robin** |
| `priority_aligned` | 0% | **100%** | 0% | 0% | SJF |
| `priority_skewed` | 0% | **100%** | 0% | 0% | SJF |
| `quantum_sensitive` | 0% | **100%** | 0% | 0% | SJF |
| `mixed` | 0% | **100%** | 0% | 0% | SJF |

Two distinct modal winners exist (SJF and Round Robin), the interactive classes favour
Round Robin by a wide margin, and **96.0%** of probe workloads have a reward-argmax equal
to the modal argmax of their encoded state — the state carries the information the agent
needs.  Full table: `results/class_verification.json`.

## 17. Results

All numbers are from the committed run (`python3 main.py experiment`; seeds 42 / 2024 /
777) and are reproduced in `results/summary.json`.

### 17.1 Training (5400 episodes, seed 42)

* mean reward: **+0.0036** over the first 100 episodes → **+0.2270** over the last 100
  (the agent learns); exploration rate 15.2%; final ε = 0.05.
* action selections (all episodes): SJF 3828, Round Robin 961, FCFS 345, Priority 266 —
  exploration visited all four actions.
* action selections (greedy only): SJF 3767, Round Robin 749, FCFS 61, Priority 0.
* **Q-table states visited: 80 of 2187** (the reachable region of the nine classes).

### 17.2 Evaluation (90 held-out workloads, seed 2024)

Overall means (all 90 workloads):

| Metric | FCFS | SJF | Round Robin | Priority | **Adaptive** |
|---|---|---|---|---|---|
| mean waiting time | 125.79 | 80.83 | 142.19 | 112.28 | **77.06** |
| mean turnaround time | 142.24 | 97.28 | 158.64 | 128.73 | **93.51** |
| mean response time | 125.79 | 80.83 | **19.08** | 112.28 | 73.43 |
| CPU utilisation (%) | 97.97 | 97.97 | 97.97 | 97.97 | 97.97 |
| throughput | 0.1421 | 0.1421 | 0.1421 | 0.1421 | 0.1421 |
| context switches | 15.67 | 15.67 | 71.19 | 15.67 | 24.44 |

The adaptive scheduler beats every baseline on waiting time (1.32× vs SJF, 1.59× vs Round
Robin, 2.32× vs FCFS, 2.00× vs Priority, workload by workload), on turnaround time
(1.17–1.90×), and on response time vs FCFS/SJF/Priority (1.91–4.02×).  Round Robin still
wins overall response time (19.08 vs 73.43): the declared reward prices waiting +
turnaround (0.55) above response (0.20), so the agent trades response for waiting on the
batch-like classes.  Utilisation and throughput are identical for all policies on every
workload — a structural property of work-conserving schedulers with costless switching —
and are reported as such.

### 17.3 Policy selected by the agent, per workload condition

| Class | FCFS | SJF | Round Robin | Priority | Mean reward | Agreement with reward-argmax |
|---|---|---|---|---|---|---|
| `short_batch` | 0 | **10** | 0 | 0 | +0.155 | 10/10 |
| `short_stream` | 0 | **10** | 0 | 0 | +0.129 | 10/10 |
| `long_batch` | 0 | **10** | 0 | 0 | +0.162 | 10/10 |
| `interactive` | 0 | 2 | **8** | 0 | +0.406 | 8/10 |
| `interactive_sparse` | 0 | 0 | **10** | 0 | +0.440 | 6/10 |
| `priority_aligned` | 0 | **10** | 0 | 0 | +0.176 | 10/10 |
| `priority_skewed` | 0 | **10** | 0 | 0 | +0.234 | 10/10 |
| `quantum_sensitive` | 0 | **10** | 0 | 0 | +0.195 | 10/10 |
| `mixed` | 0 | **10** | 0 | 0 | +0.216 | 10/10 |

**Overall agreement with the per-workload reward-argmax: 84/90 (93.3%).**  The agent's
per-class selection matches the pre-training verification's modal winner for **9 of 9
classes**.  In `interactive`, the two SJF selections are the held-out draws that contained
no long job — the state-visible case where SJF is optimal.  The 6 disagreements are all in
the two interactive classes and are all near-ties (SJF ahead of Round Robin by 0.017–0.121
on those specific draws; the class-dominant policy is Round Robin).

Per-class mean waiting / response / switches (10 held-out workloads per class):

| Class | FCFS | SJF | Round Robin | Priority | **Adaptive** |
|---|---|---|---|---|---|
| `short_batch` | 17.69 / 17.69 / 14 | **12.84** / 12.84 / 14 | 17.69 / 17.69 / 14 | 17.26 / 17.26 / 14 | **12.84** / 12.84 / 14 |
| `short_stream` | 4.93 / 4.93 / 14 | **3.73** / 3.73 / 14 | 4.93 / 4.93 / 14 | 4.87 / 4.87 / 14 | **3.73** / 3.73 / 14 |
| `long_batch` | 252.94 / 252.94 / 14 | **219.35** / 219.35 / 14 | 433.88 / 28.00 / 141 | 251.87 / 251.87 / 14 | **219.35** / 219.35 / 14 |
| `interactive` | 168.15 / 168.15 / 19 | 53.22 / 53.22 / 19 | **34.68** / 11.65 / 65 | 128.04 / 128.04 / 19 | **35.89** / 18.23 / 57 |
| `interactive_sparse` | 126.61 / 126.61 / 19 | 39.99 / 39.99 / 19 | **23.32** / 8.31 / 60 | 89.20 / 89.20 / 19 | **23.32** / 8.31 / 60 |
| `priority_aligned` | 167.65 / 167.65 / 14 | **119.22** / 119.22 / 14 | 231.67 / 25.07 / 101 | 120.71 / 120.71 / 14 | **119.22** / 119.22 / 14 |
| `priority_skewed` | 166.79 / 166.79 / 14 | **114.39** / 114.39 / 14 | 220.19 / 24.28 / 100 | 166.95 / 166.95 / 14 | **114.39** / 114.39 / 14 |
| `quantum_sensitive` | 49.63 / 49.63 / 19 | **36.34** / 36.34 / 19 | 67.86 / 27.17 / 41 | 49.69 / 49.69 / 19 | **36.34** / 36.34 / 19 |
| `mixed` | 177.70 / 177.70 / 14 | **128.44** / 128.44 / 14 | 245.51 / 24.56 / 104 | 181.93 / 181.93 / 14 | **128.44** / 128.44 / 14 |

The adaptive scheduler achieves the best (or within 4% of the best) waiting time in every
class: it equals SJF exactly on the seven SJF classes, equals Round Robin exactly on
`interactive_sparse`, and lands between SJF and Round Robin on `interactive` (35.89 vs
34.68, i.e. 3.5% worse than the best) because 2 of its 10 workloads are no-long-job draws
where SJF is optimal.

### 17.4 State coverage

* State space: **2187** states (7 variables × 3 bins).
* **Visited during training: 80 states**; **evaluated: 29 distinct states, all 29 visited
  during training, 0 unvisited, 0 decisions in unvisited states.**  (An unvisited state
  would fall back to the greedy tie-break, i.e. FCFS; the protocol was sized so that this
  does not occur.)

### 17.5 Reproducibility

Two consecutive runs produce byte-identical `metrics.csv`, `decisions.csv`,
`q_table.json` and `class_verification.json` (asserted by the test suite).  Every random
draw is seeded through a SHA-256-based `derive_seed`, and the exact configuration of the
reported run is written to `results/config.json`.

### 17.6 Honest findings

* **SJF is optimal for most of the declared workload set, and the agent says so.**  Under
  the declared reward, SJF is the reward-argmax on 7 of 9 classes (100% of their probe
  workloads).  The workload-aware behaviour is that the agent selects **Round Robin exactly
  on the interactive classes** — where preemption changes the ranking — and SJF everywhere
  else.
* **FCFS and Priority are never selected** (0 of 90 decisions).  Measured cause: the
  mean-metric reward structurally cannot prefer them — SJF weakly dominates FCFS on mean
  times, and Priority can at best tie SJF (when importance tracks job size,
  `priority_aligned`: Priority's mean reward 0.175 vs SJF's 0.185).  The pre-training
  verification confirms it: Priority is the reward-argmax on 0 of 225 probe workloads, and
  FCFS on 4 (a 16% minority of `interactive_sparse`, on draws where Round Robin
  degenerates to FCFS and SJF happens to be worse on that particular draw).
* **Round Robin still wins overall response time** (19.08 vs the adaptive's 73.43): the
  declared weights price waiting + turnaround above response, so the agent trades response
  for waiting on the batch-like classes.  The weights were not tuned.
* **Utilisation and throughput are identical for all policies on every workload**
  (work-conserving schedulers, costless switching).
* **93.3% agreement with the reward-argmax, not 100%** — the 6 misses are near-ties
  inside the interactive classes (§17.3).
* The state space is sparse (80 of 2187 states reachable by the nine classes); coverage of
  the reachable region is complete.

## 18. Limitations

1. **One decision per workload** (the project's simple architecture): the agent cannot
   change policy mid-run.
2. **Batch view**: the state sees the whole job list of the workload before it runs
   (arrival times, bursts, priorities).  That is the declared simulation model; a real
   system would estimate these quantities online.
3. **Synthetic workloads only**, single CPU, no I/O, integer time, no failures, no
   multi-core.
4. **Tabular Q-learning** with a hand-designed discretisation; no function approximation
   and no other learning algorithm (per the project's scope).
5. SJF dominates most of the declared workload set under the declared reward (§17.6);
   FCFS and Priority are never selected; the agent trades response time for waiting time
   because of the declared weights.
6. The residual 4% of probe workloads whose reward-argmax differs from their state's modal
   argmax are near-ties that a 3-bin discretisation of these variables cannot resolve.

## 19. Audit (reviewer's questions)

1. **Does the RL agent actually perform workload-aware selection?**  Yes.  It selects
   Round Robin on the interactive classes (8/10 and 10/10 held-out workloads) and SJF on
   the seven batch-like classes (10/10 each), matching the pre-training verification's modal
   winner for 9/9 classes, with 93.3% agreement with the per-workload reward-argmax.  The
   selection is a function of the encoded workload state, not of the class label: inside
   `interactive`, held-out draws without long jobs — whose state is shared with
   `short_stream` — are scheduled with SJF.
2. **Which workload classes trigger different policies?**  `interactive` and
   `interactive_sparse` trigger **Round Robin**; `short_batch`, `short_stream`,
   `long_batch`, `priority_aligned`, `priority_skewed`, `quantum_sensitive` and `mixed`
   trigger **SJF**.  FCFS and Priority are never triggered (structural, §17.6).
3. **Is the behaviour learned rather than hard-coded?**  Yes — tabular Q-learning with
   ε-greedy exploration; the selection comes from the Q-table (forced-value tests change
   the decision; evaluation is greedy and provably non-learning); nothing maps a class or
   a state to a policy in code.
4. **Are the results reproducible?**  Yes — fixed seeds (42 / 2024 / 777, recorded in
   `results/config.json`), SHA-256-derived per-draw seeds, byte-identical artefacts on
   re-runs (tested).
5. **Are all four baselines evaluated on identical workloads?**  Yes — one workload object
   per (class, repetition), scheduled by FCFS, SJF, Round Robin, Priority and the adaptive
   scheduler; fingerprints are carried in every row and the run fails if any scheduler saw
   a different workload (tested).  Evaluation workloads are disjoint from training
   workloads (tested).
6. **Any remaining deviations from the project requirements?**  None known.  The
   quantum-controller extension was removed (the RR quantum parameter stays, as part of
   Round Robin); the core flow, the six metrics, the four schedulers and the reward are
   unchanged from their declared form; no new ML algorithm or unrelated system feature was
   added.  The remaining limitations are the honest findings of §17.6 and §18.

---

### Repository notes

* Results and figures are committed so that the reported numbers can be inspected without
  re-running; re-running reproduces them byte for byte.
* `docs/DESIGN_AND_CHOICES.md` records every design decision, the diagnosis of the first
  experiment's outcome (SJF selected 120/120), and the full change history of the redesign.
* `docs/PHASE1_REQUIREMENTS_AUDIT.md` is the requirements audit performed before any code
  was written (historical).
