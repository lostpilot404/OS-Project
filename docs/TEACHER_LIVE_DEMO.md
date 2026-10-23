# Teacher live demo — actual CPU scheduling and self-healing

## A. Introduction (30 seconds)

CPU scheduling determines which ready process receives a CPU. A workload change can
make a previously reasonable policy perform poorly. This project asks whether a
simulated scheduler can monitor realized waiting, recognize degradation, choose an
intervention and verify it safely rather than adapting blindly.

This is a **single-CPU discrete-event simulation** with synthetic processes, not a
Linux kernel scheduler, real-time guarantee, hardware benchmark or production service.
There is no GUI or API to start. The primary self-healing loop can use rules or tabular
Q-learning; the short live demonstration below uses a **rule**, not a trained model.
The non-healing runtime Q-learning experiment is separate.

Reviewed base: `d7f2efe3950f35aa577c24bc7031df103b096eb4`, branch
`arena/c3d9429b-os-project`; this manual-review patch adds two focused report/plot fixes.
No branch switching is needed. Commands below run from the repository root in Bash.

## B. Short executable sequence (about 5–8 minutes)

### Before class: dependencies

Observed environment: Debian 12 x86-64, CPython 3.11.2. WSL instructions are in
[WSL_SETUP.md](WSL_SETUP.md), but actual Windows/WSL execution is not verified.
If `.venv` already works, do not recreate it. For a new environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip check
```

Core versions stay pinned. PDF verification/building additionally requires
`requirements-report.txt` and system DejaVu fonts. No network is needed for the demo
once dependencies are installed. Do not install into Debian's system Python.

### 1. Establish the executable (pause: scope)

```bash
git branch --show-current
git rev-parse HEAD
.venv/bin/python --version
.venv/bin/python main.py --help
```

Observed help lists `demo`, `healing-study`, `runtime-experiment`, `study`, `experiment`,
and `train`. **Never run `main.py` without a command in class**: it defaults to the full
healing study. The default demo is small; studies can overwrite tracked output paths.

### 2. Show a schedule you can calculate (pause: policy trade-offs)

```bash
PYTHONPATH=. .venv/bin/python docs/manual_evidence/scenarios.py
```

This small inspectable presenter script calls the real scheduler APIs, prints every
service interval and per-process outcome, compares hand-calculated values, then shows
edge cases, a no-incident healing run, a phase-change input and expected invalid-input
errors. It does not use pytest or train a model. See the worked example below.
Observed checkpoint: `HAND CALCULATIONS MATCH all four algorithms`.

### 3. Execute the real primary self-healing demo (pause: evidence)

```bash
.venv/bin/python main.py demo --results-dir /tmp/os-teacher-demo
```

Observed opening:

```text
Self-healing scheduler: deterministic quick demonstration
Real engine + default thresholds + waiting-time recovery rule; no Q training.
Times below are simulated units, not seconds. Fixtures are not study results.
```

Read the actual decision lines rather than only the summary. For example, seed 3:

```text
  d06 t=141 ready=[6,11,9,13,12,2,3,15,8] SJF -> P9 [142,144) switch=1 wait+=25
  d11 t=193 ready=[6,13,12,8] FCFS -> P6 [194,235) switch=1 wait+=127
```

These show intervention and later rollback in the running simulator. `wait+` is the
sum of queue waiting accrued during the interval, not just wall-clock duration.
The saved file is `/tmp/os-teacher-demo/demo.json`; it contains real input, decisions,
incidents and final per-process outcomes. Inspect without optional tools:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for f in json.loads(Path('/tmp/os-teacher-demo/demo.json').read_text())['fixtures']:
    i = f['incidents'][0]
    print(f['seed'], i['diagnosis'], i['action_name'], i['verdict'],
          i['rollback_target_policy'], f['metrics']['avg_waiting_time'])
PY
```

Observed fields: seed 3, fairness_starvation, set_sjf, inconclusive, FCFS,
130.46666666666667; seed 13, fairness_starvation, set_sjf, accepted, no rollback target,
220.53333333333333. Full genuine transcript: [demo.txt](manual_evidence/demo.txt).

### 4. Show research evidence, not just a chosen fixture (pause: limitations)

```bash
sed -n '1,55p' results/healing/healing_report.md
```

This is the retained generated research report, not demo output. Its metrics were
reproduced during this review in a separate directory. Explain mean wait: FCFS 130.838,
SJF 94.658, rule_wait 122.388, q_recovery 122.564. Lower is better for this metric.
The rule and Q beat FCFS on the frozen design but neither matches SJF; the Q-minus-rule
95% interval includes zero. RR improves response but has higher mean waiting and switch
cost. Do not say “RL is best”.

Open `docs/OS_Project_Final_Report.pdf` in your normal PDF viewer, particularly pages
4 (verification/reward), 10–11 (metrics/results) and 14 (demo). The PDF has **18 pages**;
its Mermaid blocks are deliberately printed source, not rendered diagrams. Historical
branch names and 500-test counts refer to its original submission, not this review.
For current verification use [MANUAL_VERIFICATION_REPORT.md](MANUAL_VERIFICATION_REPORT.md).
A representative newly rendered chart is
[healing_waiting_time.png](manual_evidence/healing_waiting_time.png).

### Recovery instructions

- Missing numpy/pandas: use `.venv/bin/python`, install `requirements.lock`, rerun pip check.
- Wrong directory/import error: `cd` to the repository root; API scripts need `PYTHONPATH=.`.
- Output path is a file or is unwritable: choose another `/tmp/...` directory. The demo
  prints “Use --results-dir with a writable directory” and exits 1. Do not use sudo.
- Existing `/tmp/os-teacher-demo/demo.json`: rerunning replaces only this disposable
  output; repeated JSON was byte-identical. Do not delete research directories.
- No PDF packages/fonts: skip PDF regeneration during class; use the existing PDF.
- A surprising verdict: read the actual ledger. Never relabel inconclusive as accepted.
- Study interrupted: keep original evidence, use a new output directory on rerun. Do not
  tune thresholds on final-test seed 23819 to obtain a nicer presentation.

## C. Worked scheduling example

All arrivals are 0; switch cost **0 for hand arithmetic**, RR quantum 2. In the real
healing demo switch cost is **1**, quantum **4**. Lower priority number is higher.

| Process | Arrival | Burst | Priority |
|---|---:|---:|---:|
| P1 | 0 | 5 | 2 |
| P2 | 0 | 3 | 1 |
| P3 | 0 | 1 | 3 |

Turnaround = completion − arrival. Waiting = turnaround − burst (including time between
RR slices). Response = first service − arrival. In P1/P2/P3 order:

| Policy | Service order | Completion / turnaround | Waiting | Mean waiting |
|---|---|---|---|---:|
| FCFS | P1 0–5, P2 5–8, P3 8–9 | 5, 8, 9 | 0, 5, 8 | 13/3 = 4.333333 |
| SJF | P3 0–1, P2 1–4, P1 4–9 | 9, 4, 1 | 4, 1, 0 | 5/3 = 1.666667 |
| Priority | P2 0–3, P1 3–8, P3 8–9 | 8, 3, 9 | 3, 0, 8 | 11/3 = 3.666667 |
| RR | P1 0–2, P2 2–4, P3 4–5, P1 5–7, P2 7–8, P1 8–9 | 9, 8, 5 | 4, 5, 4 | 13/3 = 4.333333 |

For RR P2, waiting is 8 − 0 − 3 = 5, not first-start 2. Its response is 2.
Mean turnaround is respectively 7.333333, 4.666667, 6.666667, 7.333333.
All values matched direct execution: [scenarios.txt](manual_evidence/scenarios.txt).
The same command shows P2 arriving exactly at a RR slice endpoint is admitted before
P1 is requeued, producing P1/P2/P1, not P1/P1/P2.

## D. What the self-healing demonstration proves

The monitor uses recently completed waiting times and a learned-online stable reference;
detection applies threshold + hysteresis/cooldown and load/work gates. Diagnosis uses
arrived-ready evidence, not future jobs. Seed 3 triggers at d6/t141: completed-window
mean 51.000 exceeds threshold 28.203. Diagnosis is fairness_starvation; the rule chooses
SJF, context bins `[1,2,2,0,1]` encode state 205 of 648.

- **Seed 3:** residual 0.20592237689676574 lies between −0.30 and +0.30. Verdict is
  inconclusive; FCFS is restored at t193. It is **not** a confirmed failure. Measured
  mean wait is 130.467 versus independently executed FCFS 150.600.
- **Seed 13:** trigger t243, residual 0.437454899124583; accepted at t319. SJF remains
  in force. Mean wait 220.533 versus FCFS 255.800.

Acceptance is a bounded online evidence decision, not proof of globally optimal
scheduling. A useful intervention can still be inconclusive. Neither fixture demonstrates
Q learning or an unbiased performance estimate. Seeds were chosen for contrasting paths.

An additional actual default-threshold replay, described below, produces **two rejected
RR interventions and safe fallback**. Do not confuse it with the accepted waiting-rule demo.

## E. Optional extended checks (before/after class, not the main presentation)

### Separate runtime RL training and inference (small)

```bash
PYTHONPATH=. .venv/bin/python docs/manual_evidence/runtime_learning.py
sed -n '1,65p' /tmp/os-teacher-runtime/runtime_report.md
```

Observed: 30 episodes, one seed, two families; 162 states × four actions, Q CSV 648 rows,
106 visited state/actions in the pipeline; six final-test workloads × six methods =
36 rows. The validation-selected learned demonstration has six switches. The direct
controller check additionally trained 68 nonzero Q entries and made 19 inference decisions
without changing Q or visit counts; reward −97 equals −sum(waiting)/4.
The pipeline reuses trained models in memory; CSV is an audit export, not a model loader.
This tiny run's Q mean waiting 100.467 is worse than FCFS 87.389 and SJF 55.433.
Timing fields can vary; no performance superiority claim is justified.

### Changing-workload evaluation (small, intentionally insufficient for inference)

```bash
PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py \
  --reps 1 --train-episodes 2 --drift phase_transition --out /tmp/os-teacher-drift
sed -n '1,100p' /tmp/os-teacher-drift/drift_phase_transition_estTrue_cv0.3_report.md
```

Observed: seven base workloads, 35 Q records, 70 total arm records; prefix preserved,
suffix burst/priority distribution changes toward short_jobs with original arrivals/IDs.
Noise is modelled, not calibrated. Fixed SJF still sees exact bursts (oracle). With one
workload per family, deterministic-arm family-stratified bootstrap intervals collapse;
**do not present these smoke intervals as general statistical evidence**.

### Primary full study, actual rejection/fallback and integrity replay

```bash
.venv/bin/python main.py healing-study --results-dir /tmp/os-teacher-healing \
  --figures-dir /tmp/os-teacher-healing-figures
PYTHONPATH=. .venv/bin/python docs/manual_evidence/output_integrity.py
PYTHONPATH=. .venv/bin/python docs/manual_evidence/render_plots.py
```

Run the demo and drift commands above first: integrity script reads their `/tmp` files.
The study trains five models × 1,200 traces, evaluates held-out arms and audits; allow
roughly a minute or more. It was run safely here, not silently substituted with fixtures.
Integrity replay uses poisson_arrivals/final_test repetition 21, fingerprint
`c555eb898c254d71`, **fairness rule**. It produces two `set_round_robin` interventions,
residuals −0.316225 and −0.383115, rollback to FCFS, then pins FCFS after t189.
No thresholds are weakened. It is an explanatory replay of already-evaluated test data,
not fresh independent evidence or a reason to tune the design.

New plots go to `/tmp/os-teacher-healing-figures`. Use these regenerated figures: the
historical action-effect PNG contains a plotting defect documented in the manual report.
Several no-op reward-point labels still overlap; use CSV rows for their exact values.

### Optional final PDF build and automated verification

```bash
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python tools/build_plain_report.py --source docs/OS_Project_Final_Report.md \
  --output /tmp/os-teacher-final-report.pdf
.venv/bin/python tools/verify_final_delivery.py
```

The verifier checks PDF properties and exact demo/evidence links, not all visual layout.
Fresh sample page renders: [page 1](manual_evidence/report_page_01.png),
[page 14](manual_evidence/report_page_14.png). See [TEACHER_QA.md](TEACHER_QA.md)
for questions grounded in code and evidence.
