# Teacher live demo — actual CPU scheduling, self-healing, and the dashboard

## A. Introduction (30 seconds)

CPU scheduling determines which ready process receives a CPU. A workload change can make
a previously reasonable policy perform poorly. This project asks whether a simulated
scheduler can monitor realized waiting, recognize degradation, choose an intervention and
verify it safely rather than adapting blindly.

This is a **single-CPU discrete-event simulation** with synthetic processes, not a Linux
kernel scheduler, real-time guarantee, hardware benchmark or production service. The
primary self-healing loop can use rules or tabular Q-learning; the short live
demonstration uses a **rule**, not a trained model. The non-healing runtime Q-learning
experiment is a separate workflow (§F).

Everything shown in class is produced by the project's own code: the scheduler in
`scheduler/`, the control loop in `healing/`, and the dashboard builder in `dashboard/`,
which only *reads* the saved output. The dashboard does not simulate anything in the
browser and contains no hand-typed numbers or screenshots.

## A2. Interactive console — the recommended classroom path

The static page below (§B) replays a saved file. The live console runs the engine from the
browser: pick a scenario, press **Run simulation**, and the Python backend executes the
scheduler, the workload generator and the healing loop *at that moment*. Every number, chart
and verdict on screen comes from that execution; nothing is pre-recorded.

Start it once (full setup, API reference and troubleshooting:
[LIVE_DASHBOARD.md](LIVE_DASHBOARD.md)):

```powershell
# Windows PowerShell, from the repository root
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe tools\serve_dashboard.py
# then open http://127.0.0.1:8000/   (stop with Ctrl+C in the same window)
```

Presentation sequence (8 minutes):

1. **Launch** — the header pill reads *Engine ready · Python 3.11.x*. Say plainly: a local
   FastAPI service is calling the project's own scheduler; the browser only draws results.
2. **Simulation Studio → Small teaching workload → Run simulation.** Three processes, four
   policies. Point at mean waiting 4.33 u (FCFS) and the table on the **Workload Explorer**
   (P1 0, P2 5, P3 8). Switch the **Policy view** tabs to SJF: 1.67 u, order P3 → P2 → P1.
   (The Studio's own **Policy view** selector shows the same trace once the run exists; the
   tab, the selector and the Overview all read one field.)
   The numbers are the hand calculation in §D because the page fails loudly if they ever
   stop matching.
3. **Successful recovery (seed 13) → Run simulation.** Read the provenance line out loud
   ("Executed now … run …", engine milliseconds), then the lifecycle: detection at t=243
   (105.17 > 75.70), diagnosis *fairness starvation*, action `set_sjf`, observation to
   t=319, residual +0.4375 ≥ +0.30 → **accepted**, SJF kept.
4. **Inconclusive verification and rollback (seed 3) → Run simulation.** Same rule, different
   fate: detection at t=141, SJF selected, residual +0.2059 inside the null range → the
   verdict is **inconclusive**, not rejected, so the controller rolls back to FCFS. Use this
   for the safety argument: the system does not claim a win it cannot prove.
5. **Policy Comparison → Run comparison.** Four separate executions on the identical
   workload, with the best by mean waiting marked. State that this is one workload, not
   general evidence.
6. **Research Results.** The committed final-test aggregate (FCFS 130.84, SJF 94.66,
   waiting-time rule 122.39, Q-learning 122.56) is labelled *historical, read-only*; the
   Q-minus-rule interval includes zero, so the study does **not** establish that Q-learning
   beats the rule.
7. **Limits.** Open **About & Method**: simulation not kernel, charts replay completed
   traces, research never mixed with the current run. Offer the JSON export for inspection.

If the network or the server is unavailable, the console says so and refuses to draw
anything; the static page from §B remains the offline fallback.

## B. Short executable sequence (about 5–8 minutes)

The classroom flow is:

1. run the real simulation (about 5 seconds),
2. build and open the dashboard,
3. explain one Gantt chart with the small worked example,
4. show the actual detection → intervention → rollback lifecycle,
5. switch to the accepted intervention,
6. explain the research comparison and its limitations,
7. finish with what is and is not implemented.

### Before class: dependencies

The dashboard adds **no new dependency**. It needs exactly the packages the simulator
already uses (`numpy`, `pandas`, `matplotlib`, `pytest` in `requirements.lock`): charts are
inline SVG generated with the Python standard library.

Windows PowerShell, from the repository root (this is the environment you already ran):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip check
```

Linux/WSL mirror used for the verification recorded below:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip check
```

PDF verification/building additionally requires `requirements-report.txt` and system
DejaVu fonts. No network is needed for the demo or the dashboard once dependencies are
installed.

### 1. Start the real simulation (pause: scope)

```powershell
.\.venv\Scripts\python.exe main.py demo --results-dir "$env:USERPROFILE\os-teacher-demo"
```

```bash
# Linux/WSL equivalent (verified)
.venv/bin/python main.py demo --results-dir /tmp/os-teacher-demo
```

Observed opening:

```text
Self-healing scheduler: deterministic quick demonstration
Real engine + default thresholds + waiting-time recovery rule; no Q training.
Times below are simulated units, not seconds. Fixtures are not study results.
```

Read one decision line, not only the summary. Seed 3:

```text
  d06 t=141 ready=[6,11,9,13,12,2,3,15,8] SJF -> P9 [142,144) switch=1 wait+=25
  d11 t=193 ready=[6,13,12,8] FCFS -> P6 [194,235) switch=1 wait+=127
```

`wait+` is the queue waiting accrued during that interval, not its duration. The file
`demo.json` (Windows: `%USERPROFILE%\os-teacher-demo\demo.json`) holds the real input,
every dispatch, both incidents and the final per-process outcomes.

Do not omit the command word: **`main.py` with no command defaults to the full healing
study**, which trains models instead of running the five-second demo.

### 2. Generate and open the dashboard (pause: “this is the run you just made”)

```powershell
.\.venv\Scripts\python.exe tools\build_teacher_dashboard.py --input "$env:USERPROFILE\os-teacher-demo\demo.json" --output "$env:USERPROFILE\os-teacher-demo\index.html"
Start-Process "$env:USERPROFILE\os-teacher-demo\index.html"
```

```bash
# Linux/WSL equivalent (verified)
.venv/bin/python tools/build_teacher_dashboard.py \
  --input /tmp/os-teacher-demo/demo.json --output /tmp/os-teacher-demo/index.html
```

Observed output:

```text
Teacher dashboard written
-------------------------
  demo input        : demo.json
  scenarios         : seeds 3, 13
  teaching example  : 4 policies, hand calculation verified
  research panel    : healing_summary.json (10 arms)
  output            : ...\os-teacher-demo\index.html
  size              : 213.0 KB (self-contained HTML, inline SVG)
  sha256            : 230a34caf560495e
Open the file in a browser; no server or network connection is used.
```

The page is a single file: styles, script and every chart are inside it. It opens from the
file system with no server, no CDN and no internet connection, so a projector laptop
cannot lose it mid-lecture. `--output` is optional (it defaults to `index.html` next to
`demo.json`), and an existing dashboard is only replaced with `--force`; the builder never
writes anywhere near `results/healing/`, `results/runtime/`, `results/study/` or
`figures/`.

### 3. Explain a Gantt chart with the worked example (about 2 minutes)

On the dashboard: **“Scheduling timeline”** first (the seed-3 Gantt chart), then
**“Worked example”**.

The top lane of the Gantt chart is the CPU itself — the exact sequence of dispatches the
engine recorded — and each lower lane is one process with arrival (hollow circle) and
completion (tick). Colour is the policy in force, so the intervention appears as a colour
change, and the hatched slivers are the configured 1-unit context switches. Hover or click
any block to read that interval's pid, policy, start, end, switch cost and accrued waiting.

Then use the four-policy worked example, which the builder **recomputes** with the real
`scheduler/` classes and checks against the hand calculation (the page prints
“match … the documented calculation”; the build fails instead of drawing a stale table):

All arrivals 0; switch cost **0 for hand arithmetic**, RR quantum 2. In the live healing
demo switch cost is **1** and quantum **4**. Lower priority number is higher.

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

For RR, P2 waits 8 − 0 − 3 = 5 even though its first response is at t=2: waiting counts
time between slices. Mean turnaround is 7.333333, 4.666667, 6.666667, 7.333333. The
selector buttons show each policy; with JavaScript disabled all four charts are on the
page. Say plainly that SJF wins *this* three-job pattern and needs burst times, so it is a
different trade-off, not a free improvement. Full arithmetic and edge cases:
`PYTHONPATH=. .venv/bin/python docs/manual_evidence/scenarios.py` →
`HAND CALCULATIONS MATCH all four algorithms`
([scenarios.txt](manual_evidence/scenarios.txt)).

### 4. Show the actual detection, intervention and rollback (about 2 minutes)

On the dashboard: **“Self-healing lifecycle”**, seed 3 (the default).

The eight cards are built from the incident record: trigger time and threshold, diagnosis
and context state, the selected action, the observation window, the verifier's verdict and
the final decision. Seed 3's recorded evidence:

- trigger at **t=141** (decision 6): completed-window mean **51.000** exceeds the threshold
  **28.203** (reference 23.75 + 1.25 bands of 3.5625), after 2 suspect evaluations;
- diagnosis `fairness_starvation`; context bins `[1,2,2,0,1]` encode state **205 of 648**;
- the rule selects `set_sjf`; the change first runs at **t=142**, after the 1-unit switch;
- verification closes at **t=193** with residual **+0.2059**, inside the declared null range
  (−0.30 … +0.30), so the verdict is **inconclusive**;
- FCFS is restored at t=193; `confirmed_failure` stays **false**.

Make the distinction explicit, because it is the point of the design: **inconclusive is not
rejected**. A rejected window (residual ≤ −0.30) is a *confirmed failure* that advances the
failure counter and, twice in a row, pins the safe fallback configuration. An inconclusive
window rolled back here because the declared posture never keeps a configuration it could
not verify. Note also that the intervention *did* lower the final mean waiting time
(130.467 vs 150.600 for FCFS) — a short window cannot certify a whole-trace outcome, and
the verifier is a safety filter, not a benefit estimator.

### 5. Switch to the accepted intervention (about 1 minute)

Click **“Seed 13 · accepted”** in the lifecycle section (the selector stays in step with the
timeline and performance sections). Recorded evidence: trigger at **t=243** (window mean
105.167 vs threshold 75.703), the same `set_sjf` rule choice at t=244, residual **+0.4375**
→ **accepted** at t=319, and every later dispatch still runs under SJF. Final mean waiting
220.533 vs 255.800 for FCFS.

Then say what acceptance means: the window closed more of the trigger gap than the
phase-aware no-action reference predicted, so the change became the known-good
configuration. It is bounded online evidence, not proof of global optimality. The dashboards
for the two seeds are: [seed 3](manual_evidence/dashboard/gantt_seed3.png),
[seed 13](manual_evidence/dashboard/gantt_seed13.png) (Gantt),
[seed 3](manual_evidence/dashboard/lifecycle_seed3.png),
[seed 13](manual_evidence/dashboard/lifecycle_seed13.png) (lifecycle),
[seed 3](manual_evidence/dashboard/verification_seed3.png),
[seed 13](manual_evidence/dashboard/verification_seed13.png) (verification bands).

### 6. Explain the research comparison and its limitations (about 1 minute)

Click the **“Research results”** tab (top of the page), or open
`results/healing/healing_report.md` for the generated tables. The research panel reads the
committed held-out study — a different artifact from the two demo fixtures — and is drawn
from `results/healing/healing_summary.json` plus `healing_comparisons.csv`:

| Arm | Mean waiting (simulated units) |
|---|---:|
| FCFS | 130.838 |
| SJF | 94.658 |
| Waiting-time rule | 122.388 |
| Q-learning recovery | 122.564 |

Lower is better. SJF is best on mean waiting in this design; both recovery approaches beat
FCFS; the Q-minus-rule paired difference is +0.18 with a 95% interval [−0.04, +0.48] that
includes zero, so **the evidence does not establish Q-learning as better than the rule**.
The panel prints exactly these sentences from the values, so a re-run cannot leave a stale
claim behind. Keep three subsystems apart: the self-healing controller, the rule-based
recovery policy inside it, and the separate runtime Q-learning workflow in
`results/runtime/` (four dispatch policies, no verifier, no rollback).

Mention the limitations on the same slide: synthetic workloads, one CPU, exact bursts for
the fixed baselines (SJF is an oracle, not an equal-information comparison), a verifier
that leaves most helpful interventions inconclusive, and finite-sample intervals. The two
demo seeds are illustrative fixtures, not evidence of general superiority.

### 7. Finish: what is and is not implemented (about 30 seconds)

The dashboard's last section, “What is and is not implemented”, lists: single-CPU
simulation with synthetic processes; no I/O, caches, multicore or kernel module; simulated
time units; detection is an operational condition, not fault proof; inconclusive rollbacks;
and the research chart is the frozen study, not a new experiment. Nothing in this
repository modifies a real kernel scheduler.

## C. Dashboard details, recovery and requirements

### Interaction model

| Interaction | Where | Works without JavaScript |
|---|---|---|
| Switch demo scenario (seed 3 / 13) | Timeline, lifecycle, performance | All scenarios are rendered; the selector is a view filter |
| Select teaching policy | Worked example | All four policies are rendered |
| Inspect a dispatch interval | Hover any block (native tooltip), click/tab to read details | Hover tooltip and the full interval table inside “Every recorded dispatch interval” |
| Reveal incident / verification details | `<details>` fold-outs | Yes |
| Move between pages | Left rail, browser Back/Forward | Every page has its own URL (`/#/studio`, `/#/workload`, …); refresh keeps the page |
| Navigate sections | Section links | Yes |

If the script fails, it removes its own `js` class, so the page falls back to showing every
section rather than hiding content.

### Recovery instructions

- **Python launcher not found / wrong interpreter**: run `.\.venv\Scripts\python.exe --version` (Windows) or `.venv/bin/python --version`; reinstall `requirements.lock` and
  rerun `pip check`. The dashboard needs no package beyond those.
- **`error: demo input not found: …`** — the path is wrong or `main.py demo` was never run.
  Use the exact `--results-dir` you passed to the demo.
- **`error: demo.json: invalid JSON at line L column C`** — the file is truncated or was
  edited. Rerun `main.py demo` into the same folder; it is a disposable output.
- **`error: … fixtures[0] (seed 3).metrics.avg_waiting_time …`** — the trace contradicts its
  recorded metrics or outcomes (hand-edited file). Rerun the demo instead of patching JSON.
- **`error: output already exists: …`** — you already built a dashboard. Pass `--force` to
  replace that HTML file, or write to a new name.
- **`error: refusing to write into the research evidence directory …`** — choose an output
  under the demo folder (`--output <demo-dir>/index.html`). Research artifacts are never
  write targets.
- **Wrong working directory / import error** — run the builder from the repository root, or
  with the same interpreter that ran the demo; `tools/build_teacher_dashboard.py` adds the
  repository root to `sys.path` itself.
- **Page opens but looks unstyled** — the file was edited or truncated; rebuild with
  `--force`. Never hand-edit the generated HTML.
- **A surprising verdict** — read the incident record on the page or in `demo.json`. Never
  relabel inconclusive as accepted.
- **Study interrupted** (optional section) — keep original evidence, rerun into a new output
  directory, never tune thresholds on final-test seed 23819.

### Requirements and verification status

- Python 3.11.2 and `requirements.lock` (`numpy`, `pandas`, `matplotlib`, `pytest`) — the
  same environment that runs the simulator. Importing the project's scheduler package pulls
  in `numpy` through `workload/`, so the dashboard is not runnable in a bare interpreter,
  but it adds nothing to install. Its charts use only the Python standard library (no
  Plotly/D3/Chart.js), and Pillow is needed only by the evidence script described below.
- **Verified here**: Debian 12, CPython 3.11.2 — demo, dashboard build, the full test suite
  (582 tests), and inspection of the generated HTML/SVG. The page was opened in a headless
  environment only through its SVG and markup; this sandbox has **no browser**, so the
  screenshots below were rasterised from the SVG embedded in the generated `index.html`
  (`docs/manual_evidence/dashboard/render_chart_png.py`), not taken with a screenshot tool.
- **Live console (optional)**: `pip install -r requirements-dashboard.txt` adds FastAPI and
  Uvicorn on top of the same environment. It was exercised in a headless Chromium session
  here (see [manual_evidence/web/VERIFICATION.md](manual_evidence/web/VERIFICATION.md)),
  driven against the running server on Linux; see [LIVE_DASHBOARD.md](LIVE_DASHBOARD.md) for
  the API, the startup command and its own recovery list. The static page needs no server.
- **Not verified here**: the Windows PowerShell commands (for both the builder and the live
  console). They use the same interpreter and paths you already ran the demo with, and both
  entry points avoid anything platform-specific (paths are handled by `pathlib`), but execute
  them once before class.
- Machine-readable provenance is on the page itself: the demo file name and the first 16
  hex characters of its SHA-256, plus the research file names.

### Captured evidence (from the actual generated dashboard)

| Section | Files |
|---|---|
| Scheduling timeline | [seed 3](manual_evidence/dashboard/gantt_seed3.png), [seed 13](manual_evidence/dashboard/gantt_seed13.png) |
| Self-healing lifecycle | [seed 3](manual_evidence/dashboard/lifecycle_seed3.png), [seed 13](manual_evidence/dashboard/lifecycle_seed13.png) |
| Verification bands | [seed 3](manual_evidence/dashboard/verification_seed3.png), [seed 13](manual_evidence/dashboard/verification_seed13.png) |
| Worked example | [mean waiting](manual_evidence/dashboard/teaching_waiting_time.png), [SJF](manual_evidence/dashboard/teaching_sjf.png), [Round Robin](manual_evidence/dashboard/teaching_round_robin.png) |
| Performance | [seed 3](manual_evidence/dashboard/performance_seed3_waiting.png), [seed 13](manual_evidence/dashboard/performance_seed13_waiting.png) |
| Research | [mean waiting by arm](manual_evidence/dashboard/research_waiting_time.png) |

## D. Worked scheduling example (same numbers as §B3)

The hand calculation is reproduced above; the dashboard recomputes it at build time. All
values matched direct execution: [scenarios.txt](manual_evidence/scenarios.txt). The same
command shows that P2 arriving exactly at a Round-Robin slice endpoint is admitted before
P1 is requeued, producing P1/P2/P1, not P1/P1/P2.

## E. What the self-healing demonstration proves

The monitor uses recently completed waiting times and an online stable reference; detection
applies threshold + hysteresis/cooldown and load/work gates. Diagnosis uses arrived-ready
evidence, not future jobs.

- **Seed 3:** residual 0.20592237689676574 lies between −0.30 and +0.30. Verdict is
  inconclusive; FCFS is restored at t193. It is **not** a confirmed failure. Measured mean
  wait is 130.467 versus independently executed FCFS 150.600.
- **Seed 13:** trigger t243, residual 0.437454899124583; accepted at t319. SJF remains in
  force. Mean wait 220.533 versus FCFS 255.800.

Acceptance is a bounded online evidence decision, not proof of globally optimal scheduling.
A useful intervention can still be inconclusive. Neither fixture demonstrates Q learning or
an unbiased performance estimate. Seeds were chosen for contrasting paths. Full genuine
transcript: [demo.txt](manual_evidence/demo.txt).

An additional actual default-threshold replay produces **two rejected RR interventions and
safe fallback** (§F). Do not confuse it with the accepted waiting-rule demo.

## F. Optional extended checks (before/after class, not the main presentation)

### Separate runtime RL training and inference (small)

```bash
PYTHONPATH=. .venv/bin/python docs/manual_evidence/runtime_learning.py
sed -n '1,65p' /tmp/os-teacher-runtime/runtime_report.md
```

Observed: 30 episodes, one seed, two families; 162 states × four actions, Q CSV 648 rows,
106 visited state/actions in the pipeline; six final-test workloads × six methods = 36 rows.
The validation-selected learned demonstration has six switches. The direct controller check
additionally trained 68 nonzero Q entries and made 19 inference decisions without changing Q
or visit counts; reward −97 equals −sum(waiting)/4. The pipeline reuses trained models in
memory; CSV is an audit export, not a model loader. This tiny run's Q mean waiting 100.467 is
worse than FCFS 87.389 and SJF 55.433. Timing fields can vary; no performance superiority
claim is justified.

### Changing-workload evaluation (small, intentionally insufficient for inference)

```bash
PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py \
  --reps 1 --train-episodes 2 --drift phase_transition --out /tmp/os-teacher-drift
sed -n '1,100p' /tmp/os-teacher-drift/drift_phase_transition_estTrue_cv0.3_report.md
```

Observed: seven base workloads, 35 Q records, 70 total arm records; prefix preserved, suffix
burst/priority distribution changes toward short_jobs with original arrivals/IDs. Noise is
modelled, not calibrated. Fixed SJF still sees exact bursts (oracle). With one workload per
family, deterministic-arm family-stratified bootstrap intervals collapse; **do not present
these smoke intervals as general statistical evidence**.

### Primary full study, actual rejection/fallback and integrity replay

```bash
.venv/bin/python main.py healing-study --results-dir /tmp/os-teacher-healing \
  --figures-dir /tmp/os-teacher-healing-figures
PYTHONPATH=. .venv/bin/python docs/manual_evidence/output_integrity.py
PYTHONPATH=. .venv/bin/python docs/manual_evidence/render_plots.py
```

Run the demo and drift commands above first: the integrity script reads their `/tmp` files.
The study trains five models × 1,200 traces, evaluates held-out arms and audits; allow
roughly a minute or more. It was run safely here, not silently substituted with fixtures.
Integrity replay uses poisson_arrivals/final_test repetition 21, fingerprint
`c555eb898c254d71`, **fairness rule**. It produces two `set_round_robin` interventions,
residuals −0.316225 and −0.383115, rollback to FCFS, then pins FCFS after t189. No thresholds
are weakened. It is an explanatory replay of already-evaluated test data, not fresh
independent evidence or a reason to tune the design.

New plots go to `/tmp/os-teacher-healing-figures`. Use these regenerated figures: the
historical action-effect PNG contains a plotting defect documented in the manual report.
Several no-op reward-point labels still overlap; use CSV rows for their exact values.

The dashboard's research panel reads the committed `results/healing/` artifacts, so it does
not need this rerun. To point it at a fresh copy instead, pass `--research
/tmp/os-teacher-healing/healing_summary.json`.

### Optional final PDF build and automated verification

```bash
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python tools/build_plain_report.py --source docs/OS_Project_Final_Report.md \
  --output /tmp/os-teacher-final-report.pdf
.venv/bin/python tools/verify_final_delivery.py
```

The verifier checks PDF properties and exact demo/evidence links, not all visual layout.
Fresh sample page renders: [page 1](manual_evidence/report_page_01.png),
[page 14](manual_evidence/report_page_14.png). See [TEACHER_QA.md](TEACHER_QA.md) for
questions grounded in code and evidence.
