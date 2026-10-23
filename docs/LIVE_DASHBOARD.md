# Live dashboard — run the scheduler from a browser

The live console is the interactive face of this project: a small FastAPI service that
calls the **same** scheduler, runtime simulator, healing controller and metric functions the
command line uses, plus a browser frontend that renders what the engine returns.

It is not a second simulator. `dashboard/service.py` validates a request, calls
`experiments.demo.run_fixture` / `run_healing_workload` / `scheduler.POLICY_CLASSES`, and
serialises the recorded trace. The browser contains no scheduling, detection, recovery or
metric logic; the only thing JavaScript does is send requests and draw returned values.

> **Scope reminder.** This is a discrete-event simulation of a single CPU with synthetic
> processes. Nothing here is a kernel scheduler, real-time guarantee or hardware benchmark.
> All times are *simulated time units*.

---

## 1. Requirements

| Item | Value |
|---|---|
| Python | 3.11 (the recorded environment is CPython 3.11.2) |
| Project packages | `requirements.txt` / `requirements.lock` — `numpy`, `pandas`, `matplotlib`, `pytest` |
| Dashboard packages | `requirements-dashboard.txt` — adds `fastapi 0.121.2`, `uvicorn 0.40.0`, `httpx 0.28.1` |
| Browser | any current Chrome, Edge or Firefox. No CDN, web font or network access is used |

`httpx` is only needed for the API tests (`fastapi.testclient`). The server itself needs
FastAPI and Uvicorn.

## 2. Install and start

### Windows PowerShell (primary)

```powershell
# from the repository root, with the project virtual environment already created
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt

.\.venv\Scripts\python.exe tools\serve_dashboard.py
```

The script prints the address and tries to open your default browser:

```
Adaptive CPU Scheduling Dashboard
  serving : http://127.0.0.1:8000/
  engine  : C:\path\to\OS-Project
  stop    : press Ctrl+C in this terminal
```

Open **http://127.0.0.1:8000/** if the window did not open by itself.

### Linux / WSL

```bash
.venv/bin/python -m pip install -r requirements-dashboard.txt
.venv/bin/python tools/serve_dashboard.py
```

### Options

```text
--host 127.0.0.1    interface to bind (default: loopback only)
--port 8000         port to listen on
--no-browser        do not open a browser window
--log-level info    uvicorn log level
```

Stop the server with **Ctrl+C** in the terminal that started it. Nothing is left running
and no repository file is modified.

## 3. The interface, page by page

The console is a single page with hash routing, so every page has a URL that can be
refreshed, bookmarked and reached with the browser's Back and Forward buttons. The left rail
lists destinations only — no example workloads, seeds or recovery examples live there.

| Page | URL | What it shows |
|---|---|---|
| Overview | `/#/overview` | engine health, the current simulation, and the runs kept in this server session |
| Simulation Studio | `/#/studio` | the scenario library, the configuration the engine will receive, and the execute action |
| Workload Explorer | `/#/workload` | metrics, the dispatch timeline with per-block inspection, and the process table |
| Recovery Analysis | `/#/recovery` | the event timeline, the verification band and the recorded incident detail |
| Policy Comparison | `/#/compare` | four policies executed on the current run's workload |
| Research Results | `/#/research` | the committed held-out aggregates, labelled historical |
| About & Method | `/#/about` | purpose, architecture, definitions, units, limitations and how to run it |

An unknown or empty fragment opens the Overview. Analysis pages read the **current run**;
with no run in the session they show an empty state that links to the Studio rather than
inventing values. Running a new simulation replaces the current run and clears the
comparison that belonged to the previous one.

## 4. The demonstration, in order

1. **Simulation Studio → Small teaching workload → Run simulation.** Three processes, four
   policies, all executed now; the numbers match the hand calculation in
   [TEACHER_LIVE_DEMO.md](TEACHER_LIVE_DEMO.md) §D (FCFS 4.33, SJF 1.67, Priority 3.67,
   Round Robin 4.33 mean waiting units).
2. Switch the **Policy view** tabs on the Workload Explorer to compare the same jobs under
   each policy, and click a block in the timeline (or focus it and press Enter) to read the
   recorded dispatch interval.
3. **Simulation Studio → Successful recovery (seed 13) → Run simulation.** FCFS degrades,
   SJF is selected, the observation window closes **accepted** (+0.4375 ≥ +0.30) and SJF
   stays in force.
4. **Inconclusive verification and rollback (seed 3) → Run simulation.** The same rule
   selects SJF, but the residual +0.2059 lies inside the null range, so the verdict is
   **inconclusive** and the controller rolls back to FCFS. Inconclusive ≠ rejected.
5. **Policy Comparison → Run comparison**: four separate executions on the identical
   process list.
6. **Research Results**: the committed final-test aggregates, labelled historical and never
   mixed with the live run.
7. **About & Method**: what the console does and does not claim; every analysis page also
   exports the raw result JSON.

A custom workload can be typed in on the Studio page (pid, arrival, burst, priority), run
under any policy, and optionally passed through the healing loop.

## 4. What each scenario runs

| Scenario | Engine path | Verified outcome |
|---|---|---|
| `teaching` | `dashboard.teaching.build_teaching_panel` → `scheduler.POLICY_CLASSES` (quantum 2, switch cost 0) | hand-verified: waiting FCFS `[0,5,8]`, SJF `[4,1,0]`, Priority `[3,0,8]`, RR `[4,5,4]` |
| `recovery-accepted` | `experiments.demo.run_fixture(13)` — controller + `RuntimeSimulator` + `compute_metrics` | trigger t=243, FCFS→SJF, closes t=319, residual +0.4375 → **accepted**, SJF kept; mean wait 220.53 vs FCFS 255.80 |
| `recovery-rollback` | `experiments.demo.run_fixture(3)` | trigger t=141, FCFS→SJF, closes t=193, residual +0.2059 → **inconclusive**, rollback to FCFS, no confirmed failure; mean wait 130.47 vs FCFS 150.60 |
| `custom` | `scheduler.POLICY_CLASSES` or `run_healing_workload` on the submitted processes | whatever the engine computes; no expectation is displayed |

Both fixtures are re-executed on every request; nothing is cached from a previous session.
`tests/test_dashboard_api.py` re-runs them and compares the API payload with
`experiments.demo.run_fixture` and with direct `scheduler`/`compute_metrics` calls.

## 5. HTTP API

Base URL `http://127.0.0.1:8000`. Interactive OpenAPI docs are at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | service status **and** an engine probe (a one-process FCFS schedule is executed) |
| `GET` | `/api/examples` | the four loadable scenarios, their options and expectations |
| `GET` | `/api/runs` | compact summaries of the runs currently in memory, newest first (used by the Overview page) |
| `POST` | `/api/simulations` | validate and execute a scenario; returns the run id and the full result |
| `GET` | `/api/simulations/{run_id}` | re-read a stored result |
| `GET` | `/api/simulations/{run_id}/export` | download exactly the stored result as JSON |
| `POST` | `/api/comparisons` | run all four policies on a stored run’s workload (or an explicit process list) |
| `GET` | `/api/research` | the committed study tables, read-only |

### Request bodies

```jsonc
// POST /api/simulations
{ "scenario": "recovery-accepted", "options": {} }

// custom workload, optional self-healing loop
{ "scenario": "custom",
  "options": { "processes": [ {"pid": 1, "arrival_time": 0, "burst_time": 8, "priority": 2} ],
               "policy": "SJF", "quantum": 4, "switching_cost": 1, "healing": false } }

// POST /api/comparisons
{ "run_id": "<id from a previous POST>" }          // or
{ "processes": [ ... ], "quantum": 4, "switching_cost": 0 }
```

`GET /api/runs` answers with `{"runs": [...], "count": n, "history_limit": 32}`. Each entry
carries only what a list needs — `run_id`, `scenario_id`, `scenario_name`, `kind`,
`executed_at`, `status`, `final_policy`, `mean_waiting_time`, `verdict` — so the Overview page
never downloads full traces to show the session's history. It is empty before the first run
and reports the registry's real limit, not a constant.

### Response shape (`POST /api/simulations`, HTTP 201)

```jsonc
{
  "run_id": "…",
  "status": "completed",
  "server_ms": 12.7,
  "result": {
    "kind": "healing" | "teaching" | "custom",
    "scenario": { "id", "name", "description", "expectation", … },
    "executed_at": "2026-10-23T11:02:03+00:00",
    "duration_ms": 4.8,                       // engine time only
    "settings": { "quantum", "switching_cost", "policy", "healing", … },
    "workload": { "fingerprint", "processes": [ … ] },
    "schedule": { "intervals": [ … ], "segments": [ … ], "final_policy", … },
    "metrics": { "avg_waiting_time", "avg_turnaround_time", "avg_response_time",
                 "cpu_utilization", "throughput", "context_switches", … },
    "baseline": { … FCFS on the same workload, or null for a plain custom run },
    "outcomes": [ { "pid", "arrival_time", "burst_time", "priority",
                    "start_time", "completion_time", "waiting_time",
                    "turnaround_time", "response_time" } ],
    "healing": { "triggered": true, "summary", "stages": [ … 8 stages … ],
                 "incident": { "verdict", "verifier_reason", "closure_residual",
                               "settings_before_policy", "settings_after_policy",
                               "rollback_target_policy", … } } | null,
    "charts": { "gantt": "<svg …>", "lifecycle": "<svg …>", … },
    "notes": [ … ]
  }
}
```

### Errors

Every failure uses one shape, and the message is written for a person:

```json
{ "error": { "type": "invalid_request", "message": "processes[0]: burst_time must be > 0, got 0",
             "field": "processes" } }
```

| Status | `type` | When |
|---|---|---|
| 422 | `invalid_request` | unknown scenario/option/policy, invalid workload, malformed JSON |
| 404 | `unknown_run` | the run fell out of the bounded history |
| 429 | `busy` | more than 4 simulations running at once |
| 500 | `engine_failure` / `engine_output_rejected` | the engine raised, or its output failed schema validation |
| 503 | `research_unavailable` | the committed study artifacts cannot be read |

## 6. Isolation and safety

- Requests never contain paths. The server chooses every path it uses; research artifacts
  are opened **read-only** and nothing is ever written inside the repository.
- Each run gets its own `tempfile.mkdtemp` directory holding `result.json`; the directory is
  deleted when the run leaves the 32-run history or when the server shuts down.
- At most four simulations execute concurrently; further requests get HTTP 429 instead of
  queueing behind a slow engine.
- No shell command is constructed from input; the engine is called as Python functions.
- Custom workloads are bounded: 1–40 processes, burst ≤ 10 000, arrival ≤ 100 000,
  quantum 1–64, switch cost 0–10.

## 7. Verification status

**Verified on Linux** (Debian 12, CPython 3.11.2) with a real browser engine —
HeadlessChrome 138.0.7204.0 driven by Puppeteer — against the running server: startup and
health, all four scenarios, recovery/rollback, custom workloads and validation, policy
comparisons, export, backend-loss and recovery, duplicate-submission and stale-response
handling, keyboard operation, and 1440 / 900 / 420 px layouts. Records and screenshots:
[docs/manual_evidence/web/](manual_evidence/web/) for the first interface, and
[docs/manual_evidence/web-v2/](manual_evidence/web-v2/) for the current one — every page at
1440 / 900 / 420 px, the empty states, and 50 scripted browser checks with their results in
`browser-checks.json`.

**Not verified here:** Windows PowerShell. The commands above are the Windows equivalents of
the Linux commands that were executed; run them once before class. No desktop browser
session was possible in this environment (the browser ran headless), so visual checks were
made from headless screenshots at fixed viewports.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: missing dependency 'fastapi'` on startup | run `pip install -r requirements-dashboard.txt` with the same interpreter |
| Page loads unstyled and stays on *Connecting…* | the browser used a cached copy of the page and never fetched `/static/app.js`. The server now sends the page with `no-store` and a content-stamped script URL; reload with Ctrl+Shift+R (Cmd+Shift+R) or open a new tab |
| Page shows **Backend unavailable** | the server stopped or the port is wrong; restart `tools/serve_dashboard.py` and press **Retry connection** |
| `[Errno 98] Address already in use` | another process owns port 8000; start with `--port 8001` and open that address |
| `GET /api/research` returns 503 | the committed `results/healing/healing_summary.json` is missing; run from the repository root |
| **that run is not in the server's history** | the run aged out of the 32-run history after a restart; press **Run simulation** again |
| Tables/charts look cramped | the charts scroll horizontally inside their panel on narrow screens; resize or scroll the chart area |
| Verdict differs from the guide | read the incident record on **Recovery Analysis**; never relabel inconclusive as accepted |
| A page says *No simulation in this session yet* | that page describes the current run; run a simulation in the Studio first |

The original, JavaScript-free static page is still available for printing or for a machine
without a server: `python tools/build_teacher_dashboard.py --input results/demo/demo.json
--output results/demo/index.html` (see [TEACHER_LIVE_DEMO.md](TEACHER_LIVE_DEMO.md)).
