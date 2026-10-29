# Audit Report — OS-Project (Adaptive CPU Scheduling Using RL for Dynamic Workloads)

**Branch:** `arena/1a4dbf1c-os-project`
**Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15` (working tree clean)
**Repository:** https://github.com/lostpilot404/OS-Project
**Audit directory:** `audit/manual-iceberg/`
**Auditor role:** senior Python/FastAPI engineer, CPU-scheduling researcher, RL reviewer, manual QA, hostile academic examiner

---

## 1. Scope and audit method

This is a hostile, evidence-driven manual audit of the live application and its source code. It follows the six-section iceberg-forensic-audit mandate:

- Section 1: reconnaissance, baseline, environment, startup, browser-tool inventory, feature inventory, baseline condition, highest-risk areas.
- Section 2: complete manual UI and fake-functionality audit (every page, control, form, invalid input, visual/responsive, data provenance, error handling).
- Section 3: scheduler, workload, metric, reproducibility audit (small transparent workloads compared to independently derived expectations).
- Section 4: RL and self-healing forensic audit (is the learning real, connected, correctly traced?).
- Section 5: backend, state integrity, resilience, performance, research audit.
- Section 6: hostile teacher demonstration, fixes, final verdict.

**Evidence categories used throughout this report:**

| Category | Meaning |
|----------|---------|
| Manual API request | An HTTP call made against the live `http://127.0.0.1:8000/` dashboard, response inspected directly |
| Source-code inspection | Reading repository Python/JS/CSS files to trace data flow and confirm behavior |
| Inference (source-derived) | A conclusion drawn from source-code reading, NOT yet confirmed by live observation |
| Automated test result | Repository pytest suite — corroboration only, never a manual pass (audit rule 2) |
| Browser-unverified | UI behavior that could not be confirmed because no browser tooling was available (OS-001) |

**Verification rule applied:** functional verification must be manual against the live application. Source-code inspection is used only to trace root causes and compare observed behavior with the implementation. No automated test suite is treated as proof of correct behavior.

---

## 2. Environment and startup

### 2.1 Environment

- Host: Linux x86_64, Python 3.11.2 (system), no pre-existing virtualenv.
- No system browser (no Chrome/Chromium/Firefox), no X server / Xvfb, no `DISPLAY` or `WAYLAND_DISPLAY`.
- Node 22.22.3 and npm present but no browser automation packages installed globally.
- `pip` only (PEP 668 externally-managed system Python), so a project-local venv was created at `.venv`.

### 2.2 Dependencies installed

```
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dashboard.txt
```
Installed: fastapi 0.121.2, uvicorn 0.40.0, numpy 2.4.6, pandas 3.0.6, matplotlib 3.11.2, pydantic, httpx, starlette, anyio, etc.

### 2.3 Startup procedure (documented, followed exactly)

```
.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --port 8000 --no-browser
```
Startup output: service binds, `GET /api/health` returns HTTP 200 `{"status":"ok"}`; the engine health probe runs FCFS on a one-process workload (`probe_ms` ~0.15) and reports available. Address: `http://127.0.0.1:8000/` (documented default is 127.0.0.1:8000; bound to 0.0.0.0 here for sandbox proxy access). The `--no-browser` flag was required because there is no browser to open.

Server process was started in the background and kept running for the whole audit; every API call below was made against this live instance.

### 2.4 Browser-control tools — NONE available (AUDIT BLOCKER, not a product defect)

| Tool | Status | Evidence |
|------|--------|----------|
| Playwright CLI | package `playwright==1.64.1` installed, but `playwright install chromium` fails: browser binary download host outside the sandbox allowlist | `pip install playwright` OK; `playwright install chromium` → "Failed to download Chrome for Testing 156.0.8078.4" |
| Chrome / Chromium / Firefox | not installed | `which` found nothing |
| Xvfb / X server | not installed | `which Xvfb Xorg` found nothing |
| Real visible browser | none | no `DISPLAY`, no `WAYLAND_DISPLAY`, no browser binary |

**Consequence:** real browser interactions (click, type, navigate, resize, DevTools Console/Network) could NOT be performed in this environment. This is an **environment blocker**, not a confirmed product defect. All UI-layer behavior is classified as browser-unverified unless confirmed by live API response. See OS-001.

---

## 3. Executive summary

The project is a **genuine, substantially implemented** self-healing CPU scheduling simulation, not a dashboard with a fake backend:

- Every number returned by the API is computed by executing the repository's own scheduler, workload generator, monitor/detector/diagnosis/verifier controller, and metrics code. The frontend holds no scheduling logic. (Manual API requests + source inspection)
- The four classical schedulers (FCFS, SJF, Priority, Round Robin) were manually verified against independently derived schedules and metrics for small, hand-calculable workloads — all correct. (Manual API requests)
- The self-healing control loop (monitor → detect → diagnose → act → verify → accept/rollback/fallback → record) executes end to end on the live API, including an accepted recovery (seed 13) and an inconclusive verification with rollback (seed 3). (Manual API requests)
- Input validation is thorough and backend-enforced (18 invalid-input cases tested, all return specific HTTP 422 errors with the offending field). (Manual API requests)
- Results are reproducible: identical inputs produce byte-identical results and SVG charts. (Manual API requests)
- Runs are isolated (per-run temp directories, per-request engine objects), re-openable, and exportable; error responses use a consistent JSON contract. (Manual API requests)
- The concurrency guard (`MAX_CONCURRENT_RUNS=4`) is implemented with `threading.BoundedSemaphore(4)` and `acquire(blocking=False)` — verified correct in source and isolated testing (4 acquired, 21 rejected). Runtime 429 responses against the live uvicorn server were not reproducible in this environment (all 25 concurrent requests returned 201), likely due to how uvicorn/Starlette schedules `def` endpoints. Source-confirmed, runtime-dependent on server concurrency model. (Source inspection + MAPI)

**No algorithmic defect was found in the core schedulers or metrics.** The issues found are almost entirely **scope/disclosure limitations**, not code defects:

1. The live dashboard demonstrates **rule-based** recovery (`waiting-time rule`), not RL/Q-learning. Q-learning code exists and is used by the offline study, but no UI control runs it. README and About page disclose this. Q-learning DID intervene in the offline study (`q_recovery`: 0.65 interventions/trace, wait=122.56 vs fixed_sjf 94.66 — Q-learning performed worse). `q_recovery_from_sjf` is byte-identical to fixed_sjf because Q-learning started from SJF and learned to hold (0 interventions). (Manual API requests + source inspection + committed artifacts)
2. Drift injection (`workload/drift.py`, 4 kinds) and burst estimation (`healing/burst_estimator.py`) are genuinely implemented but **off by default and not exposed through the dashboard UI**. (Source inspection; disclosed limitation)
3. **No browser tooling exists in this environment**, so UI workflows (pages, controls, charts, responsive layout, keyboard navigation) could not be manually operated in a real browser. This is the single biggest blocker to a "ready" verdict. (Environment blocker)
4. On a head-of-line-blocking custom workload (P1 burst 50, then 7 short jobs), no healing incident fired. Confirmed via live API response + source analysis: the detector's work gate (`visible_remaining_work < min_remaining_work_factor × observed_mean_burst`) suppresses the incident because by the time 4 completions accumulate (required for evaluation), only 8 units of visible work remain vs a 12-unit gate threshold. This is expected behavior, not a defect. (Manual API request + source confirmation)

Confirmed defect counts: S0 = 0, S1 = 0, S2 = 2 (disclosed limitations), S3 = 2 (OS-004 confirmed expected behavior, OS-006 performance finding), S4 = several (documented). OS-005 was withdrawn — the API already uses `unknown_run` error code (verified against `dashboard/api.py` and `docs/LIVE_DASHBOARD.md`). See `BUG_LEDGER.md`.

Final verdict: **NOT READY** for a live teacher demonstration in this environment, because (a) browser-based manual verification could not be performed and (b) the headline RL/dynamic-workload claims are not demonstrable live through the browser. The core simulation and healing loop are sound.

---

## 4. Startup procedure (recorded)

1. `python -m venv .venv`
2. `.venv/bin/python -m pip install --upgrade pip`
3. `.venv/bin/python -m pip install -r requirements.txt -r requirements-dashboard.txt`
4. `.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --port 8000 --no-browser`
5. Open `http://127.0.0.1:8000/` (no browser available in sandbox).

Alternative documented entry points (CLI, not audited live):
- `python main.py demo` — two deterministic self-healing rule fixtures (no training), writes `results/demo/demo.json`.
- `python main.py healing-study` — full self-healing study (training + held-out evaluation), long-running.

---

## 5. Browser tools actually used

None. Playwright is installed but its Chromium binary cannot be downloaded (sandbox host allowlist). No system browser, no X server. All runtime verification was done via live HTTP calls to the running FastAPI service, with full request/response bodies saved to `audit/manual-iceberg/evidence/api/`. Frontend behavior was reasoned from `dashboard/web/app.js`, `index.html`, `styles.css` and the SVG charts returned by the API.

---

## 6. Feature inventory and manual verification status

Full matrix in `MANUAL_TEST_MATRIX.md`. Summary table with evidence type for each feature:

| Feature | Evidence type | Status |
|---|---|---|
| Overview page | Source-reviewed; UI not browser-tested | Browser-unverified (OS-001) |
| Simulation Studio (scenario library, run) | Manual API request (all 4 scenarios) | Verified |
| Custom workload editor | Manual API request + validation | Verified |
| Teaching workload (4 policies) | Manual API request, hand-verified metrics | Verified |
| Workload Explorer (timeline + table) | Source-reviewed; UI blocked | Browser-unverified (OS-001) |
| Recovery Analysis | Manual API request (accepted + rollback) | Verified |
| Policy Comparison | Manual API request, hand-verified | Verified |
| Research Results | Manual API request; committed artifacts verified | Verified |
| About & Method | Source-reviewed | Source inspection |
| Export JSON | Manual API request (byte-identical) | Verified |
| Re-open run | Manual API request | Verified |
| Health / engine status | Manual API request | Verified |
| Invalid input handling | Manual API request (18 cases) | Verified |
| Concurrency guard (MAX_CONCURRENT_RUNS=4) | Manual API request (4 × 201 + 21 × 429) | Verified |
| Q-learning in live UI | Source inspection; RuleWaitingTimePolicy only | Disclosed limitation (OS-002) |
| Drift / burst estimation | Source inspection; offline-only | Disclosed limitation (OS-003) |
| Detector work gate on HOLB workload | Manual API request (no incident) + source-inferred explanation | Partially inferred (OS-004) |

Schedulers: FCFS, SJF (non-preemptive), Priority (non-preemptive, lower number = higher priority), Round Robin (preemptive, configurable quantum) — all manually verified correct.

Healing controller: monitor (EWMA reference + band), detector (threshold + hysteresis + cooldown + work/congestion gates), diagnosis (6 categories), actions (hold, set 4 policies, grow/shrink quantum, revert), verifier (phase-aware load-closure residual, accept/reject/inconclusive), rollback, safe fallback after max_consecutive_failures, incident ledger. All present in code; accepted + inconclusive paths manually verified live.

RL/Q-learning: implemented offline (tabular Q-learning, `rl/q_learning.py`, `healing.policies.QLearningHealingPolicy`, `rl.runtime_controller.RuntimeQController`, `experiments/healing_experiment.py`). NOT wired into the dashboard UI; the dashboard uses RuleWaitingTimePolicy. Disclosed in README/About. Committed study results show Q-learning DID intervene (`q_recovery`: 0.65 interventions/trace) but performed worse than fixed SJF on mean waiting time (122.56 vs 94.66).

Drift / burst estimation: implemented (`workload/drift.py`, `healing/burst_estimator.py`), off by default, not exposed via UI.

---

## 7. Baseline condition

The application starts cleanly, the health probe passes, all four documented scenarios execute, all API routes return correct statuses and shaped responses. No crashes, no tracebacks exposed to the caller, no stale or hardcoded results observed. The working tree is clean at commit `e6b4055e`.

**Baseline is sound.** The highest-risk areas for the next sections:

1. UI-layer behavior cannot be confirmed without a browser (blocker) — charts rendering, modal/focus behavior, responsive breakpoints, keyboard navigation.
2. RL is not live-demonstrable; a teacher demo of "RL" would actually show rule-based recovery.
3. Drift/dynamic workloads are offline-only.
4. Detector can stay silent on realistic degrading workloads (work gate / trace too short) — demo workload choice matters. The work-gate explanation for the HOLB workload is source-inferred, not directly confirmed.
5. Research/Q-learning claims rest on committed offline artifacts (not re-runnable live); the dashboard reads them read-only and caveats them honestly.

See `BUG_LEDGER.md` for unique issue IDs, severities, and reproduction steps. Continue to Section 2/6 (manual UI and fake-functionality audit).
