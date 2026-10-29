# Final Demo Checklist — OS-Project

**Branch:** `arena/1a4dbf1c-os-project` | **Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`
**Based only on features that actually exist and were verified live.** Unsupported claims (live RL demonstration, dynamic/drift workload through the browser) are EXCLUDED — see `FAKE_FUNCTIONALITY_AUDIT.md` (OS-002, OS-003) and `BUG_LEDGER.md` (OS-001 browser blocker).

**Prerequisites (verified startup procedure):**
1. `python -m venv .venv`
2. `.venv/bin/python -m pip install --upgrade pip`
3. `.venv/bin/python -m pip install -r requirements.txt -r requirements-dashboard.txt`
4. `.venv/bin/python tools/serve_dashboard.py --host 127.0.0.1 --port 8000` (default; opens the browser automatically on platforms that have one)
5. Open `http://127.0.0.1:8000/`

**Demo sequence (each step verified live at the API level; UI-click steps require a real browser — see OS-001):**

| # | Step | What to do | What to confirm | Evidence |
|---|------|-----------|-----------------|----------|
| 1 | Health | Backend reachable | `/api/health`: `status: ok`, engine `available: true`, probe ok | `health.json` |
| 2 | Catalogue | Load scenarios | 4 scenarios: teaching, recovery-accepted, recovery-rollback, custom | `examples.json` |
| 3 | Teaching workload | Studio → "Small teaching workload" → Run | FCFS 4.33, SJF 1.67, Priority 3.67, RR 4.33 mean waiting (hand-calculable) | `sim_teaching.json` |
| 4 | Teaching policy view | Switch the policy-view selector between the four policies | Each policy's executed trace shown first; same 3 processes each time | `sim_teaching.json` (teaching.policies) |
| 5 | Classical comparison | Studio → custom or any run → "Run comparison" | 4 policies executed separately on the identical workload (same fingerprint); like-for-like table + bars | `compare_explicit.json` |
| 6 | Accepted recovery | Studio → "Successful recovery (seed 13)" → Run | Degradation detected, FCFS→SJF, verification closes ACCEPTED, SJF stays to end; baseline FCFS 255.8 vs healing 220.53 waiting | `sim_recovery_accepted.json` |
| 7 | Rollback recovery | Studio → "Inconclusive verification and rollback (seed 3)" → Run | Degradation detected, FCFS→SJF, window INCONCLUSIVE, rolled back to FCFS, no confirmed failure; final policy FCFS | `sim_recovery_rollback.json` |
| 8 | Custom workload | Studio → "Custom workload" → edit process table → Run | Your exact process table scheduled; metrics recomputed from the trace; invalid inputs rejected with field-specific 422 errors | `sim_custom.json`; C1–C18 |
| 9 | Export | From an open run → "Export JSON" | Download starts; file opens; content matches the run; filename `os-dashboard-<scenario>-<runid>.json` | `sim_teaching_export.json` |
| 10 | Re-open run | Overview → earlier run → "Open" | Restored result shown, labelled "Restored from this session"; not a new execution | `sim_teaching_reopen.json` |
| 11 | Research results | Research Results page | Committed held-out study arms (FCFS, SJF, RR, Priority, detector-only, fairness rule, waiting-time rule, Q-learning recovery) read from `healing_summary.json`; caveats visible (Q-learning not established as better than the rule) | `research.json` |
| 12 | Invalid input | Custom editor → burst = 0 or negative, duplicate pid, quantum out of range | HTTP 422 with the offending field named; no traceback; previous run retained | C1–C18 |

**What NOT to demonstrate (not supported live):**
- Q-learning / RL training or inference through the browser (the live self-healing loop uses the waiting-time rule; Q-learning exists only in the offline study).
- Drift / burst-injection dynamic workloads through the browser (implemented offline, off by default, no UI control).
- Any workflow not in the table above.

**Finish state:** leave the server running or stop it with Ctrl+C; no data is written into the repository (runs live in per-run temp directories, cleared on shutdown).
