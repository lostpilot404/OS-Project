# Fake-Functionality Audit — OS-Project

**Branch:** `arena/1a4dbf1c-os-project`  
**Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`  
**Audit directory:** `audit/manual-iceberg/`

**Method:** every user-facing feature was exercised against the live API (MAPI) and/or traced through source code (SRC). A feature is classified **fake** only if evidence shows it is disconnected, hardcoded, or decorative. A feature is **genuine but not live** if real code exists and is exercised somewhere (offline or CLI) but not through the dashboard UI.

**Classification scale:**
- GENUINE — real implementation, exercised live, output is computed not canned
- GENUINE (OFFLINE) — real implementation, exercised offline/CLI, not wired into dashboard UI
- DISCLOSED LIMITATION — real implementation, intentionally not live, disclosed in README/About
- FAKE — hardcoded, disconnected, or decorative (none found)
- UNVERIFIABLE — cannot be tested in this environment (no browser)

---

## Feature-by-feature audit

| # | Feature | Classification | Evidence | Basis |
|---|---------|---------------|----------|-------|
| F1 | Scheduler engine (FCFS/SJF/Priority/RR) | GENUINE | MAPI + SRC | `POST /api/simulations` executes `scheduler/fcfs.py`, `sjf.py`, `priority.py`, `rr.py`; outputs match independent hand calculations exactly (11 cases, Category D) |
| F2 | Workload generator (fixed + mixed seeds) | GENUINE | MAPI + SRC | teaching scenario uses fixed workload; recovery scenarios use mixed seeds 13/3; outcomes deterministic and reproducible |
| F3 | Metrics computation (avg_waiting, turnaround, etc.) | GENUINE | MAPI + SRC | `dashboard/model.py` cross-checks metrics vs per-process outcomes; `DashboardError` raised on mismatch; hand-verified values match |
| F4 | Healing controller (monitor→detect→diagnose→act→verify) | GENUINE | MAPI | accepted recovery (seed 13) and inconclusive rollback (seed 3) both execute end to end live |
| F5 | Healing accepted path | GENUINE | MAPI | recovery-accepted: detected t=243, fairness_starvation, set_sjf, ACCEPTED, SJF kept |
| F6 | Healing rollback path | GENUINE | MAPI | recovery-rollback: detected t=141, set_sjf, INCONCLUSIVE, rolled back to FCFS |
| F7 | Healing safe-fallback (max_consecutive_failures) | GENUINE (SRC) | SRC | present in `healing/controller.py`; not triggered by any manual scenario tested |
| F8 | RuleWaitingTimePolicy (live healing policy) | GENUINE | MAPI + SRC | the only healing policy wired into `dashboard/service.py`; fires on recovery-accepted/recovery-rollback |
| F9 | QLearningHealingPolicy | GENUINE (OFFLINE) | SRC + ART | real class in `healing/policies.py`, used by `experiments/healing_experiment.py`; NOT wired into dashboard UI; committed study shows it intervened (0.65/trace) but performed worse than fixed SJF |
| F10 | RuntimeQController | GENUINE (OFFLINE) | SRC | `rl/runtime_controller.py` real, used by `main.py healing-study`; not in dashboard |
| F11 | Tabular Q-learning (`rl/q_learning.py`) | GENUINE (OFFLINE) | SRC + ART | real implementation; committed `results/healing/healing_summary.json` produced by it; q_recovery_from_sjf arm holds (0 interventions) |
| F12 | Drift injection (`workload/drift.py`) | GENUINE (OFFLINE) | SRC | 4 drift kinds implemented; off by default; no UI/API control; used in offline held-out evaluation |
| F13 | Burst estimation (`healing/burst_estimator.py`) | GENUINE (OFFLINE) | SRC | linear + EMA estimators implemented; off by default; no UI/API control |
| F14 | Dashboard frontend (React-ish vanilla JS + SVG charts) | GENUINE (UI unverifiable) | SRC | `dashboard/web/app.js` calls the same API endpoints audited live; chart SVG generated server-side; DOM behavior cannot be tested (no browser) |
| F15 | Research page data | GENUINE (committed artifacts) | MAPI + ART | `GET /api/research` reads `healing_summary.json` + `healing_comparisons.csv` read-only; page honestly caves q_recovery vs rule_wait "CI includes zero" |
| F16 | Export / re-open | GENUINE | MAPI | `GET /api/simulations/{run_id}/export` returns byte-identical payload; re-open matches stored results |
| F17 | Input validation | GENUINE | MAPI | 18 invalid cases, all HTTP 422 with named field |
| F18 | Concurrency guard (MAX_CONCURRENT_RUNS=4) | GENUINE | MAPI | 25 rapid requests → 4 × 201 + 21 × 429; semaphore enforced live |

---

## Features NOT fake (disconnected/hardcoded checks)

- **No hardcoded scheduler results:** every `POST /api/simulations` response is computed by executing scheduler classes (`dashboard/teaching.py` raises `DashboardError` on hand-calculation mismatch).
- **No precomputed charts:** chart SVG is generated per run from computed metrics; identical inputs produce byte-identical charts, different inputs produce different charts.
- **No mock metrics in research page:** research aggregates come from committed `healing_summary.json` / `healing_comparisons.csv`, produced by the offline study (read-only in dashboard).
- **No canned validation messages:** each 422 names the specific offending field with a context-specific message.
- **No fake healing:** `healing.triggered: False` is genuinely returned when no incident fires (HOLB workload); `True` with accepted/rollback outcomes on the two recovery scenarios.

---

## Features that ARE limited (not fake, but not live)

| # | Feature | Limitation | Disclosed? |
|---|---------|------------|------------|
| F9–F11 | Q-learning / RL | Not wired into dashboard UI; offline study only | Yes — README and About page state demo uses rule-based policy; Q-learning is the offline study |
| F12 | Drift injection | Off by default; no UI control | Partially — code exists, no UI surface |
| F13 | Burst estimation | Off by default; no UI control | Partially — code exists, no UI surface |
| F14 | Frontend DOM behavior | Cannot be manually tested (no browser) | Environment blocker (OS-001), not a product limitation |

---

## Verdict on fake functionality

**No feature was found to be fake, disconnected, hardcoded, or decorative.** Everything the dashboard presents is computed by the repository's own code, executed live against real scheduler classes. The only gaps are:

1. **RL is offline-only** (F9–F11): genuine Q-learning code, genuinely exercised in the committed offline study, genuinely not live in the dashboard. Disclosed.
2. **Drift/burst estimation offline-only** (F12–F13): genuine implementations, off by default, no UI control. Disclosed in code defaults; not surfaced in UI.
3. **Frontend unverifiable** (F14): genuine implementation, but no browser tooling to manually click/type/inspect. Environment blocker.

A hostile examiner would conclude the application is **not a Potemkin dashboard** — the backend is real and the numbers are real — but it **does not demonstrate RL live**, and its UI layer could not be manually validated in this environment.
