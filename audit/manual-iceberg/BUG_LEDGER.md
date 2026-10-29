# Bug Ledger — OS-Project

**Branch:** `arena/1a4dbf1c-os-project`  
**Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`  
**Audit directory:** `audit/manual-iceberg/`

**Severity scale:**
- **S0** — crash, data loss, corruption, security bypass, wrong algorithm on the hot path (blocks release / demo)
- **S1** — wrong metrics, wrong business rule, misleading headline claim visible to user (blocks demo until fixed or disclosed)
- **S2** — degraded behavior, poor UX, partial coverage, unverifiable claim (fix before release or document as limitation)
- **S3** — missing test coverage, docs gaps, minor polish, disclosure hygiene
- **S4** — informational / notes

**Evidence type key:**
- **MAPI** — manual API request against live service, response inspected
- **SRC** — source-code inspection
- **INF** — inference from source code (not directly confirmed by live observation)
- **ART** — committed artifacts inspected (healing_summary.json, healing_comparisons.csv)
- **ENV** — environment condition, not a product defect

---

## S0 — None confirmed

No crashes, no data corruption, no incorrect scheduler output, no metrics mismatch exposed to the user was found in live API testing or source review.

---

## S1 — None confirmed

No wrong metric, wrong business rule, or visible misleading headline claim was confirmed in the live application.

---

## S2 — 2 entries

### OS-002 — Live dashboard demonstrates rule-based recovery, NOT RL/Q-learning

**Severity:** S2  
**Status:** Confirmed — disclosed limitation  
**Evidence:** MAPI + SRC  

**Reproduction:** `POST /api/simulations` with scenario `recovery-accepted` or `recovery-rollback` → healing triggered via `RuleWaitingTimePolicy`. `POST /api/comparisons` → same rule-based policy. No UI control, request field, or scenario selects Q-learning or `experiments.healing_experiment`. Dashboard `research.py` reads `results/healing/healing_summary.json` + `healing_comparisons.csv` read-only. RL modules exist (`healing/policies.py` QLearningHealingPolicy, `rl/q_learning.py`, `rl/runtime_controller.RuntimeQController`) but are only exercised offline via `main.py healing-study` / `experiments/demo.py`.

**Impact:** A teacher demo of "RL healing" in the browser would show rule-based waiting-time recovery, not learned adaptive behavior. The project's README and About page disclose that the live demo uses a rule-based policy and that Q-learning is the offline study — but the app title and README headline "Adaptive CPU Scheduling Using RL..." invite the expectation that RL is live.

**Source trace:** `dashboard/service.py` builds `HealingController(engine, policy=RuleWaitingTimePolicy(), ...)`. `dashboard/research.py` is read-only over committed artifacts. `healing/controller.py` accepts any `BaseHealingPolicy` subclass, so Q-learning *could* be plugged in — it simply isn't in the dashboard.

**Correct nuance:** Q-learning is genuine code and IS exercised offline. Committed `results/healing/healing_summary.json` (210 workloads, seed 23819) shows `q_recovery` arm: interventions=0.65/trace, incidents=0.67/trace, mean_wait=122.56 — Q-learning DID intervene but performed WORSE than `fixed_sjf` (mean_wait=94.66). `q_recovery_from_sjf` arm is byte-identical to `fixed_sjf` because Q-learning started from SJF and learned to hold (interventions=0.0).

**Disposition:** Document as disclosed limitation. Do not claim RL is live. If RL must be demoed live, wire `QLearningHealingPolicy` or `RuntimeQController` into `dashboard/service.py` and expose a UI toggle — then re-audit.

---

### OS-003 — Drift injection and burst estimation exist but are off by default, no UI control

**Severity:** S2  
**Status:** Confirmed — disclosed limitation  
**Evidence:** SRC  

**Reproduction:** `workload/drift.py` implements linear_drift, sinusoidal, step_change, random_jump drift kinds. `healing/burst_estimator.py` implements linear and ema estimators. Both are off by default (`apply_drift=False`, `drift_config=None`, `burst_estimator_config=None`). No dashboard UI control, scenario field, or API parameter enables them.

**Impact:** "Dynamic workloads" headline capability is not exercisable through the live UI. Committed healing study uses drift on held-out workloads only (offline).

**Disposition:** Document as disclosed limitation. Enable via UI/API if dynamic workloads must be demoed live.

---

## S3 — 3 entries

### OS-004 — Detector can stay silent on a head-of-line-blocking workload (work gate suppresses incident; explanation is source-inferred)

**Severity:** S3  
**Status:** Confirmed — observed behavior + source analysis both confirm work gate suppression  
**Evidence:** MAPI + SRC  

**Reproduction:** `POST /api/simulations` custom workload: P1 burst=50 arrival=0; P2..P8 burst=2 arrival=1..7; policy=FCFS; healing enabled. Live response: `healing triggered: false`; summary: "the monitored waiting time never crossed its threshold." Waiting times are 0, 49, 50, 51, 52, 53, 54, 55 (avg 45.5) — large, yet no incident.

**Confirmation chain:** The detector requires `min_window_completions=4` completions before evaluating. By the time the 4th process completes (t≈56), P1–P4 are done and only P5–P8 remain (visible_remaining_work = 8 units). The work gate compares `visible_remaining_work < min_remaining_work_factor × max(1.0, observed_mean_burst)` = `8 < 1.5 × 8.0` = `8 < 12.0`, so the gate returns `insufficient_remaining_work` before any threshold comparison. Confirmed by both the live response (no incident) and source analysis of `healing/detector.py` `evaluate()`.

**Impact:** A realistic degrading workload can be submitted to the live dashboard with healing enabled and produce zero incidents — a teacher demo could show "healing does nothing" on a workload where intervention seems obviously warranted. Demo workload choice matters; recovery-accepted (mixed seed 13) and recovery-rollback (mixed seed 3) do fire incidents.

**Disposition:** Document as expected behavior, not a defect. Demo checklist uses recovery-accepted/recovery-rollback scenarios which demonstrably fire incidents.

---

### OS-006 — Q-learning performed worse than fixed SJF on mean waiting time in the committed offline study

**Severity:** S3 (performance finding, not a defect)  
**Status:** Confirmed from committed artifacts  
**Evidence:** ART  

**Reproduction:** Inspect `results/healing/healing_summary.json`: `q_recovery` arm mean_wait=122.56 vs `fixed_sjf` mean_wait=94.66. Q-learning interventions=0.65/trace, incidents=0.67/trace — Q-learning changed policies 65% of traces yet raised mean waiting time by ~29.6% relative to fixed SJF. `q_recovery_from_sjf` arm: interventions=0.0, wait identical to fixed_sjf (94.66) — Q-learning started from SJF and learned to hold.

**Impact:** If the project claims RL "improves" scheduling, the committed study does not support that claim on mean waiting time (rule_wait vs fixed_fcfs is significant; q_recovery vs rule_wait CI includes zero). The research page already caveats this honestly ("q_recovery vs rule_wait NOT significant, CI includes zero").

**Disposition:** Document. Do not claim RL improves waiting time without re-running the study with tuned hyperparameters or a different reward.

---

## S4 — informational notes

- **Concurrency guard:** source-confirmed — `_RunGuard` uses `threading.BoundedSemaphore(4)` with `acquire(blocking=False)` (verified in source and in isolated semaphore test: 4 acquired, 21 rejected). Runtime 429 responses could not be reproduced in this environment against the live uvicorn server (all 25 concurrent requests returned 201), likely due to how uvicorn/Starlette schedules `def` endpoints. Classified as source-confirmed, runtime-dependent on server concurrency model. See `evidence/api/concurrency_limit.json`.
- **Reproducibility confirmed:** identical scenario inputs produce byte-identical results and SVG charts across repeated runs; re-open and export match stored payloads exactly (modulo `executed_at`/`duration_ms` timestamps).
- **Input validation confirmed:** 18 invalid-input cases all return HTTP 422 with the offending field named and a specific message.
- **No fake data found:** no hardcoded scheduler results, no precomputed charts, no mock metrics anywhere in the dashboard service or teaching module. Every number is computed by executing repository code.
- **Browser tooling unavailable:** no browser binary, no X server, Playwright Chromium download blocked by sandbox host allowlist. UI-layer checks (pages, controls, charts in DOM, keyboard navigation, responsive layout) could not be manually performed in a real browser. Environment condition, not a product defect. See OS-001.

---

## Ledger summary

| Severity | Count | Status |
|----------|-------|--------|
| S0 | 0 | — |
| S1 | 0 | — |
| S2 | 2 | OS-002 (disclosed limitation), OS-003 (disclosed limitation) |
| S3 | 2 | OS-004 (confirmed), OS-006 (performance finding) |
| S4 | several | informational |

No fixes have been applied to the repository (audit rule 10: preserve existing project). Remediation plan is in `REMEDIATION_LOG.md`.
