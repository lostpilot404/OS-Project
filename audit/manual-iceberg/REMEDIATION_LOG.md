# Remediation Log — OS-Project Manual Audit

**Branch:** `arena/1a4dbf1c-os-project`  
**Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`  
**Audit directory:** `audit/manual-iceberg/`  
**Rule:** no fixes applied to the repository during this audit (preserve existing project). This log is a prioritized plan only.

**Priority scale:** P1 = blocks a live demo / must fix or disclose before demo; P2 = should fix before release; P3 = polish; P4 = informational / optional.

---

## Remediation items

### R1 — Wire Q-learning (or expose RL toggle) into the dashboard — P1 (disclosure alternative acceptable)

**Issue:** OS-002 — live dashboard uses `RuleWaitingTimePolicy`, not Q-learning. RL is genuine but offline-only.

**Options:**
- (a) Wire `QLearningHealingPolicy` / `RuntimeQController` into `dashboard/service.py` and add a UI toggle (`policy: rule | q-learning`), then re-audit the live healing path with Q-learning.
- (b) Keep rule-based live demo and strengthen disclosure: README headline, About page, and demo script must state plainly that the live demo is rule-based and that Q-learning results are from an offline committed study.

**Evidence:** `experiments/demo.run_healing_workload` constructs `SelfHealingController(..., RuleWaitingTimePolicy(config.healing), ...)`; `dashboard/service.py` calls `run_fixture` which uses `run_healing_workload`. `healing/controller.py` accepts any `BaseHealingPolicy` subclass; committed `results/healing/healing_summary.json` shows q_recovery intervened (0.65/trace) but performed worse than fixed_sjf (122.56 vs 94.66).

**Status:** Not applied. Disclosed as limitation OS-002.

---

### R2 — Enable or disclose drift injection and burst estimation — P2 (disclosure alternative acceptable)

**Issue:** OS-003 — `workload/drift.py` (4 drift kinds) and `healing/burst_estimator.py` (linear + EMA) are implemented but off by default, with no UI/API control.

**Options:**
- (a) Add scenario/API fields for `drift_config` and `burst_estimator_config`, default off, documented.
- (b) Keep off by default and disclose in README/About that dynamic workloads are offline-only.

**Evidence:** source inspection of `workload/drift.py` (`drift_enabled=False`) and `healing/burst_estimator.py` (`burst_estimator_enabled=False`) and `healing/config.py` (`HealingConfig.burst_estimator_enabled: bool = False`, `HealingConfig.drift_enabled: bool = False`); both off by default.

**Status:** Not applied. Disclosed as limitation OS-003.

---

### R3 — Choose demo workloads that demonstrably cross the detector threshold — P1 for demo

**Issue:** OS-004 — HOLB-style workload (P1 burst=50 + short jobs) produced no healing incident. Confirmed as expected behavior: the detector's work gate suppresses incidents when visible remaining work (8 units) < 1.5 × mean burst (12 units). Not a defect — no code change needed.

**Action:** Use demo workloads that demonstrably cross the detector threshold (recovery-accepted seed 13, recovery-rollback seed 3). Do not rely on the HOLB workload to demonstrate healing.

**Evidence:** `evidence/api/sim_custom_holb.json` — `healing triggered: false`; source analysis of `healing/detector.py` `evaluate()`.

**Status:** Not applicable — expected behavior, not a defect. Demo checklist uses recovery-accepted/recovery-rollback scenarios.

---

### R4 — Browser-based UI verification — P1 (environment blocker, outside repo scope)

**Issue:** OS-001 — no browser binary, no X server, Playwright Chromium download blocked. UI-layer behavior (pages, controls, SVG charts in DOM, keyboard navigation, responsive layout, DevTools) could not be manually verified in a real browser.

**Action:** Outside the repository's scope. Requires a sandbox with a browser and display server, or manual verification by a human operator with browser access. Until then, all UI checks are BLOCKED; API-level behavior is fully verified.

**Evidence:** environment inventory (Section 2 of AUDIT_REPORT.md); EVIDENCE_INDEX E06.

**Status:** Not applied (cannot be applied from within this environment).

---

## Items verified — no remediation needed

| Item | Finding | Evidence |
|------|---------|----------|
| Concurrency guard (`MAX_CONCURRENT_RUNS=4`) | source-confirmed: `threading.BoundedSemaphore(4)` with `acquire(blocking=False)`; isolated semaphore test: 4 acquired, 21 rejected; runtime 429s not reproducible against live uvicorn in this environment | `evidence/api/concurrency_limit.json` |
| Scheduler correctness (FCFS/SJF/Priority/RR) | all 11 hand-verified cases match exactly | `MANUAL_TEST_MATRIX.md` Category D |
| Input validation | 18/18 invalid cases return specific 422 | `MANUAL_TEST_MATRIX.md` Category C |
| Reproducibility | byte-identical results and charts across runs | `MANUAL_TEST_MATRIX.md` B11 |
| Re-open / export match stored payloads | verified | `MANUAL_TEST_MATRIx.md` B7–B8 |
| No fake/hardcoded data | all numbers computed by repository code | `FAKE_FUNCTIONALITY_AUDIT.md` |

---

## Application of fixes

No remediation item has been applied to the repository. Working tree remains clean at commit `e6b4055e`. If the project team elects to apply R1–R2, each change must be followed by re-running the corresponding manual API checks (Categories B, E) and re-verifying reproducibility before any demo.
