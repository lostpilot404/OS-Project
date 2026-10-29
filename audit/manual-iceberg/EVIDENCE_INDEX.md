# Evidence Index — OS-Project Manual Audit

**Branch:** `arena/1a4dbf1c-os-project`  
**Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`  
**Audit directory:** `audit/manual-iceberg/`  
**Live service:** `http://127.0.0.1:8000/`  
**Capture tool:** `audit/manual-iceberg/evidence/api/capture.py` (urllib only, manual)

**Evidence type key:** MAPI = manual API request; SRC = source inspection; ART = committed artifacts; INF = source-inferred; ENV = environment condition.

---

## E01 — API response captures (manual, live service)

All files under `audit/manual-iceberg/evidence/api/`. Each is a raw JSON response body from `POST`/`GET` against the running FastAPI service, saved by `capture.py` then summarized by `inspect.py`.

| Ref | File | Endpoint | Scenario / input | Evidence type | Confirms |
|-----|------|----------|------------------|---------------|----------|
| E01a | `health.json` | GET /api/health | — | MAPI | service up, engine available |
| E01b | `examples.json` | GET /api/examples | — | MAPI | 4-scenario catalogue |
| E01c | `sim_teaching.json` | POST /api/simulations | scenario=teaching | MAPI | teaching executes, 4 policies, metrics correct (D1–D4) |
| E01d | `sim_recovery_accepted.json` | POST /api/simulations | scenario=recovery-accepted | MAPI | healing accepted: detected t=243, set_sjf, ACCEPTED (E1) |
| E01e | `sim_recovery_rollback.json` | POST /api/simulations | scenario=recovery-rollback | MAPI | healing rollback: detected t=141, INCONCLUSIVE (E2) |
| E01f | `sim_custom.json` | POST /api/simulations | custom FCFS workload | MAPI | custom metrics correct (D9) |
| E01g | `sim_custom_healing.json` | POST /api/simulations | custom + healing | MAPI | `healing.triggered` field present (E4) |
| E01h | `sim_custom_holb.json` | POST /api/simulations | P1 burst=50 + 7× burst=2 | MAPI + INF | no incident fired (E5); work-gate explanation source-inferred (OS-004) |
| E01i | `compare_explicit.json` | POST /api/comparisons | explicit 3-process workload | MAPI | like-for-like comparison, metrics correct (D5–D8) |
| E01j | `compare_from_run.json` | POST /api/comparisons | from run_id | MAPI | comparison derived from stored run |
| E01k | `compare_empty.json` | POST /api/comparisons | {} | MAPI | 422 validation (C18) |
| E01l | `runs.json` | GET /api/runs | — | MAPI | run list |
| E01m | `sim_teaching_reopen.json` | GET /api/simulations/{run_id} | from E01c | MAPI | re-open matches stored (B7) |
| E01n | `sim_teaching_export.json` | GET /api/simulations/{run_id}/export | from E01c | MAPI | byte-identical export (B8) |
| E01o | `404_unknown_run.json` | GET /api/simulations/does-not-exist/export | missing run | MAPI | 404 `unknown_run` code (OS-005 withdrawn — API already uses `unknown_run`) |
| E01p | `422_invalid_scenario.json` | POST /api/simulations | scenario=bogus | MAPI | 422 validation (B10) |
| E01q | `422_*.json` (17 more files) | POST /api/simulations / comparisons | invalid inputs C1–C17 | MAPI | all 18 validation cases return specific 422 |
| E01r | `concurrency_limit.json` | POST /api/simulations ×25 | rapid concurrent burst | MAPI | 4 × 201 + 21 × 429; MAX_CONCURRENT_RUNS=4 enforced (F1) |
| E01s | `research.json` | GET /api/research | — | MAPI | committed aggregates; q_recovery vs rule_wait CI includes zero (G6) |

**Total API captures: 19 files (E01a–E01s), all saved 2026-10-29.**

## E02 — Capture and inspection scripts

| Ref | File | Purpose | Evidence type |
|-----|------|---------|---------------|
| E02a | `evidence/api/capture.py` | urllib-based manual capture of all E01 endpoints | MAPI tooling |
| E02b | `evidence/api/inspect.py` | summary inspector over captured payloads | MAPI tooling |

## E03 — Manual test matrix

| Ref | File | Content | Evidence type |
|-----|------|---------|---------------|
| E03a | `MANUAL_TEST_MATRIX.md` | Categories A–H; 63 checks; 52 PASS, 0 FAIL, 1 PARTIAL, 3 LIMITATION, 7 BLOCKED | MAPI + SRC + ENV |

## E04 — Committed artifacts inspected (offline study)

| Ref | File | Content | Evidence type | Confirms |
|-----|------|---------|---------------|----------|
| E04a | `results/healing/healing_summary.json` | 210-workload healing study (seed 23819): q_recovery interventions=0.65/trace, wait=122.56 vs fixed_sjf 94.66; q_recovery_from_sjf holds (0.0, identical); rule_wait significant, q_recovery vs rule_wait CI includes zero | ART | G2, G3, OS-002, OS-006 |
| E04b | `results/healing/healing_comparisons.csv` | per-trace comparison rows used by research page | ART | G6 |

## E05 — Source-code inspection (no execution substitute)

| Ref | File(s) | Content | Evidence type | Confirms |
|-----|---------|---------|---------------|----------|
| E05a | `dashboard/api.py`, `service.py`, `scenarios.py`, `model.py`, `teaching.py`, `research.py` | FastAPI routes, engine, scenario catalogue, validating model, teaching executor, research reader | SRC | real implementation, no fake data |
| E05b | `scheduler/fcfs.py`, `sjf.py`, `priority.py`, `rr.py` | scheduler classes | SRC | real algorithms |
| E05c | `workload/models.py`, `workload/drift.py` | workload records, drift kinds | SRC | drift implemented offline |
| E05d | `healing/{monitor,detector,diagnosis,verifier,policies,controller,config,burst_estimator}.py` | full control loop + estimators | SRC | OS-004 work-gate reasoning; OS-003 burst estimation |
| E05e | `rl/q_learning.py`, `rl/runtime_controller.py`, `healing/policies.py` (QLearningHealingPolicy) | RL implementation | SRC | OS-002 |
| E05f | `dashboard/web/{index.html,app.js,styles.css}` | frontend | SRC | UI unverifiable without browser (OS-001) |
| E05g | `main.py`, `config.py`, `experiments/{demo,healing_config}.py` | CLI entry points, experiment config | SRC | offline RL entry points |

## E06 — Screenshots / browser captures (RESERVED, NOT CAPTURED)

| Ref | Intended file | Status | Reason |
|-----|---------------|--------|--------|
| E06a | `evidence/ui/homepage.png` | reserved, not captured | no browser binary (OS-001) |
| E06b | `evidence/ui/simulation-studio.png` | reserved, not captured | no browser binary |
| E06c | `evidence/ui/recovery-analysis.png` | reserved, not captured | no browser binary |
| E06d | `evidence/ui/policy-comparison.png` | reserved, not captured | no browser binary |
| E06e | `evidence/ui/research-page.png` | reserved, not captured | no browser binary |
| E06f | `evidence/ui/charts-dom.png` | reserved, not captured | no browser binary |

**No screenshots exist.** Any screenshot filename referenced in this audit is a reservation only. Browser automation is impossible in this environment (no Chrome/Chromium/Firefox, no Xvfb/Xorg, Playwright Chromium download blocked by sandbox host allowlist). See OS-001.

## E07 — Audit deliverables

| Ref | File | Status |
|-----|------|--------|
| E07a | `AUDIT_REPORT.md` | written |
| E07b | `BUG_LEDGER.md` | written (OS-001…OS-006) |
| E07c | `MANUAL_TEST_MATRIX.md` | written (63 checks) |
| E07d | `FAKE_FUNCTIONALITY_AUDIT.md` | written (18 features, none fake) |
| E07e | `EVIDENCE_INDEX.md` | this document |
| E07f | `REMEDIATION_LOG.md` | written (R1–R5) |
| E07g | `FINAL_DEMO_CHECKLIST.md` | written (demo-scoped, excludes unsupported live claims) |

---

## Evidence integrity notes

- No automated test suite was run as a substitute for manual verification (audit rule 2). The repository pytest suite exists but its results were not used to pass any check.
- No evidence was fabricated: every MAPI entry corresponds to a real HTTP call whose body is saved under `evidence/api/`; every SRC entry corresponds to a file read in the repository; every ART entry corresponds to a committed artifact inspected directly.
- Source-inferred conclusions (OS-004 work-gate chain) are explicitly labeled INF, not presented as live observation.
- No screenshots or screen recordings exist; all UI claims are source-reviewed or blocked.
