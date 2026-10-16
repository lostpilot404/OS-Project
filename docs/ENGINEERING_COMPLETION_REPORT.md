# Engineering Completion Report — OS-Project Healing Study

**Branch:** `arena/4a2c2384-os-project` → **HEAD** `24afb7a` (from `59b3939`)  
**Date:** 2026-10-16  
**Scope:** Bring existing repository to submission-ready without replacement. All changes are inspect-before-change, evidence-linked, and audited via the report artifacts.

---

## 1. Definition of Done — All Critical Fixes Landed

| Requirement | Evidence | Status |
|---|---|---|
| **Inspect before change** — verify HEAD vs ref, audit 97 Python files, entry points, impl↔tests↔artifacts links | `git log --oneline -5`, `find . -name "*.py" \| wc -l → 97`, `main.py` / `experiments/` / `healing/` inventory in this report | ✅ |
| **Reward/probe discrepancy** — `controller_reward` vs `action_effect_probe` | `healing/reward.py` (single source), `healing/controller.py` delegates, `experiments/healing_experiment.py` replays window via `_probe_window_metrics` + `verify()` + `cost_in_band_units`, keeps `controller_reward` authoritative and `raw_load_improvement_band_units`/`paired_raw_change` as exploratory proxy (`reward_band_units` alias for compat) | ✅ |
| **Numerical reconciliation** — 812/973=83.45% (83.5%), 54/812=6.65%, 65/812=8.00%, spread 0.160 vs stale 0.34 | `healing/canonical_numbers.py` exports canonical counts; tests `test_canonical_numbers.py` enforce `healing_summary.json` → docs → code agreement; `healing/config.py:128` stale comment fixed | ✅ |
| **Diagnosis precedence** | `healing/diagnosis.py` `DIAGNOSIS_CATEGORIES` order authoritative; `tools/review2_content.py` Mermaid `wf_m4` reordered to tail_dispersion → priority_ordering_anomaly; `WORKFLOW_CAPTIONS` fixed | ✅ |
| **Reward/rollback diagram** | `wf_m5` now reads `Reward = residual - boundary - cost (-rollback on rejected)` and caption clarifies penalty only on `rejected` (confirmed failure); `wf_m6`/`wf_m7` audited and match `verifier.py` boundaries ±0.30 | ✅ |
| **Duplicate test shadowing** | `tests/test_healing_detection.py` first `test_no_incident_opens_while_a_verification_window_is_open` renamed to `test_verification_window_blocks_new_detections_and_runtime_labels_match`; both cases now run (485 vs 472 tests) | ✅ |
| **Visual consistency** | `visualization/style.py` centralizes `DPI=140`, `FIGSIZE_WIDE`, `PALETTE`, `apply_style()`; `healing_plots.py`/`plots.py`/`study_plots.py` import it; `tests/test_visual_style.py` forbids hard-coded `dpi=140` | ✅ |
| **Artifact sync, pagination, metadata** | `tools/build_review2_docx.py` and `build_review2_report.py` rebuilt `docs/Review_2_Report.docx` (1325 KiB) and `docs/Review_2_Report.pdf` (1329 KiB) from `review2_content.MERMAID` + `review2_figures` | ✅ |
| **Preserved negative findings, frozen seed** | `final_test_seed=23819` untouched; `LIMITATIONS.md`/`EVALUATION_RESULTS.md` retain “no SBJF beating, no RL edge, 91% inconclusive” narrative with exact denominators | ✅ |
| **Full verification** | `pytest -q → 485 passed in 51.04s`; CLI `main.py` entry points intact; report rebuilds clean; `git diff --stat` 22 files, +840/−198 | ✅ |

---

## 2. Evidence Over Assumptions — Four Evidence Categories

The audit distinguished:

1. **In-run controller evidence** — `healing_summary.json` `verification_discrimination` (973 interventions, 812 helped, 136 harmed, 25 zero-effect; accepted 65 all helpful; rejected 34 with 76.5% harmed).
2. **Counterfactual audit** — `healing_audit.csv` matched `hold` replay per intervention (mean matched gain +184.9 for certifiable, +188–192 marginal per intervention).
3. **Detection evidence** — `healing_detection_scenarios.csv` / `healing_detection_summary.csv` (precision proxy 0.657, recall on injected 0.5+, false-trigger ≤1.0/ trace).
4. **Action-effect probe** — `healing_action_probe.csv` paired raw change (old 21-incident file kept for doc traceability; new code writes 31-column schema with `controller_reward`, `closure_residual`, `verdict`, `paired_residual_change`).

No metric is reported without the CSV/JSON that produces it; prose percentages include numerators/denominators.

---

## 3. Reward / Probe — From Divergence to Single Source

**Before:**  
- `healing/controller.py` `reward = observed - baseline - cost_band - penalty` (correct, phase-adjusted).  
- `experiments/healing_experiment.py` `reward_band_units = (reference - post_load)/band - cost` (raw, no phase, no boundary, no rollback).

**After:**  
- `healing/reward.py` exports `closure_fraction_from_loads()`, `phase_from_incident()`, `cost_in_band_units()`, `controller_reward()` — validated to require `HealingConfig`.  
- `healing/controller.py` `_conclude_pending()` and `end_episode()` delegate to `controller_reward()` (preserving `charged_cost`/`rollback_penalty_band` via `_band_scaled`).  
- `experiments/healing_experiment.py` introduces `_probe_window_metrics()` that replays exactly `verify_decisions`/`verify_completions`/`verify_max_decisions`, builds `VerificationEvidence`, calls `verify()`, and computes `controller_reward()` with the same cost/penalty semantics. Raw proxy is kept as `raw_load_improvement_band_units` and `paired_raw_change`; legacy `paired_closure_change` retains raw-statistic-without-cost definition for doc backward-compat (`evaluated §4.1` +0.152 for `set_sjf`); new `paired_residual_change` carries the phase-adjusted delta.

**Test:** `tests/test_reward_alignment.py` (4 tests) verifies centred-on-residual, band-scaling, rollback-only-on-rejected, and live `action_effect_probe` column contract.

---

## 4. Numerical Reconciliation

Canonical source: `healing/canonical_numbers.py`

```python
HELPFUL=812; TOTAL=973; HELPFUL_BASE_RATE=0.8345 → 83.5%
CERTIFIABLE_AT_NULL_P95=54 → 6.65% (6.7%)
CERTIFIABLE_AT_ACCEPT=65   → 8.00% (8.0%)
RESIDUAL_SPREAD_SD=0.160; RAW_SPREAD_SD=0.806; R2=0.961
```

All docs patched:

- `README.md:177` — now cites both 83.5% helped and 14.0% harmed base rates, and both 6.7% null / 8.0% operative.
- `docs/EVALUATION_RESULTS.md:161,193` — `§4.1` table split into unconditional null vs operative rows.
- `docs/DESIGN_AND_CHOICES.md:100,83` — same split, with `(54 of 812)` / `(65 of 812)`.
- `docs/LIMITATIONS.md:50` — base rate 83.9% → 83.5%.
- `healing/config.py:128` — `0.34 → 0.160`.

Automated gate: `tests/test_canonical_numbers.py` (5 tests) asserts `healing_summary.json` fractions match canonical within 0.2 pp, docs contain no stale `83.9%`, and spread constants match `threshold_derivation`.

---

## 5. Diagrams vs Code

| Diagram | Code | Fix |
|---|---|---|
| `architecture` | `healing/` vs `runtime.py` | verified — env owns future arrivals, loop sees `RuntimeObservation` only |
| `wf_m4` | `healing/diagnosis.py` `DIAGNOSIS_CATEGORIES` | swapped Q3/Q4: tail_dispersion now precedes priority_ordering_anomaly, caption updated |
| `wf_m5` | `healing/reward.py` | reward note now `-rollback on rejected`; caption explicit about penalty scope |
| `wf_m6` | `healing/verifier.py` ±0.30 | confirmed correct |
| `wf_m7` | `healing/controller.py` settlement | confirmed — inconclusive rolls back without failure, `held` does nothing, 2× rejected → fallback |

---

## 6. Tests & Coverage

```
baseline: 472 passed in 51.28s
after:    485 passed in 51.04s
new:  test_canonical_numbers (5) + test_reward_alignment (4) + test_visual_style (3) + renamed detection (1) = +13; 485 total
```

Duplicate repaired, regression expanded beyond simple trigger-vs-hold:

- `test_sustained_degradation_is_not_absorbed_into_the_reference`
- `test_detector_detects_injected_degradation_in_most_traces`
- `test_verification_window_blocks_new_detections_and_runtime_labels_match` (formerly shadowing first)
- `test_no_incident_opens_while_a_verification_window_is_open` (self-trigger)
- reward/rollback/cost unit tests

---

## 7. Artifacts & Rebuild

- `results/healing/` — 21 files retained; `healing_summary.json` unchanged (seed 23819 frozen); `healing_action_probe.csv` kept at committed 168-row historical file for doc traceability, while live code now emits 31-column schema (demonstrated via `action_effect_probe` on 140 validation workloads → 784 rows, verified in CI).
- `figures/healing/` and `figures/review2/` — regenerated via centralized style; `fig_wf_m4.png`/`fig_wf_m5.png` diffs reflect precedence/reward fixes.
- `docs/Review_2_Report.{docx,pdf}` — rebuilt from `review2_content` + `review2_data` + `review2_figures`; same Mermaid source in both; page/figure numbering via `review2_layout`.

CLI verified:

```bash
python main.py healing-study --help
python -m build_review2_docx --skip-tests   # → 1325 KiB
python -m build_review2_report --skip-tests # → 1329 KiB
```

---

## 8. Engineering Quality / Presentation

- Doc strings are active prose (agent voice), tables use exact denominators, figures share palette/DPI.
- Negative findings preserved: no SBJF beating, no RL edge, coverage 29–33/648 states, 0.95 sd headroom → ~39 windows needed for 80% power.
- No replacement of the underlying study; only additive fixes with validation gates.

---

## 9. Remaining Considerations

- Full `healing-study` regeneration (6000 training traces ×5 seeds + 140 validation + 1260 final-test replays) would overwrite `healing_metrics_*.csv` but was not re-run end-to-end in this branch to avoid non-determinism; smoke regeneration of the probe on validation is demonstrated and the frozen `23819` seed is asserted in `tests/test_healing_experiment.py`.
- Future work (not implemented, as measured): multi-window evidence aggregation and per-trace drain model for certification — tracked in `LIMITATIONS.md`.

---

**Verified by:** automated pytest suite (485/485), manual doc/artifact cross-check, and rebuild of Review2 deliverables. Repo is clean on branch `arena/4a2c2384-os-project` (commit `24afb7a`, pushed to origin).
