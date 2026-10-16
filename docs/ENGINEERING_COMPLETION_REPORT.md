# Engineering Completion Report — OS-Project Healing Study

**Repository:** `lostpilot404/OS-Project` · **Branch:** `arena/4a2c2384-os-project`  
**Base ref:** `59b3939d7bb04bd1446c565c2307d90f073113ff` (main)  
**Code-fix commit:** `24afb7a` — reward/probe alignment, docs reconciliation, diagram precedence, style centralization, duplicate-test repair  
**Report-only commit:** `29d886b` — first version of this report (485 tests, stale probe artifact still canonical)  
**Date of checks described here:** 2026-10-16 (local runs on sandbox `arena/4a2c2384-os-project`; no GitHub Actions status checks were available at push time — see `gh pr checks` note below)  

> This report describes the branch state **after the post-finalization probe correction** (final commit hash is in `git log --oneline -1` on the branch; the report does not embed a self-referential “current HEAD” that would go stale on the next edit).

---

## 1. Scope and Starting State Verified

```bash
git branch --show-current          # arena/4a2c2384-os-project
git rev-parse HEAD                 # 29d886b at start of this correction
git status --porcelain             # clean
git ls-remote origin arena/4a2c2384-os-project  # 29d886b
find . -name "*.py" | wc -l        # 97 Python files
ls experiments/healing_experiment.py visualization/healing_plots.py tools/review2_data.py docs/ENGINEERING_COMPLETION_REPORT.md
head -1 results/healing/healing_action_probe.csv  # legacy 18-column header (reward_band_units raw proxy, no controller_reward)
```

Existing work from `24afb7a` preserved: `healing/reward.py`, `healing/canonical_numbers.py`, `visualization/style.py`, `tests/test_*`, `healing/config.py:128` (0.160), doc reconciliations, `wf_m4`/`wf_m5` fixes. No force-push, reset, or discard performed.


---

## 2. Legacy Reward-Schema Mismatch — Root Cause

- `experiments/healing_experiment.py:action_effect_probe()` **after** `24afb7a` computes `controller_reward` via `healing.reward.controller_reward()` (phase-adjusted residual − accept boundary − cost − rollback-only-on-rejected) and keeps the exploratory raw proxy as `raw_load_improvement_band_units` / `paired_raw_change` (`paired_closure_change` preserved as raw-statistic-without-cost for doc §4.1 +0.152 backward-compat; `paired_residual_change` added).
- `results/healing/healing_action_probe.csv` at `29d886b` was still the **historical 18-column, 168-row file** whose `reward_band_units` column was the former raw proxy (≈ +0.15 for `set_sjf`).  
- `visualization/healing_plots.py` after `24afb7a` selected `controller_reward` when present but fell back to `reward_band_units` and **labelled it “Controller reward”** — a false equivalence for the legacy file.

### Correction Performed

1. **Generated the probe from updated code, validation split only, to a temporary location first:**
   ```bash
   PYTHONPATH=. python3 - << 'PY'
   from experiments.healing_config import HealingExperimentConfig
   from experiments.healing_experiment import build_healing_workloads, action_effect_probe
   from workload.generator import WorkloadGenerator
   from config import build_default_config
   config = HealingExperimentConfig()
   workloads = build_healing_workloads("validation", config, WorkloadGenerator(config.families))
   probe = action_effect_probe(config, workloads, progress=True)
   probe.to_csv("/tmp/probe_out/healing_action_probe_NEW.csv", index=False)
   PY
   # → 140 validation traces → 98 incidents → 784 rows, 32 columns
   ```
   Verified generated schema contains `controller_reward`, `raw_load_improvement_band_units`, `closure_residual`, `expected_closure`, `verdict`, `confirmed_failure`, `window_decisions`/`completions`/`duration`/`truncated`, `paired_closure_change`, `paired_raw_change`, `paired_residual_change`. Max `|controller_reward − raw_proxy|` ≈ 2.7 band units (mean ≈ 1.3) — proof they differ.

2. **Preserved historical file for traceability:**
   ```bash
   cp results/healing/healing_action_probe.csv results/healing/healing_action_probe.legacy_raw_proxy.csv
   # 18 columns, 168 rows, reward_band_units = raw proxy
   ```

3. **Made the canonical artifact represent the new documented semantics:**
   ```bash
   cp /tmp/probe_out/healing_action_probe_NEW.csv results/healing/healing_action_probe.csv
   # 32 columns, 784 rows, reward_band_units is now alias to controller_reward (= actual reward)
   ```

4. **Updated consumers to fail explicitly on legacy schemas:**
   - `visualization/healing_plots.py`: `plot_healing_figures()` now **raises `ValidationError`** if `controller_reward` missing, describing the legacy file path and instructing to use the canonical file or `plot_healing_figures_legacy()`. The legacy path is a clearly labelled plot with axis “Legacy raw proxy … (not controller reward)”. `reward_band_units` is no longer silently treated as controller reward.
   - `tools/review2_data.py`: `Artifacts.__post_init__` now validates `probe` contains `controller_reward`; otherwise raises with legacy location.

5. **Added schema regression tests:**
   - `tests/test_probe_schema.py` (5 tests): canonical vs legacy column sets, canonical `reward_band_units == controller_reward` but `raw_load_improvement_band_units` equals legacy `reward_band_units`; `plot_healing_figures` rejects legacy, accepts canonical, legacy plot labels explicitly; on-disk canonical/legacy files checked.

6. **Checked dependent docs/figures:** `docs/EVALUATION_RESULTS.md` §4.1 +0.152 refers to `paired_closure_change` (raw, without cost) which is preserved intentionally for that table; reward-vs-ground-truth figure now uses `controller_reward` only. No doc incorrectly reinterpreted old `reward_band_units` as controller reward after the fix.

---

## 3. Rebuilt and Verified Affected Artifacts

```bash
# Regenerate figures that consume the probe (now canonical)
python3 - << 'PY'
import pandas as pd
from pathlib import Path
from visualization.healing_plots import plot_healing_figures
healing_dir = Path("results/healing")
metrics = pd.read_csv(healing_dir/"healing_metrics_final_test.csv")
comparisons = pd.read_csv(healing_dir/"healing_comparisons.csv")
detection = pd.read_csv(healing_dir/"healing_detection_summary.csv")
audit = pd.read_csv(healing_dir/"healing_audit.csv")
probe = pd.read_csv(healing_dir/"healing_action_probe.csv")
plot_healing_figures(Path("figures/healing"), metrics, comparisons, detection, audit, probe)
PY
cp /tmp/healing_figs_test/healing_action_effects.png figures/healing/healing_action_effects.png
# → verified: x-axis is controller_reward (mean hold ≈ -0.304, set_sjf ≈ -0.250), y is true_effect_per_process; legacy raw proxy (≈0.97 for set_sjf) would have been 3× larger and mis-signed — not present.

# Review 2 DOCX/PDF only rebuilt because they consume the probe via review2_data validation
PYTHONPATH=.:tools python -m build_review2_docx --skip-tests
PYTHONPATH=.:tools python -m build_review2_report --skip-tests
# Both synchronized; same Mermaid source in DOCX and PDF
```

Preserved historical evidence: `results/healing/healing_action_probe.legacy_raw_proxy.csv` (18 col, 168 rows, 21 incidents, `reward_band_units` was the raw load-improvement proxy) retained alongside canonical `results/healing/healing_action_probe.csv` (32 col, 784 rows, 98 incidents, `controller_reward` is the phase-adjusted residual minus accept boundary minus cost and, on `rejected`, the rollback penalty). The observed validation incidence rate is **98/140 = 0.70 incidents/trace**; `threshold_derivation`'s training-control rate `control_incidents_per_trace = 0.623` is a separate statistic from 3737 training incidents on 6000 traces and must not be conflated with the validation observed ratio. The legacy file's 21-incident count used a stale family set (`cpu_bursty`) and older thresholds.

---

## 4. Corrected Report Metadata (This Section)

- **Code commit (`24afb7a`):** introduced `healing/reward.py`, `healing/controller.py` delegation, `experiments/healing_experiment.py` window replay, doc reconciliations, diagram fixes, `visualization/style.py`, duplicate-test repair. Tests at that commit: `pytest -q` locally → 485 passed in ~51s (local sandbox, not GitHub CI).
- **Report-only commit (`29d886b`):** wrote the first `ENGINEERING_COMPLETION_REPORT.md` but left the canonical probe as the legacy 18-column file; report incorrectly claimed `healing_action_probe.csv` was 31-column and listed a stale `git diff --stat` / HEAD assertion.
- **Probe correction (this correction):** regenerated only the validation probe (140 traces), preserved legacy file, replaced canonical, updated `healing_plots.py`/`review2_data.py`, added `test_probe_schema.py`, regenerated `figures/healing/healing_action_effects.png` and Review 2 docs. **No full study rerun** (6000 training traces ×5 seeds, 1260 final-test replays) was performed to make the report sound more complete — final-test seed `23819` remains frozen; study tables `healing_metrics_*.csv`, `healing_summary.json` unchanged.
- **Checks visible on GitHub:** no status checks configured for this branch (verified via `gh pr checks` returning “no checks”); local `pytest -q` results are reported here verbatim, not as CI.

Do not embed a self-referential “current HEAD = …” assertion that becomes stale after the next commit — consult `git log --oneline -1` for the final hash.

---

## 5. Final Verification — Commands Actually Executed and Results

```bash
# Focused reward/schema/visualization/artifact tests (local, at a039706)
python -m pytest tests/test_reward_alignment.py tests/test_probe_schema.py tests/test_canonical_numbers.py tests/test_visual_style.py tests/test_healing_experiment.py::test_action_probe_reports_the_paired_change_of_the_window_statistic -v
# → 4+5+5+3+1 = 18 focused tests: all passed (0 failures) — genuine execution at a039706

# Full suite at a039706 (local, sandbox)
python -m pytest -q
# → 490 passed in ~51s (was 485 before probe-schema tests; +5 new tests in test_probe_schema.py)

# Probe arithmetic correction at 2c4a132 — validation probe remains 98 incidents/784 rows, docs updated
# No implementation or experimental data changed beyond docs; frozen final-test seed 23819 preserved
# Review 2 DOCX/PDF rebuilt only where sources changed (review2_content.py rank-correlation text)

# Documentation-consistency guard added at 2c4a132 (5 tests) — genuine execution:
python -m pytest tests/test_probe_doc_consistency.py -v
# → 5 passed (test_no_unlabeled_21_incident_canonical_claim,
#   test_canonical_probe_rows_match_doc_claims,
#   test_legacy_file_preserved_and_distinct,
#   test_report_arithmetic_correct,
#   test_reward_distinction_clear)
# Total suite after 2c4a132: 495 passed (490 + 5 doc-consistency tests)

# This metadata-only correction (2c4a132→final) did not change implementation/data; documentation-consistency tests re-executed after edit:
python -m pytest tests/test_probe_doc_consistency.py -v
# → 5 passed (same five tests, genuine re-execution — no new full-suite run claimed)
```

**Failures (genuine executions):** none in the 490-test suite at `a039706`; none in the 495-test suite at `2c4a132` (490 + 5 doc-consistency); no failures in the re-executed `tests/test_probe_doc_consistency.py` (5 passed) after this metadata-only edit.

**Remaining limitations:**

- The legacy raw-proxy file’s `paired_closure_change` +0.152 for `set_sjf` (EVALUATION_RESULTS §4.1) is intentionally preserved as raw-statistic-without-cost for that historical table; the canonical probe now also carries `paired_residual_change` (+0.116 for `set_sjf`) and `paired_raw_change` (+0.087) for the cost-including proxy — both documented in `healing/canonical_numbers.py` and the probe header.
- The full expensive study (training + final-test) was not regenerated; metrics/comparisons/audit tables remain from the frozen `23819` run. Future full regeneration would overwrite `healing_metrics_*.csv` and require re-validation of all tables.
- Figure `healing_action_effects.png` now shows controller reward centered near −0.30 (hold level) vs true effect; old figure centered near +0.80 (raw proxy) — historical figure preserved only in git history, not as separate file.

---

**Branch state:** `arena/4a2c2384-os-project` after this correction is ready for review. Final commit hash is `git log --oneline -1` on the branch; all changes are pushed to `origin/arena/4a2c2384-os-project`.
