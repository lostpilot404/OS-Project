# Manual implementation verification — 23 October 2026

## Scope, provenance and method

Manual-review starting checkout: `arena/c3d9429b-os-project`, exact starting HEAD
`d7f2efe3950f35aa577c24bc7031df103b096eb4`. Starting `git status --short` was empty.
The manual-review phase left intentional uncommitted code/test/document/evidence
additions and performed no branch switch, reset, commit, push or historical evidence
replacement. No user changes were present to discard. The later delivery handoff
separately authorizes a new commit and push on this same task branch; use `git log -1`
and the confirmed remote reference for publication status, not this historical phase record.

Environment E1 for every check below unless noted: Debian GNU/Linux 12, Linux x86-64,
CPython 3.11.2; existing `.venv` with locked NumPy/pandas/Matplotlib/pytest plus optional
report dependencies. `pip check` passed. This is an existing Linux checkout, **not a
fresh clone, dependency reinstall, Windows or WSL execution**.

Read/reviewed README, architecture, WSL/onboarding instructions, CLI entry point, demo
source/script, final-report/limitations material, scheduler models/policies/event engine,
workload generation/drift, shared learner, runtime state/controller, healing configuration,
actions, policies/reward/verifier/controller boundaries, experiment/report orchestration,
and relevant tests. No GUI or HTTP API exists to interact with; terminal and Python APIs
are the actual application interfaces. Automated tests support but do not replace the
observations below.

### Evidence conventions

[manual_evidence/](manual_evidence/) holds small real stdout/stderr captures, presenter
scripts, PDF samples and before/after plot images. Text captures start with the exact
executed command and end with its exit code. `plot_defect_reproduction.txt` is explicitly
labelled as a pytest failure excerpt. `pdf_inspection.txt` describes the separate rendering
interaction. Scripts are not fabricated transcripts: open them and rerun their actual
calls. They contain assertions on hand expectations/invariants, not a new scheduler.
Trailing spaces in text captures were removed for Git whitespace hygiene; command
text, numeric results and substantive output were not changed.
Some direct-API failures are intentionally caught; their enclosing script exits 0 only
when the expected rejection occurred. Terminal timings are host-dependent.

New experiment outputs are in `/tmp/os-teacher-*`, not tracked `results/` or `figures/`.
Those temporary files may not persist across sessions. Exact regeneration commands are
in [TEACHER_LIVE_DEMO.md](TEACHER_LIVE_DEMO.md); compact transcripts/images persist here.
The larger original study and academic submissions remain unchanged.

## Executed checks

Each row lists expected behavior, observed evidence and disposition. Full commands for
captured terminal checks are also on the first line of the linked transcript.

| ID / purpose | Exact command or interaction | Expected → actual observed | Exit / result / evidence |
|---|---|---|---|
| M01 checkout/environment | `git status --short`; `git branch --show-current`; `git rev-parse HEAD`; `.venv/bin/python --version`; `.venv/bin/python -m pip check` | Intended branch/commit, no starting changes, consistent environment → matched; later capture includes newly created evidence folder | 0, PASS; [environment.txt](manual_evidence/environment.txt) |
| M02 startup/help | `.venv/bin/python main.py --help` | Enumerate executable commands/flags without training → six commands, demo and expensive default explicitly described | 0, PASS; [help.txt](manual_evidence/help.txt) |
| M03 live primary demo | `.venv/bin/python main.py demo --results-dir /tmp/os-teacher-demo` | Real deterministic two-fixture loop and JSON → printed arrived-ready observations, service segments, incident diagnoses, action, verdicts, rollback/keep and metrics; valid JSON inspected | 0, PASS; [demo.txt](manual_evidence/demo.txt) |
| M04 all four schedulers | `PYTHONPATH=. .venv/bin/python docs/manual_evidence/scenarios.py` | Compare hand-computed process outcomes for all-arrival-zero workload → FCFS/SJF/RR/Priority exact matches, means 4.333333/1.666667/4.333333/3.666667 | 0, PASS; [scenarios.txt](manual_evidence/scenarios.txt) |
| M05 edges/timing | Same M04 command | One late job wait 0/completion 8; explicit allowed empty schedule elapsed 0; endpoint RR arrival admitted before requeue → all four empty/single cases and P1/P2/P1 RR boundary matched | 0, PASS; scenarios.txt |
| M06 normal healing | Same M04 command; full `RuntimeSimulator` + waiting rule on 15 spaced jobs | No degradation → 15 decisions, zero incidents, zero mean waiting | 0, PASS; scenarios.txt |
| M07 triggered/accepted healing | M03, inspect seed 13 JSON and d06/d11 | Monitor → detect → diagnose → SJF → verify → retain → state 205, threshold 75.703, mean 105.167 at t243, residual 0.437455, accepted t319, subsequent SJF | 0, PASS; demo.txt and integrity.txt |
| M08 inconclusive/rollback | M03, inspect seed 3 JSON and service trace | Uncertified intervention rolls back → t141 trigger, 51.000 > 28.203, SJF action, residual 0.205922, inconclusive t193, later FCFS, confirmed_failure False | 0, PASS; demo.txt |
| M09 independent runtime RL | `PYTHONPATH=. .venv/bin/python docs/manual_evidence/runtime_learning.py` | Train real sequential controller, evaluate in-memory model, export audit data → 30 episodes/one seed, 648 Q rows (162×4), six methods × six test workloads = 36 rows; validation-selected six-switch trace | 0, PASS; [runtime_learning.txt](manual_evidence/runtime_learning.txt) |
| M10 inference/reward | Same M09 command, independent direct controller subcheck | Training changes table; inference is read-only/no exploration; return matches integrated waiting → 68 nonzero trained Q entries, 19 inference decisions, Q and visits unchanged, reward −97 = −sum(wait)/4 | 0, PASS; runtime_learning.txt |
| M11 drift input | M04 then `PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --reps 1 --train-episodes 2 --drift phase_transition --out /tmp/os-teacher-drift` | Prefix preserved and suffix distribution changes, full evaluation/report → mixed seed42 boundary t8, original arrivals/IDs preserved; seven workloads, 35 Q records, 70 combined rows, six summary rows | 0, PASS as smoke only; scenarios.txt, [drift.txt](manual_evidence/drift.txt) |
| M12 primary full study | `.venv/bin/python main.py healing-study --results-dir /tmp/os-teacher-healing --figures-dir /tmp/os-teacher-healing-figures` | Train/evaluate/audit/export separately from fixtures → actual five-seed study, reports and three figures generated; numerical tables reproduced | 0, PASS numerically; original plot defect found separately; [healing_study.txt](manual_evidence/healing_study.txt) |
| M13 genuine rejection/fallback | `PYTHONPATH=. .venv/bin/python docs/manual_evidence/output_integrity.py` after M03/M11/M12 | Full default-threshold fairness-rule replay rejects and eventually pins config → poisson_arrivals rep21, fingerprint c555eb898c254d71, two rejections, residuals −0.316225/−0.383115, rollback FCFS, fallback after t189; tail action ID 0 | 0, PASS; [integrity.txt](manual_evidence/integrity.txt) |
| M14 metric/schema checks | Same M13 command | Recompute from process/interval data and CSV → demo waiting sums 1957 and 3308 match interval sums and means; drift grouped detail means equal summary; fresh metrics 3780×32/incidents 2171×71/comparisons 80×7 equal frozen CSVs | 0, PASS; integrity.txt |
| M15 invalid CLI | `.venv/bin/python main.py not-a-command`; `.venv/bin/python main.py demo --quantum 0` | Usage error, no simulation → invalid choice/unrecognized arguments, both exit 2 | 2, PASS rejection; [invalid_command.txt](manual_evidence/invalid_command.txt), [invalid_flag.txt](manual_evidence/invalid_flag.txt) |
| M16 malformed API inputs | M04 command | Reject empty default workload, non-Process row, duplicate PID, negative arrival, zero burst/quantum, unknown family/phase target, action99 → descriptive ValidationError/ConfigurationError for every case | Caught as intended; script 0, PASS; scenarios.txt |
| M17 invalid drift CLI | `PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py --drift unsupported --out /tmp/os-teacher-invalid-drift` | Reject before training → argparse lists supported choices, no output directory | 2, PASS; [invalid_drift.txt](manual_evidence/invalid_drift.txt) |
| M18 output failure | Precreated regular file `/tmp/os-teacher-failures/not-directory`; `.venv/bin/python main.py demo --results-dir /tmp/os-teacher-failures/not-directory`; also `--results-dir /sys/os-teacher-demo` | Actionable path error/no misleading JSON → existing-file and permission-denied errors, writable-directory hint | Both 1, PASS; [output_error.txt](manual_evidence/output_error.txt), [read_only_output.txt](manual_evidence/read_only_output.txt) |
| M19 missing report source | `.venv/bin/python tools/build_plain_report.py --source /tmp/os-teacher-missing.md --output /tmp/os-teacher-missing.pdf` | Failure and no PDF → FileNotFoundError names source; failure probe confirmed no PDF | 1, PASS fail-fast, UX limitation (raw traceback); [missing_report_source.txt](manual_evidence/missing_report_source.txt) |
| M20 bad plot data / optional imports | `PYTHONPATH=. .venv/bin/python docs/manual_evidence/failure_checks.py` | Reject legacy probe before partial output; demo should not need plotting imports → schema error with regeneration advice/no directory; fresh-process simulated missing matplotlib allows demo, rejects plotting import | 0, PASS; [failures.txt](manual_evidence/failures.txt); absence is simulated, not physical uninstall |
| M21 generated PDF | `.venv/bin/python tools/build_plain_report.py --source docs/OS_Project_Final_Report.md --output /tmp/os-teacher-final-report.pdf` | Build without training → 18 pages, byte-identical to original; actual page contents inspected as below | 0, PASS sampled visual/automated; [report_build.txt](manual_evidence/report_build.txt), [pdf_inspection.txt](manual_evidence/pdf_inspection.txt) |
| M22 plot regeneration | `PYTHONPATH=. .venv/bin/python docs/manual_evidence/render_plots.py` | Correct separate scatter/no recovery bars and hold mean line → three fresh figures generated after fix; corrected action-effect image visually reviewed | 0, PASS with no-op label crowding noted; [regenerated_plots.txt](manual_evidence/regenerated_plots.txt) |
| M23 final suite, repeat demo, syntax/dependencies/links/diff | Commands in [final_checks.txt](manual_evidence/final_checks.txt) and final rerun of M03/M04/M09/M13/M20/M22 | All final-state checks after edits; repeated demo bytes equal | PASS if captured commands exit 0; authoritative counts/time in final_checks.txt |

There is no user configuration-file import command, so “missing configuration file” is
NOT APPLICABLE to the current interface, not an invented passing check. Full offline
`study`, legacy full pipeline/train CLI, all canonical drift studies, physically uninstalling
Matplotlib, clean-clone installation, Windows/WSL, real-kernel scheduling, and visual review
of every historical report page were **NOT TESTED in this pass**. Unit/integration tests
are not relabelled as those manual executions.

## Verified defects and fixes

### D01 — runtime report fabricates model count for custom runs (fixed)

Reproduction: run M09 with training_seeds=(1001,), then read runtime_report.md after its
means table. Actual before fix: “Q-learning rows average across the five independently
trained models” while the design and Q CSV show one model. Expected: report configured
model count, without changing result aggregation. Fix in
`experiments/runtime_experiment.py::_render_report` derives count from configuration.
Regression assertions in `tests/test_runtime_experiment.py` check one model and absence
of the false five-model sentence. Fresh M09 report now says “across 1 independently
trained model(s)”; data and negative performance finding are preserved.

### D02 — reward scatter reuses closed recovery figure (fixed)

Reproduction: M12, open healing_action_effects.png. Actual: recovery-outcome bars,
legend and categorical axes remain underneath the reward scatter; labels and scale
become misleading. Code in `visualization/healing_plots.py` closed figure 2 but reused
its `fig, axes` when drawing figure 3. Expected: independent scatter/diagnosis figure.
Fix: construct a new figure for action effects. Strengthened canonical-probe test captures
actual saved figure objects and verifies independence, zero recovery bar patches on the
scatter and one scatter collection. New assertion failed before fix (1 failed, 4 passed),
then passed after fix. See [reproduction excerpt](manual_evidence/plot_defect_reproduction.txt),
[before](manual_evidence/action_effects_before.png), [after](manual_evidence/action_effects_after.png).

### D03 — dashed “hold level” uses all actions' mean (fixed)

Code/figure review after D02 found the marker took `probe[controller_reward].mean()`
while claiming hold. Expected: mean over action_name == hold, no marker if no hold rows.
Fixed subset selection and added test with distinct hold/overall means (line at −0.30,
not −0.275). No reward definition, scheduling logic or research metric was changed.

No thresholds/seed design were retuned. Historical PNGs/reports remain untouched as
requested; their old visualization issue is not concealed. Use newly regenerated figures
for class, not the historical action-effect PNG.

### Harness corrections (not product defects)

The first presenter-script draft used nonexistent slice attributes `start`/`end`; corrected
to actual `start_time`/`end_time` before capturing the final scenarios transcript. The first
optional-dependency probe installed an import blocker after Matplotlib was cached, so it
did not test absence; corrected to a fresh interpreter before any plotting import. These
initial harness failures are not represented as successful product tests. Genuine final
captures correspond to the final source. No test expectation was weakened.

## Research integrity and report review

Observed fresh primary means: FCFS 130.837778, SJF 94.657778, waiting rule 122.388254,
Q recovery 122.564063. Q-minus-rule interval [−0.04,+0.48] supports no superiority claim.
Ledger preserves Q rejected/inconclusive counts 2/625, waiting-rule inconclusive 129,
and fairness-rule rejected 31. Two accepted/inconclusive demo fixtures are illustrative
only. Counterfactual audit is offline and never visible to live inference. The stable
controlled study still has 0.70 incidents/trace and 68.6% detected traces: the detector is
not perfectly specific. Do not hide the fairness rule's worse waiting or the Q smoke loss.

Runtime smoke: Q wait 100.466667 versus FCFS 87.388889 and SJF 55.433333. It demonstrates
training/consumption/inference, not superiority. Phase-transition smoke uses only one
workload/family and two training episodes/model; deterministic within-family bootstrap
intervals can collapse. These values are not fresh generalizable research conclusions.
The input target short_jobs is a configured distribution, not a guarantee every suffix
burst is tiny. Estimator noise is uncalibrated; fixed SJF remains an oracle comparator.

Inspected newly generated healing/runtime/drift report text and tables against inputs,
CSV shapes, grouped means and preserved incidents. Reviewed all three newly generated
healing figures visually; D02/D03 came from that inspection, not just file existence.
Corrected scatter still has overlapping labels for near-identical no-op points: this is a
remaining readability limitation; exact CSV rows and the waiting chart are clearer for
classroom explanation. Do not interpret overlapped labels as additional performance data.

Final PDF: generated 18 pages; all rendered at 96 DPI with PyMuPDF 1.26.6 installed only
in `/tmp/os-pdf-tools`, not added as a project dependency. **Visually reviewed pages
1, 4, 6, 10, 14 and 18**, not all pages. Sample text, margins, equations, page numbers,
source diagrams and result prose were legible without noticed clipping; page 18 is
intentionally short. Retained actual rendered [page1](manual_evidence/report_page_01.png)
and [page14](manual_evidence/report_page_14.png). PDF remains the historical submission:
its branch names/test counts describe that original run, not current HEAD. The literal
Mermaid workflow blocks are printed source by design. Automated verifier separately
checks all-page text/numbering/operators, seven source blocks and deterministic PDF bytes;
that is not all-page visual certification. Earlier PDF/DOCX submissions were preserved,
not rebuilt or visually recertified in this review.

## Teacher dashboard review (visual layer added after the manual review)

The classroom dashboard (`dashboard/`, `tools/build_teacher_dashboard.py`) is a consumer
of recorded output, not a second simulator. Reviewed on the same environment E1:

* `main.py demo --results-dir /tmp/dashbuild` then
  `tools/build_teacher_dashboard.py --input /tmp/dashbuild/demo.json --output
  /tmp/dashbuild/index.html` wrote a 213.0 KB single-file page (16 inline SVG charts).
* The builder cross-checks the trace before drawing: per-process waits must sum to the
  recorded totals, busy/idle/switch counts and the fingerprint must agree with the
  intervals and both runs, every incident must sit on a recorded dispatch, and an incident
  that records a rollback must match the policy the trace ends under. Deliberately broken
  copies (edited metric, edited outcome, wrong verdict label, overlapping dispatch, wrong
  fingerprint) are rejected with the field named; nothing is written.
* Output handling observed: existing file refused without `--force`, non-HTML suffix
  refused, input path refused, and any path inside `results/healing/`, `results/runtime/`,
  `results/study/`, `results/drift/`, `figures/` or `docs/manual_evidence/` refused. A
  failed build leaves the destination byte-identical and no temporary file behind.
* Generated-page checks: no `http(s)://` URL, no `src=`, no `<link>`, unique element ids,
  every `url(#id)` reference resolves, balanced tags, and the demo file's SHA-256 prefix on
  the page matches the input `demo.json`. Page text reproduces the recorded values (130.47
  vs 150.60 for seed 3, 220.53 vs 255.80 for seed 13; study means 130.84 / 94.66 / 122.39 /
  122.56) and labels seed 3 *inconclusive* with rollback to FCFS and seed 13 *accepted*
  with SJF kept — the two are never interchanged.
* The research panel is generated from `results/healing/healing_summary.json` and
  `healing_comparisons.csv`; its sentences are derived from the values, including that the
  Q-minus-rule interval [−0.04, +0.48] includes zero. The three-process worked example is
  executed with the real `scheduler/` classes at build time and fails the build if the
  documented hand calculation stops matching.
* New tests: 46 in `tests/test_teacher_dashboard.py`; full suite **582 passed**.
* Visual review was performed on the SVG itself and on PNGs rasterised from the page's
  embedded SVG ([dashboard/](manual_evidence/dashboard/), `render_chart_png.py`), because
  this sandbox has no browser. **The page was not opened in a real browser here, and the
  Windows PowerShell commands were not executed here**; both are marked unverified in
  [TEACHER_LIVE_DEMO.md](TEACHER_LIVE_DEMO.md). Tracked research artifacts, figures and
  submission documents were not modified (only `.gitignore`, README and teacher docs).

## Live dashboard application (interactive layer, this pass)

The static page above replays a saved `demo.json`. This pass adds a **running application**:
`tools/serve_dashboard.py` starts a FastAPI service (`dashboard/api.py`) that executes the
project's own engine on request, and `dashboard/web/` serves the console that drives it.
`dashboard/service.py` is the only adapter: it validates a request, calls
`experiments.demo.run_fixture` / `run_healing_workload` / `scheduler.POLICY_CLASSES`, validates
the result through the existing `dashboard.model` schema and returns it. There is no second
scheduler in Python or in JavaScript.

### Environment E2 (manual browser session)

* Linux x86-64 container, CPython 3.11.2, FastAPI 0.121.2, Uvicorn 0.40.0.
* Browser: **HeadlessChrome 138.0.7204.0** (`@sparticuz/chromium` 138.0.2 through
  `puppeteer-core` 24), driven by scripted keyboard/mouse interaction against
  `http://127.0.0.1:8000`. The image ships without NSS/NSPR, so a local ABI shim
  (`/tmp/nss-stub`, outside the repository) was linked by `LD_LIBRARY_PATH` to let the
  browser start. This is a real rendering engine executing the real page and real requests,
  but it is **not** a desktop Chrome/Edge-Firefox window on Windows: font fallbacks, native
  scrollbars and OS widgets are those of the headless Linux shell.
* Server command: `.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --port 8000
  --no-browser --log-level warning`.

### Executed browser checks

| ID | Procedure | Expected → actual observed | Result |
|---|---|---|---|
| W01 | Open `/`, wait for the scenario list | Page renders, engine pill reports the live probe → "Engine ready · 3.11.2"; `/api/health` reported `status ok`, probe 0.14 ms; four scenario cards | PASS |
| W02 | Block all `/api/` requests, then press Run | Explicit failure, no substitute values → banner "The simulation did not run. The backend could not be reached…"; pill "Backend unavailable"; Retry button appears; the previous result stays labelled as previous | PASS |
| W03 | Restore the backend, press Retry connection, run again | Recovery without reload → pill "Engine ready · 3.11.2"; new run id and new result rendered | PASS |
| W04 | Load each example in turn | Scenario name, description, fixed parameters and expected behaviour match the catalogue; the previous result is cleared, not left on screen → teaching (processes 3, quantum 2, switch 0), seed 13 (`mixed`, waiting-time rule), seed 3, custom (editable table) | PASS |
| W05 | Run **Successful recovery (seed 13)** | Accepted recovery from the engine → verdict `accepted`, badge "accepted → SJF", residual +0.4375 vs accept boundary +0.30, window t=243→319, mean wait 220.53 vs FCFS 255.80 (Δ −35.27), 14 switches, 97.18 % utilisation, elapsed 497 | PASS |
| W06 | Run **Inconclusive verification and rollback (seed 3)** | Inconclusive rollback, not a success → verdict `inconclusive`, residual +0.2059 inside [−0.30, +0.30], "neither proven nor rejected", final policy FCFS, confirmed failure "no", mean wait 130.47 vs FCFS 150.60 (Δ −20.13) | PASS |
| W07 | Run the teaching workload, switch policy tabs, click a block | Hand-checkable values from the real schedulers → 4.33 / 7.33 / 4.33 waiting / turnaround / response, 2 switches, 100 % utilisation, throughput 0.3333, elapsed 9; Priority tab shows P2 start 0 / completion 3 / wait 0; selected block read-out "P1 Priority, t=3 → t=8 (5 u), arrival 0, burst/priority 5/2, completion 8, total waiting 3 u" | PASS |
| W08 | Custom workload: change P3 burst 9 → 12, policy SJF, run | The edited configuration reaches the backend → new fingerprint, mean waiting 8.40, 4 switches, workload subtitle shows the new burst | PASS |
| W09 | Custom workload: burst 0, then duplicate pid | Client rejects with an actionable message and sends nothing → inline "Process 1: burst must be at least 1 time unit." + banner, **0 POSTs**, result panel still hidden; duplicate pid "Duplicate pid 1."; after correcting to 6 the same form ran (mean 9.20) | PASS |
| W10 | Compare all four policies (teaching and seed 13) | Four separate executions on the identical processes → teaching: SJF 1.67 best vs FCFS 4.33; seed 13: FCFS 255.80, SJF 172.13, RR 405.53, Priority 254.60, best SJF highlighted, quantum/switch cost stated | PASS |
| W11 | Trip the duplicate-submission guard, then change scenario mid-run | One request only → button disabled and labelled "Running…", three rapid clicks produced **1 POST**; scenario card clicks ignored while running (teaching stayed unselected); result belonged to the started scenario | PASS |
| W12 | Delay a comparison, then start a newer run before it returns | The late response must not overwrite the newer result → comparison section stayed "No comparison executed for this run yet"; run provenance showed the newer rollback run | PASS |
| W13 | Export the displayed result | Downloaded JSON equals the displayed run → `os-dashboard-recovery-rollback-32c07992.json`, `run_id` matches the displayed id, exported `avg_waiting_time` 130.466667, verdict `inconclusive`, final policy FCFS | PASS |
| W14 | Keyboard-only flow | Skip link → nav → scenario cards → Run → policy select all reachable; Enter on a card selects it, Enter on Run executes ("Completed in 1.06 ms…"); ArrowRight in the policy tablist switches the trace; focus ring visible on cards, buttons and dispatch blocks | PASS |
| W15 | Layouts at 1440×980, 900×900, 420×900 | No page-level horizontal overflow (`scrollWidth == innerWidth` at all three widths); charts scroll inside their own frame; tables scroll horizontally; text remains legible; sticky result nav and run button stay usable | PASS |
| W16 | Research panel | Committed aggregates, clearly separated from the current run → badge "HISTORICAL · READ-ONLY", source `healing_summary.json`, split `final_test`, 210 workloads, seed 23819, 130.84 / 94.66 / 122.39 / 122.56, Q−rule +0.18 (95 % CI −0.04 to +0.48) with the "did not establish" caveat visible | PASS |
| W17 | API error surface (through the UI and by HTTP) | Every rejection is explicit → 422 with field-named messages for unknown scenario, empty workload, burst 0, duplicate pid, quantum 0/65, unknown policy, unsupported option, malformed JSON; 404 for an unknown run; 500 with a message (no traceback) when the engine raises | PASS |

Console output during the final pass contained **no** JavaScript errors, failed requests or
warnings, other than the two `net::ERR_FAILED` entries produced deliberately by the
backend-loss test (W02).

### Defects found in the browser and fixed

1. The empty `#banner` container rendered as a 1-px box because `.banner` overrode the
   `hidden` attribute — fixed with a global `[hidden] { display: none !important; }`.
2. The Run button was clipped behind the sticky sidebar block at 980 px height — the sidebar
   now scrolls its scenario list while the run block stays visible.
3. The custom-workload process editor stretched across the full panel with misaligned
   columns — constrained to 640 px with right-aligned numeric cells.
4. A tie against FCFS printed "0.0 % lower than FCFS"; it now reads "identical to the FCFS
   baseline".
5. The teaching scenario reported "Throughput n/a" because the adapter dropped two engine
   metrics — `dashboard.teaching` now carries `throughput` and `cpu_utilization` from
   `compute_metrics`.
6. The research panel stacked a full-width chart above a half-empty column; it now shows the
   chart, then primary arms and caveats side by side.

### Automated checks added this pass

* `tests/test_dashboard_api.py` — 46 tests: health (including a simulated engine failure),
  catalogue, all four scenarios cross-checked against direct engine calls, hand-calculation
  assertions, custom workloads (all four policies, healing on/off), validation matrix,
  malformed JSON, engine-failure and schema-rejection paths, stored-run read/export,
  comparison by run id and by explicit workload, per-run directory isolation, four
  concurrent runs, bounded history with directory eviction, shutdown cleanup, "no writes
  inside the repository", the 429 guard, frontend self-containment, and research fidelity
  against `results/healing/healing_summary.json`.
* Full suite **628 passed** (`pytest -q`, 61.3 s) after the change; `pip check` reports no
  broken requirements; `tools/verify_final_delivery.py` exits 0.
* The CLI demo is unchanged: `main.py demo` output and `demo.json` are byte-identical to the
  pre-refactor baseline (`c3f4826d…`), because `experiments/demo.py` now calls the same
  function the API uses.

### Evidence

Screenshots from those headless sessions and the machine-readable record are in
[manual_evidence/web/](manual_evidence/web/) (`browser-checks.json` holds the captured
values, `VERIFICATION.md` the procedure log). They are **headless-browser screenshots**, not
desktop-window screenshots, and the Windows PowerShell commands were **not executed here**.

## Final disposition and remaining limitations

Read [TEACHER_LIVE_DEMO.md](TEACHER_LIVE_DEMO.md) for the short executed sequence and
optional study commands; [TEACHER_QA.md](TEACHER_QA.md) distinguishes the three learning
workflows and directs implementation questions to source/runtime evidence.

Only focused report/plot/service code and regression assertions changed; scheduling/learning
methodology and existing source assets/results/submissions are preserved. The full test
count is 628 (536 at the time of the original review, +46 for the static dashboard, +46 for
the live API). Final actual timing/results are in final_checks.txt and the handoff.
The manual review itself did not authorize publication. A subsequent explicit user
authorization permits the final delivery commit and push on `arena/c3d9429b-os-project`.
Fresh pre-publication verification is captured in
[delivery_checks.txt](manual_evidence/delivery_checks.txt); earlier transcripts above
remain historical execution evidence, not a substitute for that rerun.

Known limits: no clean clone or Windows/WSL test; no real kernel integration; limited
PDF visual sample; synthetic workloads/oracle information/sparse learning coverage;
raw traceback for missing report input; crowded no-op plot annotations; no persistent
model loader demonstrated. This is evidence of the exercised implementation, **not a
claim of exhaustive verification of every environment or research workflow**.
