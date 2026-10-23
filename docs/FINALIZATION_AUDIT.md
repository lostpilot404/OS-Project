# Current engineering audit — 23 October 2026

## Provenance and repository disposition

This maintenance pass started on `arena/c3d9429b-os-project` at
`5275cd31a53cfd724dd336d54184e73c6421d80d`. Starting tracked/untracked/ignored
status was empty; there was no pre-existing work to discard or back up. Origin is
`https://github.com/lostpilot404/OS-Project.git`. The audit phase performed no branch
switch, reset, commit, push or research-evidence replacement. The subsequent delivery
handoff authorizes a new local commit on this same branch; use `git log -1` for the
delivery commit, and compare it to the starting HEAD above for the full patch. No push
is authorized by this handoff. This section supersedes earlier maintenance status
claims below, not their historical experiment provenance.

Initial inventory: 220 tracked files: docs 21, evaluation 6, experiments 16, figures 24,
healing 12, results 54, rl 12, scheduler 7, tests 41, tools 8, visualization 5,
workload 5, plus nine root files. No unrelated upload, disposable tracked cache,
local environment or personal file was confirmed. **No files were deleted or moved.**
The older reports, PDF/DOCX submissions, raw-proxy archive, figures and drift failures
are legitimate project history, not clutter. Local test caches and `.venv` generated
by verification are ignored, not deliverables. Secret-pattern scanning of tracked text
found no private-key headers, GitHub token patterns or AWS access-key patterns; this
is not an exhaustive security certification. `.gitignore` now excludes local `.env`
files and private key extensions, retaining `.env.example` eligibility.

## Implemented fixes

1. `workload/generator.py`: phase_transition was advertised but always raised because
   it omitted the required target. The convenience API now uses a documented
   short_jobs suffix target, with a keyword override; original arrivals/IDs and stable
   prefix remain unchanged. `experiments/drift_uncertainty_experiment.py` supplies
   the same explicit target and validates drift before training/output creation.
   An actual smoke run exposed a second target-less DriftSpec in report formatting;
   it was fixed to use the declared injection constant.
2. Drift report prose previously embedded fixed performance numbers and overrode
   CI interpretation based on drift name, including declaring a positive lower bound
   to include zero. New prose interprets each measured **unrounded** interval, reports
   actual record counts and removes unsupported per-run superiority/practical-effect
   claims. Corrected waiting formula to completion minus arrival minus service.
3. Drift regression aggregation was conditional on exactly 70/350 records, leaving
   other repetition counts without Q regression rates. It now groups by family,
   repetition and fingerprint, averages training seeds, and checks matching keys.
4. `visualization/healing_plots.py`: invalid legacy probe input previously wrote two
   partial figures before failing. Schema validation now occurs before side effects.
   Empty-arm legend calls are guarded, not warnings-suppressed. The tail chart title
   now correctly says mean per-trace P95, not pooled tail.
5. README's estimator snippet used undefined scheduler_config and implied CLI custom
   config flags; it now constructs the scheduler config and identifies the Python API.
   Added TEAMMATE_ONBOARDING; clarified optional report dependencies.

Regression coverage: seven new drift tests (four parameterized CI cases included),
plus strengthened existing legacy-schema test asserting no output directory is made.
The new integration assertion initially incorrectly expected the reference arm itself
to have a regression rate; corrected the test to match documented reference semantics
(None), not an implementation defect. No existing failing test was removed or weakened.

## Actual execution evidence

Host: Debian Linux x86-64, Python 3.11.2. A direct system pip install was refused by
PEP 668, and system Python had no pytest. Created a fresh **local virtual environment**;
installed unchanged core pins plus optional report requirements successfully. This is
not a clean-clone test or WSL execution. Optional report transitive dependencies are
not fully locked; core requirements.lock remains unchanged.

Executed from repository root (all successful except the explicitly described initial
system install and initial phase-transition smoke):

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock -r requirements-report.txt
.venv/bin/python -m pip check
.venv/bin/python main.py --help
.venv/bin/python main.py demo --results-dir /tmp/os-audit-demo
.venv/bin/python main.py experiment --results-dir /tmp/os-audit-legacy \
  --figures-dir /tmp/os-audit-legacy-figures
.venv/bin/python main.py healing-study --results-dir /tmp/os-audit-healing \
  --figures-dir /tmp/os-audit-healing-figures
.venv/bin/python main.py runtime-experiment --results-dir /tmp/os-audit-runtime
.venv/bin/python main.py train
.venv/bin/python -m pytest -q tests/test_drift_reporting.py tests/test_probe_schema.py
.venv/bin/python tools/verify_final_delivery.py
.venv/bin/python -m pytest -q
```

Initial complete suite: **529 passed, one warning in 46.53 s**. After fixes: **536
passed in 46.78 s**, no warnings. Focused drift/probe run: **12 passed in 2.43 s**.
Final-state rerun is also required after documentation edits (see final response).
No lint/type tool is configured; syntax compilation and git diff whitespace checks are
used rather than inventing a static-analysis pass.

Demo JSON contains real inputs, observations, intervals, incidents and metrics. Its
fixtures produce mean waits 130.46666666666667 and 220.53333333333333; the first is
inconclusive/rollback, the second accepted. They are rules, not trained RL results.
The PDF verifier independently runs the demo and matches the captured transcript,
checks frozen means and artifact counts, regenerates the final PDF byte-for-byte,
checks text/color/page operators across **18 pages**, seven Mermaid source blocks,
and page numbering. This pass does **not** claim new page-by-page visual inspection.
Other historical PDFs/DOCX were preserved, not rebuilt or visually recertified.

## Research integrity and scope

The isolated full healing rerun produced final-test metrics (3780 x 32), incidents
(2171 x 71), and comparisons (80 x 7) with **no differing columns** versus committed
CSV tables (pandas equality). This is actual re-execution, not just a report assertion.
Directly recomputed means from committed metric rows:
FCFS 130.83778, SJF 94.65778, rule_wait 122.38825, q_recovery 122.56406,
rule_fairness 136.78000. This supports the documented negative/inconclusive findings:
healing improves FCFS waiting but loses to SJF, and Q is not established superior to
rule_wait. No seed/threshold/hyperparameter was tuned and no frozen output was replaced.

Drift CLI phase_transition was exercised with one repetition and two training episodes
per seed: seven workloads, 35 Q records. This proves workflow/output plumbing only,
not performance. Canonical drift studies were not rerun; old drift reports remain
historical evidence. New report generation no longer repeats their hard-coded
interpretations. Uncertainty SJF retains exact bursts (oracle, not equal information).
Nearest-rank P95 for 15 jobs equals max; recovery delay is verification closure, not
independently confirmed recovery. Sparse Q coverage, synthetic traces, no I/O/multicore,
and limited verifier power remain substantial academic limitations.

Full offline `study` was not rerun (3600 episodes/seed, multiple Q arms); reduced
integration tests cover its design/outputs. Complete Review 2 report rebuilding,
Windows/WSL setup, alternate Python/platform support, actual kernel integration,
and visual recertification of historical PDFs are not claimed. No frontend/API exists.
All initial Markdown local-link targets resolved; final checks repeat link validation.

## Historical audit (retained provenance; not current verification)

# Finalization audit and verification record

## Checkout examined before edits

- Date: 2026-10-23. Actual start: `fd8f103d94f267844b3ac78256ab98e987306df0`.
- Branch: `arena/49fb0ff9-os-project`; initial working tree clean.
- Origin: `https://github.com/lostpilot404/OS-Project.git`.
- Initial remote refs: main plus `arena/1ffa6b9e-os-project` and
  `arena/630f2a01-os-project`. Neither this new session branch nor the requested
  `arena/4a2c2384-os-project` existed remotely at audit time. The session branch
  is retained and will be pushed explicitly; no other branch is changed.
- Requested starting commit `ea30833` was not the checked-out HEAD. No reset,
  branch switch, forced push or historical artifact replacement was performed.

## Severity-ordered verified issues and disposition

1. **High: no safe short presentation command.** The CLI default launches the full
   healing study, including training and audits, and defaults to committed result
   directories. Added explicit `demo`, using the real engine/controller with default
   thresholds and two seeded rule fixtures, writing separately to gitignored
   `results/demo/`. Documented the dangerous no-command default rather than changing
   a working compatibility contract.
2. **High: environment instructions could silently use Ubuntu's wrong Python.**
   The lock records Python 3.11.2, while generic `python3 -m venv` is ambiguous on
   Ubuntu. Added WSL2 instructions with an exact private CPython installation,
   locked packages, separate report packages and local Linux checkout paths.
3. **High: historical report content is stale for this checkout.** The old plain
   report quotes 472 tests and an old branch/probe provenance. It is preserved,
   not overwritten. The new final report uses the actual checkout, current test
   run, canonical 784-row/98-incident probe and separate legacy 168-row/21-incident
   raw-proxy artifact. It distinguishes shared controller reward from the raw proxy.
4. **Medium: no lecturer script connecting output to real code.** Added a timed
   guide, actual Linux transcript, source map, truthful Q-learning explanation and
   fallback steps. The rule demo is not called learned recovery or a study estimate.
5. **Medium: output failures need actionable feedback in the new command.** Demo
   directory failures identify a writable `--results-dir` remedy. Nested/spaced
   paths, deterministic JSON, CLI help and execution outside repository CWD are tested.
6. **Scope limitation, not a code defect:** supplied `OS(2).pdf` was not present in
   the accessible checkout/attachments. Nine requested academic sections are retained
   using the existing report structure, with no claim to have inspected that PDF.
7. **Unverified environment:** no Windows WSL host was available. Debian Linux tests
   support portability, not a claim that WSL execution or Ubuntu installation was tested.

## Source and evidence findings

`main.py` and README explicitly identify self-healing as primary. `scheduler/runtime.py`
owns the live queue/event loop. `healing/` owns monitor, detector, diagnosis, 648-state
encoder, eight recovery actions, rules/Q policy, verifier, rollback and ledger.
`rl/q_learning.py` supplies the shared table learner; runtime and offline controllers in
`rl/` belong to distinct workflows. `workload/` generates seeded synthetic workloads;
`evaluation/` computes schedule metrics and paired inference. `experiments/` orchestrates
training/splits/audit, not a second scheduler implementation.

The primary study uses train seeds 7101–7105, validation 8201 and frozen test 23819;
`experiments/healing_config.py` and committed `results/healing/` are authoritative.
The non-healing runtime appendix instead uses 162 states, four actions and test 19301.
The offline selector/study uses whole-workload features; its outputs are not pooled here.

The existing report builder supports `--source` and `--output` and already renders only
plain text/Mermaid source with local fonts; no stylistic rewrite was needed. The final
report has its own names and Markdown source. Review 2 files are untouched.

No Windows-specific path dependency or case-sensitive import failure was found in the
exercised Linux workflow. The new demo was also launched by absolute `main.py` path from
a different directory and a directory containing spaces. Core and report packages
installed from the actual lock/manifests on Python 3.11.2; `pip check` passed.

## Executed checks

- `python --version`: Python 3.11.2, Debian 12 Linux x86-64.
- Created `.venv`; installed `requirements.lock` and `requirements-report.txt` successfully.
- `python -m pip check`: no broken requirements.
- `python main.py --help`: includes the implemented demo and preserved study commands.
- `python main.py demo`: two real fixtures and JSON output, no training or final evaluation.
- `python -m pytest -q tests/test_demo.py`: 5 passed in 0.69 seconds on the first focused run.
- `python -m pytest -q`: 500 passed, 1 warning in 38.37 seconds on the first full run.
  The warning is the existing empty-legend case in the legacy-probe plotting rejection test.
- Final verification and PDF inspection are recorded in `docs/FINAL_DELIVERY_CHECKS.md`.

No full expensive research study was rerun. Small experiment integrations executed by the
test suite do not constitute a rerun of the canonical study. Historical prior 495-pass
runs are not presented as this session's run. Frozen evidence, including seed 23819,
was neither tuned nor regenerated.
