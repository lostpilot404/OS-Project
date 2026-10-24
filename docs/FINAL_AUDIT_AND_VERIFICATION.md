# Final audit and verification

This report states what was inspected, what was found, what was changed, and what remains
unverified. It covers the checkout at `arena/0c334afe-os-project` (branched from `8c5aaba`)
with the changes in §11 applied in the working tree. Nothing was committed or pushed.

It separates three kinds of statement on purpose: things **executed here**, things **read in
the sources** (static audit), and things that were **not done** (§1, §12). A static finding is
not a substitute for executing on the target machine, and a headless browser is not a human
session.

## 1. Status at a glance

| Area | State | Where it was established |
|---|---|---|
| Scheduling rules, metrics, controller, Q-learning wiring | Inspected, hand-computed, covered by tests | This sandbox (Linux), CPython 3.11.2 |
| HTTP API: validation, error shapes, history, concurrency cap | `curl` against the running service + FastAPI `TestClient` | This sandbox |
| Seven console pages at 1440 px | Scripted headless Chromium: 83 named checks, 22 screenshots | This sandbox, `docs/manual_evidence/final-desktop/` |
| Console re-render after the last code changes | Scripted smoke check, 9/9 checks | This sandbox |
| Automated suite | `674 passed` under a UTF-8 locale **and** under a forced non-UTF-8 locale | This sandbox |
| Windows portability (paths, temp files, encodings, console, subprocess) | Static AST audit of every shipped module + execution under an ASCII locale | This sandbox |
| **Human-operated desktop browser session** | **Not performed — blocked** (§5) | — |
| **Native Windows execution** (PowerShell, `.venv\Scripts\python.exe`) | **Not performed — blocked** (§4.4) | — |
| **WSL check** | **Not available in the sandbox** (§4.4) | — |

No scheduling rule, threshold, metric definition, experimental result, recovery decision or
pre-existing test assertion was changed. The study and drift artifacts were read, never
recomputed.

## 2. What this environment is

The work ran in a Linux container, not on the target Windows machine:

```text
$ uname -srm          → Linux 6.1.158 x86_64
$ cat /etc/os-release → Debian GNU/Linux 12 (bookworm)   (host: e2b.local)
$ .venv/bin/python --version → Python 3.11.2
$ .venv/bin/python -m pip check → No broken requirements found.
$ git status --short --branch → ## arena/0c334afe-os-project
$ git log -1 --oneline → 8c5aaba Merge pull request #14 from lostpilot404/arena/6f055996-os-project
```

The documented Windows environment uses CPython 3.11.9; the sandbox interpreter is 3.11.2, so
every number below is **this** interpreter's, not the Windows `.venv`'s.

Absent from the container: any display (`DISPLAY` and `WAYLAND_DISPLAY` unset, no
`/tmp/.X11-unix`), any packaged virtual framebuffer (`apt-get install xvfb` →
“Unable to locate package xvfb”; the same for `apt-get download xvfb`), any installed Chrome,
Edge or Firefox (only an extracted `chromium` build under `/tmp`, used headless), any
Playwright browser cache, and any Windows executable (`powershell.exe`, `pwsh`, `wsl.exe` and
`cmd.exe` are all missing; there is no `C:\Users\Srikiran\OS-Project`). The Windows machine
described in the task is therefore **not reachable from this session**, and no command that
ran here can be reported as a Windows result.

## 3. Evidence categories

| Kind | What it proves | What it does not prove |
|---|---|---|
| **Independent arithmetic** | Waits, turnaround, response, utilisation and residuals recomputed by hand from arrival/burst/quantum match the API | Nothing about presentation |
| **Automated test** (674) | Behaviour the suite pins, including 14 new regression tests | Not that the interface looks right |
| **API probe** (`curl`, `TestClient`) | Status codes, error bodies, headers, payload contents | Not that the browser renders them |
| **Scripted browser run** (HeadlessChrome 153, 1440 × 900) | The DOM the application actually produces, plus screenshots | Not a human session, not Windows, not another engine |
| **Static audit** (AST over shipped modules) | Absence of POSIX-only calls; explicit encodings; path handling | Not execution |
| **Locale-simulated execution** (`LC_ALL=C PYTHONUTF8=0`) | Behaviour when the default text encoding is not UTF-8 — the Windows failure mode | Still Linux, still not cp1252 exactly |

## 4. Windows: what was audited, what was fixed, what remains

### 4.1 Static audit of the shipped modules

Every module under `dashboard/`, `healing/`, `experiments/`, `evaluation/`, `rl/`,
`scheduler/`, `visualization/`, `tools/`, plus `main.py`, `config.py` and `errors.py` was
parsed and searched. Findings:

| Concern | Finding |
|---|---|
| POSIX-only APIs | None. `fcntl`, `os.fork`, `pwd`, `grp`, `resource`, `os.kill`, `signal` and `start_new_session` do not appear in shipped code |
| Hard-coded POSIX paths | None in shipped code. `/tmp` appears only in the ad-hoc scripts under `docs/manual_evidence/` (not part of the delivered workflow) and in usage docstrings |
| Temporary directories | `dashboard/api.py` uses `tempfile.mkdtemp()` under the OS temp directory, writes `.result.json.tmp` and `os.replace()`s it into place, and removes directories with `shutil.rmtree(..., ignore_errors=True)` |
| Atomic builds | `dashboard/build.py` already uses `NamedTemporaryFile(delete=False)` followed by `os.replace()` — the pattern that works when the destination exists on Windows |
| Console output | No non-ASCII text is printed by `main.py`, `tools/serve_dashboard.py`, `tools/build_teacher_dashboard.py`, `tools/build_plain_report.py` or the demo path; the teacher page is written with an explicit encoding |
| Demo transcript | `docs/examples/demo_linux_output.txt` is ASCII-only (0 non-ASCII lines); its one path-bearing line is normalised by the verifier, so the comparison is valid on Windows |
| Subprocess | Only `tools/build_review2_report.py` shells out (`sys.executable`, `git`), and it already catches `OSError`/`SubprocessError`, which covers a missing `git.exe` |
| Browser opening | `tools/serve_dashboard.py` opens a window from a `threading.Timer` after one second and only when `--no-browser` is absent; the server itself binds `127.0.0.1` by default |
| Static assets | Served from `dashboard/web/` through `StaticFiles` with no path arithmetic, a SHA-1 content stamp and `Cache-Control: no-cache` |
| Text I/O encoding | **This was the one real class of defect** — see §4.2 |

### 4.2 Defects in the Windows portability class, found and fixed

Python resolves `Path.read_text()`, `Path.write_text()` and text-mode `open()` with the
*locale* encoding when no encoding is named. On an English Windows install that is cp1252,
which cannot even decode `−` (U+2212), `≥` (U+2265) or `²` (U+00B2) — all of which occur in
generated reports and committed documents.

The failure was reproduced by running the suite with a non-UTF-8 default encoding:

```text
$ LC_ALL=C LANG=C PYTHONUTF8=0 .venv/bin/python -m pytest -q
14 failed, 657 passed, 1 warning in 40.44s
```

Fourteen tests failed, for four distinct reasons: report generation crashed writing `−`
(`experiments/drift_uncertainty_experiment.py`), a committed summary was read without an
encoding (`healing/canonical_numbers.py`, `tools/review2_data.py`,
`tools/verify_final_delivery.py`), tests read UTF-8 documents and `app.js` without an encoding,
and one test wrote into `/tmp`, which does not exist on Windows.

Fixes: 42 text-mode call sites now name their encoding — 12 in 5 shipped modules, 30 in 10 test
modules — and `tests/test_probe_schema.py` uses pytest's `tmp_path` instead of `/tmp`. One unrelated
latent defect surfaced while doing this: the module docstring of
`tools/build_teacher_dashboard.py` contained Windows paths with invalid escape sequences
(`\.venv`, `\Scripts`), which Python 3.12 reports as `SyntaxWarning` and later versions will
reject; the docstring is now raw text. Two regression tests guard the class
(`tests/test_encoding_discipline.py`): one parses every shipped module and fails on any
text-mode call without an explicit encoding; the other loads the committed summary in a
subprocess started with `-X utf8=0`, `LC_ALL=C`, `PYTHONUTF8=0`.

After the fixes, the same command reports `674 passed, 1 warning in 38.73s` — identical to the
UTF-8 run (`674 passed, 1 warning in 40.22s`). Further execution under the same ASCII locale:

| Check | Observed |
|---|---|
| `main.py demo --results-dir /tmp/os-demo-ascii` | exit 0; output identical to `docs/examples/demo_linux_output.txt` after replacing the printed path (`diff` empty) |
| `tools/verify_final_delivery.py` | exit 0; “Verified 18 pages…”, “Verified canonical/legacy probe counts, frozen means, exact demo transcript and reproducible PDF bytes.” |
| `tools/build_teacher_dashboard.py --input …/demo.json` | 213.0 KB, sha256 `230a34caf560495e` — unchanged by the docstring edit |
| Dashboard server under the ASCII locale | `/api/health` 200; two `POST /api/simulations` 201; `/export` 200 with `Content-Disposition: attachment; filename="os-dashboard-recovery-accepted-0e248276.json"`; `/` 200; no traceback in the server log |

**What this does not establish:** that PowerShell parses the documented commands, that
`uvicorn` binds identically on Windows, that Windows Defender or file locking behaves, or that
a real Chrome/Edge window renders the pages. Those are the §4.4 items.

### 4.3 The shortest Windows check, with expected values

Run from the repository root in PowerShell; each step is independent, so a failure localises
itself. This is the procedure the audit could not execute.

```powershell
git status --short --branch          # ## arena/0c334afe-os-project, 30 modified, 4 untracked
git log -1 --oneline                 # 8c5aaba Merge pull request #14 ...
.\.venv\Scripts\python.exe --version # 3.11.x
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe tools\serve_dashboard.py
```

Then open <http://127.0.0.1:8000/> in a desktop browser and check, in this order:

| Step | Expected |
|---|---|
| DevTools → Console | Empty (no exceptions, no failed requests) |
| DevTools → Network | `index.html` 200 `no-store`; `styles.css?v=…` and `app.js?v=…` 200; `/api/health` 200 “ready” |
| Studio → teaching example → Run | 4.33 / 1.67 / 3.67 / 4.33 mean waiting for FCFS / SJF / Priority / Round Robin; note reads “mean waiting 1.67 u, FCFS 4.33 u. Showing SJF.” after choosing SJF in *Policy view* |
| Recovery → accepted example | “FCFS → SJF”, close `t=319`, residual `+0.4375`, verdict `accepted`, both judgment stages green |
| Recovery → rollback example | “rolled back to FCFS”, verdict `inconclusive`, both judgment stages amber, never described as a success |
| Policy Comparison | FCFS 150.60, SJF 85.13, Round Robin 211.87 (83 switches, 78.77 %), Priority 156.13 |
| Export JSON | `os-dashboard-recovery-accepted-<8 hex>.json`, identical to the stored result |
| `Get-ChildItem $env:TEMP -Filter os-dashboard-*` | One directory per live run; all removed after Ctrl+C |
| Ctrl+C, then `Get-NetTCPConnection -LocalPort 8000` | Server stopped, port released, no repository file modified (`git status` unchanged) |

Also worth one run each: `.\.venv\Scripts\python.exe -m pytest -q` (expect `674 passed`),
`python tools\verify_final_delivery.py`, and
`python main.py demo --results-dir "$env:TEMP\os-demo"` — compare its output with
`docs/examples/demo_linux_output.txt`; every line matches except the last, which prints the
output path you chose.

### 4.4 WSL

`wsl.exe` does not exist in this container and the Windows machine is not reachable, so the
WSL feature state cannot be inspected from here. Nothing was installed to fake it. On the
target machine the check is one command — `wsl.exe -l -v` — and the project's own record is
that **actual execution inside WSL remains unverified** (`README.md` §“Quick lecturer demo”),
which this audit does not change. Windows stays the primary target; WSL is a secondary
convenience.

## 5. Interactive browser status

**Not performed. The blocker is physical, not a matter of effort:** this container has no
display server and no way to obtain one — `Xvfb` and the other virtual-framebuffer packages
are not in the image's package index, `DISPLAY` is unset, there is no X socket, and there is
no Chrome, Edge, Firefox or Playwright browser installed, only an extracted headless Chromium
build. A browser started here can never show a window to a person, so a “visible-browser
pass” cannot be produced honestly. Anything that looked like one would be a screenshot of a
headless process, which is exactly what the evidence folder already documents as scripted.

What stands in its place, and what it covers:

- `docs/manual_evidence/final-desktop/` — 22 screenshots at 1440 × 900 and
  `browser-checks.json` with **83 named checks in 4 sessions, all passing**, captured with
  HeadlessChrome 153.0.8010.0 driven by Puppeteer. It includes the seven pages with
  navigation, refresh, Back/Forward and empty states; the teaching and recovery flows; the
  comparison states including an injected failure and retry; the stale-run refusal; the
  backend-down banner; the export; and the research/limitations text.
- A post-change smoke check run after the last code edits, writing no evidence files:
  **9/9** — accepted recovery (`data-tone` tail `ok, ok`), inconclusive rollback
  (`warn, warn`, badge `rgb(147, 100, 28)`), Studio policy view “Showing SJF” with
  “mean waiting 1.67 u, FCFS 4.33 u”, Workload Explorer under SJF, and no page errors.
  The changes made after the screenshots were an indentation fix, an equivalent derivation of
  the constant 648, encoding arguments and a docstring, so the captures still describe the
  current behaviour; the API regression tests pin the same payloads.

The shortest human click-through, for the person at the keyboard (about ten minutes, using the
values above): load the page and watch the console and Network panel; visit each of the seven
pages, refresh one, then use Back and Forward; run the teaching example and open a Gantt block
both by mouse and with Tab/Enter; run the accepted example and then the rollback example and
compare the two lifecycle panels side by side; run the comparison and re-run it; stop the
server with Ctrl+C, press *Retry* in the page and watch the banner, then restart and retry;
export a run and open the JSON. If every row of §4.3 matches, the interactive pass is done.

## 6. Defects found and fixed

| # | Defect (evidence) | Fix | Where |
|---|---|---|---|
| 1 | A healing-enabled run that declared **no incident** crashed the Recovery page with `ReferenceError: schedule is not defined` (browser console; empty `stages`) | The page passed an undefined `schedule` to the Gantt builder; it now passes the run's own schedule, and the empty lifecycle renders the “no incident” text | `dashboard/web/app.js` |
| 2 | An inconclusive **rollback** was painted with the success tone on its “Final decision” stage; Verification was amber while the decision was green | Stages carry a semantic `tone_key` from the controller's verdict; the interface colours from it | `dashboard/charts.py`, `dashboard/service.py`, `dashboard/render.py`, `dashboard/web/app.js` |
| 3 | Teaching payload carried a **zero FCFS baseline** and no turnaround/response totals | The baseline is the executed FCFS run; totals and switches-per-process are filled like every other scenario | `dashboard/service.py` |
| 4 | A custom workload with `healing: true` **silently ignored** a requested non-FCFS policy while the Studio still showed it | The API rejects the mismatch (422, `field: "policy"`); the Studio disables the selector and explains why | `dashboard/service.py`, `dashboard/web/app.js` |
| 5 | The Studio's **Policy view** selector did nothing unless the Workload page was open | One source of truth: the tab, the selector and the next request share `state.teachingPolicy` | `dashboard/web/app.js` |
| 6 | A run that left the bounded history answered `type: "not_found"` while the contract promises **`unknown_run`** | The 404 handler keeps the type declared by the route | `dashboard/api.py` |
| 7 | A malformed JSON body reported `1: JSON decode error` — a character offset presented as a field name | The error names `body` | `dashboard/api.py` |
| 8 | An unknown scenario message arrived wrapped in literal double quotes (`str(KeyError(...))` is a repr) | The declared text is used | `dashboard/service.py` |
| 9 | The recovery page hard-coded the tabular state space as **`/648`** | The payload carries `state_space_size`, now derived with `math.prod` from the encoder's radices | `dashboard/service.py`, `dashboard/web/app.js` |
| 10 | The Research page stated the study's means but nothing about what the **verifier** did | A panel reads the committed summary: 973 interventions, 65 accepted (all helpful), 8.0 % certifiable, 34 rejected (76.5 % harmful), 0.62 incidents per control trace, declared false-accept 4.6 % | `dashboard/service.py`, `dashboard/web/app.js` |
| 11 | Four statements in two documents described the arrival-storm paired difference as `[0.00, 0.56] (includes 0)`; the committed interval is `[0.0036048, 0.5638381]` | Corrected with the effect size named; the conclusion is unchanged | `docs/LEAKAGE_FIX_AND_DRIFT_REPORT.md`, `docs/FINAL_SUBMISSION_REPORT.md` |
| 12 | Smaller interface items: Studio lead claimed the run opens elsewhere; empty state said “in this session yet”; About heading did not match the navigation label; a dead branch in `setRunCardBusy`; an inline style in the inspector; process chips after a mid-trace recovery used the final policy; the “no incident” panel borrowed the success class; a `held` verdict had no styling | Each corrected in place | `dashboard/web/app.js`, `dashboard/web/styles.css` |
| 13 | **Windows/locale:** 5 shipped modules and 10 test modules read or wrote UTF-8 text without naming an encoding; 14 tests failed under a non-UTF-8 locale, and the drift report itself could not be written under cp1252 | Explicit `encoding="utf-8"` at 42 call sites; regression tests for the class (static audit + subprocess under `-X utf8=0`) | `experiments/*.py`, `healing/canonical_numbers.py`, `tools/*.py`, `tests/*` |
| 14 | `tools/build_teacher_dashboard.py` docstring contained invalid escape sequences in its PowerShell example (`\.venv`, `\Scripts`), a `SyntaxWarning` on 3.12 and a future error | The docstring is a raw string; the usage block is one line | `tools/build_teacher_dashboard.py` |
| 15 | `tests/test_probe_schema.py` wrote figures into `/tmp`, which does not exist on Windows | Uses pytest's `tmp_path` | `tests/test_probe_schema.py` |

Examined and deliberately left alone: the Overview headline “A self-healing CPU scheduler,
observed end to end” instead of the word “Overview”. The document title and the active
navigation item both read *Overview*, and the headline states the project rather than
repeating the menu, so this is the landing-page design.

An earlier suspicion that `experiments/healing_experiment.py` and `experiments/study.py`
omitted encodings did not survive inspection: those writes already pass `encoding="utf-8"`
(only two JSON writers in `study.py` did not, and they are now explicit). The audit is
reported as it came out, not as first assumed.

## 7. Verification record

| Area | What was done | Result |
|---|---|---|
| A. Startup, assets, console | Fresh load at 1440 px; asset statuses, console and failed-request log | 200s for `styles.css` and `app.js`, no page errors, no horizontal overflow |
| B. Seven routes | Each route opened, refreshed, then reached again with Back/Forward | All seven keep their route and heading; no page errors |
| C. Teaching example | Run, tab switch, Gantt block by mouse and by keyboard | 4.33 / 1.67 / 3.67 / 4.33 as documented; the inspector reads the recorded interval and says when a waiting increment was not recorded |
| D. Recovery accepted (seed 13) | Run, read the lifecycle and the verification band | residual +0.4375, accepted, both judgment stages green, SJF kept |
| E. Rollback (seed 3) | Run, read the lifecycle and the wording | residual +0.2059, inconclusive, both judgment stages amber, “rolled back to FCFS” |
| F. Policy comparison | In-flight state, duplicate clicks, stale reset, injected 500, dismiss, retry | One request per click; button disabled while running; failure explains itself and clears the stale table; retry works |
| G. Custom workload | Four-process workload typed and hand-checked; burst 0; healing on | 4.25 u mean, rows as computed; burst 0 blocked before any request; healing disables the policy selector |
| H. History and reload | Footer, Overview fact, `/api/runs`; reload; restore | Counts agree; a reload keeps history and closes the open run; a restored run is labelled as restored |
| I. Export and research provenance | Export vs stored result; Research page | Identical JSON; source, split and seed named; aggregates explicitly not this session's data |
| J. Error handling | Backend stopped mid-session; banner; dismiss; restart; retry; a run that ages out | “The simulation did not run…”, pill *Backend unavailable*, banner dismissible, retry restores *Engine ready*; a stale run refused with “that run is not in the server's history” and no invented numbers |
| K. Post-change smoke | Accepted and rollback lifecycle tones, teaching policy view, Workload Explorer, page errors | 9/9 checks, `ok, ok` / `warn, warn`, badge `rgb(147, 100, 28)`, no errors |
| L. Locale independence | Full suite under `LC_ALL=C PYTHONUTF8=0`; demo, verifier, teacher build and server under the same locale | 674 passed; transcript identical; verifier exit 0; 213.0 KB / `230a34caf560495e`; API 201/200 with the documented filename |

Screenshots 01–22 are the post-fix captures; 04, 05, 06 and 13 were re-captured after the
lifecycle changes.

## 8. Commands and observed output

```text
$ .venv/bin/python -m pytest -q
674 passed, 1 warning in 40.22s

$ LC_ALL=C LANG=C PYTHONUTF8=0 .venv/bin/python -m pytest -q
674 passed, 1 warning in 38.73s

$ .venv/bin/python -m pip check
No broken requirements found.

$ .venv/bin/python tools/verify_final_delivery.py
Verified 18 pages: text-only black/white, page numbers, seven Mermaid sources.
Verified canonical/legacy probe counts, frozen means, exact demo transcript and reproducible PDF bytes.
Visual inspection and WSL execution are separate from these automated checks.

$ .venv/bin/python main.py demo --results-dir /tmp/os-demo-ascii
Trigger d6 t=243: completed-window wait=105.167 > threshold=75.703
Verdict=accepted; reason=verified_phase_residual; residual=0.437454899124583; reward=0.07078823245791635
SUMMARY mean wait=220.533; FCFS same input=255.800; difference=-35.267
(exit 0; transcript identical to docs/examples/demo_linux_output.txt)

$ .venv/bin/python tools/build_teacher_dashboard.py --input /tmp/os-demo-ascii/demo.json
size              : 213.0 KB (self-contained HTML, inline SVG)
sha256            : 230a34caf560495e
```

API probes (current tree, service on `:8000`):

| Request | Answer |
|---|---|
| `POST {"scenario":"nope"}` | 422 `invalid_request`, field `scenario`, message without the `KeyError` quotes |
| `POST teaching` with `policy_view: "XYZ"` | 422, field `policy_view`, lists the four policies |
| `POST custom`, burst 0 with a priority | 422, field `processes`, “burst_time must be > 0, got 0” |
| `POST custom` with `healing: true` and `policy: "SJF"` | 422, field `policy`, explains that the loop starts from FCFS |
| `POST /api/comparisons {}` | 422, “send either 'run_id' or a non-empty 'processes' list” |
| `GET /api/simulations/<unknown>` and `/export` | 404 `unknown_run` |
| `POST` with a truncated body | 422, field `body` |
| `GET /api/nope` | 404 `not_found` |

The 429 path is covered by `tests/test_dashboard_api.py` (`_RunGuard` with capacity 1) and was
not provoked over HTTP, because that needs four simultaneous multi-second runs.

## 9. Evidence paths

- `docs/manual_evidence/final-desktop/` — 22 screenshots at 1440 px, `browser-checks.json`
  (4 sessions, 83 checks, all passing) and a README describing each capture and its limits.
- `docs/manual_evidence/README.md` — index of every evidence folder, old and new.
- `docs/LIVE_DASHBOARD.md` — API contract, scenario behaviour, Windows and Linux start-up,
  troubleshooting, and the browser-verification status.
- `docs/TEACHER_LIVE_DEMO.md`, `docs/DEMO_SCRIPT_FOR_LECTURER.md`, `docs/TEACHER_QA.md` —
  the classroom path, checked against the current payloads.
- Older evidence (`web/`, `web-v2/`, `web-v3/`, `web-v4/`) is untouched.

## 10. Limitations and open items

- **Windows was not executed.** §4 gives the static audit, the locale-simulated execution and
  the exact procedure with expected values; it is not a substitute for §4.3 on the target
  machine, and this report does not claim otherwise.
- **No human browser session.** §5 explains the blocker and what the scripted evidence covers.
- One browser engine (Chromium), one viewport family (1440 × 900 for the final pass). The
  earlier 900/420 px passes are in `web-v2/`/`web-v4/`.
- The drift and held-out study were **not** re-run; their numbers were verified against the
  committed CSVs and the report text.
- Q-learning remains undertrained on the evaluated design (a few dozen of 648 contexts
  visited); no superiority claim is made anywhere, and the Research page says so.
- The verifier is a safety filter, not a benefit estimator: it accepted 65 of 973
  interventions, and 91–92 % of helpful rule interventions end inconclusive. That is a design
  limitation, documented rather than hidden.
- The scripted browser flows are not part of the repository; re-running the evidence means
  re-writing them. The README gives the entry points, and every assertion is listed by name in
  `browser-checks.json`.

## 11. Git state

- Branch `arena/0c334afe-os-project`, based on `8c5aaba`; **nothing committed or pushed**.
- Modified (30 files, +373 / −125): `dashboard/{api,charts,render,service}.py`,
  `dashboard/web/{app.js,styles.css}`, `healing/{reward,canonical_numbers}.py`,
  `experiments/{study,drift_uncertainty_experiment}.py`,
  `tools/{build_teacher_dashboard,review2_data,verify_final_delivery}.py`,
  11 modules under `tests/`, `README.md`,
  `docs/{FINAL_SUBMISSION_REPORT,LEAKAGE_FIX_AND_DRIFT_REPORT,LIVE_DASHBOARD,TEACHER_LIVE_DEMO}.md`,
  `docs/manual_evidence/README.md`.
- Added (untracked): `docs/FINAL_AUDIT_AND_VERIFICATION.md`, `docs/manual_evidence/final-desktop/`
  (22 PNGs, `browser-checks.json`, `README.md`), `tests/test_dashboard_audit_fixes.py`,
  `tests/test_encoding_discipline.py`.

## 12. Definition of done

- [x] The Recovery page renders every healing outcome the engine can produce, including “no
      incident” and a rollback, with tones that match the verdict.
- [x] The Studio, Overview and Workload pages describe the same trace and the same policy.
- [x] A custom workload cannot ask for a policy the self-healing loop would ignore.
- [x] API error shapes match the documented contract, including `unknown_run` and `body`.
- [x] The interface shows where every research number comes from and adds what the verifier
      measured, without claiming Q-learning superiority.
- [x] Documentation matches the current tree; one imprecise drift interval was corrected.
- [x] The demo and the static teacher page reproduce exactly.
- [x] 674 tests pass under a UTF-8 locale **and** under a forced non-UTF-8 locale, including
      14 new regression tests.
- [x] Scripted browser evidence with 83 checks and 22 screenshots is recorded, and a
      post-change smoke check passes 9/9.
- [x] Shipped modules contain no POSIX-only calls, no hard-coded temporary paths and no
      text-mode file access without an explicit encoding.
- [ ] A human click-through of the console (§5) — blocked: no display and no browser window
      can exist in this environment.
- [ ] Native Windows execution of §4.3 — blocked: no Windows machine, PowerShell or WSL is
      reachable from this session.
