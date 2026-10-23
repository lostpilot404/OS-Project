# Interface redesign — manual browser verification

Records the browser checks made **after** the interface rewrite. The earlier interface has
its own evidence in [`../web/`](../web/); nothing there was overwritten.

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-23 |
| Host | Linux container (Debian), CPython 3.11.2 |
| Server | `.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --no-browser` |
| Browser engine | HeadlessChrome 138.0.7204.0, Puppeteer (puppeteer-core 24) |
| Viewports | 1440 × 1000, 900 × 900, 420 × 860 |
| Network | loopback only; the page made **no** external request (checked in the script) |

The browser ran **headless**, against the real server and the real Python engine. It was not
a desktop session, so native window chrome, OS-level zoom and screen-reader output were not
exercised beyond the DOM-level checks below.

## What the scripted run did

`browser-checks.json` in this directory holds every check with its result. Summary:

| # | Check | Result |
|---|---|---|
| 1 | Deep link `#/recovery` on a cold session renders Recovery Analysis, marks the nav item, sets the title, and shows an empty state whose primary action opens the Studio | pass |
| 2 | Unknown fragment `#/does-not-exist` opens the Overview | pass |
| 3 | All seven routes render exactly one page with the expected `h1` and exactly one `aria-current="page"` nav item | pass (14 checks) |
| 4 | The rail contains seven destination links and no scenario, seed or example control | pass |
| 5 | Teaching run reports `Completed in N ms · run <id>`, sends exactly one POST | pass |
| 6 | Teaching arithmetic: FCFS 4.33, SJF 1.67, Round Robin 4.33, Priority 3.67 mean waiting units | pass (4 checks) |
| 7 | The policy-view tabs are a real control: the metric row, timeline and process table all follow the selected policy | pass |
| 8 | Clicking a dispatch block fills the inspector; the same slices are focusable and respond to <kbd>Enter</kbd> | pass |
| 9 | Clicking the “Waiting” column sorts ascending and publishes `aria-sort="ascending"` | pass |
| 10 | Back twice returns Studio → Overview; Forward replays Studio → About | pass |
| 11 | A reload on `#/research` keeps the page | pass |
| 12 | Seed 13 comparison: FCFS 255.80, SJF 172.13, Round Robin 405.53, Priority 254.60 on the same processes | pass |
| 13 | Running a different simulation clears the previous comparison instead of mixing runs | pass |
| 14 | Restoring a run from the Overview announces “It is not a new execution”, and the restored id differs | pass |
| 15 | The exported JSON is the run shown on screen (same `run_id`) | pass |
| 16 | A triple click on *Run simulation* produces one POST; the button, label and scenario cards are disabled while in flight | pass |
| 17 | A burst of 0 is rejected in the browser with a message and **zero** requests | pass |
| 18 | A corrected burst runs through `POST /api/simulations` (mean waiting 9.20 u) | pass |
| 19 | No page has a dead control: every `data-action` is handled, every link is a route or an API download, no empty buttons | pass |
| 20 | The skip link moves focus to `<main>` without changing the route | pass |
| 21 | No external requests, no console errors | pass |

Totals: **50 / 50 checks passed** (some rows above group several assertions).
Machine-readable record: [`browser-checks.json`](browser-checks.json).

## Screenshots

Full-page captures at 1440 px and viewport captures at 900 and 420 px. Each set was taken in
one session so the same run is visible on every page.

| Page | 1440 (viewport) | 1440 (full page) | 900 | 420 |
|---|---|---|---|---|
| Overview | `1440-01-overview.png` | `1440-01-overview-full.png` | `900-01-overview.png` | `420-01-overview.png` |
| Simulation Studio | `1440-02-studio.png` | `1440-02-studio-full.png` | `900-02-studio.png` | `420-02-studio.png` |
| Workload Explorer | `1440-03-workload.png` | `1440-03-workload-full.png` | `900-03-workload.png` | `420-03-workload.png` |
| Recovery Analysis | `1440-04-recovery.png` | `1440-04-recovery-full.png` | `900-04-recovery.png` | `420-04-recovery.png` |
| Policy Comparison | `1440-05-compare.png` | `1440-05-compare-full.png` | `900-05-compare.png` | `420-05-compare.png` |
| Research Results | `1440-06-research.png` | `1440-06-research-full.png` | `900-06-research.png` | `420-06-research.png` |
| About & Method | `1440-07-about.png` | `1440-07-about-full.png` | `900-07-about.png` | `420-07-about.png` |

Additional states:

| File | Shows |
|---|---|
| `1440-00-overview-no-run.png` | Overview before any run in the session |
| `1440-00-workload-empty.png` | Workload Explorer with no run: empty state with “Open Simulation Studio” |
| `1440-08-workload-teaching.png` | Teaching run, FCFS policy view |
| `1440-09-workload-teaching-sjf.png` | The same run switched to the SJF policy view |
| `1440-10-recovery-no-loop.png` | Recovery Analysis for a run that had no control loop: an honest “not part of this run” state |

## What was looked at

* **Layout.** One content column, 1180 px maximum, with the rail fixed at the left. At 900 px
  the Studio keeps its execute card beside the configuration; at 420 px everything stacks and
  the rail becomes a compact two-line top bar listing all seven destinations (no hidden
  horizontal scroll).
* **Spacing.** A single 4 px-based scale drives every gap and panel inset; panel padding is
  20–24 px, section rhythm is 28–32 px, and dense tables are the only place where text sits
  closer than 12 px.
* **Hierarchy.** Serif page titles (34 px), sans-serif section headings (15 px), 13–14 px
  body, and a muted metadata size used for units and provenance only. No page repeats its
  `h1` inside a panel, and the Workload page no longer repeats the scenario name as a title.
* **Colour.** One sea-blue accent for interactive elements; four policy colours used only in
  charts, legends and PID keys; accepted/inconclusive badges pair colour with a word and a
  dot, so nothing is signalled by colour alone. No gradients, glow, emoji or decorative art.
* **Empty / loading / error states.** Empty states name what is missing and link to the
  Studio; a failed run leaves the previous run visible and says so; a lost backend replaces
  the engine pill with *Backend unavailable* and offers *Retry connection*.
* **Example library.** The four scenarios, grouped as Teaching example / Self-healing
  recovery / Custom workload, sit in the Studio as a list of rows with a name, one sentence
  and a Select action. The rail has none of them.
* **Progressive disclosure.** The Studio shows the scenario list, then the configuration,
  then execution; comparison results appear only after a comparison is requested; the
  Research page keeps the interpretation limits beside the numbers and drops the duplicated
  arm list that repeated the chart.

## Fixed during this pass

* The custom-workload editor table had lost its styling in the rewrite; `.process-editor`
  rules were restored (hover, focus, numerals, remove button).
* The Studio ran its scenario/configuration column beside the execute card, which pushed the
  scenario list into a narrow strip. The list is now full width above the two-column area.
* The teaching policy tabs sat inside the timeline panel and, on the Workload page, changed
  the chart but not the metric row or process table. They are now one control above the
  metric row and drive all three.
* `noRunPage()` was called with the wrong argument order on three pages, so their empty states
  printed the wrong heading and body.
* The skip link (`#view`) matched the router's “unknown hash” rule and re-rendered the page;
  in-page fragments are now ignored by the router and the link focuses `<main>`.
* The favicon placeholder was a data-URL SVG containing `http://`, which the
  self-containment test (correctly) rejects; it is now an empty data URL.
* `.detail-cols` used `1fr auto` columns with right-aligned values, which squeezed long
  lists into one character per line; the layout is now `max-content minmax(0, 1fr)` with
  left-aligned, wrapping values.
* The Recovery page repeated the whole dispatch timeline that the Workload Explorer already
  shows; it now links to it. The Overview no longer carries both a current-run card and a
  duplicate “Where to go” list of every destination.

## A stale shell in the preview browser

Opening the preview in the user's browser once showed the shell with no stylesheet and no
behaviour: the navigation as plain links, "Connecting…" never resolving, the footer text
stacked underneath. The server log for that visit contained a single `GET /` — the browser
never asked for `/static/styles.css` or `/static/app.js`, so it was rendering a copy of the
page it had cached earlier rather than anything the server sent.

The interface was not at fault, but the setup allowed it to happen again on every restart.
Two changes remove that class of failure:

* `GET /` is now served with `Cache-Control: no-store, must-revalidate`, and the shell's
  asset URLs carry a content hash (`/static/app.js?v=8e393c0d27`). Static files are served
  with `no-cache`, so they are revalidated on every load and an edited script can never be
  paired with an older page. Three tests cover the headers, the stamps and the revalidation.
* If `app.js` does not run within 2.5 s, the page replaces the empty region with a visible
  "The console could not start" note that names the fix, instead of leaving the reader on
  "Connecting…".

Verified after the change: a cold load requested `/`, `/static/styles.css`, `/static/app.js`,
`/api/health`, `/api/examples` and `/api/runs`, all `200`; the rail computed `position: fixed`,
the Overview rendered, and the engine pill read *Engine ready · 3.11.2*
(`1440-11-cold-load.png`). The 50 scripted checks and the full suite were re-run afterwards.

If you still see the unstyled version, it is a cached copy: reload with Ctrl+Shift+R (or
Cmd+Shift+R), or open the preview in a new tab.

## Known limitations of this evidence

* Headless only: no native window chrome, no OS font rendering differences, no screen reader.
* Only Chromium 138 was available; Firefox and Safari were not exercised.
* Screenshots are static captures at three widths; scrolling behaviour, sticky positioning
  and hover states were checked in the DOM but cannot be fully judged from stills.
* Windows PowerShell remains unverified here (see `docs/LIVE_DASHBOARD.md` §7).
