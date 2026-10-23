# Browser evidence — v4 (defect fixes)

Captured against the real dashboard (`tools/serve_dashboard.py`, uvicorn on `0.0.0.0:8000`)
with a real headless Chromium — not rendered or mocked SVGs. Every check below ran in that
same session; the raw log is `browser-checks.json`.

- Browser: HeadlessChrome/138.0.7204.0 via puppeteer-core 24 / @sparticuz chromium 138
- Viewports: 1440x1000, 420x900
- Started: 2026-10-23T18:00:50.767Z  ·  Finished: 2026-10-23T18:01:04.614Z
- Result: **36/36 checks passed**, 9 screenshots

## What this run deliberately exercised

1. A comparison request was delayed by 1.5 s to observe the disabled "Running four policies…"
   state, then allowed to finish; two extra clicks while it was in flight produced no extra POSTs.
2. One `POST /api/comparisons` was answered with an injected HTTP 500 (that single failure is
   the only non-2xx response in the log) to prove the button recovers and the banner explains it.
3. One `POST /api/simulations` was answered with an injected HTTP 500 to prove a failed run
   changes no counter.
4. Three sequential runs (`teaching`, `recovery-rollback`, `recovery-accepted`) to compare the
   footer, the Overview fact, `/api/runs` and `/api/health` after each one.

## Screenshots

- `01-comparison-complete-1440.png` — Policy Comparison after a completed run: button ready again (209,400 bytes)
- `02-comparison-error-1440.png` — After a failed comparison: button restored, banner shown (124,154 bytes)
- `03-overview-counts-1440.png` — Overview after several runs: footer and sidebar agree (178,051 bytes)
- `04-workload-1440.png` — Workload Explorer at 1440 px (unchanged desktop layout) (161,518 bytes)
- `05-studio-1440.png` — Simulation Studio at 1440 px (212,813 bytes)
- `06-workload-420.png` — Workload Explorer at 420 px: fitted timeline, stacked process rows (162,956 bytes)
- `07-workload-420-actual-size.png` — 420 px Workload with the timeline at actual size (161,986 bytes)
- `08-studio-420.png` — Simulation Studio at 420 px: parameter summary reflows (186,456 bytes)
- `09-compare-420.png` — Policy Comparison at 420 px: stacked policy metrics (185,320 bytes)

## Checks

- PASS — stylesheet and script load with 200
- PASS — button starts ready with the empty label
- PASS — button says "Running four policies…" and is disabled while the request is open
- PASS — button returns to a ready state after success
- PASS — repeated clicks while running did not fire duplicate requests
- PASS — captured 01-comparison-complete-1440.png
- PASS — the ready button really re-runs the comparison
- PASS — an API failure restores the button and explains itself
- PASS — captured 02-comparison-error-1440.png
- PASS — the banner Dismiss control actually removes the banner
- PASS — a retry after the failure produces the comparison table
- PASS — after every run the footer, the sidebar and /api/runs agree
- PASS — a failed run changes no counter, and the overview still agrees
- PASS — the failed-run banner clears on request
- PASS — captured 03-overview-counts-1440.png
- PASS — counters survive a full page reload
- PASS — the run list reports how much of the history it is showing
- PASS — 1440 px Workload: ordinary table, no chart toolbar, chart at full width
- PASS — captured 04-workload-1440.png
- PASS — 1440 px Studio: four examples in three groups, one-line summary rows, ready Run button
- PASS — captured 05-studio-1440.png
- PASS — 1440 px Comparison: normal table, four rows, ready button
- PASS — navigation reaches Recovery Analysis with a real verdict and lifecycle
- PASS — JSON export returns the run that is on screen
- PASS — 420 px Workload: no page scroll, all nine columns shown as labelled rows
- PASS — 420 px Workload: the whole timeline fits with legible labels
- PASS — captured 06-workload-420.png
- PASS — 420 px: "Actual size" brings back the scrollable full-detail timeline without breaking the page
- PASS — captured 07-workload-420-actual-size.png
- PASS — 420 px Studio: summary values sit on their own left-aligned line inside the card
- PASS — captured 08-studio-420.png
- PASS — 420 px Comparison: bar charts keep legible labels by scrolling rather than shrinking
- PASS — 420 px Comparison: four policies stacked with all seven columns, ready button
- PASS — captured 09-compare-420.png
- PASS — the only failed request is the deliberately injected one
- PASS — no console errors beyond the injected 500

The earlier folders (`web/`, `web-v2/`, `web-v3/`) are left untouched; `web-v3/` is the
evidence that identified these defects, and this folder shows them fixed.
