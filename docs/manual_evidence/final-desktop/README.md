# Browser evidence — final desktop pass

Captured against the dashboard in this working tree (`tools/serve_dashboard.py`, uvicorn on
`0.0.0.0:8000`) with a scripted headless Chromium. Every file here was produced by that run;
the raw check log is `browser-checks.json`.

- Browser: HeadlessChrome/153.0.8010.0 (puppeteer-core, `@sparticuz/chromium` 153)
- Host: Debian GNU/Linux 12 (bookworm), CPython 3.11.2
- Viewport: 1440 × 900, full-page screenshots
- Sessions: 4 (startup/routes/teaching/recovery/comparison · custom/history/export/restore/research · backend loss and retry · a run that ages out of the history)
- Result: **83/83 checks passed**, 22 screenshots

## What this is, and what it is not

This is *scripted* browser evidence: Puppeteer drove the real application in a real Chromium
process, but nobody sat in front of it. It is supporting evidence, not a substitute for a
person clicking through the console, and it says nothing about Windows rendering, other
browser engines or physical devices. No interactive desktop browser was available in this
environment, so a human pass over the same pages is still the step that remains open.

## What each session exercised

1. **Startup, navigation, teaching, recovery, comparison.** Clean console and network on
   load; all seven destinations, each refreshed and reached again with back/forward;
   the teaching example (FCFS 0/5/8 waits, SJF 1.67 u) with a dispatch block inspected by
   mouse and by keyboard; the accepted (seed 13) and inconclusive-rollback (seed 3) recovery
   pages, including the tone of every lifecycle stage; a comparison observed mid-flight
   (button disabled, two extra clicks produced no extra request), completed, failed once on
   an injected HTTP 500, dismissed, retried, and cleared when a new run replaced the workload.
2. **Custom workloads and the session record.** A four-process workload typed into the
   editor and hand-checked (`P1 0/6/3`, `P2 1/2/1`, `P3 2/8/2`, `P4 10/4/0` → waits
   `0/5/6/6`, mean 4.25 u); a burst of 0 stopped in the form with no request sent; a
   healing-enabled run that declared no incident (the Recovery page that used to crash);
   the footer, the Overview fact and `/api/runs` agreeing on the run count; the export
   endpoint byte-for-byte equal to the stored result; a reload that keeps the server history
   but closes the open run; a run restored from history; Research Results and About & Method.
3. **Backend loss and recovery.** With the server stopped, a run reported that the
   simulation did not run, the engine pill turned to *Backend unavailable*, the banner could
   be dismissed, and after a restart the retry control returned *Engine ready · 3.11.2* and
   a new run succeeded.
4. **An aged-out run.** After a restart, a comparison requested for a run the new server
   process had never seen was refused with *that run is not in the server's history* — no
   numbers were invented.

## Screenshots

| File | What it shows |
|---|---|
| `01-overview-1440.png` | Overview on a fresh load: engine ready, no run open |
| `02-studio-1440.png` | Simulation Studio with the four scenarios |
| `03-teaching-studio-1440.png` | Teaching example executed, four policies compared |
| `04-teaching-workload-inspector-1440.png` | Workload Explorer, SJF trace, dispatch block selected |
| `05-recovery-accepted-1440.png` | Recovery Analysis (seed 13): accepted, both judgment stages green |
| `06-recovery-rollback-1440.png` | Recovery Analysis (seed 3): inconclusive, rollback to FCFS |
| `07-comparison-running-1440.png` | Comparison in flight, button disabled |
| `08-comparison-complete-1440.png` | Comparison completed on the 15-process workload |
| `09-comparison-error-1440.png` | Injected HTTP 500: banner, no stale table, retry available |
| `10-custom-workload-1440.png` | Typed four-process workload executed |
| `11-custom-workload-trace-1440.png` | Its trace and process table |
| `12-custom-invalid-1440.png` | A burst of 0 rejected before any request |
| `13-recovery-no-incident-1440.png` | Healing run that declared no incident |
| `14-workload-healing-1440.png` | Its Workload Explorer trace |
| `15-overview-history-1440.png` | Overview with the session history |
| `16-empty-state-1440.png` | A run page after reload, with the history explanation |
| `17-restored-run-1440.png` | A result reopened from history |
| `18-research-1440.png` | Research Results: provenance and measured limits |
| `19-about-1440.png` | About & Method |
| `20-backend-down-1440.png` | A run while the backend is stopped |
| `21-after-retry-1440.png` | After the backend returns and the retry is used |
| `22-stale-run-1440.png` | Comparison for a run that no longer exists |

## Reproducing

```bash
# from the repository root, with the dependencies installed
.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --port 8000 --no-browser
# then drive the same flows with puppeteer-core against http://127.0.0.1:8000/
```

The check scripts are not committed (they live outside the repository); every assertion they
made is listed by name in `browser-checks.json` together with its observed value.

## Post-change check

After the captures, the working tree changed in four ways that do not touch rendering: an
indentation fix in `app.js`, an equivalent derivation of the tabular state-space constant
(648), explicit `encoding="utf-8"` arguments, and one docstring. To confirm the pages still
render the recorded values, a separate scripted pass ran against the same server and stored no
screenshots (so the captures above stay the reference): **9/9 checks passed**.

| Check | Observed |
|---|---|
| seed 13 still renders an accepted verdict | close `t=319`, residual `+0.4375`, verdict `accepted` |
| the two judgment stages keep the success tone | `data-tone` tail `ok, ok` |
| seed 3 still renders an inconclusive rollback | “rolled back to FCFS”, `inconclusive_residual_within_null_range` |
| the rollback judgment stages stay amber | `data-tone` tail `warn, warn` |
| the inconclusive badge keeps its colour | `rgb(147, 100, 28)` |
| the Studio result follows the selected teaching policy | “mean waiting 1.67 u, FCFS 4.33 u. Showing SJF.” |
| the teaching payload reports the hand-checked means | 1.67 and 4.33 present |
| the Workload Explorer reports the executed teaching run | “under SJF” for the three-process workload |
| no page errors during the run | none |

Two draft checks in that pass were wrong in the *script* and were corrected before the numbers
above were recorded: the Studio results note is `.result-note` (not `.run-result`), and the
restore notice lives in `#banner` outside `#view`. Both were verified again directly — the
history table lists the 10 runs the API returns, and opening one shows
“Restored a result from this session’s history. It is not a new execution.”
