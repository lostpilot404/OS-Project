# Manual browser verification — live dashboard

Environment: Linux x86-64 container, CPython 3.11.2, FastAPI 0.121.2 / Uvicorn 0.40.0.
Browser: **HeadlessChrome 138.0.7204.0** (`@sparticuz/chromium` 138.0.2 via `puppeteer-core`
24), driven against `http://127.0.0.1:8000`. The image has no NSS/NSPR libraries, so a
local ABI shim outside the repository was used (`LD_LIBRARY_PATH=/tmp/nss-stub`) to start
Chromium. This is a real browser engine rendering the real page and issuing real requests,
but it is **not** a desktop Chrome/Edge/Firefox window on Windows; native widgets,
scrollbars and font fallbacks are those of the headless Linux build.

Server command:

```bash
.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --port 8000 --no-browser --log-level warning
```

Date of the captured session: 2026-10-23.

## Screenshots

| File | Shows |
|---|---|
| `01-workspace-before-run.png` | 1440×980 initial state: scenario list, run setup for the teaching example, empty workspace |
| `02-teaching-timeline.png` | Teaching run executed: metric band, provenance with run id, FCFS timeline |
| `03-teaching-workload.png` | Per-process table for the teaching run (waiting 0 / 5 / 8) |
| `04-teaching-priority-inspection.png` | Priority tab with a dispatch block selected and its read-out |
| `05-policy-comparison-teaching.png` | Four-policy comparison on the three teaching processes |
| `06-accepted-timeline.png` | Seed 13 timeline with the detection marker and FCFS→SJF band |
| `07-accepted-recovery.png` | Seed 13 lifecycle, verification band and incident record (accepted) |
| `08-rollback-recovery.png` | Seed 3 lifecycle and verification band (inconclusive, rollback to FCFS) |
| `09-research-evidence.png` | Historical research panel with source and caveats |
| `10-custom-editor.png` | Custom workload editor (process table, policy, quantum, switch cost) |
| `11-custom-sjf-result.png` | Custom workload executed under SJF after editing a burst time |
| `12-custom-invalid-input.png` | Invalid burst rejected before any request is sent |
| `13-narrow-420px.png` | 420 px viewport with the result and chart frames |
| `14-narrow-top.png` | 420 px viewport top: header, scenario list, run block |

`browser-checks.json` records the values captured during the session (health text, scenario
names, metric read-outs, recovery call-outs, custom-workload numbers, console messages).

## Test log

| ID | Procedure | Expected | Actual | Status |
|---|---|---|---|---|
| W01 | Open `/` | Page loads, backend status accurate | "Engine ready · 3.11.2"; `/api/health` `status ok`, probe 0.14 ms | PASS |
| W02 | Block `/api/*`, press Run | Clear failure, no substitute values | Banner "The simulation did not run…"; pill "Backend unavailable"; previous result labelled previous | PASS |
| W03 | Unblock, press Retry, run | Recovery | "Engine ready · 3.11.2"; new run rendered | PASS |
| W04 | Load each example | Name/description/parameters match, stale result cleared | All four correct; result panel cleared on switch | PASS |
| W05 | Run seed 13 | Accepted recovery | `accepted → SJF`, residual +0.4375, window t=243→319, mean wait 220.53 vs 255.80 | PASS |
| W06 | Run seed 3 | Inconclusive rollback | `inconclusive`, residual +0.2059 in [−0.30, +0.30], final FCFS, no confirmed failure | PASS |
| W07 | Teaching run + tabs + block click | Hand-checkable values | 4.33/7.33/4.33, 2 switches, 100 %, throughput 0.3333, elapsed 9; block read-out correct | PASS |
| W08 | Custom: burst 9→12, SJF | New configuration reaches backend | New fingerprint, mean wait 8.40, 4 switches | PASS |
| W09 | Custom: burst 0 and duplicate pid | Actionable validation, no fabricated result | Inline error + banner, 0 POSTs, result hidden; corrected form ran (9.20) | PASS |
| W10 | Compare policies | Four executions, correct labels/units | Teaching SJF 1.67 best; seed 13 SJF 172.13 best vs FCFS 255.80 | PASS |
| W11 | Rapid repeated clicks while running | No duplicate submissions | 1 POST, button disabled/"Running…", scenario switch ignored | PASS |
| W12 | Delay comparison, start new run | Late response discarded | Comparison section reset; provenance shows the newer run | PASS |
| W13 | Export the displayed result | Download equals displayed run | Filename and `run_id` match; 130.466667 exported vs 130.47 displayed | PASS |
| W14 | Keyboard-only workflow | Core workflow operable | Tab order covers nav/cards/Run/select; Enter runs; ArrowRight switches tabs | PASS |
| W15 | 1440 / 900 / 420 px | No overflow, readable | `scrollWidth == innerWidth` at all three; charts scroll inside their frames | PASS |
| W16 | Research panel | Historical data separated and caveated | "HISTORICAL · READ-ONLY", values match `healing_summary.json`, Q−rule caveat visible | PASS |
| W17 | API errors through the UI/HTTP | Explicit, field-named messages | 422/404/500 all as documented in `docs/LIVE_DASHBOARD.md` | PASS |

A follow-up smoke pass after the last documentation edits re-ran the accepted-recovery
scenario plus the four-policy comparison and opened the raw JSON viewer: badge
"accepted → SJF", comparison note naming the same run id as the displayed result
(`run 451a1596`), raw-viewer summary "11.9 KB of JSON below (SVG charts omitted)", and no
console messages.

Checks that **could not** be performed here, and are therefore not claimed: a Windows
desktop browser session, PowerShell execution, touch devices, and screen-reader
announcements (only the ARIA structure and keyboard operation were exercised).
