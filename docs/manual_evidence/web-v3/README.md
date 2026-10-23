# Browser capture — v3 (the defect record)

This folder is the capture that **found** the interface defects fixed afterwards. It is kept
unchanged as evidence; the corrected state is in [`../web-v4/`](../web-v4/).

## Screenshots

| File | Page |
|---|---|
| `01-overview.png` | Overview |
| `02-studio-library.png` | Simulation Studio, example library |
| `03-workload-explorer.png` | Workload Explorer |
| `04-recovery-analysis.png` | Recovery Analysis |
| `05-policy-comparison.png` | Policy Comparison with results |
| `06-research-results.png` | Research Results |
| `07-about-method.png` | About & Method |
| `08-studio-420px.png` | Studio at 420 px |
| `09-workload-420px.png` | Workload Explorer at 420 px (follow-up capture) |
| `10-compare-stale-button.png` | the defect: button still disabled on "Running four policies…" after the comparison finished (follow-up capture) |

The first eight come from the main walk-through; the last two were taken by the follow-up
scripts that isolated the two interface defects.

## The defects this folder records

1. **Comparison button state.** After a comparison completed, the button stayed disabled and
   labelled "Running four policies…" with the four result rows already on screen
   (`10-compare-stale-button.png`). It returned to "Run comparison again" only after leaving the
   page. The busy flag was cleared after the final paint, so the paint read the stale value.
2. **Run counters disagreed.** The footer took its count from `/api/health`, which is polled
   every 25 s, while the sidebar took a different one from the run list, so immediately after a
   run the two figures differed. The follow-up capture records this as the check
   *"footer run counter refreshes only on the 25 s health poll (reported, not fixed)"*.
3. **420 px layout.** The process table cut its headers off mid-column, the Gantt showed only a
   fraction of the timeline, and the Studio summary stranded its last value against the edge.

## Raw reports

- `capture-report.json` — 44 checks from the full walk-through, the page measurements at 1440
  and 420 px, the console log and the network responses. One check failed: a self-check of the
  workload page ("real teaching trace"). The follow-up run re-measured the same page
  ("summary, table and CPU lane agree") and it passed, so that failure was in the check, not in
  the page.
- `recapture-report.json` — five follow-up checks: the re-measured workload page, route
  behaviour on a hard reload, the footer counter refreshing only on the health poll (recorded as
  reported, not fixed), the 420 px rule that the page itself never scrolls sideways while charts
  and tables scroll inside their panels, and the capture of `09-workload-420px.png`.

## Environment

Linux container (Debian), CPython 3.11.2, FastAPI/Uvicorn served by
`.venv/bin/python tools/serve_dashboard.py --host 0.0.0.0 --no-browser`. The headless image
lacks NSS/NSPR, so Chromium was started with a local ABI shim outside the repository
(`LD_LIBRARY_PATH=/tmp/nss-stub`). The screenshots are genuine renders at fixed viewports, but
they are not a desktop browser window and say nothing about Windows or physical phones.
