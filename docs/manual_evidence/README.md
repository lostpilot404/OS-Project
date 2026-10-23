# Manual evidence

Everything in this directory was produced by running the code in this repository, not written
by hand. Each folder is self-contained and says which build of the interface it belongs to.

| Folder | What it records |
|---|---|
| [`dashboard/`](dashboard/) | command-line and rendered-chart checks for the static teacher dashboard: captured stdout/stderr, the rendered SVG/PNG charts and the scripts that produced them |
| [`web/`](web/) | the first browser interface, verified page by page at 1440 / 900 / 420 px; see [`VERIFICATION.md`](web/VERIFICATION.md) |
| [`web-v2/`](web-v2/) | the redesigned interface: every page at three widths, empty states and 50 scripted browser checks; see [`VERIFICATION.md`](web-v2/VERIFICATION.md) and [`browser-checks.json`](web-v2/browser-checks.json) |
| [`web-v3/`](web-v3/) | the capture that **found** three defects (comparison button left disabled, footer and sidebar counts disagreeing, 420 px layout); see [`README.md`](web-v3/README.md) |
| [`web-v4/`](web-v4/) | the verification **after** those defects were fixed: 36 browser checks, including the button lifecycle, an injected API failure, run-count agreement and the 420 px reflow; see [`README.md`](web-v4/README.md) |

Nothing in an older folder is rewritten when a newer one is added: `web-v3/` is kept as the
record of the defects, and `web-v4/` shows the same screens afterwards.

Captures are real browser screenshots of the running dashboard (`tools/serve_dashboard.py`)
driven by headless Chromium through Puppeteer — not rendered substitutes. They were taken in a
Linux container, so they say nothing about Windows rendering or about physical phones; those
remain unverified. The commands to repeat any capture are recorded in the folder's own report.
