# Release Baseline Record — OS-Project

**Branch:** `arena/1a4dbf1c-os-project`
**HEAD commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`
**Remote:** `https://github.com/lostpilot404/OS-Project.git` (origin)
**Default branch:** `main` (origin/HEAD -> origin/main)
**Remote main SHA:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15` (identical to local)
**Divergence:** None — local branch == origin/main

## Working-tree state at baseline

Modified (tracked):
- `docs/OS_Project_Final_Report.md` — added References section (5 verified citations)

Untracked:
- `audit/` — prior manual audit package (manual-iceberg/ + baseline/)
- `docs/OS_Project_Final_Report.docx` — generated DOCX of the final report
- `submission/` — submission package (checklist, source inventory, change summary, ZIP)

## Python environment
- `.venv/` at repo root (Python 3.11.2), requirements installed

## Test status at baseline
- `pytest tests/` → 674 passed, 0 failed (executed in prior session)

## Submission blockers at baseline
- Browser testing BLOCKED: no browser binary, no X server, Playwright Chromium download blocked by sandbox host allowlist
- RL is offline-only in the live dashboard (disclosed limitation)
- Concurrency guard runtime 429s not reproducible against live uvicorn (source-confirmed)
