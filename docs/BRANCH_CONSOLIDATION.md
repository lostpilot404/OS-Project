# Consolidation into main

The owner requested that outstanding branch changes be brought into `main`.
The integration was prepared on `arena/49fb0ff9-os-project` for a single pull request.

## Branch audit

- `main` started at `fd8f103d94f267844b3ac78256ab98e987306df0`.
- `arena/49fb0ff9-os-project` at `d3b6a3814bb7dea734557560577ddb6fbd619ebe`
  supplied the actual demo, WSL guide, academic report and five demo tests.
- `arena/1ffa6b9e-os-project` at `7bfc40d68be9b4c14b2e44b0b340dbcc90484b46`
  supplied drift generation, optional burst estimation, reliability metrics,
  experiment artifacts, tests and its research reports. All ten outstanding commits
  are included through a normal merge, rather than manually copying selected files.
- `arena/630f2a01-os-project` at `fcd9a71c6faed15a0c6907e9610b28b3bc9ef703`
  was already incorporated by PR #9. Its file tree was identical to the starting
  `main`; no new source changes needed importing from it.

The README conflict was resolved by retaining both the quick-demo instructions and
reliability-extension material. Current setup and lecturer guides recommend `main`.
Historical report provenance and dated test counts remain historical, not new claims.
The final PDF describes the default frozen study and demo; the optional drift study
is documented separately in `docs/FINAL_SUBMISSION_REPORT.md` and
`docs/LEAKAGE_FIX_AND_DRIFT_REPORT.md`. Its results are not pooled with the frozen study.

## Integration checks actually executed

On Debian 12 Linux x86-64 / Python 3.11.2, installed `requirements.lock` and
`requirements-report.txt` into a fresh `.venv`, then ran:

- `python -m pytest -q`: **529 passed, 1 warning in 47.20 seconds**.
- `python main.py demo`: actual default-threshold fixture output unchanged.
- `python tools/verify_final_delivery.py`: passed, including exact demo transcript,
  canonical/legacy probe counts, frozen means, and repeatable 18-page PDF bytes.
- `python -m pip check`: no broken requirements.

The warning is the existing Matplotlib empty-legend warning in the legacy-probe
rejection test. No expensive research experiment was rerun. Existing frozen healing,
runtime and offline artifacts, old reports, final PDF and incoming drift CSVs were
preserved. The incoming CSVs use CRLF line endings; whitespace review uses Git's
`cr-at-eol` setting so those unchanged CSV bytes are not treated as trailing spaces.

No branch was force-pushed, reset or deleted. The initial local files matched the
already-pushed demo delivery byte for byte; a safety stash was retained before the
checkout fast-forward. Normal users can use only `main`; the active Arena session
branch is retained for session tracking. Pull-request merge state and the final
main commit are verified on GitHub after the merge rather than predicted here.
