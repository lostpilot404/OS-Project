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
