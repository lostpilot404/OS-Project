# Final delivery checks — 23 October 2026

## Environment and actual execution

Debian GNU/Linux 12 (bookworm), Linux x86-64, CPython 3.11.2. The checkout started clean
at `fd8f103d94f267844b3ac78256ab98e987306df0` on `arena/49fb0ff9-os-project`.
The delivered commit is available with `git log -1 --oneline`; this file does not embed
its own self-referential commit hash. No Windows WSL instance was available. Ubuntu/WSL
installation instructions were reviewed, not executed on Ubuntu or WSL.

Executed successfully:

```bash
python -m pip install -r requirements.lock -r requirements-report.txt
python -m pip check
python main.py --help
python main.py demo
python -m pytest -q tests/test_demo.py
python -m pytest -q
python tools/build_plain_report.py \
  --source docs/OS_Project_Final_Report.md \
  --output docs/OS_Project_Final_Report.pdf
python tools/verify_final_delivery.py
python -m compileall -q main.py experiments/demo.py \
  tools/verify_final_delivery.py tests/test_demo.py
git diff --check
```

The interpreter for these commands was `.venv/bin/python`; the environment was created
with `/usr/bin/python` at version 3.11.2. Dependency installation succeeded and pip check
reported no broken requirements. Five focused demo tests passed in 0.69 seconds.
The first complete suite returned **500 passed, 1 warning in 38.37 seconds**; the final
complete rerun returned **500 passed, 1 warning in 35.97 seconds**. The one warning is
an existing Matplotlib empty-legend warning in
`tests/test_probe_schema.py::test_healing_plots_rejects_legacy_schema`, not a failing test.
CLI tests also exercise invalid-command exit code 2, directory errors and absolute entry
point execution from another working directory.

## Real demo evidence

`python main.py demo` produced the 93-line capture `docs/examples/demo_linux_output.txt`
and the disposable `results/demo/demo.json`. A fresh execution matches that transcript
exactly after normalizing only the chosen output directory. Repeated JSON output is
byte-identical. The default study configuration is serialized in the JSON.

- Mixed seed 3: 15 processes, mean waiting 130.46666666666667; separately executed FCFS
  mean 150.6. State 205, SJF recovery, residual 0.20592237689676574, inconclusive, rollback
  to FCFS at time 193. The verdict is not called a failure or acceptance.
- Mixed seed 13: 15 processes, mean waiting 220.53333333333333; FCFS mean 255.8.
  State 205, SJF recovery, residual 0.437454899124583, accepted, configuration retained.
- Both are illustrative rule-controlled executions, not trained Q-policy evaluations.
  No trained model, experiment result or success message was substituted for execution.

The test checks final per-process arithmetic against the metrics and independently
against the sum of interval waiting increments. A fixture's observed benefit does not
change its actual verification verdict.

## PDF source, structure and page-by-page review

Final source: `docs/OS_Project_Final_Report.md`.
Generated PDF: `docs/OS_Project_Final_Report.pdf`, **18 pages**.
All 18 pages were rendered and visually inspected individually. After the final prose
corrections, changed pages 13, 14 and 15 were rendered and visually reinspected; all other
page renderings were unchanged. No clipped text, missing commands, broken numbering,
missing sections, accidental images/tables or colored elements were found. The final
source-reference page is intentionally short. No table of contents or diagram images
were introduced. Commands and all seven Mermaid blocks are printed in monospaced text;
no code block is split over a page boundary.

Page review coverage:

- Page 1: title, scope, provenance and abstract; plain black/white.
- Pages 2–4: methodology, observation, state formula, Q update, reward and verification.
- Pages 5–8: architecture and all seven intact Mermaid source blocks, module interaction,
  workload families and split introduction.
- Pages 9–12: dataset collection, tools, metrics, bootstrap design, frozen results,
  verification limits, detection, coverage and implementation evidence.
- Pages 13–14: real quick-demo behavior, artifact meaning, rule/Q distinction and Linux
  versus WSL verification status.
- Page 15: limitations and proposed future work, distinguishing the 8.0% operative
  phase-aware certification share from the 6.7% older unconditional boundary.
- Pages 16–17: conclusion, commands, independent workflows, probe and branch provenance.
- Page 18: implementation and numerical evidence references.

Automated verification in `tools/verify_final_delivery.py` additionally checks:

- All ten numbered sections, seven literal Mermaid flowcharts, no source table/image markup.
- Every code line fits the report's frame at its actual font size.
- No PDF XObjects, inline images or image-paint operators; only black/white color operators.
- Nonempty text and correct printed page number on every page.
- Regeneration from Markdown gives the **same PDF bytes**.
- Frozen means and artifact row counts, canonical probe 784 rows/98 incidents, separate
  historical raw-proxy archive 168 rows/21 incidents, unchanged final-test seed 23819.
- A fresh real demo matches the committed capture and the report's demo mean values.

A separate local PyMuPDF 1.26.6 inspection checked that every text bounding box lies
within the reviewed page margins and rendered the review pages. PyMuPDF is not required
to build the PDF or run the project; only the optional report dependencies are needed for
the committed verifier. The verification script does not claim to replace visual review.

Final file hashes (SHA-256):

```text
Markdown:
225df7ed0eb554fabcc44fbb3e83722f18ed622dd24caccfcfc16af1d7cce725
PDF:
9eadd87419a38f9347c396633951725fda3455f19b8cd6ac303493db88e111c1
Captured demo output:
ddcf0d0c01cb556fe2c72c0997a3c8dea8e725b4b1bb98a8df80c25b2cdd3f29
```

## Preservation and limits

The full expensive research study was **not rerun**. Existing `results/` and `figures/`
research artifacts, old plain report and Review 2 PDF/DOCX were not changed. No seed,
configuration, threshold or algorithm was retuned on final-test data. The demo and small
experiment tests do not replace the frozen study. The requested `OS(2).pdf` was not
accessible, so its exact layout could not be checked. The final report follows the nine
listed academic requirements and existing assessment structure instead.

The demo is intentionally a rule-policy demonstration of the primary healing loop,
not evidence of live Q training or RL superiority. The report accurately notes that
healing Q bootstrapping masks unvisited next actions, while live action selection also
uses the configuration availability mask. These are distinct guarantees. Single-CPU,
synthetic workloads, exact arrived burst knowledge, sparse Q coverage and limited
verification power remain research limitations, not defects hidden by the demo.
