# Teammate onboarding

## Purpose and scope

This is a single-CPU discrete-event simulation of adaptive scheduling, not a kernel
scheduler or real hardware benchmark. The primary research system detects degradation,
diagnoses observable queue conditions, selects a recovery action, then verifies it and
keeps, rolls back or safely pins an accepted configuration. Workloads are synthetic.
Read README, ARCHITECTURE and LIMITATIONS before interpreting performance claims.

## Code map and subsystem boundaries

- `config.py`, `errors.py`: validated shared configuration and domain exceptions.
- `workload/`: immutable process/workload/result models, seeded generators, fingerprints,
  and opt-in drift. Phase transition replaces suffix bursts/priorities, not arrivals;
  the convenience generator defaults its target to `short_jobs` (override with
  `phase_target_family`). Priority inversion here is a priority-skew model, not lock inversion.
- `scheduler/`: standalone FCFS, nonpreemptive SJF, Round Robin and priority baselines;
  `runtime.py` owns events, queue/service accounting and causal observations.
- `healing/`: primary monitor, detector, diagnosis, eight recovery actions, 648-state
  encoder, rule/Q policies, reward, verification, rollback/fallback, incident ledger,
  and optional burst estimator. Start with `controller.py::select_action`, then
  `observe_decision`, `_conclude_pending` and `end_episode`.
- `rl/`: shared tabular learner plus **separate** offline selection and non-healing
  runtime controllers. Runtime appendix uses 162 states/four policy actions. Offline
  selection chooses a whole-workload policy; it is not the healing loop.
- `experiments/`: split/seed design, training, evaluation, probes and report orchestration.
  `healing_config.py` declares frozen seeds and rejects retired streams.
- `evaluation/`: schedule metrics, paired crossed bootstrap, matched counterfactual
  analysis, reliability. Counterfactual replay knows the finished trace; the online
  controller must not. Do not feed audit outcomes back into evaluation-time decisions.
- `visualization/`: plot generation/style; `tools/`: academic PDF/DOCX report builders
  and final-delivery verification. `tests/` includes unit and small integration runs.
- `results/{healing,runtime,study,drift}/`, `figures/`: retained research evidence,
  including negative findings and a separately labelled legacy raw reward proxy.
  `docs/`: methodology, limitations, presentation instructions and historical submissions.

## Install and run (repository root, Linux)

Recorded interpreter is Python 3.11.2; Windows/WSL setup is reviewed, not executed here.
Use a virtual environment rather than Debian's externally managed system Python:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip check
.venv/bin/python main.py --help
.venv/bin/python main.py demo
.venv/bin/python -m pytest -q
```

Demo uses two deterministic rule-policy fixtures, writes ignored
`results/demo/demo.json`, and does not train. **Invoking `main.py` without a command starts the full healing
study**, not the demo. Explicit study commands can overwrite evidence: choose new paths.

```bash
.venv/bin/python main.py healing-study --results-dir /tmp/healing-new \
  --figures-dir /tmp/healing-new-figures
.venv/bin/python main.py experiment --results-dir /tmp/legacy-new \
  --figures-dir /tmp/legacy-new-figures
.venv/bin/python main.py runtime-experiment --results-dir /tmp/runtime-new
PYTHONPATH=. .venv/bin/python experiments/drift_uncertainty_experiment.py \
  --reps 1 --train-episodes 2 --drift phase_transition --out /tmp/drift-smoke
```

The last command is a smoke test, **not research evidence**. Canonical drift runs use
10 repetitions and 200 episodes/seed. The healing study uses 1,200 episodes per seed
across five seeds and takes roughly a minute on the audit host. Offline `study` uses
3,600 episodes per seed and multiple arms; inspect `study_config.py` before running.
`train` trains only the legacy offline selector and prints summaries.

Optional reports (system DejaVu fonts required):

```bash
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python tools/verify_final_delivery.py
.venv/bin/python tools/build_plain_report.py --source docs/OS_Project_Final_Report.md \
  --output /tmp/final-report.pdf
```

Keep PDF and Markdown synchronized. Review 2 has separate builders; do not overwrite
historical submissions merely to update maintenance prose.

## Design decisions, results and next steps

Deterministic RNG streams, immutable observations, disjoint workload splits, action
availability masks, held-out evaluation and matched replay protect experimental intent.
The estimator is a modelled EWMA/noise mechanism, not a calibrated real burst predictor.
Fixed SJF has oracle burst knowledge in uncertainty comparisons. Healing helps versus
FCFS on the retained design but loses to SJF; Q is not established as superior to the
waiting-time rule. Sparse Q coverage and inconclusive verification are important results.
Per-trace nearest-rank P95 equals max for 15 jobs. Reliability recovery delay measures
verification closure, not proof of sustained recovery. Regression aggregation requires
matched workloads (and seed averaging), not arbitrary index alignment.

Start development with a focused test and separate outputs. Never tune on frozen seed
23819 or silently replace results. Future work: realistic trace calibration, more
coverage, equal-information baselines and stronger verification; these are not implemented
claims. See FINALIZATION_AUDIT for current checks and historical reports for provenance.
