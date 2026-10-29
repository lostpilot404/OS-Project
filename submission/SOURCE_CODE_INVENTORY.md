# Source-Code Inventory — OS-Project

**Branch:** `arena/1a4dbf1c-os-project`  
**Commit:** `e6b4055e29e29bec6deaabb20a18d3ba8c6b6a15`

## Core Engine

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~15403 bytes | CLI entry point: `demo`, `healing-study`, `study`, `runtime-experiment`, `experiment`, `train` commands |
| `config.py` | ~33986 bytes | Scheduler configuration (quantum, switch cost, priority sense, bounds) |
| `errors.py` | ~883 bytes | ValidationError, ConfigurationError exception classes |

## Workload Model & Generation

| File | Lines | Purpose |
|------|-------|---------|
| `workload/models.py` | ~18863 bytes | Process, Workload, ExecutionSlice, ProcessOutcome, ScheduleResult, build_schedule_result with metric definitions |
| `workload/generator.py` | ~10742 bytes | WorkloadGenerator with 7 families, seeded NumPy sampling, derive_seed procedure |
| `workload/fingerprint.py` | ~12657 bytes | Workload fingerprinting, split integrity, disjointness checks |
| `workload/drift.py` | ~12657 bytes | Drift injection: burst_shock, arrival_storm, phase_transition, priority_inversion (off by default) |

## Schedulers

| File | Lines | Purpose |
|------|-------|---------|
| `scheduler/base.py` | ~7563 bytes | Timeline (slice merging), SchedulingPolicy, NonPreemptiveReadyQueuePolicy engine |
| `scheduler/fcfs.py` | ~1067 bytes | First-Come, First-Served (non-preemptive) |
| `scheduler/sjf.py` | ~1224 bytes | Shortest Job First (non-preemptive) |
| `scheduler/priority.py` | ~1756 bytes | Priority scheduling (non-preemptive, lower number = higher priority) |
| `scheduler/round_robin.py` | ~3636 bytes | Round Robin (preemptive, configurable quantum) |
| `scheduler/runtime.py` | ~22062 bytes | RuntimeSimulator (event-driven, causal observations, interval records) |

## Healing / Self-Controller

| File | Lines | Purpose |
|------|-------|---------|
| `healing/config.py` | ~18842 bytes | HealingConfig: frozen thresholds for monitor, detector, verifier, safety fallback |
| `healing/monitor.py` | ~14666 bytes | CausalMonitor: rolling completion windows, EWMA reference/band, queue-load accrual |
| `healing/detector.py` | ~9732 bytes | DegradationDetector: threshold, hysteresis, cooldown, work gate, load gate, intervention cap |
| `healing/diagnosis.py` | ~11101 bytes | Diagnosis from visible ready-set evidence (6 categories) |
| `healing/actions.py` | ~8024 bytes | Eight recovery actions (hold, set FCFS/SJF/RR/Priority, grow/shrink quantum, revert) |
| `healing/policies.py` | ~21623 bytes | HealingContext, HealingContextEncoder (648 states), HealingPolicy, RuleWaitingTimePolicy, QLearningHealingPolicy |
| `healing/verifier.py` | ~14666 bytes | Phase-adjusted residual verifier: accept/reject/inconclusive verdicts |
| `healing/controller.py` | ~60527 bytes | SelfHealingController: monitor→detect→diagnose→act→verify→rollback/fallback state machine |
| `healing/reward.py` | ~6676 bytes | Controller reward: phase-adjusted residual minus intervention cost and rollback penalty |
| `healing/burst_estimator.py` | ~20359 bytes | EWMA per-bucket burst predictor with log-normal noise (off by default) |
| `healing/canonical_numbers.py` | ~3786 bytes | Canonical numeric constants |

## Reinforcement Learning

| File | Lines | Purpose |
|------|-------|---------|
| `rl/q_learning.py` | ~13904 bytes | Tabular Q-learning agent (alpha=0.10, gamma=1.00, epsilon decay 1.0→0.05) |
| `rl/runtime_controller.py` | ~5639 bytes | RuntimeQController for offline study |
| `rl/runtime_state.py` | ~3376 bytes | Runtime state representation |
| `rl/state.py` | ~13748 bytes | State encoding for RL |
| `rl/objective.py` | ~13117 bytes | Objective function for RL training |
| `rl/reward.py` | ~6236 bytes | Reward computation |
| `rl/adaptive.py` | ~7955 bytes | Adaptive components |
| `rl/selector.py` | ~19167 bytes | Action selection with masking |
| `rl/explain.py` | ~22099 bytes | Explanation generation |
| `rl/fingerprint_state.py` | ~4982 bytes | State fingerprinting |

## Experiments

| File | Lines | Purpose |
|------|-------|---------|
| `experiments/demo.py` | ~10865 bytes | Demo fixtures (run_healing_workload, run_fixture, run_demo) — used by live dashboard |
| `experiments/healing_config.py` | — | HealingExperimentConfig: frozen study design, arms, seeds, Q-learning settings |
| `experiments/healing_experiment.py` | — | Training, evaluation, artifact writing, offline audits |
| `experiments/healing_detection.py` | — | Controlled degradation scenario probes |
| `experiments/healing_thresholds.py` | — | Control-based threshold derivation |

## Evaluation

| File | Lines | Purpose |
|------|-------|---------|
| `evaluation/metrics.py` | ~10739 bytes | Schedule metrics computation |
| `evaluation/paired.py` | ~7113 bytes | Paired bootstrap comparisons (family-stratified, crossed) |
| `evaluation/comparison.py` | ~9555 bytes | Policy comparison |
| `evaluation/counterfactual.py` | ~5530 bytes | Counterfactual replay |
| `evaluation/reliability.py` | ~12347 bytes | Reliability metrics (P95, starvation, Jain fairness, etc.) |

## Dashboard (Live API)

| File | Lines | Purpose |
|------|-------|---------|
| `dashboard/api.py` | ~21569 bytes | FastAPI app: health, examples, simulations, runs, export, comparisons, research |
| `dashboard/service.py` | ~54461 bytes | Service layer: scenario execution, healing view, metrics view, charts |
| `dashboard/model.py` | ~40632 bytes | Schema validation: Fixture, Incident, Metrics, Outcome, parse_fixture, DashboardError |
| `dashboard/teaching.py` | ~6766 bytes | Teaching panel: four-policy worked example with hand-verified metrics |
| `dashboard/scenarios.py` | ~3886 bytes | Scenario catalogue: teaching, recovery-accepted, recovery-rollback, custom |
| `dashboard/research.py` | ~15077 bytes | Read-only research panel over committed healing_summary.json + comparisons CSV |
| `dashboard/charts.py` | ~28700 bytes | SVG chart generation (Gantt, bars, lifecycle flow, verification band) |
| `dashboard/render.py` | ~78639 bytes | Lifecycle stages, rendering |
| `dashboard/build.py` | ~2156 bytes | Dashboard build utilities |
| `dashboard/cli.py` | ~3886 bytes | Dashboard CLI |
| `dashboard/web/index.html` | — | Frontend HTML shell |
| `dashboard/web/app.js` | — | Frontend application logic |
| `dashboard/web/styles.css` | — | Frontend styles |

## Tools

| File | Purpose |
|------|---------|
| `tools/serve_dashboard.py` | Start the live FastAPI dashboard server |
| `tools/build_teacher_dashboard.py` | Build self-contained HTML teacher dashboard from demo.json |

## Tests

| File | Purpose |
|------|---------|
| `tests/test_fcfs.py` | FCFS scheduler tests |
| `tests/test_sjf.py` | SJF scheduler tests |
| `tests/test_priority.py` | Priority scheduler tests |
| `tests/test_round_robin.py` | Round Robin scheduler tests |
| `tests/test_metrics.py` | Metrics computation tests |
| `tests/test_healing_monitor.py` | Monitor tests |
| `tests/test_healing_detector.py` | Detector tests |
| `tests/test_healing_diagnosis.py` | Diagnosis tests |
| `tests/test_healing_verification.py` | Verifier tests |
| `tests/test_healing_controller.py` | Controller tests |
| `tests/test_healing_policies.py` | Healing policy tests |
| `tests/test_healing_experiment.py` | Experiment tests |
| `tests/test_q_learning.py` | Q-learning tests |
| `tests/test_reward.py` | Reward tests |
| `tests/test_reward_alignment.py` | Reward alignment tests |
| `tests/test_dashboard_api.py` | API tests |
| `tests/test_dashboard_frontend.py` | Frontend tests |
| `tests/test_teacher_dashboard.py` | Teacher dashboard tests |
| `tests/test_reproducibility.py` | Reproducibility tests |
| `tests/test_experiments.py` | Experiment tests |
| `tests/test_drift_and_reliability.py` | Drift/reliability tests |
| `tests/test_drift_reporting.py` | Drift reporting tests |
| `tests/test_explanations.py` | Explanation tests |
| `tests/test_counterfactual.py` | Counterfactual tests |
| `tests/test_runtime_scheduler.py` | Runtime scheduler tests |
| `tests/test_runtime_experiment.py` | Runtime experiment tests |
| `tests/test_fingerprint.py` | Fingerprint tests |
| `tests/test_generator.py` | Generator tests |
| `tests/test_objective.py` | Objective tests |
| `tests/test_output_validation.py` | Output validation tests |
| `tests/test_selector.py | Selector tests |
| `tests/test_state.py` | State tests |
| `tests/test_schedule_result.py` | Schedule result tests |
| `tests/test_adaptive.py` | Adaptive tests |
| `tests/test_burst_estimator_terminology.py` | Burst estimator terminology tests |
| `tests/test_canonical_numbers.py` | Canonical numbers tests |
| `tests/test_encoding_discipline.py` | Encoding discipline tests |
| `tests/test_experiment_design.py` | Experiment design tests |
| `tests/test_probe_doc_consistency.py` | Probe documentation consistency tests |
| `tests/test_probe_schema.py` | Probe schema tests |
| `tests/test_study_design.py` | Study design tests |
| `tests/test_visual_style.py` | Visual style tests |
| `tests/test_workload_models.py` | Workload model tests |
| `tests/test_demo.py` | Demo tests |
| `tests/test_dashboard_audit_fixes.py` | Dashboard audit fix tests |
| `tests/helpers.py` | Test helpers |

## Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | numpy==2.4.6, pandas==3.0.6, matplotlib==3.11.2, pytest==9.1.1 |
| `requirements-dashboard.txt` | FastAPI + Uvicorn dependencies |
| `requirements-report.txt` | Report generation dependencies |
| `requirements.lock` | Pinned dependency versions |
| `pytest.ini` | Test configuration: testpaths=tests, pythonpath=., addopts=-ra |

## Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview, setup, demo instructions, known limitations |
| `docs/ARCHITECTURE.md` | System architecture |
| `docs/DESIGN_AND_CHOICES.md` | Design decisions |
| `docs/METHODOLOGY.md` | Research methodology |
| `docs/EVALUATION_RESULTS.md` | Experimental results with statistical analysis |
| `docs/LIMITATIONS.md` | Scope limitations and threats to validity |
| `docs/LIVE_DASHBOARD.md` | API documentation, error shapes, troubleshooting |
| `docs/TEACHER_LIVE_DEMO.md` | Teacher demonstration procedure |
| `docs/DEMO_SCRIPT_FOR_LECTURER.md` | 5–8 minute lecturer script |
| `docs/TEACHER_QA.md` | Teacher Q&A |
| `docs/OS_Project_Final_Report.md` | Final research report (markdown) |
| `docs/OS_Project_Final_Report.pdf` | Final research report (PDF) |
| `docs/OS_Project_Final_Report.docx` | Final research report (DOCX) |
| `docs/OS_Project_Report_Plain.md` | Plain report (historical) |
| `docs/OS_Project_Report_Plain.pdf` | Plain report PDF (historical) |
| `docs/Review_2_Report.docx` | Review 2 report |
| `docs/Review_2_Report.pdf` | Review 2 report PDF |
| `docs/FINAL_SUBMISSION_REPORT.md` | Final submission report (from different branch state) |
| `docs/FINALIZATION_AUDIT.md` | Finalization audit |
| `docs/FINAL_AUDIT_AND_VERIFICATION.md` | Final audit and verification |
| `docs/ENGINEERING_COMPLETION_REPORT.md` | Engineering completion report |
| `docs/MANUAL_VERIFICATION_REPORT.md` | Manual verification report |
| `docs/LEAKAGE_FIX_AND_DRIFT_REPORT.md` | Leakage fix and drift report |
| `docs/APPENDIX_RUNTIME_SIMULATOR.md` | Runtime simulator appendix |
| `docs/TEAMMATE_ONBOARDING.md` | Teammate onboarding |
| `docs/WSL_SETUP.md` | WSL2 setup instructions |
| `docs/BRANCH_CONSOLIDATION.md` | Branch consolidation notes |
| `docs/FINAL_DELIVERY_CHECKS.md` | Final delivery checks |
| `docs/manual_evidence/` | Manual evidence captures |
| `docs/figures/` | Figure assets |
| `docs/examples/` | Example files |

## Research Artifacts (Committed, Read-Only)

| File | Purpose |
|------|---------|
| `results/healing/healing_summary.json` | Frozen healing study: 210 workloads, seed 23819 |
| `results/healing/healing_comparisons.csv` | Per-trace comparison rows |
| `results/healing/healing_metrics_final_test.csv` | Final test metrics |
| `results/healing/healing_audit.csv` | Action-effect audit probe (784 rows, 98 incidents) |
| `results/drift/` | Drift experiment artifacts |
| `results/runtime/` | Runtime experiment artifacts |
| `results/study/` | Offline study artifacts |

## Audit Package

| File | Purpose |
|------|---------|
| `audit/manual-iceberg/AUDIT_REPORT.md` | Full audit report |
| `audit/manual-iceberg/BUG_LEDGER.md` | Bug ledger (OS-001…OS-006) |
| `audit/manual-iceberg/MANUAL_TEST_MATRIX.md` | Manual test matrix (63 checks) |
| `audit/manual-iceberg/FAKE_FUNCTIONALITY_AUDIT.md` | Fake-functionality audit (18 features) |
| `audit/manual-iceberg/EVIDENCE_INDEX.md` | Evidence index (E01–E07) |
| `audit/manual-iceberg/REMEDIATION_LOG.md` | Remediation plan (R1–R4) |
| `audit/manual-iceberg/FINAL_DEMO_CHECKLIST.md` | Teacher demo checklist |
| `audit/manual-iceberg/evidence/api/` | 19 captured API responses |

## Submission Package

| File | Purpose |
|------|---------|
| `submission/SUBMISSION_CHECKLIST.md` | This file |
| `submission/SOURCE_CODE_INVENTORY.md` | Source-code inventory |
| `submission/FINAL_CHANGE_SUMMARY.md` | Concise final change summary |
