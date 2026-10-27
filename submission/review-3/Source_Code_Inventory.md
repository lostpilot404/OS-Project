# Source Code Inventory

**Project:** Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads  
**Repository:** https://github.com/lostpilot404/OS-Project  
**Branch:** arena/c2630615-os-project

---

## 1. Core Source Modules

### Configuration and Entry Points
| File | Lines (approx.) | Description |
|------|-----------------|-------------|
| `main.py` | ~100 | CLI entry point with subcommands (demo, study, healing-study, runtime-experiment) |
| `config.py` | ~400 | Dataclasses: SchedulerConfig, WorkloadFamilyConfig, QLearningConfig, TrainingConfig, ExperimentConfig |
| `errors.py` | ~30 | Custom exception types (ConfigurationError, ValidationError) |

### Scheduling Engine (`scheduler/`)
| File | Description |
|------|-------------|
| `scheduler/__init__.py` | Package init, exports POLICY_CLASSES |
| `scheduler/base.py` | Abstract base class for schedulers |
| `scheduler/fcfs.py` | First-Come First-Served implementation |
| `scheduler/sjf.py` | Shortest Job First implementation |
| `scheduler/round_robin.py` | Round Robin with configurable quantum |
| `scheduler/priority.py` | Priority-based scheduling |
| `scheduler/runtime.py` | RuntimeSimulator: discrete-event engine with causal observation boundary |

### Workload Generation (`workload/`)
| File | Description |
|------|-------------|
| `workload/__init__.py` | Package init |
| `workload/models.py` | Process, Workload, ProcessOutcome, ScheduleResult dataclasses |
| `workload/generator.py` | WorkloadGenerator: generate(family, seed), generate_with_drift() |
| `workload/fingerprint.py` | SHA-256 content fingerprint for workload identity |
| `workload/drift.py` | DriftSpec, apply_drift: mid-trace workload phase shifts |

### Self-Healing Controller (`healing/`)
| File | Description |
|------|-------------|
| `healing/__init__.py` | Package init |
| `healing/config.py` | HealingConfig: all thresholds, gates, costs, bounds |
| `healing/monitor.py` | CausalMonitor: rolling EWMA reference and dispersion band |
| `healing/detector.py` | DegradationDetector: hysteresis, cooldown, work/congestion gates |
| `healing/diagnosis.py` | ReadyEvidence, diagnose: map ready-set to primary cause |
| `healing/actions.py` | 8-action recovery space, SchedulerSettings, apply_healing_action |
| `healing/policies.py` | HealingContextEncoder (648 states), MappedRulePolicy, QLearningHealingPolicy |
| `healing/verifier.py` | Tri-state verification: accepted/rejected/inconclusive with phase residuals |
| `healing/controller.py` | SelfHealingController: the complete state machine |
| `healing/reward.py` | controller_reward: reward computation for Q-learning |
| `healing/burst_estimator.py` | BurstEstimator: modelled EWMA + log-normal noise predictor |
| `healing/canonical_numbers.py` | Canonical numerical constants |

### Reinforcement Learning (`rl/`)
| File | Description |
|------|-------------|
| `rl/__init__.py` | Package init |
| `rl/q_learning.py` | QLearningAgent: tabular Q-learning with epsilon-greedy |
| `rl/state.py` | State encoder for offline selector |
| `rl/adaptive.py` | Offline policy selector |
| `rl/selector.py` | Policy selection interface |
| `rl/reward.py` | Reward models |
| `rl/runtime_state.py` | Runtime state representation |
| `rl/runtime_controller.py` | Runtime Q-learning controller |
| `rl/runtime_heuristic.py` | Runtime heuristic baseline |
| `rl/explain.py` | Explanation utilities |
| `rl/fingerprint_state.py` | Fingerprint-based state encoding |
| `rl/objective.py` | Objective function definitions |

### Evaluation (`evaluation/`)
| File | Description |
|------|-------------|
| `evaluation/__init__.py` | Package init |
| `evaluation/metrics.py` | compute_metrics: waiting, turnaround, response, utilization, throughput, switches |
| `evaluation/paired.py` | Family-stratified crossed bootstrap comparisons |
| `evaluation/counterfactual.py` | Counterfactual replay audit |
| `evaluation/comparison.py` | Comparison utilities |
| `evaluation/reliability.py` | P95, starvation, fairness, recovery delay, false-intervention rates |

### Experiments (`experiments/`)
| File | Description |
|------|-------------|
| `experiments/__init__.py` | Package init |
| `experiments/healing_config.py` | Frozen study design: seeds, splits, arms |
| `experiments/healing_experiment.py` | Main study runner: training, evaluation, audit, coverage, figures |
| `experiments/healing_thresholds.py` | Threshold derivation from no-action control |
| `experiments/healing_detection.py` | Controlled degradation scenarios, recall, delay |
| `experiments/healing_thresholds.py` | Derivation of declared boundaries |
| `experiments/demo.py` | Demo fixture execution |
| `experiments/study.py` | Offline policy-selection study |
| `experiments/study_config.py` | Study configuration |
| `experiments/train.py` | Training loop |
| `experiments/evaluate.py` | Evaluation loop |
| `experiments/report.py` | Report generation |
| `experiments/run_experiment.py` | Legacy experiment runner |
| `experiments/runtime_experiment.py` | Runtime simulator appendix |
| `experiments/runtime_config.py` | Runtime configuration |
| `experiments/drift_uncertainty_experiment.py` | Drift + uncertainty experiment |

### Visualization (`visualization/`)
| File | Description |
|------|-------------|
| `visualization/__init__.py` | Package init |
| `visualization/plots.py` | Standard figures from study artifacts |
| `visualization/healing_plots.py` | Healing-specific figures |
| `visualization/study_plots.py` | Study-specific figures |
| `visualization/style.py` | Monochrome plot styling |

### Dashboard (`dashboard/`)
| File | Description |
|------|-------------|
| `dashboard/__init__.py` | Package init |
| `dashboard/api.py` | FastAPI routes: health, examples, simulations, comparisons, research |
| `dashboard/service.py` | Simulation service adapter |
| `dashboard/scenarios.py` | Scenario library metadata |
| `dashboard/model.py` | Dashboard data model and validation |
| `dashboard/teaching.py` | Teaching example execution |
| `dashboard/research.py` | Research results reader |
| `dashboard/charts.py` | Inline SVG chart generation |
| `dashboard/render.py` | HTML template rendering |
| `dashboard/build.py` | Static dashboard builder |
| `dashboard/cli.py` | CLI for dashboard tools |
| `dashboard/web/index.html` | Main page (7 sections: Overview, Studio, Workload, Recovery, Compare, Research, About) |
| `dashboard/web/app.js` | Client-side JavaScript |
| `dashboard/web/styles.css` | Monochrome responsive styles |

### Tools (`tools/`)
| File | Description |
|------|-------------|
| `tools/serve_dashboard.py` | Launch the FastAPI dashboard server |
| `tools/build_teacher_dashboard.py` | Build static HTML classroom dashboard |
| `tools/build_review2_report.py` | Generate Review 2 PDF report |
| `tools/build_review2_docx.py` | Generate Review 2 DOCX report |
| `tools/build_plain_report.py` | Generate plain-text PDF report |
| `tools/review2_content.py` | Review 2 narrative content |
| `tools/review2_layout.py` | Review 2 PDF layout |
| `tools/review2_figures.py` | Review 2 figure generation |
| `tools/review2_data.py` | Review 2 data reader |
| `tools/verify_final_delivery.py` | Delivery verification script |

## 2. Test Suite (`tests/`)

58 test files covering all modules. Key tests:

| Test File | Coverage |
|-----------|----------|
| `test_fcfs.py`, `test_sjf.py`, `test_round_robin.py`, `test_priority.py` | Four scheduler policies |
| `test_metrics.py` | Metric calculations |
| `test_q_learning.py` | Q-learning agent |
| `test_healing_controller.py` | Self-healing controller state machine |
| `test_healing_detector.py` | Degradation detection |
| `test_healing_verification.py` | Tri-state verifier |
| `test_healing_monitor.py` | Causal monitor |
| `test_healing_diagnosis.py` | Diagnosis module |
| `test_healing_policies.py` | Recovery policies |
| `test_healing_experiment.py` | Full experiment pipeline |
| `test_dashboard_api.py` | FastAPI endpoints |
| `test_dashboard_frontend.py` | Frontend functionality |
| `test_reproducibility.py` | Deterministic reproducibility |
| `test_generator.py` | Workload generation |

## 3. Research Artifacts (`results/`)

### `results/healing/` — Frozen Healing Study (21 artifacts)
- `healing_summary.json` — Complete study summary
- `healing_comparisons.csv` — 10 paired comparisons
- `healing_audit.csv` — Counterfactual replay audit
- `healing_incidents_final_test.csv` — All incidents on held-out split
- `healing_incidents_validation.csv` — All incidents on validation split
- `healing_metrics_final_test.csv` — Per-arm metrics on test
- `healing_metrics_validation.csv` — Per-arm metrics on validation
- `healing_episodes_final_test.csv` — Per-episode results
- `healing_coverage.csv` — Q-table coverage
- `healing_coverage_curve.csv` — Coverage over training
- `healing_q_seed_variability.csv` — Seed-to-seed variability
- `healing_training_episodes.csv` — Training episode data
- `healing_action_probe.csv` — Canonical action probe (784 rows)
- `healing_action_probe.legacy_raw_proxy.csv` — Legacy probe (168 rows)
- `healing_detection_scenarios.csv` — Detection scenarios
- `healing_detection_summary.csv` — Detection summary
- `healing_threshold_derivation.csv` — Threshold derivation
- `healing_verification_models.csv` — Phase model comparison
- `healing_demo.json` — Demo trace
- `healing_report.md` — Generated report
- `healing_q_table.csv` — Q-table values

### `results/study/` — Offline Policy Selection Study
- `design.json`, `manifest.json`, `summary.json`
- `decisions.csv`, `counterfactuals.csv`, `robustness.csv`
- `report.md`

### `results/runtime/` — Runtime Experiment
- `runtime_summary.json`, `runtime_report.md`
- Training, validation, and test metrics CSVs
- Q-table and workload manifest

### `results/drift/` — Drift Experiments
- 3 experiment conditions × 4 files each

## 4. Documentation (`docs/`)

| File | Description |
|------|-------------|
| `ARCHITECTURE.md` | Module map, data flow, causal boundary |
| `METHODOLOGY.md` | Object of study, workload families, splits, metrics, statistics |
| `EVALUATION_RESULTS.md` | Every number traceable to artifacts |
| `LIMITATIONS.md` | Complete limitations with measurements |
| `DESIGN_AND_CHOICES.md` | Non-obvious decisions, rejected alternatives |
| `LIVE_DASHBOARD.md` | Dashboard API, error shapes, troubleshooting |
| `TEACHER_LIVE_DEMO.md` | Classroom demonstration guide |
| `TEACHER_QA.md` | Anticipated questions and answers |
| `FINAL_AUDIT_AND_VERIFICATION.md` | End-to-end audit record |
| `FINALIZATION_AUDIT.md` | Maintenance verification |
| `FINAL_DELIVERY_CHECKS.md` | Delivery verification |
| `FINAL_SUBMISSION_REPORT.md` | Submission report |
| `ENGINEERING_COMPLETION_REPORT.md` | Engineering record |
| `DEMO_SCRIPT_FOR_LECTURER.md` | 5-8 minute lecturer script |
| `TEAMMATE_ONBOARDING.md` | New contributor guide |
| `WSL_SETUP.md` | WSL2 setup instructions |
| `APPENDIX_RUNTIME_SIMULATOR.md` | Runtime simulator appendix |
| `BRANCH_CONSOLIDATION.md` | Branch management record |
| `LEAKAGE_FIX_AND_DRIFT_REPORT.md` | Burst estimator and drift report |
| `MANUAL_VERIFICATION_REPORT.md` | Browser verification evidence |
| `Review_2_Report.pdf` | Review 2 report (PDF) |
| `Review_2_Report.docx` | Review 2 report (DOCX) |
| `OS_Project_Final_Report.md` | Academic report source |
| `OS_Project_Final_Report.pdf` | Academic report (PDF) |
| `OS_Project_Report_Plain.md` | Plain-text report source |
| `OS_Project_Report_Plain.pdf` | Plain-text report (PDF) |

## 5. Figures (`figures/`)

- `figures/review2/` — 12 Review 2 figures (architecture, workflows, results)
- `figures/healing/` — 3 healing study figures
- `figures/study/` — 5 offline study figures

## 6. Dependencies

| File | Description |
|------|-------------|
| `requirements.txt` | Runtime dependencies (numpy, pandas, matplotlib) |
| `requirements.lock` | Pinned versions for reproducibility |
| `requirements-dashboard.txt` | Dashboard dependencies (fastapi, uvicorn) |
| `requirements-report.txt` | Report generation (reportlab, python-docx) |
| `pytest.ini` | Pytest configuration |
