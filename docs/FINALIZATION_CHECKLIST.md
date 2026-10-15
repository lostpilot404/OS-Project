# Finalization checklist

Statuses: **Complete**, **Needs Fixing**, **Blocked**, **Not Verified**.

Official title preserved: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic
Workloads**. Everything below was verified by running the commands in this repository in this
session; nothing is copied from an earlier report. Where a number is quoted it is the output of
the command listed next to it.

## 1. Starting state of this round (verified, not assumed)

| Item | Verified value | Status |
|---|---|---|
| Repository | `/home/user/OS-Project`, remote `https://github.com/lostpilot404/OS-Project.git` | Complete |
| Branch | `arena/befa2b94-os-project` | Complete |
| Commit at the start of the round | `3866faf506c75ab959a219c88ef4de864be34e21`, identical to `origin/arena/befa2b94-os-project` | Complete |
| `main` | untouched at `c26e2a47b637128400f90fe10dec1ce9cf473270` (`origin/main`) | Complete |
| Environment | `.venv` on Python 3.11.2 with numpy 2.4.6, pandas 3.0.6, matplotlib 3.11.2, pytest 9.1.1; `requirements.lock` unchanged in this round | Complete |
| Work carried over | the previous round left the repaired loop, the harness and most of the documentation uncommitted in this work tree; this round finished the propagation, re-derived nothing and re-ran everything from the frozen source | Complete |

## 2. The verification mismatch: cause, fix, evidence

| Question from the task | Answer | Where it is recorded |
|---|---|---|
| Which metric/reference does the online verifier use? | the *phase-adjusted residual* of the normalised load closure, against two boundaries derived from the no-action control of the **training** split (+0.30 / −0.30) | `healing/verifier.py`, `healing/config.py`, `results/healing/healing_verification_models.csv` |
| Does it compare against a baseline for the same workload phase? | yes — a declared 4-coefficient structural reference (`post_load_baseline = −2.422506, −0.008145, 0.998700, −0.082333`) predicting the *no-action* post-window queue level from the trigger's load level, ready-set size and visible work; it explains 96.06 % of the post-load variance and cuts the residual spread from 0.806 to 0.160 | `healing/config.py::post_load_baseline`, `healing/verifier.py::expected_closure`, `healing_threshold_derivation.csv` |
| Does the short window capture the action's effect, or reject on transient queue changes? | both, and the study measures how much: the best action moves the statistic by +0.152, ≈ 0.95 of the residual spread that one window shows, so ≈ 39 paired windows would be needed for 80 % power. A five-decision window can certify a minority of genuinely helpful interventions; that limit is reported, not hidden behind a lower boundary | `healing_action_probe.csv::paired_closure_change`, `healing_summary.json::verification_discrimination` |
| Are short-term waiting changes confused with the final total-waiting benefit? | they were: the earlier single-threshold rule called every non-accepted window a failure and rolled helpful actions back. The matched-replay audit (`matched_gain`) measures the final benefit and is now cross-tabulated against the verdict per arm | `docs/EVALUATION_RESULTS.md` §3–§5, `results/healing/healing_audit.csv` |
| Does a helpful action get a negative RL reward? | no: the reward is the verifier's own residual minus its accept boundary, plus the action cost when the configuration changed; the rollback penalty is charged **only** on a confirmed failure. An inconclusive window charges the action cost and nothing else | `healing/controller.py::_conclude_pending` |
| Do online verification and the offline audit measure different things? | they measure different quantities — one closed window of evidence online, a matched counterfactual replay offline — and the disagreement is quantified rather than assumed away | `docs/EVALUATION_RESULTS.md` §4.1, §5 |
| Can verification improve without leaking future outcomes? | the verdict and the reward read only quantities observed inside the closed window; a test pins that an arrival after the window closed cannot change the incident's verdict or reward | `tests/test_healing_verification.py`, `tests/test_healing_controller.py` |
| Is a *richer* phase reference justified? | no, within the measured uncertainty. Against the declared reference, on the 700 judged validation interventions (559 helpful, 117 harmful): the two-load-geometry reference and the earlier free closure fit certify **fewer** helpful interventions (paired difference −0.084 [−0.169, −0.005] and −0.084 [−0.168, −0.004], 2000-draw cluster bootstrap); a ready-count-only reference (+0.021 [−0.021, +0.065]) and clock-augmented references (+0.000) include zero; a monitor-window-augmented reference is +0.043 [+0.010, +0.087] and was **not** adopted, because it conditions the expectation on the monitor's own window maturity rather than on the workload phase, and the declared reference already certifies 15.0 % of helpful windows with 0 of 117 harmful ones | `results/healing/healing_verification_models.csv` (`paired_delta_*` columns), `docs/METHODOLOGY.md` |

**Corrections implemented (smallest defensible, no threshold lowered to accept more):**

1. tri-state verification (`accepted` / `rejected` / `inconclusive`) with a declared posture for
   the middle case (`inconclusive_rollback = true`: rolled back, never counted as a failure,
   never triggering the safety fallback, charging only the cost of the attempted change);
2. a truncated window at the trace end is concluded as `inconclusive`
   (`insufficient_evidence_window_truncated`) instead of being logged as `censored` with NaN;
3. the reward uses the verifier's own phase-aware statistic, so learning cannot be rewarded for
   something the verifier could not see;
4. the audit aggregate no longer mixes intervention rows with one-per-episode "never heal
   again" rows in one denominator; both are reported in their own tables with their own `n`;
5. verification discrimination (rank AUC, certification ceiling, enrichment with denominators)
   and the paired uncertainty of the candidate comparison are machine-readable artifacts rather
   than narrative (`healing_verification_models.csv`).

## 3. Verification commands and outcomes (this session)

```bash
cd /home/user/OS-Project
.venv/bin/python -m pytest -q                      # 472 passed in 50.20 s, exit 0
.venv/bin/python main.py healing-study             # exit 0 (frozen run, declared held-out seed 23819)
.venv/bin/python main.py healing-study --results-dir /tmp/det3/results \
                                       --figures-dir /tmp/det3/figures
                                                   # exit 0 (independent regeneration)
.venv/bin/python -m experiments.healing_thresholds # exit 0 (derivation only)
.venv/bin/python -m compileall -q healing experiments evaluation rl scheduler workload main.py
                                                   # exit 0 (no output)
git diff --check                                   # clean, exit 0
```

Clean-checkout verification (fresh clone of this branch, fresh environment):

```bash
git clone <repository> /tmp/clean-check && cd /tmp/clean-check
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.lock   # exit 0
.venv/bin/python -m pytest -q                                                   # 472 passed in 51.83 s, exit 0
.venv/bin/python main.py healing-study --results-dir /tmp/clean/results \
                                       --figures-dir /tmp/clean/figures         # exit 0
diff -r /tmp/clean/results results/healing                                      # 0 lines
diff -r /tmp/clean/figures figures/healing                                      # 0 lines
```

So the committed artifacts are not only reproducible in the tree they were produced in: a clean
clone of the pushed branch rebuilds them byte for byte. `requirements.lock` is not modified in
this round, and every command above was run from the frozen source.

Determinism: the independent regeneration reproduced **all 21 artifacts** under
`results/healing/` and **all 3 figures** under `figures/healing/` byte for byte (`diff -r` over
both directories: 0 lines). An additional canonical re-run changed exactly one file — the
artifact whose columns the round extended — and left the other 23 byte-identical. Note that
`--no-probe` legitimately changes `healing_action_effects.png` (the probe is its only source),
so the determinism check is run with the probe enabled.

Targeted evidence quoted in this round (frozen run, held-out seed `23819`):

| Measurement | Value | Artifact |
|---|---:|---|
| accepted interventions that helped | 65 / 65 (100 %; base rate 83.9 %) | `healing_summary.json::verification_discrimination` |
| rejected interventions that were harmful | 26 / 34 (76.5 %; base rate 14.0 %) | same |
| rank AUC of the residual → helped | 0.9514 (raw closure 0.6130) | same |
| helpful interventions that clear the declared accept boundary | 6.65 % (54 / 812) | same |
| paired change of the statistic for `set_sjf` | +0.152 (≈ 0.95 residual sd) | `healing_action_probe.csv` |
| windows of the same action needed for 80 % power | ≈ 39 | derived in `docs/EVALUATION_RESULTS.md` §4.1 |
| safety fallback episodes | 1 on the held-out split (`rule_fairness`, repetition 21, episode 201, decision 20, 2 consecutive failures) and 1 on validation | `healing_detection_summary.csv`, `healing_incidents_final_test.csv` |
| detector false triggers | 0.62 events/trace on the training control, 0.65 on the held-out control, 0.70 per stable trace in the injection probe | `healing_threshold_derivation.csv`, `healing_detection_summary.csv` |
| Q state-action coverage | 29–33 of 648 states visited: 1.1–1.3 % of the declared 5184-pair table, 3.8–7.2 % of the pairs reachable from the observed contexts | `healing_coverage.csv`, `healing_coverage_curve.csv` |

## 4. Pipeline stage status (all ten stages)

| Stage | Implementation | Test | Status |
|---|---|---|---|
| 1. Reproducible workloads | `workload/generator.py` | `test_generator.py`, split-disjointness test | Complete |
| 2. Event simulation | `scheduler/runtime.py` | `test_runtime_scheduler.py` (23) | Complete |
| 3. Causal telemetry | `healing/monitor.py` | `test_healing_monitor.py` (8) | Complete |
| 4. Detection | `healing/detector.py` | `test_healing_detector.py` (14), `test_healing_detection.py` (6) | Complete |
| 5. Diagnosis | `healing/diagnosis.py` | `test_healing_diagnosis.py` (11) | Complete |
| 6. Recovery decision (rule / Q, masked) | `healing/policies.py`, `rl/q_learning.py` | `test_healing_policies.py` (8) | Complete |
| 7. Apply the action | `healing/actions.py`, `healing/controller.py` | `test_healing_controller.py` (10) | Complete |
| 8. Verify over a declared window | `healing/verifier.py` | `test_healing_verification.py` (18) | Complete |
| 9. Accept / roll back / safe fallback | `healing/controller.py` | verification + controller tests | Complete |
| 10. Record decision, evidence, outcome | `IncidentRecord`, `EpisodeRecord`, artifacts | `test_healing_experiment.py` (11) | Complete |

## 5. Experimental integrity

| Requirement | Status | Evidence |
|---|---|---|
| Splits disjoint by construction | Complete | `SPLIT_TAGS` + `derive_seed`; fingerprint test |
| Held-out seed declared **before** the frozen evaluation | Complete | `23819` in `experiments/healing_config.py`; `20501`, `20719`, `21617`, `72601`, `21977` retired in the same file and refused as final-test seeds by validation |
| No parameters chosen on the final-test split | Complete | `healing_thresholds.py` refuses the final-test split; the reference and the boundaries come from the training control and are checked out of sample on the validation control (false accept 0.0211, false reject 0.0105); the gate and posture were chosen on training/validation |
| Identical workloads across arms | Complete | one workload builder; paired statistics keyed by fingerprint |
| Paired comparisons with uncertainty | Complete | `evaluation/paired.py`, 1000 replicates, `healing_comparisons.csv`; the candidate-reference comparison adds a 2000-draw cluster bootstrap (`healing_verification_models.csv`) |
| Every intervention causally audited | Complete | matched replay (`forced_holds`) + `divergence_decision_index` check |
| No future information in the online verdict or reward | Complete | `tests/test_healing_verification.py` (later arrivals cannot change a closed incident) |
| Unfavourable results reported | Complete | SJF beats every recovery arm; the fairness rule is harmful (false-recovery rate 0.767); only 6.65 % of helpful interventions can be certified; the detector's false-trigger frequency and the Q-coverage ceiling are published |
| No claim of RL superiority | Complete | README §8, `EVALUATION_RESULTS.md` §2–4, `LIMITATIONS.md` §3 |
| Reproducible | Complete | 21 artifacts and 3 figures byte-identical across independent runs (§3) |

## 6. Completed / Needs Fixing / Blocked / Not Verified

| Item | Status |
|---|---|
| Recovery-verification mismatch diagnosed, corrected and documented | **Complete** |
| Truncated-window and audit-denominator defects fixed, with tests | **Complete** |
| Detector covered by controlled injection tests (recall, delay, false triggers, no self-trigger) | **Complete** |
| Q-coverage analysed against observed-reachable state-action pairs, with a coverage curve | **Complete** (analysis) |
| Counterfactual metrics mathematically consistent (denominators separated per intervention/episode) | **Complete** |
| Docs match the final source and the regenerated artifacts | **Complete** |
| Safe fallback exercised on a full trace | **Complete** — fired once in `rule_fairness` on the held-out split (and once on validation); the branch is also unit-tested |
| The loop does not beat fixed SJF; the RL arm does not beat the rule | **Needs Fixing** (research-level: action set, workload families and reward informativeness, not a code defect) |
| Only 6.65 % of genuinely helpful interventions can be certified; 91.0–92.1 % end inconclusive | **Needs Fixing** (needs more evidence per decision — several windows per action or a better phase reference; measured, not implemented) |
| Detector false triggers (0.62/trace on the training control, 0.65 on the held-out control, 0.70 per stable trace in the probe) | **Needs Fixing** (documented trade-off with the congestion-gate sweep: raising the gate to 1.25 removes the false triggers and with them almost all true detections, 0.042 events/trace) |
| Tabular-Q coverage (1.1–1.3 % of the declared table and 3.8–7.2 % of the reachable pairs, flat after ~900 episodes) | **Needs Fixing** (event supply, not training budget; reported rather than replaced by a more complex algorithm) |
| Real-OS integration, burst prediction, non-stationary arrivals | **Not Verified** (out of scope; no claim made) |
| Behaviour on hardware or real traces | **Not Verified** (simulator only) |
| Anything requiring more compute or external data than this environment provides | **Blocked** (none identified for the reported results) |

## 7. Repository hygiene (this round)

| Check | Result |
|---|---|
| Complete diff inspected before staging | Complete (38 files: 19 source/doc files and 19 artifacts; §8 records the commit) |
| `git diff --check` | clean, exit 0 |
| Formatting / linting / static analysis | the repository configures none (`pytest.ini` only); `.venv/bin/python -m compileall -q healing experiments evaluation rl scheduler workload main.py` exits 0 |
| Temporary files, debug output, caches, `.venv` | none tracked; the model-choice scratch ran outside the repository, and the numbers it produced are reproducible from `healing_verification_models.csv` |
| Secrets / `.env` / credentials | none |
| Machine-specific or absolute paths in tracked files | only inside `docs/TRANSFORMATION_AUDIT.md`'s record of the session environment, which is documentation |
| Unused dependencies | none added this round; `requirements.lock` unchanged (numpy/pandas/matplotlib/pytest) |
| Documented install and commands verified | Complete — fresh clone of this branch, fresh `.venv` from `requirements.lock`, full suite 472 passed, and the study regenerates the committed artifacts byte for byte (§3) |

## 8. Git state at the end of the session

| Item | Value |
|---|---|
| Working branch | `arena/befa2b94-os-project` |
| Commit at the start of the round | `3866faf506c75ab959a219c88ef4de864be34e21`, identical to `origin/arena/befa2b94-os-project` |
| Source, tests, docs and artifacts | `64036348c250246856bb359acf070e833b0518a6` — "Repair recovery verification, prove it, and freeze the study on a declared held-out seed" |
| Push | `git push origin arena/befa2b94-os-project` → `3866faf..6403634` (fast-forward, no force, no history rewrite), exit 0 |
| Remote verification | `git ls-remote origin refs/heads/arena/befa2b94-os-project` and `gh api repos/lostpilot404/OS-Project/branches/arena/befa2b94-os-project --jq .commit.sha` both report `64036348c250246856bb359acf070e833b0518a6`, identical to the local HEAD; `git rev-list --left-right --count` reports 0 behind / 0 ahead |
| Checklist record commit | `f19dab3640d8f11bd253392217077cc853b92472` — this file only, pushed as `6403634..f19dab3` (fast-forward, exit 0) |
| Documentation corrections after that record | `de1085a` — "Correct four doc claims that disagreed with the regenerated artifacts": the fallback episode, the demonstration trace's repetition, the judgeable out-of-sample denominator and two stale numbers in `LIMITATIONS.md`, plus the paired-uncertainty paragraph in `METHODOLOGY.md`; pushed as `f19dab3..de1085a` (fast-forward, exit 0) |
| Final verification | after those pushes `git ls-remote` reports the branch head equal to the local tip and `git ls-remote origin refs/heads/main` still reports `c26e2a47b637128400f90fe10dec1ce9cf473270`; the commit that contains this row is the branch tip and its SHA is quoted in the session report |
| `main` before the merge | untouched at `c26e2a47b637128400f90fe10dec1ce9cf473270`; it is an ancestor of this branch, so the merge below is a clean fast-forwardable merge |
| Merge into `main` | authorised by the project owner in this session and performed through a pull request from this branch with `gh pr merge --merge` (no force, no history rewrite). The merge commit and the resulting `main` head are quoted in the session report (`git log --merges -1 origin/main`). |
| Other pull requests | pull request #5 belongs to another session's branch (`arena/5e0b2eac-os-project`) and is left open and untouched; nothing was closed in this round |

`results/healing/` (21 files) and `figures/healing/` (3 files) in the commit above are the
artifacts of the frozen run on the declared held-out seed `23819`, regenerated after the last
source change and reproduced byte for byte by an independent run. The `21977` tables that remain
in `docs/EVALUATION_RESULTS.md` §10.1 and in `docs/LIMITATIONS.md` are labelled historical: that
stream was examined while the structural phase reference was chosen and is retired, not evidence
for the current rule.
