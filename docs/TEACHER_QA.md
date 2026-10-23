# Teacher questions and answers

These answers distinguish the primary healing study, the non-healing runtime appendix,
and the legacy offline selector. Current direct observations and provenance are in
[MANUAL_VERIFICATION_REPORT.md](MANUAL_VERIFICATION_REPORT.md); commands and arithmetic
are in [TEACHER_LIVE_DEMO.md](TEACHER_LIVE_DEMO.md).

## Purpose, scheduling and metrics

**What is the exact problem?**
Recognize degradation of realized scheduling performance under changing synthetic load,
choose a recovery configuration, verify it causally, and avoid persisting uncertified
changes. The objective is safe adaptive behavior, not merely a lower demo mean.
`healing/controller.py` coordinates the pipeline; `scheduler/runtime.py` executes it.

**Why these four baselines?**
FCFS gives a simple FIFO control; SJF tests burst-aware waiting-time ordering; Priority
models importance-based dispatch; RR models time slicing and responsiveness. They expose
waiting/response/overhead trade-offs instead of comparing only against a weak policy.
See `scheduler/{fcfs,sjf,priority,round_robin}.py`.

**How do they differ?**
Standalone FCFS orders ready jobs by arrival/PID; nonpreemptive SJF by burst/arrival/PID;
nonpreemptive Priority by priority/arrival/PID (lower number higher by default). RR uses a
FIFO queue and configurable quantum, admitting arrivals at a slice endpoint before
requeueing its unfinished job. SRTF is not implemented. The evolving runtime queue preserves
its own FIFO tie ordering across switches, rather than reconstructing a standalone queue.

**Can a high-priority arrival interrupt a running job?**
Not under this nonpreemptive Priority implementation. RR limits service to a quantum;
other dispatches run to completion. There is no interrupt-driven real OS execution.

**How does workload shape matter?**
Burst variation makes SJF ordering useful; arrival congestion changes queue age and load;
priority/burst correlations change Priority outcomes. RR can reduce first response but
increase total waiting and PID-switch overhead. Future arrival details are not provided
to controllers. Workload generation and drift live in `workload/`.

**How are waiting, turnaround and response calculated?**
Turnaround = completion − arrival. Waiting = completion − arrival − burst, including
waiting between RR slices. Response = first service − arrival. `ProcessOutcome` in
`workload/models.py` defines these. The teacher example's RR P2 has waiting 5 but response 2.
`evaluation/metrics.py` aggregates outcomes. Utilization is CPU-busy/elapsed ×100;
throughput is completed-process count/elapsed. Units are simulated, not seconds.

**Why does reported per-trace P95 often equal maximum?**
Nearest-rank P95 is the ceil(0.95*n)-th ordered wait. At n=15 that is the 15th value,
so equality is a definition artifact. Mean per-trace P95 is not pooled process-wait P95.
The drift workflow separately computes pooled P95.

## Learning: be explicit about the subsystem

**What does reinforcement learning mean here?**
A tabular Q learner updates values for state/action pairs from reward using
`Q <- Q + alpha*(r + gamma*max Q(next) - Q)`, omitting the future term at terminal
transitions. Training explores; evaluation is deterministic and read-only. It is not a
neural network, imitation of a teacher schedule, or direct kernel scheduler training.
Shared implementation: `rl/q_learning.py`.

**What are the primary healing state and actions?**
`HealingContextEncoder` in `healing/policies.py`: diagnosis (6), excess-band category (3),
ready-count category (3), current policy (4), visible-work/reference category (3):
6×3×3×4×3 = **648 states**. `healing/actions.py` defines **eight** actions: hold, set FCFS,
SJF, RR or Priority, shrink/grow quantum, revert to last accepted. Availability masks
exclude ineffective choices; requests that do not change active behavior collapse to hold.
The demonstration's state is 205, but the rule does not choose by maximizing Q.

**What is the healing reward?**
`healing/reward.py::controller_reward`: phase-adjusted closure residual minus acceptance
boundary, minus scaled intervention cost when changed, minus a rollback penalty only for
confirmed failure. Hold has no intervention cost. Missing residual uses a neutral zero.
Inconclusive is not a confirmed failure. Reward is not the end-of-trace counterfactual gain;
that audit is offline and unavailable to the live controller.

**What is different about the runtime Q-learning appendix?**
`rl/runtime_state.py` uses ready size, completed count, median ready remaining burst,
mean ready arrival age (not accumulated waiting), and observed priority spread:
3×3×3×3×2 = **162 states**. Actions are the **four dispatch policies**, not the eight healing
actions. `scheduler/runtime.py` supplies `r=-delta_wait/q`; default gamma=1 makes total
return equal negative total wait divided by q. It updates at dispatch/quantum epochs,
not only at degradation incidents. It has no healing verification/rollback pipeline.

**What was actually demonstrated about runtime learning?**
A real 30-episode/one-seed small pipeline wrote a 648-row state/action Q export, evaluated
four baselines plus heuristic and Q on six shared workloads, and selected a six-switch
illustrative trace from validation. It used trained in-memory models; it did **not** reload
a saved CSV as a deployable model. An additional direct-controller check trained nonzero
Q values, then made 19 inference decisions without altering Q or visit counts. This
confirms learning/inference plumbing, not general performance superiority.

**How does unseen-state handling work?**
Runtime training forces exploration for wholly unseen states; greedy choices and bootstrap
use visited actions. Runtime evaluation falls back to the configured causal heuristic (or
FCFS if none is supplied) for an unseen state. Healing has its own rule fallback and
configuration availability mask. Do not conflate bootstrap visit masks with action
availability masks. See `rl/runtime_controller.py` and `healing/policies.py`.

**What is the legacy offline selector?**
It sees whole-workload features and selects one policy before running that workload.
Its terminal one-step decision is not causal within-trace adaptation. `rl/adaptive.py`,
`experiments/study.py`, and `experiments/{train,evaluate}.py` preserve this independent work.
The manual classroom demo does not rely on rerunning it.

## Monitoring, diagnosis and safety

**How is degradation detected?**
`healing/monitor.py` aggregates completed waiting and causal load intervals, maintaining
reference and dispersion. `healing/detector.py` applies reference + 1.25 bands, two suspect
evaluations, telemetry/work/load/ready gates, cooldown and an intervention cap. Detection
is a declared operational condition, not an infallible diagnosis of a hardware fault.
In the new full run, stable synthetic controls still triggered 0.70 incidents/trace;
68.6% of those traces had an incident. Do not claim zero false alarms.

**How does diagnosis work?**
`healing/diagnosis.py` applies deterministic evidence rules with precedence to arrived-ready
jobs: e.g. starvation, head-of-line blocking, priority anomaly, tail dispersion or congestion.
It records primary and additional matched labels. It is not supervised classification or
proof of a real lock fault. The priority_inversion workload is priority skew, not mutex
priority inheritance.

**How are recovery actions selected?**
Either a predeclared mapping (`RuleWaitingTimePolicy`/fairness rule), hold-only control,
or a trained Q policy. The short demo explicitly uses the waiting rule: starvation maps
to SJF. The full study labels each arm and training seed separately.

**What is verified?**
Queue-load closure is compared with a phase-aware expected no-action closure using only
trigger-visible quantities. Residual >= +0.30 is accepted, <= −0.30 rejected, otherwise
inconclusive, subject to enough observations. Default window targets five decisions and
three completions, capped at fourteen decisions; truncated/thin windows cannot certify.
`healing/verifier.py` and controller closure logic implement this.

**What does rollback mean?**
Restore the recorded last accepted/initial scheduler configuration for later dispatches;
it does not rewind simulation time or undo completed CPU work. Inconclusive changes also
roll back under the default safety posture but do not count as confirmed failures.
Two consecutive confirmed rejections pin the last accepted/safe configuration and stop
adaptation for that trace. The fresh fairness-rule replay demonstrated this with real
rejections, unchanged thresholds, and FCFS after t189.

**Why can mean waiting improve yet the intervention be inconclusive?**
The verifier has a limited online observation window; final waiting measures the completed
trace. They answer different questions. Seed 3 improves final mean versus FCFS but has
residual 0.205922 and rolls back. This is not an accepted success or failed program.

## Reproducibility, findings and limitations

**How are experiments reproducible and leakage controlled?**
Seeded NumPy generators and stable seed derivation identify streams; process-content
fingerprints verify pairing and split separation. Primary training seeds 7101–7105,
validation 8201, final test 23819 are declared in `experiments/healing_config.py`;
retired streams are rejected. Runtime appendix uses an independent design (default final
test 19301; tiny manual run uses 3001). Threshold derivation uses training controls;
evaluation is read-only. Replaying already-seen test traces for explanation does not
create a new independent test set. Wall-clock timing fields are not byte-reproducible.

**What do the retained results actually show?**
On the primary 210-workload design: mean wait FCFS 130.83778, SJF 94.65778,
rule_wait 122.38825, Q recovery 122.56406. Fresh metrics, incidents and comparisons
matched the committed tables. These are conditional synthetic-study findings, not
production OS results. The fairness rule worsens waiting while buying response improvements.

**Why can SJF outperform the proposed method?**
It starts with exact arrived burst knowledge and favorable waiting-oriented ordering;
healing starts from FCFS, waits for evidence, and often rolls back. Under estimator-enabled
drift, fixed SJF still sees truth while adaptive arms receive modelled estimates, so it
is an **oracle**, not an equal-information comparison. SJF is not a universal optimum
for arbitrary arrival patterns/objectives or starvation constraints.

**Does Q demonstrably outperform the waiting rule?**
No. The frozen Q-minus-rule mean-wait difference is approximately +0.18 with 95% interval
[−0.04,+0.48], not evidence of superiority. The tiny runtime smoke also loses to FCFS/SJF.
A successful training run or a selected learned switch is not proof of better scheduling.

**What about the one-repetition drift experiment?**
It verifies that phase-transition inputs and result/report plumbing work. Its deterministic
family-stratified intervals can be degenerate because there is only one workload per
family. Do not cite it as a well-powered statistical performance study.

**What are the main limitations and improvements?**
Synthetic workloads; one CPU; no I/O, caches, multicore or kernel integration; exact burst
assumptions; uncalibrated EWMA/noise estimator; sparse Q coverage; detector false triggers;
limited verification power and many inconclusive rollbacks. Improvements require realistic
traces, equal-information baselines, broader independent evaluation and stronger verifier
calibration—not tuning on the frozen test stream. See `docs/LIMITATIONS.md`.

**Does it replace Linux's scheduler?**
No. Python data structures simulate process arrival, dispatch and accounting. No kernel
module, scheduler syscall, real process priority control or hardware timing is implemented.

## Questions that test the implementation

1. **Show that RR waiting differs from response.** Run `scenarios.py` and explain P2's
   completion 8, burst 3, waiting 5 and first response 2; point to `ProcessOutcome`.
2. **Where is rollback visible in execution, not a label?** Run `main.py demo`; compare
   seed 3 decisions d06 and d11, then read the saved incident and service intervals.
3. **Is an accepted change necessarily globally best?** Show seed 13's local verdict,
   then compare full-study SJF and Q means/intervals.
4. **Did Q values change at inference?** Execute `runtime_learning.py`; point to its
   before/after table and visit-count checks and `RuntimeQController.observe_transition`.
5. **Can the controller see future arrivals?** Inspect `RuntimeObservation` in
   `scheduler/runtime.py` and the ready IDs/timestamps printed by the real demo.
6. **Does phase_transition actually alter data?** Run `scenarios.py`; compare the printed
   suffix bursts/priorities and preserved prefix before boundary t8.
7. **Where is an actual failed intervention?** After extended runs, execute
   `output_integrity.py`; inspect two rejected RR incidents and FCFS pinned after t189.
8. **Can you trust an existing PNG?** Compare captured action-effect before/after images
   and the regression in `tests/test_probe_schema.py`. Existence alone is insufficient.
9. **Where do these numbers come from?** Open fresh CSVs and rerun `output_integrity.py`,
   rather than quoting an unsupported performance headline.
