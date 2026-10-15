# Design and choices

Official title: **Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads**

Every non-obvious decision of the self-healing scheduler, why it was made, what it replaced,
and what measurement was behind it. Values in *italics* are derivations or measurements, not
opinions; the artifacts are named for each.

## 1. Where the healing loop lives

**Decision.** The healing loop is a *controller* on top of the existing event engine
(`healing/controller.py` implements the `RuntimeController` protocol of
`scheduler/runtime.py`), not a fork of the engine.

**Why.** The engine is already tested for scheduling semantics
(`tests/test_runtime_scheduler.py` pins it to the four reference schedulers), and keeping one
engine means the healing arms and the fixed baselines run on exactly the same code path. The
only engine change was additive: an optional per-decision `quantum` override, so a recovery
action can change the Round-Robin quantum without disturbing the reward normalization
(documented in `RuntimeActionDecision` and `RuntimeRun`).

**Rejected alternative.** A second simulator inside `healing/`. Rejected because two engines
drift, and because then the baselines would no longer be the same measurement instrument as
the treatment.

## 2. What "degradation" is defined to be

**Decision.** A degradation event is a *statistical* event: the rolling mean waiting time of
the last 8 completions exceeds `reference + 1.25 × band` for 2 consecutive evaluations,
outside a 3-decision cooldown, with at least 2 ready processes, with enough visible work to
matter, and with the queue at least as congested as the trace's own normal load.

**Why.** Each clause exists because of a measured failure without it:

* *hysteresis* (2 evaluations) stops single-window spikes from being incidents;
* the *cooldown* stops back-to-back interventions;
* the *congestion gate* (`min_load_factor = 1.0`) removes 45 % of the control's events
  (*1.52 → 0.86 events per trace* in `healing_threshold_derivation.csv`): a trace that is
  draining needs no help, and its waiting-time windows are exactly what a naive detector
  latches onto;
* the *work gate* compares visible remaining **CPU work** with the *observed mean burst*, not
  with the waiting-time reference. The earlier version compared CPU work with a waiting-time
  level — two different units — and blocked almost every intervention on traces whose jobs are
  much shorter than their waiting times.

**Rejected alternative.** A warm-up period (declare nothing before *N* completions), which is
the textbook fix for a trace that starts with a backlog. Measured on the training control:
`N = 12` → 0.072 events/trace, `N = 24` → **0.000** events/trace (the reference is then
initialized from an already-congested window, so nothing ever exceeds it again). Warm-up was
therefore rejected as a cliff, not a knob: it does not remove false triggers, it removes the
detector. The residual false-trigger rate (0.62/trace) is reported as a limitation instead.

## 3. What the verification rule measures

**Decision (revised after the verification-mismatch investigation).** Every window is judged
on the **phase-adjusted residual** of the normalised load closure
`(trigger_load − post_load) / max(trigger_load − reference_load, load_band)`: the declared
*phase model* is subtracted from that statistic, and the residual is compared with two
boundaries derived from the **no-action control** of the training split (95th percentile
*+0.263 → declared +0.30*, 5th percentile *−0.257 → declared −0.30*, rounded outward to the
declared step of 0.05). The window produces one of
three verdicts — `accepted`, `rejected` (a *confirmed* failure), `inconclusive` (insufficient
evidence) — and the controller applies the declared posture: `accepted` keeps the change,
`rejected` and `inconclusive` roll back to the last accepted configuration
(`inconclusive_rollback = true`), while only `rejected` increments the failure counter and
attaches the rollback penalty.

**Why the residual and not the raw closure.** The old rule compared the raw closure with a
single threshold, *2.4*. Measured on the audit, that rule was not a benefit test: the trigger
itself is a local maximum of the monitored statistic, and the statistic decays afterwards
whether or not anything changed (*control median closure 1.406*, which is also the median for
helpful interventions), so the old rule rolled back helpful changes and — because the reward
was tied to the same statistic — taught the learner that they were bad. Two measured facts
made the fix possible: (i) the closure of a no-action window is predictable from evidence
available **at the trigger** (the trigger's own queue level, its ready-set
size and the visible remaining work) — the declared phase model removes *80.2 %* of that
statistic's no-action variance and cuts its
residual spread from *0.806 to 0.160*; (ii) the effect of the best action on the same
statistic is *+0.152* on average (*0.95 residual sd*, `healing_action_probe.csv`).

**Why three verdicts instead of two.** With one window's evidence the honest answer is often
"cannot tell": at a 5 % false-accept budget, only *6.7 %* of the genuinely helpful
interventions clear the declared null boundary at all (`verification_discrimination` in
`healing_summary.json`), and certifying an action whose effect is 0.95 of the residual spread
would still need about *39 windows* of the same action. Collapsing that case into `rejected` would be a
false statement about the action *and* would drive the safety fallback on ambiguity, so
`inconclusive` is kept semantically distinct: it rolls the change back (the loop never keeps a
configuration it could not verify) but is not a failure, carries no rollback penalty and
cannot trigger the fallback.

**Measured, on the validation split only.** Rolling back the inconclusive windows costs about
one wait unit on the arms whose actions are usually useful (`rule_wait` 118.18 vs 117.13;
`q_recovery` 118.70 vs 117.87) and saves 11 wait units on the arm whose actions are usually
harmful (`rule_fairness` 133.40 vs 144.77), while halving the number of configuration changes
(196 vs 102 policy switches). The conservative posture was therefore kept and is declared as
`inconclusive_rollback = true`.

**Also measured, and worth stating plainly.** The revised rule accepts a population that is
enriched in benefit (*65 of 65 accepted interventions helped* on the reported split, against a
base rate of 83.9 %, and no harmful intervention was accepted) and rejects a population that is
enriched in harm (*76.5 % of rejected interventions were harmful* against a base rate of
14.0 %, a 5.5× enrichment). Its ranking is strong (rank AUC of the residual for predicting
"this intervention helped" is *0.951*, against *0.613* for the raw closure) but its
certification rate is low (6.7 % of helpful interventions). The rule is a safety gate that
filters the tails; it is not a benefit estimator.

**Why the correction was not simply a laxer threshold.**
`(trigger_load − post_load) / max(trigger_load − reference_load, load_band)`, and the
threshold is derived as the *95th percentile of that statistic under the no-action control*
(train: *2.375 → declared 2.4*), giving a measured false-accept rate of *4.5 %* on the control.

Laxer boundaries would certify no-change windows at the same rate: both operative boundaries
are percentile points of the control residual (*95th and 5th*), and their measured error is
*3.4 % / 3.1 %* on the training control and *2.1 % / 1.1 %* out of sample on the 98 control
windows of the validation split. A rule that accepted the helpful 98 % would have to accept
roughly the same share of windows in which nothing happened, and the ledger would stop meaning
anything. The conservative gate plus an honest `inconclusive` category was chosen over a
flattering gate. The route to genuinely more certified recoveries is more evidence per
decision (several windows per action), not a looser boundary — see `docs/LIMITATIONS.md`.

**Earlier rule, for the record.** The two rules before this one are documented rather than
deleted: the *absolute improvement* rule (post-window load below the trace's normal level by
0.5 band widths and 5 % relative) accepted *94.6 %* of no-action windows; the *single-threshold
closure* rule that replaced it was calibrated (false-accept 4.5 %) but treated every
non-accepted window as a failure, which is the mismatch this revision fixes.

**Truncated windows.** A window still open when the trace ends is concluded as `inconclusive`
with the reason `insufficient_evidence_window_truncated` and follows the same posture as any
other inconclusive window, so the ledger has no "censored" category and no incident is
discarded or reported with missing fields.

## 4. The reward the learner maximizes

**Decision.** `reward = residual − accept_residual_factor − charged_cost`, where `residual` is
the *same phase-adjusted statistic the verdict is taken on*, `charged_cost` is 0 for a hold and
1 % of the reference waiting time for a configuration change (converted into band units), plus
the declared rollback penalty *only* when the window is a **confirmed failure** (`rejected`).

**Why this shape.** The reward and the verdict are the same measurement, so the learner cannot
be rewarded for something the verifier could not see — and, symmetrically, it is no longer
punished for a change the verifier merely could not confirm: an `inconclusive` window charges
the action's cost and nothing else. Holding is scored on exactly the same scale, so doing
nothing wins when the statistic does not exceed the declared boundary by more than the cost of
acting.

**No future information.** The reward uses only quantities observed inside the window that just
closed: the window's own closure statistic, the trigger-phase evidence, and the declared
constants. The counterfactual replays that produce `matched_gain` are computed *after* the
study and are never visible to the controller (`healing/controller.py` has no access to the
workload, to future arrivals or to the audit). A test pins that a later arrival cannot change
an incident that closed before it.

**Measured history.** An earlier version charged 5 % per change, which exceeded the best
achievable improvement of *any* action on these families (~3 % of the reference waiting time
for `set_sjf`, measured by the action-effect probe), and made every action worse than holding.
The declared 1 % is below that measured ceiling on purpose.

**Honest limitation.** The reward is a weak proxy for the *size* of the effect: across all
probed rows its pooled Pearson correlation with the measured effect is *0.10*, and the pooled
number understates it because different traces have different scales. What matters for
learning is the ranking *within* an incident: mean within-incident rank correlation *0.622*
(median *0.875*), and `argmax(reward)` is the truly best action in *15 of 21* incidents
(`healing_action_probe.csv`). Five of the eight actions (`hold`, `set_fcfs`, `shrink_quantum`,
`grow_quantum`, `revert_to_last_accepted`) have exactly zero effect from the declared FCFS
start, so the real choice is among three actions, and `set_sjf` is the best of them in 15 of
21 probed incidents.

## 5. What the recovery actions may touch

**Decision.** One configuration: the dispatch policy and the Round Robin quantum. Eight
actions, of which `hold` and `revert_to_last_accepted` are safety-oriented.

**Masking.** Before the policy is asked, `available_actions(...)` removes every action that
cannot change the live configuration: quantum actions under a non-Round-Robin policy, set-policy
actions that are already in force, and revert when there is no different accepted configuration.
A rule arm asked for a masked-out action falls back to its declared alternative and finally to
`hold`, and records why. *This was a real defect*: `MappedRulePolicy.select` originally ignored
the mask, so the fairness rule spent incidents requesting quantum changes that collapsed to
no-ops (54 `shrink_quantum` requests in an early smoke run), which inflated the "intervention"
count with changes that could not happen.

**Rejected alternative.** A broader action space (I/O priority, affinity, preemption
thresholds). Rejected as unmodelled: the simulator has no such mechanisms, so the actions
would be names without effects.

## 6. The comparison set

**Decision.** Ten predeclared comparisons over eleven arm definitions: four untouched fixed
schedulers, a detection-only control, two predeclared rules, and learned controllers (five
seeds) with and without SJF as the initial configuration.

**Why.** The control answers "does acting help?"; the fixed schedulers answer "is the loop
better than the best static choice?"; the `*_from_sjf` arms answer "does the loop damage an
already good configuration?", which is the most realistic deployment question and, as it turns
out, where the loop looks best: it is inert (`q_recovery_from_sjf`: 0 interventions, identical
to SJF) or nearly so (`rule_wait_from_sjf`: +0.28 wait units).

## 7. Counterfactual audit instead of an in-run oracle

**Decision.** Every intervention is audited by a **matched replay**: the same controller, on
the same workload, with the same recorded decision prefix, forcing *that one incident* to hold
(`forced_holds`) while every later decision is taken as usual
(`experiments/healing_experiment.py::_audit_episode`). The prefix identity is checked
(`divergence_decision_index` must equal the audited incident's decision index), so the
counterfactual is causally anchored at the decision. `matched_gain = hold_total_waiting −
actual_total_waiting`. A second, separately reported counterfactual replays the prefix and then
never heals again (`hold_forever_gain`, one row per episode).

**Why.** No run-time rule can attribute an effect from a single trace. The audit can, and it is
the only place in the repository where "the recovery helped" is a measurement rather than a
verdict. It is also what exposes the fairness rule (*80.9 % false recoveries*) and the
verifier's inability to certify ordinary improvements.

**Denominators.** The two counterfactuals answer different questions and are reported with
their own denominators: every rate in `healing_audit.csv` and in the per-arm summary is a
*per-intervention* quantity (the false-recovery rate is the share of audited interventions with
`matched_gain ≤ 0`), and the never-heal rows are aggregated per *episode* in their own table.
The two views agree closely on the mean gain (`rule_wait` +178.78 per intervention vs +188.85
per episode), which is the check that neither is an artefact of its denominator.

## 8. Seed hygiene

**Decision.** Splits are disjoint by construction and seeds are declared before use. Five
streams are retired as evidence: `9401` (harness development), `20501` (an early end-to-end
smoke run), and `20719`, `21617` and `72601` — each of which was declared as a held-out seed,
evaluated once, and then *examined* while the verification rule, the reward and the audit
denominators were reworked. Substantive changes after looking at a stream invalidate that
stream as evidence, and `21977` is retired the same way after the phase reference was replaced
by the physical observation row; `23819` was declared before the final evaluation
of the frozen system and is refused as a development seed by the configuration's own
validation (`HealingExperimentConfig.__post_init__`).

**Why.** Otherwise "held out" means nothing. The retirement is recorded in
`experiments/healing_config.py` and in `docs/METHODOLOGY.md` §2.
