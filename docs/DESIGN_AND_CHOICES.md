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

**Decision.** The verdict uses the **normalised load closure**
`(trigger_load − post_load) / max(trigger_load − reference_load, load_band)`, and the
threshold is derived as the *95th percentile of that statistic under the no-action control*
(train: *2.375 → declared 2.4*), giving a measured false-accept rate of *4.5 %* on the control.

**Why not absolute improvement.** The earlier rule (post-window load below the trace's normal
level by at least 0.5 band widths and 5 % relative) accepted *94.6 %* of no-action windows on
the training control, because a trigger is a local maximum of the monitored statistic and the
statistic decays afterwards by itself. That is a false-accept rate of ~95 % for a rule that
claims to verify recovery. The closure statistic is scale-free, causal (all three quantities
are observed before the verdict) and its null distribution is measurable, so the threshold can
be declared from data instead of taste.

**Known cost (measured, not hidden).** With that threshold the rule rejects 97.6 %
(`rule_wait`) and 89.7 % (`q_recovery`) of the interventions that the counterfactual audit
shows *were helpful*. The loop is therefore safe but forgetful: it undoes most of the
improvements it finds. Fixing this needs a matched counterfactual baseline at run time, which
a live scheduler cannot obtain from a single trace; in this study the audit supplies it
off-line. See `docs/LIMITATIONS.md` §5.

**Rejected alternative.** Making the threshold lax enough to keep the helpful changes. It was
rejected because it would certify the same *fraction of no-change windows* as successes, i.e.
the acceptance ledger would stop meaning anything. A conservative gate plus an honest audit was
chosen over a flattering gate.

## 4. The reward the learner maximizes

**Decision.** `reward = closure − accept_closure_factor − charged_cost`, where
`charged_cost` is 0 for a hold, 1 % of the reference waiting time for a configuration change
(converted into band units), and 2 % extra on a rollback.

**Why this shape.** Holding and acting are scored on the same scale, so doing nothing wins
exactly when the observed closure does not exceed the declared threshold by more than the cost
of acting. That is what makes "act only when it pays" learnable at all.

**Measured history.** An earlier version charged 5 % per change, which exceeded the best
achievable improvement of *any* action on these families (~3 % of the reference waiting time
for `set_sjf`, measured by the action-effect probe), and made every action worse than holding.
The declared 1 % is below that measured ceiling on purpose.

**Honest limitation.** The reward correlates only weakly with the measured truth
(*Pearson 0.10*, argmax agreement *15/21* in `healing_action_probe.csv`). It ranks
`set_sjf` above the no-ops and `set_round_robin` below them; it is not a calibrated value
function.

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
to SJF) or nearly so (`rule_wait_from_sjf`: +0.24 wait units).

## 7. Counterfactual audit instead of an in-run oracle

**Decision.** Every intervention is audited by replaying the same trace with the recorded
decision prefix and then holding the pre-incident configuration
(`healing/controller.py::replay_hold_counterfactual`), which is legitimate because the engine is
deterministic. `intervention_gain = hold_total_waiting − actual_total_waiting`.

**Why.** No run-time rule can attribute an effect from a single trace. The audit can, and it is
the only place in the repository where "the recovery helped" is a measurement rather than a
verdict. It is also what exposes the fairness rule (80.1 % false recoveries) and the
verification's false-reject problem.

## 8. Seed hygiene

**Decision.** Splits are disjoint by construction and seeds are declared before use; two
earlier streams are retired as evidence (`9401` from harness development, `20501` from an early
end-to-end smoke run before the gates and the verification rule were finalised). The final
study ran once on the newly declared held-out seed `20719`.

**Why.** Otherwise "held out" means nothing. The retirement is recorded in
`experiments/healing_config.py` and in `docs/METHODOLOGY.md` §2.
