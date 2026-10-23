# Lecturer demonstration script (5–8 minutes)

## 1. Before the presentation

Complete `docs/WSL_SETUP.md` once. In Windows PowerShell, confirm `wsl --list --verbose`
shows version 2. In the Ubuntu terminal:

```bash
cd "$HOME/OS-Project"
source .venv/bin/activate
python --version
python -m pip check
git branch --show-current
python main.py --help
python -m pytest -q tests/test_demo.py
```

Use the delivered branch `arena/49fb0ff9-os-project`. The tested host was Debian 12,
Python 3.11.2; actual Windows WSL execution remains unverified. Run the demo once
before presenting, enlarge terminal text, and have this guide and the source open.
After installation the demo needs no network. Do not run `python main.py` alone:
its default is the expensive full study.

## 2. The one demonstration command

```bash
python main.py demo
```

This is a real execution, not a playback of `results/healing/healing_demo.json`.
It uses `RuntimeSimulator`, `SelfHealingController`, `HealingContextEncoder`, the
actual workload generator, waiting-time rule and metrics. It generates 15 processes
for each of two illustrative `mixed` seeds, 3 and 13. Those seeds were selected for
clear contrasting verdicts, not for an unbiased performance estimate. It uses the
unchanged study defaults, including quantum 4 and switch cost 1. It fits no model.

## 3. What to say and show

### Minute 0–1: scope and input

Say: “The official title is Adaptive CPU Scheduling Using Reinforcement Learning for
Dynamic Workloads. This checkout's primary implementation extends runtime scheduling
with a self-healing control loop. I will show that loop using the predeclared waiting-time
rule so we do not need a long training run. Q-learning is a separate recovery policy
in the same controller, evaluated in the frozen experiment.”

Point to `INPUT mixed seed=3` and the process list. Explain `(arrival, burst, priority)`.
These are simulated time units. Exact burst lengths are known after arrival, an explicit
simplification. Fingerprints identify inputs; no future process is passed to the controller.

### Minute 1–2: actual scheduling

Point to a decision line: `t` is the observation time, `ready` lists arrived unfinished
PIDs, the named policy chooses a PID, and `[start,end)` is its service interval.
`switch=1` accounts for changing the running process. Idle time is handled by the engine.
`wait+=` is the interval's total ready-queue waiting accrual, not elapsed wall-clock time.
The JSON preserves the complete observation, including remaining bursts and priorities.

Explain: “FCFS and RR dispatch the queue head. SJF chooses the shortest remaining job.
Priority chooses the configured highest priority. Non-RR segments are non-preemptive;
RR stops after a quantum or completion. A policy change is not itself a PID context switch.”

### Minute 2–3: monitor, incident, state and action

Scroll to the first concluded incident. The monitor sees completed-process waiting times,
not predicted final metrics. The trigger compares the recent completion-window mean
with reference plus 1.25 dispersion bands, after two suspect evaluations and the gates.
The ledger is printed after execution; its timestamp tells when the event actually occurred.

For seed 3, the trigger is at decision 6/time 141: mean wait 51.000 exceeds 28.203.
The primary diagnosis is `fairness_starvation`; other matched evidence remains recorded.
This is a deterministic diagnosis label, not proof of an underlying kernel fault.

Point to bins `[1, 2, 2, 0, 1]` and state 205. The five dimensions are diagnosis,
excess over threshold, ready size, current policy, and visible work relative to reference
waiting time. Their radices are 6, 3, 3, 4, 3: 648 states, eight recovery actions.
The rule selects `set_sjf`; it does not maximize Q values in this demonstration.
Even the rule run records the encoded context to make its evidence inspectable.

### Minute 3–4: verification and rollback

For seed 3, SJF is applied at decision 6. Five later decisions/completions provide a
verification window. Its phase-adjusted closure residual is approximately 0.206:
between −0.30 and +0.30, hence `inconclusive`, not `rejected` and not “successful”.
At time 193 the controller rolls back to FCFS; the later decision trace shows it.
Inconclusive does not increment confirmed failures. It means insufficient evidence
to certify the intervention under the declared rule, even though the final mean improves.

For seed 13, the same rule triggers at time 243. The residual is approximately 0.437,
so the verifier reports `accepted` and SJF remains in force after time 319.
Do not promise a rejection in these two examples: neither produces one. A rejection
would require residual at most −0.30 with sufficient evidence; it rolls back and counts
as a confirmed failure. Two consecutive confirmed failures disable further adaptation.
Controller tests separately exercise that path.

### Minute 4–5: measurements and artifacts

Point to both summaries. Waiting per process is completion minus arrival minus burst;
response is first service minus arrival. Mean waiting is the sum divided by 15.
The engine also integrates waiting over service and switching intervals, so the sum of
`wait+=` equals total process waiting. Utilization is 100 times service busy time divided
by simulated elapsed time; switching and idle time do not count as busy CPU service.

`results/demo/demo.json` stores both inputs, full observations, decisions, incidents,
per-process outcomes, metrics and separately executed FCFS metrics on the same inputs.
It is disposable and can be regenerated; it does not replace study evidence.
The negative reported mean-wait difference favors this rule on that fixture only.

### Minute 5–6: Q-learning and the actual findings

Open `healing/policies.py` and `healing/reward.py` if needed. The learned alternative
chooses among available recovery actions using epsilon-greedy training and a tabular
Q update. It learns at incident verification, not from the runtime engine's per-dispatch
waiting-cost reward. Greedy evaluation is read-only, considers learned available actions,
and holds for an unseen context. The rule does not need a learned state to act.

Reward = residual − accept boundary − changed-action cost − confirmed-failure penalty.
The costs use reference waiting time converted into band units. An inconclusive result
is not automatically zero reward. An offline raw queue-load proxy is not this reward.

Quote only the frozen healing study: seed 23819, 210 traces per arm, five trained models.
Rule mean waiting 122.39; Q recovery 122.56; detection-only FCFS 130.84; fixed SJF 94.66.
Q minus rule is +0.18 with 95% interval [−0.04,+0.48]. Say: “No RL advantage is established;
recovery helps relative to FCFS, but neither reaches fixed SJF.” Do not present either
small fixture as evidence for those aggregate claims.

### Minute 6–8: questions and limits

Use the answers below. Finish by saying: “This is an auditable simulator study, not a
Linux kernel scheduler, and it has not been tested on actual WSL in this delivery.”

## 4. Source files to open

- `experiments/demo.py`: two fixture executions and printing; no alternate scheduler.
- `main.py`: `demo` dispatch, output error handling, other study commands.
- `scheduler/runtime.py`: observations, live queue, service, switching, waiting accrual.
- `healing/monitor.py` and `healing/detector.py`: telemetry, threshold and suppression gates.
- `healing/diagnosis.py`: evidence labels and precedence.
- `healing/policies.py`: state encoding, rule maps, masked Q-learning and unseen holds.
- `healing/controller.py`: incident lifecycle, verification, reward delivery, rollback.
- `healing/verifier.py` and `healing/reward.py`: phase baseline, verdict and shared reward.
- `rl/q_learning.py`: tabular update and visit counts.
- `evaluation/metrics.py`: final schedule measurements.
- `experiments/healing_config.py`: primary frozen seeds and settings.
- `results/healing/healing_summary.json`: frozen aggregate results, not demo metrics.

## 5. Likely questions and accurate answers

- **How is an action chosen?** The live loop gates detection, diagnoses visible evidence,
  then calls a recovery policy with masked available actions. This demo uses the declared
  waiting-time map. The learned arm uses greedy learned Q values in evaluation.
- **What does the controller know?** Arrived unfinished jobs, exact arrived burst estimates,
  priorities, time, counters and realized completions. Not future arrivals or final metrics.
- **How is waiting counted?** Per-process completion − arrival − burst, also integrated
  interval by interval for every waiting process, including switching delays and arrivals
  during a segment. Context switches are changes of PID, not every policy decision.
- **What role does Q-learning play here?** It is an implemented alternative recovery
  policy evaluated in the full study. It is not trained or used by this short rule demo.
- **What if the state is new?** Healing Q evaluation explicitly holds when no available
  action has been visited. During training an unseen context forces random exploration
  among available actions. The separate runtime Q controller instead uses a causal heuristic
  fallback; do not confuse those two behaviors.
- **How are fixed policies compared?** Identical generated traces, standalone baselines,
  paired workload fingerprints, independent model seeds and family-stratified crossed
  bootstrap intervals. Five model rows are not five different workloads.
- **What does the experiment establish?** Conditional improvements over FCFS/control,
  no demonstrated Q advantage over the rule, and inferior waiting relative to fixed SJF.
- **Why roll back something helpful?** An offline counterfactual can reveal benefit that
  a short online window cannot certify. Inconclusive is not synonymous with harmful.
- **Is this adapting to arbitrary real workload shifts?** No. Primary traces have synthetic
  family distributions; injection scenarios test specific degradation cases. No real traces,
  multicore, I/O, deadlines, burst prediction errors or kernel integration are evaluated.
- **Why not show the reference runtime scheduler instead?** Source and README designate
  healing as primary. The preserved runtime workflow has 162 states, four actions, its own
  seed 19301 and separate results. The absent `OS(2).pdf` could not be inspected here.
- **Were the results regenerated?** Not the expensive study. This delivery ran the quick
  demo and tests. Historical probe, canonical probe and final-test evidence were preserved.

## 6. Actual captured output

Full capture: `docs/examples/demo_linux_output.txt`, command `python main.py demo`
inside the activated Python 3.11.2 environment on Debian 12 Linux. Excerpt:

```text
SUMMARY mean wait=130.467; FCFS same input=150.600; difference=-20.133
SUMMARY mean wait=220.533; FCFS same input=255.800; difference=-35.267
```

These are fixture results, not averages from the research experiment. Exact full precision
values and incident evidence are regenerated in `results/demo/demo.json`.

## 7. Recovery during the presentation

```bash
cd "$HOME/OS-Project"
source .venv/bin/activate
which python
python --version
python -m pip check
python main.py demo --results-dir "$HOME/os-demo-output"
```

If a package is missing, install `requirements.lock` before continuing. If offline and the
environment cannot be repaired, explain the limitation and use the explicitly labelled
captured output as a fallback, not as a claimed live execution. A permission error is fixed
by using a writable output directory. Do not change thresholds, final-test seeds or study
artifacts to repair a presentation. The optional report build is not needed for the demo.
