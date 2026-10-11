"""Shared data layer for the Review 2 PDF and DOCX builders.

Everything the reports show is loaded from the committed experiment artifacts
under ``results/`` (and, for the dataset statistics, from the project's own
seeded generator), so the two documents can never disagree with each other or
with the recorded experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHOD_ORDER = [
    "FCFS",
    "SJF",
    "Round Robin",
    "Priority",
    "Causal heuristic",
    "Runtime Q-learning",
]
SHORT_NAME = {
    "FCFS": "FCFS",
    "SJF": "SJF",
    "Round Robin": "RR",
    "Priority": "Priority",
    "Causal heuristic": "Heuristic",
    "Runtime Q-learning": "Runtime Q",
}


class Artifacts:
    def __init__(self, root: Path, with_dataset: bool = True):
        self.root = root
        runtime = root / "results" / "runtime"
        self.summary = json.loads((runtime / "runtime_summary.json").read_text())
        self.final = pd.read_csv(runtime / "runtime_final_test_metrics.csv")
        self.val = pd.read_csv(runtime / "runtime_validation_metrics.csv")
        self.train = pd.read_csv(runtime / "runtime_training_metrics.csv")
        self.paired = pd.read_csv(runtime / "runtime_paired_comparisons.csv")
        self.coverage = pd.read_csv(runtime / "runtime_state_action_coverage.csv")
        self.q_table = pd.read_csv(runtime / "runtime_q_table.csv")
        self.manifest = pd.read_csv(runtime / "runtime_workload_manifest.csv")
        self.demo = json.loads((runtime / "runtime_learned_switch_demo.json").read_text())
        self.legacy = json.loads((root / "results" / "summary.json").read_text())
        self.cfg = self.summary["configuration"]
        self.families = list(self.cfg["validation_and_test_families"])
        self.workloads: pd.DataFrame = pd.DataFrame()
        self.processes: pd.DataFrame = pd.DataFrame()
        if with_dataset:
            self._load_dataset()

    def _load_dataset(self) -> None:
        """Rebuild the held-out dataset with the project's own seeded generator."""
        from experiments.runtime_config import RuntimeExperimentConfig
        from experiments.runtime_experiment import build_runtime_workloads

        items = build_runtime_workloads("final_test", RuntimeExperimentConfig())
        workload_rows = []
        process_rows = []
        for item in items:
            workload = item.workload
            workload_rows.append(
                {
                    "family": item.family,
                    "repetition": item.repetition,
                    "seed": item.seed,
                    "fingerprint": workload.fingerprint,
                    "size": workload.size,
                    "total_burst": workload.total_burst_time,
                    "mean_burst": workload.mean_burst_time,
                    "max_arrival": workload.max_arrival_time,
                    "arrival_span": workload.arrival_span,
                }
            )
            for process in workload.processes:
                process_rows.append(
                    {
                        "family": item.family,
                        "repetition": item.repetition,
                        "fingerprint": workload.fingerprint,
                        "pid": process.pid,
                        "arrival_time": process.arrival_time,
                        "burst_time": process.burst_time,
                        "priority": process.priority,
                    }
                )
        self.workloads = pd.DataFrame(workload_rows)
        self.processes = pd.DataFrame(process_rows)

    def mean_sd(self, split: str, method: str, metric: str) -> Tuple[float, float]:
        frame = self.final if split == "final" else self.val
        values = frame.loc[frame["method"] == method, metric].to_numpy(dtype=float)
        return float(values.mean()), float(values.std(ddof=1))

    def summary_mean(self, split: str, method: str, metric: str) -> float:
        key = "final_test_means" if split == "final" else "validation_means"
        return float(self.summary[key][method][metric])

    def family_stat(self, family: str, column: str) -> float:
        return float(self.workloads.loc[self.workloads["family"] == family, column].mean())


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def ci_of(data: Artifacts, target: str, reference: str, metric: str) -> Tuple[float, float, float]:
    """Look up a predeclared paired difference and its 95% interval.

    Only pairs produced by the experiment's own comparison table are used, so the
    report never mixes a committed interval with one recomputed at build time.
    """
    sel = data.paired[
        (data.paired["target"] == target)
        & (data.paired["reference"] == reference)
        & (data.paired["metric"] == metric)
    ]
    if sel.empty:
        raise KeyError(f"no committed paired comparison for {target} vs {reference} / {metric}")
    row = sel.iloc[0]
    return float(row["mean_difference"]), float(row["ci95_low"]), float(row["ci95_high"])


MODULES = [
    ("M1", "Workload generator and dataset builder",
     "workload/generator.py, workload/models.py",
     "family config + seed", "validated Workload with fingerprint",
     "Generates the seven synthetic families, validates every process (positive burst, non-negative "
     "arrival, priority bounds) and records split/seed/fingerprint metadata."),
    ("M2", "Discrete-event runtime environment",
     "scheduler/runtime.py",
     "Workload + controller + (quantum, switch cost)", "schedule trace and six metrics",
     "Owns the private event list, the single FIFO ready queue, remaining bursts, switch-cost "
     "accounting, the causal observation and the interval reward."),
    ("M3", "Causal observation and state encoder",
     "rl/runtime_state.py",
     "RuntimeObservation", "state index in [0, 162)",
     "Extracts five causal features, discretises them relative to the quantum and flattens them in "
     "mixed-radix order. Accepts an observation only — never a Workload."),
    ("M4", "Sequential Q-learning controller",
     "rl/runtime_controller.py",
     "observation (encode) / reward (update)", "policy action; Q table and visit counts",
     "Tabular Q-learning over 648 pairs: ε-greedy exploration, first-visit forcing, "
     "unvisited-action masking, read-only greedy evaluation and a counted heuristic fallback."),
    ("M5", "Predeclared heuristic controller",
     "rl/runtime_heuristic.py",
     "RuntimeObservation", "policy action",
     "Non-RL adaptive rule (RR → Priority → SJF → FCFS) sharing the same observation "
     "contract, declared before any test result was inspected."),
    ("M6", "Experiment orchestration, metrics and reporting",
     "experiments/runtime_experiment.py, evaluation/metrics.py",
     "configuration + trained models", "CSV/JSON artifacts, paired CIs, markdown report",
     "Builds disjoint splits, trains five agents, validates, runs the held-out test, computes paired "
     "bootstrap intervals and writes every artifact."),
]


WORKFLOWS = [
    ("M1. Workload generator and dataset builder",
     "workload/generator.py -> workload/models.py",
     [
         ("Select family configuration", "7 frozen parameter sets (burst, arrival, priority)"),
         ("Derive the workload seed", "SHA-256 (master seed, split tag, family index, repetition)"),
         ("Sample 15 processes", "burst, arrival time and priority drawn per family rules"),
         ("Validate the workload", "positive bursts, ordered arrivals, priority bounds"),
         ("Fingerprint and tag", "SHA-256 workload fingerprint plus split and repetition metadata"),
         ("Assemble the split", "6,000 train / 140 validation / 210 held-out workloads"),
     ]),
    ("M2. Discrete-event runtime environment",
     "scheduler/runtime.py - one iteration per decision epoch",
     [
         ("Advance to the next epoch", "dispatch point, completion or Round-Robin quantum boundary"),
         ("Admit arrivals", "all arrivals at or before the epoch, in (arrival time, pid) order"),
         ("Freeze the observation", "arrived and ready work only, the future stays private"),
         ("Ask the controller for an action", "FCFS, SJF, Round Robin or Priority"),
         ("Charge the switch cost", "1 unit if the running PID changes; the first dispatch is free"),
         ("Execute the segment", "min(quantum, remaining) for RR, otherwise to completion"),
         ("Settle the endpoint", "admit arrivals, then RR tail re-enqueue or mark completed"),
         ("Return the interval reward", "r = -dW / q, measured by the environment"),
     ]),
    ("M3. Causal observation and state encoder",
     "rl/runtime_state.py",
     [
         ("Receive the observation", "immutable RuntimeObservation, never a Workload object"),
         ("Extract five causal features", "ready count, completed count, median remaining burst, "
                                          "mean arrival age, priority spread"),
         ("Discretise each feature", "bin edges expressed relative to the quantum q = 4"),
         ("Flatten in mixed-radix order", "3 x 3 x 3 x 3 x 2 collapses to one integer"),
         ("Return the state index", "s in [0, 162), consumed by the controller only"),
     ]),
    ("M4. Sequential Q-learning controller",
     "rl/runtime_controller.py",
     [
         ("Encode the observation", "s = encoder(observation)"),
         ("Select an action", "training: epsilon-greedy plus forced random on first visit; "
                              "evaluation: greedy over visited actions"),
         ("Handle unseen states", "explicit fallback to the causal heuristic, counted and reported"),
         ("Receive the reward", "r = -dW / q from the environment"),
         ("Update the table", "Q(s,a) += alpha * (r + gamma * max over visited a' of Q(s',a') "
                              "- Q(s,a)); terminal omits the bootstrap"),
         ("Persist Q values and visits", "written to runtime_q_table.csv per training seed"),
     ]),
    ("M5. Predeclared heuristic controller",
     "rl/runtime_heuristic.py - fixed rules, no learning",
     [
         ("Receive the observation", "the same frozen RuntimeObservation the learner sees"),
         ("Test the Round Robin rule", "at least 3 ready jobs and mean arrival age >= quantum"),
         ("Test the Priority rule", "at least 2 ready jobs with distinct priorities"),
         ("Test the SJF rule", "largest visible remaining burst >= 2 x smallest visible burst"),
         ("Fall through to FCFS", "the default when none of the three rules fires"),
         ("Return the action", "deterministic; no state, no Q update, identical in training and "
                               "evaluation"),
     ]),
    ("M6. Experiment orchestration, metrics and reporting",
     "experiments/runtime_experiment.py, evaluation/metrics.py",
     [
         ("Build the three splits", "independent seeded streams for training, validation and test"),
         ("Assert split disjointness", "zero SHA-256 fingerprint overlap, fails loudly otherwise"),
         ("Train five agents", "1,200 sequential episodes per seed over six families"),
         ("Validate", "140 workloads, used for reporting and demo selection only"),
         ("Run the held-out test", "210 untouched workloads, greedy and read-only"),
         ("Derive metrics from the traces", "waiting, turnaround, response, utilisation, "
                                            "throughput and context switches per workload"),
         ("Compare pairwise", "paired differences plus a 1,000-replicate crossed bootstrap"),
         ("Write the artifacts", "CSV tables, Q table, coverage, demo trace, summary JSON and the "
                                 "generated report"),
     ]),
]


def module_outputs(data) -> List[Tuple[str, str]]:
    """Concrete evidence, per module, that the module runs and produces output."""
    train = data.coverage[data.coverage["split"] == "training"]
    test = data.coverage[data.coverage["split"] == "final_test"]
    heuristic_rows = int((data.final["method"] == "Causal heuristic").sum())
    return [
        ("M1", f"{len(data.manifest):,} manifest rows; {len(data.workloads):,} held-out workloads "
               f"({len(data.processes):,} processes) regenerated from the declared seeds, with the "
               f"measured burst, arrival and priority statistics of section 5"),
        ("M2", f"{len(data.final):,} final-test metric rows produced from simulated traces; "
               f"fixed-action runs reproduce the standalone FCFS, SJF, Round Robin and Priority "
               f"schedules exactly in the test suite"),
        ("M3", f"162-state encoder; {int(train['visited_states'].min())}-"
               f"{int(train['visited_states'].max())} distinct states visited per agent during "
               f"training"),
        ("M4", f"{len(data.q_table):,} Q values stored; "
               f"{int(train['visited_state_actions'].min())}-"
               f"{int(train['visited_state_actions'].max())} of 648 state-action pairs visited; "
               f"{int(test['unseen_state_decisions'].min())}-"
               f"{int(test['unseen_state_decisions'].max())} counted unseen-state fallbacks per "
               f"model on the held-out test"),
        ("M5", f"heuristic rows present for all {heuristic_rows:,} held-out workloads; mean "
               f"waiting "
               f"{fmt(data.summary_mean('final', 'Causal heuristic', 'avg_waiting_time'))} with "
               f"{fmt(data.summary_mean('final', 'Causal heuristic', 'policy_switch_count'), 2)} "
               f"policy switches per trace"),
        ("M6", f"{len(data.paired):,} paired comparisons with 95% bootstrap intervals, the "
               f"coverage table, the learned-switch demo trace and the generated "
               f"runtime_report.md"),
    ]

