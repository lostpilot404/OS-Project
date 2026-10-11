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
     "workload/generator.py → workload/models.py",
     [
         ("Select family configuration", "7 frozen parameter sets (burst, arrival, priority)"),
         ("Derive the workload seed", "SHA-256 (master seed, split tag, family index, repetition)"),
         ("Sample 15 processes", "burst, arrival time and priority drawn per family rules"),
         ("Validate the workload", "positive bursts, ordered arrivals, priority bounds"),
         ("Fingerprint and tag", "SHA-256 workload fingerprint + split/repetition metadata"),
         ("Assemble the split", "6,000 train / 140 validation / 210 held-out workloads"),
     ]),
    ("M2. Discrete-event runtime environment",
     "scheduler/runtime.py — one iteration per decision epoch",
     [
         ("Advance to the next epoch", "dispatch point, completion or Round-Robin quantum boundary"),
         ("Admit arrivals", "all arrivals at or before the epoch, in (arrival_time, pid) order"),
         ("Freeze the observation", "arrived/ready work only — the future stays private"),
         ("Ask the controller for an action", "FCFS, SJF, Round Robin or Priority"),
         ("Charge the switch cost", "1 unit if the running PID changes; first dispatch is free"),
         ("Execute the segment", "min(quantum, remaining) for RR, otherwise to completion"),
         ("Settle the endpoint", "admit arrivals, then RR tail re-enqueue or mark completed"),
         ("Return the interval reward", "r = −ΔW / q, measured by the environment"),
     ]),
    ("M3. Causal observation and state encoder",
     "rl/runtime_state.py",
     [
         ("Receive the observation", "immutable RuntimeObservation (no Workload object)"),
         ("Extract five causal features", "ready count, completed count, median remaining burst, "
                                          "mean arrival age, priority spread"),
         ("Discretise each feature", "bin edges expressed relative to the quantum q = 4"),
         ("Flatten in mixed-radix order", "3 × 3 × 3 × 3 × 2 → one integer"),
         ("Return the state index", "s ∈ [0, 162); consumed by the controller only"),
     ]),
    ("M4. Sequential Q-learning controller",
     "rl/runtime_controller.py",
     [
         ("Encode the observation", "s = encoder(observation)"),
         ("Select an action", "training: ε-greedy + forced random on first visit; "
                              "evaluation: greedy over visited actions"),
         ("Handle unseen states", "explicit fallback to the causal heuristic (counted and reported)"),
         ("Receive the reward", "r = −ΔW / q from the environment"),
         ("Update the table", "Q(s,a) += α(r + γ max_visited Q(s′,·) − Q(s,a)); "
                              "terminal omits the bootstrap"),
         ("Persist Q values and visits", "written to runtime_q_table.csv per training seed"),
     ]),
    ("M5. Experiment orchestration and evaluation",
     "experiments/runtime_experiment.py",
     [
         ("Build the three splits", "independent seeded streams per split"),
         ("Assert split disjointness", "zero SHA-256 fingerprint overlap (fails loudly otherwise)"),
         ("Train five agents", "1,200 sequential episodes per seed, six families"),
         ("Validate", "140 workloads — reporting and demo selection only, never tuning"),
         ("Run the held-out test", "210 untouched workloads, greedy and read-only"),
         ("Compare pairwise", "paired differences + 1,000-replicate crossed bootstrap"),
         ("Write the artifacts", "CSV/JSON tables, Q table, coverage, demo trace, report"),
     ]),
    ("M6. Metrics and reporting",
     "evaluation/metrics.py, evaluation/comparison.py",
     [
         ("Collect the schedule trace", "start/end time per (pid, segment)"),
         ("Derive per-process times", "waiting, turnaround and response per process"),
         ("Aggregate per workload", "means, CPU utilisation, throughput, context switches"),
         ("Aggregate per method and family", "mean ± SD across workloads (and model seeds)"),
         ("Compute paired differences", "target − reference per workload, per seed for Q-learning"),
         ("Render the outputs", "CSV tables, summary JSON, markdown report, figures"),
     ]),
]
