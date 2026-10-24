"""Data layer for the Review 2 report.

Everything the report states is read from the **committed artifacts of the primary system**
(the self-healing scheduling study, ``results/healing/``) or, where the report describes a
preserved secondary workflow, from that workflow's own committed artifacts
(``results/study/``, ``results/runtime/``).  Nothing is typed in by hand, and no artifact
that the repository does not track is required: the legacy offline pipeline writes
``results/summary.json`` on demand and is therefore *not* used here.

The workload statistics of Section 5 are computed by calling the project's own seeded
generator on the split configuration the study declares, so the dataset description cannot
drift from the dataset that was actually evaluated.

Usage::

    from review2_data import Artifacts
    data = Artifacts(Path("."))          # repository root
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

#: Arm display metadata, in the order the study declares them.
ARMS: Tuple[Tuple[str, str, str], ...] = (
    ("fixed_fcfs", "Fixed FCFS", "fixed"),
    ("fixed_sjf", "Fixed SJF", "fixed"),
    ("fixed_round_robin", "Fixed Round Robin", "fixed"),
    ("fixed_priority", "Fixed Priority", "fixed"),
    ("detector_only", "Detector only (control)", "control"),
    ("rule_fairness", "Rule: fairness", "rule"),
    ("rule_wait", "Rule: waiting time", "rule"),
    ("rule_wait_from_sjf", "Rule from SJF", "rule"),
    ("q_recovery", "Q-learning recovery", "learned"),
    ("q_recovery_from_sjf", "Q-learning from SJF", "learned"),
)

#: The eight recovery actions of the declared action space.
ACTIONS: Tuple[str, ...] = (
    "hold", "set_fcfs", "set_sjf", "set_round_robin", "set_priority",
    "shrink_quantum", "grow_quantum", "revert_to_last_accepted",
)

#: The five features of the learning state, with their declared bin counts.
STATE_FEATURES: Tuple[Tuple[str, str, int], ...] = (
    ("Primary diagnosis", "head-of-line blocking, fairness starvation, priority anomaly, "
                          "tail dispersion, queue congestion, unexplained", 6),
    ("Excess over threshold", "< 1.5k, < 2.5k, >= 2.5k band widths (k = detection multiplier)", 3),
    ("Ready-queue size", "1-2, 3-4, 5 or more", 3),
    ("Policy in force", "FCFS, SJF, Round Robin, Priority", 4),
    ("Visible remaining work", "< 3, < 8, >= 8 multiples of the reference waiting time", 3),
)


def fmt(value: Optional[float], digits: int = 2) -> str:
    """Format an optional number, printing ``n/a`` instead of ``nan``."""
    if value is None:
        return "n/a"
    value = float(value)
    if value != value:  # NaN
        return "n/a"
    return f"{value:,.{digits}f}"


@dataclass
class Artifacts:
    """Committed evidence of the primary study, plus the preserved secondary workflows."""

    root: Path
    with_dataset: bool = True
    healing_dir: Path = field(init=False)
    summary: Dict = field(init=False, default_factory=dict)
    final: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    val: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    incidents_final: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    incidents_val: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    episodes_final: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    audit: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    detection: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    scenarios: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    coverage_curve: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    probe: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    training: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    thresholds: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    demo: Dict = field(init=False, default_factory=dict)
    study: Dict = field(init=False, default_factory=dict)
    runtime: Dict = field(init=False, default_factory=dict)
    dataset: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        root = Path(self.root)
        self.healing_dir = root / "results" / "healing"
        read = pd.read_csv
        self.summary = json.loads(
            (self.healing_dir / "healing_summary.json").read_text(encoding="utf-8")
        )
        self.final = read(self.healing_dir / "healing_metrics_final_test.csv")
        self.val = read(self.healing_dir / "healing_metrics_validation.csv")
        self.incidents_final = read(self.healing_dir / "healing_incidents_final_test.csv")
        self.incidents_val = read(self.healing_dir / "healing_incidents_validation.csv")
        self.episodes_final = read(self.healing_dir / "healing_episodes_final_test.csv")
        self.audit = read(self.healing_dir / "healing_audit.csv")
        self.comparisons = read(self.healing_dir / "healing_comparisons.csv")
        self.detection = read(self.healing_dir / "healing_detection_summary.csv")
        self.scenarios = read(self.healing_dir / "healing_detection_scenarios.csv")
        self.coverage = read(self.healing_dir / "healing_coverage.csv")
        self.coverage_curve = read(self.healing_dir / "healing_coverage_curve.csv")
        self.probe = read(self.healing_dir / "healing_action_probe.csv")
        # Strict canonical-schema enforcement for the probe: the historical file
        # kept as healing_action_probe.legacy_raw_proxy.csv used reward_band_units
        # for the raw proxy; the canonical file must carry controller_reward.
        if "controller_reward" not in self.probe.columns:
            from errors import ValidationError

            raise ValidationError(
                "Artifacts probe is legacy schema (missing 'controller_reward'). "
                "The canonical results/healing/healing_action_probe.csv must be the "
                "31+ column file generated by action_effect_probe() after the reward "
                "fix (contains controller_reward, raw_load_improvement_band_units, "
                "closure_residual, verdict). Legacy file is preserved as "
                "healing_action_probe.legacy_raw_proxy.csv for traceability."
            )
        self.training = read(self.healing_dir / "healing_training_episodes.csv")
        self.thresholds = read(self.healing_dir / "healing_threshold_derivation.csv")
        self.demo = json.loads(
            (self.healing_dir / "healing_demo.json").read_text(encoding="utf-8")
        )
        self.study = json.loads(
            (root / "results" / "study" / "summary.json").read_text(encoding="utf-8")
        )
        self.runtime = json.loads(
            (root / "results" / "runtime" / "runtime_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if self.with_dataset:
            self.dataset = _dataset_statistics(root)

    # -- declared configuration --------------------------------------------------------
    @property
    def healing_config(self) -> Dict:
        """The declared thresholds of the healing loop (from the summary artifact)."""
        return self.summary["healing_config"]

    @property
    def q_learning(self) -> Dict:
        """The declared Q-learning hyper-parameters."""
        return self.summary["q_learning"]

    @property
    def software(self) -> Dict:
        """Recorded interpreter, library and platform versions of the study run."""
        return self.summary["software"]

    @property
    def final_test_seed(self) -> int:
        return int(self.summary["final_test_seed"])

    @property
    def validation_seed(self) -> int:
        return int(self.summary["validation_seed"])

    @property
    def training_seeds(self) -> List[int]:
        return [int(seed) for seed in self.summary["training_seeds"]]

    @property
    def bootstrap_seed(self) -> int:
        """Bootstrap seed, from the artifact when it is recorded and the declared design
        otherwise (the summary of an older run may not carry it)."""
        if "bootstrap_seed" in self.summary:
            return int(self.summary["bootstrap_seed"])
        if not self.thresholds.empty and "bootstrap_seed" in set(self.thresholds["quantity"]):
            return int(self.threshold("bootstrap_seed"))
        root = self.root
        try:
            import sys

            if str(Path(root).resolve()) not in sys.path:
                sys.path.insert(0, str(Path(root).resolve()))
            from experiments.healing_config import HealingExperimentConfig  # noqa: PLC0415

            return int(HealingExperimentConfig().bootstrap_seed)
        except Exception:  # pragma: no cover - incomplete checkout
            return 0

    def audit_summary_row(self, arm: str) -> Dict:
        """One row of the recorded counterfactual-audit summary, by arm."""
        for row in self.summary["audit_summary"]:
            if row["arm"] == arm:
                return dict(row)
        return {}

    def seed_variability(self, arm: str = "q_recovery") -> pd.DataFrame:
        """Per-training-seed held-out result of a learned arm (spread across seeds)."""
        rows = [dict(row) for row in self.summary["q_seed_variability"] if row["arm"] == arm]
        return pd.DataFrame(rows)

    # -- convenience views over the artifacts -----------------------------------------
    def arm_rows(self, split: str = "final_test") -> List[Dict]:
        """One summary row per arm, ordered by mean waiting time (best first)."""
        rows = []
        for arm, label, kind in ARMS:
            means = [r for r in self.summary["means"] if r["arm"] == arm]
            if not means:
                continue
            row = dict(means[0])
            row["label"] = label
            row["kind"] = kind
            rows.append(row)
        rows.sort(key=lambda r: float(r["avg_waiting_time"]))
        return rows

    def mean(self, arm: str, metric: str) -> float:
        """Mean of ``metric`` for ``arm`` on the held-out split."""
        for row in self.summary["means"]:
            if row["arm"] == arm:
                return float(row[metric])
        return float("nan")

    def metric_by_trace(self, arm: str, metric: str) -> pd.Series:
        """Per-trace values of ``metric`` for ``arm`` on the held-out split."""
        subset = self.final[self.final["arm"] == arm]
        return subset[metric].astype(float).reset_index(drop=True)

    def comparison(self, target: str, reference: str, metric: str = "avg_waiting_time") -> Dict:
        """One paired-bootstrap comparison row (difference and 95 % interval)."""
        subset = self.comparisons[
            (self.comparisons["target"] == target)
            & (self.comparisons["reference"] == reference)
            & (self.comparisons["metric"] == metric)
        ]
        if subset.empty:
            return {}
        row = subset.iloc[0]
        return {
            "mean_difference": float(row["mean_difference"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
            "favours": str(row["favours"]),
        }

    def verdict_counts(self, split: str = "final_test") -> pd.DataFrame:
        """Interventions, verdicts and fallback episodes per healing arm."""
        frame = self.detection[self.detection["split"] == split]
        frame = frame[frame["interventions"] > 0]
        columns = ["arm", "episodes", "incidents", "interventions", "accepted", "rejected",
                   "inconclusive", "held", "episodes_with_fallback"]
        return frame[columns].copy()

    def audit_rows(self, split: str = "final_test") -> pd.DataFrame:
        """Per-intervention audit summary for the arms that actually intervened."""
        frame = self.audit[
            (self.audit["split"] == split) & (self.audit["verdict"].astype(str) != "episode_level")
        ]
        rows = []
        for arm, subset in frame.groupby("arm"):
            gain = subset["matched_gain"].astype(float)
            verdict = subset["verdict"].astype(str)
            rows.append({
                "arm": arm,
                "interventions": len(subset),
                "mean_gain": float(gain.mean()),
                "median_gain": float(gain.median()),
                "helped_share": float((gain > 0).mean()),
                "accepted": int((verdict == "accepted").sum()),
                "rejected": int((verdict == "rejected").sum()),
            })
        return pd.DataFrame(rows).sort_values("interventions", ascending=False)

    def discrimination(self) -> Dict:
        """The verifier's measured discrimination on the held-out interventions."""
        return self.summary["verification_discrimination"]

    def scenario_summary(self) -> pd.DataFrame:
        """Detector behaviour per controlled scenario (stable / burst / storm)."""
        rows = []
        for name, subset in self.scenarios.groupby("scenario"):
            rows.append({
                "scenario": name,
                "traces": len(subset),
                "detected": int(subset["detected"].sum()),
                "detection_share": float(subset["detected"].mean()),
                "incidents_per_trace": float(subset["incidents"].mean()),
                "median_delay_units": float(subset.loc[subset["detected"], "detection_delay_time"].median()),
                "median_delay_decisions": float(subset.loc[subset["detected"], "detection_delay_decisions"].median()),
            })
        return pd.DataFrame(rows).sort_values("scenario")

    def threshold(self, name: str) -> float:
        """One row of the threshold-derivation artifact, by quantity name."""
        subset = self.thresholds[self.thresholds["quantity"] == name]
        return float(subset.iloc[0]["value"]) if not subset.empty else float("nan")

    # -- preserved secondary workflows --------------------------------------------------
    def study_arms(self) -> List[Tuple[str, float, float]]:
        """Offline selector study: ``(arm, mean objective cost, mean regret)``."""
        arms = self.study["test"]["arms"]
        return [(name, float(values["mean_objective_cost"]), float(values["mean_regret"]))
                for name, values in arms.items()]

    def runtime_means(self) -> Dict[str, float]:
        """Runtime appendix: mean waiting time of every method on its held-out split."""
        return {name: float(values["avg_waiting_time"])
                for name, values in self.runtime["final_test_means"].items()
                if "avg_waiting_time" in values}


def _dataset_statistics(root: Path) -> pd.DataFrame:
    """Per-family statistics of the evaluation splits, from the project's own generator.

    The generator is imported here rather than reimplemented, so the table describes the
    exact instances the study evaluated.  A failure to import (for example when the report
    is built from a partial checkout) degrades to an empty frame instead of a wrong number.
    """
    try:
        import sys

        if str(root.resolve()) not in sys.path:
            sys.path.insert(0, str(root.resolve()))
        from experiments.healing_config import SPLIT_TAGS, HealingExperimentConfig  # noqa: PLC0415
        from workload.generator import WorkloadGenerator  # noqa: PLC0415

        config = HealingExperimentConfig()
        generator = WorkloadGenerator(config.families)
        rows: Dict[str, Dict] = {}
        for split, repetitions in (("validation", config.validation_repetitions),
                                   ("final_test", config.final_test_repetitions)):
            master = (config.validation_seed if split == "validation"
                      else config.final_test_seed)
            for family_index, family in enumerate(config.families):
                stats = rows.setdefault(
                    family.name,
                    {"family": family.name, "processes": family.num_processes, "traces": 0,
                     "burst_total": 0, "burst_min": None, "burst_max": None, "span_max": 0},
                )
                for repetition in range(repetitions):
                    from config import derive_seed  # noqa: PLC0415

                    seed = derive_seed(master, SPLIT_TAGS[split], family_index, repetition)
                    workload = generator.generate(family.name, seed)
                    bursts = [process.burst_time for process in workload.processes]
                    arrivals = [process.arrival_time for process in workload.processes]
                    stats["traces"] += 1
                    stats["burst_total"] += sum(bursts)
                    stats["burst_min"] = min(bursts) if stats["burst_min"] is None else min(
                        stats["burst_min"], min(bursts))
                    stats["burst_max"] = max(bursts) if stats["burst_max"] is None else max(
                        stats["burst_max"], max(bursts))
                    stats["span_max"] = max(stats["span_max"], max(arrivals) - min(arrivals))
        for stats in rows.values():
            stats["burst_mean"] = stats.pop("burst_total") / (stats["traces"] * stats["processes"])
            stats["mean_total_work"] = stats["burst_mean"] * stats["processes"]
        return pd.DataFrame(list(rows.values())).sort_values("family").reset_index(drop=True)
    except Exception:  # pragma: no cover - only when the checkout is incomplete
        return pd.DataFrame()


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    data = Artifacts(Path(__file__).resolve().parent.parent)
    print("held-out arms, sorted by mean waiting time:")
    for row in data.arm_rows():
        print(f"  {row['label']:<28} {row['avg_waiting_time']:9.2f}   traces={row['traces']}")
    print("\npaired comparisons:")
    for _, row in data.comparisons.iterrows():
        if row["metric"] == "avg_waiting_time":
            print(f"  {row['target']:<22} - {row['reference']:<18} "
                  f"{row['mean_difference']:+8.2f} [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]")
    print("\ndataset families:")
    print(data.dataset.to_string(index=False) if not data.dataset.empty else "  (not built)")
