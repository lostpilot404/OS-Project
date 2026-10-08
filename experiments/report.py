"""Markdown result reports rendered only from generated result summaries."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

from config import ACTION_NAMES, SELECTOR_LABEL

BEGIN_RESULTS_MARKER = "<!-- BEGIN GENERATED RESULTS -->"
END_RESULTS_MARKER = "<!-- END GENERATED RESULTS -->"


def render_readme_summary(summary: Mapping[str, object]) -> str:
    """Render compact README tables using only values in the generated summary."""
    training = summary["training"]
    evaluation = summary["evaluation"]
    tables = summary["tables"]
    assert isinstance(training, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(tables, Mapping)
    stat_rows = list(tables["policy_summary_baselines"]) + list(
        tables["policy_summary_selector"]
    )
    stats = {row["policy"]: row for row in stat_rows}

    lines = [
        f"**Run design:** {evaluation['observed_unique_workloads']} unique workloads, "
        f"{evaluation['selector_decision_rows']} model decisions across "
        f"{training['replicates']} independently trained seeds "
        f"({', '.join(str(seed) for seed in training['seeds'])}); "
        f"{evaluation['repetitions_per_family']} workloads per condition. Repeated model "
        "decisions are not counted as additional workloads.",
        "",
        "Mean ± sample SD. Baseline SD is across unique workloads; selector SD is across "
        "training-seed × workload pairs.",
        "",
        "| Method | Waiting time | Turnaround time | Response time | CPU utilization (%) | Throughput | Context switches | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (*ACTION_NAMES, SELECTOR_LABEL):
        row = stats[method]
        cells = [
            method,
            _mean_sd(row, "avg_waiting_time"),
            _mean_sd(row, "avg_turnaround_time"),
            _mean_sd(row, "avg_response_time"),
            _mean_sd(row, "cpu_utilization"),
            _mean_sd(row, "throughput"),
            _mean_sd(row, "context_switches"),
            str(row["observations"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "**Selected policies by workload condition** (counts across independent models):",
            "",
            "| Condition | Unique workloads | Model decisions | FCFS | SJF | Round Robin | Priority | Unseen-state fallbacks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tables["policy_selection"]:
        lines.append(
            f"| {row['family']} | {row['unique_workloads']} | {row['model_decisions']} | "
            f"{row['count_FCFS']} | {row['count_SJF']} | "
            f"{row[f'count_{ACTION_NAMES[2]}']} | {row['count_Priority']} | "
            f"{row['unseen_state_decisions']} |"
        )
    lines.extend(
        [
            "",
            "Full generated results, including per-seed decisions, held-out coverage, "
            "variability and reproducibility metadata: [`results/report.md`](results/report.md).",
        ]
    )
    return "\n".join(lines)


def update_readme_results(readme_path: Path, summary: Mapping[str, object]) -> None:
    """Replace the marked README block with values derived from ``summary.json``."""
    content = readme_path.read_text(encoding="utf-8")
    if content.count(BEGIN_RESULTS_MARKER) != 1 or content.count(END_RESULTS_MARKER) != 1:
        raise ValueError("README must contain exactly one generated-results marker pair")
    before, remainder = content.split(BEGIN_RESULTS_MARKER)
    generated, after = remainder.split(END_RESULTS_MARKER)
    del generated
    updated = (
        before
        + BEGIN_RESULTS_MARKER
        + "\n"
        + render_readme_summary(summary)
        + "\n"
        + END_RESULTS_MARKER
        + after
    )
    readme_path.write_text(updated, encoding="utf-8")


def render_report(summary: Mapping[str, object]) -> str:
    """Render the detailed experiment report from ``summary.json``-compatible data."""
    configuration = summary["configuration"]
    software = summary["software"]
    training = summary["training"]
    evaluation = summary["evaluation"]
    tables = summary["tables"]
    assert isinstance(configuration, Mapping)
    assert isinstance(software, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(tables, Mapping)

    training_runs = list(training["runs"])
    state_config = configuration["state"]
    baseline_rows = list(tables["policy_summary_baselines"])
    selector_rows = list(tables["policy_summary_selector"])
    policy_stats = {row["policy"]: row for row in baseline_rows + selector_rows}
    selection_rows = list(tables["policy_selection"])
    ratio_rows = list(tables["selector_vs_baselines"])
    ratio_index = {(row["policy"], row["metric"]): row for row in ratio_rows}
    state_rows = list(tables["state_occupancy"])
    metrics = [
        ("avg_waiting_time", "Waiting"),
        ("avg_turnaround_time", "Turnaround"),
        ("avg_response_time", "Response"),
        ("cpu_utilization", "CPU util. (%)"),
        ("throughput", "Throughput"),
        ("context_switches", "Context switches"),
    ]
    methods = [*ACTION_NAMES, SELECTOR_LABEL]
    lines: List[str] = [
        "# Generated experiment results",
        "",
        "> This file is generated from `summary.json` by `experiments.report.render_report`; do not edit result values by hand.",
        "",
        "## Scope and run design",
        "",
        "The evaluated system is an **offline workload-aware policy selector**. Each model "
        "makes one batch decision from a complete synthetic workload description and applies "
        "one policy to the entire workload. It is not an OS scheduler and does not switch "
        "policies during execution.",
        "",
        f"- Training: {training['replicates']} independent seeds "
        f"({', '.join(str(seed) for seed in training['seeds'])}), "
        f"{training['episodes_per_run']} terminal episodes per seed.",
        f"- Evaluation: {evaluation['observed_unique_workloads']} unique workloads "
        f"({evaluation['repetitions_per_family']} per condition across "
        f"{len(evaluation['families'])} conditions).",
        f"- Selector decisions: {evaluation['selector_decision_rows']} model-decision rows "
        f"({training['replicates']} independently trained models per unique workload). "
        "These repeated rows are not additional unique workloads.",
        f"- Metric records: {evaluation['metric_rows']} rows, comprising "
        f"{evaluation['baseline_metric_rows']} paired baseline records and "
        f"{evaluation['selector_metric_rows']} learned-selector records.",
        f"- Training conditions: {', '.join(evaluation['training_families'])}.",
        f"- Held-out condition(s): {', '.join(evaluation['held_out_families']) or 'none'}.",
        "- All four baselines receive the same workload as each learned decision; evaluation "
        "fingerprints are disjoint from the training workloads.",
        "",
        "## Results across evaluation workloads",
        "",
        "Entries are arithmetic mean ± sample standard deviation. Baseline variability is "
        "across unique workload instances; selector variability is across model-seed × "
        "workload decisions. Context-switch count is per workload. The reward uses the "
        "separate derived context-switches-per-process metric.",
        "",
    ]
    headers = ["Policy", *(label for _, label in metrics), "n"]
    lines.extend(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
    for method in methods:
        row = policy_stats.get(method)
        if not row:
            continue
        cells = [method, *(_mean_sd(row, metric) for metric, _label in metrics)]
        cells.append(str(row.get("observations", "")))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Learned policy choices by condition",
            "",
            f"Counts below aggregate {training['replicates']} independently trained selectors "
            f"over the same {evaluation['repetitions_per_family']} unique workload instances "
            "per condition. `unseen-state` counts identify evaluation states with no training "
            "update; their equal zero Q-values fall back to the deterministic lowest-index "
            "action (FCFS) and are not evidence of a learned choice.",
            "",
        ]
    )
    selection_headers = [
        "Condition",
        "Unique workloads",
        "Model decisions",
        *(f"{name} n" for name in ACTION_NAMES),
        "Unseen-state decisions",
    ]
    lines.extend(
        ["| " + " | ".join(selection_headers) + " |", "|" + "---|" * len(selection_headers)]
    )
    for row in selection_rows:
        cells = [
            str(row["family"]),
            str(row["unique_workloads"]),
            str(row["model_decisions"]),
            *(str(row[f"count_{name}"]) for name in ACTION_NAMES),
            str(row["unseen_state_decisions"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Per-seed selection stability",
            "",
            "The per-seed table in `summary.json` is generated from `decisions.csv` and should "
            "be consulted alongside the pooled counts; pooled counts alone do not establish "
            "that every independent training run learned the same condition-specific action.",
            "",
            "| Seed | Condition | Unique workloads | FCFS | SJF | Round Robin | Priority | Unseen-state decisions |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tables["policy_selection_by_seed"]:
        lines.append(
            f"| {row['training_seed']} | {row['family']} | {row['unique_workloads']} | "
            f"{row['count_FCFS']} | {row['count_SJF']} | "
            f"{row[f'count_{ACTION_NAMES[2]}']} | {row['count_Priority']} | "
            f"{row['unseen_state_decisions']} |"
        )

    lines.extend(
        [
            "",
            "## Paired waiting-time and turnaround comparisons",
            "",
            "Aggregate reductions are computed from the reported means as "
            "`100 × (baseline mean − selector mean) / baseline mean`. They are not averages "
            "of per-workload percentage ratios.",
            "",
            "| Baseline | Waiting-time reduction | Turnaround-time reduction | Paired model-workload observations | Unique workloads |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for baseline in ACTION_NAMES:
        waiting = ratio_index[(baseline, "avg_waiting_time")]
        turnaround = ratio_index[(baseline, "avg_turnaround_time")]
        lines.append(
            f"| {baseline} | {_reduction(waiting)} | {_reduction(turnaround)} | "
            f"{waiting['model_workload_pairs']} | {waiting['unique_workloads']} |"
        )

    state_variables = state_config["state_variables_in_mixed_radix_order"]
    state_bins = state_config["bins_per_variable"]
    n_states = state_config["n_discrete_states"]
    lines.extend(
        [
            "",
            "## State coverage and limitations",
            "",
            f"The tabular state has {len(state_variables)} variables, {state_bins} bins per "
            f"variable, and {n_states} possible states; coverage differs by training seed. "
            "Held-out Poisson arrivals are excluded from training by design. State overlap "
            "with training conditions is not guaranteed; unseen held-out states use the "
            "documented zero-initialization tie fallback. Accordingly, held-out-family "
            "performance is a limited generalization check, not evidence of broad "
            "distributional generalization.",
            "",
            "| Condition | Unique workloads | Distinct states in evaluation set | Model decisions | State-seen rate | Unseen-state decisions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in state_rows:
        lines.append(
            f"| {row['family']} | {row['unique_workloads']} | {row['distinct_states']} | "
            f"{row['model_decisions']} | {_pct(row['state_seen_rate'] * 100)} | "
            f"{row['unseen_state_decisions']} |"
        )

    lines.extend(
        [
            "",
            "### Training-seed variability",
            "",
            "| Training seed | Mean episodic reward | States visited / possible | Random-action exploration rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for run in training_runs:
        lines.append(
            f"| {run['seed']} | {_num(run['mean_reward_all'])} | "
            f"{run['visited_states']} / {run['n_states']} | "
            f"{_pct(100 * run['random_exploration_episode_rate'])} |"
        )
    lines.extend(
        [
            "",
            "The exploration rate is the observed fraction of episode decisions that took the "
            "epsilon-greedy random-action branch; it is distinct from the epsilon probability "
            "and from the fraction of actions differing from greedy.",
            "",
            "## Reproducibility",
            "",
            "- Runtime: Python {python_version} ({python_implementation}).".format(**software),
            "- Dependency versions: "
            + ", ".join(
                f"{name} {version or 'not installed'}"
                for name, version in software["packages"].items()
            )
            + ".",
            "- Locked dependency manifest SHA-256: `"
            + str(software["requirements_lock_sha256"])
            + "`.",
            "- Rerun with `python main.py experiment`; omit plotting with "
            "`python main.py experiment --no-figures`.",
            "- Machine-readable inputs/results are `config.json`, `workloads.csv`, "
            "`training_history.json`, `q_table.json`, `metrics.csv`, `decisions.csv`, "
            "and `summary.json` in this directory.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_sd(row: Mapping[str, object], metric: str) -> str:
    """Format one generated mean and sample standard deviation."""
    return f"{_num(row[f'{metric}_mean'])} ± {_num(row[f'{metric}_sd'])}"


def _reduction(row: Mapping[str, object]) -> str:
    """Calculate aggregate relative reduction from the generated means."""
    baseline = float(row["mean_baseline"])
    selector = float(row["mean_selector"])
    if baseline == 0.0:
        return "n/a"
    return _pct((baseline - selector) / baseline * 100.0)


def _num(value: object, digits: int = 2) -> str:
    """Format a finite numeric report value."""
    return f"{float(value):.{digits}f}"


def _pct(value: float) -> str:
    """Format a percentage with one decimal digit."""
    return f"{value:.1f}%"
