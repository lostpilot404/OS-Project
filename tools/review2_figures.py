"""Matplotlib figures for the Review 2 report.

Every figure is rendered from the checked-in experiment artifacts only; no
experiment is re-run here.  All figures are written as PNG files so that
``tools/build_review2_report.py`` can embed them in the PDF.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NAVY = "#1F3864"
STEEL = "#2E75B6"
LIGHT = "#D9E2F3"
GREY = "#5A5A5A"

PALETTE: Dict[str, str] = {
    "FCFS": "#8DA0CB",
    "SJF": "#FC8D62",
    "Round Robin": "#66C2A5",
    "Priority": "#E78AC3",
    "Causal heuristic": "#A6D854",
    "Runtime Q-learning": NAVY,
}

METHOD_ORDER = [
    "FCFS",
    "SJF",
    "Round Robin",
    "Priority",
    "Causal heuristic",
    "Runtime Q-learning",
]

POLICY_COLORS = {
    "FCFS": "#8DA0CB",
    "SJF": "#FC8D62",
    "Round Robin": "#66C2A5",
    "Priority": "#E78AC3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.5,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Diagram helpers
# --------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, body, facecolor=LIGHT, edgecolor=NAVY, fs=7.5):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.1,
            edgecolor=edgecolor,
            facecolor=facecolor,
            mutation_scale=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=fs + 0.5,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=fs - 0.8,
        color="#333333",
        linespacing=1.35,
    )


def _arrow(ax, x1, y1, x2, y2, color=NAVY, label=None, rad=0.0, ls="-"):
    ax.add_patch(
        mpatches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.012,
            label,
            ha="center",
            va="center",
            fontsize=6.6,
            color=color,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9),
            zorder=6,
        )


def architecture_diagram(path: Path) -> Path:
    """Layered module architecture of the runtime-adaptive scheduler."""
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax, 0.03, 0.82, 0.42, 0.14,
        "workload/generator.py",
        "7 seeded families, 15 processes each\nSHA-256 derived workload seeds",
    )
    _box(
        ax, 0.55, 0.82, 0.42, 0.14,
        "experiments/runtime_experiment.py",
        "train / validation / held-out test splits\nseeds, bootstrap, report writing",
    )

    _box(
        ax, 0.03, 0.56, 0.94, 0.19,
        "scheduler/runtime.py  —  single-CPU discrete-event environment",
        "private future-arrival event queue  •  one persistent FIFO ready queue  •  remaining bursts,\n"
        "arrivals, completions  •  switch cost 1  •  quantum 4  •  six schedule metrics",
        facecolor="#EAF0FA",
    )

    _box(
        ax, 0.03, 0.36, 0.30, 0.13,
        "RuntimeObservation",
        "arrived + ready work only\n(immutable, frozen)",
        facecolor="#FFF6E5",
        edgecolor="#B8860B",
    )
    _box(
        ax, 0.37, 0.36, 0.28, 0.13,
        "rl/runtime_state.py",
        "5 causal features\n3x3x3x3x2 = 162 states",
        facecolor="#FFF6E5",
        edgecolor="#B8860B",
    )
    _box(
        ax, 0.69, 0.36, 0.28, 0.13,
        "rl/runtime_controller.py",
        "tabular Q-learning\n162 x 4 = 648 pairs",
        facecolor="#FFF6E5",
        edgecolor="#B8860B",
    )

    _box(
        ax, 0.03, 0.16, 0.94, 0.13,
        "action: FCFS  |  SJF  |  Round Robin  |  Priority   →   next execution segment",
        "a switch changes only the next selection rule and service length; the "
        "queue, remaining bursts and switch accounting are preserved",
        facecolor="#E8F5E9",
        edgecolor="#2E7D32",
    )

    _box(
        ax, 0.03, 0.02, 0.94, 0.10,
        "reward  r_t = -ΔW_t / max(1, q)   (γ = 1  ⇒  Σ r_t = -total waiting time / q)",
        "evaluation/metrics.py  •  rl/runtime_heuristic.py (predeclared non-RL baseline)",
        facecolor="#F3F3F3",
        edgecolor=GREY,
    )

    _arrow(ax, 0.24, 0.82, 0.24, 0.75)
    _arrow(ax, 0.76, 0.82, 0.76, 0.75, label="workloads")
    _arrow(ax, 0.30, 0.56, 0.18, 0.49, label="observation")
    _arrow(ax, 0.35, 0.42, 0.37, 0.42, label="encode")
    _arrow(ax, 0.65, 0.42, 0.69, 0.42, label="select")
    _arrow(ax, 0.83, 0.36, 0.83, 0.30, label="action")
    _arrow(ax, 0.50, 0.16, 0.50, 0.12, label="ΔW over segment")
    _arrow(ax, 0.90, 0.16, 0.985, 0.16)
    _arrow(ax, 0.985, 0.16, 0.985, 0.655, ls="--", label="next epoch")
    _arrow(ax, 0.985, 0.655, 0.972, 0.655)
    _arrow(ax, 0.33, 0.42, 0.37, 0.42, label="encode")

    ax.text(
        0.5, 0.965,
        "Figure: Runtime-adaptive scheduling architecture (Review 2 core)",
        ha="center", fontsize=8, color=GREY,
    )
    return _finish(fig, path)


def decision_loop(path: Path) -> Path:
    """Flowchart of one decision epoch."""
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("Decision epoch t", "dispatch point or RR quantum\nboundary (or completion)"),
        ("Admit arrivals", "all arrivals at or before the epoch\nin (arrival_time, pid) order"),
        ("Build observation", "ready processes, remaining bursts,\narrived/completed counts, ages"),
        ("Encode state s_t", "5 causal features → 162 states"),
        ("Select action a_t", "ε-greedy in training / greedy in\nevaluation; fallback if unseen"),
        ("Execute segment", "run min(quantum, remaining) or to\ncompletion; charge switch cost"),
        ("Reward + update", "r_t = -ΔW_t / q ;\nQ(s,a) += α(r + γ max Q(s',a') - Q)"),
    ]
    y = 0.90
    for title, body in steps:
        _box(ax, 0.12, y - 0.10, 0.76, 0.095, title, body, fs=7.6)
        if y < 0.90:
            _arrow(ax, 0.5, y + 0.005, 0.5, y - 0.005)
        y -= 0.125

    _arrow(ax, 0.12, 0.045, 0.05, 0.045)
    _arrow(ax, 0.05, 0.045, 0.05, 0.90)
    _arrow(ax, 0.05, 0.90, 0.12, 0.90, label="not terminal")

    _box(
        ax, 0.62, 0.005, 0.32, 0.07,
        "Terminal: workload complete",
        "no bootstrap on the last transition",
        facecolor="#FDECEA",
        edgecolor="#C44E52",
        fs=7.0,
    )
    _arrow(ax, 0.88, 0.045, 0.88, 0.075)
    ax.text(
        0.5, 0.965,
        "Figure: One sequential decision epoch",
        ha="center", fontsize=8, color=GREY,
    )
    return _finish(fig, path)


# --------------------------------------------------------------------------
# Result figures
# --------------------------------------------------------------------------
_METRIC_LABEL = {
    "avg_waiting_time": ("Mean waiting time", "time units"),
    "avg_turnaround_time": ("Mean turnaround time", "time units"),
    "avg_response_time": ("Mean response time", "time units"),
    "cpu_utilization": ("CPU utilization", "%"),
    "throughput": ("Throughput", "processes / time unit"),
    "context_switches": ("Context switches", "count / trace"),
}


def metric_bars(final_df: pd.DataFrame, path: Path) -> Path:
    """Six metric panels, mean +/- SD across the 210 held-out workloads."""
    _style()
    metrics = list(_METRIC_LABEL)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.9))
    for ax, metric in zip(axes.ravel(), metrics):
        title, unit = _METRIC_LABEL[metric]
        means, errs, colors = [], [], []
        for method in METHOD_ORDER:
            values = final_df.loc[final_df["method"] == method, metric].to_numpy(dtype=float)
            means.append(values.mean())
            errs.append(values.std(ddof=1))
            colors.append(PALETTE[method])
        bars = ax.bar(range(len(METHOD_ORDER)), means, yerr=errs, color=colors,
                      edgecolor="#333333", linewidth=0.4, capsize=2,
                      error_kw=dict(linewidth=0.7, ecolor="#333333"))
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(["FCFS", "SJF", "RR", "Prio", "Heur.", "Q"], fontsize=7)
        ax.set_title(f"{title} ({unit})", fontsize=8.5)
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:,.1f}" if abs(mean) >= 10 else f"{mean:.3f}",
                ha="center", va="bottom", fontsize=6.3, color="#222222",
            )
        ax.margins(y=0.22)
    fig.suptitle(
        "Held-out final test: 210 workloads (seed 19301), 5 Q-learning model seeds",
        fontsize=8.6, color=GREY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, path)


def metric_boxplots(final_df: pd.DataFrame, path: Path) -> Path:
    """Per-workload spread of waiting and response time."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, metric in zip(axes, ["avg_waiting_time", "avg_response_time"]):
        data = [
            final_df.loc[final_df["method"] == m, metric].to_numpy(dtype=float)
            for m in METHOD_ORDER
        ]
        box = ax.boxplot(
            data, tick_labels=["FCFS", "SJF", "RR", "Prio", "Heur.", "Q"],
            patch_artist=True, widths=0.6, showfliers=False, medianprops=dict(color="black"),
        )
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(PALETTE[method])
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.6)
        ax.set_title(_METRIC_LABEL[metric][0])
        ax.set_ylabel(_METRIC_LABEL[metric][1])
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Distribution across the 210 held-out workloads (whiskers = 1.5 IQR)",
        fontsize=8.4, color=GREY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path)


def family_bars(final_df: pd.DataFrame, path: Path, families: Sequence[str]) -> Path:
    """Per-family waiting and response time grouped bars."""
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True)
    short = ["FCFS", "SJF", "RR", "Prio", "Heur.", "Q"]
    for ax, metric in zip(axes, ["avg_waiting_time", "avg_response_time"]):
        x = np.arange(len(families))
        width = 0.135
        for i, method in enumerate(METHOD_ORDER):
            values = [
                final_df.loc[
                    (final_df["method"] == method) & (final_df["family"] == fam), metric
                ].mean()
                for fam in families
            ]
            ax.bar(x + (i - 2.5) * width, values, width, label=short[i],
                   color=PALETTE[method], edgecolor="#333333", linewidth=0.35)
        ax.set_title(_METRIC_LABEL[metric][0] + f" ({_METRIC_LABEL[metric][1]})")
        ax.set_ylabel(_METRIC_LABEL[metric][1], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(np.arange(len(families)))
    axes[1].set_xticklabels([f.replace("_", "\n") for f in families], fontsize=6.8)
    axes[1].legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.suptitle(
        "Per-family means on the held-out final test (30 workloads per family; "
        "poisson_arrivals was never trained on)",
        fontsize=8.4, color=GREY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _finish(fig, path)


def training_curves(train_df: pd.DataFrame, path: Path) -> Path:
    """Sequential learning dynamics across 1,200 episodes per seed."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    window = 100
    seeds = sorted(train_df["training_seed"].unique())

    ax = axes[0]
    for seed in seeds:
        series = (
            train_df.loc[train_df["training_seed"] == seed, "avg_waiting_time"]
            .rolling(window, min_periods=window)
            .mean()
        )
        ax.plot(series.to_numpy(), linewidth=0.9, alpha=0.85, label=f"seed {seed}")
    overall = train_df.groupby("episode")["avg_waiting_time"].mean().rolling(window).mean()
    ax.plot(overall.to_numpy(), color="black", linewidth=1.6, label="mean of seeds")
    ax.set_title(f"Training mean waiting time ({window}-episode rolling mean)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("time units")
    ax.legend(fontsize=6.4, ncol=2)

    ax = axes[1]
    for seed in seeds:
        series = (
            train_df.loc[train_df["training_seed"] == seed, "episode_reward"]
            .rolling(window, min_periods=window)
            .mean()
        )
        ax.plot(series.to_numpy(), linewidth=0.9, alpha=0.85, label=f"seed {seed}")
    ax.set_title(f"Training episodic return ({window}-episode rolling mean)")
    ax.set_xlabel("training episode")
    ax.set_ylabel(r"$\Sigma r_t$  (higher = less waiting)")
    ax.legend(fontsize=6.4, ncol=2)
    fig.suptitle(
        "5 independent agents x 1,200 sequential episodes (6,000 training workloads)",
        fontsize=8.4, color=GREY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path)


def bootstrap_forest(paired_df: pd.DataFrame, path: Path) -> Path:
    """Paired 95% crossed-bootstrap confidence intervals."""
    _style()
    wanted = [
        ("Runtime Q-learning", "SJF", "avg_waiting_time"),
        ("Runtime Q-learning", "SJF", "avg_response_time"),
        ("Runtime Q-learning", "SJF", "context_switches"),
        ("Round Robin", "SJF", "avg_waiting_time"),
        ("Causal heuristic", "SJF", "avg_waiting_time"),
        ("Runtime Q-learning", "Causal heuristic", "avg_waiting_time"),
        ("Runtime Q-learning", "Causal heuristic", "avg_response_time"),
        ("Runtime Q-learning", "Causal heuristic", "context_switches"),
    ]
    rows = []
    for target, reference, metric in wanted:
        sel = paired_df[
            (paired_df["target"] == target)
            & (paired_df["reference"] == reference)
            & (paired_df["metric"] == metric)
        ]
        if not sel.empty:
            rows.append(sel.iloc[0])

    labels = [
        f"{r['target'].replace('Runtime ', '')} - {r['reference']}\n"
        f"({r['metric'].replace('avg_', '').replace('_', ' ')})"
        for r in rows
    ]
    diffs = [float(r["mean_difference"]) for r in rows]
    lows = [float(r["ci95_low"]) for r in rows]
    highs = [float(r["ci95_high"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = np.arange(len(rows))
    colors = ["#2E7D32" if d < 0 else "#C44E52" for d in diffs]
    ax.errorbar(
        diffs, y,
        xerr=[np.array(diffs) - np.array(lows), np.array(highs) - np.array(diffs)],
        fmt="o", color="#333333", ecolor="#333333", elinewidth=1.0, capsize=3,
        markersize=4.5, linestyle="none",
    )
    ax.scatter(diffs, y, color=colors, zorder=5, s=26)
    ax.axvline(0, color=NAVY, linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("paired mean difference (target - reference); negative favours the target")
    ax.set_title("Paired 95% family-stratified crossed-bootstrap intervals (1,000 replicates, seed 104729)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _finish(fig, path)


def gantt_trace(demo: Dict, path: Path) -> Path:
    """Illustrative learned within-trace switching timeline."""
    _style()
    decisions = demo["decisions"]
    policy_by_key = {(d["time"], d["chosen_pid"]): d["policy"] for d in decisions}
    segments = demo["trace"]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    pids = sorted({int(s["pid"]) for s in segments})
    ypos = {pid: i for i, pid in enumerate(pids)}
    used = set()
    for seg in segments:
        pid = int(seg["pid"])
        policy = policy_by_key.get((seg["start"], pid))
        if policy is None:
            policy = next(
                (d["policy"] for d in decisions if d["chosen_pid"] == pid and d["time"] <= seg["start"]),
                "FCFS",
            )
        used.add(policy)
        ax.barh(
            ypos[pid],
            seg["end"] - seg["start"],
            left=seg["start"],
            height=0.62,
            color=POLICY_COLORS.get(policy, GREY),
            edgecolor="white",
            linewidth=0.5,
        )
    for time in demo["policy_switch_times"]:
        ax.axvline(time, color="#C44E52", linewidth=0.7, linestyle=":", alpha=0.8)
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([f"P{pid}" for pid in pids], fontsize=6.6)
    ax.set_xlabel("simulated time")
    ax.set_title(
        f"Illustrative validation trace — seed {demo['training_seed']}, {demo['family']} "
        f"rep {demo['repetition']}\n{demo['decision_count']} decisions, "
        f"{demo['policy_switch_count']} policy switches "
        f"({demo['metrics']['avg_waiting_time']:.2f} wait / "
        f"{demo['metrics']['avg_response_time']:.2f} response)",
        fontsize=8.6,
    )
    ax.grid(axis="y", visible=False)
    handles = [
        mpatches.Patch(facecolor=POLICY_COLORS[p], edgecolor="#333333", label=p)
        for p in ["FCFS", "SJF", "Round Robin", "Priority"]
        if p in used
    ]
    handles.append(
        mpatches.Patch(facecolor="white", edgecolor="#C44E52", linestyle=":",
                       label="policy switch")
    )
    ax.legend(handles=handles, fontsize=6.8, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    return _finish(fig, path)


def coverage_heatmap(q_table: pd.DataFrame, coverage: pd.DataFrame, path: Path) -> Path:
    """State-action visit coverage of the tabular controller."""
    _style()
    visits = (
        q_table.pivot_table(index="state_index", columns="action", values="visit_count",
                            aggfunc="sum")
        .reindex(index=range(162), columns=range(4))
        .fillna(0.0)
        .to_numpy()
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    image = ax.imshow(visits, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("action")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["FCFS", "SJF", "RR", "Prio"], fontsize=7)
    ax.set_ylabel("causal state index (162)")
    ax.set_title("Visited (state, action) counts, pooled over 5 seeds")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label="visits")

    ax = axes[1]
    train = coverage[coverage["split"] == "training"].sort_values("training_seed")
    seeds = [str(int(s)) for s in train["training_seed"]]
    pairs = train["visited_state_actions"].to_numpy(dtype=float)
    states = train["visited_states"].to_numpy(dtype=float)
    x = np.arange(len(seeds))
    ax.bar(x - 0.19, pairs, 0.36, label="state-action pairs (of 648)", color=NAVY)
    ax.bar(x + 0.19, states, 0.36, label="states (of 162)", color="#8DA0CB")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=7)
    ax.set_xlabel("training seed")
    ax.set_title("Tabular coverage per seed", fontsize=9)
    ax.legend(fontsize=6.4, loc="lower right")
    ax.set_ylim(0, 300)
    for i, (p, s) in enumerate(zip(pairs, states)):
        ax.text(i - 0.19, p + 4, f"{int(p)}", ha="center", fontsize=6)
        ax.text(i + 0.19, s + 4, f"{int(s)}", ha="center", fontsize=6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _finish(fig, path)


def legacy_policy_share(policy_selection: List[Dict], path: Path) -> Path:
    """Offline selector policy share by workload family."""
    _style()
    families = [row["family"] for row in policy_selection]
    policies = ["FCFS", "SJF", "Round Robin", "Priority"]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    bottom = np.zeros(len(families))
    for policy in policies:
        shares = np.array([float(row[f"share_{policy}"]) for row in policy_selection])
        ax.bar(families, shares, 0.6, bottom=bottom, label=policy,
               color=POLICY_COLORS[policy], edgecolor="white", linewidth=0.5)
        bottom += shares
    ax.set_ylabel("share of decisions")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=6.6)
    ax.set_title("Legacy offline selector: policy chosen per workload family (210 workloads)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return _finish(fig, path)


def legacy_training(history: Dict, path: Path) -> Path:
    """Offline selector training reward and epsilon decay."""
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    twin = ax.twinx()
    for run in history["runs"]:
        detail = run["episodes_detail"]
        seed = run["summary"]["seed"]
        reward = pd.Series(detail["reward"]).rolling(100, min_periods=100).mean()
        ax.plot(reward.to_numpy(), linewidth=0.9, alpha=0.9, label=f"seed {seed}")
        twin.plot(detail["epsilon"], linewidth=0.7, color="black", linestyle="--",
                  alpha=0.6)
    ax.set_xlabel("training episode (one whole workload per episode)")
    ax.set_ylabel("rolling mean reward (window 100)")
    twin.set_ylabel(r"$\epsilon$", fontsize=9)
    twin.set_ylim(0, 1.05)
    twin.grid(False)
    ax.set_title("Legacy offline selector: reward trend and epsilon decay (5 seeds)")
    ax.legend(fontsize=6.6, ncol=5, loc="lower right")
    fig.tight_layout()
    return _finish(fig, path)


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))
