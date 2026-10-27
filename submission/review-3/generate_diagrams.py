#!/usr/bin/env python3
"""Generate monochrome technical diagrams for the Review 3 submission paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "diagrams")
os.makedirs(OUT, exist_ok=True)

DPI = 200
LW = 1.2
FS_TITLE = 11
FS_NODE = 8
FS_SMALL = 7


def _base_fig(w, h):
    fig, ax = plt.subplots(1, 1, figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, label, fontsize=FS_NODE, style="-"):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.05", linewidth=LW,
        edgecolor="black", facecolor="white", linestyle=style,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, wrap=True,
            clip_on=True)


def _diamond(ax, x, y, w, h, label, fontsize=FS_SMALL):
    verts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = plt.Polygon(verts, closed=True, fill=False, edgecolor="black", linewidth=LW)
    ax.add_patch(poly)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2, label="", label_offset=(0, 0)):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=LW),
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.5,
                fontstyle="italic")


# ============================================================================
# Figure 1: Overall System Architecture
# ============================================================================
def fig_architecture():
    fig, ax = _base_fig(10, 7)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_title("Figure 1. Overall system architecture of the adaptive CPU scheduling pipeline",
                 fontsize=FS_TITLE, pad=12)

    # Row 1: Workload generation
    _box(ax, 1.5, 8, 2.2, 0.6, "Workload\nGenerator\n(workload/)")
    _box(ax, 4.5, 8, 2.2, 0.6, "Workload\nFingerprint\n(fingerprint.py)")
    _box(ax, 7.5, 8, 2.2, 0.6, "Drift Injector\n(drift.py)")
    _box(ax, 10.5, 8, 2.2, 0.6, "Burst\nEstimator\n(burst_estimator.py)")

    _arrow(ax, 2.6, 8, 3.4, 8)
    _arrow(ax, 5.6, 8, 6.4, 8)
    _arrow(ax, 8.6, 8, 9.4, 8)

    # Row 2: Scheduler + Controller
    _box(ax, 3.0, 6.2, 2.8, 0.7, "Discrete-Event\nScheduler\n(scheduler/runtime.py)", fontsize=7.5)
    _box(ax, 7.0, 6.2, 2.8, 0.7, "Runtime\nObservation\n(RuntimeObservation)", fontsize=7.5)
    _box(ax, 10.5, 6.2, 2.2, 0.7, "Policy Classes\n(FCFS/SJF/RR/Pri)", fontsize=7)

    _arrow(ax, 6.0, 8, 4.0, 6.6)
    _arrow(ax, 4.4, 6.2, 5.6, 6.2)
    _arrow(ax, 8.4, 6.2, 9.4, 6.2)
    _arrow(ax, 9.4, 8, 10.5, 6.6)

    # Row 3: Healing controller loop
    _box(ax, 2.0, 4.2, 2.2, 0.6, "Causal Monitor\n(monitor.py)")
    _box(ax, 5.0, 4.2, 2.2, 0.6, "Degradation\nDetector\n(detector.py)")
    _box(ax, 8.0, 4.2, 2.2, 0.6, "Diagnosis\n(diagnosis.py)")
    _box(ax, 10.8, 4.2, 1.8, 0.6, "Recovery\nPolicy\n(policies.py)")

    _arrow(ax, 4.0, 5.9, 2.8, 4.5)
    _arrow(ax, 3.1, 4.2, 3.9, 4.2)
    _arrow(ax, 6.1, 4.2, 6.9, 4.2)
    _arrow(ax, 9.1, 4.2, 9.9, 4.2)

    # Row 4: Verification
    _box(ax, 2.5, 2.3, 2.2, 0.6, "Verification\nWindow\n(verifier.py)")
    _box(ax, 5.5, 2.3, 2.2, 0.6, "Accept / Reject\n/ Inconclusive")
    _box(ax, 8.5, 2.3, 2.2, 0.6, "Rollback /\nFallback\n(actions.py)")
    _box(ax, 10.8, 2.3, 1.6, 0.6, "Incident\nLedger")

    _arrow(ax, 10.8, 3.9, 10.8, 2.6)
    _arrow(ax, 3.0, 3.9, 2.8, 2.6)
    _arrow(ax, 3.6, 2.3, 4.4, 2.3)
    _arrow(ax, 6.6, 2.3, 7.4, 2.3)
    _arrow(ax, 9.6, 2.3, 10.0, 2.3)

    # Row 5: Metrics and output
    _box(ax, 2.0, 0.6, 2.5, 0.6, "Metrics\n(evaluation/metrics.py)")
    _box(ax, 5.5, 0.6, 2.5, 0.6, "Reliability\n(evaluation/reliability.py)")
    _box(ax, 9.0, 0.6, 2.5, 0.6, "Paired Comparison\n(evaluation/paired.py)")

    _arrow(ax, 3.0, 2.0, 2.5, 0.9)
    _arrow(ax, 5.5, 2.0, 5.5, 0.9)
    _arrow(ax, 8.5, 2.0, 8.5, 0.9)

    # Feedback arrow
    ax.annotate(
        "", xy=(1.2, 6.2), xytext=(1.2, 2.3),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8,
                        connectionstyle="arc3,rad=-0.3", linestyle="--"),
    )
    ax.text(0.3, 4.2, "Feedback\n(action applied\nto scheduler)", ha="center", va="center",
            fontsize=6, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_architecture.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Figure 2: Self-Healing Loop Workflow
# ============================================================================
def fig_healing_loop():
    fig, ax = _base_fig(10, 8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_title("Figure 2. Self-healing controller workflow (one trace)",
                 fontsize=FS_TITLE, pad=12)

    # Main flow
    _box(ax, 5.0, 8.2, 2.0, 0.5, "Generate\nworkload")
    _box(ax, 5.0, 7.2, 2.0, 0.5, "Simulate\n(dispatch slice)")
    _box(ax, 5.0, 6.2, 2.0, 0.5, "Monitor\n(EWMA reference)")
    _diamond(ax, 5.0, 5.0, 2.8, 0.8, "Degradation\ndetected?")
    _box(ax, 5.0, 3.8, 2.0, 0.5, "Diagnose\ncause")
    _box(ax, 5.0, 2.8, 2.0, 0.5, "Select recovery\naction")
    _box(ax, 5.0, 1.8, 2.0, 0.5, "Apply action\nto scheduler")
    _box(ax, 5.0, 0.8, 2.0, 0.5, "Verification\nwindow")

    _arrow(ax, 5.0, 7.95, 5.0, 7.45)
    _arrow(ax, 5.0, 6.95, 5.0, 6.45)
    _arrow(ax, 5.0, 5.95, 5.0, 5.4)
    _arrow(ax, 5.0, 4.6, 5.0, 4.05)
    _arrow(ax, 5.0, 3.55, 5.0, 3.05)
    _arrow(ax, 5.0, 2.55, 5.0, 2.05)
    _arrow(ax, 5.0, 1.55, 5.0, 1.05)

    # Decision branches
    _box(ax, 8.0, 0.8, 1.6, 0.5, "Accepted\n(keep change)")
    _box(ax, 8.0, 1.8, 1.6, 0.5, "Rejected\n(rollback + failure)")
    _box(ax, 8.0, 2.8, 1.6, 0.5, "Inconclusive\n(rollback, no penalty)")

    _diamond(ax, 8.0, 0.0, 2.2, 0.6, "2 consecutive\nfailures?")
    _box(ax, 9.5, 0.0, 1.0, 0.4, "Safe\nfallback")

    # Arrows for verdicts
    ax.annotate("", xy=(7.2, 0.8), xytext=(6.0, 0.8),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.text(6.6, 0.95, "good", ha="center", fontsize=6)

    ax.annotate("", xy=(7.2, 1.8), xytext=(6.0, 1.8),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.text(6.6, 1.95, "bad", ha="center", fontsize=6)

    ax.annotate("", xy=(7.2, 2.8), xytext=(6.0, 2.8),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.text(6.6, 2.95, "unclear", ha="center", fontsize=6)

    # Fallback
    ax.annotate("", xy=(8.0, 0.3), xytext=(8.0, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.annotate("", xy=(9.0, 0.0), xytext=(8.9, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.text(8.5, -0.2, "yes", ha="center", fontsize=6)

    # Rejected feedback
    ax.annotate("", xy=(9.2, 1.8), xytext=(9.2, 0.3),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8,
                                connectionstyle="arc3,rad=-0.5", linestyle="--"))
    ax.text(9.6, 1.0, "failure counter", ha="center", fontsize=5.5, fontstyle="italic")

    # No detection loop back
    ax.annotate("", xy=(8.5, 5.0), xytext=(8.5, 7.2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8,
                                connectionstyle="arc3,rad=-0.3", linestyle="--"))
    ax.text(9.3, 6.2, "no: continue\nsimulating", ha="center", fontsize=6, fontstyle="italic")

    # Record
    _box(ax, 2.0, 0.8, 1.6, 0.5, "Incident\nLedger")
    ax.annotate("", xy=(2.8, 1.05), xytext=(5.0, 1.05),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, linestyle=":"))
    ax.text(3.8, 1.3, "record every\ndecision", ha="center", fontsize=5.5, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_healing_loop.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_healing_loop.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Figure 3: Module Workflow - Workload Generation & Scheduling
# ============================================================================
def fig_workflow_m1():
    fig, ax = _base_fig(10, 6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title("Figure 3. Workload generation and scheduling workflow",
                 fontsize=FS_TITLE, pad=12)

    _box(ax, 1.5, 5.8, 2.0, 0.5, "Family + Seed\nselection")
    _box(ax, 4.5, 5.8, 2.0, 0.5, "WorkloadGenerator\n.generate()")
    _box(ax, 7.5, 5.8, 2.0, 0.5, "Workload object\n(fingerprinted)")

    _box(ax, 1.5, 4.5, 2.0, 0.5, "Optional: drift\nspec (drift.py)")
    _box(ax, 4.5, 4.5, 2.0, 0.5, "generate_with_\ndrift()")
    _box(ax, 7.5, 4.5, 2.0, 0.5, "Drifted Workload\n(phase shift)")

    _box(ax, 5.0, 3.0, 2.5, 0.5, "RuntimeSimulator\n.run(workload, ctrl)")
    _box(ax, 5.0, 1.8, 2.5, 0.5, "Event loop:\ndispatch, switch, complete")
    _box(ax, 5.0, 0.6, 2.5, 0.5, "ScheduleResult\n(per-process outcomes)")

    _arrow(ax, 2.5, 5.8, 3.5, 5.8)
    _arrow(ax, 5.5, 5.8, 6.5, 5.8)
    _arrow(ax, 2.5, 4.5, 3.5, 4.5)
    _arrow(ax, 5.5, 4.5, 6.5, 4.5)
    _arrow(ax, 7.5, 5.5, 6.0, 3.3)
    _arrow(ax, 5.0, 2.75, 5.0, 2.05)
    _arrow(ax, 5.0, 1.55, 5.0, 0.85)
    _arrow(ax, 6.5, 4.5, 6.0, 3.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_wf_m1.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_wf_m1.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Figure 4: Detection and Diagnosis Workflow
# ============================================================================
def fig_workflow_m2():
    fig, ax = _base_fig(10, 7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title("Figure 4. Degradation detection and diagnosis workflow",
                 fontsize=FS_TITLE, pad=12)

    _box(ax, 5.0, 7.0, 2.5, 0.5, "Completion notice\nfrom scheduler")
    _box(ax, 5.0, 6.0, 2.5, 0.5, "Update rolling window\n(EWMA ref + band)")
    _diamond(ax, 5.0, 4.8, 3.0, 0.9, "window_mean > ref\n+ k * band ?")
    _box(ax, 5.0, 3.5, 2.5, 0.5, "Hysteresis check\n(h consecutive)")
    _diamond(ax, 5.0, 2.3, 3.0, 0.9, "Gates pass?\n(cooldown, load, work)")
    _box(ax, 2.0, 2.3, 1.8, 0.5, "Build ready\nevidence")
    _box(ax, 2.0, 1.1, 1.8, 0.5, "Diagnose:\nprimary cause")
    _box(ax, 5.0, 1.1, 1.8, 0.5, "Declare\nincident")
    _box(ax, 8.0, 4.8, 1.5, 0.4, "Suppress\n(reason logged)")

    _arrow(ax, 5.0, 6.75, 5.0, 6.25)
    _arrow(ax, 5.0, 5.75, 5.0, 5.25)
    _arrow(ax, 5.0, 4.35, 5.0, 3.75)
    _arrow(ax, 5.0, 3.25, 5.0, 2.75)
    _arrow(ax, 3.5, 2.3, 2.9, 2.3)
    _arrow(ax, 2.0, 2.05, 2.0, 1.35)
    _arrow(ax, 2.9, 1.1, 4.1, 1.1)

    # Yes / No
    ax.text(5.6, 4.5, "yes", fontsize=6.5)
    ax.text(6.0, 4.8, "no", fontsize=6.5)
    ax.arrow(6.5, 4.8, 0.75, 0, head_width=0.1, head_length=0.08, fc="black", ec="black")

    # Gates pass
    ax.text(5.6, 2.0, "yes", fontsize=6.5)
    ax.text(6.0, 2.3, "no", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_wf_m2.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_wf_m2.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Figure 5: Verification and Acceptance Workflow
# ============================================================================
def fig_workflow_m3():
    fig, ax = _base_fig(10, 7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_title("Figure 5. Verification, acceptance, and rollback workflow",
                 fontsize=FS_TITLE, pad=12)

    _box(ax, 5.0, 7.0, 2.5, 0.5, "Action applied;\nwindow opened")
    _box(ax, 5.0, 6.0, 2.5, 0.5, "Collect completions\nover verify_decisions")
    _box(ax, 5.0, 5.0, 2.5, 0.5, "Compute phase-adjusted\nclosure residual")
    _diamond(ax, 5.0, 3.7, 3.0, 0.9, "residual >= accept\nboundary (+0.30)?")
    _diamond(ax, 5.0, 2.3, 3.0, 0.9, "residual <= reject\nboundary (-0.30)?")

    _box(ax, 8.5, 3.7, 1.5, 0.4, "ACCEPTED\n(keep, learn)")
    _box(ax, 8.5, 2.3, 1.5, 0.4, "REJECTED\n(rollback, fail +1)")
    _box(ax, 5.0, 1.0, 2.0, 0.4, "INCONCLUSIVE\n(rollback, no penalty)")

    _arrow(ax, 5.0, 6.75, 5.0, 6.25)
    _arrow(ax, 5.0, 5.75, 5.0, 5.25)
    _arrow(ax, 5.0, 4.75, 5.0, 4.15)
    _arrow(ax, 5.0, 3.25, 5.0, 2.75)
    _arrow(ax, 5.0, 1.85, 5.0, 1.2)

    # Yes for accept
    ax.annotate("", xy=(7.75, 3.7), xytext=(6.5, 3.7),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.text(7.0, 3.85, "yes", fontsize=6.5)

    # No, then reject check
    ax.text(5.6, 3.4, "no", fontsize=6.5)

    # Yes for reject
    ax.annotate("", xy=(7.75, 2.3), xytext=(6.5, 2.3),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=LW))
    ax.text(7.0, 2.45, "yes", fontsize=6.5)

    ax.text(5.6, 2.0, "no", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_wf_m3.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_wf_m3.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Figure 6: Frontend-to-Backend Simulation Flow
# ============================================================================
def fig_workflow_m4():
    fig, ax = _base_fig(10, 6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title("Figure 6. Frontend-to-backend simulation execution flow",
                 fontsize=FS_TITLE, pad=12)

    _box(ax, 1.5, 5.5, 2.2, 0.5, "Browser UI\n(app.js)")
    _box(ax, 4.5, 5.5, 2.2, 0.5, "HTTP Request\nPOST /api/simulations")
    _box(ax, 7.5, 5.5, 2.2, 0.5, "FastAPI Route\n(api.py)")
    _box(ax, 7.5, 4.2, 2.2, 0.5, "Service Layer\n(service.py)")
    _box(ax, 4.5, 4.2, 2.2, 0.5, "Simulation Engine\n(demo.run_fixture)")
    _box(ax, 1.5, 4.2, 2.2, 0.5, "Model Validator\n(model.py)")
    _box(ax, 1.5, 2.8, 2.2, 0.5, "JSON Response\n(workload, metrics,\nincidents)")
    _box(ax, 4.5, 2.8, 2.2, 0.5, "Charts\n(inline SVG)")
    _box(ax, 7.5, 2.8, 2.2, 0.5, "Render in UI\n(index.html)")

    _arrow(ax, 2.6, 5.5, 3.4, 5.5)
    _arrow(ax, 5.6, 5.5, 6.4, 5.5)
    _arrow(ax, 7.5, 5.2, 7.5, 4.45)
    _arrow(ax, 6.4, 4.2, 5.6, 4.2)
    _arrow(ax, 3.4, 4.2, 2.6, 4.2)
    _arrow(ax, 2.6, 2.8 + 0.95, 2.6, 3.05)
    _arrow(ax, 3.7, 2.8, 3.4, 2.8)
    _arrow(ax, 5.6, 2.8, 6.4, 2.8)

    # Return arrow
    ax.annotate("", xy=(8.5, 5.2), xytext=(8.5, 3.05),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8,
                                connectionstyle="arc3,rad=-0.3", linestyle="--"))
    ax.text(9.3, 4.2, "JSON +\nSVG", ha="center", fontsize=6, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_wf_m4.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_wf_m4.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Figure 7: Methodology Pipeline
# ============================================================================
def fig_methodology():
    fig, ax = _base_fig(10, 5)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_title("Figure 7. Experimental methodology pipeline",
                 fontsize=FS_TITLE, pad=12)

    stages = [
        (1.0, "Workload\nFamilies\n(7 families)"),
        (3.0, "Train / Valid /\nTest Split\n(seeds 7101-7105,\n8201, 23819)"),
        (5.2, "Arms\n(10 arms:\nfixed, rule,\nQ-learning)"),
        (7.4, "Evaluation\n(210 traces/arm,\nbootstrap 1000)"),
        (9.6, "Counterfactual\nAudit &\nReport"),
    ]
    for x, label in stages:
        _box(ax, x, 2.5, 1.8, 1.2, label, fontsize=7)

    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.9
        x2 = stages[i + 1][0] - 0.9
        _arrow(ax, x1, 2.5, x2, 2.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_methodology.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, "fig_methodology.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ============================================================================
# Generate all
# ============================================================================
if __name__ == "__main__":
    fig_architecture()
    fig_healing_loop()
    fig_workflow_m1()
    fig_workflow_m2()
    fig_workflow_m3()
    fig_workflow_m4()
    fig_methodology()
    print(f"Diagrams saved to {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")
