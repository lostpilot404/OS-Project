"""Centralized figure style for all published plots.

Every figure in ``figures/`` must use this module so that DPI, font sizes,
colour palettes and axis styling stay consistent. The test
``tests/test_visual_style.py`` checks that no figure module hard-codes a
divergent palette.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Canonical rendering parameters — change here, not in individual plot modules.
DPI = 140
FIGSIZE_WIDE = (13, 4.8)
FIGSIZE_SINGLE = (6.5, 4.5)
FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 11
LABEL_SIZE = 10
TICK_SIZE = 8
LEGEND_SIZE = 8

# Shared palette: map arm / diagnosis names to colours so the same arm looks
# identical in every figure.
PALETTE = {
    # Healing experiment arms
    "fixed_fcfs": "#2c3e50",
    "fixed_sjf": "#7fb069",
    "detector_only": "#95a5a6",
    "rule_fairness": "#e74c3c",
    "rule_wait": "#2980b9",
    "q_recovery": "#8e44ad",
    # Offline study arms
    "ql_orig": "#4c72b0",
    "ql_risk": "#c44e52",
    "ql_risk_origstate": "#dd8452",
    "orig_fingerprint": "#937860",
    "ql_orig_fingerprint": "#937860",
    "best_fixed": "#55a868",
    "fingerprint_rule": "#8172b3",
    "random": "#ccb974",
    # Diagnosis colours (aligned with DIAGNOSIS_CATEGORIES)
    "head_of_line_blocking": "#2980b9",
    "fairness_starvation": "#e74c3c",
    "tail_dispersion": "#f39c12",
    "priority_ordering_anomaly": "#8e44ad",
    "queue_congestion": "#16a085",
    "unexplained": "#95a5a6",
    # Generic
    "primary": "#3f6fb5",
    "accent": "#c0392b",
    "neutral": "#7f8c8d",
}

_HEALING_ORDER = (
    "fixed_fcfs",
    "fixed_sjf",
    "detector_only",
    "rule_fairness",
    "rule_wait",
    "q_recovery",
)


def apply_style() -> None:
    """Apply the canonical Matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": FONT_FAMILY,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def color_for(key: str, fallback: str = "#333333") -> str:
    return PALETTE.get(key, fallback)
