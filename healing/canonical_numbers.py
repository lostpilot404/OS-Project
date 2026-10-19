"""Canonical numerical claims derived from ``results/healing/healing_summary.json``.

This module is the single source of truth for the headline percentages and
calibration constants that appear in README, docs/* and the Review2 report.
Documentation text should be updated to match the values exported here; the
automated checks in :mod:`tests.test_canonical_numbers` enforce that the prose
does not drift from the evidence on disk.

All percentages are stored as fractions and as pre-formatted strings so both
code and documentation generators can import the same constants.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

SUMMARY_PATH = pathlib.Path("results/healing/healing_summary.json")

# Canonical counts taken from ``verification_discrimination`` in the summary.
# These are frozen from the final_test split unless a methodological defect is
# found; do not edit them by hand — regenerate the summary instead.
HELPFUL = 812
HARMFUL = 136
ZERO_EFFECT = 25
TOTAL_INTERVENTIONS = 973  # 812 + 136 + 25
ACCEPTED = 65  # operative boundary, all helpful
REJECTED = 34  # 26 harmful of 34
CERTIFIABLE_AT_NULL_P95 = 54  # unconditional 95th percentile
CERTIFIABLE_AT_ACCEPT = 65  # phase-adjusted operative boundary

# Derived fractions
HELPFUL_BASE_RATE = HELPFUL / TOTAL_INTERVENTIONS  # 0.8345 -> 83.5%
HARMFUL_BASE_RATE = HARMFUL / TOTAL_INTERVENTIONS  # 0.1398 -> 14.0%
CERTIFIABLE_SHARE_AT_NULL = CERTIFIABLE_AT_NULL_P95 / HELPFUL  # 0.0665 -> 6.7%
CERTIFIABLE_SHARE_AT_ACCEPT = CERTIFIABLE_AT_ACCEPT / HELPFUL  # 0.0800 -> 8.0%
REJECTED_HARMFUL_RATE = 26 / REJECTED  # 0.7647 -> 76.5%

# Calibration spreads
RAW_SPREAD_SD = 0.806
RESIDUAL_SPREAD_SD = 0.160  # canonical: threshold_derivation baseline_residual_sd 0.1599
RESIDUAL_R2 = 0.961  # 0.9606 rounded to three decimals
PHASE_MODEL_VARIANCE_REMOVED = 0.802  # 80.2%

# Human-readable formatted versions used in prose and figure captions
FMT_HELPFUL_BASE_RATE = "83.5%"
FMT_HARMFUL_BASE_RATE = "14.0%"
FMT_CERTIFIABLE_AT_NULL = "6.7%"
FMT_CERTIFIABLE_AT_ACCEPT = "8.0%"
FMT_REJECTED_HARMFUL = "76.5%"
FMT_RAW_SPREAD = "0.806"
FMT_RESIDUAL_SPREAD = "0.160"


@dataclass(frozen=True)
class VerificationDiscrimination:
    helpful: int
    harmful: int
    zero_effect: int
    total: int
    accepted: int
    certifiable_at_null_p95: int
    certifiable_at_accept: int


CANONICAL = VerificationDiscrimination(
    helpful=HELPFUL,
    harmful=HARMFUL,
    zero_effect=ZERO_EFFECT,
    total=TOTAL_INTERVENTIONS,
    accepted=ACCEPTED,
    certifiable_at_null_p95=CERTIFIABLE_AT_NULL_P95,
    certifiable_at_accept=CERTIFIABLE_AT_ACCEPT,
)


def load_summary_discrimination(path: pathlib.Path | str = SUMMARY_PATH) -> dict:
    """Load ``verification_discrimination`` from the on-disk summary."""
    summary = json.loads(pathlib.Path(path).read_text())
    return summary["verification_discrimination"]


def assert_summary_matches_canonical(path: pathlib.Path | str = SUMMARY_PATH) -> None:
    """Raise :class:`AssertionError` if the on-disk summary drifts from the canonical counts."""
    disc = load_summary_discrimination(path)
    assert disc["helped"] == HELPFUL, f"helped {disc['helped']} != {HELPFUL}"
    assert disc["harmful"] == HARMFUL, f"harmful {disc['harmful']} != {HARMFUL}"
    assert disc["zero_effect"] == ZERO_EFFECT
    assert disc["n_interventions"] == TOTAL_INTERVENTIONS
    got_null = float(disc.get("certifiable_share_of_helpful_at_null_p95", 0))
    assert abs(got_null - CERTIFIABLE_SHARE_AT_NULL) < 1e-3, f"null share {got_null} != {CERTIFIABLE_SHARE_AT_NULL}"
    got_accept = float(disc.get("certifiable_share_of_helpful_at_accept_residual_factor", 0))
    assert abs(got_accept - CERTIFIABLE_SHARE_AT_ACCEPT) < 1e-3
