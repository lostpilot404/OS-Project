"""Reconciliation: docs and summary must agree on canonical percentages."""

import json
import pathlib
import re

from healing.canonical_numbers import (
    CERTIFIABLE_AT_ACCEPT,
    CERTIFIABLE_AT_NULL_P95,
    CERTIFIABLE_SHARE_AT_ACCEPT,
    CERTIFIABLE_SHARE_AT_NULL,
    HELPFUL,
    HARMFUL,
    RAW_SPREAD_SD,
    RESIDUAL_R2,
    RESIDUAL_SPREAD_SD,
    TOTAL_INTERVENTIONS,
)


def test_summary_discrimination_matches_canonical_counts() -> None:
    summary = json.loads(pathlib.Path("results/healing/healing_summary.json").read_text(encoding="utf-8"))
    disc = summary["verification_discrimination"]
    assert disc["helped"] == HELPFUL
    assert disc["n_interventions"] == TOTAL_INTERVENTIONS
    assert disc["helped"] + disc["harmful"] + disc["zero_effect"] == TOTAL_INTERVENTIONS
    # Fractions stored in JSON must match the canonical derivation within 0.1 pp
    assert abs(float(disc["certifiable_share_of_helpful_at_null_p95"]) - CERTIFIABLE_SHARE_AT_NULL) < 0.002
    assert abs(float(disc["certifiable_share_of_helpful_at_accept_residual_factor"]) - CERTIFIABLE_SHARE_AT_ACCEPT) < 0.002
# Threshold derivation spreads
    deriv = summary["threshold_derivation"]
    assert abs(float(deriv["baseline_residual_sd"]) - RESIDUAL_SPREAD_SD) < 0.005
    assert abs(float(deriv["all_events_residual_sd"]) - RESIDUAL_SPREAD_SD) < 0.005


def test_docs_do_not_contain_stale_83_9_percent() -> None:
    # Every remaining occurrence of the old 83.9% would be a drift.
    for path in pathlib.Path("docs").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "83.9" not in text, f"stale 83.9% in {path}"
    assert "83.9" not in pathlib.Path("README.md").read_text(encoding="utf-8")


def test_docs_use_correct_base_rate() -> None:
    # Spot-check that the three reconciled files now state 83.5% base rate.
    for fname in ["docs/EVALUATION_RESULTS.md", "docs/DESIGN_AND_CHOICES.md", "docs/LIMITATIONS.md", "README.md"]:
        text = pathlib.Path(fname).read_text(encoding="utf-8")
        assert "83.5" in text, f"expected 83.5% in {fname}"


def test_healing_config_comment_uses_canonical_spread() -> None:
    text = pathlib.Path("healing/config.py").read_text(encoding="utf-8")
    assert "0.160" in text
    # Old value must be gone
    assert "0.34 and more than doubles" not in text


def test_canonical_percentages_are_internally_consistent() -> None:
    # Numerators/denominators must produce the quoted percentages
    assert round(HELPFUL / TOTAL_INTERVENTIONS * 100, 1) == 83.5
    assert round(CERTIFIABLE_AT_NULL_P95 / HELPFUL * 100, 1) == 6.7
    assert round(CERTIFIABLE_AT_ACCEPT / HELPFUL * 100, 1) == 8.0
    assert round(HARMFUL / TOTAL_INTERVENTIONS * 100, 1) == 14.0
