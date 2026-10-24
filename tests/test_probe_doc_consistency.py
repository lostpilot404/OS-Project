"""Regression: docs must not present 21-incident/168-row stats as canonical probe results."""

import pathlib
import re

CANONICAL_ROWS = 784
CANONICAL_INCIDENTS = 98
LEGACY_ROWS = 168
LEGACY_INCIDENTS = 21

DOCS = [
    "docs/EVALUATION_RESULTS.md",
    "docs/DESIGN_AND_CHOICES.md",
    "docs/LIMITATIONS.md",
    "docs/ENGINEERING_COMPLETION_REPORT.md",
    "tools/review2_content.py",
]

def _load(path):
    return pathlib.Path(path).read_text(encoding="utf-8")

def test_no_unlabeled_21_incident_canonical_claim():
    # Any mention of 21 incidents or 168 rows as if it were the current canonical probe is stale.
    # Historical mentions are allowed only when they explicitly cite the legacy artifact.
    for doc in DOCS:
        text = _load(doc)
        # Find sentences that mention "21 incidents" without legacy marker nearby
        for m in re.finditer(r"21 incidents", text):
            # Look back 200 chars and forward 200 chars for legacy marker
            start = max(0, m.start()-300)
            end = min(len(text), m.end()+300)
            snippet = text[start:end]
            # Must contain legacy marker or historical qualifier
            assert "legacy" in snippet.lower() or "historical" in snippet.lower() or "legacy_raw_proxy" in snippet, \
                f"{doc}: unlabeled 21-incident claim at {m.start()}: {snippet[:200]}"
        for m in re.finditer(r"168 rows", text):
            start = max(0, m.start()-300)
            end = min(len(text), m.end()+300)
            snippet = text[start:end]
            assert "legacy" in snippet.lower() or "historical" in snippet.lower(), \
                f"{doc}: unlabeled 168-row claim"

def test_canonical_probe_rows_match_doc_claims():
    # Docs that claim canonical 98 incidents ×8 =784 rows must match actual file
    probe = pathlib.Path("results/healing/healing_action_probe.csv")
    assert probe.exists()
    import pandas as pd
    df = pd.read_csv(probe)
    assert len(df) == CANONICAL_ROWS, f"canonical rows {len(df)} != {CANONICAL_ROWS}"
    assert len(df)//8 == CANONICAL_INCIDENTS
    # Ensure docs mention 784 or 98 when they claim canonical
    # Check EVALUATION_RESULTS.md Section 7 header must mention 784
    text = _load("docs/EVALUATION_RESULTS.md")
    assert "784 rows" in text or "784" in text, "EVALUATION_RESULTS should mention canonical 784 rows"
    assert "98" in text and "validation" in text.lower(), "EVALUATION should mention 98 validation incidents"

def test_legacy_file_preserved_and_distinct():
    import pandas as pd
    legacy = pathlib.Path("results/healing/healing_action_probe.legacy_raw_proxy.csv")
    assert legacy.exists()
    df = pd.read_csv(legacy)
    assert len(df) == LEGACY_ROWS
    assert "controller_reward" not in df.columns
    assert "reward_band_units" in df.columns

def test_report_arithmetic_correct():
    text = _load("docs/ENGINEERING_COMPLETION_REPORT.md")
    assert "98/140 = 0.70" in text, "report must state 98/140=0.70 not 0.62"
    assert "0.623" in text or "0.623" in text.lower() or "control_incidents_per_trace" in text, "training control rate should be distinguished"
    # Must not conflate
    assert "140 traces × 0.62" not in text, "stale 0.62 conflation must be removed"

def test_reward_distinction_clear():
    # Docs must distinguish raw closure, cost-including raw proxy, phase-adjusted residual, controller_reward
    for doc in ["docs/EVALUATION_RESULTS.md", "docs/DESIGN_AND_CHOICES.md"]:
        text = _load(doc)
        assert "controller_reward" in text, f"{doc} must mention controller_reward"
        assert "raw_load_improvement" in text or "raw proxy" in text.lower(), f"{doc} must mention raw proxy"
        assert "phase-adjusted" in text.lower() or "phase-adjusted" in text, f"{doc} must mention phase-adjusted residual"
