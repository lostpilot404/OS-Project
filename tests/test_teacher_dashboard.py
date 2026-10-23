"""Tests of the teacher dashboard: validation, labelling, safety and generated HTML.

The dashboard is a consumer of recorded output, so these tests check three things:

* it refuses input it cannot draw truthfully (missing, malformed, inconsistent);
* labels, verdicts and comparisons follow the recorded values rather than the seed;
* the generated file is self-contained, escaped and written atomically.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from dashboard.build import PROTECTED_DIRECTORIES, REPOSITORY_ROOT, build_dashboard
from dashboard.cli import main as cli_main
from dashboard.model import DashboardError, parse_demo
from dashboard.render import format_number, format_signed
from dashboard.research import load_research
from dashboard.teaching import build_teaching_panel
from main import main as demo_main

ROOT = REPOSITORY_ROOT
RESEARCH = ROOT / "results/healing/healing_summary.json"


@pytest.fixture(scope="module")
def demo_payload(tmp_path_factory) -> dict:
    """The real demo payload, produced once by the presentation command."""
    target = tmp_path_factory.mktemp("dashboard-demo")
    assert demo_main(["demo", "--results-dir", str(target)]) == 0
    payload = json.loads((target / "demo.json").read_text(encoding="utf-8"))
    yield payload


@pytest.fixture()
def demo_file(tmp_path, demo_payload) -> Path:
    path = tmp_path / "demo.json"
    path.write_text(json.dumps(demo_payload), encoding="utf-8")
    return path


@pytest.fixture()
def built(tmp_path, demo_file) -> Path:
    output = tmp_path / "index.html"
    build_dashboard(input_path=demo_file, output_path=output)
    return output


# --------------------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------------------


def test_valid_demo_json_parses_into_typed_scenarios(demo_payload):
    document = parse_demo(demo_payload, source="demo.json")
    assert [fixture.seed for fixture in document.fixtures] == [3, 13]
    for fixture in document.fixtures:
        assert len(fixture.processes) == 15
        assert len(fixture.outcomes) == 15
        assert len(fixture.intervals) == 15
        assert fixture.metrics.num_processes == 15
        # The stored aggregate equals the recomputation from the per-process rows.
        assert fixture.total_waiting_from_outcomes == fixture.metrics.total_waiting_time
        assert fixture.mean_waiting_from_outcomes == pytest.approx(fixture.metrics.avg_waiting_time)
        # Every recorded dispatch is contiguous with the trace it belongs to.
        assert fixture.intervals[0].start == min(row.start_time for row in fixture.outcomes)


def test_missing_file_is_reported_and_nothing_is_written(tmp_path):
    output = tmp_path / "index.html"
    with pytest.raises(DashboardError, match="demo input not found"):
        build_dashboard(input_path=tmp_path / "absent.json", output_path=output)
    assert not output.exists()


def test_malformed_json_reports_position(tmp_path):
    source = tmp_path / "demo.json"
    source.write_text('{"fixtures": [}', encoding="utf-8")
    with pytest.raises(DashboardError, match="invalid JSON at line 1"):
        build_dashboard(input_path=source, output_path=tmp_path / "index.html")
    assert not (tmp_path / "index.html").exists()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.pop("fixtures"), "missing required field 'fixtures'"),
        (lambda data: data["fixtures"][0].pop("metrics"), "missing required field 'metrics'"),
        (
            lambda data: data["fixtures"][0]["metrics"].pop("avg_waiting_time"),
            "missing metric field",
        ),
        (
            lambda data: data["fixtures"][0]["metrics"].__setitem__("avg_waiting_time", 999.0),
            "avg_waiting_time",
        ),
        (
            lambda data: data["fixtures"][0].pop("fcfs_metrics"),
            "fcfs_metrics",
        ),
        (
            lambda data: data["fixtures"][0]["incidents"][0].__setitem__("verdict", "probably-ok"),
            "unknown verdict",
        ),
        (
            lambda data: data["fixtures"][0]["incidents"][0].__setitem__(
                "action_name", "set_everything"
            ),
            "unknown recovery action",
        ),
        (
            lambda data: data["fixtures"][0]["incidents"][0].__setitem__(
                "settings_after_policy", "EDF"
            ),
            "unknown policy",
        ),
        (
            lambda data: data["fixtures"][0]["decisions"][1].__setitem__("start", 0),
            "starts before the previous interval ended",
        ),
        (
            lambda data: data["fixtures"][0]["outcomes"][0].__setitem__("completion_time", 9999),
            "last dispatch ends at 65 but the outcome completes at 9999",
        ),
        (
            lambda data: data["fixtures"][0]["metrics"].__setitem__(
                "total_waiting_time",
                data["fixtures"][0]["metrics"]["total_waiting_time"] + 5,
            ),
            "total_waiting_time",
        ),
        (
            lambda data: data["fixtures"][0].__setitem__("fingerprint", "0000"),
            "workload_fingerprint",
        ),
    ],
)
def test_inconsistent_payloads_are_rejected(demo_payload, mutate, message):
    payload = copy.deepcopy(demo_payload)
    mutate(payload)
    with pytest.raises(DashboardError, match=message):
        parse_demo(payload, source="demo.json")


def test_absent_metric_fields_are_named(demo_payload):
    payload = copy.deepcopy(demo_payload)
    for name in ("cpu_utilization", "context_switches", "total_elapsed_time"):
        payload["fixtures"][0]["metrics"].pop(name)
    with pytest.raises(DashboardError, match="cpu_utilization, context_switches, total_elapsed_time"):
        parse_demo(payload)


def test_optional_metric_fields_may_be_absent(demo_payload):
    payload = copy.deepcopy(demo_payload)
    metrics = payload["fixtures"][0]["metrics"]
    for name in ("cpu_busy_time", "cpu_idle_time", "throughput", "total_response_time"):
        metrics.pop(name)
    document = parse_demo(payload)
    fixture = document.fixtures[0]
    assert fixture.metrics.cpu_busy_time is None
    assert fixture.metrics.has_runtime_detail is False


def test_confirmed_failure_must_match_the_rejected_verdict(demo_payload):
    payload = copy.deepcopy(demo_payload)
    incident = payload["fixtures"][0]["incidents"][0]  # inconclusive
    incident["confirmed_failure"] = True
    with pytest.raises(DashboardError, match="confirmed_failure must be true exactly"):
        parse_demo(payload)


def test_incident_outside_the_trace_is_rejected(demo_payload):
    payload = copy.deepcopy(demo_payload)
    payload["fixtures"][0]["incidents"][0]["decision_index"] = 99
    with pytest.raises(DashboardError, match="outside the recorded trace"):
        parse_demo(payload)


def test_rollback_must_match_the_trace_ending(demo_payload):
    payload = copy.deepcopy(demo_payload)
    payload["fixtures"][0]["incidents"][0]["rollback_target_policy"] = "SJF"
    with pytest.raises(DashboardError, match="records a rollback to SJF"):
        parse_demo(payload)


# --------------------------------------------------------------------------------------
# Labels, verdicts and comparisons
# --------------------------------------------------------------------------------------


def test_policy_and_verdict_labels_follow_the_records(demo_payload):
    document = parse_demo(demo_payload)
    first, second = document.fixtures
    assert first.primary_incident.verdict == "inconclusive"
    assert first.primary_incident.action_name == "set_sjf"
    assert first.primary_incident.settings_before_policy == "FCFS"
    assert first.final_policy == "FCFS"
    assert first.primary_incident.rollback_target_policy == "FCFS"

    assert second.primary_incident.verdict == "accepted"
    assert second.primary_incident.action_name == "set_sjf"
    assert second.final_policy == "SJF"
    assert second.primary_incident.rollback_target_policy == ""
    # An inconclusive window is not a confirmed failure; an accepted one is not either.
    assert first.primary_incident.confirmed_failure is False
    assert second.primary_incident.confirmed_failure is False
    assert first.primary_incident.is_error is False


def test_waiting_time_comparisons_match_the_recorded_metrics(demo_payload):
    document = parse_demo(demo_payload)
    for fixture in document.fixtures:
        metrics, baseline = fixture.metrics, fixture.fcfs_metrics
        assert fixture.mean_waiting_difference == pytest.approx(
            metrics.avg_waiting_time - baseline.avg_waiting_time
        )
        assert fixture.mean_waiting_reduction_fraction == pytest.approx(
            (baseline.avg_waiting_time - metrics.avg_waiting_time) / baseline.avg_waiting_time
        )
        # Both runs must be the same workload: the demo says so and the fingerprint agrees.
        assert metrics.workload_fingerprint == baseline.workload_fingerprint == fixture.fingerprint
        # Per-process rows reproduce the aggregate mean exactly.
        assert sum(row.waiting_time for row in fixture.outcomes) / len(fixture.outcomes) == pytest.approx(
            metrics.avg_waiting_time
        )
        assert fixture.total_idle == sum(interval.idle_before for interval in fixture.intervals)


def test_formatting_never_invents_values():
    assert format_number(None) == "n/a"
    assert format_number(float("nan")) == "n/a"
    assert format_number(130.46666666666667) == "130.47"
    assert format_signed(0.0) == "0.00"
    assert format_signed(0.001) == "0.00"
    assert format_signed(-20.1333333) == "-20.13"
    assert format_signed(None) == "n/a"


def test_teaching_example_runs_the_real_schedulers():
    panel = build_teaching_panel()
    assert panel.hand_verified is True
    names = [policy.name for policy in panel.policies]
    assert names == ["FCFS", "SJF", "Round Robin", "Priority"]
    by_name = {policy.name: policy for policy in panel.policies}
    assert [row.waiting_time for row in by_name["FCFS"].outcomes] == [0, 5, 8]
    assert [row.waiting_time for row in by_name["SJF"].outcomes] == [4, 1, 0]
    assert [row.waiting_time for row in by_name["Priority"].outcomes] == [3, 0, 8]
    assert [row.waiting_time for row in by_name["Round Robin"].outcomes] == [4, 5, 4]
    assert by_name["Round Robin"].context_switches == 5
    assert panel.best_by_waiting == "SJF"
    assert panel.quantum == 2 and panel.switching_cost == 0


# --------------------------------------------------------------------------------------
# Research panel
# --------------------------------------------------------------------------------------


def test_research_summary_is_read_from_the_committed_artifacts():
    research = load_research(RESEARCH)
    assert research.split == "final_test"
    assert research.workloads is not None and research.workloads > 0
    fcfs = research.arm("fixed_fcfs")
    sjf = research.arm("fixed_sjf")
    rule = research.arm("rule_wait")
    learned = research.arm("q_recovery")
    assert sjf.mean_waiting_time < fcfs.mean_waiting_time
    assert rule.mean_waiting_time < fcfs.mean_waiting_time
    assert learned.mean_waiting_time < fcfs.mean_waiting_time
    assert research.best_primary.arm == "fixed_sjf"
    comparison = research.q_vs_rule
    assert comparison is not None
    assert comparison.difference == pytest.approx(
        learned.mean_waiting_time - rule.mean_waiting_time
    )
    assert comparison.excludes_zero is False  # the artifact's interval includes zero
    findings = " ".join(research.finding_lines())
    assert "not measurably better" in findings
    assert "improve on FCFS" in findings


def test_missing_research_summary_is_actionable(tmp_path):
    with pytest.raises(DashboardError, match="research summary not found"):
        load_research(tmp_path / "healing_summary.json")


def test_research_summary_without_required_arms_is_rejected(tmp_path):
    payload = json.loads(RESEARCH.read_text(encoding="utf-8"))
    payload["means"] = [row for row in payload["means"] if row["arm"] != "rule_wait"]
    path = tmp_path / "healing_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DashboardError, match="missing the arm"):
        load_research(path)


# --------------------------------------------------------------------------------------
# Output path safety and failure handling
# --------------------------------------------------------------------------------------


def test_output_must_be_html(tmp_path, demo_file):
    with pytest.raises(DashboardError, match="output must be an .html file"):
        build_dashboard(input_path=demo_file, output_path=tmp_path / "index.txt")


def test_output_may_not_be_the_input(tmp_path, demo_file):
    same_path = tmp_path / "demo.html"
    same_path.write_bytes(demo_file.read_bytes())
    before = same_path.read_bytes()
    with pytest.raises(DashboardError, match="output path is the input demo file"):
        build_dashboard(input_path=same_path, output_path=same_path)
    assert same_path.read_bytes() == before


def test_output_may_not_overwrite_an_existing_dashboard(tmp_path, demo_file):
    output = tmp_path / "index.html"
    output.write_text("previous dashboard", encoding="utf-8")
    with pytest.raises(DashboardError, match="already exists"):
        build_dashboard(input_path=demo_file, output_path=output)
    assert output.read_text(encoding="utf-8") == "previous dashboard"
    result = build_dashboard(input_path=demo_file, output_path=output, force=True)
    assert result.bytes_written > 0
    assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_output_may_not_land_in_research_directories(demo_file):
    protected = PROTECTED_DIRECTORIES[0]
    with pytest.raises(DashboardError, match="refusing to write into the research evidence directory"):
        build_dashboard(input_path=demo_file, output_path=protected / "index.html")
    assert not (protected / "index.html").exists()


def test_failed_build_leaves_the_destination_untouched(tmp_path, demo_payload):
    source = tmp_path / "demo.json"
    broken = copy.deepcopy(demo_payload)
    broken["fixtures"][1]["metrics"]["avg_waiting_time"] = 1.0
    source.write_text(json.dumps(broken), encoding="utf-8")
    output = tmp_path / "index.html"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(DashboardError):
        build_dashboard(input_path=source, output_path=output, force=True)
    assert output.read_text(encoding="utf-8") == "keep me"
    assert list(tmp_path.glob(".dashboard-*.tmp")) == []


def test_cli_reports_errors_without_a_traceback(tmp_path, capsys):
    exit_code = cli_main(["--input", str(tmp_path / "missing.json"), "--output", str(tmp_path / "i.html")])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: demo input not found" in captured.err
    assert "Traceback" not in captured.err


def test_cli_defaults_output_next_to_the_input(tmp_path, demo_file, capsys):
    assert cli_main(["--input", str(demo_file)]) == 0
    captured = capsys.readouterr()
    assert "Teacher dashboard written" in captured.out
    assert (tmp_path / "index.html").exists()


# --------------------------------------------------------------------------------------
# Generated HTML
# --------------------------------------------------------------------------------------


def test_dashboard_html_is_self_contained(built: Path):
    html = built.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html and "</style>" in html
    assert re.search(r"<svg[^>]+viewBox", html)
    # No external resources of any kind: the page must open offline from a file path.
    assert "http://" not in html and "https://" not in html
    assert "src=" not in html
    assert "<link" not in html
    assert "@import" not in html
    assert "cdn." not in html.lower()
    assert "integrity=" not in html.lower()


def test_timeline_blocks_carry_process_details_for_inspection(built: Path):
    html = built.read_text(encoding="utf-8")
    chart = re.search(r'<svg[^>]+aria-labelledby="gantt-3-title".*?</svg>', html, re.S).group(0)
    block = re.search(r'<g class="slice".*?</g>', chart, re.S).group(0)
    for name in ("data-pid", "data-policy", "data-start", "data-end", "data-duration",
                 "data-switch", "data-waiting", "data-burst", "data-priority",
                 "data-arrival", "data-completion", "data-process-waiting"):
        assert name + "=" in block, name
    # The first recorded dispatch is P4 [0,12): burst 12, priority 2, arrives at 0.
    assert 'data-pid="4"' in block and 'data-start="0"' in block and 'data-end="12"' in block
    assert 'data-burst="12"' in block and 'data-priority="2"' in block


def test_dashboard_markup_has_unique_ids_and_resolvable_references(built: Path):
    import collections

    html = built.read_text(encoding="utf-8")
    ids = re.findall(r'\sid="([^"]+)"', html)
    duplicates = [value for value, count in collections.Counter(ids).items() if count > 1]
    assert duplicates == []
    for reference in set(re.findall(r"url\(#([^)]+)\)", html)):
        assert f'id="{reference}"' in html, f"unresolved SVG reference #{reference}"


def test_dashboard_embeds_every_chart_it_claims(built: Path):
    html = built.read_text(encoding="utf-8")
    for seed in (3, 13):
        assert f'id="gantt-{seed}-title"' in html
        assert f'id="lifecycle-{seed}-title"' in html
        assert f'id="band-{seed}-title"' in html
        assert f'data-readout="{seed}"' in html
    for index in range(4):
        assert f'data-readout="{100 + index}"' in html
    assert 'id="teaching-waiting-title"' in html
    assert 'id="research-waiting-title"' in html
    for index in range(4):
        assert f'id="teaching-{index}-title"' in html
    # Every recorded dispatch is drawn twice per scenario (CPU lane and process lane).
    for seed in (3, 13):
        chart = re.search(
            rf'<svg[^>]+aria-labelledby="gantt-{seed}-title".*?</svg>', html, re.S
        ).group(0)
        assert chart.count('class="slice"') == 2 * 15


def test_build_is_deterministic(tmp_path, demo_file):
    """Identical input must produce an identical file: no timestamps, no randomness."""
    first = build_dashboard(input_path=demo_file, output_path=tmp_path / "a.html")
    second = build_dashboard(input_path=demo_file, output_path=tmp_path / "b.html")
    assert first.sha256 == second.sha256
    assert (tmp_path / "a.html").read_bytes() == (tmp_path / "b.html").read_bytes()


def test_dashboard_states_both_seed_outcomes_differently(built: Path):
    html = built.read_text(encoding="utf-8")
    assert "rolled back to FCFS" in html
    assert "SJF kept in force" in html
    assert "Inconclusive is not rejected" in html
    assert "Accepted." in html
    assert "replay recorded runs" in html
    # The two fixture means and the FCFS comparison are rendered from the payload.
    assert "130.47" in html and "150.60" in html
    assert "220.53" in html and "255.80" in html


def test_dashboard_separates_demo_fixtures_from_research(built: Path):
    html = built.read_text(encoding="utf-8")
    assert 'id="research"' in html
    assert "frozen held-out study" in html
    assert "not measurably better" in html
    assert "Q-learning recovery" in html
    # The two demo seeds are not presented as study arms.
    research_section = html.split('id="research"', 1)[1].split('id="limits"', 1)[0]
    assert "seed 3" not in research_section.lower()
    assert "seed 13" not in research_section.lower()


def test_embedding_escapes_hostile_values(tmp_path, demo_payload):
    payload = copy.deepcopy(demo_payload)
    payload["scope"] = '</script><script>alert("scope")</script>'
    payload["fixtures"][0]["family"] = 'mixed <img src=x onerror="alert(1)">'
    payload["fixtures"][0]["incidents"][0]["verifier_reason"] = "<b>injected</b> & done"
    source = tmp_path / "demo.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "index.html"
    build_dashboard(input_path=source, output_path=output)
    html = output.read_text(encoding="utf-8")
    assert "<script>alert(" not in html
    assert '<img src=x onerror' not in html
    assert "<b>injected</b>" not in html
    assert "&lt;img src=x onerror=" in html
    assert "&lt;b&gt;injected&lt;/b&gt; &amp; done" in html
    # The embedded JSON payload cannot terminate its script element.
    payload_text = re.search(
        r'<script id="dashboard-data" type="application/json">(.*?)</script>', html, re.S
    ).group(1)
    assert "</script" not in payload_text
    assert "\\u003c/script" in payload_text
    json.loads(payload_text)  # still valid JSON


def test_dashboard_renders_without_research_or_javascript(built: Path):
    html = built.read_text(encoding="utf-8")
    assert "<noscript>" in html
    # Default: every scenario panel (scheduling, lifecycle, performance) is in the
    # document; the script only decides which one is visible.
    assert html.count('data-panel-scenario="3"') == 3
    assert html.count('data-panel-scenario="13"') == 3
    assert 'data-panel-mode="demo"' in html
    assert 'data-panel-mode="research"' in html


def test_no_research_build_still_produces_a_complete_page(tmp_path, demo_file):
    output = tmp_path / "index.html"
    result = build_dashboard(
        input_path=demo_file, output_path=output, include_research=False
    )
    html = output.read_text(encoding="utf-8")
    assert result.research_included is False
    assert "was not loaded for this build" in html
    assert "research-waiting-title" not in html
    assert "<svg" in html


# --------------------------------------------------------------------------------------
# The documented command keeps working
# --------------------------------------------------------------------------------------


def test_builder_script_runs_as_documented(tmp_path, demo_file):
    import subprocess
    import sys

    output = tmp_path / "index.html"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_teacher_dashboard.py"),
            "--input",
            str(demo_file),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Teacher dashboard written" in completed.stdout
    assert output.exists()


def test_demo_command_is_unchanged(tmp_path, capsys):
    target = tmp_path / "demo"
    assert demo_main(["demo", "--results-dir", str(target)]) == 0
    text = capsys.readouterr().out
    assert "Verdict=accepted" in text
    assert "Verdict=inconclusive" in text
    assert (target / "demo.json").exists()
    # The builder adds the dashboard next to the trace and never touches the trace itself.
    before = (target / "demo.json").read_bytes()
    build_dashboard(input_path=target / "demo.json", output_path=target / "index.html")
    assert (target / "index.html").exists()
    assert before == (target / "demo.json").read_bytes()
