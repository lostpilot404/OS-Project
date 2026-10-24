"""Regression tests for the interface and contract defects found in the 2026 audit.

Each test pins a concrete behaviour that was wrong before the audit and would silently rot
again: the recovery page crashed on a healing run that declared no incident, an inconclusive
rollback was painted with the success colour, the teaching payload carried placeholder FCFS
metrics, a 404 for an aged-out run lost its error type, a malformed body reported a character
offset instead of the body, and a custom workload accepted a policy the loop would ignore.

Browser behaviour cannot be proved here; these tests protect the data the browser renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.api import RunRegistry, create_app
from dashboard.charts import verdict_tone_key
from dashboard.service import STATE_SPACE_SIZE


def _app_client() -> TestClient:
    registry = RunRegistry(limit=4)
    return TestClient(create_app(registry))


def test_healing_run_without_an_incident_has_an_explicit_empty_lifecycle():
    """The Recovery page must be able to render the payload of a run that never triggered."""
    with _app_client() as client:
        response = client.post(
            "/api/simulations",
            json={
                "scenario": "custom",
                "options": {
                    "processes": [
                        {"pid": 1, "arrival_time": 0, "burst_time": 6, "priority": 3},
                        {"pid": 2, "arrival_time": 1, "burst_time": 2, "priority": 1},
                    ],
                    "healing": True,
                },
            },
        )
    assert response.status_code == 201
    healing = response.json()["result"]["healing"]
    assert healing["triggered"] is False
    assert healing["stages"] == []
    assert healing["incident"] is None
    assert "No intervention" in healing["final_text"]


def test_lifecycle_stages_carry_a_semantic_tone_key():
    with _app_client() as client:
        result = client.post(
            "/api/simulations", json={"scenario": "recovery-rollback"}
        ).json()["result"]
    stages = result["healing"]["stages"]
    assert len(stages) == 8
    assert all(stage["tone_key"] in {"accent", "muted", "warn", "ok", "bad"} for stage in stages)
    assert stages[0]["tone_key"] == "accent"
    assert stages[1]["tone_key"] == "muted"
    assert stages[2]["tone_key"] == "warn"
    # An inconclusive rollback is not a success: both judgment stages say so.
    assert result["healing"]["incident"]["verdict"] == "inconclusive"
    assert stages[6]["tone_key"] == "warn"
    assert stages[7]["tone_key"] == "warn"
    assert stages[7]["title"] == "Final decision"


def test_accepted_recovery_keeps_both_judgment_stages_green():
    with _app_client() as client:
        result = client.post(
            "/api/simulations", json={"scenario": "recovery-accepted"}
        ).json()["result"]
    stages = result["healing"]["stages"]
    assert result["healing"]["incident"]["verdict"] == "accepted"
    assert stages[6]["tone_key"] == "ok"
    assert stages[7]["tone_key"] == "ok"


def test_verdict_tone_never_turns_failure_into_success():
    assert verdict_tone_key("accepted") == "ok"
    assert verdict_tone_key("rejected") == "bad"
    assert verdict_tone_key("inconclusive") == "warn"
    assert verdict_tone_key("held") == "muted"


def test_teaching_payload_reports_the_same_metric_fields_as_other_scenarios():
    with _app_client() as client:
        result = client.post("/api/simulations", json={"scenario": "teaching"}).json()["result"]
    policies = {policy["name"]: policy for policy in result["teaching"]["policies"]}
    fcfs = policies["FCFS"]["metrics"]
    assert fcfs["total_waiting_time"] == 13
    assert fcfs["total_turnaround_time"] == 22
    assert fcfs["total_response_time"] == 13
    for policy in policies.values():
        metrics = policy["metrics"]
        assert metrics["total_waiting_time"] == sum(
            row["waiting_time"] for row in policy["outcomes"]
        )
        assert metrics["context_switches_per_process"] == pytest.approx(
            metrics["context_switches"] / metrics["num_processes"], abs=1e-6
        )


def test_teaching_fixtures_use_the_executed_fcfs_run_as_the_baseline():
    """The shared Fixture type requires an FCFS baseline; zeros would be fabricated data."""
    from dashboard.service import _teaching_fixtures
    from dashboard.teaching import build_teaching_panel

    panel = build_teaching_panel()
    fixtures = _teaching_fixtures(panel)
    executed = next(policy for policy in panel.policies if policy.name == "FCFS")
    for fixture in fixtures.values():
        assert fixture.fcfs_metrics.avg_waiting_time == executed.mean_waiting_time
        assert fixture.fcfs_metrics.total_waiting_time == sum(
            row.waiting_time for row in executed.outcomes
        )
        assert fixture.fcfs_metrics.context_switches == executed.context_switches
        assert fixture.fcfs_metrics.num_processes == len(executed.outcomes)


def test_a_healing_run_refuses_a_policy_the_loop_would_ignore():
    with _app_client() as client:
        response = client.post(
            "/api/simulations",
            json={
                "scenario": "custom",
                "options": {
                    "processes": [
                        {"pid": 1, "arrival_time": 0, "burst_time": 6, "priority": 2},
                        {"pid": 2, "arrival_time": 1, "burst_time": 2, "priority": 1},
                    ],
                    "policy": "SJF",
                    "healing": True,
                },
            },
        )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["field"] == "policy"
    assert "FCFS" in error["message"]


def test_unknown_run_keeps_its_error_type():
    with _app_client() as client:
        body = client.get("/api/simulations/" + "0" * 8).json()
    assert body["error"]["type"] == "unknown_run"
    assert "history" in body["error"]["message"]


def test_a_malformed_body_reports_the_body_not_a_character_offset():
    with _app_client() as client:
        response = client.post(
            "/api/simulations",
            content='{"scenario": ',
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["field"] == "body"
    assert error["message"].startswith("body:")


def test_incident_payload_carries_the_state_space_size():
    with _app_client() as client:
        result = client.post(
            "/api/simulations", json={"scenario": "recovery-rollback"}
        ).json()["result"]
    incident = result["healing"]["incident"]
    assert STATE_SPACE_SIZE == 648
    assert incident["state_space_size"] == STATE_SPACE_SIZE
    assert 0 <= incident["context_state_index"] < STATE_SPACE_SIZE


def test_an_unknown_scenario_reports_the_declared_text_not_a_keyerror_repr():
    with _app_client() as client:
        response = client.post("/api/simulations", json={"scenario": "nope"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["field"] == "scenario"
    # str(KeyError(...)) would wrap the whole message in double quotes.
    assert not error["message"].startswith('"')
    assert error["message"].startswith("unknown scenario 'nope'")
    assert "teaching" in error["message"]


def test_the_research_verifier_panel_quotes_the_committed_study():
    """The Research page's verifier figures must come from the frozen summary, not prose."""
    summary = json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "healing" / "healing_summary.json")
        .read_text(encoding="utf-8")
    )
    discrimination = summary["verification_discrimination"]
    derivation = summary["threshold_derivation"]
    with _app_client() as client:
        limits = client.get("/api/research").json()["measured_limits"]
    assert limits["interventions"] == discrimination["n_interventions"] == 973
    assert limits["accepted"] == discrimination["accepted_n"] == 65
    assert limits["accepted_helped_share"] == pytest.approx(discrimination["accepted_helped_share"])
    assert limits["certifiable_share"] == pytest.approx(
        discrimination["certifiable_share_of_helpful_at_accept_residual_factor"]
    )
    assert limits["rejected"] == discrimination["rejected_n"] == 34
    assert limits["rejected_harmful_share"] == pytest.approx(
        discrimination["rejected_harmful_share"]
    )
    assert limits["control_incidents_per_trace"] == pytest.approx(
        derivation["control_incidents_per_trace"]
    )
    assert limits["declared_false_accept_rate"] == pytest.approx(
        derivation["declared_false_accept_rate_on_control"]
    )
