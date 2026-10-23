"""Tests of the live dashboard API and its binding to the simulation engine.

These tests exercise the HTTP layer with FastAPI's test client and compare every number it
returns with a direct call into the project's scheduler/metric code, so a regression in the
plumbing cannot pass unnoticed. They also cover validation, error shapes, run isolation,
export fidelity and the guarantees that nothing is written inside the repository.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dashboard.service as service
from config import SchedulerConfig
from dashboard.api import HISTORY_LIMIT, RunRegistry, create_app
from dashboard.scenarios import SCENARIOS
from evaluation.metrics import compute_metrics
from experiments.demo import run_fixture
from scheduler import FCFS, SJF, Priority, RoundRobin
from workload.models import Process, Workload

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSES = [
    {"pid": 1, "arrival_time": 0, "burst_time": 8, "priority": 2},
    {"pid": 2, "arrival_time": 1, "burst_time": 4, "priority": 1},
    {"pid": 3, "arrival_time": 2, "burst_time": 9, "priority": 3},
]


@pytest.fixture()
def registry():
    store = RunRegistry(limit=6)
    yield store
    store.close()


@pytest.fixture()
def client(registry):
    app = create_app(registry)
    with TestClient(app) as test_client:
        yield test_client


def _post(client, scenario, options=None):
    return client.post("/api/simulations", json={"scenario": scenario, "options": options or {}})


# --------------------------------------------------------------------------------------
# Health and catalogue
# --------------------------------------------------------------------------------------


def test_health_reports_a_working_engine(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    engine = body["engine"]
    assert engine["available"] is True
    assert "scheduled a one-process probe" in engine["detail"]
    assert set(engine["policies"]) == {"FCFS", "SJF", "Round Robin", "Priority"}
    assert body["simulation"]["max_concurrent_runs"] >= 1
    assert body["simulation"]["stored_runs"] == 0
    assert response.headers["cache-control"] == "no-store"


def test_health_reports_an_engine_failure_as_degraded(monkeypatch):
    from dashboard import api as api_module

    def broken():
        return {"available": False, "detail": "boom", "probe_ms": 0.0}

    monkeypatch.setattr(api_module, "_engine_probe", broken)
    app = create_app(RunRegistry(limit=2))
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["engine"]["available"] is False


def test_examples_lists_the_documented_scenarios(client):
    body = client.get("/api/examples").json()
    ids = [scenario["id"] for scenario in body["scenarios"]]
    assert ids == ["teaching", "recovery-accepted", "recovery-rollback", "custom"]
    assert body["count"] == 4
    for scenario in body["scenarios"]:
        assert scenario["name"] and scenario["description"] and scenario["expectation"]


# --------------------------------------------------------------------------------------
# Scenario execution
# --------------------------------------------------------------------------------------


def test_teaching_run_matches_the_hand_calculation(client):
    response = _post(client, "teaching", {"policy_view": "SJF"})
    assert response.status_code == 201
    body = response.json()
    result = body["result"]
    assert body["run_id"]
    assert result["kind"] == "teaching"
    assert result["metrics"]["avg_waiting_time"] == pytest.approx(1.666666, abs=1e-6)
    assert result["teaching"]["hand_verified"] is True
    assert result["teaching"]["selected"] == "SJF"
    # every policy panel carries its own executed outcome table
    by_name = {row["name"]: row for row in result["teaching"]["policies"]}
    assert [row["waiting_time"] for row in by_name["FCFS"]["outcomes"]] == [0, 5, 8]
    assert [row["completion_time"] for row in by_name["Round Robin"]["outcomes"]] == [9, 8, 5]
    assert result["charts"]["gantt_variants"]["Priority"].startswith("<svg")
    assert "http://" not in result["charts"]["gantt"]


def test_successful_recovery_scenario_is_accepted_and_keeps_sjf(client):
    result = _post(client, "recovery-accepted").json()["result"]
    incident = result["healing"]["incident"]
    assert result["healing"]["triggered"] is True
    assert incident["verdict"] == "accepted"
    assert incident["settings_before_policy"] == "FCFS"
    assert incident["settings_after_policy"] == "SJF"
    assert not incident["rollback_target_policy"]   # empty string: nothing to roll back to
    assert result["schedule"]["final_policy"] == "SJF"
    assert "stayed in force" in result["healing"]["summary"]
    assert result["healing"]["stages"][-1]["title"] == "Final decision"


def test_rollback_scenario_is_inconclusive_not_rejected(client):
    result = _post(client, "recovery-rollback").json()["result"]
    incident = result["healing"]["incident"]
    assert incident["verdict"] == "inconclusive"
    assert incident["confirmed_failure"] is False
    assert incident["rollback_target_policy"] == "FCFS"
    assert result["schedule"]["final_policy"] == "FCFS"
    summary = result["healing"]["summary"]
    assert "inconclusive" in summary and "neither proven nor rejected" in summary
    assert incident["closure_residual"] == pytest.approx(0.20592237689676574)


def test_documented_fixtures_reproduce_the_verified_engine_run(client):
    """The API must return exactly what the CLI fixture produces, not a re-derived copy."""
    for scenario_id, seed in (("recovery-accepted", 13), ("recovery-rollback", 3)):
        expected = run_fixture(seed).payload
        result = _post(client, scenario_id).json()["result"]
        # the API rounds to 6 decimals for the wire format; compare on that grid
        assert result["metrics"]["avg_waiting_time"] == pytest.approx(
            expected["metrics"]["avg_waiting_time"], abs=1e-6
        )
        assert result["baseline"]["avg_waiting_time"] == pytest.approx(
            expected["fcfs_metrics"]["avg_waiting_time"], abs=1e-6
        )
        assert result["healing"]["incident"]["verdict"] == expected["incidents"][0]["verdict"]
        assert result["workload"]["fingerprint"] == expected["fingerprint"]


def test_custom_workload_matches_a_direct_engine_call(client):
    result = _post(
        client,
        "custom",
        {"processes": PROCESSES, "policy": "SJF", "quantum": 3, "switching_cost": 1},
    ).json()["result"]
    workload = Workload("custom", [Process(**row) for row in PROCESSES])
    expected = compute_metrics(SJF(SchedulerConfig(round_robin_quantum=3, switching_cost=1)).run(workload))
    assert result["metrics"]["avg_waiting_time"] == pytest.approx(expected.avg_waiting_time)
    assert result["metrics"]["avg_turnaround_time"] == pytest.approx(expected.avg_turnaround_time)
    assert result["metrics"]["context_switches"] == expected.context_switches
    assert result["metrics"]["total_elapsed_time"] == expected.total_elapsed_time
    assert result["workload"]["fingerprint"] == workload.fingerprint
    assert result["baseline"] is None
    # recorded intervals tile the makespan exactly and never overlap
    spans = [(row["start"], row["end"]) for row in result["schedule"]["intervals"]]
    assert spans == sorted(spans)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end
    assert sum(row["duration"] for row in result["schedule"]["intervals"]) <= expected.total_elapsed_time


def test_custom_workload_can_run_the_self_healing_loop(client):
    result = _post(
        client, "custom", {"processes": PROCESSES, "healing": True, "policy": "FCFS"}
    ).json()["result"]
    assert result["healing"] is not None
    assert "triggered" in result["healing"]
    assert result["schedule"]["final_policy"] in {"FCFS", "SJF", "Round Robin", "Priority"}
    # a short workload may legitimately never trigger; the payload must still be explicit
    if not result["healing"]["triggered"]:
        assert "no incident" in result["healing"]["summary"].lower()


def test_all_four_policies_execute_on_the_same_workload(client):
    for policy in ("FCFS", "SJF", "Priority", "Round Robin"):
        response = _post(client, "custom", {"processes": PROCESSES, "policy": policy})
        assert response.status_code == 201, response.text
        assert response.json()["result"]["settings"]["policy"] == policy


# --------------------------------------------------------------------------------------
# Validation and error handling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, message",
    [
        ({"scenario": "does-not-exist"}, "unknown scenario"),
        ({"scenario": "custom", "options": {"processes": []}}, "at least one process"),
        (
            {"scenario": "custom", "options": {"processes": [{**PROCESSES[0], "burst_time": 0}]}},
            "burst_time must be > 0",
        ),
        (
            {"scenario": "custom", "options": {"processes": [{**PROCESSES[0], "priority": -1}]}},
            "priority must be >= 0",
        ),
        (
            {"scenario": "custom", "options": {"processes": [PROCESSES[0], dict(PROCESSES[0])]}},
            "duplicate pid",
        ),
        ({"scenario": "custom", "options": {"quantum": 0}}, "quantum must be between"),
        ({"scenario": "custom", "options": {"quantum": 65}}, "quantum must be between"),
        ({"scenario": "custom", "options": {"switching_cost": 11}}, "switching_cost must be between"),
        ({"scenario": "custom", "options": {"healing": "yes"}}, "must be true or false"),
        ({"scenario": "custom", "options": {"policy": "LIFO"}}, "unknown policy"),
        ({"scenario": "teaching", "options": {"processes": PROCESSES}}, "does not accept option"),
        ({"scenario": "teaching", "options": {"policy_view": "LIFO"}}, "unknown policy"),
    ],
)
def test_invalid_requests_are_rejected_with_a_message(client, body, message):
    response = client.post("/api/simulations", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request"
    assert message in error["message"]


def test_non_integer_and_missing_fields_are_rejected(client):
    response = client.post(
        "/api/simulations",
        json={"scenario": "custom", "options": {"processes": [{**PROCESSES[0], "pid": "one"}]}},
    )
    assert response.status_code == 422
    assert "whole number" in response.json()["error"]["message"]

    response = client.post(
        "/api/simulations",
        json={"scenario": "custom", "options": {"processes": [{"pid": 1, "arrival_time": 0}]}},
    )
    assert response.status_code == 422
    assert "missing" in response.json()["error"]["message"]


def test_workload_size_and_burst_limits_are_enforced(client):
    too_many = [{"pid": i, "arrival_time": 0, "burst_time": 1, "priority": 1} for i in range(41)]
    response = _post(client, "custom", {"processes": too_many})
    assert response.status_code == 422
    assert "at most 40" in response.json()["error"]["message"]

    huge = _post(client, "custom", {"processes": [{**PROCESSES[0], "burst_time": 10_001}]})
    assert huge.status_code == 422
    assert "burst_time must be <=" in huge.json()["error"]["message"]


def test_malformed_json_is_reported_as_a_client_error(client):
    response = client.post(
        "/api/simulations", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request"


def test_missing_body_field_names_the_field(client):
    response = client.post("/api/simulations", json={"options": {}})
    assert response.status_code == 422
    assert "scenario" in response.json()["error"]["message"]


def test_raw_engine_failure_is_reported_without_a_traceback(client, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated engine explosion")

    monkeypatch.setattr(service, "execute", explode)
    response = _post(client, "teaching")
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "engine_failure"
    assert "simulated engine explosion" in error["message"]
    assert "Traceback" not in error["message"]


def test_dashboard_schema_rejection_is_reported(client, monkeypatch):
    from dashboard.model import DashboardError

    def reject(*_args, **_kwargs):
        raise DashboardError("fixtures[0]: metrics contradict outcomes")

    monkeypatch.setattr(service, "execute", reject)
    response = _post(client, "teaching")
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "engine_output_rejected"


# --------------------------------------------------------------------------------------
# Stored runs, export and comparisons
# --------------------------------------------------------------------------------------


def test_a_stored_run_can_be_read_back_and_exported(client):
    created = _post(client, "recovery-rollback").json()
    run_id = created["run_id"]

    fetched = client.get(f"/api/simulations/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["result"] == created["result"]

    export = client.get(f"/api/simulations/{run_id}/export")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/json")
    assert f"os-dashboard-recovery-rollback-{run_id[:8]}.json" in export.headers["content-disposition"]
    document = json.loads(export.content)
    assert document["run_id"] == run_id
    assert document["result"]["healing"]["incident"]["verdict"] == "inconclusive"


def test_recent_runs_is_empty_before_any_simulation(client):
    body = client.get("/api/runs").json()
    # The fixture registry keeps six runs; the endpoint reports that limit, not the default.
    assert body == {"runs": [], "count": 0, "history_limit": 6}


def test_recent_runs_summarises_runs_newest_first(client):
    first = _post(client, "teaching").json()
    second = _post(client, "recovery-rollback").json()

    body = client.get("/api/runs").json()
    assert body["count"] == 2
    assert [run["run_id"] for run in body["runs"]] == [second["run_id"], first["run_id"]]

    newest = body["runs"][0]
    assert set(newest) == {
        "run_id",
        "scenario_id",
        "scenario_name",
        "kind",
        "executed_at",
        "status",
        "final_policy",
        "mean_waiting_time",
        "verdict",
    }
    result = second["result"]
    assert newest["scenario_id"] == "recovery-rollback"
    assert newest["scenario_name"] == result["scenario"]["name"]
    assert newest["kind"] == "healing"
    assert newest["status"] == "completed"
    assert newest["executed_at"] == result["executed_at"]
    assert newest["final_policy"] == result["schedule"]["final_policy"]
    assert newest["mean_waiting_time"] == result["metrics"]["avg_waiting_time"]
    assert newest["verdict"] == "inconclusive"


def test_recent_runs_reports_no_verdict_for_a_plain_run(client):
    created = _post(client, "teaching").json()
    summary = client.get("/api/runs").json()["runs"][0]
    assert summary["run_id"] == created["run_id"]
    assert summary["verdict"] is None
    assert summary["kind"] == "teaching"


def test_recent_runs_agrees_with_the_stored_run(client):
    created = _post(client, "recovery-accepted").json()
    summary = client.get("/api/runs").json()["runs"][0]
    stored = client.get(f"/api/simulations/{created['run_id']}").json()["result"]
    assert summary["mean_waiting_time"] == stored["metrics"]["avg_waiting_time"]
    assert summary["final_policy"] == stored["schedule"]["final_policy"]
    assert summary["verdict"] == stored["healing"]["incident"]["verdict"]


def test_recent_runs_only_reads_history(client):
    created = _post(client, "teaching").json()
    before = client.get("/api/runs").json()
    for _ in range(3):
        assert client.get("/api/runs").json() == before
    assert client.get("/api/health").json()["simulation"]["stored_runs"] == 1
    assert client.get(f"/api/simulations/{created['run_id']}").status_code == 200


def test_recent_runs_is_bounded_by_the_registry(client):
    # The fixture registry keeps six runs, so the seventh evicts the oldest entry.
    created = [_post(client, "teaching").json()["run_id"] for _ in range(7)]
    body = client.get("/api/runs").json()
    assert body["count"] == 6
    assert body["history_limit"] == 6
    kept = [run["run_id"] for run in body["runs"]]
    assert kept == list(reversed(created[1:]))
    assert client.get(f"/api/simulations/{created[0]}").status_code == 404


def test_unknown_run_ids_produce_a_clear_404(client):
    for path in ("/api/simulations/nope", "/api/simulations/nope/export"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["type"] in {"unknown_run", "not_found"}
        assert "run" in response.json()["error"]["message"].lower() or "not found" in response.json()["error"]["message"].lower()


def test_comparison_by_run_id_matches_a_direct_engine_run(client):
    created = _post(client, "custom", {"processes": PROCESSES, "policy": "FCFS"}).json()
    result = client.post("/api/comparisons", json={"run_id": created["run_id"]}).json()["result"]

    workload = Workload("custom", [Process(**row) for row in PROCESSES])
    config = SchedulerConfig(round_robin_quantum=4, switching_cost=0)
    expected = {
        policy.name: compute_metrics(policy(config).run(workload)).avg_waiting_time
        for policy in (FCFS, SJF, RoundRobin, Priority)
    }
    got = {row["name"]: row["metrics"]["avg_waiting_time"] for row in result["policies"]}
    assert got == pytest.approx(expected)
    assert result["best_by_waiting"] == min(expected, key=expected.get)
    assert result["source_run_id"] == created["run_id"]
    assert "http://" not in result["charts"]["waiting"]


def test_comparison_can_be_requested_with_an_explicit_workload(client):
    result = client.post(
        "/api/comparisons",
        json={"processes": PROCESSES, "quantum": 2, "switching_cost": 1},
    ).json()["result"]
    assert [row["name"] for row in result["policies"]] == ["FCFS", "SJF", "Round Robin", "Priority"]
    assert result["settings"] == {"quantum": 2, "switching_cost": 1}


def test_comparison_requires_a_workload(client):
    response = client.post("/api/comparisons", json={})
    assert response.status_code == 422
    assert response.json()["error"]["field"] == "processes"


def test_comparison_rejects_an_invalid_workload(client):
    response = client.post(
        "/api/comparisons", json={"processes": [{"pid": 1, "arrival_time": 0, "burst_time": 0, "priority": 1}]}
    )
    assert response.status_code == 422
    assert "burst_time" in response.json()["error"]["message"]


# --------------------------------------------------------------------------------------
# Isolation, concurrency and repository safety
# --------------------------------------------------------------------------------------


def test_each_run_gets_its_own_directory_outside_the_repository(client, registry):
    first = _post(client, "teaching").json()["run_id"]
    second = _post(client, "teaching").json()["run_id"]
    run_a, run_b = registry.get(first), registry.get(second)
    assert run_a is not None and run_b is not None
    assert run_a.directory != run_b.directory
    assert run_a.directory.exists() and run_b.directory.exists()
    for run in (run_a, run_b):
        assert REPO_ROOT not in run.directory.parents
        assert (run.directory / "result.json").read_bytes() == run.json_bytes


def test_concurrent_runs_do_not_share_state(client, registry):
    outcomes = {}
    errors = []

    def worker(scenario, options):
        try:
            response = _post(client, scenario, options)
            errors.append(response.status_code) if response.status_code != 201 else None
            outcomes[scenario + str(options.get("policy", ""))] = response.json()
        except Exception as error:  # pragma: no cover - reported through the assertion
            errors.append(repr(error))

    threads = [
        threading.Thread(target=worker, args=("recovery-rollback", {})),
        threading.Thread(target=worker, args=("recovery-accepted", {})),
        threading.Thread(target=worker, args=("custom", {"processes": PROCESSES, "policy": "SJF"})),
        threading.Thread(target=worker, args=("teaching", {})),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors
    ids = {payload["run_id"] for payload in outcomes.values()}
    assert len(ids) == 4
    verdicts = {
        key: payload["result"]["healing"]["incident"]["verdict"]
        for key, payload in outcomes.items()
        if payload["result"]["healing"] and payload["result"]["healing"]["triggered"]
    }
    assert verdicts["recovery-rollback"] == "inconclusive"
    assert verdicts["recovery-accepted"] == "accepted"


def test_history_is_bounded_and_evicted_directories_are_removed():
    registry = RunRegistry(limit=2)
    try:
        first = registry.add("teaching", {"x": 1})
        directory = first.directory
        registry.add("teaching", {"x": 2})
        registry.add("teaching", {"x": 3})
        assert registry.get(first.run_id) is None
        assert not directory.exists()
        assert len(registry.ids()) == 2
    finally:
        registry.close()


def test_shutdown_removes_run_directories(registry):
    app = create_app(registry)
    with TestClient(app) as client:
        run_id = _post(client, "teaching").json()["run_id"]
        directory = registry.get(run_id).directory
        assert directory.exists()
    assert not directory.exists()


def test_simulations_never_write_inside_the_repository(client):
    before = sorted(str(path) for path in REPO_ROOT.rglob("*") if ".git" not in path.parts and ".venv" not in path.parts)
    for scenario in ("teaching", "recovery-accepted", "recovery-rollback"):
        assert _post(client, scenario).status_code == 201
    assert client.post(
        "/api/comparisons", json={"processes": PROCESSES}
    ).status_code == 201
    after = sorted(str(path) for path in REPO_ROOT.rglob("*") if ".git" not in path.parts and ".venv" not in path.parts)
    new_paths = [path for path in after if path not in before]
    # pytest's own caches are the only tolerated additions
    assert all("__pycache__" in path or ".pytest_cache" in path for path in new_paths), new_paths


def test_run_guard_refuses_more_than_capacity():
    from fastapi import HTTPException

    from dashboard.api import _RunGuard

    guard = _RunGuard(capacity=1)
    with guard:
        with pytest.raises(HTTPException) as error:
            with guard:
                pass
    assert error.value.status_code == 429
    # the semaphore is released again
    with guard:
        pass


def test_request_paths_cannot_reach_the_filesystem():
    from dashboard.api import SimulationRequest

    assert set(SimulationRequest.model_fields) == {"scenario", "options"}
    assert not hasattr(service, "write")


# --------------------------------------------------------------------------------------
# Research evidence and static assets
# --------------------------------------------------------------------------------------


def test_research_endpoint_returns_the_committed_values(client):
    body = client.get("/api/research").json()
    summary = json.loads((REPO_ROOT / "results" / "healing" / "healing_summary.json").read_text())
    arms = {row["arm"]: row for row in body["arms"]}
    assert len(arms) == len(summary["means"])
    for raw in summary["means"]:
        assert arms[raw["arm"]]["mean_waiting_time"] == pytest.approx(
            raw["avg_waiting_time"], abs=1e-6
        )
    assert body["split"] == summary["split"]
    assert body["final_test_seed"] == summary["final_test_seed"]
    assert "not the output" in body["caveats"][0]
    assert "did not establish Q-learning" in body["caveats"][1]
    assert body["charts"]["waiting"].startswith("<svg")


def test_research_endpoint_reports_missing_artifacts(client, monkeypatch, tmp_path):
    monkeypatch.setattr(service, "RESEARCH_SUMMARY_PATH", tmp_path / "absent.json")
    response = client.get("/api/research")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "research_unavailable"


def test_frontend_is_served_and_is_self_contained(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "Adaptive CPU Scheduling" in index.text
    assert "/static/app.js" in index.text
    assert "/static/styles.css" in index.text
    for forbidden in ("http://", "https://", "cdn.", "@import url", "integrity="):
        assert forbidden not in index.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_shell_is_never_cached_and_stamps_its_assets(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "no-store" in index.headers["cache-control"]
    # The shell points at content-stamped asset URLs, so a reload never pairs an old page
    # with a new (or missing) script.
    assert re.search(r"/static/app\.js\?v=[0-9a-f]{10}", index.text)
    assert re.search(r"/static/styles\.css\?v=[0-9a-f]{10}", index.text)
    assert client.get("/").text == index.text


def test_static_assets_are_revalidated(client):
    for path in ("/static/app.js", "/static/styles.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"


def test_shell_stamp_follows_the_asset_bytes(client, monkeypatch, tmp_path):
    shell = (REPO_ROOT / "dashboard" / "web" / "index.html").read_text()
    (tmp_path / "index.html").write_text(shell)
    (tmp_path / "app.js").write_text("// first\n")
    (tmp_path / "styles.css").write_text("/* first */\n")
    monkeypatch.setattr("dashboard.api.WEB_DIR", tmp_path)
    with TestClient(create_app()) as fresh:
        first = fresh.get("/").text
        assert "?v=" + hashlib.sha1(b"// first\n").hexdigest()[:10] in first
        (tmp_path / "app.js").write_text("// second\n")
        second = fresh.get("/").text
        assert "?v=" + hashlib.sha1(b"// first\n").hexdigest()[:10] not in second
        assert "?v=" + hashlib.sha1(b"// second\n").hexdigest()[:10] in second


def test_frontend_says_so_when_the_script_never_runs(client):
    # A page that loaded without its script must not sit on "Connecting…" in silence.
    body = client.get("/").text
    assert "The console could not start" in body
    assert "/static/app.js" in body


def test_frontend_script_uses_only_relative_api_calls():
    script = (Path(__file__).resolve().parent.parent / "dashboard" / "web" / "app.js").read_text()
    assert "http://" not in script and "https://" not in script
    for route in ("/api/health", "/api/examples", "/api/simulations", "/api/comparisons", "/api/research"):
        assert route in script
    # no scheduling logic in the browser: no engine formulas may be reimplemented
    assert "completion - arrival - burst" not in script

def test_scenario_catalogue_is_deep_copied():
    import copy

    first = copy.deepcopy(SCENARIOS[3]["defaults"]["processes"])
    first[0]["burst_time"] = 999
    assert SCENARIOS[3]["defaults"]["processes"][0]["burst_time"] != 999

