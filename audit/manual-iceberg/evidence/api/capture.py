"""Manual evidence capture against the live dashboard API (browser unavailable).

Drives every real endpoint with the standard library only: /api/health, /api/examples,
POST /api/simulations for each scenario, GET /api/runs, GET /api/simulations/{run_id},
GET /api/simulations/{run_id}/export, POST /api/comparisons, GET /api/research.

Run: .venv/bin/python audit/manual-iceberg/evidence/api/capture.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parent


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def save(name, status, raw):
    (OUT / f"{name}.status").write_text(f"{status}\n")
    (OUT / f"{name}.json").write_bytes(raw)
    try:
        pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        (OUT / f"{name}.pretty.json").write_text(pretty, encoding="utf-8")
    except Exception:
        (OUT / f"{name}.pretty.json").write_text(raw.decode("utf-8", "replace"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    status, raw = get("/api/health")
    save("health", status, raw)

    status, raw = get("/api/examples")
    save("examples", status, raw)

    for scenario in ("teaching", "recovery-accepted", "recovery-rollback", "custom"):
        status, raw = post("/api/simulations", {"scenario": scenario, "options": {}})
        save(f"sim_{scenario}", status, raw)

    # Custom workload with self-healing enabled.
    status, raw = post(
        "/api/simulations",
        {
            "scenario": "custom",
            "options": {
                "processes": [
                    {"pid": 1, "arrival_time": 0, "burst_time": 8, "priority": 2},
                    {"pid": 2, "arrival_time": 1, "burst_time": 4, "priority": 1},
                    {"pid": 3, "arrival_time": 2, "burst_time": 9, "priority": 3},
                ],
                "policy": "FCFS",
                "quantum": 4,
                "switching_cost": 0,
                "healing": True,
            },
        },
    )
    save("sim_custom_healing", status, raw)

    # Extract a run_id from the custom run to exercise detail/export/runs endpoints.
    custom_path = OUT / "sim_custom.json"
    run_id = None
    if custom_path.exists():
        try:
            payload = json.loads(custom_path.read_text())
            run_id = payload.get("run_id")
        except Exception:
            pass

    if run_id:
        status, raw = get(f"/api/simulations/{run_id}")
        save(f"run_detail_{run_id[:8]}", status, raw)
        status, raw = get(f"/api/simulations/{run_id}/export")
        save(f"run_export_{run_id[:8]}", status, raw)
        # Stale/nonexistent run (error path).
        status, raw = get("/api/simulations/does-not-exist/export")
        save("run_export_missing", status, raw)

    status, raw = get("/api/runs")
    save("runs", status, raw)

    # Comparison against the stored custom run.
    if run_id:
        status, raw = post(
            "/api/comparisons",
            {"run_id": run_id, "quantum": 4, "switching_cost": 0},
        )
        save(f"compare_from_run_{run_id[:8]}", status, raw)

    # Explicit process-list comparison (default custom workload).
    status, raw = post(
        "/api/comparisons",
        {
            "processes": [
                {"pid": 1, "arrival_time": 0, "burst_time": 8, "priority": 2},
                {"pid": 2, "arrival_time": 1, "burst_time": 4, "priority": 1},
                {"pid": 3, "arrival_time": 2, "burst_time": 9, "priority": 3},
            ],
            "quantum": 4,
            "switching_cost": 0,
        },
    )
    save("compare_explicit", status, raw)

    # Invalid comparison (no workload) — error path.
    status, raw = post("/api/comparisons", {})
    save("compare_invalid", status, raw)

    # Invalid scenario — error path.
    status, raw = post("/api/simulations", {"scenario": "nope", "options": {}})
    save("sim_invalid_scenario", status, raw)

    status, raw = get("/api/research")
    save("research", status, raw)


if __name__ == "__main__":
    main()
