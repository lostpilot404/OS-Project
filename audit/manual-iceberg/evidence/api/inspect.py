"""Inspect captured scenario payloads."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent  # evidence/api dir


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def brief(name):
    d = load(name)
    r = d["result"]
    h = r.get("healing") or {}
    inc = h.get("incident")
    print("=" * 70)
    print(name.upper(), "| kind:", r["kind"], "| executed_at:", r.get("executed_at"), "| duration_ms:", r.get("duration_ms"))
    print("SETTINGS:", r["settings"])
    print("WORKLOAD:", r["workload"]["family"], "seed", r["workload"]["seed"], "fingerprint", r["workload"]["fingerprint"], "procs:", r["workload"]["process_count"])
    print("METRICS:", {k: r["metrics"][k] for k in ("policy_name", "avg_waiting_time", "avg_turnaround_time", "avg_response_time", "cpu_utilization", "context_switches", "total_elapsed_time")})
    print("HEALING triggered:", h.get("triggered"), "| summary:", h.get("summary"))
    print("FINAL TEXT:", h.get("final_text"))
    if inc:
        print("INCIDENT: time", inc["simulated_time"], "| diagnosis:", inc["diagnosis"], "| matched:", inc["diagnosis_matched"])
        print("  action:", inc["action_name"], "changed:", inc["action_changed"], "verdict:", inc["verdict"], "reason:", inc["verifier_reason"])
        print("  settings before:", inc["settings_before_policy"], inc["settings_before_quantum"], "-> after:", inc["settings_after_policy"], inc["settings_after_quantum"])
        print("  rollback target:", inc["rollback_target_policy"], inc["rollback_target_quantum"], "| fallback:", inc["fallback_triggered"], "| confirmed_failure:", inc["confirmed_failure"])
        print("  closure_residual:", inc["closure_residual"], "| accept>=", inc["accept_residual_factor"], "| reject<=", inc["reject_residual_factor"])
        print("  reward:", inc["reward"], "| window_end:", inc["window_end_time"], "| truncated:", inc["window_truncated"])
        print("  verification decisions/completions:", inc["verification_decisions"], inc["verification_completions"], "| closure_fraction:", inc["closure_fraction"])
    print("SCHEDULE: initial", r["schedule"]["initial_policy"], "-> final", r["schedule"]["final_policy"], "| intervals:", len(r["schedule"]["intervals"]), "| idle:", r["schedule"]["idle_units"], "| switch:", r["schedule"]["switch_units"])
    print("BASELINE:", (r["baseline"] or {}).get("avg_waiting_time"))
    print("OUTCOMES:", [(o["pid"], o["arrival_time"], o["burst_time"], o["priority"], o["start_time"], o["completion_time"], o["waiting_time"]) for o in r["outcomes"]])
    print("STAGES:", [(s["title"], s.get("tone_key")) for s in h.get("stages", [])])


for name in ("sim_teaching", "sim_recovery-accepted", "sim_recovery-rollback", "sim_custom", "sim_custom_healing"):
    brief(name)
