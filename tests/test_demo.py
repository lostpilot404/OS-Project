"""Presentation command must exercise the real event engine, not prerecorded text."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from main import main

ROOT = Path(__file__).resolve().parents[1]


def test_demo_real_outputs_and_deterministic_replay(tmp_path, capsys):
    target = tmp_path / "new directory" / "demo"
    assert main(["demo", "--results-dir", str(target)]) == 0
    first = (target / "demo.json").read_bytes()
    data = json.loads(first)
    text = capsys.readouterr().out
    assert "no Q training" in text
    assert "Verdict=accepted" in text
    assert "Verdict=inconclusive" in text
    assert "rollback to FCFS; confirmed_failure=False" in text
    assert "state=205/648" in text
    assert str(target / "demo.json") in text
    for fixture, verdict in zip(data["fixtures"], ("inconclusive", "accepted")):
        assert len(fixture["processes"]) == 15
        assert len(fixture["outcomes"]) == 15
        assert fixture["incidents"][0]["verdict"] == verdict
        assert fixture["incidents"][0]["context_state_index"] == 205
        total = sum(o["completion_time"] - o["arrival_time"] - o["burst_time"] for o in fixture["outcomes"])
        assert total == fixture["metrics"]["total_waiting_time"]
        assert total == sum(d["waiting_increment"] for d in fixture["decisions"])
        assert fixture["metrics"]["avg_waiting_time"] == total / 15
        for decision in fixture["decisions"]:
            observation = decision["observation"]
            assert all(p["arrival_time"] <= observation["current_time"] for p in observation["ready_processes"])
    first_incident = data["fixtures"][0]["incidents"][0]
    tail = [d for d in data["fixtures"][0]["decisions"] if d["observation"]["current_time"] >= first_incident["window_end_time"]]
    assert tail and tail[0]["policy"] == "FCFS"
    assert main(["demo", "--results-dir", str(target)]) == 0
    assert first == (target / "demo.json").read_bytes()


def test_demo_cli_from_other_directory(tmp_path):
    command = [sys.executable, str(ROOT / "main.py"), "demo", "--results-dir", str(tmp_path / "out")]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Saved real inputs" in result.stdout
    assert (tmp_path / "out" / "demo.json").exists()


def test_demo_default_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    assert (tmp_path / "results/demo/demo.json").exists()


def test_demo_output_failure_is_actionable(tmp_path):
    path = tmp_path / "file"
    path.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(SystemExit, match="Use --results-dir with a writable directory"):
        main(["demo", "--results-dir", str(path)])
    assert path.read_text() == "do not overwrite"


def test_cli_help_and_invalid_command():
    for arguments, expected in [(["--help"], 0), (["nonexistent"], 2)]:
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == expected
