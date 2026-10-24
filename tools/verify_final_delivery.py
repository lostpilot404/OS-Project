"""Check final report constraints and demo/evidence links without running the study.

Run from any directory with the project's core + requirements-report dependencies:
    python tools/verify_final_delivery.py
Visual inspection is separate; this script does not claim to replace it.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader
from pypdf.generic import ContentStream
from reportlab.pdfbase.pdfmetrics import stringWidth

from experiments.demo import run_demo
from tools.build_plain_report import _register_fonts, build


def check() -> None:
    source_path = ROOT / "docs/OS_Project_Final_Report.md"
    pdf_path = ROOT / "docs/OS_Project_Final_Report.pdf"
    source = source_path.read_text(encoding="utf-8")
    assert not re.search(r"^\s*\|", source, re.M), "No tables permitted"
    assert not re.search(r"!\[|<img|<table|style |classDef ", source)
    for number in range(1, 11):
        assert re.search(rf"^## {number}\. ", source, re.M), f"Missing section {number}"
    assert source.count("```mermaid") == 7
    assert "500 passed" in source and "WSL" in source
    _register_fonts()
    in_code = False
    for line in source.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif in_code:
            assert stringWidth(line, "ReportMono", 8) < 470, f"Code exceeds frame: {line}"
    assert not in_code

    summary = json.loads(
        (ROOT / "results/healing/healing_summary.json").read_text(encoding="utf-8")
    )
    assert summary["final_test_seed"] == 23819
    for arm in ("fixed_fcfs", "fixed_sjf", "rule_wait", "q_recovery"):
        row = next(row for row in summary["means"] if row["arm"] == arm)
        assert f"{row['avg_waiting_time']:.2f}" in source
    for filename, count, incidents in [
        ("healing_action_probe.csv", 784, 98),
        ("healing_action_probe.legacy_raw_proxy.csv", 168, 21),
    ]:
        with (ROOT / "results/healing" / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == count
        assert len({(r["workload_fingerprint"], r["decision_index"]) for r in rows}) == incidents
    for filename, count in [("healing_metrics_final_test.csv", 3780), ("healing_incidents_final_test.csv", 2171), ("healing_audit.csv", 3256)]:
        with (ROOT / "results/healing" / filename).open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == count

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            payload = run_demo(target)
        actual = captured.getvalue().replace(str(target / "demo.json"), "results/demo/demo.json")
        expected = (ROOT / "docs/examples/demo_linux_output.txt").read_text(encoding="utf-8")
        assert actual == expected, "Captured output must match a fresh real execution"
        for fixture in payload["fixtures"]:
            metric = fixture["metrics"]["avg_waiting_time"]
            assert f"{metric:.3f}" in source
        copy = target / "report.pdf"
        pages = build(source_path, copy)
        assert hashlib.sha256(copy.read_bytes()).digest() == hashlib.sha256(pdf_path.read_bytes()).digest(), "PDF must be rebuilt from authoritative Markdown"

    reader = PdfReader(str(pdf_path), strict=True)
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        assert text and f"Page {number}" in text
        resources = page["/Resources"].get_object()
        assert not resources.get("/XObject"), "No images or form diagrams permitted"
        for operands, operator in ContentStream(page.get_contents(), reader).operations:
            if operator in (b"rg", b"RG", b"g", b"G"):
                assert all(float(value) in (0.0, 1.0) for value in operands)
                assert len(set(float(value) for value in operands)) == 1, "Only black/white permitted"
            assert operator not in (b"k", b"K", b"Do", b"BI"), "Unexpected color/image operator"
    pdf_text = "\n".join(page.extract_text() for page in reader.pages)
    assert pdf_text.count("flowchart") == 7, "Print all Mermaid source, do not render it"
    print(f"Verified {pages} pages: text-only black/white, page numbers, seven Mermaid sources.")
    print("Verified canonical/legacy probe counts, frozen means, exact demo transcript and reproducible PDF bytes.")
    print("Visual inspection and WSL execution are separate from these automated checks.")


if __name__ == "__main__":
    check()
