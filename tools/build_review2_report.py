"""Build the Review 2 report PDF from the committed artifacts.

    python tools/build_review2_report.py                    # write the canonical PDF
    python tools/build_review2_report.py --output /tmp/x.pdf --skip-tests

The narrative lives in :mod:`review2_content` and the page furniture in
:mod:`review2_layout`; this module only renders the content blocks to ReportLab flowables,
so the PDF and the DOCX built by :mod:`build_review2_docx` cannot disagree.  Nothing here
invents a number: every value comes from the artifacts through :mod:`review2_data`, and the
figures are rendered from those same artifacts.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import KeepTogether, NextPageTemplate, PageBreak, Spacer  # noqa: E402
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

import review2_content as content  # noqa: E402
import review2_figures as figures  # noqa: E402
from review2_data import Artifacts  # noqa: E402
from review2_layout import (  # noqa: E402
    CONTENT_W,
    REPORT_TITLE,
    REVIEW_LINE,
    ReviewDoc,
    S,
    bullets,
    caption,
    code_block,
    figure,
    h1,
    h2,
    kpi_strip,
    make_table,
    para,
)

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "Review_2_Report.pdf"
DEFAULT_FIGURES = REPO_ROOT / "figures" / "review2"


# --------------------------------------------------------------------------------------
# evidence quoted in the report
# --------------------------------------------------------------------------------------
def collect_test_counts(root: Path, timeout: int = 1800) -> Tuple[str, Dict[str, int]]:
    """Run the suite and report what it observed, per module.

    Returns the sentence that goes on the cover page and the per-module counts, so the
    report quotes the number it actually observed rather than a remembered one.
    """
    try:
        run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root,
                             capture_output=True, text=True, timeout=timeout)
        summary = ""
        for line in reversed(run.stdout.splitlines()):
            if "passed" in line or "failed" in line or "error" in line:
                summary = line.strip()
                break
        collection = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                                    cwd=root, capture_output=True, text=True, timeout=timeout)
        counts: Dict[str, int] = {}
        for line in collection.stdout.splitlines():
            if "::" in line:
                module = line.split("::")[0].split("/")[-1]
                counts[module] = counts.get(module, 0) + 1
        if summary:
            return f"{summary} (python -m pytest -q, {len(counts)} test modules)", counts
    except (OSError, subprocess.SubprocessError):
        pass
    return "test suite present; execution unavailable in this environment", {}


def git_revision(root: Path) -> str:
    """Short revision of the working tree; reported by the CLI, not stamped into the PDF.

    A document cannot name the commit that contains it, so the printed page cites the
    artifacts and the reproduction commands instead.
    """
    try:
        revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return f"{revision}{' (working tree)' if dirty else ''}" if revision else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------------------
# flowables
# --------------------------------------------------------------------------------------
def render_block(block: Tuple, story: List, fig_paths: Dict[str, Path]) -> None:
    """Turn one typed content block into flowables."""
    kind, payload = block[0], block[1]
    if kind == "h1":
        story.append(h1(payload))
    elif kind == "h2":
        story.append(h2(payload))
    elif kind == "p":
        story.append(para(payload))
    elif kind == "bullets":
        story.extend(bullets(payload))
    elif kind == "numbers":
        story.extend(bullets([f"<b>{index + 1}.</b> {item}"
                              for index, item in enumerate(payload)]))
    elif kind == "code":
        story.append(code_block(payload["text"]))
        if payload.get("caption"):
            story.append(caption(payload["caption"]))
    elif kind == "table":
        # content declares column widths as fractions of the text column
        table = make_table(payload["rows"], [CONTENT_W * w for w in payload["widths"]],
                           align_right=tuple(payload["align_right"]),
                           header=payload["header"])
        group: List = [table]
        if payload.get("caption"):
            group.insert(0, Spacer(1, 0.18 * cm))
            group.append(caption(payload["caption"]))
        story.append(KeepTogether(group) if len(payload["rows"]) <= 8 else group[0])
        if len(payload["rows"]) > 8:
            story.extend(group[1:])
    elif kind == "kpi":
        story.append(Spacer(1, 0.12 * cm))
        story.append(kpi_strip(payload))
        story.append(Spacer(1, 0.12 * cm))
    elif kind == "fig":
        path = fig_paths.get(payload["key"])
        if path and path.exists():
            story.extend(figure(path, payload["caption"], width=CONTENT_W * payload["width"],
                                max_height=payload.get("max_height", 20.6) * cm))
        else:  # pragma: no cover - only when a figure could not be rendered
            story.append(para(f"[figure {payload['key']} unavailable]", "Small"))
    else:  # pragma: no cover - guards against a typo in the content module
        raise ValueError(f"unknown block kind: {kind!r}")


# --------------------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------------------
def cover_page(data: Artifacts, generated: str, test_line: str) -> List:
    fields = content.cover_fields(data, test_line, generated)
    story: List = [Spacer(1, 0.35 * cm)]
    story.append(para("OPERATING SYSTEMS PROJECT", "cover_meta"))
    story.append(para(REVIEW_LINE.upper(), "cover_meta"))
    story.append(Spacer(1, 0.25 * cm))
    story.append(para(REPORT_TITLE, "title"))
    story.append(para("Self-healing scheduling in a discrete-event CPU simulator", "subtitle"))
    story.append(
        para(
            "Causal arrived-work observation  •  monitor, detect, diagnose, act, verify  •  "
            "rule-based and tabular Q-learning recovery  •  accepted, rolled back or "
            "safe-fallback settlement  •  seven seeded workload families  •  held-out "
            "crossed-bootstrap evaluation with a counterfactual audit",
            "Small",
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(make_table([["Field", "Value"]] + fields["meta"],
                            [CONTENT_W * 0.24, CONTENT_W * 0.76]))
    story.append(Spacer(1, 0.28 * cm))
    story.append(kpi_strip(fields["kpis"]))
    story.append(Spacer(1, 0.28 * cm))
    story.append(para(fields["abstract"], "Abstract"))
    story.append(Spacer(1, 0.18 * cm))
    story.append(para("Every number in this report is read at build time from the "
                      "committed artifacts under results/healing (and, for the preserved "
                      "secondary workflows, results/study and results/runtime); the figures "
                      "are rendered from the same files. Reproduce this document with the "
                      "two commands in Appendix A.1.", "Small"))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    return story


def toc_page() -> List:
    story: List = [h1("Contents")]
    story.append(para("The report follows the Review 2 rubric item by item: the nine "
                      "numbered headings below are the nine required items, followed by an "
                      "appendix of reproduction commands, seeds and preserved workflows.",
                      "Small"))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC0", parent=S["H1"], fontSize=9.4, leading=15, spaceAfter=1.5),
        ParagraphStyle("TOC1", parent=S["H2"], fontSize=8.4, leading=12.5, leftIndent=14,
                       spaceAfter=0.5),
    ]
    story.append(toc)
    story.append(PageBreak())
    return story


def build(root: Path = REPO_ROOT, output: Path = DEFAULT_OUTPUT,
          figures_dir: Path = DEFAULT_FIGURES, with_tests: bool = True,
          with_dataset: bool = True, docs_copy: Path | None = None) -> Path:
    """Render the report to ``output`` and return the path written."""
    generated = _datetime.date.today().isoformat()
    test_line, _counts = (collect_test_counts(root) if with_tests
                          else ("test suite not re-run for this build", {}))
    data = Artifacts(root, with_dataset=with_dataset)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = figures.render_all(data, content.MERMAID, figures_dir)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = ReviewDoc(str(output), generated=generated)
    story: List = []
    story += cover_page(data, generated, test_line)
    story += toc_page()
    for block in content.sections(data):
        render_block(block, story, fig_paths)
    document.multiBuild(story)
    if docs_copy:
        Path(docs_copy).parent.mkdir(parents=True, exist_ok=True)
        Path(docs_copy).write_bytes(output.read_bytes())
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Review 2 report PDF.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--docs-copy", type=Path, default=None,
                        help="also write the PDF to this path (e.g. a docs/ copy)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="do not re-run test collection for the cover page")
    parser.add_argument("--no-dataset", action="store_true",
                        help="skip the generator call that fills the dataset table")
    args = parser.parse_args(argv)
    path = build(root=args.root, output=args.output, figures_dir=args.figures_dir,
                 with_tests=not args.skip_tests, with_dataset=not args.no_dataset,
                 docs_copy=args.docs_copy)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB) "
          f"[source revision {git_revision(args.root)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
