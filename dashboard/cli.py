"""Command-line interface of the teacher-dashboard builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from dashboard.build import BuildResult, build_dashboard, default_research_path
from dashboard.model import DashboardError

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (exposed for tests and ``--help``)."""
    parser = argparse.ArgumentParser(
        prog="build_teacher_dashboard.py",
        description=(
            "Build the self-contained classroom dashboard from a saved demo.json. The page "
            "is written once, opens from the file system and needs no server or network."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="DEMO_JSON",
        help="demo.json produced by 'python main.py demo --results-dir DIR'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="INDEX_HTML",
        help="destination .html file (default: index.html next to the input demo.json)",
    )
    research = parser.add_mutually_exclusive_group()
    research.add_argument(
        "--research",
        type=Path,
        default=None,
        metavar="SUMMARY_JSON",
        help=(
            "study summary to read for the research panel "
            "(default: results/healing/healing_summary.json)"
        ),
    )
    research.add_argument(
        "--no-research",
        action="store_true",
        help="build without the research panel (for a demo-only copy)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing dashboard file (never touches research artifacts)",
    )
    parser.add_argument("--quiet", action="store_true", help="print only errors")
    return parser


def _summary(result: BuildResult, demo_name: str, research_name: Optional[str]) -> str:
    size_kb = result.bytes_written / 1024.0
    lines = [
        "Teacher dashboard written",
        "-------------------------",
        f"  demo input        : {demo_name}",
        f"  scenarios         : seeds {', '.join(str(seed) for seed in result.seeds)}",
        f"  teaching example  : {result.teaching_policies} policies, hand calculation verified",
        f"  research panel    : {research_name if result.research_included else 'not included'} ({result.research_arms} arms)",
        f"  output            : {result.output_path}",
        f"  size              : {size_kb:.1f} KB (self-contained HTML, inline SVG)",
        f"  sha256            : {result.sha256[:16]}",
        "Open the file in a browser; no server or network connection is used.",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the builder and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path = args.input.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else input_path.parent / "index.html"
    )
    include_research = not args.no_research
    research_path = args.research.expanduser() if args.research else default_research_path()
    try:
        result = build_dashboard(
            input_path=input_path,
            output_path=output_path,
            research_path=research_path,
            include_research=include_research,
            force=args.force,
        )
    except DashboardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(
            _summary(
                result,
                input_path.name,
                research_path.name if include_research else None,
            )
        )
    return 0
