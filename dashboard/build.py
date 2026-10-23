"""Build the teacher dashboard from a saved demo trace (and the committed study tables).

The builder is deliberately conservative about the file system:

* every input is validated before anything is written;
* the output must be an ``.html`` file, the input and the research evidence are never
  targets, and an existing dashboard is not replaced without ``--force``;
* the page is written to a temporary file in the destination directory and moved into
  place only after the complete document exists, so a failure cannot leave a truncated
  ``index.html`` behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dashboard.model import DashboardError, DemoDocument, parse_demo
from dashboard.render import render_dashboard
from dashboard.research import ResearchSummary, load_research
from dashboard.teaching import TeachingPanel, build_teaching_panel

__all__ = [
    "BuildResult",
    "build_dashboard",
    "default_research_path",
    "load_demo",
    "REPOSITORY_ROOT",
    "PROTECTED_DIRECTORIES",
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Directories whose contents are research evidence or generated report assets. The
#: dashboard never writes there, so a mistyped ``--output`` cannot land on top of the
#: frozen artifacts.
PROTECTED_DIRECTORIES = (
    REPOSITORY_ROOT / "results" / "healing",
    REPOSITORY_ROOT / "results" / "runtime",
    REPOSITORY_ROOT / "results" / "study",
    REPOSITORY_ROOT / "results" / "drift",
    REPOSITORY_ROOT / "figures",
    REPOSITORY_ROOT / "docs" / "manual_evidence",
)

_HTML_SUFFIXES = (".html", ".htm")


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a successful build."""

    output_path: Path
    bytes_written: int
    sha256: str
    seeds: tuple
    teaching_policies: int
    research_arms: int
    research_included: bool


def default_research_path() -> Path:
    """Return the repository's committed study summary path."""
    return REPOSITORY_ROOT / "results" / "healing" / "healing_summary.json"


def _read_text(path: Path, label: str) -> str:
    if not path.exists():
        raise DashboardError(f"{label} not found: {path}")
    if not path.is_file():
        raise DashboardError(f"{label} is not a file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DashboardError(f"{label} is not valid UTF-8 text: {path} ({exc})") from exc
    except OSError as exc:
        raise DashboardError(f"cannot read {label} {path}: {exc}") from exc


def load_demo(path: Path) -> tuple[DemoDocument, bytes]:
    """Read, validate and hash a ``demo.json`` file.

    Returns:
        The validated document and the exact bytes that were parsed (the hash is shown on
        the page as provenance).
    """
    path = Path(path)
    text = _read_text(path, "demo input")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DashboardError(
            f"{path.name}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    document = parse_demo(payload, source=path.name)
    return document, text.encode("utf-8")


def _check_output_path(output: Path, input_path: Path) -> Path:
    output = Path(output)
    resolved = output.expanduser().resolve()
    if resolved.suffix.lower() not in _HTML_SUFFIXES:
        raise DashboardError(
            f"output must be an .html file, got {resolved.name!r}; "
            "choose a path such as index.html"
        )
    if resolved == input_path.expanduser().resolve():
        raise DashboardError("output path is the input demo file; choose a separate .html path")
    if resolved.exists() and resolved.is_dir():
        raise DashboardError(f"output path is a directory: {resolved}")
    for directory in PROTECTED_DIRECTORIES:
        try:
            resolved.relative_to(directory.resolve())
        except ValueError:
            continue
        raise DashboardError(
            f"refusing to write into the research evidence directory {directory}; "
            "write the dashboard next to the demo output instead (for example "
            "--output <demo-dir>/index.html)"
        )
    return resolved


def _write_atomically(path: Path, content: str, *, force: bool) -> bytes:
    """Write ``content`` to ``path`` via a temporary file; return the encoded bytes."""
    resolved = path
    if resolved.exists() and not force:
        raise DashboardError(
            f"output already exists: {resolved}; pass --force to replace this dashboard "
            "(research artifacts are never written by this tool)"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8")
    if not data.rstrip().lower().endswith(b"</html>"):
        raise DashboardError("internal error: generated document is incomplete; nothing was written")
    handle = None
    temporary: Optional[Path] = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb", dir=str(resolved.parent), prefix=".dashboard-", suffix=".tmp", delete=False
        )
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        if resolved.exists() and not force:  # pragma: no cover - defensive re-check
            raise DashboardError(f"output appeared while building: {resolved}")
        os.replace(temporary, resolved)
        temporary = None
    except OSError as exc:
        raise DashboardError(f"cannot write dashboard to {resolved}: {exc}") from exc
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
    return data


def build_dashboard(
    *,
    input_path: Path,
    output_path: Path,
    research_path: Optional[Path] = None,
    include_research: bool = True,
    force: bool = False,
) -> BuildResult:
    """Validate the inputs and write the dashboard.

    Args:
        input_path: ``demo.json`` written by ``main.py demo``.
        output_path: Destination ``.html`` file.
        research_path: Study summary to read; defaults to the repository's committed
            ``results/healing/healing_summary.json``.
        include_research: When ``False`` the research panel states that it was not loaded.
        force: Replace an existing dashboard file.

    Returns:
        A :class:`BuildResult` describing what was written.

    Raises:
        DashboardError: If an input is missing/malformed/inconsistent, the destination is
            unsafe, or the file cannot be written.
    """
    input_path = Path(input_path).expanduser()
    document, raw = load_demo(input_path)
    resolved_output = _check_output_path(Path(output_path).expanduser(), input_path)

    research: Optional[ResearchSummary] = None
    if include_research:
        research_file = Path(research_path).expanduser() if research_path else default_research_path()
        research = load_research(research_file)

    teaching: TeachingPanel = build_teaching_panel()

    provenance = {
        "input_name": input_path.name,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input_sha256_short": hashlib.sha256(raw).hexdigest()[:16],
        "generator": "tools/build_teacher_dashboard.py",
    }
    html = render_dashboard(document, teaching, research, provenance)
    data = _write_atomically(resolved_output, html, force=force)
    return BuildResult(
        output_path=resolved_output,
        bytes_written=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        seeds=tuple(fixture.seed for fixture in document.fixtures),
        teaching_policies=len(teaching.policies),
        research_arms=len(research.chart_arms) if research else 0,
        research_included=research is not None,
    )
