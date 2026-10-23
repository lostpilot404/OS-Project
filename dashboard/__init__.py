"""Self-contained teacher dashboard built from recorded simulator output.

The package is a *consumer* of the project's artifacts:

* :mod:`dashboard.model` validates ``demo.json`` and cross-checks it against the
  per-process outcomes and recorded metrics;
* :mod:`dashboard.teaching` executes the documented three-process example with the real
  scheduler classes;
* :mod:`dashboard.research` reads the committed held-out study tables;
* :mod:`dashboard.charts` and :mod:`dashboard.render` draw inline SVG and assemble one
  HTML file that opens from disk with no server, CDN or network.

Nothing here decides scheduling, recomputes study results or trains a model.
"""

from dashboard.build import BuildResult, build_dashboard, default_research_path
from dashboard.model import DashboardError, DemoDocument, parse_demo
from dashboard.research import ResearchSummary, load_research
from dashboard.teaching import TeachingPanel, build_teaching_panel

__all__ = [
    "BuildResult",
    "DashboardError",
    "DemoDocument",
    "ResearchSummary",
    "TeachingPanel",
    "build_dashboard",
    "build_teaching_panel",
    "default_research_path",
    "load_research",
    "parse_demo",
]
