"""Build the self-contained classroom dashboard from a saved demo trace.

Usage (from the repository root, same interpreter that ran the demo)::

    python tools/build_teacher_dashboard.py \
        --input results/demo/demo.json \
        --output results/demo/index.html

Windows PowerShell (one line, or continue with a backtick):

    .\.venv\Scripts\python.exe tools\build_teacher_dashboard.py --input "$env:USERPROFILE\os-teacher-demo\demo.json" --output "$env:USERPROFILE\os-teacher-demo\index.html"

The script only reads ``demo.json`` (and, unless ``--no-research`` is given, the committed
``results/healing/healing_summary.json`` plus ``healing_comparisons.csv``). It writes one
HTML file whose charts are inline SVG: no server, no CDN and no network access are needed
to open it. Existing files are not replaced without ``--force``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``python tools/build_teacher_dashboard.py`` puts ``tools/`` on sys.path, so the
# repository root (which holds ``dashboard/``, ``scheduler/`` and ``config.py``) has to be
# added explicitly. This is what makes the documented Windows command work from any
# working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dashboard.cli import main  # noqa: E402 - path setup must happen first

if __name__ == "__main__":
    raise SystemExit(main())
