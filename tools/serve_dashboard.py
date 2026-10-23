"""Start the live teacher dashboard (FastAPI + the project's simulation engine).

Usage
-----
    python tools/serve_dashboard.py                    # http://127.0.0.1:8000
    python tools/serve_dashboard.py --port 9000        # another port
    python tools/serve_dashboard.py --host 0.0.0.0     # reachable from other machines
    python tools/serve_dashboard.py --no-browser       # do not try to open a window

The server binds to loopback by default. It never writes inside the repository: each run
gets its own temporary directory under the operating system's temp path, and the
committed research artifacts are only read.

Requirements: ``pip install -r requirements-dashboard.txt`` (installs FastAPI and Uvicorn
on top of the pinned project requirements).
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Serve the live CPU scheduling dashboard on a local port."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the dashboard in the default browser",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level (default: info)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the server until interrupted."""
    args = parse_args(argv)
    try:
        import uvicorn

        from dashboard.api import app
    except ModuleNotFoundError as error:
        missing = error.name or "a dependency"
        print(f"error: missing dependency {missing!r}.", file=sys.stderr)
        print(
            "Install the dashboard requirements first:\n"
            "  python -m pip install -r requirements-dashboard.txt",
            file=sys.stderr,
        )
        return 2
    except Exception as error:  # pragma: no cover - startup guard
        print(f"error: the dashboard could not start: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print("Adaptive CPU Scheduling Dashboard")
    print(f"  serving : {url}")
    print(f"  engine  : {REPOSITORY_ROOT}")
    print("  stop    : press Ctrl+C in this terminal")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
