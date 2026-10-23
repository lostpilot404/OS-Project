"""HTTP API for the live scheduling dashboard.

Run it with ``python tools/serve_dashboard.py`` (or ``uvicorn dashboard.api:app``).  The
API is a thin, validated shell around :mod:`dashboard.service`, which in turn calls the
project's scheduler, runtime simulator, healing controller and metric code.  There is no
second simulation engine anywhere in this package.

Routes
------
``GET  /api/health``
    Liveness of the service *and* of the simulation engine: a one-process schedule is
    executed as a probe, so a broken engine cannot report healthy.
``GET  /api/examples``
    The loadable scenarios and the options each one accepts.
``POST /api/simulations``
    Validate a request, execute it now and return the recorded result.
``GET  /api/runs``
    Compact summaries of this session's most recent runs (newest first), for the overview.
``GET  /api/simulations/{run_id}``
    Re-read a result the server still holds in memory (bounded history).
``GET  /api/simulations/{run_id}/export``
    Download exactly the stored result as JSON.
``POST /api/comparisons``
    Run all four policies on one workload (either the workload of a stored run or an
    explicit process list).
``GET  /api/research``
    The committed held-out study tables, read-only.

Isolation and safety
--------------------
* Requests never carry filesystem paths; the server chooses every path it touches.
* Each run is stored in its own ``tempfile.mkdtemp`` directory, removed when the run
  leaves the bounded history or when the server shuts down.
* Research artifacts are opened read-only; nothing is written inside the repository.
* Concurrent runs never share state: the engine objects are constructed per request and
  the registry assigns an independent directory per run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard import service
from dashboard.model import DashboardError
from dashboard.scenarios import catalogue as scenario_catalogue

__all__ = ["app", "create_app", "RunRegistry"]

LOGGER = logging.getLogger("dashboard.api")

#: How many completed runs stay addressable (and on disk) at once.
HISTORY_LIMIT = 32
#: At most this many simulations execute at the same time; further requests get HTTP 429.
MAX_CONCURRENT_RUNS = 4

WEB_DIR = Path(__file__).resolve().parent / "web"


class SimulationRequest(BaseModel):
    """Body of ``POST /api/simulations``."""

    scenario: str = Field(..., description="Scenario id from GET /api/examples")
    options: Dict[str, Any] = Field(default_factory=dict, description="Scenario options")


class ComparisonRequest(BaseModel):
    """Body of ``POST /api/comparisons``.

    Either reference a stored run (``run_id``) or send a process list explicitly; the
    workload is validated by the engine either way.
    """

    run_id: Optional[str] = None
    processes: Optional[List[Dict[str, Any]]] = None
    quantum: int = 4
    switching_cost: int = 0


@dataclass(frozen=True)
class StoredRun:
    """One completed run: its payload, its exported bytes and its private directory."""

    run_id: str
    scenario_id: str
    created_at: float
    payload: Dict[str, Any]
    json_bytes: bytes
    directory: Path


class RunRegistry:
    """Bounded, thread-safe store of completed runs with isolated directories."""

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._runs: "OrderedDict[str, StoredRun]" = OrderedDict()
        self._closed = False

    def add(self, scenario_id: str, payload: Dict[str, Any]) -> StoredRun:
        """Persist one result in a fresh temporary directory and register it."""
        if self._closed:
            raise RuntimeError("the run registry is closed")
        run_id = uuid.uuid4().hex
        directory = Path(tempfile.mkdtemp(prefix=f"os-dashboard-{run_id[:8]}-"))
        document = {"run_id": run_id, "status": "completed", "result": payload}
        json_bytes = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        target = directory / "result.json"
        temporary = directory / ".result.json.tmp"
        temporary.write_bytes(json_bytes)
        os.replace(temporary, target)
        stored = StoredRun(
            run_id=run_id,
            scenario_id=scenario_id,
            created_at=time.time(),
            payload=payload,
            json_bytes=json_bytes,
            directory=directory,
        )
        with self._lock:
            self._runs[run_id] = stored
            while len(self._runs) > self._limit:
                _, evicted = self._runs.popitem(last=False)
                shutil.rmtree(evicted.directory, ignore_errors=True)
        return stored

    def get(self, run_id: str) -> Optional[StoredRun]:
        with self._lock:
            return self._runs.get(run_id)

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._runs)

    def size(self) -> int:
        """Number of runs the registry currently holds.

        This is the one figure the interface may show as "runs in this session": it counts
        every retained run, not just the summaries a single request happens to return.
        """
        with self._lock:
            return len(self._runs)

    @property
    def limit(self) -> int:
        """Number of runs this registry keeps before evicting the oldest."""
        return self._limit

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return a compact summary of the most recent runs, newest first.

        Only the fields the interface shows are returned (no charts), so the overview page
        can list this session's activity without downloading every payload.
        """
        with self._lock:
            runs = list(self._runs.values())[-limit:]
        runs.reverse()
        summaries = []
        for run in runs:
            result = run.payload
            healing = result.get("healing") or {}
            incident = healing.get("incident") or {}
            summaries.append(
                {
                    "run_id": run.run_id,
                    "scenario_id": run.scenario_id,
                    "scenario_name": (result.get("scenario") or {}).get("name", run.scenario_id),
                    "kind": result.get("kind", ""),
                    "executed_at": result.get("executed_at"),
                    "status": "completed",
                    "final_policy": (result.get("schedule") or {}).get("final_policy"),
                    "mean_waiting_time": (result.get("metrics") or {}).get("avg_waiting_time"),
                    "verdict": incident.get("verdict"),
                }
            )
        return summaries

    def close(self) -> None:
        with self._lock:
            self._closed = True
            while self._runs:
                _, evicted = self._runs.popitem(last=False)
                shutil.rmtree(evicted.directory, ignore_errors=True)


class _RunGuard:
    """Cap concurrent engine executions and reject excess requests instead of queueing."""

    def __init__(self, capacity: int = MAX_CONCURRENT_RUNS) -> None:
        self._semaphore = threading.BoundedSemaphore(capacity)

    def __enter__(self) -> "_RunGuard":
        if not self._semaphore.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail={
                    "type": "busy",
                    "message": (
                        "the simulation service is already running the maximum number of "
                        "simulations; wait for one to finish and try again"
                    ),
                },
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._semaphore.release()


def _engine_probe() -> Dict[str, Any]:
    """Execute a one-process schedule to prove the simulation engine is importable and working."""
    started = time.perf_counter()
    try:
        from config import SchedulerConfig
        from scheduler import FCFS
        from workload.models import Process, Workload

        workload = Workload("health-probe", [Process(1, 0, 1, 1)])
        result = FCFS(SchedulerConfig()).run(workload)
        if result.total_elapsed_time != 1:
            raise RuntimeError(
                f"health probe produced elapsed time {result.total_elapsed_time}, expected 1"
            )
    except Exception as error:  # pragma: no cover - exercised by the health test
        LOGGER.exception("engine health probe failed")
        return {
            "available": False,
            "detail": f"{type(error).__name__}: {error}",
            "probe_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    return {
        "available": True,
        "detail": "FCFS scheduled a one-process probe workload successfully",
        "probe_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _error(status: int, kind: str, message: str, **extra: Any) -> JSONResponse:
    body = {"error": {"type": kind, "message": message, **extra}}
    return JSONResponse(status_code=status, content=body)


def create_app(registry: Optional[RunRegistry] = None) -> FastAPI:
    """Build the FastAPI application (a factory so tests can use an isolated registry)."""
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        application.state.registry.close()

    application = FastAPI(
        lifespan=lifespan,
        title="Adaptive CPU Scheduling Dashboard",
        version="1.0.0",
        description=(
            "Live interface to the project's self-healing CPU scheduling simulation engine. "
            "Every response is produced by executing the engine at request time."
        ),
    )
    application.state.registry = registry or RunRegistry()
    application.state.guard = _RunGuard()
    application.state.started_at = time.time()

    @application.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = first.get("msg", "invalid request")
        return _error(
            422,
            "invalid_request",
            f"{location}: {message}" if location else message,
            field=location or None,
        )

    @application.exception_handler(404)
    async def _not_found(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else "not found"
        return _error(404, "not_found", message)

    @application.middleware("http")
    async def _no_store(request: Request, call_next):  # type: ignore[no-untyped-def]
        response: Response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/api/health")
    def health() -> Dict[str, Any]:
        engine = _engine_probe()
        info = service.engine_info()
        return {
            "status": "ok" if engine["available"] else "degraded",
            "service": "os-project-dashboard",
            "version": application.version,
            "uptime_s": round(time.time() - application.state.started_at, 1),
            "engine": {**info, **engine},
            "simulation": {
                "max_concurrent_runs": MAX_CONCURRENT_RUNS,
                "history_limit": application.state.registry.limit,
                "stored_runs": application.state.registry.size(),
            },
        }

    @application.get("/api/examples")
    def examples() -> Dict[str, Any]:
        catalogue = list(scenario_catalogue())
        return {"scenarios": catalogue, "count": len(catalogue)}

    @application.post("/api/simulations")
    def create_simulation(request: SimulationRequest) -> JSONResponse:
        started = time.perf_counter()
        with application.state.guard:
            try:
                result = service.execute(request.scenario, request.options)
            except service.SimulationError as error:
                raise SimulationHTTPError(
                    422,
                    "invalid_request",
                    error.message,
                    {"field": error.field} if error.field else {},
                ) from error
            except DashboardError as error:
                LOGGER.exception("engine output rejected by the dashboard schema")
                raise SimulationHTTPError(
                    500,
                    "engine_output_rejected",
                    f"the engine produced output the dashboard schema rejected: {error}",
                ) from error
            except Exception as error:
                LOGGER.exception("simulation failed")
                raise SimulationHTTPError(
                    500,
                    "engine_failure",
                    f"{type(error).__name__}: {error}",
                ) from error
        stored = application.state.registry.add(request.scenario, result)
        return JSONResponse(
            status_code=201,
            content={
                "run_id": stored.run_id,
                "status": "completed",
                "server_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "result": result,
            },
        )

    @application.get("/api/runs")
    def recent_runs() -> Dict[str, Any]:
        registry = application.state.registry
        runs = registry.recent()
        # ``count`` is how many runs the registry holds (the same figure /api/health reports
        # as ``stored_runs``); ``returned`` is how many summaries this response carries, which
        # is capped so the overview never downloads a long history. Reporting the page size as
        # the count made the interface show two different "runs in this session" figures.
        return {
            "runs": runs,
            "count": registry.size(),
            "returned": len(runs),
            "history_limit": registry.limit,
        }

    @application.get("/api/simulations/{run_id}")
    def get_simulation(run_id: str) -> JSONResponse:
        stored = application.state.registry.get(run_id)
        if stored is None:
            raise SimulationHTTPError(
                404,
                "unknown_run",
                "that run is not in the server's history; run the scenario again",
            )
        return JSONResponse(
            content={"run_id": stored.run_id, "status": "completed", "result": stored.payload}
        )

    @application.get("/api/simulations/{run_id}/export")
    def export_simulation(run_id: str) -> Response:
        stored = application.state.registry.get(run_id)
        if stored is None:
            raise SimulationHTTPError(
                404,
                "unknown_run",
                "that run is no longer available, so there is nothing to export",
            )
        filename = f"os-dashboard-{stored.scenario_id}-{stored.run_id[:8]}.json"
        return Response(
            content=stored.json_bytes,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @application.post("/api/comparisons")
    def create_comparison(request: ComparisonRequest) -> JSONResponse:
        processes: Iterable[Dict[str, Any]]
        quantum = request.quantum
        switching_cost = request.switching_cost
        source_run_id = request.run_id
        if request.run_id:
            stored = application.state.registry.get(request.run_id)
            if stored is None:
                raise SimulationHTTPError(
                    404,
                    "unknown_run",
                    "that run is not in the server's history; run the scenario again",
                )
            processes = stored.payload["workload"]["processes"]
            quantum = int(stored.payload.get("settings", {}).get("quantum", quantum))
            switching_cost = int(
                stored.payload.get("settings", {}).get("switching_cost", switching_cost)
            )
        elif request.processes:
            processes = request.processes
        else:
            raise SimulationHTTPError(
                422,
                "invalid_request",
                "send either 'run_id' or a non-empty 'processes' list",
                {"field": "processes"},
            )
        with application.state.guard:
            try:
                comparison = service.compare_policies(
                    list(processes), quantum=quantum, switching_cost=switching_cost,
                    source_run_id=source_run_id,
                )
            except service.SimulationError as error:
                raise SimulationHTTPError(
                    422,
                    "invalid_request",
                    error.message,
                    {"field": error.field} if error.field else {},
                ) from error
            except Exception as error:
                LOGGER.exception("comparison failed")
                raise SimulationHTTPError(
                    500, "engine_failure", f"{type(error).__name__}: {error}"
                ) from error
        return JSONResponse(status_code=201, content={"status": "completed", "result": comparison})

    @application.get("/api/research")
    def research() -> Dict[str, Any]:
        try:
            return service.research_panel()
        except Exception as error:
            LOGGER.exception("research artifacts could not be read")
            raise SimulationHTTPError(
                503,
                "research_unavailable",
                f"the committed research artifacts could not be read: {error}",
            ) from error

    application.add_exception_handler(
        SimulationHTTPError,
        lambda _request, exc: _error(exc.status_code, exc.kind, exc.message, **exc.extra),
    )

    if WEB_DIR.exists():
        # Assets must never be served from a stale cache: the page and its script are edited
        # together, and a cached copy of the shell that points at an older script is exactly
        # how a browser ends up rendering unstyled HTML with no behaviour.
        class _RevalidatedStatic(StaticFiles):
            """Static files are revalidated on every load, so an edited script is never stale."""

            async def get_response(self, path, scope):
                response = await super().get_response(path, scope)
                response.headers.setdefault("Cache-Control", "no-cache")
                return response

        application.mount("/static", _RevalidatedStatic(directory=WEB_DIR), name="static")

        @application.get("/")
        def index() -> HTMLResponse:
            # The asset URLs carry a content stamp so a reload after an edit always fetches
            # the matching stylesheet and script, while the HTML itself is never stored.
            shell = (WEB_DIR / "index.html").read_text(encoding="utf-8")
            for name in ("styles.css", "app.js"):
                asset = WEB_DIR / name
                stamp = hashlib.sha1(asset.read_bytes()).hexdigest()[:10] if asset.exists() else "0"
                shell = shell.replace(f'href="/static/{name}"', f'href="/static/{name}?v={stamp}"')
                shell = shell.replace(f'src="/static/{name}"', f'src="/static/{name}?v={stamp}"')
            return HTMLResponse(
                shell, headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}
            )

    return application


class SimulationHTTPError(HTTPException):
    """An error with the structure the frontend renders (type, message, extra fields)."""

    def __init__(
        self, status_code: int, kind: str, message: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.kind = kind
        self.message = message
        self.extra = extra or {}


app = create_app()
