"""Inline SVG charts for the teacher dashboard.

Everything here is plain string building with the Python standard library: the dashboard
must open from the filesystem on a Windows laptop with no web server, no CDN and no
network. Each chart is emitted as an ``<svg>`` element with a ``viewBox`` so it scales on
a projector and in print, and carries ``<title>`` tooltips plus ``data-*`` attributes that
the page's small script uses for the "inspect this block" panel.

Charts are drawn from the validated objects of :mod:`dashboard.model` and
:mod:`dashboard.research`; no value is invented here.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from dashboard.model import Fixture, Incident, Interval, PolicySegment
from dashboard.research import ResearchArm

__all__ = [
    "POLICY_COLORS",
    "GanttRow",
    "Marker",
    "BarItem",
    "LifecycleStage",
    "gantt_chart",
    "horizontal_bars",
    "verification_band",
    "lifecycle_flow",
    "stage_colours",
    "verdict_tone_key",
    "research_bars",
]

#: Restrained policy palette (also used for the lifecycle and the legends).
POLICY_COLORS = {
    "FCFS": "#3d5a80",
    "SJF": "#2f6f4f",
    "Round Robin": "#9a6b1f",
    "Priority": "#6b4a86",
}
POLICY_SHORT = {"FCFS": "FCFS", "SJF": "SJF", "Round Robin": "RR", "Priority": "PRIO"}

INK = "#16212b"
INK_SOFT = "#455666"
MUTED = "#64748b"
GRID = "#e3e9f0"
LINE = "#c9d3de"
SWITCH_FILL = "#b9c4d1"
IDLE_FILL = "#eef1f4"
OK = "#2f6f4f"
WARN = "#a8691b"
BAD = "#a03d3d"
ACCENT = "#1f5f8b"
LEARNED = "#6b4a86"

_TICK_STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000)


def esc(value: object) -> str:
    """HTML-escape an arbitrary value for text and attribute contexts."""
    return html.escape(str(value), quote=True)


def _f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _tick_step(domain: float, target: int = 7) -> int:
    domain = max(float(domain), 1.0)
    for step in _TICK_STEPS:
        if domain / step <= target:
            return step
    return _TICK_STEPS[-1]


def _ticks(domain: float, target: int = 7) -> List[int]:
    step = _tick_step(domain, target)
    count = int(domain // step)
    return [step * index for index in range(count + 1)]


def _svg_open(name: str, width: int, height: int, title: str, desc: str = "") -> str:
    described = f"{name}-desc"
    body = [
        f'<svg class="chart" role="img" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMinYMin meet" aria-labelledby="{name}-title"',
    ]
    if desc:
        body[0] += f' aria-describedby="{described}"'
    body[0] += ">"
    body.append(f'<title id="{name}-title">{esc(title)}</title>')
    if desc:
        body.append(f'<desc id="{described}">{esc(desc)}</desc>')
    return "".join(body)


@dataclass(frozen=True)
class GanttRow:
    """One horizontal lane of the Gantt chart.

    ``burst_time``, ``priority`` and ``waiting_time`` are optional extras used for the
    hover/selection read-out; they are always values recorded for that process.
    """

    pid: int
    arrival_time: int
    completion_time: int
    burst_time: Optional[int] = None
    priority: Optional[int] = None
    waiting_time: Optional[int] = None


@dataclass(frozen=True)
class Marker:
    """A vertical annotation line at a recorded simulation time."""

    time: int
    label: str
    tone: str = WARN


def _wrap_text(text: str, width: int, limit: int = 3) -> List[str]:
    """Greedy word wrap used for in-chart labels (SVG has no automatic wrapping)."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
        lines[-1] = lines[-1][: max(width - 1, 1)] + "\u2026"
    return lines


def _policy_fill(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#64748b")


def gantt_chart(
    *,
    name: str,
    title: str,
    description: str,
    rows: Sequence[GanttRow],
    intervals: Sequence[Interval],
    segments: Sequence[PolicySegment],
    domain_end: int,
    markers: Sequence[Marker] = (),
    show_policy_row: bool = True,
    width: int = 1080,
    seed: int = 0,
) -> str:
    """Draw recorded dispatch intervals as a Gantt chart (one lane per process).

    The top lane is the CPU itself: the exact sequence of recorded dispatches. Below it
    every process gets a lane so its own service slices can be compared. Colours encode
    the policy in force for that dispatch; nothing is interpolated.
    """
    left = 78
    right = 24
    top = 62 if markers else 18
    cpu_height = 28
    gap_after_cpu = 8
    policy_height = 18
    gap_after_policy = 12
    row_height = 22
    row_gap = 3
    axis_band = 44
    plot_width = max(width - left - right, 120)
    row_count = len(rows)
    policy_y = top + cpu_height + gap_after_cpu
    rows_top = policy_y + (policy_height + gap_after_policy if show_policy_row else 0)
    height = rows_top + row_count * (row_height + row_gap) + axis_band
    domain = max(int(domain_end), 1)

    def x(time_value: float) -> float:
        return left + (max(0.0, min(float(time_value), float(domain))) / domain) * plot_width

    by_pid = {row.pid: row for row in rows}

    def process_attributes(interval: Interval) -> str:
        """Recorded process-level facts, so a block can be inspected on its own."""
        row = by_pid.get(interval.pid)
        if row is None:
            return ""
        parts = [
            f'data-arrival="{row.arrival_time}"',
            f'data-completion="{row.completion_time}"',
        ]
        if row.burst_time is not None:
            parts.append(f'data-burst="{row.burst_time}"')
        if row.priority is not None:
            parts.append(f'data-priority="{row.priority}"')
        if row.waiting_time is not None:
            parts.append(f'data-process-waiting="{row.waiting_time}"')
        return " " + " ".join(parts)

    def tooltip(interval: Interval) -> str:
        text = (
            f"P{interval.pid} running {interval.policy} from t={interval.start} to "
            f"t={interval.end} ({interval.duration} time units)"
        )
        row = by_pid.get(interval.pid)
        if row is not None and row.burst_time is not None:
            text += (
                f"; process arrives at t={row.arrival_time}, burst {row.burst_time}, "
                f"priority {row.priority}"
            )
            if row.waiting_time is not None:
                text += f", total waiting {row.waiting_time} time units"
        return text

    parts: List[str] = [_svg_open(name, width, height, title, description)]
    hatch_id = f"switch-hatch-{name}"
    parts.append(
        f'<defs><pattern id="{hatch_id}" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{IDLE_FILL}"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{SWITCH_FILL}" stroke-width="2"/>'
        "</pattern></defs>"
    )

    axis_y = height - axis_band + 8

    # Grid and axis labels first, so the data sits on top of them.
    for tick in _ticks(domain):
        xpos = x(tick)
        parts.append(
            f'<line x1="{xpos:.1f}" y1="{top - 6}" x2="{xpos:.1f}" y2="{axis_y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xpos:.1f}" y="{axis_y + 15:.1f}" text-anchor="middle" '
            f'font-size="10.5" fill="{MUTED}">{tick}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{axis_y:.1f}" x2="{left + plot_width}" y2="{axis_y:.1f}" '
        f'stroke="{LINE}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        f'font-size="10.5" fill="{MUTED}">Simulated time (arbitrary time units)</text>'
    )

    # Row labels.
    parts.append(
        f'<text x="{left - 10}" y="{top + cpu_height / 2 + 4:.1f}" text-anchor="end" '
        f'font-size="11" font-weight="600" fill="{INK_SOFT}">CPU</text>'
    )
    for index, row in enumerate(rows):
        ypos = rows_top + index * (row_height + row_gap)
        parts.append(
            f'<text x="{left - 10}" y="{ypos + row_height / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="10.5" fill="{INK_SOFT}">P{row.pid}</text>'
        )

    # Policy strip: which configuration the engine was actually dispatching under.
    if show_policy_row:
        parts.append(
            f'<text x="{left - 10}" y="{policy_y + policy_height / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="10.5" fill="{INK_SOFT}">policy</text>'
        )
        for segment in segments:
            x1, x2 = x(segment.start), x(segment.end)
            segment_width = max(x2 - x1, 1.4)
            colour = _policy_fill(segment.policy)
            parts.append(
                f'<rect x="{x1:.1f}" y="{policy_y}" width="{segment_width:.1f}" '
                f'height="{policy_height}" rx="2" fill="{colour}" fill-opacity="0.22" '
                f'stroke="{colour}" stroke-opacity="0.55" stroke-width="1">'
                f'<title>{esc(segment.policy)} in force from t={segment.start} to '
                f"t={segment.end} ({segment.decisions} dispatches)</title></rect>"
            )
            if segment_width >= 54:
                parts.append(
                    f'<text x="{(x1 + x2) / 2:.1f}" y="{policy_y + policy_height / 2 + 4:.1f}" '
                    f'text-anchor="middle" font-size="10" font-weight="600" fill="{colour}">'
                    f"{esc(POLICY_SHORT.get(segment.policy, segment.policy))}</text>"
                )

    # CPU lane: one contiguous strip of the exact dispatch sequence. Switch and idle
    # overlays are emitted first, so a hatch can never paint over a dispatch block.
    labels: List[str] = []
    overlays: List[str] = []
    previous: Optional[Interval] = None
    for interval in intervals:
        if previous is None:
            continue
        gap_start, gap_end = previous.end, interval.start
        if interval.switch_before:
            switch_x2 = x(gap_start + interval.switch_before)
            overlays.append(
                f'<rect x="{x(gap_start):.1f}" y="{top}" '
                f'width="{max(switch_x2 - x(gap_start), 1.2):.1f}" height="{cpu_height}" '
                f'fill="url(#{hatch_id})" stroke="{LINE}" stroke-width="0.6">'
                f"<title>Context switch: {interval.switch_before} time unit(s) between "
                f"P{previous.pid} and P{interval.pid}</title></rect>"
            )
        if interval.idle_before:
            idle_start = gap_start + interval.switch_before
            overlays.append(
                f'<rect x="{x(idle_start):.1f}" y="{top}" '
                f'width="{max(x(gap_end) - x(idle_start), 1.2):.1f}" height="{cpu_height}" '
                f'fill="{IDLE_FILL}" stroke="{LINE}" stroke-width="0.6">'
                f"<title>CPU idle for {interval.idle_before} time unit(s)</title></rect>"
            )
        previous = interval
    parts.extend(overlays)

    for interval in intervals:
        x1, x2 = x(interval.start), x(interval.end)
        block_width = max(x2 - x1, 1.4)
        colour = _policy_fill(interval.policy)
        description = tooltip(interval)
        parts.append(
            f'<g class="slice" data-seed="{seed}" '
            f'data-pid="{interval.pid}" data-policy="{esc(interval.policy)}" '
            f'data-start="{interval.start}" data-end="{interval.end}" '
            f'data-switch="{interval.switch_before}" '
            f'data-waiting="{"" if interval.waiting_increment is None else interval.waiting_increment}" '
            f'data-duration="{interval.duration}"{process_attributes(interval)} '
            f'tabindex="0" role="button" aria-label="{esc(description)}">'
            f'<rect x="{x1:.1f}" y="{top}" width="{block_width:.1f}" height="{cpu_height}" '
            f'rx="2.5" fill="{colour}" stroke="#ffffff" stroke-width="1"/>'
            f"<title>{esc(description)}</title></g>"
        )
        # Label only when the block is a few pixels wider than the text needs; narrow
        # dispatches stay unlabelled rather than colliding with their neighbours.
        if block_width >= 6 * len(f"P{interval.pid}"):
            labels.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{top + cpu_height / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="10" font-weight="600" fill="#ffffff" '
                f'pointer-events="none">{esc(f"P{interval.pid}")}</text>'
            )

    # One lane per process: its own service slices, arrival and completion.
    for index, row in enumerate(rows):
        base = rows_top + index * (row_height + row_gap)
        centre = base + row_height / 2
        for interval in intervals:
            if interval.pid != row.pid:
                continue
            x1, x2 = x(interval.start), x(interval.end)
            block_width = max(x2 - x1, 1.4)
            description = tooltip(interval)
            parts.append(
                f'<g class="slice" data-seed="{seed}" '
                f'data-pid="{interval.pid}" data-policy="{esc(interval.policy)}" '
                f'data-start="{interval.start}" data-end="{interval.end}" '
                f'data-switch="{interval.switch_before}" '
                f'data-waiting="{"" if interval.waiting_increment is None else interval.waiting_increment}" '
                f'data-duration="{interval.duration}"{process_attributes(interval)} '
                f'tabindex="0" role="button" aria-label="{esc(description)}">'
                f'<rect x="{x1:.1f}" y="{base + 3}" width="{block_width:.1f}" height="{row_height - 6}" '
                f'rx="2" fill="{_policy_fill(interval.policy)}" fill-opacity="0.9" '
                f'stroke="#ffffff" stroke-width="0.8"/>'
                f"<title>{esc(description)}</title></g>"
            )
        parts.append(
            f'<circle cx="{x(row.arrival_time):.1f}" cy="{centre:.1f}" r="2.4" fill="#ffffff" '
            f'stroke="{MUTED}" stroke-width="1.2">'
            f"<title>P{row.pid} arrives at t={row.arrival_time}</title></circle>"
        )
        parts.append(
            f'<line x1="{x(row.completion_time):.1f}" y1="{centre - 5:.1f}" '
            f'x2="{x(row.completion_time):.1f}" y2="{centre + 5:.1f}" stroke="{INK}" '
            f'stroke-width="1.6"><title>P{row.pid} completes at t={row.completion_time}'
            f"</title></line>"
        )

    parts.extend(labels)
    for marker in markers:
        xpos = x(marker.time)
        parts.append(
            f'<line x1="{xpos:.1f}" y1="{top - 8}" x2="{xpos:.1f}" y2="{axis_y:.1f}" '
            f'stroke="{marker.tone}" stroke-width="1.4" stroke-dasharray="5 4" '
            f'stroke-opacity="0.9"/>'
        )

    # Marker labels last so they are readable.
    for position, marker in enumerate(sorted(markers, key=lambda item: item.time)):
        xpos = x(marker.time)
        label_y = top - 26 if position % 2 == 0 else top - 10
        anchor = "start" if xpos < left + plot_width * 0.55 else "end"
        offset = 5 if anchor == "start" else -5
        parts.append(
            f'<text x="{xpos + offset:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" '
            f'font-size="10.5" font-weight="600" fill="{marker.tone}">{esc(marker.label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------------------
# Horizontal bars
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BarItem:
    """One bar: a label, a value and the colour that identifies its series."""

    label: str
    value: float
    colour: str
    note: str = ""
    strong: bool = False


def horizontal_bars(
    *,
    name: str,
    title: str,
    description: str,
    items: Sequence[BarItem],
    unit: str,
    width: int = 760,
    max_value: Optional[float] = None,
    value_digits: int = 2,
    tick_target: int = 6,
) -> str:
    """Draw labelled horizontal bars with a value next to each bar and a shared axis."""
    label_width = min(300, max(150, int(max((len(item.label) for item in items), default=12) * 6.9) + 20))
    right = 96
    top = 14
    row_height = 34
    axis_band = 38
    plot_width = max(width - label_width - right, 120)
    height = top + row_height * len(items) + axis_band
    domain = max(max_value if max_value is not None else max((item.value for item in items), default=1.0), 1e-9)
    domain = domain * 1.04

    def x(value: float) -> float:
        return label_width + (max(0.0, min(float(value), domain)) / domain) * plot_width

    axis_y = height - axis_band + 6
    parts: List[str] = [_svg_open(name, width, height, title, description)]
    for tick in _ticks(domain, tick_target):
        xpos = x(tick)
        parts.append(
            f'<line x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{axis_y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xpos:.1f}" y="{axis_y + 15:.1f}" text-anchor="middle" '
            f'font-size="10.5" fill="{MUTED}">{tick}</text>'
        )
    parts.append(
        f'<line x1="{label_width}" y1="{axis_y:.1f}" x2="{label_width + plot_width}" '
        f'y2="{axis_y:.1f}" stroke="{LINE}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{label_width + plot_width / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        f'font-size="10.5" fill="{MUTED}">{esc(unit)}</text>'
    )
    for index, item in enumerate(items):
        base = top + index * row_height
        bar_height = 20
        parts.append(
            f'<text x="{label_width - 12}" y="{base + bar_height / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="11.5" '
            f'font-weight="{"700" if item.strong else "500"}" fill="{INK}">'
            f"{esc(item.label)}</text>"
        )
        parts.append(
            f'<rect x="{label_width}" y="{base + 1}" width="{max(x(item.value) - label_width, 1.4):.1f}" '
            f'height="{bar_height}" rx="3" fill="{item.colour}">'
            f"<title>{esc(item.label)}: {_f(item.value, value_digits)} {esc(unit)}"
            f"{esc(' (' + item.note + ')') if item.note else ''}</title></rect>"
        )
        parts.append(
            f'<text x="{x(item.value) + 8:.1f}" y="{base + bar_height / 2 + 4:.1f}" '
            f'font-size="11.5" font-weight="600" fill="{INK_SOFT}">{_f(item.value, value_digits)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def research_bars(
    *,
    name: str,
    title: str,
    description: str,
    arms: Sequence[ResearchArm],
    width: int = 820,
) -> str:
    """Mean waiting time per study arm, coloured by family (fixed / rule / learned)."""
    colours = {"fixed": "#3d5a80", "control": "#7d8b99", "rule": "#2f6f4f", "learned": LEARNED}
    items = [
        BarItem(
            label=arm.label,
            value=arm.mean_waiting_time,
            colour=colours.get(arm.kind, "#64748b"),
            note="lower is better",
            strong=arm.is_primary,
        )
        for arm in arms
    ]
    return horizontal_bars(
        name=name,
        title=title,
        description=description,
        items=items,
        unit="Mean waiting time (simulated time units per process)",
        width=width,
    )


# --------------------------------------------------------------------------------------
# Verification band
# --------------------------------------------------------------------------------------


def verification_band(
    *, name: str, title: str, description: str, incident: Incident, width: int = 820
) -> str:
    """Draw the verifier's accept/reject boundaries and the observed residual.

    The bands come from the incident record itself (``accept_residual_factor``,
    ``reject_residual_factor``); the marker is the recorded ``closure_residual``.
    """
    residual = incident.closure_residual
    height = 186
    left = 26
    right = 26
    plot_width = width - left - right
    plot_y = 64
    plot_height = 44
    if residual is None:
        residual = 0.0
    domain = max(
        abs(residual) * 1.3,
        incident.accept_residual_factor * 1.6,
        abs(incident.reject_residual_factor) * 1.6,
        0.35,
    )

    def x(value: float) -> float:
        return left + ((max(-domain, min(domain, value)) + domain) / (2 * domain)) * plot_width

    bands = (
        (-domain, incident.reject_residual_factor, BAD, "reject"),
        (incident.reject_residual_factor, incident.accept_residual_factor, WARN, "inconclusive"),
        (incident.accept_residual_factor, domain, OK, "accept"),
    )
    parts: List[str] = [_svg_open(name, width, height, title, description)]
    for start, end, colour, label in bands:
        x1, x2 = x(start), x(end)
        parts.append(
            f'<rect x="{x1:.1f}" y="{plot_y}" width="{max(x2 - x1, 1.0):.1f}" '
            f'height="{plot_height}" fill="{colour}" fill-opacity="0.16" '
            f'stroke="{colour}" stroke-opacity="0.5" stroke-width="0.8"/>'
        )
        if x2 - x1 >= 6.2 * len(label) + 10:
            parts.append(
                f'<text x="{(x1 + x2) / 2:.1f}" y="{plot_y + plot_height / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="10.5" fill="{colour}" font-weight="600">'
                f"{esc(label)}</text>"
            )
    axis_y = plot_y + plot_height + 4
    parts.append(
        f'<line x1="{left}" y1="{axis_y:.1f}" x2="{left + plot_width}" y2="{axis_y:.1f}" '
        f'stroke="{LINE}" stroke-width="1"/>'
    )
    tick_values = [(-domain, _f(-domain)), (incident.reject_residual_factor, _f(incident.reject_residual_factor)), (0.0, "0"), (incident.accept_residual_factor, "+" + _f(incident.accept_residual_factor)), (domain, "+" + _f(domain))]
    for value, label in tick_values:
        xpos = x(value)
        parts.append(
            f'<line x1="{xpos:.1f}" y1="{axis_y:.1f}" x2="{xpos:.1f}" y2="{axis_y + 5:.1f}" '
            f'stroke="{MUTED}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xpos:.1f}" y="{axis_y + 18:.1f}" text-anchor="middle" '
            f'font-size="10.5" fill="{MUTED}">{esc(label)}</text>'
        )
    footnote = _wrap_text(
        "Phase-aware closure residual: at or below the reject boundary the change failed; between "
        "the boundaries it is inconclusive; at or above the accept boundary it is verified.",
        int(max(60, (width - left - right) / 5.9)),
        3,
    )
    for index, line in enumerate(footnote):
        parts.append(
            f'<text x="{left:.1f}" y="{height - 12 - (len(footnote) - 1 - index) * 14:.1f}" '
            f'font-size="10.5" fill="{MUTED}">{esc(line)}</text>'
        )
    marker_x = x(residual)
    parts.append(
        f'<line x1="{marker_x:.1f}" y1="{plot_y - 14:.1f}" x2="{marker_x:.1f}" '
        f'y2="{axis_y:.1f}" stroke="{INK}" stroke-width="1.6"/>'
    )
    parts.append(
        f'<circle cx="{marker_x:.1f}" cy="{plot_y - 14:.1f}" r="6" fill="{INK}" '
        f'stroke="#ffffff" stroke-width="1.6">'
        f"<title>Observed closure residual {_f(residual, 4)}</title></circle>"
    )
    anchor = "start" if marker_x < left + plot_width * 0.6 else "end"
    offset = 9 if anchor == "start" else -9
    parts.append(
        f'<text x="{marker_x + offset:.1f}" y="{plot_y - 22:.1f}" text-anchor="{anchor}" '
        f'font-size="11" font-weight="700" fill="{INK}">observed residual {_f(residual, 4)}</text>'
    )
    parts.append(
        f'<text x="{left:.1f}" y="22" font-size="11.5" fill="{INK_SOFT}">'
        f"Verdict: <tspan font-weight=\"700\" fill=\"{INK}\">{esc(incident.verdict)}</tspan>"
        f" &#183; reason: {esc(incident.verifier_reason)}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------------------
# Lifecycle flow
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleStage:
    """One stage of the monitor -> ... -> decision loop.

    ``tone`` is the display colour used by the inline SVG. ``tone_key`` names the state in
    words (``accent``, ``muted``, ``warn``, ``ok``, ``bad``) so a consumer that colours DOM
    elements never has to reverse-engineer a hex value or guess from the stage title.
    """

    title: str
    lines: Tuple[str, ...]
    tone: str
    detail: str = ""
    tone_key: str = "accent"


def lifecycle_flow(
    *, name: str, title: str, description: str, stages: Sequence[LifecycleStage], width: int = 1080
) -> str:
    """Draw the control loop as an ordered flow of stage cards."""
    gap = 9
    box_height = 126
    top = 10
    height = top + box_height + 18
    box_width = (width - 2 * gap) / max(len(stages), 1) - gap
    title_chars = max(8, int(box_width / 7.4))
    parts: List[str] = [_svg_open(name, width, height, title, description)]
    for index, stage in enumerate(stages):
        x = gap + index * (box_width + gap)
        parts.append(
            f'<g><title>{esc(stage.detail or " ".join((stage.title, *stage.lines)))}</title>'
            f'<rect x="{x:.1f}" y="{top}" width="{box_width:.1f}" height="{box_height}" '
            f'rx="5" fill="#ffffff" stroke="{stage.tone}" stroke-width="1.4"/>'
            f'<rect x="{x:.1f}" y="{top}" width="{box_width:.1f}" height="5" rx="2.5" '
            f'fill="{stage.tone}"/>'
            f'<text x="{x + 9:.1f}" y="{top + 24:.1f}" font-size="10" font-weight="700" '
            f'fill="{stage.tone}" letter-spacing="0.4">STEP {index + 1}</text>'
        )
        title_lines = _wrap_text(stage.title, title_chars, 2)
        for title_index, title_line in enumerate(title_lines):
            parts.append(
                f'<text x="{x + 9:.1f}" y="{top + 41 + title_index * 14:.1f}" font-size="11.5" '
                f'font-weight="700" fill="{INK}">{esc(title_line)}</text>'
            )
        body_top = top + 41 + 14 * len(title_lines) + 3
        for line_index, line in enumerate(stage.lines[:3]):
            parts.append(
                f'<text x="{x + 9:.1f}" y="{body_top + line_index * 13:.1f}" '
                f'font-size="10" fill="{INK_SOFT}">{esc(line)}</text>'
            )
        parts.append("</g>")
        if index < len(stages) - 1:
            arrow_x = x + box_width
            parts.append(
                f'<path d="M {arrow_x + 2:.1f} {top + box_height / 2 - 4:.1f} l 5 4 l -5 4" '
                f'fill="none" stroke="{MUTED}" stroke-width="1.4" stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def interval_rows(fixture: Fixture) -> List[GanttRow]:
    """Return one Gantt lane per process, in pid order, with its recorded timing."""
    return [
        GanttRow(
            pid=outcome.pid,
            arrival_time=outcome.arrival_time,
            completion_time=outcome.completion_time,
            burst_time=outcome.burst_time,
            priority=outcome.priority,
            waiting_time=outcome.waiting_time,
        )
        for outcome in sorted(fixture.outcomes, key=lambda row: row.pid)
    ]


def stage_colours(verdict: str) -> Tuple[str, str]:
    """Return the accent colour for the verification and decision stages."""
    if verdict == "accepted":
        return OK, OK
    if verdict == "rejected":
        return BAD, BAD
    if verdict == "inconclusive":
        return WARN, WARN
    if verdict == "held":
        return MUTED, MUTED
    return ACCENT, ACCENT


def verdict_tone_key(verdict: str) -> str:
    """Return the semantic tone of a verdict: ``ok``, ``bad``, ``warn``, ``muted`` or ``accent``.

    The live console colours the verification and final-decision stages from this key. It
    used to compare the stage's hex colour against three hard-coded values, which painted
    an inconclusive rollback with the "accent" colour and the final decision with the
    success colour in every case.
    """
    return {
        "accepted": "ok",
        "rejected": "bad",
        "inconclusive": "warn",
        "held": "muted",
    }.get(verdict, "accent")
