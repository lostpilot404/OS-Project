"""Figures for the Review 2 report, rendered from the committed artifacts.

Two kinds of figure live here.

* **Diagram figures** are rendered from the Mermaid source that the DOCX builder embeds
  verbatim, by :func:`render_mermaid`: a small layered layout that understands the subset of
  Mermaid this report uses (``flowchart LR/TB/TD``, ``subgraph``, rectangle / stadium /
  diamond nodes, chained and labelled edges, dotted edges, back edges).  Rendering the same
  text in both deliverables is what keeps them in agreement.
* **Data figures** are drawn from the artifacts loaded by :mod:`review2_data`, so a figure
  cannot show a number the study did not record.

Nothing here re-runs an experiment.  ``python tools/review2_figures.py`` renders every figure
into ``figures/review2/`` for inspection.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.path as mpl_path  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from review2_data import Artifacts  # noqa: E402

NAVY = "#1F3864"
STEEL = "#2E75B6"
LIGHT = "#D9E2F3"
PALE = "#F2F5FB"
GREY = "#5A5A5A"
GREEN = "#3F7D3F"
RED = "#A63A3A"
AMBER = "#B8860B"

OUT_OF_SCOPE = {"architecture"}  # drawn with primitives, Mermaid source still carried in DOCX


# --------------------------------------------------------------------------------------
# shared styling
# --------------------------------------------------------------------------------------
def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.0,
        "axes.edgecolor": "#8C9BB5",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })


def _finish(fig, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def wrap(text: str, width: int) -> str:
    """Wrap a label onto at most a few lines, without breaking words."""
    return "\n".join(textwrap.wrap(text, width)) or text


def _plain(text: str) -> str:
    return re.sub(r"<br\s*/?>", "\n", text)


def measure_text(text: str, font_size: float) -> Tuple[float, float]:
    """Measure a multi-line label in inches with the figure font, at ``font_size`` points.

    Node boxes are sized from this measurement rather than from a per-character estimate, so
    a label cannot overflow its box whichever characters it contains.
    """
    fig = plt.figure(figsize=(0.1, 0.1), dpi=200)
    renderer = fig.canvas.get_renderer()
    widths, heights = [], []
    for line in text.split("\n"):
        artist = fig.text(0.0, 0.0, line, fontsize=font_size, family="DejaVu Sans")
        box = artist.get_window_extent(renderer=renderer)
        widths.append(box.width / fig.dpi)
        heights.append(box.height / fig.dpi)
        artist.remove()
    plt.close(fig)
    return (max(widths) if widths else 0.0, sum(heights))


# --------------------------------------------------------------------------------------
# Mermaid rendering
# --------------------------------------------------------------------------------------
_TERM_RE = re.compile(
    r'([A-Za-z][\w]*)\s*'
    r'(?:\(\[\s*"?(.*?)"?\s*\]\)'      # stadium: ID(["label"])
    r'|\[\[(.*?)\]\]'                    # subroutine: ID[["label"]]
    r'|\[(.*?)\]'                          # rectangle: ID["label"]
    r'|\{(.*?)\})?')                       # diamond: ID{"label"}
_EDGE_SPLIT_RE = re.compile(r'(-\.[^.]*?\.->|--\s*(?:"[^"]*"\s*)?-->|-->)')


def _edge_label(token: str) -> Tuple[str, bool]:
    """Extract the label and the dotted flag from one edge token."""
    dotted = token.startswith("-.")
    match = re.search(r'"([^"]*)"', token)
    return (match.group(1) if match else "", dotted)


def _term(token: str, add_node) -> str:
    """Register one node term (``ID``, ``ID["label"]``, ``ID(["label"])``, ``ID{"label"}``)."""
    match = _TERM_RE.match(token.strip())
    if not match:
        return ""
    node_id = match.group(1)
    stadium, subroutine, rectangle, diamond = (match.group(index) for index in (2, 3, 4, 5))
    if stadium is not None:
        shape, label = "stadium", stadium
    elif subroutine is not None:
        shape, label = "rect", subroutine
    elif rectangle is not None:
        shape, label = "rect", rectangle
    elif diamond is not None:
        shape, label = "diamond", diamond
    else:
        shape, label = "rect", node_id
    add_node(node_id, (label or node_id).strip().strip('"'), shape)
    return node_id


def parse_mermaid(text: str) -> Dict:
    """Parse the Mermaid subset used by this report into nodes, edges and clusters.

    Supported: ``flowchart LR|TB|TD``, ``subgraph NAME["title"] … end``, node shapes
    rectangle, stadium and diamond, node definitions written inline on an edge line, chains
    such as ``A --> B --> C`` (with or without an intervening labelled edge), labelled edges
    (``-- "yes" -->``) and dotted edges (``-. "text" .->``).
    """
    nodes: List[Tuple[str, str, str]] = []
    index: Dict[str, int] = {}
    edges: List[Tuple[str, str, str, bool]] = []
    clusters: List[Tuple[str, List[str]]] = []
    direction = "TB"
    cluster: List[str] = []
    in_cluster = False

    def add_node(node_id: str, label: str, shape: str) -> None:
        if not node_id:
            return
        if node_id not in index:
            index[node_id] = len(nodes)
            nodes.append((node_id, label, shape))
        elif label != node_id:
            position = index[node_id]
            nodes[position] = (node_id, label, shape)
        if in_cluster and node_id not in cluster:
            cluster.append(node_id)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        if line.startswith("flowchart") or line.startswith("graph"):
            direction = line.split()[1]
            continue
        if line.startswith("subgraph"):
            match = re.match(r'subgraph\s+([\w]+)\s*(?:\["(.*)"\])?', line)
            cluster = []
            in_cluster = True
            clusters.append((match.group(2) if match and match.group(2) else "group", cluster))
            continue
        if line == "end":
            in_cluster = False
            continue
        # a plain node statement
        if not _EDGE_SPLIT_RE.search(line):
            _term(line, add_node)
            continue
        # a chain of terms joined by edges
        parts = _EDGE_SPLIT_RE.split(line)
        previous = ""
        for piece in parts:
            if _EDGE_SPLIT_RE.fullmatch(piece):
                pending = piece
                continue
            node_id = _term(piece, add_node)
            if previous and node_id:
                label, dotted = _edge_label(pending)
                edges.append((previous, node_id, label, dotted))
            if node_id:
                previous = node_id
    clusters = [(name, members) for name, members in clusters if members]
    return {"direction": direction, "nodes": nodes, "edges": edges, "clusters": clusters}


def _layers(parsed: Dict) -> Tuple[Dict[str, int], List[List[str]]]:
    """Longest-path layering; edges pointing backwards in the layering become back edges."""
    order = [node[0] for node in parsed["nodes"]]
    index = {name: position for position, name in enumerate(order)}
    layer = {name: 0 for name in order}
    forward = [(s, t) for s, t, _, _ in parsed["edges"] if index[s] <= index[t]]
    for _ in range(len(order) + 2):
        changed = False
        for source, target in forward:
            if layer[target] < layer[source] + 1:
                layer[target] = layer[source] + 1
                changed = True
        if not changed:
            break
    rows: List[List[str]] = [[] for _ in range(max(layer.values()) + 1 if layer else 1)]
    for name in order:
        rows[layer[name]].append(name)
    return layer, rows


#: fraction of a diamond's bounding box that the rhombus itself covers; arrows aimed at the
#: bounding box would otherwise appear to start and end inside the shape
DIAMOND_INSET = 0.62


def _anchor(source: Tuple[float, float, float, float],
            target: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float],
                                                                Tuple[float, float]]:
    """Pick facing sides for an arrow between two node boxes ``(x, y, width, height)``."""
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    dx, dy = tx - sx, ty - sy
    if dx == 0 and dy == 0:
        return (sx, sy + sh / 2.0), (tx, ty - th / 2.0)
    if abs(dx) >= abs(dy):
        start = (sx + (sw / 2.0 if dx > 0 else -sw / 2.0), sy)
        end = (tx - (tw / 2.0 if dx > 0 else -tw / 2.0), ty)
    else:
        start = (sx, sy + (sh / 2.0 if dy > 0 else -sh / 2.0))
        end = (tx, ty - (th / 2.0 if dy > 0 else -th / 2.0))
    return start, end


def render_mermaid(text: str, path: Path, font_size: float = 7.0,
                   wrap_width: int = 26, node_pad: float = 0.13, gap_cross: float = 0.20,
                   gap_along: float = 0.48, max_per_row: int = 5) -> Path:
    """Render the Mermaid subset used by this report to a PNG.

    The layout is a layered drawing computed in inches: node sizes come from their text at
    ``font_size`` points (labels are wrapped to ``wrap_width`` characters), layers are
    stacked along the declared flow direction, and a long left-to-right chain is wrapped
    after ``max_per_row`` layers so that it fits a page instead of becoming an unreadably
    thin strip.  Edges become arrows between the facing sides of their end nodes; an edge
    that points backwards in the layering is drawn as an arc.
    """
    _style()
    parsed = parse_mermaid(text)
    layer, rows = _layers(parsed)
    shapes = {name: shape for name, _, shape in parsed["nodes"]}
    labels = {}
    for name, label, shape in parsed["nodes"]:
        text = _plain(label)
        measure = min(wrap_width, 19) if shape == "diamond" else wrap_width
        if shape == "diamond":
            text = text.replace("\n", " ")
        labels[name] = "\n".join(wrap(part, measure) for part in text.split("\n"))
    horizontal = parsed["direction"] == "LR"

    measured = {name: measure_text(labels[name], font_size) for name in labels}

    def size(name: str) -> Tuple[float, float]:
        text_w, text_h = measured[name]
        lines = labels[name].count("\n") + 1
        width = text_w + 2 * node_pad
        height = max(0.28, text_h * (lines / max(lines, 1)) * 1.06 + 2 * node_pad * 0.8)
        if shapes[name] == "diamond":
            # the label sits inside the rhombus: the bounding box must be substantially wider
            # and taller than the text, because the shape narrows away from its centre
            width = text_w * 1.62 + 2 * node_pad
            height = text_h * 2.00 + 2 * node_pad * 0.8 + 0.06
        return width, height

    sizes = {name: size(name) for name, _, _ in parsed["nodes"]}

    def visual(name: str) -> Tuple[float, float, float, float]:
        """The shape an arrow should stop at: the rhombus for a diamond, the box otherwise."""
        x, y = positions[name]
        width, height = sizes[name]
        if shapes[name] == "diamond":
            return (x, y, width * DIAMOND_INSET, height * DIAMOND_INSET)
        return (x, y, width, height)

    # ---- geometry ---------------------------------------------------------------------
    positions: Dict[str, Tuple[float, float]] = {}
    if horizontal:
        columns = [0] * len(rows)          # column index of each layer inside its row
        row_of = [0] * len(rows)
        for position in range(len(rows)):
            row_of[position] = position // max_per_row
            columns[position] = position % max_per_row
        column_width: Dict[Tuple[int, int], float] = {}
        for position, row in enumerate(rows):
            key = (row_of[position], columns[position])
            column_width[key] = max(sizes[name][0] for name in row)
        row_heights: Dict[int, float] = {}
        for position, row in enumerate(rows):
            row_heights[row_of[position]] = max(
                row_heights.get(row_of[position], 0.0),
                max(sizes[name][1] for name in row))
        row_offsets: Dict[int, float] = {}
        running = 0.0
        for row_index in sorted(row_heights):
            row_offsets[row_index] = running
            running += row_heights[row_index] + gap_along * 1.7
        column_offsets: Dict[Tuple[int, int], float] = {}
        for row_index in sorted(row_heights):
            running = 0.0
            for column in range(max_per_row):
                key = (row_index, column)
                if key not in column_width:
                    continue
                column_offsets[key] = running
                running += column_width[key] + gap_along
        for position, row in enumerate(rows):
            row_index, column = row_of[position], columns[position]
            x_centre = column_offsets[(row_index, column)] + column_width[(row_index, column)] / 2.0
            used = sum(sizes[name][1] for name in row) + gap_cross * (len(row) - 1)
            cursor = row_offsets[row_index] + row_heights[row_index] / 2.0 - used / 2.0
            for name in row:
                width, height = sizes[name]
                positions[name] = (x_centre, -(cursor + height / 2.0))
                cursor += height + gap_cross
    else:
        layer_heights = [max(sizes[name][1] for name in row) for row in rows]
        offsets: List[float] = []
        running = 0.0
        for height in layer_heights:
            offsets.append(running)
            running += height + gap_along
        for position, row in enumerate(rows):
            y_centre = -(offsets[position] + layer_heights[position] / 2.0)
            used = sum(sizes[name][0] for name in row) + gap_cross * (len(row) - 1)
            cursor = -used / 2.0
            for name in row:
                width, height = sizes[name]
                positions[name] = (cursor + width / 2.0, y_centre)
                cursor += width + gap_cross

    node_row = {name: (row_of[position] if horizontal else layer[name])
                for position, row in enumerate(rows) for name in row}
    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for name, members in parsed["clusters"]:
        member_positions = [positions[m] for m in members if m in positions]
        if not member_positions:
            continue
        pad = 0.26
        boxes[f"__cluster__{name}"] = (
            min(p[0] for p in member_positions) - pad,
            max(p[0] for p in member_positions) + pad,
            min(p[1] for p in member_positions) - pad,
            max(p[1] for p in member_positions) + pad,
        )

    node_bottom = {name: positions[name][1] - sizes[name][1] / 2.0 for name in positions}
    node_top = {name: positions[name][1] + sizes[name][1] / 2.0 for name in positions}
    node_left = {name: positions[name][0] - sizes[name][0] / 2.0 for name in positions}
    node_right = {name: positions[name][0] + sizes[name][0] / 2.0 for name in positions}
    # bands (not layers): in a wrapped horizontal layout one band holds several layers
    band_bottom: Dict[int, float] = {}
    band_top: Dict[int, float] = {}
    for name, band in node_row.items():
        band_bottom[band] = min(band_bottom.get(band, 1e9), node_bottom[name])
        band_top[band] = max(band_top.get(band, -1e9), node_top[name])
    row_bottom, row_top = band_bottom, band_top

    # the viewport must contain the node *boxes*, not their centres
    xs = [node_left[n] for n in positions] + [node_right[n] for n in positions]
    ys = [node_bottom[n] for n in positions] + [node_top[n] for n in positions]
    for box in boxes.values():
        xs += [box[0], box[1]]
        ys += [box[2], box[3]]
    for source, target, _label, _dotted in parsed["edges"]:
        if layer[target] <= layer[source] and node_row[source] == node_row[target]:
            sx, sy = positions[source]
            tx, ty = positions[target]
            if abs(sy - ty) < 0.05:
                ys.append(sy + sizes[source][1] / 2.0 + 0.45 * abs(sx - tx) / 2.0 + 0.34)
    # count the side lanes the drawing will need, so the canvas can be wide enough for them
    side_lanes = 0
    for source, target, _label, _dotted in parsed["edges"]:
        if layer[target] <= layer[source]:
            continue
        if node_row[target] > node_row[source] + 1 and positions[target][0] <= positions[source][0]:
            side_lanes += 1
    bottom_lane = min(ys) - 0.34
    left_lane = min(xs) - 0.30
    right_edge = max(xs) + 0.30 + 0.26 * side_lanes
    top_edge = max(ys) + 0.30
    xlim = (left_lane - 0.18, right_edge)
    ylim = (bottom_lane - 0.22, top_edge)
    fig, ax = plt.subplots(figsize=(max(2.6, xlim[1] - xlim[0]), max(1.5, ylim[1] - ylim[0])))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")

    for key, box in boxes.items():
        ax.add_patch(mpatches.FancyBboxPatch(
            (box[0], box[2]), box[1] - box[0], box[3] - box[2],
            boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=0.9,
            edgecolor=STEEL, facecolor=PALE, linestyle=(0, (4, 2)), zorder=1))
        ax.text(box[0] + 0.06, box[3] - 0.05, key[len("__cluster__"):], fontsize=6.2,
                color=STEEL, fontweight="bold", va="top", ha="left", zorder=2)

    faces = {"rect": "#EAF0FA", "stadium": "#E3F0E3", "diamond": "#FBF3DC"}
    for name, _, shape in parsed["nodes"]:
        x, y = positions[name]
        width, height = sizes[name]
        if shape == "diamond":
            ax.add_patch(mpatches.Polygon(
                [(x, y + height / 2.0), (x + width / 2.0, y), (x, y - height / 2.0),
                 (x - width / 2.0, y)], closed=True, linewidth=1.0, edgecolor=NAVY,
                facecolor=faces[shape], zorder=3))
        else:
            radius = height / 2.0 if shape == "stadium" else 0.05
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - width / 2.0, y - height / 2.0), width, height,
                boxstyle=f"round,pad=0.01,rounding_size={radius}", linewidth=1.0,
                edgecolor=NAVY, facecolor=faces[shape], zorder=3))
        ax.text(x, y, labels[name], ha="center", va="center", fontsize=font_size,
                color="#111111", zorder=4, linespacing=1.28)

    lane_use = {"wrap": 0, "side": 0, "back": 0}
    label_slots: Dict[Tuple[float, float], int] = {}

    for source, target, label, dotted in parsed["edges"]:
        sx, sy = positions[source]
        tx, ty = positions[target]
        sw, sh = sizes[source]
        tw, th = sizes[target]
        vx, vy, vw, vh = visual(source)
        ux, uy, uw, uh = visual(target)
        style = (0, (2.4, 1.8)) if dotted else "solid"
        back = layer[target] <= layer[source]
        row_s, row_t = node_row[source], node_row[target]
        rad = 0.0
        label_xy = None

        if not back and ty < sy - 0.05:
            if abs(tx - sx) < 0.35:
                points = [(vx, vy - vh / 2.0), (tx, uy + uh / 2.0)]
                label_xy = (sx + 0.52, (sy - sh / 2.0 + ty + th / 2.0) / 2.0)
            elif row_t == row_s + 1:
                channel = (row_bottom[row_s] + row_top[row_t]) / 2.0
                points = [(vx, vy - vh / 2.0), (vx, channel), (tx, channel),
                          (tx, uy + uh / 2.0)]
                label_xy = ((sx + tx) / 2.0, channel + 0.13)
            elif row_t == row_s:
                channel = row_bottom[row_s] - 0.20 - 0.24 * lane_use["wrap"]
                lane_use["wrap"] += 1
                points = [(sx, sy - sh / 2.0), (sx, channel), (tx, channel),
                          (tx, ty + th / 2.0)]
                label_xy = ((sx + tx) / 2.0, channel + 0.13)
            else:
                # skips a whole row: leave sideways, run down a clear side lane and come back
                # in along a channel just above the target row
                index = lane_use["side"]
                lane_use["side"] += 1
                right = tx <= sx
                lane = (right_edge + 0.24 + 0.24 * index) if right else (
                    left_lane - 0.24 - 0.24 * index)
                run_y = row_top[row_t] + 0.13 + 0.20 * index
                start = (sx + sw / 2.0, sy) if right else (sx - sw / 2.0, sy)
                points = [start, (lane, sy), (lane, run_y), (tx, run_y), (tx, ty + th / 2.0)]
                label_xy = ((lane + tx) / 2.0, run_y + 0.12)
        elif not back:
            # sideways: an arrow between the facing sides is enough
            start, end = _anchor((sx, sy, sw, sh), (tx, ty, tw, th))
            points = [start, end]
            label_xy = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0 + 0.10)
        elif row_s == row_t:
            # a feedback arrow inside one band: arc over the top of that band
            start = (vx, vy + vh / 2.0)
            end = (tx, ty + th / 2.0)
            points = [start, end]
            rad = -0.45 if tx < sx else 0.45
            label_xy = (tx, start[1] + 0.26)
        else:
            # feedback into an upper row: out of the bottom, under everything, up the left
            # lane, then along the clear channel just above the target row
            offset = 0.26 * lane_use["back"]
            lane_use["back"] += 1
            here = bottom_lane - offset
            upper = row_bottom[row_t] - 0.22 - offset
            points = [(sx, sy - sh / 2.0), (sx, here), (left_lane - offset, here),
                      (left_lane - offset, upper), (tx, upper), (tx, ty - th / 2.0)]
            label_xy = ((left_lane - offset + tx) / 2.0, upper - 0.14)

        mpl_points = mpl_path.Path(points, [mpl_path.Path.MOVETO]
                                  + [mpl_path.Path.LINETO] * (len(points) - 1))
        ax.add_patch(mpatches.FancyArrowPatch(
            path=mpl_points, arrowstyle="-|>", mutation_scale=7, linewidth=0.9,
            color=STEEL if dotted else NAVY, linestyle=style,
            connectionstyle=f"arc3,rad={rad}", zorder=5))
        if label and label_xy:
            key = (round(label_xy[0], 2), round(label_xy[1], 2))
            taken = label_slots.get(key, 0)
            label_slots[key] = taken + 1
            ax.text(label_xy[0] + 0.38 * taken, label_xy[1] - 0.155 * taken, label,
                    ha="center", va="center", fontsize=6.0, color="#333333", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.94))
    return _finish(fig, path)


# --------------------------------------------------------------------------------------
# data figures
# --------------------------------------------------------------------------------------
def dataset_profile(data: Artifacts, path: Path) -> Path:
    _style()
    frame = data.dataset
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5))
    y = np.arange(len(frame))
    labels = [name.replace("_", " ") for name in frame["family"]]
    axes[0].barh(y, frame["span_max"], color=LIGHT, edgecolor=NAVY, height=0.62)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("latest arrival of the trace (time units)")
    axes[0].set_title("Arrival spread")
    for position, value in zip(y, frame["span_max"]):
        axes[0].text(value + 1.2, position, f"{int(value)}", va="center", fontsize=6.4,
                     color=GREY)
    axes[0].set_xlim(0, float(frame["span_max"].max()) * 1.22)
    for position, (_, row) in enumerate(frame.iterrows()):
        axes[1].plot([row["burst_min"], row["burst_max"]], [position, position],
                     color=NAVY, linewidth=2.0, solid_capstyle="round", zorder=3)
        axes[1].scatter([row["burst_mean"]], [position], color=STEEL, s=16, zorder=4,
                        label="mean" if position == 0 else None)
    axes[1].set_yticks(y, ["" for _ in labels])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("CPU burst (time units)")
    axes[1].set_title("Burst range and mean")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Workload families over the evaluation splits ("
                 f"{int(frame['traces'].iloc[0])} traces each)", fontsize=8.6, color=NAVY, y=1.12)
    return _finish(fig, path)


def waiting_time(data: Artifacts, path: Path) -> Path:
    _style()
    rows = data.arm_rows()
    names = [row["label"] for row in rows]
    means = [float(row["avg_waiting_time"]) for row in rows]
    p95 = [float(row["p95_waiting_time"]) for row in rows]
    colors = {"fixed": "#B7C4DD", "control": "#C9A227", "rule": "#2E75B6",
              "learned": NAVY}
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    ax.bar(x - 0.2, means, width=0.4, color=[colors[row["kind"]] for row in rows],
           edgecolor=NAVY, linewidth=0.6, label="mean waiting time")
    ax.bar(x + 0.2, p95, width=0.4, color=[colors[row["kind"]] for row in rows],
           alpha=0.38, edgecolor=NAVY, linewidth=0.6, hatch="//", label="95th percentile")
    for position, value in zip(x - 0.2, means):
        ax.text(position, value + 4, f"{value:.1f}", ha="center", fontsize=6.2, color="#222222")
    ax.set_xticks(x, [wrap(name, 13) for name in names], fontsize=6.6)
    ax.set_ylabel("waiting time (simulated units)")
    ax.set_title("Held-out waiting time per arm (210 traces; learned arms 5 × 210)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#E3E8F2", linewidth=0.7)
    ax.set_axisbelow(True)
    return _finish(fig, path)


FOREST_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("q_recovery", "rule_wait", "learned − rule"),
    ("rule_wait", "detector_only", "rule − control"),
    ("q_recovery", "detector_only", "learned − control"),
    ("rule_wait", "fixed_sjf", "rule − fixed SJF"),
    ("q_recovery", "fixed_sjf", "learned − fixed SJF"),
    ("q_recovery_from_sjf", "fixed_sjf", "learned from SJF − fixed SJF"),
    ("rule_fairness", "fixed_fcfs", "fairness rule − fixed FCFS"),
)


def forest(data: Artifacts, path: Path) -> Path:
    _style()
    rows = []
    for target, reference, label in FOREST_ROWS:
        comparison = data.comparison(target, reference)
        if comparison:
            rows.append((label, comparison))
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    y = np.arange(len(rows))[::-1]
    for position, (label, comparison) in zip(y, rows):
        low, high = comparison["ci95_low"], comparison["ci95_high"]
        mean = comparison["mean_difference"]
        significant = (low > 0) or (high < 0)
        colour = RED if (significant and mean > 0) else (GREEN if significant else GREY)
        ax.plot([low, high], [position, position], color=colour, linewidth=1.6)
        ax.scatter([mean], [position], color=colour, s=22, zorder=4)
        ax.text(high + 0.8, position, f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]",
                va="center", fontsize=6.2, color=colour)
    ax.axvline(0.0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, [row[0] for row in rows], fontsize=6.8)
    ax.set_xlabel("paired difference in mean waiting time (simulated units)")
    ax.set_title("Paired differences with 95 % bootstrap intervals (positive = target is worse)")
    ax.grid(axis="x", color="#E3E8F2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(min(-4.0, min(row[1]["ci95_low"] for row in rows) - 3.0),
                max(row[1]["ci95_high"] for row in rows) + 7.5)
    return _finish(fig, path)


def verdicts(data: Artifacts, path: Path) -> Path:
    _style()
    frame = data.verdict_counts()
    audit = [data.audit_summary_row(arm) for arm in
             ("rule_wait", "rule_fairness", "q_recovery")]
    audit = [row for row in audit if row]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = np.arange(len(frame))[::-1]
    totals = frame["interventions"].astype(float)
    accepted = frame["accepted"].astype(float)
    rejected = frame["rejected"].astype(float)
    inconclusive = frame["inconclusive"].astype(float)
    axes[0].barh(y, 100 * accepted / totals, color=GREEN, height=0.55, label="accepted")
    axes[0].barh(y, 100 * rejected / totals, left=100 * accepted / totals, color=RED,
                 height=0.55, label="rejected")
    axes[0].barh(y, 100 * inconclusive / totals,
                 left=100 * (accepted + rejected) / totals, color="#C9C9C9", height=0.55,
                 label="inconclusive")
    axes[0].set_yticks(y, [str(arm) for arm in frame["arm"]])
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("share of interventions (%)")
    axes[0].set_title("In-run verdicts (n annotated)")
    axes[0].legend(frameon=False, loc="lower center", ncol=3, fontsize=6.2,
                   bbox_to_anchor=(0.5, -0.52))
    for position, arm, total in zip(y, frame["arm"], totals):
        axes[0].text(101, position, f"n={int(total)}", va="center", fontsize=6.2, color=GREY)
    axes[0].set_xlim(0, 118)

    x = np.arange(len(audit))
    helped = [row["helped"] for row in audit]
    harmful = [row["harmful"] for row in audit]
    axes[1].bar(x - 0.19, helped, width=0.38, color=GREEN, label="helped")
    axes[1].bar(x + 0.19, harmful, width=0.38, color=RED, label="harmful")
    axes[1].set_xticks(x, [str(row["arm"]).replace("_", "\n") for row in audit], fontsize=6.2)
    axes[1].set_ylabel("audited interventions")
    axes[1].set_title("Counterfactual audit")
    axes[1].legend(frameon=False, fontsize=6.4)
    for position, row in zip(x, audit):
        axes[1].text(position, max(row["helped"], row["harmful"]) + 4,
                     f"false rec. {row['false_recovery_rate']:.0%}", ha="center",
                     fontsize=5.8, color=GREY)
    axes[1].set_ylim(0, max(max(helped), max(harmful)) * 1.35)
    return _finish(fig, path)


def demo_gantt(data: Artifacts, path: Path) -> Path:
    _style()
    demo = data.demo
    decisions = demo["decisions"]
    incident = demo["incidents"][0] if demo["incidents"] else {}
    colours = {"fcfs": "#8DA0CB", "sjf": "#FC8D62", "round_robin": "#66C2A5",
               "priority": "#E78AC3"}
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    for decision in decisions:
        policy = str(decision["policy"]).lower()
        start, end = float(decision["executed_from"]), float(decision["executed_to"])
        ax.barh(0.0, max(end - start, 0.25), left=start, height=0.42,
                color=colours.get(policy, "#BBBBBB"),
                edgecolor=NAVY if decision.get("policy_changed") else "white", linewidth=1.0)
        ax.text((start + end) / 2.0, 0.0, f"P{int(decision['selected_pid'])}",
                ha="center", va="center", fontsize=5.6, color="#111111")
    if incident:
        trigger = float(incident["simulated_time"])
        ax.axvline(trigger, color=RED, linewidth=1.2, linestyle="--")
        ax.annotate(f"detected: {str(incident['diagnosis']).replace('_', ' ')} → "
                    f"{incident['action_name']} → {incident['verdict']}",
                    xy=(trigger, 0.30), xytext=(trigger, 0.62), ha="center", fontsize=6.4,
                    color=RED, arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.8))
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.95)
    ax.set_xlabel("simulated time (decision intervals)")
    ax.set_title(f"Demonstration trace: {demo['workload']['family']}, "
                 f"arm {demo['arm']['name']}, {len(decisions)} decision intervals")
    handles = [mpatches.Patch(color=colour, label=name.upper())
               for name, colour in colours.items()]
    handles.append(mpatches.Patch(facecolor="white", edgecolor=NAVY,
                                  label="change applied"))
    ax.legend(handles=handles, frameon=False, ncol=5, fontsize=6.0, loc="lower center",
              bbox_to_anchor=(0.5, -0.62))
    return _finish(fig, path)


def coverage(data: Artifacts, path: Path) -> Path:
    _style()
    frame = data.coverage
    curve = data.coverage_curve
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5),
                             gridspec_kw={"width_ratios": [1.0, 1.1]})
    x = np.arange(len(frame))
    axes[0].bar(x, frame["observed_reachable_state_action_pairs"], width=0.62,
                color=LIGHT, edgecolor=NAVY, label="reachable from observed contexts")
    axes[0].bar(x, frame["visited_state_action_pairs"], width=0.34, color=NAVY,
                label="visited during training")
    axes[0].set_xticks(x, [f"seed\n{int(seed)}" for seed in frame["training_seed"]], fontsize=6.2)
    axes[0].set_ylabel("state-action pairs")
    axes[0].set_title("Coverage after 1200 training traces")
    axes[0].legend(frameon=False, fontsize=6.0, loc="upper left")
    for position, row in zip(x, frame.itertuples()):
        axes[0].text(position, row.visited_state_action_pairs + 1.2,
                     f"{row.state_action_coverage_of_observed_reachable:.0%}",
                     ha="center", fontsize=5.9, color=GREY)
    axes[0].set_ylim(0, float(frame["observed_reachable_state_action_pairs"].max()) * 1.3)

    for seed, subset in curve.groupby("training_seed"):
        axes[1].plot(subset["episodes"], subset["visited_state_action_pairs"], marker="o",
                     markersize=2.4, linewidth=1.0, label=f"seed {int(seed)}")
    axes[1].set_xlabel("training traces (episodes)")
    axes[1].set_ylabel("visited state-action pairs")
    axes[1].set_title("Coverage growth")
    axes[1].legend(frameon=False, fontsize=6.0)
    axes[1].grid(color="#E3E8F2", linewidth=0.7)
    axes[1].set_axisbelow(True)
    return _finish(fig, path)


# --------------------------------------------------------------------------------------
# architecture (primitives; the Mermaid source is carried in the DOCX)
# --------------------------------------------------------------------------------------
def architecture(data: Artifacts, path: Path) -> Path:
    """Two-band architecture panel: the environment above, the healing loop below.

    The loop is read as a ring: the environment hands the controller causal observations,
    the controller works left to right through monitor, detector, diagnosis, policy and
    verifier, and replies with one configuration along a lane outside the bands.  The
    post-run counterfactual audit is drawn as a dashed path underneath, because it is
    computed after the trace has finished and is never visible inside the loop.
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, width, height, title, body, fs=6.7):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.07",
            linewidth=1.0, edgecolor=NAVY, facecolor=LIGHT, zorder=3))
        ax.text(x + width / 2.0, y + height * 0.70, title, ha="center", va="center",
                fontsize=fs + 0.6, fontweight="bold", color="#111111", zorder=4)
        ax.text(x + width / 2.0, y + height * 0.30, body, ha="center", va="center",
                fontsize=fs - 0.9, color="#333333", linespacing=1.3, zorder=4)

    def arrow(points, label="", colour=NAVY, dotted=False, label_xy=None, rotation=0,
              label_ha="center"):
        path = mpl_path.Path(points, [mpl_path.Path.MOVETO]
                             + [mpl_path.Path.LINETO] * (len(points) - 1))
        ax.add_patch(mpatches.FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=colour,
            linestyle=(0, (2.4, 1.8)) if dotted else "solid", zorder=5))
        if label:
            x, y = label_xy if label_xy else points[len(points) // 2]
            ax.text(x, y, label, ha=label_ha, va="center", fontsize=6.0, color=colour,
                    rotation=rotation, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95))

    # ---- bands ------------------------------------------------------------------------
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.10, 4.55), 9.80, 1.75, boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=0.9, edgecolor=STEEL, facecolor=PALE, linestyle=(0, (4, 2)), zorder=1))
    ax.text(0.30, 6.20, "Environment — scheduler/runtime.py (owns the future and the "
            "accounting)", fontsize=6.9, color=STEEL, fontweight="bold", va="top", zorder=2)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.10, 0.10), 9.80, 3.62, boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=0.9, edgecolor=STEEL, facecolor=PALE, linestyle=(0, (4, 2)), zorder=1))
    ax.text(9.72, 3.64, "Self-healing controller — healing/ (causal inputs only)",
            fontsize=6.9, color=STEEL, fontweight="bold", va="top", ha="right", zorder=2)

    # ---- environment -----------------------------------------------------------------
    box(0.45, 4.85, 2.90, 1.05, "Private event list",
        "future arrivals\nnever exposed to the controller")
    box(3.55, 4.85, 2.90, 1.05, "Single FIFO ready queue",
        "arrived work, remaining bursts\none CPU, one switch cost")
    box(6.65, 4.85, 2.90, 1.05, "Accounting",
        "completions, waiting times,\ndecision records, schedule trace")

    # ---- healing loop -----------------------------------------------------------------
    box(0.45, 2.10, 2.90, 1.05, "Monitor",
        "rolling completed-work statistic,\nreference and dispersion band")
    box(3.55, 2.10, 2.90, 1.05, "Detector",
        "threshold, hysteresis, cooldown,\nwork and congestion gates")
    box(6.65, 2.10, 2.90, 1.05, "Diagnosis",
        "visible ready-set evidence →\none primary cause")
    box(0.45, 0.55, 2.90, 1.05, "Policy",
        "declared rule or tabular\nQ-learning over 648 states")
    box(3.55, 0.55, 2.90, 1.05, "Verifier",
        "phase baseline → residual →\naccepted / rejected / inconclusive")
    box(6.65, 0.55, 2.90, 1.05, "Controller",
        "apply, roll back, safe fallback,\nreward, incident ledger")

    # ---- causal input, internal flow, reply, and the post-run audit --------------------
    arrow([(1.90, 4.83), (1.90, 3.17)], "RuntimeObservation (arrived work only)",
          label_xy=(3.70, 4.16))
    arrow([(3.36, 2.62), (3.56, 2.62)])
    arrow([(6.46, 2.62), (6.66, 2.62)])
    arrow([(3.36, 1.08), (3.56, 1.08)])
    arrow([(6.46, 1.08), (6.66, 1.08)])
    arrow([(6.75, 2.08), (2.35, 1.62)], "diagnosis + masked actions", label_xy=(5.05, 1.90))
    arrow([(8.10, 1.62), (8.10, 2.05), (10.35, 2.05), (10.35, 5.38), (9.57, 5.38)],
          "one configuration (policy, quantum)", label_xy=(10.62, 3.60), rotation=90)
    arrow([(8.10, 0.55), (8.10, 0.30), (1.60, 0.30), (1.60, 0.53)], colour=STEEL,
          dotted=True, label="counterfactual replay after the run (post-run, never visible "
                              "inside the loop)", label_xy=(6.30, 0.44))
    return _finish(fig, path)


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
MERMAID_OPTIONS = {
    "methodology_pipeline": {"wrap_width": 20, "max_per_row": 5},
    "architecture": {"wrap_width": 26, "max_per_row": 3},
    "wf_m1": {"wrap_width": 24, "max_per_row": 4},
    "wf_m2": {"wrap_width": 24, "max_per_row": 4},
    "wf_m3": {"wrap_width": 22, "max_per_row": 5},
    "wf_m4": {"wrap_width": 18, "max_per_row": 2},
    "wf_m5": {"wrap_width": 20, "max_per_row": 3},
    "wf_m6": {"wrap_width": 22, "max_per_row": 5},
    "wf_m7": {"wrap_width": 24, "max_per_row": 4},
    "wf_m8": {"wrap_width": 24, "max_per_row": 3},
}


DATA_FIGURES = {
    "dataset_profile": dataset_profile,
    "waiting_time": waiting_time,
    "forest": forest,
    "verdicts": verdicts,
    "demo_gantt": demo_gantt,
    "coverage": coverage,
    "architecture": architecture,
}


def render(key: str, data: Artifacts, out_dir: Path, mermaid: Dict[str, str]) -> Path:
    """Render one figure by key. Mermaid keys are drawn from their own source text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fig_{key}.png"
    if key in DATA_FIGURES:
        return DATA_FIGURES[key](data, path)
    if key in mermaid:
        return render_mermaid(mermaid[key], path, **MERMAID_OPTIONS.get(key, {}))
    raise KeyError(f"no figure defined for {key!r}")


def render_all(data: Artifacts, mermaid: Dict[str, str], out_dir: Path) -> Dict[str, Path]:
    """Render every figure the report references, returning ``key -> path``."""
    keys = list(DATA_FIGURES) + [key for key in mermaid if key not in OUT_OF_SCOPE]
    return {key: render(key, data, out_dir, mermaid) for key in keys}


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from review2_content import MERMAID  # noqa: PLC0415

    root = Path(__file__).resolve().parent.parent
    artifacts = Artifacts(root)
    for key, path in render_all(artifacts, MERMAID, root / "figures" / "review2").items():
        print(f"{key:24} -> {path.relative_to(root)} "
              f"({path.stat().st_size / 1024:.0f} KiB)")
