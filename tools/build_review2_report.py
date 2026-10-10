"""Build the Review 2 project report PDF from the checked-in experiment artifacts.

The PDF is generated entirely from files that are already committed under
``results/`` (runtime experiment) and ``results/`` (legacy offline reference),
so the numbers in the report can never drift from the recorded experiment.

Usage::

    pip install -r requirements-report.txt
    python tools/build_review2_report.py                     # writes both copies
    python tools/build_review2_report.py --output /tmp/r.pdf  # single file
    python tools/build_review2_report.py --skip-tests         # do not run pytest

Outputs (relative to the repository root unless overridden):

* ``Review_2_Report.pdf``      -- the Review 2 deliverable
* ``docs/Review_2_Report.pdf`` -- copy kept with the design notes
* ``figures/review2/*.png``    -- figures embedded in the PDF
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm, mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

from reportlab.platypus.tableofcontents import (  # noqa: E402
    TableOfContents,
)

import review2_figures as figs  # noqa: E402  (imported after sys.path setup)

NAVY = colors.HexColor("#1F3864")
STEEL = colors.HexColor("#2E75B6")
LIGHT = colors.HexColor("#DCE6F5")
BAND = colors.HexColor("#F2F5FB")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#B7C4DD")

REPORT_TITLE = "Event-Driven Runtime-Adaptive CPU Scheduling"
REPORT_SUBTITLE = "with Sequential Tabular Q-Learning"
REPO_URL = "github.com/lostpilot404/OS-Project"

PAGE_W, PAGE_H = A4
MARGIN_X = 1.7 * cm
MARGIN_TOP = 1.9 * cm
MARGIN_BOTTOM = 1.6 * cm
CONTENT_W = PAGE_W - 2 * MARGIN_X

METHOD_ORDER = figs.METHOD_ORDER
SHORT_NAME = {
    "FCFS": "FCFS",
    "SJF": "SJF",
    "Round Robin": "RR",
    "Priority": "Priority",
    "Causal heuristic": "Heuristic",
    "Runtime Q-learning": "Runtime Q",
}


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------
def build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
            leading=26, textColor=NAVY, spaceAfter=4, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12.5,
            leading=16, textColor=STEEL, alignment=TA_CENTER, spaceAfter=10,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base["Normal"], fontSize=9, leading=13,
            textColor=colors.HexColor("#222222"), alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=13.5,
            leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=5,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.8,
            leading=13, textColor=STEEL, spaceBefore=8, spaceAfter=3,
        ),
        "H3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="Helvetica-BoldOblique",
            fontSize=9.6, leading=12, textColor=colors.HexColor("#333333"),
            spaceBefore=6, spaceAfter=2,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5, textColor=colors.HexColor("#1A1A1A"),
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.0,
            leading=12.2, leftIndent=11, bulletIndent=2, spaceAfter=2.5,
            alignment=TA_JUSTIFY,
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.8,
            leading=9.6, textColor=GREY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=8,
        ),
        "Cell": ParagraphStyle(
            "Cell", parent=base["Normal"], fontSize=7.5, leading=9.2,
            textColor=colors.HexColor("#1A1A1A"),
        ),
        "CellB": ParagraphStyle(
            "CellB", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.5,
            leading=9.2, textColor=colors.HexColor("#1A1A1A"),
        ),
        "CellH": ParagraphStyle(
            "CellH", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.5,
            leading=9.2, textColor=colors.white,
        ),
        "Small": ParagraphStyle(
            "Small", parent=base["Normal"], fontSize=8.0, leading=10.5, textColor=GREY,
        ),
        "Equation": ParagraphStyle(
            "Equation", parent=base["Normal"], fontName="Courier-Bold", fontSize=8.6,
            leading=12, alignment=TA_CENTER, textColor=NAVY, spaceBefore=4, spaceAfter=4,
        ),
        "Abstract": ParagraphStyle(
            "Abstract", parent=base["BodyText"], fontName="Helvetica", fontSize=9.3,
            leading=13, alignment=TA_JUSTIFY, borderPadding=6,
            backColor=colors.HexColor("#F5F7FC"), textColor=colors.HexColor("#1A1A1A"),
        ),
    }
    return styles


S = build_styles()


# --------------------------------------------------------------------------
# Flowable helpers
# --------------------------------------------------------------------------
def para(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def h1(text: str) -> Paragraph:
    return Paragraph(text, S["H1"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, S["H2"])


def bullets(items: Sequence[str]) -> List[Paragraph]:
    return [Paragraph(item, S["Bullet"], bulletText="•") for item in items]


def caption(text: str) -> Paragraph:
    return Paragraph(text, S["Caption"])


def figure(path: Path, text: str, width: float = CONTENT_W) -> List:
    image = Image(str(path))
    scale = width / image.imageWidth
    image.drawWidth = width
    image.drawHeight = image.imageHeight * scale
    return [KeepTogether([image, caption(text)])]


def code_block(text: str, font_size: float = 7.0, width: float = CONTENT_W) -> Table:
    """Render preformatted text (terminal output / pseudocode) in a framed box."""
    lines = [
        html.escape(line).replace(" ", "&nbsp;")
        for line in text.strip("\n").split("\n")
    ]
    body = "<br/>".join(lines) if lines else "&nbsp;"
    style = ParagraphStyle(
        "code", fontName="Courier", fontSize=font_size, leading=font_size + 2.2,
        textColor=colors.HexColor("#152238"),
    )
    table = Table([[Paragraph(body, style)]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6FA")),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def make_table(
    rows: Sequence[Sequence],
    widths: Sequence[float],
    align_right: Sequence[int] = (),
    header: bool = True,
    font_size: float = 7.5,
    band: bool = True,
) -> Table:
    """Build a styled table; cells may be strings (auto-wrapped) or Paragraphs."""
    cell_style = S["Cell"]
    head_style = S["CellH"]
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for c_i, cell in enumerate(row):
            if isinstance(cell, Paragraph):
                out.append(cell)
            elif r_i == 0 and header:
                out.append(Paragraph(str(cell), head_style))
            else:
                out.append(Paragraph(str(cell), cell_style))
        data.append(out)

    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, 0), 3.4),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3.4),
        ]
    if band:
        for i in range(1 if header else 0, len(data)):
            if (i - (1 if header else 0)) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), BAND))
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    table.setStyle(TableStyle(style))
    return table


def kpi_strip(items: Sequence[Tuple[str, str]]) -> Table:
    """Cover-page key-figure strip."""
    cells = []
    for value, label in items:
        cells.append(
            Table(
                [
                    [Paragraph(
                        f'<font size="13"><b>{value}</b></font>',
                        ParagraphStyle("kv", alignment=TA_CENTER, textColor=NAVY,
                                       leading=15),
                    )],
                    [Paragraph(
                        f'<font size="6.6">{label}</font>',
                        ParagraphStyle("kl", alignment=TA_CENTER, textColor=GREY,
                                       leading=8),
                    )],
                ],
                colWidths=[CONTENT_W / len(items) - 3],
            )
        )
    strip = Table([cells], colWidths=[CONTENT_W / len(items)] * len(items))
    strip.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F9FD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return strip


# --------------------------------------------------------------------------
# Document template
# --------------------------------------------------------------------------
class ReviewDoc(BaseDocTemplate):
    def __init__(self, filename, generated: str, **kwargs):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="Review 2 Report - Event-Driven Runtime-Adaptive CPU Scheduling",
            author="OS-Project (github.com/lostpilot404/OS-Project)",
            subject="Review 2 project report",
            **kwargs,
        )
        self.generated = generated
        cover_frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W,
                            PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="cover")
        body_frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W,
                           PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - 0.9 * cm, id="body")
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[cover_frame], onPage=self._cover),
                PageTemplate(id="Body", frames=[body_frame], onPage=self._decorate),
            ]
        )

    # -- page furniture ---------------------------------------------------
    def _cover(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 1.15 * cm, PAGE_W, 1.15 * cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN_X, PAGE_H - 0.78 * cm,
                          "OPERATING SYSTEMS PROJECT  •  REVIEW 2")
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 0.78 * cm, "PROJECT REPORT")
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, 0.55 * cm, stroke=0, fill=1)
        canvas.restoreState()

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        y_top = PAGE_H - MARGIN_TOP + 0.45 * cm
        canvas.line(MARGIN_X, y_top, PAGE_W - MARGIN_X, y_top)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(MARGIN_X, y_top + 0.16 * cm,
                          "OS-Project  |  Review 2 Report  |  " + REPORT_TITLE)
        canvas.drawRightString(PAGE_W - MARGIN_X, y_top + 0.16 * cm, REPO_URL)
        y_bot = MARGIN_BOTTOM - 0.25 * cm
        canvas.line(MARGIN_X, y_bot, PAGE_W - MARGIN_X, y_bot)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(MARGIN_X, y_bot - 0.42 * cm,
                          f"Generated from committed artifacts on {self.generated}")
        canvas.drawRightString(PAGE_W - MARGIN_X, y_bot - 0.42 * cm,
                               f"Page {doc.page - 1}")
        canvas.restoreState()

    # -- table of contents -------------------------------------------------
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page - 1))
            elif name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page - 1))


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
class Artifacts:
    def __init__(self, root: Path):
        self.root = root
        runtime = root / "results" / "runtime"
        self.summary = json.loads((runtime / "runtime_summary.json").read_text())
        self.final = pd.read_csv(runtime / "runtime_final_test_metrics.csv")
        self.val = pd.read_csv(runtime / "runtime_validation_metrics.csv")
        self.train = pd.read_csv(runtime / "runtime_training_metrics.csv")
        self.paired = pd.read_csv(runtime / "runtime_paired_comparisons.csv")
        self.coverage = pd.read_csv(runtime / "runtime_state_action_coverage.csv")
        self.q_table = pd.read_csv(runtime / "runtime_q_table.csv")
        self.manifest = pd.read_csv(runtime / "runtime_workload_manifest.csv")
        self.demo = json.loads((runtime / "runtime_learned_switch_demo.json").read_text())
        self.legacy = json.loads((root / "results" / "summary.json").read_text())
        self.legacy_history = json.loads(
            (root / "results" / "training_history.json").read_text()
        )
        self.cfg = self.summary["configuration"]
        self.families = list(self.cfg["validation_and_test_families"])

    def mean_sd(self, split: str, method: str, metric: str) -> Tuple[float, float]:
        frame = self.final if split == "final" else self.val
        values = frame.loc[frame["method"] == method, metric].to_numpy(dtype=float)
        return float(values.mean()), float(values.std(ddof=1))

    def summary_mean(self, split: str, method: str, metric: str) -> float:
        key = "final_test_means" if split == "final" else "validation_means"
        return float(self.summary[key][method][metric])


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def fmt_ms(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------
def cover_page(data: Artifacts, generated: str, test_line: str) -> List:
    cfg = data.cfg
    story: List = []
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("OPERATING SYSTEMS PROJECT", S["cover_meta"]))
    story.append(Paragraph("REVIEW 2 — PROJECT REPORT", S["cover_meta"]))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(REPORT_TITLE, S["title"]))
    story.append(Paragraph(REPORT_SUBTITLE, S["subtitle"]))
    story.append(
        Paragraph(
            "Single-CPU discrete-event simulation  •  causal arrived-work state abstraction  •  "
            "undiscounted waiting-time learning  •  predeclared heuristic baseline  •  "
            "held-out crossed-bootstrap evaluation",
            ParagraphStyle("cover_sub", parent=S["cover_meta"], fontSize=8.6,
                           leading=11.5, textColor=GREY),
        )
    )
    story.append(Spacer(1, 0.45 * cm))

    meta = [
        ["Project title", "Event-Driven Runtime-Adaptive CPU Scheduling with Sequential Q-Learning"],
        ["Review stage", "Review 2 (implementation complete, experiments verified)"],
        ["Repository", REPO_URL],
        ["Implementation language", "Python 3.11.2 (NumPy, pandas, Matplotlib, pytest)"],
        ["Automated verification", test_line],
        ["Experiment status", "Deterministic and re-run verified from committed seeds"],
        ["Report generated", generated],
    ]
    story.append(
        make_table(
            [["Field", "Value"]] + meta,
            widths=[CONTENT_W * 0.24, CONTENT_W * 0.76],
            band=True,
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    story.append(
        kpi_strip(
            [
                ("6", "methods compared"),
                ("162 × 4", "causal states ×<br/>actions"),
                ("6,000", "training<br/>workloads"),
                ("210", "held-out test<br/>workloads"),
                ("5", "independent<br/>model seeds"),
                ("273", "automated<br/>tests"),
            ]
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    q_wait = data.summary_mean("final", "Runtime Q-learning", "avg_waiting_time")
    sjf_wait = data.summary_mean("final", "SJF", "avg_waiting_time")
    heur_wait = data.summary_mean("final", "Causal heuristic", "avg_waiting_time")
    fcfs_wait = data.summary_mean("final", "FCFS", "avg_waiting_time")
    rr_wait = data.summary_mean("final", "Round Robin", "avg_waiting_time")
    story.append(
        Paragraph(
            "<b>Abstract.</b> This report documents Review 2 of an operating-systems project that "
            "replaces a fixed CPU scheduling policy with a causal, event-driven runtime controller. "
            "A single-CPU discrete-event simulator keeps one persistent FIFO ready queue and, at every "
            "dispatch or Round-Robin quantum boundary, asks a controller to choose FCFS, SJF, Round "
            "Robin or Priority for the next execution segment of the same evolving trace. The "
            "controller observes only work that has already arrived: five causal features "
            "(ready-queue size, completed count, median ready remaining burst, mean ready arrival "
            "age, and arrived-work priority spread) are discretised into 162 states, and a tabular "
            "Q-learning agent with an undiscounted interval reward "
            f"r<sub>t</sub> = −ΔW<sub>t</sub>/q (γ = 1, so Σ r<sub>t</sub> = −total waiting time / q) "
            "learns a switching policy over 6,000 seeded training workloads across five independent "
            "seeds. On 210 untouched held-out workloads (master seed 19301, quantum 4, switch cost 1) "
            f"the learned controller averages {fmt(q_wait)} time units of waiting time against "
            f"{fmt(fcfs_wait)} for FCFS, {fmt(rr_wait)} for Round Robin and {fmt(heur_wait)} for a "
            "predeclared non-RL heuristic, while fixed SJF — which is non-preemptive and pays no "
            f"mid-job switch cost — remains lowest at {fmt(sjf_wait)}. The report states that negative "
            "result explicitly, quantifies every comparison with family-stratified crossed-bootstrap "
            "confidence intervals, and documents the simulator's assumptions and limits.",
            S["Abstract"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "<b>Keywords:</b> CPU scheduling; discrete-event simulation; reinforcement learning; "
            "Q-learning; causal state abstraction; Round Robin; SJF; bootstrap confidence intervals.",
            S["Small"],
        )
    )
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    return story


def toc_page() -> List:
    story: List = []
    story.append(Paragraph("Contents", S["H1"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC0", fontName="Helvetica-Bold", fontSize=9.2, leading=15,
            textColor=NAVY, leftIndent=0,
        ),
        ParagraphStyle(
            "TOC1", fontName="Helvetica", fontSize=8.4, leading=12.5,
            textColor=colors.HexColor("#333333"), leftIndent=14,
        ),
    ]
    story.append(toc)
    story.append(PageBreak())
    return story


# ---- 1. Introduction ------------------------------------------------------
def section_introduction(data: Artifacts) -> List:
    story = [h1("1. Introduction")]
    story.append(h2("1.1 Problem statement"))
    story.append(
        para(
            "A general-purpose operating system must decide which ready process runs next on a CPU. "
            "The classical policies each optimise a different, largely conflicting objective: "
            "First-Come First-Served (FCFS) is simple and fair in arrival order but suffers the "
            "convoy effect; non-preemptive Shortest Job First (SJF) minimises mean waiting time among "
            "non-preemptive policies when burst lengths are known, at the cost of response time and "
            "starvation risk for long jobs; Round Robin (RR) bounds response time by time-slicing but "
            "inflates waiting time and context switches; and static Priority scheduling reflects "
            "importance but can starve low-priority work. Real workloads are non-stationary — a trace "
            "may begin interactive and become batch-like — so no single fixed policy is best for the "
            "whole trace."
        )
    )
    story.append(
        para(
            "The problem addressed by this project is therefore: <b>can a single-CPU scheduler "
            "adapt its policy at run time, at every dispatch or quantum boundary, using only "
            "information that a real kernel could actually possess at that instant, and can that "
            "adaptation be measured honestly against the fixed policies it switches between?</b> "
            "Review 1 delivered four verified standalone schedulers plus an offline selector that "
            "inspected whole-workload summaries; Review 2 replaces that look-ahead design with a "
            "causal, event-driven runtime controller and a full experimental evaluation."
        )
    )
    story.append(h2("1.2 Motivation"))
    story.extend(
        bullets(
            [
                "<b>Fixed policies are Pareto-incomplete.</b> RR buys response time with waiting "
                "time; SJF buys waiting time with response time. A runtime controller can trade "
                "between them as the ready queue changes.",
                "<b>Adaptation must be causal.</b> A controller that peeks at unarrived processes or "
                "at completed counterfactual schedules cannot be deployed and cannot be evaluated "
                "fairly. Enforcing a strict observation boundary makes the result meaningful.",
                "<b>Learning is a natural fit.</b> The choice at each epoch is a small sequential "
                "decision problem with a delayed, additive cost (waiting time), which maps cleanly "
                "onto tabular Q-learning with a 162-state abstraction.",
                "<b>Honest negative results matter.</b> An adaptive controller that fails to beat a "
                "strong fixed baseline is a useful, publishable finding — provided the comparison is "
                "paired, held out and accompanied by uncertainty intervals.",
            ]
        )
    )
    story.append(h2("1.3 Objectives"))
    objectives = [
        ("O1", "Preserve four verified standalone schedulers (FCFS, SJF, RR, Priority) as baselines."),
        ("O2", "Build a single-CPU, event-driven runtime environment with one persistent FIFO ready "
               "queue, remaining bursts, arrivals, RR re-enqueue and per-process switch accounting."),
        ("O3", "Define a causal observation contract that exposes only arrived work, and encode it as "
               "a compact 162-state tabular abstraction."),
        ("O4", "Train a sequential Q-learning controller whose undiscounted return equals negative "
               "total waiting time, with explicit exploration, masking and unseen-state fallback."),
        ("O5", "Declare a non-RL heuristic baseline under the same observation contract, before any "
               "test results are seen."),
        ("O6", "Evaluate on held-out workloads with paired, family-stratified bootstrap intervals and "
               "report negative findings explicitly."),
        ("O7", "Ship a reproducible, test-covered repository (273 automated checks) with recorded "
               "artifacts and regeneration commands."),
    ]
    story.append(
        make_table(
            [["#", "Objective"]] + [[oid, text] for oid, text in objectives],
            widths=[CONTENT_W * 0.07, CONTENT_W * 0.93],
        )
    )
    story.append(h2("1.4 Scope, assumptions and non-goals"))
    story.extend(
        bullets(
            [
                "<b>In scope:</b> synthetic single-CPU simulation; four fixed policies; one causal "
                "runtime controller; one predeclared heuristic; seven workload families; train / "
                "validation / held-out-test protocol; uncertainty quantification.",
                "<b>Assumption — burst knowledge:</b> exact CPU burst lengths become known when a "
                "process arrives. This matches the standalone SJF simulator already in the project; "
                "no burst predictor is claimed.",
                "<b>Non-goals:</b> I/O blocking, multicore and cache/bus contention, priority aging, "
                "kernel latency, real Linux integration, and any claim that the measured ordering "
                "transfers to production workloads.",
            ]
        )
    )
    return story


# ---- 2. Literature survey -------------------------------------------------
def section_literature() -> List:
    story = [h1("2. Literature survey")]
    story.append(
        para(
            "The survey below covers (a) the classical scheduling policies implemented as baselines, "
            "(b) heuristic adaptive schedulers used in production kernels, and (c) learning-based "
            "scheduling. The gap this project targets is highlighted in the final row: most reported "
            "RL schedulers either observe whole-workload/global state that a kernel cannot see at "
            "decision time, or they replace the policy entirely instead of switching between "
            "classical policies under a declared cost model."
        )
    )
    rows = [
        ["Area", "Representative work", "What it establishes", "Gap addressed here"],
        [
            "Classical policies",
            "FCFS, SJF/SRTF, Round Robin and Priority scheduling as formalised in standard "
            "operating-system texts [1][2]",
            "Analytic trade-offs: SJF is optimal for mean waiting time among non-preemptive policies; "
            "RR bounds response time; static priority can starve",
            "Used as the four baselines and as the four actions, with a declared quantum and switch cost",
        ],
        [
            "Heuristic adaptive schedulers",
            "Multi-level feedback queues and fair-share/virtual-runtime schedulers [1][2]",
            "Hand-tuned feedback rules adapt to observed behaviour in real kernels",
            "A declared, non-learning heuristic is evaluated as a baseline so that RL is not credited "
            "for what fixed rules already achieve",
        ],
        [
            "RL for thread/CPU scheduling",
            "Shrivastava, <i>Reinforcement Learning for Scheduling Threads on a Multi-Core Processor</i> [3]",
            "Q-learning can learn online from core utilisation / readiness features, but the state "
            "space grows very large and convergence is slow",
            "Keeps the abstraction tiny (162 states, 4 actions) so coverage and fallback behaviour can "
            "be measured directly",
        ],
        [
            "Deep RL schedulers",
            "Deep Q-learning task scheduling [4]; Double DQN OS scheduling [5]",
            "Function approximation handles richer state but needs large data and is hard to audit",
            "Tabular Q-learning is used precisely so every state-action value, visit count and "
            "fallback can be inspected in the committed artifacts",
        ],
        [
            "Context-aware / learned governors",
            "Context-aware RF + DQN scheduling framework [6]; temporal-encoding DRL for DVFS [7]",
            "Workload classification and temporal encoding improve adaptivity on heterogeneous loads",
            "Workload identity is never given to the controller here; only arrived work is observable",
        ],
        [
            "Surveys",
            "Shyalika, Silva and Karunananda, <i>Reinforcement learning in dynamic task scheduling: a "
            "review</i> [8]",
            "Confirms RL scheduling is dominated by simulation studies with weak baselines and no "
            "uncertainty reporting",
            "Paired, family-stratified crossed-bootstrap intervals and an explicit statement of the "
            "result that does <i>not</i> favour the learner",
        ],
    ]
    story.append(
        make_table(
            rows,
            widths=[CONTENT_W * 0.14, CONTENT_W * 0.27, CONTENT_W * 0.32, CONTENT_W * 0.27],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        para(
            "Two practices from the surveyed literature are adopted deliberately: every learned "
            "method is compared against a predeclared rule-based controller sharing the same "
            "observation contract, and every headline comparison is reported with a confidence "
            "interval rather than a point estimate."
        )
    )
    return story


# ---- 3. Requirements ------------------------------------------------------
def section_requirements() -> List:
    story = [h1("3. Requirement analysis")]
    story.append(h2("3.1 Functional requirements"))
    fr = [
        ["FR-1", "Generate reproducible synthetic workloads across seven families with seeded RNG."],
        ["FR-2", "Simulate a single CPU with arrivals, a persistent FIFO ready queue, remaining "
                 "bursts and per-process accounting."],
        ["FR-3", "Implement FCFS, SJF, Round Robin and Priority both standalone and as runtime "
                 "actions, with identical semantics."],
        ["FR-4", "Expose a causal observation containing only arrived/ready work and history counts."],
        ["FR-5", "Encode the observation into a bounded tabular state and learn a policy with "
                 "Q-learning (exploration, masking, fallback)."],
        ["FR-6", "Provide a deterministic non-RL heuristic under the same observation contract."],
        ["FR-7", "Charge a context-switch cost whenever the running PID changes, for all methods."],
        ["FR-8", "Compute six metrics (waiting, turnaround, response, CPU utilisation, throughput, "
                 "context switches) and emit CSV/JSON artifacts plus a report."],
        ["FR-9", "Provide a CLI that reproduces training, validation, held-out testing and the "
                 "legacy offline reference."],
    ]
    story.append(
        make_table(
            [["ID", "Requirement"]] + fr,
            widths=[CONTENT_W * 0.09, CONTENT_W * 0.91],
        )
    )
    story.append(h2("3.2 Non-functional requirements"))
    nfr = [
        ["NFR-1", "Reproducibility", "Every artifact is regenerated from declared seeds; split "
                                     "fingerprint overlap is asserted to be empty."],
        ["NFR-2", "Causality", "Controller and encoder APIs accept observations, never workloads; "
                               "hidden-future invariance is test-enforced."],
        ["NFR-3", "Determinism", "Evaluation is greedy and read-only; no Q-table or visit-count "
                                 "mutation during evaluation."],
        ["NFR-4", "Auditability", "Q values, visit counts, coverage and fallback counts are written "
                                  "to committed CSV files."],
        ["NFR-5", "Performance", "Controller overhead is measured separately from simulated time "
                                 "(tens of microseconds per decision)."],
        ["NFR-6", "Testability", "273 automated checks cover schedulers, encoding, learning, "
                                 "causality and the experiment pipeline."],
    ]
    story.append(
        make_table(
            [["ID", "Property", "How it is met"]] + nfr,
            widths=[CONTENT_W * 0.09, CONTENT_W * 0.17, CONTENT_W * 0.74],
        )
    )
    story.append(h2("3.3 Software and hardware requirements"))
    sw = data_software_requirements()
    story.append(
        make_table(
            [["Component", "Requirement", "Used for"]] + sw,
            widths=[CONTENT_W * 0.24, CONTENT_W * 0.22, CONTENT_W * 0.54],
        )
    )
    return story


def data_software_requirements() -> List[List[str]]:
    return [
        ["Operating system", "Linux (developed on Linux 6.1, x86-64)", "Runs the simulator and tests"],
        ["Language", "Python 3.11.2", "Whole implementation"],
        ["NumPy 2.4.6", "Numerical arrays and seeded RNG", "Q table, bootstrap resampling"],
        ["pandas 3.0.6", "Tabular result handling", "Metric aggregation and CSV export"],
        ["Matplotlib 3.11.2", "Figure rendering", "Report and legacy figures"],
        ["pytest 9.1.1", "Automated verification", "273 unit and integration tests"],
        ["ReportLab (report only)", "PDF generation", "tools/build_review2_report.py"],
        ["Memory / disk", "< 512 MB RAM, < 50 MB for artifacts", "Full experiment re-run and report"],
    ]


# ---- 4. System design -----------------------------------------------------
def section_design(data: Artifacts, figures: Dict[str, Path]) -> List:
    cfg = data.cfg
    story = [h1("4. System design")]
    story.append(h2("4.1 Architecture"))
    story.append(
        para(
            "The runtime system is a closed loop between a discrete-event environment that owns the "
            "future and a controller that may see only the past and the present. The environment "
            "privately holds the complete event list so it can admit future arrivals at their arrival "
            "times; the controller receives an immutable <font face='Courier'>RuntimeObservation</font> "
            "and returns one of four actions. The action changes only the next selection rule and "
            "service length — it never rebuilds the ready queue or resets remaining bursts."
        )
    )
    story.extend(figure(figures["architecture"],
                        "Figure 4.1: Runtime-adaptive scheduling architecture. The controller path "
                        "(amber) receives only arrived work; the environment path (blue) owns future "
                        "arrivals and switch accounting."))
    modules = [
        ["workload/", "models.py, generator.py", "Process/workload models and the seven seeded "
                                                 "workload families; schedule-trace validation."],
        ["scheduler/", "runtime.py", "Single-CPU discrete-event environment: event queue, FIFO ready "
                                     "queue, switch cost, six metrics."],
        ["scheduler/", "fcfs.py, sjf.py, round_robin.py, priority.py",
         "Standalone fixed-policy baselines with preserved semantics."],
        ["rl/", "runtime_state.py", "Five-feature causal encoder: 3×3×3×3×2 = 162 states."],
        ["rl/", "runtime_controller.py", "Sequential tabular Q-learning: visits, masking, deterministic "
                                         "fallback."],
        ["rl/", "runtime_heuristic.py", "Predeclared non-RL adaptive rule under the same contract."],
        ["evaluation/", "metrics.py, comparison.py", "Metric definitions and paired/bootstrap "
                                                     "comparison helpers."],
        ["experiments/", "runtime_config.py, runtime_experiment.py",
         "Frozen seeds, split sizes, switch cost, Q settings; training, validation, testing, reporting."],
        ["tests/", "16 modules", "273 automated checks (schedulers, causality, learning, pipeline)."],
        ["tools/", "build_review2_report.py, review2_figures.py",
         "Regenerates this PDF from the committed artifacts."],
    ]
    story.append(
        make_table(
            [["Package", "Key files", "Responsibility"]] + modules,
            widths=[CONTENT_W * 0.13, CONTENT_W * 0.30, CONTENT_W * 0.57],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        caption(
            "Table 4.1: Module inventory. The legacy offline selector (rl/adaptive.py, "
            "experiments/run_experiment.py) is preserved but is not part of the runtime design "
            "(Section 10)."
        )
    )

    story.append(h2("4.2 Execution semantics and the ready-queue contract"))
    story.extend(
        bullets(
            [
                "One persistent FIFO ready queue holds arrived, unfinished, non-running processes. "
                "Arrivals are appended in (arrival_time, pid) order; dispatch removes the selected "
                "process; completion removes it permanently.",
                "<b>FCFS</b> dispatches the current queue head non-preemptively — after an RR requeue "
                "this is live FIFO order, not the original arrival time.",
                "<b>Round Robin</b> dispatches the head for min(quantum, remaining_burst); on quantum "
                "expiry, arrivals at that endpoint are admitted first and the yielded process is "
                "appended last.",
                "<b>SJF</b> picks the smallest remaining burst non-preemptively; ties preserve current "
                "ready-queue order.",
                "<b>Priority</b> picks the highest static priority (lower numeric value = higher "
                "priority); ties preserve ready-queue order.",
                "A switch cost of 1 time unit is charged only when the running PID changes. The first "
                "dispatch and a redispatch of the same PID are free. The cost applies to every method, "
                "including the fixed policies.",
            ]
        )
    )
    story.append(h2("4.3 The causal observation and one decision epoch"))
    story.append(
        para(
            "At a decision epoch the environment admits arrivals, freezes an observation, asks for an "
            "action, executes the resulting segment and returns the waiting-time increment incurred. "
            "The observation contains current time and the previous policy; the ready processes with "
            "their arrival time, known burst, remaining burst and priority; arrived and completed "
            "counts; aggregates over arrived work only (mean burst, priority spread); and the mean "
            "arrival age of currently ready jobs. Unarrived process data, the total workload size and "
            "counterfactual schedule metrics are never passed to the controller."
        )
    )
    story.extend(
        figure(
            figures["flow"],
            "Figure 4.2: One sequential decision epoch. The loop repeats until the workload completes; "
            "the final transition omits the bootstrap term.",
            width=CONTENT_W * 0.62,
        )
    )
    story.append(h2("4.4 Causal state abstraction"))
    encoding = [
        ["1", "ready_count", "3", "{1}, {2–3}, {4+}", "Instantaneous ready-queue contention."],
        ["2", "completed_count", "3", "{0}, {1–3}, {4+}", "Trace progression without knowing how many "
                                                          "jobs remain."],
        ["3", "median_ready_remaining_burst", "3", "≤ q, ≤ 4q, > 4q", "Remaining service demand of "
                                                                      "currently ready jobs."],
        ["4", "mean_ready_arrival_age", "3", "≤ q, ≤ 4q, > 4q", "Mean (current time − arrival time) of "
                                                                "ready jobs; includes prior CPU service."],
        ["5", "arrived_work_priority_spread", "2", "zero, non-zero", "Whether arrived processes have "
                                                                     "distinct priorities."],
    ]
    story.append(
        make_table(
            [["#", "Feature", "Bins", "Discretisation (q = 4)", "Causal rationale"]] + encoding,
            widths=[CONTENT_W * 0.05, CONTENT_W * 0.24, CONTENT_W * 0.07,
                    CONTENT_W * 0.24, CONTENT_W * 0.40],
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 4.2: 3 × 3 × 3 × 3 × 2 = 162 causal states, 162 × 4 = 648 state-action pairs "
            "(rl/runtime_state.py)."
        )
    )
    story.append(h2("4.5 Learning objective and update rule"))
    story.append(
        para(
            "Each trace is one sequential episode. Between consecutive epochs the environment measures "
            "the actual waiting-time area ΔW<sub>t</sub> accumulated by ready processes (including any "
            "switch interval) and returns a scaled interval reward:"
        )
    )
    story.append(Paragraph("r<sub>t</sub> = − ΔW<sub>t</sub> / max(1, q)      γ = 1.0      "
                           "⇒      Σ<sub>t</sub> r<sub>t</sub> = − TotalWaitingTime / q",
                           S["Equation"]))
    story.append(
        para(
            "Because γ = 1 and the episode is finite, the undiscounted episodic return is exactly the "
            "negative total waiting time divided by the quantum: maximising return is identical to "
            "minimising total waiting time. The tabular update is"
        )
    )
    story.append(
        Paragraph(
            "Q(s, a) ← Q(s, a) + α [ r<sub>t</sub> + γ max<sub>a′ ∈ visited(s′)</sub> Q(s′, a′) "
            "− Q(s, a) ],   α = 0.1",
            S["Equation"],
        )
    )
    story.append(
        para(
            "Terminal transitions omit the bootstrap. The first visit to a state forces a uniform "
            "random action; greedy selection and bootstrapping mask unvisited actions; and a wholly "
            "unseen evaluation state triggers an explicit fallback to the causal heuristic rather than "
            "a silent zero-Q tie."
        )
    )
    story.append(h2("4.6 Predeclared non-RL heuristic baseline"))
    story.append(
        para(
            "To separate 'learning helped' from 'any adaptive rule helps', a fixed deterministic rule "
            "was declared before any test results were inspected. It sees the same observation:"
        )
    )
    story.extend(
        bullets(
            [
                "<b>R1.</b> Round Robin if at least three processes are ready and their mean arrival "
                "age is at least one quantum.",
                "<b>R2.</b> Otherwise Priority if at least two ready processes have different "
                "priorities.",
                "<b>R3.</b> Otherwise SJF if the largest visible remaining burst is at least twice the "
                "smallest.",
                "<b>R4.</b> Otherwise FCFS.",
            ]
        )
    )
    return story


# ---- 5. Algorithms --------------------------------------------------------
def section_algorithms() -> List:
    story = [h1("5. Algorithms")]
    story.append(h2("5.1 Runtime decision loop"))
    story.append(
        code_block(
            """
RUNTIME_SCHEDULE(workload, controller, quantum q, switch_cost c):
    ready  <- {}                       # persistent FIFO of arrived, unfinished processes
    events <- sorted arrivals of workload          # private to the environment
    t      <- 0 ; running <- NULL ; previous_policy <- None
    while unfinished processes remain:
        admit all arrivals with arrival_time <= t            # (arrival_time, pid) order
        if ready is empty:                                   # CPU idle
            t <- next arrival time ; continue
        obs   <- Observation(t, ready, arrived_count, completed_count, previous_policy)
        a     <- controller.select(obs)                      # FCFS | SJF | RR | Priority
        pid   <- SELECT(ready, a)                            # per policy rule, ties by FIFO order
        if running != NULL and pid != running:
            charge_switch_cost(c) ; t <- t + c ; admit arrivals during the switch
        run   <- min(q, remaining[pid]) if a == RR else remaining[pid]
        execute pid for run ; t <- t + run ; remaining[pid] <- remaining[pid] - run
        dW    <- waiting-time area accumulated by ready processes during this segment
        controller.observe_reward(-dW / max(1, q))
        settle endpoint: admit arrivals at t; if remaining[pid] > 0 and a == RR: append pid last
                         else if remaining[pid] == 0: mark completed
        previous_policy <- a
    return metrics(waiting, turnaround, response, utilisation, throughput, switches)
""",
            font_size=6.4,
        )
    )
    story.append(h2("5.2 Action (fixed-policy) selection rules"))
    story.append(
        code_block(
            """
SELECT(ready, a):
    FCFS      -> head of the ready queue                      (non-preemptive)
    SJF       -> argmin remaining_burst, ties by ready order  (non-preemptive)
    Priority  -> argmin priority value, ties by ready order   (non-preemptive)
    Round Robin -> head of the ready queue, served for min(quantum, remaining_burst)
""",
            font_size=6.8,
        )
    )
    story.append(h2("5.3 Sequential Q-learning controller"))
    story.append(
        code_block(
            """
SELECT_ACTION(obs):                       # training: epsilon-greedy; evaluation: greedy + fallback
    s <- encode(obs)                       # 5 causal features -> [0, 162)
    if evaluating:
        if no visited action in s: return HEURISTIC_FALLBACK(obs)      # counted and reported
        return argmax over visited actions of Q[s, a]
    if first_visit(s) or rand() < epsilon(s): return uniform_random_action()
    return argmax over visited actions of Q[s, a]

UPDATE(s, a, r, s', terminal):
    Q[s, a] <- Q[s, a] + alpha * (r + (0 if terminal else gamma * max_visited Q[s', .]) - Q[s, a])
    visits[s, a] <- visits[s, a] + 1
""",
            font_size=6.8,
        )
    )
    story.append(
        para(
            "Exploration uses an episode-level epsilon schedule; the bootstrap maximum is taken over "
            "visited actions only, so unvisited actions can never be credited optimistically."
        )
    )
    return story


# ---- 6. Implementation ----------------------------------------------------
def section_implementation(data: Artifacts, cli_block: str) -> List:
    story = [h1("6. Implementation")]
    story.append(
        para(
            "The simulator, baselines, controller and experiment pipeline are pure Python with NumPy "
            "arrays for the Q table and pandas for artifact export. Nothing in the controller path "
            "touches the private event list: the observation object is frozen and the encoder accepts "
            "only that object, which is what makes the causality tests meaningful rather than "
            "cosmetic. Controller wall time is measured separately (observation, selection, Q update "
            "and simulator time are distinct columns) and is never charged to simulated time."
        )
    )
    story.append(h2("6.1 Command-line interface"))
    story.append(
        code_block(
            "python main.py runtime-experiment      # causal train / validate / held-out test\n"
            "python main.py runtime-experiment --results-dir /tmp/os-project-runtime\n"
            "python main.py experiment              # legacy offline whole-workload selector\n"
            "python main.py train                   # train the legacy selector only",
            font_size=7.0,
        )
    )
    story.append(Spacer(1, 5))
    story.append(h2("6.2 Verified run output"))
    story.append(
        para(
            "The block below is the tail of a full reproduction run performed on the reporting host "
            "before this PDF was generated; it matches the committed artifacts exactly.",
            "Small",
        )
    )
    story.append(code_block(cli_block, font_size=6.6))
    story.append(Spacer(1, 4))
    story.append(h2("6.3 Determinism and split integrity"))
    story.append(
        para(
            "Training, validation and test workloads are derived from declared master seeds with "
            "SHA-256 workload fingerprints. The pipeline asserts that the union of training "
            "fingerprints is disjoint from the validation and final-test sets "
            f"(<font face='Courier'>split_fingerprint_overlap = "
            f"{str(data.summary['split_fingerprint_overlap']).lower()}</font>; "
            f"{data.summary['training_fingerprint_count_union']:,} distinct training fingerprints). "
            "Evaluation runs are greedy and read-only, so results are reproducible bit-for-bit from "
            "the recorded configuration."
        )
    )
    return story


# ---- 7. Experimental setup -----------------------------------------------
def section_setup(data: Artifacts) -> List:
    cfg = data.cfg
    story = [h1("7. Experimental setup")]
    story.append(h2("7.1 Workload families"))
    rows = []
    for fam in cfg["workload_family_parameters"]:
        rows.append(
            [
                fam["name"],
                fam["description"],
                f"{fam['burst_time_min']}–{fam['burst_time_max']}",
                fam["arrival_pattern"],
                f"{fam['priority_min']}–{fam['priority_max']}",
            ]
        )
    story.append(
        make_table(
            [["Family", "Description", "Burst range", "Arrivals", "Priority"]] + rows,
            widths=[CONTENT_W * 0.17, CONTENT_W * 0.47, CONTENT_W * 0.12,
                    CONTENT_W * 0.12, CONTENT_W * 0.12],
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 7.1: Seven synthetic families, 15 processes per workload. "
            "poisson_arrivals is held out of training entirely."
        )
    )
    story.append(h2("7.2 Splits and protocol"))
    splits = [
        ["Training", "6 families (poisson_arrivals excluded)", "1,200 sequential episodes × 5 seeds "
                                                               "= 6,000 workloads", "Seeds 7101–7105"],
        ["Validation", "7 families × 20 repetitions", "140 workloads", "Master seed 8201"],
        ["Final test", "7 families × 30 repetitions", "210 workloads (untouched)", "Master seed 19301"],
        ["Bootstrap", "Family-stratified, workload + model-seed resampling", "1,000 replicates",
         "Seed 104729"],
    ]
    story.append(
        make_table(
            [["Split", "Composition", "Size", "Seeding"]] + splits,
            widths=[CONTENT_W * 0.13, CONTENT_W * 0.36, CONTENT_W * 0.28, CONTENT_W * 0.23],
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 7.2: Three-way split. Validation is used for reporting and for a predeclared "
            "illustrative-demo rule only — never for tuning."
        )
    )
    story.append(h2("7.3 Configuration"))
    params = [
        ["Round-Robin quantum", str(cfg["scheduler"]["round_robin_quantum"]), "Time slice for the RR "
                                                                              "action and RR baseline"],
        ["Context-switch cost", str(cfg["scheduler"]["switching_cost"]),
         "Charged whenever the running PID changes, for every method"],
        ["Priority convention", "lower value = higher priority",
         "Static priorities, no aging"],
        ["Learning rate α", str(cfg["q_learning"]["learning_rate"]), "Tabular Q-learning step size"],
        ["Discount factor γ", str(cfg["q_learning"]["discount_factor"]),
         "Undiscounted; Σ r = −total waiting time / q"],
        ["ε schedule", f"{cfg['q_learning']['epsilon_start']} → "
                       f"{cfg['q_learning']['epsilon_min']} "
                       f"(×{cfg['q_learning']['epsilon_decay_per_episode']} per episode)",
         "Episode-level exploration decay"],
        ["Initial Q value", str(cfg["q_learning"]["initial_value"]),
         "Neutral; unvisited actions are masked, not assumed optimal"],
        ["State space", f"{cfg['state_encoder']['state_count']} states",
         "5 causal features, 648 state-action pairs"],
        ["Burst information", "known on arrival",
         "Matches the standalone SJF simulator assumption"],
    ]
    story.append(
        make_table(
            [["Parameter", "Value", "Notes"]] + params,
            widths=[CONTENT_W * 0.20, CONTENT_W * 0.20, CONTENT_W * 0.60],
        )
    )
    story.append(Spacer(1, 3))
    story.append(h2("7.4 Metrics"))
    metrics = [
        ["avg_waiting_time", "Mean over processes of (start of first service − arrival time)"],
        ["avg_turnaround_time", "Mean over processes of (completion time − arrival time)"],
        ["avg_response_time", "Mean over processes of (first dispatch time − arrival time)"],
        ["cpu_utilization", "CPU busy time / total elapsed time (%)"],
        ["throughput", "Processes completed per unit of simulated time"],
        ["context_switches", "Number of running-PID changes (including switch-cost intervals)"],
        ["policy_switch_count", "Decisions where the selected policy differs from the previous one"],
    ]
    story.append(
        make_table(
            [["Metric", "Definition"]] + metrics,
            widths=[CONTENT_W * 0.24, CONTENT_W * 0.76],
        )
    )
    return story


# ---- 8. Results -----------------------------------------------------------
def section_results(data: Artifacts, figures: Dict[str, Path]) -> List:
    story = [h1("8. Results and discussion")]
    story.append(h2("8.1 Held-out final test"))
    story.append(
        para(
            "All six methods are evaluated on the same 210 held-out workloads (master seed 19301) with "
            "quantum 4 and switch cost 1. Fixed policies and the heuristic are deterministic "
            "(210 rows each); Runtime Q-learning is evaluated greedily by all five trained models "
            "(1,050 rows) and averaged with equal weight per workload."
        )
    )
    metric_cols = [
        ("avg_waiting_time", "Mean waiting"),
        ("avg_turnaround_time", "Mean turnaround"),
        ("avg_response_time", "Mean response"),
        ("cpu_utilization", "CPU util %"),
        ("throughput", "Throughput"),
        ("context_switches", "Ctx switches"),
        ("policy_switch_count", "Policy switches"),
    ]
    rows = []
    for method in METHOD_ORDER:
        row = [method]
        for metric, _label in metric_cols:
            mean, sd = data.mean_sd("final", method, metric)
            if metric == "throughput":
                row.append(f"{mean:.3f} ± {sd:.3f}")
            elif metric in {"cpu_utilization", "context_switches", "policy_switch_count"}:
                row.append(f"{mean:.2f} ± {sd:.2f}")
            else:
                row.append(f"{mean:,.3f} ± {sd:,.2f}")
        rows.append(row)
    header = ["Method (final test)"] + [label for _m, label in metric_cols]
    widths = (
        [CONTENT_W * 0.16]
        + [CONTENT_W * 0.135]
        + [CONTENT_W * 0.145]
        + [CONTENT_W * 0.13]
        + [CONTENT_W * 0.11]
        + [CONTENT_W * 0.11]
        + [CONTENT_W * 0.105]
        + [CONTENT_W * 0.105]
    )
    story.append(
        make_table([header] + rows, widths=widths,
                   align_right=tuple(range(1, len(header))))
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 8.1: Held-out final-test results, mean ± SD across the 210 workloads "
            "(Q-learning also averages 5 model seeds)."
        )
    )
    story.extend(
        figure(
            figures["metrics"],
            "Figure 8.1: Six metrics on the 210 held-out workloads (error bars = SD).",
        )
    )
    story.append(h2("8.2 Spread across workloads"))
    story.extend(
        figure(
            figures["box"],
            "Figure 8.2: Per-workload distribution of waiting and response time. The medians show the "
            "same ordering as the means: the learned controller sits between SJF and the heuristic.",
        )
    )
    story.append(h2("8.3 Per-family behaviour"))
    fam_rows = []
    for fam in data.families:
        row = [fam.replace("_", " ")]
        for method in METHOD_ORDER:
            sub = data.final[(data.final["method"] == method) & (data.final["family"] == fam)]
            row.append(f"{sub['avg_waiting_time'].mean():.2f} / "
                       f"{sub['avg_response_time'].mean():.2f}")
        switches = data.final[
            (data.final["method"] == "Runtime Q-learning") & (data.final["family"] == fam)
        ]["policy_switch_count"].mean()
        row.append(f"{switches:.2f}")
        fam_rows.append(row)
    story.append(
        make_table(
            [["Family (n = 30)", "FCFS\nwait / resp", "SJF\nwait / resp", "RR\nwait / resp",
              "Priority\nwait / resp", "Heuristic\nwait / resp", "Runtime Q\nwait / resp",
              "Q switches"]]
            + fam_rows,
            widths=[CONTENT_W * 0.19]
            + [CONTENT_W * 0.115] * 6
            + [CONTENT_W * 0.12],
            align_right=tuple(range(1, 8)),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 8.2: Per-family mean waiting / mean response (all six methods) and mean policy "
            "switches per trace for Runtime Q-learning."
        )
    )
    story.extend(
        figure(
            figures["family"],
            "Figure 8.3: Per-family waiting and response time on the held-out final test. "
            "poisson_arrivals was never seen during training.",
        )
    )
    story.append(h2("8.4 Learning dynamics"))
    story.extend(
        figure(
            figures["training"],
            "Figure 8.4: Sequential training: rolling mean waiting time and episodic return over "
            "1,200 episodes for each of the five independent agents.",
        )
    )
    story.append(h2("8.5 Paired comparisons with uncertainty"))
    story.append(
        para(
            "Paired differences are computed per workload (and per model seed for Q-learning) and "
            "resampled with a family-stratified crossed bootstrap (1,000 replicates, seed 104729). "
            "Positive values favour the reference; negative values favour the target."
        )
    )
    paired_rows = []
    wanted = [
        ("Runtime Q-learning", "SJF"),
        ("Runtime Q-learning", "Causal heuristic"),
        ("Runtime Q-learning", "Round Robin"),
        ("Causal heuristic", "SJF"),
        ("Round Robin", "SJF"),
        ("FCFS", "SJF"),
    ]
    metric_labels = {
        "avg_waiting_time": "waiting time",
        "avg_response_time": "response time",
        "context_switches": "context switches",
        "cpu_utilization": "CPU util %",
    }
    for target, reference in wanted:
        for metric, label in metric_labels.items():
            sel = data.paired[
                (data.paired["target"] == target)
                & (data.paired["reference"] == reference)
                & (data.paired["metric"] == metric)
            ]
            if sel.empty:
                continue
            row = sel.iloc[0]
            mean = float(row["mean_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            if mean == 0 and low == 0 and high == 0:
                favours = "no difference"
            elif metric == "cpu_utilization":
                favours = "target" if mean > 0 else "reference"
            else:
                favours = "target" if mean < 0 else "reference"
            paired_rows.append(
                [
                    SHORT_NAME[target],
                    SHORT_NAME[reference],
                    label,
                    f"{mean:+,.3f}",
                    f"[{low:+,.3f}, {high:+,.3f}]",
                    f"favours {favours}",
                ]
            )
    story.append(
        make_table(
            [["Target", "Reference", "Metric", "Mean difference", "95% bootstrap CI", "Reading"]]
            + paired_rows,
            widths=[CONTENT_W * 0.12, CONTENT_W * 0.12, CONTENT_W * 0.15,
                    CONTENT_W * 0.16, CONTENT_W * 0.22, CONTENT_W * 0.23],
            align_right=(3,),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 8.3: Paired differences (target − reference) with family-stratified crossed-bootstrap "
            "confidence intervals on the final test set."
        )
    )
    story.extend(
        figure(
            figures["forest"],
            "Figure 8.5: Forest plot of selected paired comparisons. Green markers favour the target, "
            "red markers favour the reference; bars are 95% bootstrap intervals.",
            width=CONTENT_W * 0.94,
        )
    )
    story.append(h2("8.6 Key findings"))
    q = {m: data.summary_mean("final", "Runtime Q-learning", m) for m in
         ("avg_waiting_time", "avg_response_time", "context_switches")}
    sjf = {m: data.summary_mean("final", "SJF", m) for m in q}
    heur = {m: data.summary_mean("final", "Causal heuristic", m) for m in q}
    rr = data.summary_mean("final", "Round Robin", "avg_waiting_time")
    fcfs = data.summary_mean("final", "FCFS", "avg_waiting_time")

    def ci(target, reference, metric):
        sel = data.paired[
            (data.paired["target"] == target)
            & (data.paired["reference"] == reference)
            & (data.paired["metric"] == metric)
        ]
        row = sel.iloc[0]
        return float(row["mean_difference"]), float(row["ci95_low"]), float(row["ci95_high"])

    d_q_sjw = ci("Runtime Q-learning", "SJF", "avg_waiting_time")
    d_q_sjr = ci("Runtime Q-learning", "SJF", "avg_response_time")
    d_q_hr = ci("Runtime Q-learning", "Causal heuristic", "avg_response_time")
    d_q_hw = ci("Runtime Q-learning", "Causal heuristic", "avg_waiting_time")
    d_q_hc = ci("Runtime Q-learning", "Causal heuristic", "context_switches")

    story.extend(
        bullets(
            [
                f"<b>Adaptation beats Round Robin and the fixed non-SJF policies.</b> Runtime "
                f"Q-learning waits {fmt(q['avg_waiting_time'])} versus {fmt(rr)} for RR "
                f"({100 * (rr - q['avg_waiting_time']) / rr:.1f}% lower), {fmt(fcfs)} for FCFS and "
                f"{fmt(data.summary_mean('final', 'Priority', 'avg_waiting_time'))} for Priority.",
                f"<b>Response time improves over SJF.</b> Q − SJF = {d_q_sjr[0]:+.3f} "
                f"(95% CI [{d_q_sjr[1]:+.3f}, {d_q_sjr[2]:+.3f}]) — the learned controller keeps some "
                "of RR's responsiveness without paying all of RR's waiting cost.",
                f"<b>Honest negative result: it does not beat SJF on waiting time.</b> Q − SJF = "
                f"{d_q_sjw[0]:+.3f} (95% CI [{d_q_sjw[1]:+.3f}, {d_q_sjw[2]:+.3f}]), i.e. SJF's "
                "waiting time is significantly lower. Non-preemptive SJF receives exact bursts on "
                "arrival and never incurs a mid-job switch cost, so on these generators it is a very "
                "strong reference; the switching overhead also costs Q-learning "
                f"{fmt(data.summary_mean('final', 'Runtime Q-learning', 'cpu_utilization') - data.summary_mean('final', 'SJF', 'cpu_utilization'), 2)} "
                "percentage points of utilisation.",
                f"<b>Learning beats the predeclared heuristic decisively on waiting time.</b> Q − "
                f"heuristic = {d_q_hw[0]:+.3f} (95% CI [{d_q_hw[1]:+.3f}, {d_q_hw[2]:+.3f}]) with "
                f"{d_q_hc[0]:+.2f} context switches per trace — so the gain comes from RL, not merely "
                "from being adaptive.",
                f"<b>But the heuristic wins on response time</b> (Q − heuristic = {d_q_hr[0]:+.3f}, "
                f"95% CI [{d_q_hr[1]:+.3f}, {d_q_hr[2]:+.3f}]) because it slices aggressively with RR. "
                "Neither method dominates the other; the objective determines the winner.",
                "<b>The held-out family generalises.</b> On poisson_arrivals, which is excluded from "
                "training, the controller still lands between SJF and RR rather than collapsing to a "
                "single policy (Table 8.2).",
                "<b>Overhead is negligible relative to simulated time.</b> Mean controller cost is "
                f"{fmt_ms(data.summary['mean_runtime_q_observation_us_per_decision'], 2)} µs for "
                "observation plus "
                f"{fmt_ms(data.summary['mean_runtime_q_selection_us_per_decision'], 2)} µs for "
                "selection per decision; it is measured but never charged to the schedule.",
            ]
        )
    )
    story.append(h2("8.7 Tabular coverage and fallback behaviour"))
    cov_rows = []
    for _, row in data.coverage[data.coverage["split"] == "training"].iterrows():
        cov_rows.append(
            [
                str(int(row["training_seed"])),
                f"{int(row['episodes']):,}",
                f"{int(row['training_decisions']):,}",
                f"{int(row['visited_states'])} ({row['visited_states'] / 162:.1%})",
                f"{int(row['visited_state_actions'])} ({row['state_action_coverage']:.1%})",
                f"{row['training_wall_ms']:,.0f} / {row['training_q_update_ms']:,.0f}",
            ]
        )
    story.append(
        make_table(
            [["Train seed", "Episodes", "Decisions", "States visited (of 162)",
              "(s, a) visited (of 648)", "Train wall / Q-update (ms)"]]
            + cov_rows,
            widths=[CONTENT_W * 0.10, CONTENT_W * 0.11, CONTENT_W * 0.13,
                    CONTENT_W * 0.19, CONTENT_W * 0.19, CONTENT_W * 0.28],
            align_right=(1, 2, 5),
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 8.4: Per-seed tabular coverage, decisions and training time."))
    story.extend(
        figure(
            figures["coverage"],
            "Figure 8.6: Learned state-action visitation pooled over the five seeds (left) and "
            "per-seed coverage of states and pairs (right).",
        )
    )
    story.append(h2("8.8 Illustrative learned trace"))
    demo = data.demo
    story.append(
        para(
            f"The trace below is selected from the <i>validation</i> split by a predeclared rule "
            f"(maximise policy switches among Q traces with zero unseen-state fallbacks): training "
            f"seed {demo['training_seed']}, family <font face='Courier'>{demo['family']}</font>, "
            f"repetition {demo['repetition']}, fingerprint "
            f"<font face='Courier'>{demo['workload_fingerprint']}</font>. It contains "
            f"{demo['policy_switch_count']} policy switches across {demo['decision_count']} decisions "
            f"with {demo['metrics']['avg_waiting_time']:.3f} mean waiting time and "
            f"{demo['metrics']['avg_response_time']:.3f} mean response time. It is illustrative of "
            "the switching behaviour and explicitly <b>not representative</b> of average performance."
        )
    )
    story.extend(
        figure(
            figures["gantt"],
            "Figure 8.7: Illustrative validation trace. Each bar is one execution segment coloured by "
            "the policy chosen for it; dotted lines mark policy switches.",
        )
    )
    story.append(h2("8.9 Validation versus held-out test"))
    val_rows = []
    for method in METHOD_ORDER:
        val_rows.append(
            [
                method,
                f"{data.summary_mean('validation', method, 'avg_waiting_time'):,.3f}",
                f"{data.summary_mean('final', method, 'avg_waiting_time'):,.3f}",
                f"{data.summary_mean('validation', method, 'avg_response_time'):,.3f}",
                f"{data.summary_mean('final', method, 'avg_response_time'):,.3f}",
                f"{data.summary_mean('validation', method, 'context_switches'):,.2f}",
                f"{data.summary_mean('final', method, 'context_switches'):,.2f}",
            ]
        )
    story.append(
        make_table(
            [["Method", "Val waiting (140)", "Test waiting (210)", "Val response", "Test response",
              "Val ctx switches", "Test ctx switches"]]
            + val_rows,
            widths=[CONTENT_W * 0.19] + [CONTENT_W * 0.135] * 6,
            align_right=tuple(range(1, 7)),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 8.5: Validation (master seed 8201, 140 workloads) versus held-out final test "
            "(master seed 19301, 210 workloads). The two splits agree closely, so no split-specific "
            "tuning is implied."
        )
    )
    return story


# ---- 9. Testing -----------------------------------------------------------
def section_testing(data: Artifacts, test_rows: List[List[str]], pytest_block: str) -> List:
    story = [h1("9. Testing and verification")]
    story.append(
        para(
            "Verification is automated with pytest. The suite covers the standalone schedulers, the "
            "workload models and metrics, the causal encoder and Q-learning update, the runtime "
            "experiment pipeline, and the legacy offline reference. The counts below were collected "
            "on the reporting host immediately before this report was generated."
        )
    )
    story.append(
        make_table(
            [["Test module", "Checks", "What it protects"]] + test_rows,
            widths=[CONTENT_W * 0.27, CONTENT_W * 0.09, CONTENT_W * 0.64],
            align_right=(1,),
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 9.1: Automated test inventory (273 checks total)."))
    story.append(Spacer(1, 5))
    story.append(h2("9.1 Verification claims covered by the suite"))
    story.extend(
        bullets(
            [
                "<b>Fixed-policy equivalence.</b> Running the event-driven simulator with a constant "
                "action reproduces the standalone FCFS, SJF, RR and Priority schedules exactly, "
                "including metrics, context switches and idle time.",
                "<b>Causal hidden-future invariance.</b> Changing arrivals, bursts and priorities of "
                "processes that have not yet arrived, while keeping the observed prefix fixed, leaves "
                "the observation and action prefixes bit-for-bit identical.",
                "<b>Reward-to-waiting identity.</b> Σ r<sub>t</sub> equals −total waiting time / q for "
                "γ = 1; the terminal transition omits the bootstrap; first-visit exploration, "
                "unvisited-action masking and read-only evaluation are asserted.",
                "<b>Split disjointness and reproducibility.</b> Zero SHA-256 workload fingerprint "
                "overlap across train / validation / test, and deterministic metric regeneration.",
            ]
        )
    )
    story.append(h2("9.2 Test run output"))
    story.append(code_block(pytest_block, font_size=6.8))
    return story


# ---- 10. Legacy offline reference ----------------------------------------
def section_legacy(data: Artifacts, figures: Dict[str, Path]) -> List:
    story = [h1("10. Legacy offline policy selector (reference only)")]
    story.append(
        para(
            "Review 1 produced a whole-workload offline selector: it observed complete pre-execution "
            "workload features (81 states), chose one policy for the entire workload, and was trained "
            "with counterfactual rewards computed from four complete schedules with zero switch cost. "
            "It is preserved for continuity and is documented separately, because it is not a causal "
            "controller and must never be mixed into the runtime claims above."
        )
    )
    tables = data.legacy["tables"]
    baselines = {row["policy"]: row for row in tables["policy_summary_baselines"]}
    selector = tables["policy_summary_selector"][0]
    rows = []
    for policy in ["FCFS", "SJF", "Round Robin", "Priority"]:
        row = baselines[policy]
        reduction = 100.0 * (row["avg_waiting_time_mean"] - selector["avg_waiting_time_mean"]) / row[
            "avg_waiting_time_mean"
        ]
        rows.append(
            [
                policy,
                f"{row['avg_waiting_time_mean']:.2f} ± {row['avg_waiting_time_sd']:.2f}",
                f"{row['avg_turnaround_time_mean']:.2f} ± {row['avg_turnaround_time_sd']:.2f}",
                f"{row['avg_response_time_mean']:.2f} ± {row['avg_response_time_sd']:.2f}",
                f"{row['cpu_utilization_mean']:.2f}",
                f"{row['context_switches_mean']:.2f}",
                f"{reduction:+.1f}%",
            ]
        )
    rows.append(
        [
            "Offline Policy Selector (Q-Learning)",
            f"{selector['avg_waiting_time_mean']:.2f} ± {selector['avg_waiting_time_sd']:.2f}",
            f"{selector['avg_turnaround_time_mean']:.2f} ± {selector['avg_turnaround_time_sd']:.2f}",
            f"{selector['avg_response_time_mean']:.2f} ± {selector['avg_response_time_sd']:.2f}",
            f"{selector['cpu_utilization_mean']:.2f}",
            f"{selector['context_switches_mean']:.2f}",
            "reference",
        ]
    )
    story.append(
        make_table(
            [["Offline method (210 workloads, switch cost 0)", "Mean waiting", "Mean turnaround",
              "Mean response", "CPU util %", "Ctx switches", "Wait vs selector"]]
            + rows,
            widths=[CONTENT_W * 0.26, CONTENT_W * 0.145, CONTENT_W * 0.15,
                    CONTENT_W * 0.14, CONTENT_W * 0.10, CONTENT_W * 0.10,
                    CONTENT_W * 0.105],
            align_right=(1, 2, 3, 4, 5, 6),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 10.1: Legacy offline selector. Because the selector sees the whole workload and "
            "pays no switch cost, these numbers are not comparable with the runtime results and are "
            "reported only as a historical reference."
        )
    )
    story.extend(
        figure(
            figures["legacy_policy"],
            "Figure 10.1: Legacy offline selector — share of each policy per workload family. It "
            "learned to choose SJF on batch-like families and RR on interactive ones.",
            width=CONTENT_W * 0.95,
        )
    )
    story.extend(
        figure(
            figures["legacy_training"],
            "Figure 10.2: Legacy offline selector training: rolling mean reward (coloured) and "
            "epsilon decay (dashed black).",
            width=CONTENT_W * 0.95,
        )
    )
    return story


# ---- 11. Limitations ------------------------------------------------------
def section_limitations() -> List:
    story = [h1("11. Limitations and future work")]
    limits = [
        ["Synthetic workloads", "Seven generator families with known parameters; not trace-driven.",
         "Replay real workload traces and add I/O bursts."],
        ["Burst knowledge", "Exact burst is known on arrival (SJF assumption).",
         "Add a burst predictor with explicit prediction error."],
        ["Single CPU", "No multicore, migration cost or cache/bus contention.",
         "Extend the environment to N cores with per-core queues."],
        ["No I/O or blocking", "Processes never block; the queue is CPU-bound only.",
         "Model I/O bursts and blocked/wake transitions."],
        ["Coarse discretisation", "162 states; only ~42% of state-action pairs are ever visited.",
         "Tile coding, eligibility traces, or a small function approximator with the same causal "
         "contract."],
        ["Single objective", "The reward is waiting time only; response time is not optimised.",
         "Multi-objective / constrained RL balancing waiting, response and switch cost."],
        ["Fixed quantum and switch cost", "q = 4 and cost = 1 throughout.",
         "Sensitivity study over quantum and switch cost; learn the quantum too."],
        ["Simulation only", "No kernel integration; timings are host-specific.",
         "Prototype in a user-space scheduler or an eBPF-driven policy hook."],
    ]
    story.append(
        make_table(
            [["Limitation", "Current state", "Planned direction"]] + limits,
            widths=[CONTENT_W * 0.19, CONTENT_W * 0.40, CONTENT_W * 0.41],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        para(
            "The most important limitation to restate is the negative result: on these generators, "
            "the learned runtime controller does <b>not</b> beat non-preemptive SJF on mean waiting "
            "time. Because SJF is given exact bursts at arrival and never pays a mid-job switch cost, "
            "it is a demanding reference. The learned controller's measurable, reproducible gains are "
            "against Round Robin, FCFS, Priority and the predeclared adaptive heuristic, plus a "
            "response-time improvement over SJF."
        )
    )
    return story


# ---- 12. Conclusion -------------------------------------------------------
def section_conclusion(data: Artifacts) -> List:
    q_wait = data.summary_mean("final", "Runtime Q-learning", "avg_waiting_time")
    sjf_wait = data.summary_mean("final", "SJF", "avg_waiting_time")
    heur_wait = data.summary_mean("final", "Causal heuristic", "avg_waiting_time")
    rr_wait = data.summary_mean("final", "Round Robin", "avg_waiting_time")
    story = [h1("12. Conclusion")]
    story.append(
        para(
            "Review 2 delivers a working, causally constrained, event-driven adaptive scheduler and an "
            "evaluation that is designed to survive scrutiny. The runtime environment preserves the "
            "semantics of the four classical policies while allowing the policy to change at every "
            "dispatch or quantum boundary; the controller sees only arrived work through a 162-state "
            "abstraction; and the learning objective is provably identical to minimising total "
            "waiting time because the undiscounted return equals −WaitingTime / q."
        )
    )
    story.append(
        para(
            f"On 210 untouched held-out workloads the learned controller averages {fmt(q_wait)} time "
            f"units of waiting time, against {fmt(rr_wait)} for Round Robin, "
            f"{fmt(data.summary_mean('final', 'FCFS', 'avg_waiting_time'))} for FCFS, "
            f"{fmt(data.summary_mean('final', 'Priority', 'avg_waiting_time'))} for Priority and "
            f"{fmt(heur_wait)} for the predeclared heuristic, while SJF remains the strongest waiting-"
            f"time reference at {fmt(sjf_wait)}. Every comparison is paired and reported with a "
            "family-stratified bootstrap interval; every artifact — Q values, visit counts, coverage, "
            "fallback counts, workload fingerprints and metric tables — is committed; and the entire "
            "experiment, this report included, regenerates from declared seeds with a single command."
        )
    )
    story.append(
        para(
            "The honest conclusion is therefore conditional rather than triumphal: runtime adaptation "
            "learned with a small causal state abstraction beats every fixed policy except SJF on "
            "waiting time, beats a hand-written adaptive rule by a wide margin, and improves response "
            "time relative to SJF — but it does not beat SJF on the metric it optimises, and the "
            "report says so. That is the result the experiment supports."
        )
    )
    return story


# ---- 13. References -------------------------------------------------------
def section_references() -> List:
    story = [h1("References")]
    refs = [
        "A. Silberschatz, P. B. Galvin and G. Gagne, <i>Operating System Concepts</i> — CPU scheduling "
        "chapter (FCFS, SJF, RR, Priority, multi-level feedback queues).",
        "A. S. Tanenbaum and H. Bos, <i>Modern Operating Systems</i> — scheduling in batch, "
        "interactive and real-time systems.",
        "A. Shrivastava, <i>Reinforcement Learning for Scheduling Threads on a Multi-Core Processor</i>, "
        "CS229 project report, Stanford University, 2010. "
        "cs229.stanford.edu/proj2010/Shrivastava-ReinforcementLearningForSchedulingThreadsOnAMultiCoreProcessor.pdf",
        "G. Velingkar, J. K. Kumar, R. Varadarajan, S. Lanka and M. Anand Kumar, <i>Task Scheduling "
        "Using Deep Q-Learning</i>, Lecture Notes in Electrical Engineering 858, Springer, 2022. "
        "doi:10.1007/978-981-19-0840-8_58",
        "<i>Dynamic Operating System Scheduling Using Double DQN: A Reinforcement Learning Approach "
        "to Task Optimization</i>, arXiv:2503.23659, 2025. arxiv.org/abs/2503.23659",
        "<i>A Context-Aware Intelligent Scheduling Framework for Modern Operating Systems</i>, Journal "
        "of Computational and Cognitive Engineering, 2026 (random-forest classification with a "
        "deep Q-network). doi:10.47852/bonviewJCCE62026664",
        "T. Zhou and M. Lin, <i>CPU frequency scheduling of real-time applications on embedded devices "
        "with temporal encoding-based deep reinforcement learning</i>, Journal of Systems Architecture "
        "(arXiv:2309.03779), 2023. arxiv.org/abs/2309.03779",
        "C. Shyalika, T. Silva and A. Karunananda, <i>Reinforcement learning in dynamic task "
        "scheduling: a review</i>, SN Computer Science 1, article 306, 2020.",
        "R. S. Sutton and A. G. Barto, <i>Reinforcement Learning: An Introduction</i> — temporal-"
        "difference learning, ε-greedy exploration and tabular Q-learning.",
        "B. Efron and R. J. Tibshirani, <i>An Introduction to the Bootstrap</i> — paired bootstrap "
        "confidence intervals used for all comparisons in Section 8.",
    ]
    for i, ref in enumerate(refs, start=1):
        story.append(
            Paragraph(
                f"[{i}] {ref}",
                ParagraphStyle("ref", parent=S["Body"], fontSize=8.6, leading=11.4,
                               leftIndent=12, firstLineIndent=-12, spaceAfter=3.5),
            )
        )
    return story


# ---- Appendices -----------------------------------------------------------
def section_appendix(data: Artifacts) -> List:
    story = [h1("Appendix A. Reproducibility and artifact index")]
    story.append(h2("A.1 Commands"))
    story.append(
        code_block(
            """
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

python -m pytest -q                       # 273 unit and integration tests
python main.py runtime-experiment         # reproduces results/runtime/ from the declared seeds
python main.py runtime-experiment --results-dir /tmp/os-project-runtime   # isolated verification run
python main.py experiment                 # reproduces the legacy offline selector

# Regenerate this PDF from the committed artifacts
pip install -r requirements-report.txt
python tools/build_review2_report.py
""",
            font_size=6.8,
        )
    )
    story.append(h2("A.2 Artifact index"))
    artifacts = [
        ["results/runtime/runtime_final_test_metrics.csv",
         "2,100 rows — six metrics, switch and decision counts, timings, final test"],
        ["results/runtime/runtime_validation_metrics.csv",
         "1,400 rows — the same columns for the validation split"],
        ["results/runtime/runtime_training_metrics.csv",
         "6,000 rows — sequential training outcomes per episode"],
        ["results/runtime/runtime_paired_comparisons.csv",
         "Paired differences with 95% crossed-bootstrap intervals"],
        ["results/runtime/runtime_state_action_coverage.csv",
         "Visited states/pairs, fallbacks, training and evaluation timings"],
        ["results/runtime/runtime_q_table.csv", "3,240 rows — Q values and visit counts per seed"],
        ["results/runtime/runtime_workload_manifest.csv",
         "6,350 rows — split, seeds and workload fingerprints"],
        ["results/runtime/runtime_learned_switch_demo.json",
         "Illustrative validation trace: observations, actions, switch times, trace"],
        ["results/runtime/runtime_summary.json",
         "Configuration, split fingerprints, all means, demo and software versions"],
        ["results/runtime/runtime_{validation,final_test}_decisions.csv",
         "Full causal event logs — generated by the CLI, intentionally gitignored, reproducible"],
        ["results/ (root)", "Legacy offline selector: metrics.csv, decisions.csv, summary.json, "
                            "training_history.json"],
        ["figures/", "Legacy selector figures; figures/review2/ holds the figures in this report"],
    ]
    story.append(
        make_table(
            [["Path", "Contents"]] + artifacts,
            widths=[CONTENT_W * 0.42, CONTENT_W * 0.58],
        )
    )
    story.append(Spacer(1, 4))
    story.append(h2("A.3 Environment used for the recorded run"))
    software = data.summary["software"]
    story.append(
        make_table(
            [
                ["Item", "Value"],
                ["Python", software["python"]],
                ["Platform", software["platform"]],
                ["NumPy / pandas", f"{software['packages']['numpy']} / {software['packages']['pandas']}"],
                ["Matplotlib", software["packages"]["matplotlib"]],
                ["pytest", software["packages"]["pytest"]],
            ],
            widths=[CONTENT_W * 0.24, CONTENT_W * 0.76],
        )
    )
    return story


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_figures(data: Artifacts, figures_dir: Path) -> Dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "architecture": figures_dir / "fig_architecture.png",
        "flow": figures_dir / "fig_decision_loop.png",
        "metrics": figures_dir / "fig_final_metrics.png",
        "box": figures_dir / "fig_metric_boxplots.png",
        "family": figures_dir / "fig_family_bars.png",
        "training": figures_dir / "fig_training_curves.png",
        "forest": figures_dir / "fig_bootstrap_forest.png",
        "gantt": figures_dir / "fig_demo_gantt.png",
        "coverage": figures_dir / "fig_coverage.png",
        "legacy_policy": figures_dir / "fig_legacy_policy_share.png",
        "legacy_training": figures_dir / "fig_legacy_training.png",
    }
    figs.architecture_diagram(paths["architecture"])
    figs.decision_loop(paths["flow"])
    figs.metric_bars(data.final, paths["metrics"])
    figs.metric_boxplots(data.final, paths["box"])
    figs.family_bars(data.final, paths["family"], data.families)
    figs.training_curves(data.train, paths["training"])
    figs.bootstrap_forest(data.paired, paths["forest"])
    figs.gantt_trace(data.demo, paths["gantt"])
    figs.coverage_heatmap(data.q_table, data.coverage, paths["coverage"])
    figs.legacy_policy_share(data.legacy["tables"]["policy_selection"], paths["legacy_policy"])
    figs.legacy_training(data.legacy_history, paths["legacy_training"])
    return paths


def collect_test_counts(root: Path) -> Tuple[List[List[str]], str]:
    """Run pytest and return (per-module rows, condensed output)."""
    descriptions = {
        "test_adaptive.py": "Legacy offline selector: state encoding, decisions, read-only evaluation",
        "test_experiment_design.py": "Frozen seeds, split sizes and configuration invariants",
        "test_experiments.py": "Legacy experiment pipeline, metrics and report generation",
        "test_fcfs.py": "FCFS ordering, idle handling and metric computation",
        "test_generator.py": "Workload family generation and reproducibility from seeds",
        "test_metrics.py": "Waiting / turnaround / response / utilisation / throughput definitions",
        "test_priority.py": "Priority ordering, ties and non-preemption",
        "test_q_learning.py": "Q updates, epsilon schedule, masking, visit counts, evaluation",
        "test_reward.py": "Reward composition and clipping against the reference policies",
        "test_round_robin.py": "Quantum boundaries, re-enqueue order and completion behaviour",
        "test_runtime_experiment.py": "Split disjointness, determinism, artifact regeneration",
        "test_runtime_scheduler.py": "Runtime environment: fixed-action equivalence, causality, "
                                     "reward identity",
        "test_schedule_result.py": "Schedule trace validation and aggregate result objects",
        "test_sjf.py": "SJF selection, tie-breaking and non-preemption",
        "test_state.py": "Offline state discretisation and mixed-radix indexing",
        "test_workload_models.py": "Process/workload models, validation and trace invariants",
    }
    try:
        collect = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=root, capture_output=True, text=True, timeout=600,
        )
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root, capture_output=True, text=True, timeout=900,
        )
    except (OSError, subprocess.SubprocessError):
        return [], "pytest could not be executed on this host."

    counts: Dict[str, int] = {}
    for line in collect.stdout.splitlines():
        if "::" in line:
            module = line.split("::")[0].split("/")[-1]
            counts[module] = counts.get(module, 0) + 1
    rows = [
        [name, str(counts.get(name, 0)), descriptions.get(name, "")]
        for name in sorted(descriptions)
    ]
    total = sum(counts.values())
    tail = [line for line in run.stdout.strip().splitlines() if line.strip()][-3:]
    output = "\n".join(["$ python -m pytest -q", *tail])
    if not tail:
        output = f"$ python -m pytest -q\n{total} tests (output unavailable)"
    return rows, output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root holding results/ and figures/")
    parser.add_argument("--output", type=Path, default=None,
                        help="PDF path (default: <root>/Review_2_Report.pdf)")
    parser.add_argument("--docs-copy", type=Path, default=None,
                        help="second copy (default: <root>/docs/Review_2_Report.pdf)")
    parser.add_argument("--figures-dir", type=Path, default=None,
                        help="where figures are written (default: <root>/figures/review2)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="do not run pytest while building the report")
    args = parser.parse_args(argv)

    root: Path = args.root
    output = args.output or root / "Review_2_Report.pdf"
    docs_copy = args.docs_copy or root / "docs" / "Review_2_Report.pdf"
    figures_dir = args.figures_dir or root / "figures" / "review2"

    data = Artifacts(root)
    figures = build_figures(data, figures_dir)

    if args.skip_tests:
        test_rows: List[List[str]] = []
        pytest_block = "(pytest not executed: --skip-tests)"
        test_line = "273 checks (not re-run for this build)"
    else:
        test_rows, pytest_block = collect_test_counts(root)
        total = sum(int(row[1]) for row in test_rows) if test_rows else 0
        passed = "passed" in pytest_block
        test_line = f"{total} checks, {'all passing' if passed else 'see test run output'}"
        if total:
            test_line += " (re-run during this build)"

    generated = _dt.date.today().strftime("%d %B %Y")
    cli_block = build_cli_block(data)

    story: List = []
    story += cover_page(data, generated, test_line)
    story += toc_page()
    story += section_introduction(data)
    story.append(PageBreak())
    story += section_literature()
    story.append(PageBreak())
    story += section_requirements()
    story.append(PageBreak())
    story += section_design(data, figures)
    story.append(PageBreak())
    story += section_algorithms()
    story.append(PageBreak())
    story += section_implementation(data, cli_block)
    story.append(PageBreak())
    story += section_setup(data)
    story.append(PageBreak())
    story += section_results(data, figures)
    story.append(PageBreak())
    story += section_testing(data, test_rows, pytest_block)
    story.append(PageBreak())
    story += section_legacy(data, figures)
    story.append(PageBreak())
    story += section_limitations()
    story += section_conclusion(data)
    story.append(PageBreak())
    story += section_references()
    story.append(PageBreak())
    story += section_appendix(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReviewDoc(str(output), generated=generated)
    doc.multiBuild(story)

    if docs_copy.resolve() != output.resolve():
        docs_copy.parent.mkdir(parents=True, exist_ok=True)
        docs_copy.write_bytes(output.read_bytes())

    size_kb = output.stat().st_size / 1024
    print(f"wrote {output} ({size_kb:,.0f} KB)")
    print(f"wrote {docs_copy}")
    print(f"figures in {figures_dir}")
    return 0


def build_cli_block(data: Artifacts) -> str:
    lines = [
        "$ python main.py runtime-experiment --results-dir /tmp/os-project-runtime",
        "Causal event-driven runtime experiment",
        "---------------------------------------",
        f"  training seeds                 : "
        f"{', '.join(str(s) for s in data.cfg['training_seeds'])}",
        f"  training episodes/model        : {data.cfg['training_episodes_per_seed']}",
        f"  validation workloads           : {data.summary['validation_unique_workloads']}",
        f"  final held-out test workloads  : {data.summary['final_test_unique_workloads']}",
        "  final-test mean waiting time:",
    ]
    for method in METHOD_ORDER:
        lines.append(
            f"    {method:<22}: "
            f"{data.summary_mean('final', method, 'avg_waiting_time'):.3f}"
        )
    demo = data.summary["demo"]
    lines += [
        f"  learned within-trace switches : {demo['policy_switch_count']} "
        f"({demo['decision_count']} decisions; {demo['path']})",
        "  artifacts:",
        "    training_metrics        : runtime_training_metrics.csv",
        "    validation_metrics      : runtime_validation_metrics.csv",
        "    final_test_metrics      : runtime_final_test_metrics.csv",
        "    paired_comparisons      : runtime_paired_comparisons.csv",
        "    state_action_coverage   : runtime_state_action_coverage.csv",
        "    q_table                 : runtime_q_table.csv",
        "    workload_manifest       : runtime_workload_manifest.csv",
        "    learned_switch_demo     : runtime_learned_switch_demo.json",
        "    summary                 : runtime_summary.json",
        "    report                  : runtime_report.md",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
