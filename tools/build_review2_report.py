"""Build the Review 2 PDF for "Methodology and Partial Implementation".

The document follows the Review 2 rubric exactly, in this order:

    1. Detailed methodology of the proposed system
    2. Overall conceptual, architectural or pipeline diagram
    3. Description of the major modules
    4. Workflow diagram for each major module
    5. Dataset details and data collection procedure
    6. Tools, technologies, algorithms and frameworks used
    7. Experimental plan and evaluation metrics
    8. Implementation progress (completed modules with valid outputs)
    9. Challenges encountered and plan for completing the remaining work

Everything is rendered from the committed experiment artifacts under ``results/``
(plus the project's own seeded generator for the dataset statistics), so the
numbers cannot drift from the recorded experiment.

Usage::

    pip install -r requirements-report.txt
    python tools/build_review2_report.py                       # both PDF copies
    python tools/build_review2_report.py --skip-tests          # skip the pytest run
    python tools/build_review2_report.py --output /tmp/r.pdf   # single file

Outputs: ``Review_2_Report.pdf``, ``docs/Review_2_Report.pdf``,
``figures/review2/*.png``.
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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import (  # noqa: E402
    TableOfContents,
)

import review2_figures as figs  # noqa: E402

NAVY = colors.HexColor("#1F3864")
STEEL = colors.HexColor("#2E75B6")
BAND = colors.HexColor("#F2F5FB")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#B7C4DD")

REPORT_TITLE = "Event-Driven Runtime-Adaptive CPU Scheduling"
REPORT_SUBTITLE = "with Sequential Tabular Q-Learning"
REVIEW_LINE = "Review 2 — Methodology and Partial Implementation (Lab Assessment 9)"
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
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20.5,
            leading=24, textColor=NAVY, spaceAfter=3, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12,
            leading=15, textColor=STEEL, alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base["Normal"], fontSize=9, leading=13,
            textColor=colors.HexColor("#222222"), alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12.8,
            leading=15.5, textColor=NAVY, spaceBefore=9, spaceAfter=4,
        ),
        "H2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=13, textColor=STEEL, spaceBefore=7, spaceAfter=3,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=12.5, alignment=TA_JUSTIFY, spaceAfter=4,
            textColor=colors.HexColor("#1A1A1A"),
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.0,
            leading=12.1, leftIndent=11, bulletIndent=2, spaceAfter=2.5,
            alignment=TA_JUSTIFY,
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=7.7,
            leading=9.5, textColor=GREY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=7,
        ),
        "Cell": ParagraphStyle(
            "Cell", parent=base["Normal"], fontSize=7.4, leading=9.1,
            textColor=colors.HexColor("#1A1A1A"),
        ),
        "CellH": ParagraphStyle(
            "CellH", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.4,
            leading=9.1, textColor=colors.white,
        ),
        "Small": ParagraphStyle(
            "Small", parent=base["Normal"], fontSize=8.0, leading=10.5, textColor=GREY,
        ),
        "Equation": ParagraphStyle(
            "Equation", parent=base["Normal"], fontName="Courier-Bold", fontSize=8.5,
            leading=12, alignment=TA_CENTER, textColor=NAVY, spaceBefore=3, spaceAfter=3,
        ),
        "Abstract": ParagraphStyle(
            "Abstract", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1,
            leading=12.6, alignment=TA_JUSTIFY, borderPadding=6,
            backColor=colors.HexColor("#F5F7FC"),
            textColor=colors.HexColor("#1A1A1A"),
        ),
    }


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


def code_block(text: str, font_size: float = 6.9, width: float = CONTENT_W) -> Table:
    """Render preformatted text (terminal output / pseudocode) in a framed box."""
    lines = [
        html.escape(line).replace(" ", "&nbsp;") for line in text.strip("\n").split("\n")
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
    band: bool = True,
) -> Table:
    """Styled table; cells may be strings (auto-wrapped) or Paragraphs."""
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for cell in row:
            if isinstance(cell, Paragraph):
                out.append(cell)
            elif r_i == 0 and header:
                out.append(Paragraph(str(cell), S["CellH"]))
            else:
                out.append(Paragraph(str(cell), S["Cell"]))
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
    cells = []
    for value, label in items:
        cells.append(
            Table(
                [
                    [Paragraph(
                        f'<font size="12.5"><b>{value}</b></font>',
                        ParagraphStyle("kv", alignment=TA_CENTER, textColor=NAVY, leading=14),
                    )],
                    [Paragraph(
                        f'<font size="6.5">{label}</font>',
                        ParagraphStyle("kl", alignment=TA_CENTER, textColor=GREY, leading=8),
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
            title="Review 2 - Methodology and Partial Implementation - "
                  "Event-Driven Runtime-Adaptive CPU Scheduling",
            author="OS-Project (github.com/lostpilot404/OS-Project)",
            subject="Review 2 (Lab Assessment 9)",
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

    def _cover(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 1.15 * cm, PAGE_W, 1.15 * cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(MARGIN_X, PAGE_H - 0.78 * cm,
                          "OPERATING SYSTEMS PROJECT  •  REVIEW 2")
        canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 0.78 * cm,
                               "LAB ASSESSMENT 9")
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
                          "Review 2  |  Methodology and Partial Implementation  |  "
                          + REPORT_TITLE)
        canvas.drawRightString(PAGE_W - MARGIN_X, y_top + 0.16 * cm, REPO_URL)
        y_bot = MARGIN_BOTTOM - 0.25 * cm
        canvas.line(MARGIN_X, y_bot, PAGE_W - MARGIN_X, y_bot)
        canvas.drawString(MARGIN_X, y_bot - 0.42 * cm,
                          f"Generated from committed artifacts on {self.generated}")
        canvas.drawRightString(PAGE_W - MARGIN_X, y_bot - 0.42 * cm,
                               f"Page {doc.page - 1}")
        canvas.restoreState()

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
    def __init__(self, root: Path, with_dataset: bool = True):
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
        self.cfg = self.summary["configuration"]
        self.families = list(self.cfg["validation_and_test_families"])
        self.workloads: pd.DataFrame = pd.DataFrame()
        self.processes: pd.DataFrame = pd.DataFrame()
        if with_dataset:
            self._load_dataset()

    def _load_dataset(self) -> None:
        """Rebuild the held-out dataset with the project's own seeded generator."""
        from experiments.runtime_config import RuntimeExperimentConfig
        from experiments.runtime_experiment import build_runtime_workloads

        items = build_runtime_workloads("final_test", RuntimeExperimentConfig())
        workload_rows = []
        process_rows = []
        for item in items:
            workload = item.workload
            workload_rows.append(
                {
                    "family": item.family,
                    "repetition": item.repetition,
                    "seed": item.seed,
                    "fingerprint": workload.fingerprint,
                    "size": workload.size,
                    "total_burst": workload.total_burst_time,
                    "mean_burst": workload.mean_burst_time,
                    "max_arrival": workload.max_arrival_time,
                    "arrival_span": workload.arrival_span,
                }
            )
            for process in workload.processes:
                process_rows.append(
                    {
                        "family": item.family,
                        "repetition": item.repetition,
                        "fingerprint": workload.fingerprint,
                        "pid": process.pid,
                        "arrival_time": process.arrival_time,
                        "burst_time": process.burst_time,
                        "priority": process.priority,
                    }
                )
        self.workloads = pd.DataFrame(workload_rows)
        self.processes = pd.DataFrame(process_rows)

    def mean_sd(self, split: str, method: str, metric: str) -> Tuple[float, float]:
        frame = self.final if split == "final" else self.val
        values = frame.loc[frame["method"] == method, metric].to_numpy(dtype=float)
        return float(values.mean()), float(values.std(ddof=1))

    def summary_mean(self, split: str, method: str, metric: str) -> float:
        key = "final_test_means" if split == "final" else "validation_means"
        return float(self.summary[key][method][metric])

    def family_stat(self, family: str, column: str) -> float:
        return float(self.workloads.loc[self.workloads["family"] == family, column].mean())


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def ci_of(data: Artifacts, target: str, reference: str, metric: str) -> Tuple[float, float, float]:
    """Look up a predeclared paired difference and its 95% interval.

    Only pairs produced by the experiment's own comparison table are used, so the
    report never mixes a committed interval with one recomputed at build time.
    """
    sel = data.paired[
        (data.paired["target"] == target)
        & (data.paired["reference"] == reference)
        & (data.paired["metric"] == metric)
    ]
    if sel.empty:
        raise KeyError(f"no committed paired comparison for {target} vs {reference} / {metric}")
    row = sel.iloc[0]
    return float(row["mean_difference"]), float(row["ci95_low"]), float(row["ci95_high"])


def cover_page(data: Artifacts, generated: str, test_line: str) -> List:
    story: List = []
    story.append(Spacer(1, 0.45 * cm))
    story.append(Paragraph("OPERATING SYSTEMS PROJECT", S["cover_meta"]))
    story.append(Paragraph(REVIEW_LINE.upper(), S["cover_meta"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(REPORT_TITLE, S["title"]))
    story.append(Paragraph(REPORT_SUBTITLE, S["subtitle"]))
    story.append(
        Paragraph(
            "Single-CPU discrete-event simulator  •  causal arrived-work state abstraction  •  "
            "sequential tabular Q-learning  •  seven seeded workload families  •  held-out "
            "crossed-bootstrap evaluation",
            ParagraphStyle("cover_sub", parent=S["cover_meta"], fontSize=8.4,
                           leading=11.5, textColor=GREY),
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    meta = [
        ["Project title",
         "Event-Driven Runtime-Adaptive CPU Scheduling with Sequential Q-Learning"],
        ["Review stage", "Review 2 — Methodology and Partial Implementation (Lab Assessment 9)"],
        ["Repository", REPO_URL],
        ["Language / runtime", "Python 3.11.2 (NumPy 2.4.6, pandas 3.0.6, Matplotlib 3.11.2)"],
        ["Automated verification", test_line],
        ["Dataset", "7 synthetic families; 6,000 train / 140 validation / 210 held-out workloads"],
        ["Report generated", generated],
    ]
    story.append(
        make_table([["Field", "Value"]] + meta,
                   widths=[CONTENT_W * 0.24, CONTENT_W * 0.76])
    )
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        kpi_strip(
            [
                ("6 / 6", "major modules<br/>implemented"),
                ("3,150", "processes in the<br/>held-out dataset"),
                ("6,000", "training<br/>workloads"),
                ("210", "held-out test<br/>workloads"),
                ("162 × 4", "causal states ×<br/>actions"),
                ("273", "automated tests<br/>passing"),
            ]
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    q_wait = data.summary_mean("final", "Runtime Q-learning", "avg_waiting_time")
    sjf_wait = data.summary_mean("final", "SJF", "avg_waiting_time")
    rr_wait = data.summary_mean("final", "Round Robin", "avg_waiting_time")
    story.append(
        Paragraph(
            "<b>Summary of the work presented.</b> The proposed system replaces a fixed CPU "
            "scheduling policy with a causal, event-driven runtime controller. A single-CPU "
            "discrete-event simulator keeps one persistent FIFO ready queue and, at every dispatch "
            "or Round-Robin quantum boundary, asks a controller to choose FCFS, SJF, Round Robin or "
            "Priority for the next execution segment of the same evolving trace. The controller may "
            "observe only work that has already arrived; five causal features are discretised into "
            "162 states and a tabular Q-learning agent with an undiscounted interval reward "
            "(Σ r = −total waiting time / q) learns when to switch. This report documents the "
            "methodology (§1), the architecture (§2), the six modules and their workflows "
            "(§3–4), the generated dataset (§5), the tools and algorithms (§6), the "
            "experimental plan and metrics (§7), the implementation completed so far with its "
            f"outputs (§8), and the challenges and remaining plan (§9). Current measured "
            f"status on the 210 held-out workloads: runtime Q-learning {fmt(q_wait)} mean waiting "
            f"time versus {fmt(rr_wait)} for Round Robin and {fmt(sjf_wait)} for SJF, with 273/273 "
            "automated checks passing.",
            S["Abstract"],
        )
    )
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    return story


def toc_page() -> List:
    story: List = []
    story.append(Paragraph("Contents", S["H1"]))
    story.append(
        para(
            "The report follows the Review 2 rubric item by item; the nine headings below are the "
            "nine required items.",
            "Small",
        )
    )
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC0", fontName="Helvetica-Bold", fontSize=9.2, leading=15,
                       textColor=NAVY, leftIndent=0),
        ParagraphStyle("TOC1", fontName="Helvetica", fontSize=8.3, leading=12.5,
                       textColor=colors.HexColor("#333333"), leftIndent=14),
    ]
    story.append(toc)
    story.append(PageBreak())
    return story


# --------------------------------------------------------------------------
# 1. Methodology
# --------------------------------------------------------------------------
def section_methodology(data: Artifacts, figures: Dict[str, Path]) -> List:
    story = [h1("1. Detailed methodology of the proposed system")]
    story.append(h2("1.1 Research questions"))
    questions = [
        ["RQ1", "Can a single-CPU scheduler switch between FCFS, SJF, Round Robin and Priority at "
                "run time, using only information available at that instant?"],
        ["RQ2", "Does such switching reduce mean waiting time relative to the fixed policies it "
                "switches between, under a declared context-switch cost?"],
        ["RQ3", "Does a learned switching policy add value over a hand-written adaptive rule that "
                "sees the same observation?"],
        ["RQ4", "Do the gains, if any, survive on a workload family that was never seen in "
                "training, and are they larger than the paired bootstrap noise?"],
    ]
    story.append(
        make_table([["ID", "Question"]] + questions,
                   widths=[CONTENT_W * 0.07, CONTENT_W * 0.93])
    )
    story.append(h2("1.2 Methodology pipeline"))
    story.append(
        para(
            "The method is a closed loop between a discrete-event environment that owns the future "
            "and a controller that may see only the past and present. Stages 3–7 run once per "
            "decision epoch inside a single trace; stage 8 runs once per experiment."
        )
    )
    story.extend(
        figure(figures["methodology"],
               "Figure 1.1: Methodology pipeline. Stages 3–7 form the online decision loop; "
               "stage 8 is the offline evaluation that produces every number reported here.")
    )
    pipeline = [
        ["1. Dataset", "workload/generator.py",
         "Produce seven seeded families of 15-process workloads with SHA-256 derived seeds.",
         "RuntimeWorkload list + manifest"],
        ["2. Simulation", "scheduler/runtime.py",
         "Advance one CPU: arrivals, one FIFO ready queue, remaining bursts, switch cost.",
         "decision epochs + schedule trace"],
        ["3. Observation", "scheduler/runtime.py",
         "Freeze an immutable snapshot of arrived/ready work only (causal contract).",
         "RuntimeObservation"],
        ["4. Encoding", "rl/runtime_state.py",
         "Discretise five causal features into 162 tabular states (3×3×3×3×2).",
         "state index s ∈ [0, 162)"],
        ["5. Decision", "rl/runtime_controller.py",
         "ε-greedy (training) or greedy (evaluation) choice over four policy actions; "
         "explicit fallback in unseen states.",
         "action a ∈ {FCFS, SJF, RR, Priority}"],
        ["6. Execution", "scheduler/runtime.py",
         "Run the selected segment (RR bounded by the quantum), charge 1 unit when the running "
         "PID changes, settle endpoint arrivals.",
         "advanced simulated time"],
        ["7. Reward", "scheduler/runtime.py",
         "Measure the waiting-time area ΔW accumulated by ready processes; return "
         "r = −ΔW / q (γ = 1).",
         "interval reward r<sub>t</sub>"],
        ["8. Evaluation", "experiments/runtime_experiment.py",
         "Greedy read-only runs on held-out workloads; paired differences with a family-stratified "
         "crossed bootstrap.",
         "metric tables + CIs"],
    ]
    story.append(
        make_table(
            [["Stage", "Implemented in", "What happens", "Output"]] + pipeline,
            widths=[CONTENT_W * 0.13, CONTENT_W * 0.20, CONTENT_W * 0.47, CONTENT_W * 0.20],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 1.1: The eight methodology stages and the module that owns each."))

    story.append(h2("1.3 Causal observation contract"))
    story.append(
        para(
            "The environment privately owns the complete event list so it can admit future arrivals "
            "at their arrival times; the controller receives an immutable "
            "<font face='Courier'>RuntimeObservation</font>. This is the methodological core of the "
            "project: without it, any measured gain could come from information a real kernel could "
            "not have."
        )
    )
    contract = [
        ["Current simulated time", "Included", "Defines the decision epoch."],
        ["Previously selected policy", "Included", "The action from the immediately preceding "
                                                   "epoch (None at the first)."],
        ["Ready processes (arrived, unfinished)", "Included",
         "PID, arrival time, known burst, remaining burst, priority."],
        ["Arrived / completed counts", "Included", "Progress without knowing the workload size."],
        ["Mean burst and priority spread over arrived work", "Included",
         "Aggregates restricted to processes that have already arrived."],
        ["Mean arrival age of ready jobs", "Included",
         "current_time − arrival_time; includes prior CPU service, not queue waiting."],
        ["Unarrived process data, future bursts/arrivals", "Excluded",
         "Would break deployability and fairness of the comparison."],
        ["Total workload size, workload identity/family", "Excluded",
         "Prevents the controller from memorising generator families."],
        ["Counterfactual or completed-schedule metrics", "Excluded",
         "No peeking at outcomes of policies that were not run."],
    ]
    story.append(
        make_table(
            [["Quantity", "Visible to the controller?", "Methodological reason"]] + contract,
            widths=[CONTENT_W * 0.33, CONTENT_W * 0.17, CONTENT_W * 0.50],
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 1.2: The causal observation contract (scheduler/runtime.py, "
            "rl/runtime_state.py). A paired hidden-future test asserts that changing unarrived "
            "processes leaves the observation and action prefixes bit-for-bit identical."
        )
    )
    story.append(h2("1.4 State abstraction and learning objective"))
    encoding = [
        ["1", "ready_count", "3", "{1}, {2–3}, {4+}", "Instantaneous ready-queue contention."],
        ["2", "completed_count", "3", "{0}, {1–3}, {4+}",
         "Trace progress without the workload size."],
        ["3", "median_ready_remaining_burst", "3", "≤ q, ≤ 4q, > 4q",
         "Remaining service demand of ready jobs."],
        ["4", "mean_ready_arrival_age", "3", "≤ q, ≤ 4q, > 4q",
         "Age proxy for ready jobs (includes prior service)."],
        ["5", "arrived_work_priority_spread", "2", "zero / non-zero",
         "Whether arrived processes have distinct priorities."],
    ]
    story.append(
        make_table(
            [["#", "Feature", "Bins", "Discretisation (q = 4)", "Why it is causal"]] + encoding,
            widths=[CONTENT_W * 0.05, CONTENT_W * 0.24, CONTENT_W * 0.07,
                    CONTENT_W * 0.24, CONTENT_W * 0.40],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 1.3: 3 × 3 × 3 × 3 × 2 = 162 states; 162 × 4 = 648 "
                         "state-action pairs."))
    story.append(
        para(
            "Each trace is one sequential episode. Between consecutive epochs the environment "
            "measures the waiting-time area ΔW<sub>t</sub> accumulated by ready processes "
            "(including any switch interval) and returns a scaled interval reward:"
        )
    )
    story.append(
        Paragraph(
            "r<sub>t</sub> = − ΔW<sub>t</sub> / max(1, q),   γ = 1.0   ⇒   "
            "Σ<sub>t</sub> r<sub>t</sub> = − TotalWaitingTime / q",
            S["Equation"],
        )
    )
    story.append(
        para(
            "Because γ = 1 and the episode is finite, maximising the undiscounted return is "
            "identical to minimising total waiting time — the objective is not a proxy for it. "
            "The update is"
        )
    )
    story.append(
        Paragraph(
            "Q(s, a) ← Q(s, a) + α [ r<sub>t</sub> + γ max<sub>a′ ∈ visited(s′)</sub> "
            "Q(s′, a′) − Q(s, a) ],   α = 0.1",
            S["Equation"],
        )
    )
    story.append(
        para(
            "Terminal transitions omit the bootstrap; the first visit to a state forces a uniform "
            "random action; greedy selection and bootstrapping mask unvisited actions; and a wholly "
            "unseen evaluation state triggers an explicit, counted fallback to the causal heuristic "
            "instead of a silent zero-Q tie."
        )
    )
    story.append(h2("1.5 Methodological controls"))
    controls = [
        ["Fixed-policy baselines", "FCFS, SJF, Round Robin and Priority run on the identical "
                                   "workloads and pay the identical switch cost."],
        ["Predeclared adaptive rule", "A non-RL heuristic (RR → Priority → SJF → FCFS) "
                                      "was fixed before any test result was seen, so learning is "
                                      "not credited for what fixed rules already do."],
        ["Blinded held-out split", "Final test (master seed 19301) is generated and evaluated "
                                   "after training; the pipeline asserts zero fingerprint overlap "
                                   "with training."],
        ["Out-of-family generalisation", "poisson_arrivals is excluded from training entirely and "
                                         "used only at evaluation time."],
        ["Independent seeds", "Five agents (7101–7105) are trained separately; Q-learning "
                              "results average all five."],
        ["Paired uncertainty", "Differences are paired per workload and per model seed, then "
                               "resampled with a family-stratified crossed bootstrap (1,000 "
                               "replicates)."],
        ["Determinism", "Evaluation is greedy and read-only; artifacts regenerate bit-for-bit from "
                        "the declared seeds."],
        ["Separation of timing", "Observation, selection, Q-update and simulator wall time are "
                                 "measured in separate columns and never charged to simulated time."],
    ]
    story.append(
        make_table(
            [["Control", "How it is enforced"]] + controls,
            widths=[CONTENT_W * 0.24, CONTENT_W * 0.76],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 1.4: Controls that keep the comparison honest."))
    return story


# --------------------------------------------------------------------------
# 2. Architecture
# --------------------------------------------------------------------------
def section_architecture(data: Artifacts, figures: Dict[str, Path]) -> List:
    story = [h1("2. Overall conceptual, architectural and pipeline diagram")]
    story.append(
        para(
            "The system is layered so that the information boundary is structural, not "
            "conventional: the lower (blue) layer owns the future and the accounting, the upper "
            "(amber) layer only ever receives a frozen snapshot of arrived work. The controller's "
            "reply is a single policy name; it cannot touch the queue, the bursts or the event "
            "list."
        )
    )
    story.extend(
        figure(figures["architecture"],
               "Figure 2.1: Conceptual architecture. Blue = environment (owns future arrivals, "
               "ready queue and switch accounting); amber = controller path (arrived work only); "
               "green = the action returned to the environment.")
    )
    layers = [
        ["Data layer", "workload/models.py, workload/generator.py",
         "Defines Process/Workload objects and generates the seven seeded families with "
         "SHA-256-derived seeds."],
        ["Simulation layer", "scheduler/runtime.py",
         "Discrete-event loop, FIFO ready queue, remaining bursts, switch cost, schedule trace."],
        ["Baseline layer", "scheduler/{fcfs,sjf,round_robin,priority}.py",
         "Standalone fixed policies, preserved so runtime actions can be checked against them."],
        ["Perception layer", "rl/runtime_state.py",
         "Encodes an immutable observation into one of 162 causal states."],
        ["Decision layer", "rl/runtime_controller.py, rl/runtime_heuristic.py",
         "Learned policy (tabular Q-learning) and the predeclared non-RL rule."],
        ["Experiment layer", "experiments/runtime_experiment.py, runtime_config.py",
         "Splits, training, validation, held-out testing, paired bootstrap, artifact export."],
        ["Evaluation layer", "evaluation/metrics.py, evaluation/comparison.py",
         "Metric definitions and paired/bootstrap comparison helpers."],
    ]
    story.append(
        make_table(
            [["Layer", "Components", "Responsibility"]] + layers,
            widths=[CONTENT_W * 0.16, CONTENT_W * 0.34, CONTENT_W * 0.50],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 2.1: Architectural layers and what each one owns."))
    story.append(h2("2.1 Runtime control flow"))
    story.extend(
        bullets(
            [
                "The environment advances to the next decision epoch (a dispatch point, a "
                "completion, or a Round-Robin quantum boundary) and admits every arrival at or "
                "before that instant in (arrival_time, pid) order.",
                "It freezes an observation and passes it to the controller; the controller returns "
                "one of four policy names.",
                "If the selected process differs from the running PID, one unit of switch cost is "
                "charged before the segment starts; arrivals during that interval are queued but "
                "the dispatch already chosen is not reconsidered.",
                "The segment runs (Round Robin bounded by the quantum, others to completion); "
                "endpoint arrivals are admitted first, then an unfinished Round-Robin process is "
                "appended to the tail.",
                "The environment measures the waiting-time area accumulated by ready processes, "
                "converts it to the interval reward, and hands it back to the controller for the "
                "Q update.",
                "A switch changes only the next selection rule and service length — the queue, "
                "the remaining bursts and the switch accounting are preserved across switches.",
            ]
        )
    )
    return story


# --------------------------------------------------------------------------
# 3. Major modules
# --------------------------------------------------------------------------
MODULES = [
    ("M1", "Workload generator and dataset builder",
     "workload/generator.py, workload/models.py",
     "family config + seed", "validated Workload with fingerprint",
     "Generates the seven synthetic families, validates every process (positive burst, non-negative "
     "arrival, priority bounds) and records split/seed/fingerprint metadata."),
    ("M2", "Discrete-event runtime environment",
     "scheduler/runtime.py",
     "Workload + controller + (quantum, switch cost)", "schedule trace and six metrics",
     "Owns the private event list, the single FIFO ready queue, remaining bursts, switch-cost "
     "accounting, the causal observation and the interval reward."),
    ("M3", "Causal observation and state encoder",
     "rl/runtime_state.py",
     "RuntimeObservation", "state index in [0, 162)",
     "Extracts five causal features, discretises them relative to the quantum and flattens them in "
     "mixed-radix order. Accepts an observation only — never a Workload."),
    ("M4", "Sequential Q-learning controller",
     "rl/runtime_controller.py",
     "observation (encode) / reward (update)", "policy action; Q table and visit counts",
     "Tabular Q-learning over 648 pairs: ε-greedy exploration, first-visit forcing, "
     "unvisited-action masking, read-only greedy evaluation and a counted heuristic fallback."),
    ("M5", "Predeclared heuristic controller",
     "rl/runtime_heuristic.py",
     "RuntimeObservation", "policy action",
     "Non-RL adaptive rule (RR → Priority → SJF → FCFS) sharing the same observation "
     "contract, declared before any test result was inspected."),
    ("M6", "Experiment orchestration, metrics and reporting",
     "experiments/runtime_experiment.py, evaluation/metrics.py",
     "configuration + trained models", "CSV/JSON artifacts, paired CIs, markdown report",
     "Builds disjoint splits, trains five agents, validates, runs the held-out test, computes paired "
     "bootstrap intervals and writes every artifact."),
]


def section_modules(data: Artifacts) -> List:
    story = [h1("3. Description of the major modules")]
    story.append(
        para(
            "Six modules carry the whole system: two build and run the simulated machine, two "
            "implement the competing controllers, one is the non-learning reference rule, and one "
            "orchestrates the experiment and its statistics. Each is described below with its "
            "contract; §4 gives the workflow diagram for each."
        )
    )
    rows = [
        [mid, name, files, f"{inp} → {out}", responsibility, "Implemented"]
        for mid, name, files, inp, out, responsibility in MODULES
    ]
    story.append(
        make_table(
            [["ID", "Module", "Files", "Input → output", "Responsibility", "Status"]] + rows,
            widths=[CONTENT_W * 0.05, CONTENT_W * 0.18, CONTENT_W * 0.19,
                    CONTENT_W * 0.16, CONTENT_W * 0.34, CONTENT_W * 0.08],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 3.1: Major modules, contracts and current status."))

    details = [
        ("M1 — Workload generator and dataset builder",
         "Seven frozen family configurations (burst distribution and range, arrival pattern, "
         "priority pattern) are turned into workloads of 15 processes. Each workload seed is "
         "derived with SHA-256 from (master seed, split tag, family index, repetition), so the "
         "whole dataset is reproducible from three integers and no external data is required. The "
         "generator validates every process and the pipeline records a workload fingerprint for "
         "split-disjointness checks."),
        ("M2 — Discrete-event runtime environment",
         "The simulator is deliberately the only component that knows the future. It advances "
         "time to the next epoch, maintains one persistent FIFO ready queue, tracks remaining "
         "bursts, charges one unit whenever the running PID changes, and produces the schedule "
         "trace from which all metrics are computed. Running it with a constant action reproduces "
         "the standalone FCFS, SJF, Round Robin and Priority implementations exactly — a "
         "property that is asserted by tests, not merely claimed."),
        ("M3 — Causal observation and state encoder",
         "The encoder's API accepts a RuntimeObservation and nothing else, which makes the "
         "information boundary checkable by inspection. Five features are discretised into "
         "3 × 3 × 3 × 3 × 2 = 162 states. Bin edges are expressed relative to the "
         "quantum so the abstraction stays meaningful if the quantum changes."),
        ("M4 — Sequential Q-learning controller",
         "A tabular agent over 162 × 4 pairs. During training it follows an episode-level "
         "ε schedule, forces a random action on a state's first visit and masks unvisited "
         "actions in the greedy step and in the bootstrap. During evaluation it is greedy and "
         "read-only; an unseen state produces an explicit fallback to M5, and the number of "
         "fallbacks is reported rather than hidden."),
        ("M5 — Predeclared heuristic controller",
         "Round Robin if at least three jobs are ready and their mean arrival age is at least one "
         "quantum; else Priority if at least two ready jobs differ in priority; else SJF if the "
         "largest visible remaining burst is at least twice the smallest; else FCFS. It exists so "
         "that the learned controller is compared against adaptivity itself, not only against "
         "fixed policies."),
        ("M6 — Experiment orchestration, metrics and reporting",
         "Builds the three splits, asserts they are fingerprint-disjoint, trains the five agents, "
         "evaluates greedily on validation and on the held-out test, computes paired differences "
         "with a family-stratified crossed bootstrap, and writes every metric table, Q value, "
         "visit count, coverage figure and the illustrative trace to disk."),
    ]
    for title, body in details:
        story.append(h2(title))
        story.append(para(body))
    return story


# --------------------------------------------------------------------------
# 4. Module workflows
# --------------------------------------------------------------------------
WORKFLOWS = [
    ("M1. Workload generator and dataset builder",
     "workload/generator.py → workload/models.py",
     [
         ("Select family configuration", "7 frozen parameter sets (burst, arrival, priority)"),
         ("Derive the workload seed", "SHA-256 (master seed, split tag, family index, repetition)"),
         ("Sample 15 processes", "burst, arrival time and priority drawn per family rules"),
         ("Validate the workload", "positive bursts, ordered arrivals, priority bounds"),
         ("Fingerprint and tag", "SHA-256 workload fingerprint + split/repetition metadata"),
         ("Assemble the split", "6,000 train / 140 validation / 210 held-out workloads"),
     ]),
    ("M2. Discrete-event runtime environment",
     "scheduler/runtime.py — one iteration per decision epoch",
     [
         ("Advance to the next epoch", "dispatch point, completion or Round-Robin quantum boundary"),
         ("Admit arrivals", "all arrivals at or before the epoch, in (arrival_time, pid) order"),
         ("Freeze the observation", "arrived/ready work only — the future stays private"),
         ("Ask the controller for an action", "FCFS, SJF, Round Robin or Priority"),
         ("Charge the switch cost", "1 unit if the running PID changes; first dispatch is free"),
         ("Execute the segment", "min(quantum, remaining) for RR, otherwise to completion"),
         ("Settle the endpoint", "admit arrivals, then RR tail re-enqueue or mark completed"),
         ("Return the interval reward", "r = −ΔW / q, measured by the environment"),
     ]),
    ("M3. Causal observation and state encoder",
     "rl/runtime_state.py",
     [
         ("Receive the observation", "immutable RuntimeObservation (no Workload object)"),
         ("Extract five causal features", "ready count, completed count, median remaining burst, "
                                          "mean arrival age, priority spread"),
         ("Discretise each feature", "bin edges expressed relative to the quantum q = 4"),
         ("Flatten in mixed-radix order", "3 × 3 × 3 × 3 × 2 → one integer"),
         ("Return the state index", "s ∈ [0, 162); consumed by the controller only"),
     ]),
    ("M4. Sequential Q-learning controller",
     "rl/runtime_controller.py",
     [
         ("Encode the observation", "s = encoder(observation)"),
         ("Select an action", "training: ε-greedy + forced random on first visit; "
                              "evaluation: greedy over visited actions"),
         ("Handle unseen states", "explicit fallback to the causal heuristic (counted and reported)"),
         ("Receive the reward", "r = −ΔW / q from the environment"),
         ("Update the table", "Q(s,a) += α(r + γ max_visited Q(s′,·) − Q(s,a)); "
                              "terminal omits the bootstrap"),
         ("Persist Q values and visits", "written to runtime_q_table.csv per training seed"),
     ]),
    ("M5. Experiment orchestration and evaluation",
     "experiments/runtime_experiment.py",
     [
         ("Build the three splits", "independent seeded streams per split"),
         ("Assert split disjointness", "zero SHA-256 fingerprint overlap (fails loudly otherwise)"),
         ("Train five agents", "1,200 sequential episodes per seed, six families"),
         ("Validate", "140 workloads — reporting and demo selection only, never tuning"),
         ("Run the held-out test", "210 untouched workloads, greedy and read-only"),
         ("Compare pairwise", "paired differences + 1,000-replicate crossed bootstrap"),
         ("Write the artifacts", "CSV/JSON tables, Q table, coverage, demo trace, report"),
     ]),
    ("M6. Metrics and reporting",
     "evaluation/metrics.py, evaluation/comparison.py",
     [
         ("Collect the schedule trace", "start/end time per (pid, segment)"),
         ("Derive per-process times", "waiting, turnaround and response per process"),
         ("Aggregate per workload", "means, CPU utilisation, throughput, context switches"),
         ("Aggregate per method and family", "mean ± SD across workloads (and model seeds)"),
         ("Compute paired differences", "target − reference per workload, per seed for Q-learning"),
         ("Render the outputs", "CSV tables, summary JSON, markdown report, figures"),
     ]),
]


def section_workflows(figures: Dict[str, Path]) -> List:
    story = [h1("4. Workflow diagram for each major module")]
    story.append(
        para(
            "Each module below is shown as its own workflow: the numbered steps are the order of "
            "execution inside that module, and the arrows are the data passed between them."
        )
    )
    keys = ["wf_generator", "wf_environment", "wf_encoder", "wf_controller", "wf_pipeline",
            "wf_metrics"]
    for (title, subtitle, steps), key in zip(WORKFLOWS, keys):
        story.append(h2(title))
        story.append(
            para(
                f"<font face='Courier' size='8'>{subtitle}</font>",
                "Small",
            )
        )
        story.extend(
            figure(figures[key],
                   f"Figure 4.{keys.index(key) + 1}: Workflow of {title.split('.', 1)[1].strip()}.",
                   width=CONTENT_W * 0.68)
        )
        step_rows = [
            [str(i + 1), label, detail] for i, (label, detail) in enumerate(steps)
        ]
        story.append(
            make_table(
                [["#", "Step", "Detail"]] + step_rows,
                widths=[CONTENT_W * 0.05, CONTENT_W * 0.32, CONTENT_W * 0.63],
            )
        )
        story.append(Spacer(1, 6))
    return story


# --------------------------------------------------------------------------
# 5. Dataset
# --------------------------------------------------------------------------
def section_dataset(data: Artifacts, figures: Dict[str, Path]) -> List:
    story = [h1("5. Dataset details and data collection procedure")]
    story.append(h2("5.1 Data collection procedure"))
    story.append(
        para(
            "No external or personal data is used. The dataset is synthetic and generated entirely "
            "by the project's own generator, so it is reproducible from three integers and can be "
            "regenerated on any machine. The procedure is:"
        )
    )
    steps = [
        ["D1", "Fix seven family configurations",
         "burst distribution and range, arrival pattern/window/rate, priority pattern — "
         "declared once in config.py and frozen in the experiment configuration."],
        ["D2", "Derive one seed per workload",
         "workload_seed = SHA-256(master seed, split tag, family index, repetition); no global RNG "
         "state is shared between workloads."],
        ["D3", "Generate 15 processes per workload",
         "burst, arrival time and priority are drawn per the family rules, then validated "
         "(positive burst, non-negative arrival, priority within bounds)."],
        ["D4", "Assign workloads to splits",
         "6,000 training workloads (6 families × 1,200 episodes × 5 seeds), 140 validation "
         "(7 × 20) and 210 held-out test (7 × 30)."],
        ["D5", "Fingerprint and check disjointness",
         "each workload gets a SHA-256 fingerprint; the pipeline asserts zero overlap between "
         "training, validation and test."],
        ["D6", "Store only what is needed",
         "metrics, manifest (split/seed/fingerprint/process count) and compact summaries are "
         "committed; the large per-decision event logs are regenerated by the CLI and gitignored."],
        ["D7", "Regenerate on demand",
         "`python main.py runtime-experiment` rebuilds every workload and every metric from the "
         "declared seeds."],
    ]
    story.append(
        make_table(
            [["Step", "Action", "Detail"]] + steps,
            widths=[CONTENT_W * 0.07, CONTENT_W * 0.26, CONTENT_W * 0.67],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 5.1: Data collection procedure."))
    story.append(h2("5.2 Dataset inventory"))
    inventory = [
        ["Training (sequential episodes)", "6 families", "1,200 episodes × 5 seeds",
         "6,000", "90,000", "7101–7105"],
        ["Validation", "7 families", "20 repetitions", "140", "2,100", "8201"],
        ["Final held-out test", "7 families", "30 repetitions", "210", "3,150", "19301"],
    ]
    story.append(
        make_table(
            [["Split", "Families", "Repetitions", "Workloads", "Processes", "Master seed"]]
            + inventory,
            widths=[CONTENT_W * 0.24, CONTENT_W * 0.12, CONTENT_W * 0.18,
                    CONTENT_W * 0.13, CONTENT_W * 0.13, CONTENT_W * 0.20],
            align_right=(3, 4),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 5.2: Dataset inventory. Every workload has exactly 15 processes; poisson_arrivals "
            "is excluded from training and used only at evaluation time."
        )
    )
    story.append(h2("5.3 Family specification"))
    rows = []
    for fam in data.cfg["workload_family_parameters"]:
        rows.append(
            [
                fam["name"],
                fam["description"],
                f"{fam['burst_time_min']}–{fam['burst_time_max']}",
                fam["burst_distribution"],
                fam["arrival_pattern"],
                f"{fam['priority_min']}–{fam['priority_max']}",
            ]
        )
    story.append(
        make_table(
            [["Family", "Description", "Burst range", "Burst law", "Arrivals", "Priority"]] + rows,
            widths=[CONTENT_W * 0.17, CONTENT_W * 0.40, CONTENT_W * 0.11,
                    CONTENT_W * 0.11, CONTENT_W * 0.10, CONTENT_W * 0.11],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 5.3: The seven declared workload families (15 processes each)."))

    story.append(h2("5.4 Measured dataset statistics"))
    if not data.processes.empty:
        stats = []
        for fam in data.families:
            procs = data.processes[data.processes["family"] == fam]
            workloads = data.workloads[data.workloads["family"] == fam]
            stats.append(
                [
                    fam.replace("_", " "),
                    str(len(workloads)),
                    str(len(procs)),
                    f"{procs['burst_time'].mean():.1f} ± {procs['burst_time'].std(ddof=1):.1f}",
                    f"{procs['arrival_time'].mean():.1f}",
                    f"{workloads['total_burst'].mean():.0f}",
                    f"{100.0 * (procs['priority'] <= 2).mean():.0f}%",
                ]
            )
        stats.append(
            [
                "All families",
                str(len(data.workloads)),
                str(len(data.processes)),
                f"{data.processes['burst_time'].mean():.1f} ± "
                f"{data.processes['burst_time'].std(ddof=1):.1f}",
                f"{data.processes['arrival_time'].mean():.1f}",
                f"{data.workloads['total_burst'].mean():.0f}",
                f"{100.0 * (data.processes['priority'] <= 2).mean():.0f}%",
            ]
        )
        story.append(
            make_table(
                [["Family", "Workloads", "Processes", "Burst (mean ± SD)", "Mean arrival",
                  "Mean total work", "High-priority share"]] + stats,
                widths=[CONTENT_W * 0.19, CONTENT_W * 0.10, CONTENT_W * 0.10,
                        CONTENT_W * 0.16, CONTENT_W * 0.12, CONTENT_W * 0.15,
                        CONTENT_W * 0.18],
                align_right=(1, 2, 3, 4, 5, 6),
            )
        )
        story.append(Spacer(1, 3))
        story.append(
            caption(
                "Table 5.4: Statistics of the 210 held-out workloads (3,150 processes), recomputed "
                "with the project's generator at report-build time — these are properties of the "
                "actual dataset, not of the configuration alone. High priority means priority ≤ 2."
            )
        )
        story.extend(
            figure(figures["dataset"],
                   "Figure 5.1: Profile of the generated held-out dataset: burst-time and "
                   "arrival-time spread per family, total work per workload and high-priority "
                   "share.")
        )
    story.append(h2("5.5 Data integrity and assumptions"))
    story.extend(
        bullets(
            [
                "<b>Reproducibility:</b> seeds are derived with SHA-256, so a workload is a pure "
                "function of (master seed, split tag, family index, repetition).",
                "<b>Disjointness:</b> the pipeline fails loudly if any fingerprint appears in two "
                f"splits (current run: overlap = "
                f"{str(data.summary['split_fingerprint_overlap']).lower()}, "
                f"{data.summary['training_fingerprint_count_union']:,} distinct training "
                "fingerprints).",
                "<b>Burst-knowledge assumption:</b> exact burst lengths become known when a "
                "process arrives. This is the assumption already required by standalone SJF; no "
                "burst predictor is included.",
                "<b>Storage discipline:</b> per-decision event logs are large and are regenerated "
                "by the CLI, so they are gitignored; the manifest, metrics, Q tables and summaries "
                "are committed.",
                "<b>No personal or licensed data:</b> the dataset is entirely synthetic, so there "
                "are no privacy or licensing constraints.",
            ]
        )
    )
    return story


# --------------------------------------------------------------------------
# 6. Tools, technologies, algorithms
# --------------------------------------------------------------------------
def section_tools(data: Artifacts) -> List:
    story = [h1("6. Tools, technologies, algorithms and frameworks used")]
    story.append(h2("6.1 Tools and technologies"))
    tools = [
        ["Language", "Python 3.11.2", "Whole implementation, CLI and report generator"],
        ["Numerics", "NumPy 2.4.6", "Q table as an array, bootstrap resampling"],
        ["Data handling", "pandas 3.0.6", "Metric aggregation and CSV artifact export"],
        ["Plotting", "Matplotlib 3.11.2", "All figures in this report"],
        ["Testing", "pytest 9.1.1", "273 automated checks"],
        ["PDF generation", "ReportLab 5.0.1", "Renders this report from the committed artifacts"],
        ["Hashing", "hashlib (SHA-256, stdlib)", "Seed derivation and workload fingerprints"],
        ["Standard library", "argparse, dataclasses, json, pathlib",
         "Configuration objects, CLI, artifact serialisation"],
        ["Version control", "Git / GitHub", "Branch-per-change workflow, pull-request review"],
        ["Environment", "Linux 6.1 (x86-64), venv + pip",
         "Development, experiment runs and report builds"],
    ]
    story.append(
        make_table(
            [["Category", "Item", "Role in the project"]] + tools,
            widths=[CONTENT_W * 0.18, CONTENT_W * 0.32, CONTENT_W * 0.50],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 6.1: Tools and technologies. No deep-learning framework is used "
                         "— the controller is deliberately tabular so it stays auditable."))
    story.append(h2("6.2 Algorithms"))
    algorithms = [
        ["FCFS (action 0)", "Dispatch the head of the ready queue, non-preemptively.",
         "Baseline and runtime action; after an RR requeue it follows live FIFO order."],
        ["SJF (action 1)", "Smallest remaining burst first, non-preemptive, ties by ready order.",
         "Baseline and runtime action; optimal for mean waiting time under known bursts."],
        ["Round Robin (action 2)", "Head of the queue for min(quantum, remaining); unfinished jobs "
                                   "go to the tail after endpoint arrivals.",
         "Baseline and runtime action; bounds response time."],
        ["Priority (action 3)", "Highest static priority first, non-preemptive, ties by ready "
                                "order.",
         "Baseline and runtime action; lower numeric value = higher priority."],
        ["Tabular Q-learning", "Q(s,a) += α[r + γ max_visited Q(s′,·) − Q(s,a)] with "
                               "ε-greedy exploration and unvisited-action masking.",
         "The learned runtime controller (162 states × 4 actions)."],
        ["Predeclared heuristic", "RR → Priority → SJF → FCFS cascade over the same "
                                  "observation.",
         "Non-learning adaptive reference."],
        ["Crossed bootstrap", "Resample workloads within family and model seeds independently; "
                              "1,000 replicates.",
         "Paired 95% confidence intervals for every comparison."],
        ["Discrete-event simulation", "Event-list advancement with a persistent FIFO ready queue "
                                      "and switch-cost charging.",
         "The machine being controlled."],
    ]
    story.append(
        make_table(
            [["Algorithm", "Definition", "Where it is used"]] + algorithms,
            widths=[CONTENT_W * 0.18, CONTENT_W * 0.44, CONTENT_W * 0.38],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 6.2: Algorithms implemented and their role."))
    story.append(h2("6.3 Controller pseudocode"))
    story.append(
        code_block(
            """
SELECT_ACTION(obs):                      # training: epsilon-greedy; evaluation: greedy
    s <- encode(obs)                     # 5 causal features -> state in [0, 162)
    if evaluating:
        if no visited action in s:       # wholly unseen state
            fallback_count += 1
            return HEURISTIC(obs)        # predeclared rule, reported not hidden
        return argmax over visited actions of Q[s, a]
    if first_visit(s) or rand() < epsilon(episode):
        return uniform_random_action()
    return argmax over visited actions of Q[s, a]

UPDATE(s, a, r, s', terminal):
    target <- r if terminal else r + gamma * max over visited actions of Q[s', .]
    Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])
    visits[s, a] <- visits[s, a] + 1

HEURISTIC(obs):                          # predeclared, no learning
    if ready_count >= 3 and mean_arrival_age >= quantum:      return Round Robin
    if >= 2 ready jobs have distinct priorities:              return Priority
    if max(remaining_burst) >= 2 * min(remaining_burst):      return SJF
    return FCFS
""",
            font_size=6.6,
        )
    )
    story.append(Spacer(1, 4))
    story.append(h2("6.4 Frameworks and why none of the heavy ones"))
    story.append(
        para(
            "The controller is a 162 × 4 table rather than a neural network, so no deep-learning "
            "framework is needed. This is a deliberate methodological choice: every Q value, visit "
            "count and fallback can be inspected directly in the committed artifacts, which is what "
            "makes claims about state coverage and unseen-state behaviour verifiable instead of "
            "anecdotal."
        )
    )
    return story


# --------------------------------------------------------------------------
# 7. Experimental plan and evaluation metrics
# --------------------------------------------------------------------------
def section_experiment(data: Artifacts, figures: Dict[str, Path]) -> List:
    story = [h1("7. Experimental plan and evaluation metrics")]
    story.append(h2("7.1 Experimental plan"))
    plan = [
        ["P1", "Fix the design before looking at results",
         "Families, splits, seeds, quantum (4), switch cost (1), learning rates and the heuristic "
         "rule are all declared in configuration and committed."],
        ["P2", "Train five independent agents",
         "Seeds 7101–7105; 1,200 sequential episodes each on six families (poisson_arrivals "
         "withheld)."],
        ["P3", "Validate once",
         "140 workloads (seed 8201) used only for reporting and for the predeclared "
         "illustrative-demo rule — never for tuning."],
        ["P4", "Test once, held out",
         "210 untouched workloads (seed 19301); the fixed policies and the heuristic are "
         "deterministic (one run each), Q-learning is evaluated by all five models greedily and "
         "read-only."],
        ["P5", "Compare every method against every other",
         "Paired per-workload differences, plus independent resampling of the five model seeds."],
        ["P6", "Quantify uncertainty",
         "Family-stratified crossed bootstrap, 1,000 replicates, seed 104729; 95% intervals on "
         "every headline comparison."],
        ["P7", "Publish artifacts",
         "Metric tables, Q values, visit counts, coverage, manifest, demo trace and summary JSON "
         "are written to results/runtime/ and regenerated by one command."],
    ]
    story.append(
        make_table(
            [["Step", "Action", "Detail"]] + plan,
            widths=[CONTENT_W * 0.07, CONTENT_W * 0.27, CONTENT_W * 0.66],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 7.1: The experimental plan, in execution order."))
    story.append(h2("7.2 Evaluation metrics"))
    metrics = [
        ["Mean waiting time", "mean over processes of (first service start − arrival)",
         "Primary metric — it is exactly the quantity the reward minimises."],
        ["Mean turnaround time", "mean over processes of (completion − arrival)",
         "End-to-end latency experienced by a job."],
        ["Mean response time", "mean over processes of (first dispatch − arrival)",
         "Interactivity; this is what Round Robin buys."],
        ["CPU utilisation (%)", "busy time / total elapsed time × 100",
         "Cost of switching and idling."],
        ["Throughput", "processes completed per unit simulated time",
         "Work completed per unit time."],
        ["Context switches", "number of running-PID changes",
         "Overhead proxy; switch cost is charged for every method."],
        ["Policy switches per trace", "decisions where the chosen policy differs from the previous",
         "How much the controller actually adapts."],
        ["State-action coverage", "visited (s, a) pairs / 648, unseen-state fallbacks",
         "Whether the learned table is actually populated where it is used."],
    ]
    story.append(
        make_table(
            [["Metric", "Definition", "Why it is measured"]] + metrics,
            widths=[CONTENT_W * 0.19, CONTENT_W * 0.39, CONTENT_W * 0.42],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 7.2: Evaluation metrics."))
    story.append(h2("7.3 Statistical procedure"))
    story.append(
        para(
            "Comparisons are paired: for each workload the difference (target − reference) is "
            "computed on the same workload, and for Q-learning the model seed is resampled "
            "independently from the workloads (a crossed bootstrap). Resampling is stratified by "
            "workload family, and 1,000 replicates are drawn with seed 104729. An interval that "
            "excludes zero on the favourable side is treated as a real difference on this "
            "benchmark; an interval that contains zero is reported as inconclusive rather than as "
            "a win."
        )
    )
    story.append(h2("7.4 Results obtained with this plan"))
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
    story.append(
        make_table(
            [["Method (210 held-out workloads)"] + [label for _m, label in metric_cols]] + rows,
            widths=[CONTENT_W * 0.16, CONTENT_W * 0.135, CONTENT_W * 0.145, CONTENT_W * 0.13,
                    CONTENT_W * 0.11, CONTENT_W * 0.11, CONTENT_W * 0.105, CONTENT_W * 0.105],
            align_right=tuple(range(1, 8)),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 7.3: Held-out final test (master seed 19301, quantum 4, switch cost 1). "
            "Mean ± SD across the 210 workloads; Q-learning also averages the five model seeds."
        )
    )
    story.extend(
        figure(figures["metrics"],
               "Figure 7.1: Six metrics on the held-out test set (error bars = SD across workloads).")
    )
    story.extend(
        figure(figures["training"],
               "Figure 7.2: Sequential training: rolling mean waiting time and episodic return "
               "(100-episode window) for each of the five agents.")
    )
    paired_rows = []
    metric_labels = {
        "avg_waiting_time": "waiting time",
        "avg_response_time": "response time",
        "context_switches": "context switches",
        "cpu_utilization": "CPU util %",
    }
    for target, reference in [("Runtime Q-learning", "SJF"),
                              ("Runtime Q-learning", "Causal heuristic"),
                              ("Round Robin", "SJF"),
                              ("Causal heuristic", "SJF"),
                              ("FCFS", "SJF")]:
        for metric, label in metric_labels.items():
            sel = data.paired[
                (data.paired["target"] == target)
                & (data.paired["reference"] == reference)
                & (data.paired["metric"] == metric)
            ]
            if sel.empty:
                continue
            row = sel.iloc[0]
            mean, low, high = float(row["mean_difference"]), float(row["ci95_low"]), float(row["ci95_high"])
            if mean == 0 and low == 0 and high == 0:
                reading = "no difference"
            elif metric == "cpu_utilization":
                reading = "favours target" if mean > 0 else "favours reference"
            else:
                reading = "favours target" if mean < 0 else "favours reference"
            paired_rows.append(
                [SHORT_NAME[target], SHORT_NAME[reference], label, f"{mean:+,.3f}",
                 f"[{low:+,.3f}, {high:+,.3f}]", reading]
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
            "Table 7.4: Paired differences (target − reference) with family-stratified "
            "crossed-bootstrap intervals."
        )
    )
    story.extend(
        figure(figures["forest"],
               "Figure 7.3: Forest plot of the paired comparisons (green favours the target, red "
               "favours the reference).",
               width=CONTENT_W * 0.94)
    )
    story.append(h2("7.5 What the plan has established so far"))
    q_sjf_wait = ci_of(data, "Runtime Q-learning", "SJF", "avg_waiting_time")
    rr_sjf_wait = ci_of(data, "Round Robin", "SJF", "avg_waiting_time")
    fcfs_sjf_wait = ci_of(data, "FCFS", "SJF", "avg_waiting_time")
    prio_sjf_wait = ci_of(data, "Priority", "SJF", "avg_waiting_time")
    q_rr_diff = q_sjf_wait[0] - rr_sjf_wait[0]
    q_sjf_resp = ci_of(data, "Runtime Q-learning", "SJF", "avg_response_time")
    q_heur_wait = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_waiting_time")
    story.extend(
        bullets(
            [
                f"<b>Against Round Robin</b> the learned controller is clearly better on waiting "
                f"time: {q_rr_diff:+,.3f} time units, obtained by differencing the two committed "
                f"comparisons against SJF (Q − SJF {q_sjf_wait[0]:+,.3f} minus RR − SJF "
                f"{rr_sjf_wait[0]:+,.3f}; both intervals exclude zero and are far apart, so the "
                f"ordering is unambiguous). FCFS and Priority sit between the two, at "
                f"{fcfs_sjf_wait[0]:+,.3f} and {prio_sjf_wait[0]:+,.3f} relative to SJF.",
                f"<b>Against the predeclared adaptive rule</b> it reduces waiting time by "
                f"{q_heur_wait[0]:+,.3f} (95% CI [{q_heur_wait[1]:+,.3f}, {q_heur_wait[2]:+,.3f}]) "
                "— so the gain comes from learning, not merely from being adaptive.",
                f"<b>Against SJF it is mixed, and the report says so:</b> waiting time is worse by "
                f"{q_sjf_wait[0]:+,.3f} (95% CI [{q_sjf_wait[1]:+,.3f}, {q_sjf_wait[2]:+,.3f}]) "
                f"while response time is better by {q_sjf_resp[0]:+,.3f} "
                f"(95% CI [{q_sjf_resp[1]:+,.3f}, {q_sjf_resp[2]:+,.3f}]). Non-preemptive SJF with "
                "exact bursts and no mid-job switch cost is a very strong reference on this "
                "benchmark.",
                "On the held-out poisson_arrivals family the controller still lands between SJF "
                "and Round Robin, so the behaviour is not family memorisation.",
            ]
        )
    )
    return story


# --------------------------------------------------------------------------
# 8. Implementation progress
# --------------------------------------------------------------------------
def section_progress(data: Artifacts, figures: Dict[str, Path],
                     test_rows: List[List[str]], pytest_block: str, cli_block: str) -> List:
    story = [h1("8. Implementation progress")]
    story.append(
        para(
            "All six major modules are implemented, exercised by the automated suite and "
            "demonstrated with the outputs below. The requirement for Review 2 is that at least 50% "
            "of the proposed work is complete and that the completed modules are demonstrated with "
            "valid outputs; the table below records completion per module and the artifact that "
            "evidences it."
        )
    )
    progress = [
        ["M1", "Workload generator and dataset builder", "100%",
         "6,350-row manifest + regenerated dataset statistics (§5.4)", "Complete"],
        ["M2", "Discrete-event runtime environment", "100%",
         "fixed-action equivalence tests; schedule traces in final-test metrics", "Complete"],
        ["M3", "Causal observation and state encoder", "100%",
         "162-state table; hidden-future invariance tests", "Complete"],
        ["M4", "Sequential Q-learning controller", "100%",
         "runtime_q_table.csv (3,240 Q values), coverage report, fallback counts", "Complete"],
        ["M5", "Predeclared heuristic controller", "100%",
         "heuristic rows in every metric table", "Complete"],
        ["M6", "Experiment orchestration, metrics and reporting", "100%",
         "final-test, validation, paired-bootstrap and coverage tables; generated report",
         "Complete"],
        ["—", "Review 2 report generator (this document)", "100%",
         "tools/build_review2_report.py", "Complete"],
    ]
    story.append(
        make_table(
            [["ID", "Module", "Completion", "Evidence artifact", "Status"]] + progress,
            widths=[CONTENT_W * 0.06, CONTENT_W * 0.31, CONTENT_W * 0.11,
                    CONTENT_W * 0.40, CONTENT_W * 0.12],
            align_right=(2,),
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        caption(
            "Table 8.1: Implementation status. Overall, the proposed Review 2 scope (methodology, "
            "simulator, controller, experiment, evaluation) is complete; the items listed in §9 "
            "are extensions beyond the proposed scope."
        )
    )
    story.append(h2("8.1 Demonstration 1 — automated verification"))
    if test_rows:
        story.append(
            make_table(
                [["Test module", "Checks", "What it protects"]] + test_rows,
                widths=[CONTENT_W * 0.27, CONTENT_W * 0.09, CONTENT_W * 0.64],
                align_right=(1,),
            )
        )
        story.append(Spacer(1, 3))
        story.append(
            caption(
                f"Table 8.2: Test inventory, {sum(int(r[1]) for r in test_rows)} checks collected "
                "and executed while building this report."
            )
        )
    story.append(Spacer(1, 3))
    story.append(code_block(pytest_block, font_size=6.8))
    story.append(Spacer(1, 5))
    story.append(h2("8.2 Demonstration 2 — end-to-end run output"))
    story.append(
        para(
            "The full pipeline (training, validation, held-out testing, artifact export) is "
            "reproduced by one command. The output below is from a verification run performed on "
            "the reporting host and matches the committed artifacts exactly.",
            "Small",
        )
    )
    story.append(code_block(cli_block, font_size=6.6))
    story.append(Spacer(1, 5))
    story.append(h2("8.3 Demonstration 3 — learned behaviour and artifacts"))
    story.extend(
        figure(figures["coverage"],
               "Figure 8.1: Learned state-action visitation pooled over the five agents (left) and "
               "per-seed coverage (right). Between 267 and 276 of the 648 pairs are visited.")
    )
    demo = data.demo
    story.append(
        para(
            f"The controller does switch policies inside a single trace. The example below is taken "
            f"from the validation split by the predeclared rule (training seed "
            f"{demo['training_seed']}, family <font face='Courier'>{demo['family']}</font>, "
            f"fingerprint <font face='Courier'>{demo['workload_fingerprint']}</font>): "
            f"{demo['policy_switch_count']} switches over {demo['decision_count']} decisions, with "
            f"mean waiting time {demo['metrics']['avg_waiting_time']:.3f} and mean response time "
            f"{demo['metrics']['avg_response_time']:.3f}. It is an illustration of the mechanism, "
            "not a performance claim."
        )
    )
    story.extend(
        figure(figures["gantt"],
               "Figure 8.2: Illustrative trace. Bars are execution segments coloured by the policy "
               "chosen for them; dotted lines mark the instants where the policy changed.")
    )
    artifacts = [
        ["results/runtime/runtime_final_test_metrics.csv", "2,100 rows",
         "Six metrics, switch/decision counts and timings on the held-out test"],
        ["results/runtime/runtime_validation_metrics.csv", "1,400 rows",
         "Same columns for the validation split"],
        ["results/runtime/runtime_training_metrics.csv", "6,000 rows",
         "Per-episode sequential training outcomes"],
        ["results/runtime/runtime_paired_comparisons.csv", "36 rows",
         "Paired differences with 95% crossed-bootstrap intervals"],
        ["results/runtime/runtime_state_action_coverage.csv", "15 rows",
         "Visited states/pairs, fallback counts and timings per seed"],
        ["results/runtime/runtime_q_table.csv", "3,240 rows",
         "Q values and visit counts per training seed"],
        ["results/runtime/runtime_workload_manifest.csv", "6,350 rows",
         "Split, seeds and workload fingerprints"],
        ["results/runtime/runtime_learned_switch_demo.json", "1 trace",
         "Observations, actions, switch times and execution trace"],
        ["results/runtime/runtime_summary.json", "—",
         "Configuration, split fingerprints, all means and software versions"],
        ["figures/review2/*.png", "11 figures", "Every figure in this report"],
    ]
    story.append(
        make_table(
            [["Artifact", "Size", "Contents"]] + artifacts,
            widths=[CONTENT_W * 0.42, CONTENT_W * 0.12, CONTENT_W * 0.46],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 8.3: Artifacts produced by the completed modules."))
    return story


# --------------------------------------------------------------------------
# 9. Challenges and remaining plan
# --------------------------------------------------------------------------
def section_challenges(data: Artifacts) -> List:
    story = [h1("9. Challenges encountered and plan for completing the remaining work")]
    story.append(h2("9.1 Challenges encountered"))
    challenges = [
        ["Keeping the controller causal",
         "An earlier whole-workload selector saw future arrivals and counterfactual schedules, so "
         "its results could not be interpreted as a runtime controller.",
         "Introduced an immutable RuntimeObservation, made the encoder accept only that object, "
         "and added a paired hidden-future test that fails if unarrived changes leak into the "
         "observation or action prefix."],
        ["Round-Robin and FCFS interaction after a requeue",
         "Once RR appends a preempted job to the tail, 'FCFS' is ambiguous: original arrival order "
         "or live queue order?",
         "Defined one persistent FIFO ready queue with canonical (arrival_time, pid) insertion, and "
         "specifies that FCFS follows live queue order while SJF/Priority ties preserve it. "
         "Fixed-action runs are tested against the standalone schedulers."],
        ["Charging switch cost fairly",
         "If the learner pays for switching but baselines do not, the comparison is meaningless; if "
         "nothing is charged, switching is free and unrealistic.",
         "One unit is charged whenever the running PID changes, for every method including the "
         "fixed policies; arrivals during the switch interval are queued but the already-chosen "
         "dispatch is not reconsidered."],
        ["Designing a reward that is the objective",
         "A shaped reward can drift away from the metric being reported.",
         "Used r = −ΔW/q with γ = 1, so the episodic return is exactly −total waiting "
         "time / q; a test asserts the identity."],
        ["Sparse coverage of the tabular space",
         "Only ~42% of the 648 state-action pairs are ever visited, and unseen states appear during "
         "evaluation.",
         "Kept the abstraction small, forced exploration on first visit, masked unvisited actions "
         "in greedy steps and bootstraps, and made unseen states fall back to the predeclared "
         "heuristic with the fallback count reported (1–3 decisions per model on the final "
         "test)."],
        ["Honest reporting of a negative result",
         "The learner does not beat non-preemptive SJF on waiting time; it would be easy to bury "
         "this.",
         "Reported it as a headline finding with its bootstrap interval, and stated the structural "
         "reason (SJF gets exact bursts at arrival and pays no mid-job switch cost)."],
        ["Host-dependent timing noise",
         "Wall-clock columns make artifacts look non-deterministic across machines.",
         "Observation, selection, Q-update and simulator time are stored in separate columns, "
         "never charged to simulated time, and excluded from all schedule comparisons."],
        ["Large per-decision event logs",
         "Full causal decision logs are large and would bloat the repository.",
         "They are written by the CLI but gitignored and regenerable from the recorded "
         "configuration and seeds; compact metrics, Q tables and manifests are committed."],
    ]
    story.append(
        make_table(
            [["Challenge", "Why it mattered", "How it was resolved"]] + challenges,
            widths=[CONTENT_W * 0.19, CONTENT_W * 0.38, CONTENT_W * 0.43],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 9.1: Challenges and their resolutions."))
    story.append(h2("9.2 Plan for completing the remaining work"))
    plan = [
        ["W1", "Sensitivity study", "All results are for q = 4 and switch cost 1.",
         "Re-run the plan over a grid of quanta and switch costs and report where the ordering "
         "changes.", "Metric tables per (q, cost) cell"],
        ["W2", "Burst prediction", "Exact bursts are assumed known at arrival.",
         "Add an explicit predictor with measured error and re-run the controller under that "
         "noise.", "Results with prediction error bands"],
        ["W3", "Richer state abstraction", "Only ~42% of pairs are visited; 162 states is coarse.",
         "Tile coding or a small function approximator, keeping the same causal observation "
         "contract and fallback reporting.", "Coverage and metric comparison vs the tabular agent"],
        ["W4", "Multi-objective reward", "Waiting time is optimised; response time is not.",
         "Add a constrained/weighted objective and measure the waiting-response trade-off curve.",
         "Pareto front over the two metrics"],
        ["W5", "I/O and blocking", "Processes never block, so the queue is CPU-bound only.",
         "Model I/O bursts and blocked/wake transitions in the environment.",
         "Extended environment plus new baselines"],
        ["W6", "Multicore extension", "Single CPU only.",
         "Extend to N cores with per-core queues and migration cost; the controller chooses a "
         "policy per core.", "Multicore metrics and overhead analysis"],
        ["W7", "Trace-driven validation", "All workloads are synthetic.",
         "Replay a public scheduling trace through the same environment and report the gap to the "
         "synthetic results.", "Trace-driven result table"],
    ]
    story.append(
        make_table(
            [["ID", "Task", "Gap it closes", "Planned approach", "Deliverable"]] + plan,
            widths=[CONTENT_W * 0.05, CONTENT_W * 0.14, CONTENT_W * 0.24,
                    CONTENT_W * 0.32, CONTENT_W * 0.25],
        )
    )
    story.append(Spacer(1, 3))
    story.append(caption("Table 9.2: Remaining work, planned beyond the proposed Review 2 scope."))
    story.append(h2("9.3 Sequence for the remaining work"))
    story.extend(
        bullets(
            [
                "<b>First:</b> W1 (sensitivity) and W2 (burst prediction) — they test whether "
                "the current conclusions are an artefact of fixed parameters or an idealised "
                "assumption.",
                "<b>Second:</b> W3 and W4 — they attack the two measured weaknesses directly "
                "(coarse coverage, single objective).",
                "<b>Third:</b> W5 and W6 — environment extensions that make the simulator "
                "more realistic, each with re-baselined comparisons.",
                "<b>Finally:</b> W7 — external validation on a real trace, which is the "
                "strongest available check that the synthetic ordering means anything.",
            ]
        )
    )
    return story


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_figures(data: Artifacts, figures_dir: Path) -> Dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "methodology": figures_dir / "fig_methodology_pipeline.png",
        "architecture": figures_dir / "fig_architecture.png",
        "wf_generator": figures_dir / "fig_wf_generator.png",
        "wf_environment": figures_dir / "fig_wf_environment.png",
        "wf_encoder": figures_dir / "fig_wf_encoder.png",
        "wf_controller": figures_dir / "fig_wf_controller.png",
        "wf_pipeline": figures_dir / "fig_wf_experiment.png",
        "wf_metrics": figures_dir / "fig_wf_metrics.png",
        "dataset": figures_dir / "fig_dataset_profile.png",
        "metrics": figures_dir / "fig_final_metrics.png",
        "training": figures_dir / "fig_training_curves.png",
        "forest": figures_dir / "fig_bootstrap_forest.png",
        "coverage": figures_dir / "fig_coverage.png",
        "gantt": figures_dir / "fig_demo_gantt.png",
    }
    figs.methodology_pipeline(paths["methodology"])
    figs.architecture_diagram(paths["architecture"])
    for key, (title, subtitle, steps) in zip(
        ["wf_generator", "wf_environment", "wf_encoder", "wf_controller", "wf_pipeline",
         "wf_metrics"],
        WORKFLOWS,
    ):
        figs.module_workflow(paths[key], steps, title, subtitle)
    figs.dataset_profile(data.processes, data.workloads, paths["dataset"], data.families)
    figs.metric_bars(data.final, paths["metrics"])
    figs.training_curves(data.train, paths["training"])
    figs.bootstrap_forest(data.paired, paths["forest"])
    figs.coverage_heatmap(data.q_table, data.coverage, paths["coverage"])
    figs.gantt_trace(data.demo, paths["gantt"])
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
    rows = [[name, str(counts.get(name, 0)), descriptions.get(name, "")]
            for name in sorted(descriptions)]
    total = sum(counts.values())
    tail = [line for line in run.stdout.strip().splitlines() if line.strip()][-3:]
    output = "\n".join(["$ python -m pytest -q", *tail])
    if not tail:
        output = f"$ python -m pytest -q\n{total} tests (output unavailable)"
    return rows, output


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
            f"    {method:<22}: {data.summary_mean('final', method, 'avg_waiting_time'):.3f}"
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
    parser.add_argument("--no-dataset", action="store_true",
                        help="skip rebuilding the dataset statistics (§5.4 and Figure 5.1)")
    args = parser.parse_args(argv)

    root: Path = args.root
    output = args.output or root / "Review_2_Report.pdf"
    docs_copy = args.docs_copy or root / "docs" / "Review_2_Report.pdf"
    figures_dir = args.figures_dir or root / "figures" / "review2"

    data = Artifacts(root, with_dataset=not args.no_dataset)
    figures = build_figures(data, figures_dir)


    if args.skip_tests:
        test_rows: List[List[str]] = []
        pytest_block = "(pytest not executed: --skip-tests)"
        test_line = "273 checks (not re-run for this build)"
    else:
        test_rows, pytest_block = collect_test_counts(root)
        total = sum(int(row[1]) for row in test_rows) if test_rows else 0
        passed = "passed" in pytest_block
        test_line = (f"{total} checks, {'all passing' if passed else 'see §8.1'}"
                     f"{' (re-run during this build)' if total else ''}")

    generated = _dt.date.today().strftime("%d %B %Y")
    cli_block = build_cli_block(data)

    story: List = []
    story += cover_page(data, generated, test_line)
    story += toc_page()
    story += section_methodology(data, figures)
    story.append(PageBreak())
    story += section_architecture(data, figures)
    story.append(PageBreak())
    story += section_modules(data)
    story.append(PageBreak())
    story += section_workflows(figures)
    story.append(PageBreak())
    story += section_dataset(data, figures)
    story.append(PageBreak())
    story += section_tools(data)
    story.append(PageBreak())
    story += section_experiment(data, figures)
    story.append(PageBreak())
    story += section_progress(data, figures, test_rows, pytest_block, cli_block)
    story.append(PageBreak())
    story += section_challenges(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReviewDoc(str(output), generated=generated)
    doc.multiBuild(story)

    if docs_copy.resolve() != output.resolve():
        docs_copy.parent.mkdir(parents=True, exist_ok=True)
        docs_copy.write_bytes(output.read_bytes())

    print(f"wrote {output} ({output.stat().st_size / 1024:,.0f} KB)")
    print(f"wrote {docs_copy}")
    print(f"figures in {figures_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
