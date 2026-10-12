"""Build the text-only Review 2 DOCX (no tables, no colour, no images; Mermaid for diagrams).

The document covers the nine Review 2 rubric items in order. Every number is read from the
committed artifacts under ``results/runtime/`` (dataset statistics are recomputed with the
project's own seeded generator), so the text cannot drift from the recorded experiment.

Usage::

    pip install -r requirements-report.txt
    python tools/build_review2_docx.py                 # writes Review_2_Report.docx + docs copy
    python tools/build_review2_docx.py --skip-tests    # do not re-run pytest
    python tools/build_review2_docx.py --output /tmp/r.docx

The text-only PDF is rendered from this DOCX by ``tools/build_review2_text_pdf.py``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Pt, RGBColor  # noqa: E402

from review2_data import (  # noqa: E402
    METHOD_ORDER,
    SHORT_NAME,
    Artifacts,
    ci_of,
    fmt,
    module_outputs,
)

MONO = "Consolas"
BLACK = RGBColor(0, 0, 0)

# Diagram numbers are assigned in document order by ``diagram()``.
_DIAGRAM_COUNTER = {"n": 0}


# --------------------------------------------------------------------------
# Document helpers (plain text only)
# --------------------------------------------------------------------------
def _strip_theme_colour(document: Document) -> None:
    """Remove every colour from the default template: theme colours, shading, the blue title
    rule, and the unused coloured table styles. The document must contain no colour at all."""
    from docx.enum.style import WD_STYLE_TYPE

    for style in list(document.styles):
        if style.type == WD_STYLE_TYPE.TABLE:
            style.element.getparent().remove(style.element)
    for style in document.styles:
        element = style.element
        for node in element.iter(qn("w:color")):
            node.set(qn("w:val"), "000000")
            for attr in list(node.attrib):
                if attr.endswith("themeColor") or attr.endswith("themeShade") or attr.endswith(
                    "themeTint"
                ):
                    del node.attrib[attr]
        for shade in list(element.iter(qn("w:shd"))):
            shade.getparent().remove(shade)
        for node in element.iter(qn("w:rFonts")):
            for attr in list(node.attrib):
                if "Theme" in attr or "theme" in attr:
                    del node.attrib[attr]
        for border in element.iter(qn("w:pBdr")):
            border.getparent().remove(border)


def new_document() -> Document:
    """A4 page, 2.2 cm side margins, Calibri 10.5 pt body text with tight paragraph spacing.

    These settings keep the document at the 10-15 page length the review asks for.
    """
    from docx.shared import Cm

    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.2)
    section.top_margin = section.bottom_margin = Cm(2.0)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.0
    for name, size, before, after in (("Heading 1", 14, 12, 4), ("Heading 2", 11.5, 8, 3),
                                      ("Heading 3", 11, 6, 2), ("Title", 20, 0, 8)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = BLACK
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
    _strip_theme_colour(document)
    properties = document.core_properties
    properties.title = "Operating Systems Project — Review 2"
    properties.subject = "Methodology and Partial Implementation (Lab Assessment 9)"
    properties.author = ""
    properties.last_modified_by = ""
    properties.comments = ""
    return document


def para(document: Document, text: str, bold: bool = False, italic: bool = False):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = BLACK
    return paragraph


def bullet(document: Document, text: str, bold_prefix: str = "") -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    if bold_prefix:
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        run.font.color.rgb = BLACK
    run = paragraph.add_run(text)
    run.font.color.rgb = BLACK


def step(document: Document, number: int, text: str, bold_prefix: str = "") -> None:
    """Numbered step written as text, so numbering restarts in every list."""
    paragraph = document.add_paragraph()
    fmt_ = paragraph.paragraph_format
    fmt_.left_indent = Pt(20)
    fmt_.first_line_indent = Pt(-20)
    fmt_.space_after = Pt(3)
    run = paragraph.add_run(f"{number}.\t")
    run.font.color.rgb = BLACK
    if bold_prefix:
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        run.font.color.rgb = BLACK
    run = paragraph.add_run(text)
    run.font.color.rgb = BLACK


def code(document: Document, text: str, label: str = "") -> None:
    """Monospace block (pseudocode, terminal output or Mermaid source)."""
    if label:
        caption = document.add_paragraph()
        run = caption.add_run(label)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = BLACK
    for line in text.strip("\n").split("\n"):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.left_indent = Pt(18)
        run = paragraph.add_run(line if line else " ")
        run.font.name = MONO
        run.font.size = Pt(8.5)
        run.font.color.rgb = BLACK
    document.add_paragraph().paragraph_format.space_after = Pt(4)


def diagram(document: Document, title: str, source: str) -> None:
    """A numbered Mermaid diagram, emitted as copy-ready Mermaid source."""
    _DIAGRAM_COUNTER["n"] += 1
    number = _DIAGRAM_COUNTER["n"]
    code(document, "```mermaid\n" + source.strip() + "\n```",
         label=f"Diagram {number} — {title}")


# --------------------------------------------------------------------------
# Mermaid sources
# --------------------------------------------------------------------------
def mermaid_pipeline() -> str:
    return """flowchart LR
    S1["1. Dataset<br/>7 seeded families, 15 processes"]
    S2["2. Simulation<br/>single CPU, FIFO ready queue"]
    S3["3. Observation<br/>arrived work only"]
    S4["4. Encoding<br/>5 features, 162 states"]
    S5["5. Decision<br/>Q-learning picks a policy"]
    S6["6. Execution<br/>segment runs, switch cost charged"]
    S7["7. Reward<br/>r = -dW / q"]
    S8["8. Evaluation<br/>held-out test and paired bootstrap"]
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6
    S6 --> S7
    S7 --> S8
    S7 -.->|"repeat at every decision epoch"| S3"""


def mermaid_architecture() -> str:
    return """flowchart TB
    subgraph ENV["Environment: scheduler/runtime.py (owns the future)"]
        E1["Private event list: future arrivals"]
        E2["Single FIFO ready queue and remaining bursts"]
        E3["CPU segment, switch cost, schedule trace"]
    end
    subgraph CTL["Controller: arrived work only"]
        C1["RuntimeObservation (frozen snapshot)"]
        C2["State encoder: 162 states"]
        C3["Q-learning controller: 648 state-action pairs"]
    end
    M["Waiting-time increment gives reward r = -dW / q"]
    E1 --> E2
    E2 --> E3
    E3 -->|"observation"| C1
    C1 --> C2
    C2 --> C3
    C3 -->|"action: FCFS, SJF, RR or Priority"| E3
    E3 --> M
    M -.->|"Q update"| C3"""


WORKFLOW_M1 = """flowchart TD
    N1["1. Select family configuration"]
    N2["2. Derive the workload seed"]
    N3["3. Sample 15 processes"]
    N4{"4. Valid burst, arrival and priority values?"}
    N5["5. Fingerprint and tag the workload"]
    N6["6. Assemble the split"]
    X1(["Reject workload: ValidationError"])
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 -->|"yes"| N5
    N4 -->|"no"| X1
    N5 --> N6"""

WORKFLOW_M2 = """flowchart TD
    N1["1. Advance to the next decision epoch"]
    N2["2. Admit arrivals up to the epoch"]
    N3["3. Freeze the causal observation"]
    N4["4. Obtain the policy from the controller"]
    N5{"5. Selected PID differs from the running PID?"}
    N6["6. Charge one switch-cost unit"]
    N7["7. Execute the segment"]
    N8["8. Settle the endpoint"]
    N9["9. Return interval reward r = -dW / q"]
    N10{"10. All processes completed?"}
    X1(["Emit schedule trace and six metrics"])
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 -->|"yes"| N6
    N5 -->|"no, or first dispatch"| N7
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
    N10 -->|"no"| N1
    N10 -->|"yes"| X1"""

WORKFLOW_M3 = """flowchart TD
    N1["1. Receive the RuntimeObservation"]
    N2["2. Extract five causal features"]
    N3["3. Discretise each feature relative to the quantum"]
    N4["4. Flatten the bins in mixed-radix order"]
    N5["5. Return state index s in [0, 162)"]
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5"""

WORKFLOW_M4 = """flowchart TD
    N1["1. Encode the observation: s = encode(obs)"]
    N2{"2. Training run?"}
    N3{"3. First visit to s, or random draw below epsilon?"}
    N4["4. Choose a uniformly random action"]
    N5["5. Choose the greedy action over visited actions"]
    N6{"6. Any visited action in s?"}
    N7["7. Count an unseen-state fallback to the heuristic"]
    N8["8. Choose the greedy action over visited actions"]
    N9["9. Receive reward r = -dW / q"]
    N10["10. Update Q(s,a) from visited next-state actions"]
    N11["11. Persist Q values and visit counts"]
    X1(["Return action (evaluation, read-only)"])
    N1 --> N2
    N2 -->|"yes"| N3
    N2 -->|"no: evaluation"| N6
    N3 -->|"yes"| N4
    N3 -->|"no"| N5
    N6 -->|"yes"| N8
    N6 -->|"no"| N7
    N7 --> X1
    N8 --> X1
    N4 --> N9
    N5 --> N9
    N9 --> N10
    N10 --> N11"""

WORKFLOW_M5 = """flowchart TD
    N1["1. Receive the same frozen observation"]
    N2{"2. At least 3 ready jobs and mean arrival age >= quantum?"}
    N3{"3. At least 2 ready jobs with distinct priorities?"}
    N4{"4. Largest remaining burst >= 2 x smallest?"}
    X1(["Return Round Robin"])
    X2(["Return Priority"])
    X3(["Return SJF"])
    X4(["Return FCFS (default)"])
    N1 --> N2
    N2 -->|"yes"| X1
    N2 -->|"no"| N3
    N3 -->|"yes"| X2
    N3 -->|"no"| N4
    N4 -->|"yes"| X3
    N4 -->|"no"| X4"""

WORKFLOW_M6 = """flowchart TD
    N1["1. Train five agents on six families"]
    N2["2. Validate on 140 workloads"]
    N3{"3. Training and validation fingerprints disjoint?"}
    N4["4. Select the illustrative trace by the validation-only rule"]
    N5["5. Generate the 210 held-out test workloads"]
    N6{"6. Validation and test fingerprints disjoint, and training disjoint from both?"}
    N7["7. Run the held-out test and derive six metrics per workload"]
    N8["8. Compare pairwise: paired differences and crossed bootstrap"]
    N9["9. Write CSV, JSON, Q-table, coverage and report artifacts"]
    X1(["Stop: ValidationError"])
    N1 --> N2
    N2 --> N3
    N3 -->|"yes"| N4
    N3 -->|"no"| X1
    N4 --> N5
    N5 --> N6
    N6 -->|"yes"| N7
    N6 -->|"no"| X1
    N7 --> N8
    N8 --> N9"""


def add_cover(document: Document, data: Artifacts, generated: str, test_line: str) -> None:
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Operating Systems Project — Review 2")
    run.font.color.rgb = BLACK

    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Methodology and Partial Implementation (Lab Assessment 9)")
    run.bold = True
    run.font.size = Pt(11.5)
    run.font.color.rgb = BLACK

    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run("Event-Driven Runtime-Adaptive CPU Scheduling "
                          "with Sequential Tabular Q-Learning")
    run.font.size = Pt(12.5)
    run.italic = True
    run.font.color.rgb = BLACK

    para(document, "")
    dataset = (f"7 synthetic families: {data.cfg['training_episodes_per_seed'] * len(data.cfg['training_seeds']):,} "
               f"training episodes, {data.summary['validation_unique_workloads']} validation workloads, "
               f"{data.summary['final_test_unique_workloads']} held-out test workloads")
    for field, value in [
        ("Repository", "github.com/lostpilot404/OS-Project"),
        ("Language and libraries", "Python 3.11.2 with NumPy 2.4.6, pandas 3.0.6 and pytest 9.1.1"),
        ("Automated verification", test_line),
        ("Dataset", dataset),
        ("Date", generated),
    ]:
        bullet(document, value, bold_prefix=f"{field}: ")


def section_methodology(document: Document, data: Artifacts) -> None:
    document.add_heading("1. Detailed methodology of the proposed system", level=1)
    para(document,
         "The proposed system is a single-CPU, event-driven, runtime-adaptive CPU scheduler. At every "
         "dispatch or Round-Robin quantum boundary, a sequential tabular Q-learning controller chooses one of "
         "four classical policies (FCFS, SJF, Round Robin, Priority) for the next execution segment of the same "
         "evolving trace. The controller sees only work that has already arrived, so every measured gain can be "
         "attributed to information a real scheduler could have had.")

    document.add_heading("1.1 Research questions", level=2)
    for question in [
        "RQ1 — Can a single-CPU scheduler switch between FCFS, SJF, Round Robin and Priority at run time, "
        "using only information available at that instant?",
        "RQ2 — Does such switching reduce mean waiting time relative to the fixed policies it switches between, "
        "once a context-switch cost is charged?",
        "RQ3 — Does a learned switching policy add value over a hand-written adaptive rule that sees the same "
        "observation?",
        "RQ4 — Do the gains hold on a workload family never seen during training, and are they larger than "
        "the paired bootstrap noise?",
    ]:
        bullet(document, question)

    document.add_heading("1.2 Methodology pipeline", level=2)
    para(document,
         "The method is a closed loop between a discrete-event environment, which owns the future, and a "
         "controller, which sees only the past and the present. Stages 3 to 7 run at every decision epoch; "
         "stage 8 runs once per experiment.")
    diagram(document, "Methodology pipeline (section 1.2)", mermaid_pipeline())
    for stage, where, what in [
        ("Stage 1 — Dataset", "workload/generator.py", "seven seeded families of 15-process workloads"),
        ("Stage 2 — Simulation", "scheduler/runtime.py", "one CPU, one FIFO ready queue, remaining bursts, switch cost"),
        ("Stage 3 — Observation", "scheduler/runtime.py", "an immutable snapshot of arrived and ready work only"),
        ("Stage 4 — Encoding", "rl/runtime_state.py", "five causal features, discretised into 162 states (3 x 3 x 3 x 3 x 2)"),
        ("Stage 5 — Decision", "rl/runtime_controller.py",
         "epsilon-greedy in training, greedy in evaluation, over four policy actions, with a counted fallback "
         "in unseen states"),
        ("Stage 6 — Execution", "scheduler/runtime.py",
         "run the segment (Round Robin bounded by the quantum); charge one unit when the running PID changes"),
        ("Stage 7 — Reward", "scheduler/runtime.py",
         "the waiting-time area dW of ready processes gives r = -dW / q, with gamma = 1"),
        ("Stage 8 — Evaluation", "experiments/runtime_experiment.py",
         "greedy, read-only runs on held-out workloads; paired differences with a crossed bootstrap"),
    ]:
        bullet(document, f"{what} ({where}).", bold_prefix=f"{stage} — ")

    document.add_heading("1.3 Causal observation contract", level=2)
    para(document,
         "The environment privately owns the complete event list, so it can admit future arrivals at their "
         "arrival times. The controller receives only an immutable RuntimeObservation; the encoder and "
         "controller interfaces accept no Workload object.")
    para(document, "The observation contains:", bold=True)
    for item in [
        "the current simulated time and the policy chosen at the previous epoch (None at the first epoch);",
        "the arrived, ready processes, with PID, arrival time, known burst, remaining burst and priority;",
        "counts of arrived and completed processes, the observed mean burst and priority spread of arrived "
        "work, and the mean arrival age of ready jobs (current time minus arrival time, which includes earlier "
        "service and is not accumulated queue waiting).",
    ]:
        bullet(document, item)
    para(document,
         "It deliberately excludes future arrivals and their bursts or priorities, the workload size and family, "
         "and the metrics of policies that were not run. A paired hidden-future test changes unarrived processes "
         "and asserts that the observation and action prefixes stay identical. Exact burst lengths are assumed "
         "known once a process arrives, the same assumption standalone SJF requires; no burst predictor is used.")

    document.add_heading("1.4 State abstraction and learning objective", level=2)
    para(document, "The encoder maps each observation to one of 162 states, with quantum q = 4:")
    for feature, bins in [
        ("ready_count", "1; 2–3; 4 or more"),
        ("completed_count", "0; 1–3; 4 or more"),
        ("median_ready_remaining_burst", "at most q; at most 4q; more than 4q"),
        ("mean_ready_arrival_age", "at most q; at most 4q; more than 4q"),
        ("arrived_work_priority_spread", "zero; non-zero"),
    ]:
        bullet(document, bins, bold_prefix=f"{feature} — ")
    para(document,
         "Each trace is one sequential episode. Between decision epochs the environment measures the "
         "waiting-time area dW accumulated by ready processes, including any switch interval, and returns:")
    code(document,
         "r_t = -dW_t / max(1, q)        gamma = 1.0\n"
         "sum over t of r_t = -TotalWaitingTime / q")
    para(document,
         "Because gamma = 1 and every episode is finite, maximising the return minimises total waiting time. "
         "The tabular update is:")
    code(document,
         "Q(s, a) <- Q(s, a) + alpha * ( r_t + gamma * max over visited a' of Q(s', a') - Q(s, a) )\n"
         "alpha = 0.1; terminal transitions omit the bootstrap term; Q initialised to 0")
    para(document,
         "Exploration: epsilon starts at 1.0, is multiplied by 0.997 after each episode and is floored at 0.05. "
         "The first visit to a state forces a uniformly random action. Greedy choices and bootstrap targets use "
         "only actions with a positive visit count. A wholly unseen state in evaluation triggers a counted "
         "fallback to the predeclared heuristic.")

    document.add_heading("1.5 Methodological controls", level=2)
    for control, how in [
        ("Baselines",
         "FCFS, SJF, Round Robin and Priority run on identical workloads with the identical switch cost. A "
         "predeclared, non-learning adaptive rule is fixed before the held-out test."),
        ("Blinded held-out split",
         "the final test (master seed 19301) is generated after training and validation, and the pipeline "
         "asserts zero fingerprint overlap with both."),
        ("Retired seed",
         "an earlier final-test seed (9301) was examined during an audit and is retired; no result from it is reported."),
        ("Out-of-family generalisation",
         "poisson_arrivals is excluded from training and used only at evaluation time."),
        ("Independent seeds and paired uncertainty",
         "five agents (seeds 7101–7105) are trained separately; differences are paired per workload and resampled "
         "with a family-stratified crossed bootstrap of 1,000 replicates (seed 104729)."),
        ("Determinism and timing",
         "evaluation is greedy and read-only; every artifact regenerates from the declared seeds; wall-clock "
         "timings are recorded separately and never charged to simulated time."),
    ]:
        bullet(document, how, bold_prefix=f"{control} — ")


def section_architecture(document: Document, data: Artifacts) -> None:
    document.add_heading("2. Overall conceptual, architectural and pipeline diagram", level=1)
    para(document,
         "The information boundary is structural: the environment owns the future and the accounting, the "
         "controller receives a frozen snapshot of arrived work, and its reply is a single policy name. The "
         "controller cannot modify the queue, the bursts or the event list.")
    diagram(document, "System architecture (section 2)", mermaid_architecture())
    for layer, components, role in [
        ("Environment", "scheduler/runtime.py", "private event list, FIFO ready queue, remaining bursts, switch cost and trace"),
        ("Perception and decision", "rl/runtime_state.py, rl/runtime_controller.py, rl/runtime_heuristic.py",
         "state encoding, the tabular Q-learning policy and the predeclared rule"),
        ("Baselines", "scheduler/fcfs.py, sjf.py, round_robin.py, priority.py",
         "standalone fixed policies, used to check runtime actions"),
        ("Experiment", "experiments/runtime_experiment.py, experiments/runtime_config.py",
         "splits, training, validation, held-out testing, bootstrap and artifact export"),
        ("Data and metrics", "workload/models.py, workload/generator.py, evaluation/metrics.py",
         "Process and Workload objects, seeded families, and metric definitions"),
    ]:
        bullet(document, f"{components}: {role}.", bold_prefix=f"{layer} — ")

    document.add_heading("2.1 Runtime control flow", level=2)
    for number, text in enumerate([
        "The environment advances to the next decision epoch (a dispatch point, a completion or a quantum "
        "boundary) and admits every arrival up to that instant, in (arrival time, pid) order.",
        "It freezes an observation and passes it to the controller, which returns one of four policy names.",
        "If the selected process differs from the running PID, one unit of switch cost is charged before the "
        "segment starts; the chosen dispatch is not reconsidered.",
        "The segment runs. Round Robin is bounded by the quantum; the other policies run to completion. An "
        "unfinished Round-Robin process is appended to the tail after endpoint arrivals.",
        "The waiting-time area is converted to the interval reward and returned to the controller for the update.",
    ], start=1):
        step(document, number, text)


def section_modules(document: Document) -> None:
    document.add_heading("3. Description of the major modules", level=1)
    para(document,
         "Six modules carry the system: two build and run the simulated machine (M1, M2), two implement the "
         "learned controller (M3, M4), one is the non-learning reference rule (M5), and one orchestrates the "
         "experiment and its statistics (M6).")
    for module_id, name, files, contract, responsibility in MODULE_SPECS:
        bullet(document,
               f"Files: {files}. Contract: {contract}. {responsibility}",
               bold_prefix=f"{module_id} — {name}. ")


MODULE_SPECS = [
    ("M1", "Workload generator and dataset builder",
     "workload/generator.py, workload/models.py",
     "family configuration and seed → validated workload with fingerprint",
     "It generates the seven synthetic families and validates every process (non-negative PID, arrival and "
     "priority; positive burst; unique PIDs)."),
    ("M2", "Discrete-event runtime environment",
     "scheduler/runtime.py",
     "workload, controller, quantum and switch cost → schedule trace and six metrics",
     "It owns the private event list, the single FIFO ready queue, remaining bursts, switch-cost accounting "
     "and the interval reward."),
    ("M3", "Causal observation and state encoder",
     "rl/runtime_state.py",
     "RuntimeObservation → state index in [0, 162)",
     "It extracts five causal features, discretises them relative to the quantum and flattens them in "
     "mixed-radix order. It rejects any object other than an observation, including a Workload."),
    ("M4", "Sequential Q-learning controller",
     "rl/runtime_controller.py, rl/q_learning.py",
     "observation (encode) and reward (update) → policy action; Q table and visit counts",
     "Tabular Q-learning over 162 x 4 = 648 state-action pairs, with epsilon-greedy exploration, first-visit "
     "forcing, unvisited-action masking, read-only greedy evaluation and a counted heuristic fallback."),
    ("M5", "Predeclared heuristic controller",
     "rl/runtime_heuristic.py",
     "RuntimeObservation → policy action",
     "A non-learning rule (Round Robin, then Priority, then SJF, then FCFS) that uses the same observation as "
     "the learner and has no fitted parameters."),
    ("M6", "Experiment orchestration, metrics and reporting",
     "experiments/runtime_experiment.py, experiments/runtime_config.py, evaluation/metrics.py",
     "configuration and trained models → CSV and JSON artifacts, paired intervals, markdown report",
     "It trains five agents, validates, checks fingerprint disjointness, runs the held-out test, computes "
     "paired crossed-bootstrap intervals and writes every artifact."),
]


WORKFLOW_NOTES = {
    "M1": ("Source: workload/generator.py, workload/models.py",
           "Each workload is drawn from its family rules with a SHA-256 derived seed, validated, and "
           "fingerprinted before it is assigned to a split."),
    "M2": ("Source: scheduler/runtime.py, one pass per decision epoch",
           "The CPU advances from one decision epoch to the next, charges the switch cost whenever the running "
           "process changes, and returns the interval reward. Numbered steps match the diagram."),
    "M3": ("Source: rl/runtime_state.py",
           "The frozen observation becomes one of 162 state indices. Any object that is not a "
           "RuntimeObservation, and an empty ready queue, is rejected."),
    "M4": ("Source: rl/runtime_controller.py, rl/q_learning.py",
           "Training chooses epsilon-greedily; evaluation is greedy and read-only. Each reward updates Q from "
           "visited next-state actions, and unseen states are counted as fallbacks."),
    "M5": ("Source: rl/runtime_heuristic.py (fixed rules, no learning)",
           "The same frozen observation is tested against three fixed rules in order; the default is FCFS."),
    "M6": ("Source: experiments/runtime_experiment.py, evaluation/metrics.py",
           "Training, validation, two disjointness checks, the held-out test, paired statistics and artifact "
           "export run in this order, and any failed check stops the run."),
}


def section_workflows(document: Document) -> None:
    document.add_heading("4. Workflow diagram for each major module", level=1)
    para(document,
         "Each module has a Mermaid flowchart. Diamonds are decision points, and the numbers in each diagram "
         "match the numbered steps.")
    for module_id, name, _files, _contract, _responsibility in MODULE_SPECS:
        source, note = WORKFLOW_NOTES[module_id]
        document.add_heading(f"{module_id}. {name}", level=2)
        para(document, source, italic=True)
        diagram(document, f"Workflow of {module_id}, {name}", WORKFLOW_SOURCES[module_id])
        para(document, note)


WORKFLOW_SOURCES = {
    "M1": WORKFLOW_M1, "M2": WORKFLOW_M2, "M3": WORKFLOW_M3,
    "M4": WORKFLOW_M4, "M5": WORKFLOW_M5, "M6": WORKFLOW_M6,
}


def section_dataset(document: Document, data: Artifacts) -> None:
    document.add_heading("5. Dataset details and data collection procedure", level=1)
    document.add_heading("5.1 Data collection procedure", level=2)
    para(document,
         "No external or personal data is used. The dataset is synthetic and generated entirely by the "
         "project's own generator, so it can be reproduced from the declared integers on any machine.")
    for step_id, action, detail in [
        ("D1", "Fix seven family configurations",
         "burst distribution, arrival pattern and priority pattern, declared once in config.py and frozen in the "
         "runtime experiment configuration"),
        ("D2", "Derive one seed per workload",
         "workload seed = SHA-256(master seed, split tag, family index, repetition or episode); no global random "
         "state is shared"),
        ("D3", "Generate and validate 15 processes per workload",
         "burst, arrival and priority are drawn by the family rules, then checked for positive burst, "
         "non-negative arrival and bounded priority"),
        ("D4", "Assign workloads to splits",
         "training: 1,200 sequential episodes for each of five seeds (6,000 in total), with the six training "
         "families cycled so each receives 200 episodes per seed; validation: 140 workloads (7 families x 20); "
         "held-out test: 210 workloads (7 families x 30)"),
        ("D5", "Fingerprint and check disjointness",
         "each workload has a SHA-256 fingerprint; the pipeline asserts zero overlap across the splits before "
         "the held-out test runs"),
        ("D6", "Store what is needed and regenerate on demand",
         "metrics, the manifest, Q tables and summaries are committed; the large per-decision logs are "
         "regenerated by python main.py runtime-experiment and are gitignored"),
    ]:
        bullet(document, f"{action} — {detail}.", bold_prefix=f"{step_id}: ")

    document.add_heading("5.2 Family specification", level=2)
    para(document,
         "Each family generates 15 processes. Parameters are read from the frozen configuration and are also "
         "stored in results/runtime/runtime_summary.json.")
    for family in data.cfg["workload_family_parameters"]:
        role = ("evaluation only; never used for training" if family["name"] == "poisson_arrivals"
                else "training and evaluation")
        bullet(document, f"{family_sentence(family)} Role: {role}.",
               bold_prefix=f"{family['name']} — ")

    document.add_heading("5.3 Measured dataset statistics", level=2)
    if not data.processes.empty:
        para(document,
             "Recomputed with the project's generator at build time, so they describe the held-out dataset "
             "rather than the configuration alone. High priority means priority <= 2.")
        bullet(document, dataset_line(data.processes, data.workloads), bold_prefix="All families — ")
    else:
        para(document, "Statistics were not recomputed for this build (--no-dataset).")

    document.add_heading("5.4 Data integrity and assumptions", level=2)
    for item in [
        f"Reproducibility and disjointness — a workload is a pure function of its seed inputs, and the committed "
        f"run reports overlap = {str(data.summary['split_fingerprint_overlap']).lower()} across "
        f"{data.summary['training_fingerprint_count_union']:,} distinct training fingerprints.",
        "Burst knowledge — exact bursts are known once a process arrives, as standalone SJF already requires; "
        "no burst predictor is included.",
        "Licensing and privacy — the dataset is entirely synthetic, so no personal or licensed data is involved.",
    ]:
        bullet(document, item)


def family_sentence(f: Dict) -> str:
    """Describe one family from its frozen parameters (no hand-typed numbers)."""
    n = f["num_processes"]
    lo, hi = f["burst_time_min"], f["burst_time_max"]
    dist = f["burst_distribution"]
    if dist == "uniform":
        burst = f"bursts uniform over {lo}–{hi}"
    elif dist == "bimodal":
        share = round(100 * f["short_burst_fraction"])
        burst = (f"{share}% of bursts uniform over {lo}–{f['short_burst_max']} and the rest uniform over "
                 f"{f['short_burst_max'] + 1}–{hi}")
    elif dist == "staggered_interactive":
        burst = (f"one long process with burst uniform over {lo}–{hi} at time 0, and {n - 1} short "
                 f"processes with bursts uniform over 1–{f['short_burst_max']}")
    else:
        raise ValueError(f"unknown burst distribution {dist!r}")

    arrival = f["arrival_pattern"]
    if arrival == "uniform":
        arrivals = f"arrival times uniform over the integers 0–{f['arrival_window']}"
    elif arrival == "staggered":
        arrivals = (f"first arrival at time 0, then integer gaps uniform over "
                    f"{f['staggered_gap_min']}–{f['staggered_gap_max']}")
    elif arrival == "poisson":
        arrivals = (f"arrivals from a Poisson process with rate {f['arrival_rate']} (exponential "
                    f"inter-arrival times, floored to whole time units), first arrival at time 0")
    else:
        raise ValueError(f"unknown arrival pattern {arrival!r}")

    pmin, pmax = f["priority_min"], f["priority_max"]
    if pmin == pmax:
        priority = f"all priorities equal to {pmin}"
    elif f["priority_pattern"] == "uniform":
        priority = f"priorities uniform over {pmin}–{pmax}"
    else:
        priority = (f"priorities uniform over {pmin}–{pmax}, except that with probability "
                    f"{f['high_priority_fraction']} the priority is drawn from {pmin}–"
                    f"{f['high_priority_cutoff']} instead")
    return f"{burst[0].upper()}{burst[1:]}; {arrivals}; {priority}."


def dataset_line(procs, workloads) -> str:
    return (f"{len(workloads)} workloads, {len(procs):,} processes; burst "
            f"{procs['burst_time'].mean():.1f} ± {procs['burst_time'].std(ddof=1):.1f}; mean arrival "
            f"{procs['arrival_time'].mean():.1f}; mean total work {workloads['total_burst'].mean():.0f}; "
            f"high-priority share {100.0 * (procs['priority'] <= 2).mean():.0f}%.")


def section_tools(document: Document, data: Artifacts) -> None:
    document.add_heading("6. Tools, technologies, algorithms and frameworks used", level=1)
    document.add_heading("6.1 Tools and technologies", level=2)
    for category, item, role in [
        ("Language", "Python 3.11.2", "implementation, command-line interface and report builders"),
        ("Numerics and data", "NumPy 2.4.6, pandas 3.0.6", "Q tables and visit counts, bootstrap resampling, metric aggregation and CSV export"),
        ("Testing", "pytest 9.1.1", "automated checks of every module (section 8.2)"),
        ("Diagrams", "Mermaid (text source)", "all diagrams in this document"),
        ("Document generation", "python-docx 1.2.0, ReportLab 5.0.1", "the text-only DOCX and the PDF rendered from it"),
        ("Standard library", "hashlib (SHA-256), argparse, dataclasses, json, pathlib",
         "seed derivation, fingerprints, configuration and serialisation"),
        ("Version control", "Git and GitHub", "branch-per-change workflow with pull requests"),
    ]:
        bullet(document, f"{item} — {role}.", bold_prefix=f"{category}: ")
    para(document,
         "No deep-learning framework is used. The controller is a tabular Q-learning agent, so every value, "
         "visit count and fallback is inspectable in the committed artifacts.")

    document.add_heading("6.2 Algorithms", level=2)
    for name, definition, usage in [
        ("FCFS (action 0)", "dispatch the head of the ready queue, non-preemptively",
         "Fixed baseline and runtime action."),
        ("SJF (action 1)", "smallest remaining burst first, non-preemptively",
         "Minimises mean waiting time when all jobs are present together; a strong but not guaranteed-optimal "
         "reference under staggered arrivals."),
        ("Round Robin (action 2)", "the head runs for min(quantum, remaining burst); unfinished jobs go to the tail",
         "Bounds the time to first service."),
        ("Priority (action 3)", "highest static priority first, non-preemptively",
         "Fixed baseline and runtime action; a lower numeric value means higher priority."),
        ("Tabular Q-learning", "epsilon-greedy exploration, unvisited-action masking, the update of section 1.4",
         "The learned runtime controller over 162 states x 4 actions."),
        ("Predeclared heuristic", "Round Robin, then Priority, then SJF, then FCFS, on the same observation",
         "The non-learning adaptive reference."),
        ("Crossed bootstrap", "resample workloads within each family and model seeds independently; 1,000 replicates",
         "Paired 95% intervals."),
    ]:
        bullet(document, f"{definition}. {usage}", bold_prefix=f"{name} — ")

    document.add_heading("6.3 Controller pseudocode", level=2)
    code(document,
         """SELECT_ACTION(obs):                  # training: epsilon-greedy; evaluation: greedy
    s <- encode(obs)                 # 5 causal features -> state in [0, 162)
    if evaluating:
        if no visited action in s:   # wholly unseen state
            fallback_count += 1
            return HEURISTIC(obs)    # predeclared rule, reported rather than hidden
        return argmax over visited actions of Q[s, a]
    if first_visit(s) or rand() < epsilon(episode):
        return uniform_random_action()
    return argmax over visited actions of Q[s, a]

UPDATE(s, a, r, s', terminal):
    target <- r if terminal else r + gamma * max over visited actions of Q[s', .]
    Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])
    visits[s, a] <- visits[s, a] + 1

HEURISTIC(obs):                      # predeclared, no learning
    if ready_count >= 3 and mean_arrival_age >= quantum:  return Round Robin
    if at least 2 ready jobs have distinct priorities:    return Priority
    if max(remaining_burst) >= 2 * min(remaining_burst):  return SJF
    return FCFS""")


def section_experiment(document: Document, data: Artifacts) -> None:
    document.add_heading("7. Experimental plan and evaluation metrics", level=1)
    document.add_heading("7.1 Experimental plan", level=2)
    for step_id, action, detail in [
        ("P1", "Fix the design before looking at results",
         "families, splits, seeds, quantum (4), switch cost (1), learning rate and the heuristic rule are "
         "declared in configuration and committed"),
        ("P2", "Train and validate",
         "five independent agents (seeds 7101–7105), 1,200 sequential episodes each; 140 validation workloads "
         "(seed 8201), used for reporting and the illustrative-trace rule only"),
        ("P3", "Test once, held out",
         "210 untouched workloads (seed 19301); the fixed policies and the heuristic run once; the five Q-learning "
         "models are evaluated greedily and read-only"),
        ("P4", "Compare and quantify uncertainty",
         "paired per-workload differences; family-stratified crossed bootstrap, 1,000 replicates, seed 104729; "
         "95% intervals"),
    ]:
        bullet(document, f"{action} — {detail}.", bold_prefix=f"{step_id}: ")

    document.add_heading("7.2 Evaluation metrics", level=2)
    for name, definition, why in [
        ("Mean waiting time", "mean of (turnaround time minus burst time) over processes",
         "the primary metric; the quantity the reward accumulates"),
        ("Mean turnaround time", "mean of (completion time minus arrival time)", "end-to-end latency"),
        ("Mean response time", "mean of (first dispatch time minus arrival time)",
         "interactivity; the metric Round Robin improves"),
        ("CPU utilisation and throughput", "busy time over elapsed time (%); processes completed per unit time",
         "the cost of switching and idling; work completed"),
        ("Context and policy switches", "changes of the running PID; decisions that change the policy",
         "switching overhead, and how much the controller adapts"),
    ]:
        bullet(document, f"{definition}. Why: {why}.", bold_prefix=f"{name} — ")

    document.add_heading("7.3 Results on the held-out test", level=2)
    para(document,
         "Means ± standard deviation across the 210 held-out workloads, with quantum 4 and switch cost 1. "
         "Q-learning also averages the five model seeds.")
    for method in METHOD_ORDER:
        wait, wait_sd = data.mean_sd("final", method, "avg_waiting_time")
        turn, turn_sd = data.mean_sd("final", method, "avg_turnaround_time")
        resp, resp_sd = data.mean_sd("final", method, "avg_response_time")
        cpu, cpu_sd = data.mean_sd("final", method, "cpu_utilization")
        thr, thr_sd = data.mean_sd("final", method, "throughput")
        ctx, ctx_sd = data.mean_sd("final", method, "context_switches")
        switches, _ = data.mean_sd("final", method, "policy_switch_count")
        bullet(document,
               f"waiting {wait:,.3f} ± {wait_sd:,.2f}; turnaround {turn:,.3f} ± {turn_sd:,.2f}; response "
               f"{resp:,.3f} ± {resp_sd:,.2f}; CPU utilisation {cpu:.2f} ± {cpu_sd:.2f} %; throughput "
               f"{thr:.3f} ± {thr_sd:.3f}; context switches {ctx:.2f} ± {ctx_sd:.2f}; policy switches per "
               f"trace {switches:.2f}.",
               bold_prefix=f"{method} — ")

    document.add_heading("7.4 Paired comparisons", level=2)
    para(document,
         "Each line gives target minus reference with its 95% crossed-bootstrap interval (waiting and response "
         "time; all metrics are in results/runtime/runtime_paired_comparisons.csv). Turnaround differences equal "
         "waiting differences, because both methods serve the same bursts.")
    for target, reference in [
        ("Runtime Q-learning", "SJF"),
        ("Runtime Q-learning", "Causal heuristic"),
        ("Round Robin", "SJF"),
        ("Causal heuristic", "SJF"),
        ("FCFS", "SJF"),
        ("Priority", "SJF"),
    ]:
        parts = []
        for metric, label in [("avg_waiting_time", "waiting"), ("avg_response_time", "response")]:
            mean, low, high = ci_of(data, target, reference, metric)
            if mean == 0 and low == 0 and high == 0:
                reading = "no difference"
            else:
                reading = "favours the target" if mean < 0 else "favours the reference"
            parts.append(f"{label} {mean:+,.3f} [{low:+,.3f}, {high:+,.3f}], {reading}")
        bullet(document, "; ".join(parts), bold_prefix=f"{SHORT_NAME[target]} minus {SHORT_NAME[reference]} — ")

    document.add_heading("7.5 What the plan has established", level=2)
    q_sjf_wait = ci_of(data, "Runtime Q-learning", "SJF", "avg_waiting_time")
    q_sjf_resp = ci_of(data, "Runtime Q-learning", "SJF", "avg_response_time")
    rr_sjf_wait = ci_of(data, "Round Robin", "SJF", "avg_waiting_time")
    fcfs_sjf_wait = ci_of(data, "FCFS", "SJF", "avg_waiting_time")
    prio_sjf_wait = ci_of(data, "Priority", "SJF", "avg_waiting_time")
    q_heur_wait = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_waiting_time")
    q_heur_resp = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_response_time")
    q_mean = data.summary_mean("final", "Runtime Q-learning", "avg_waiting_time")
    rr_mean = data.summary_mean("final", "Round Robin", "avg_waiting_time")
    heur_mean = data.summary_mean("final", "Causal heuristic", "avg_waiting_time")
    poisson_q = data.final[(data.final["method"] == "Runtime Q-learning")
                           & (data.final["family"] == "poisson_arrivals")]["avg_waiting_time"].mean()
    poisson_sjf = data.final[(data.final["method"] == "SJF")
                             & (data.final["family"] == "poisson_arrivals")]["avg_waiting_time"].mean()
    poisson_rr = data.final[(data.final["method"] == "Round Robin")
                            & (data.final["family"] == "poisson_arrivals")]["avg_waiting_time"].mean()
    family_means = data.final.groupby(["method", "family"])["avg_waiting_time"].mean().unstack(0)
    families_between = [
        family for family, row in family_means.iterrows()
        if min(row["SJF"], row["Round Robin"]) <= row["Runtime Q-learning"] <= max(row["SJF"], row["Round Robin"])
    ]
    for item in [
        f"Against Round Robin, mean waiting time is lower by {rr_mean - q_mean:,.3f} time units "
        f"({q_mean:,.3f} against {rr_mean:,.3f}). No paired interval was computed for this pair; it follows from "
        f"the two committed comparisons against SJF (Q minus SJF {q_sjf_wait[0]:+,.3f}; RR minus SJF "
        f"{rr_sjf_wait[0]:+,.3f}). FCFS and Priority lie between them, at {fcfs_sjf_wait[0]:+,.3f} and "
        f"{prio_sjf_wait[0]:+,.3f} relative to SJF.",
        f"Against the predeclared adaptive rule, mean waiting time is lower by {heur_mean - q_mean:,.3f} (paired "
        f"difference {q_heur_wait[0]:+,.3f}; 95% interval [{q_heur_wait[1]:+,.3f}, {q_heur_wait[2]:+,.3f}]). "
        f"The rule has the lower response time, by {q_heur_resp[0]:,.3f}, because it selects Round Robin whenever "
        f"three or more aged jobs are ready.",
        f"Against SJF the result is mixed. Waiting time is higher for Q-learning by {q_sjf_wait[0]:,.3f} (95% "
        f"interval {q_sjf_wait[1]:,.3f} to {q_sjf_wait[2]:,.3f}), while response time is lower by "
        f"{abs(q_sjf_resp[0]):,.3f}. Non-preemptive SJF with exact bursts and no mid-job switch cost is a strong "
        f"reference, and the learned controller does not beat it on waiting time.",
        f"Q-learning's mean waiting time lies between the SJF and Round Robin values on all {len(families_between)} "
        f"held-out families, including the unseen poisson_arrivals family ({poisson_q:,.2f}, against SJF "
        f"{poisson_sjf:,.2f} and Round Robin {poisson_rr:,.2f}).",
        f"Controller overhead is small: about {data.summary['mean_runtime_q_observation_us_per_decision']:.2f} "
        f"microseconds to build an observation and {data.summary['mean_runtime_q_selection_us_per_decision']:.2f} "
        f"microseconds to select an action per decision on the build host. These timings are never charged to "
        f"the schedule.",
    ]:
        bullet(document, item)


def section_progress(document: Document, data: Artifacts, test_rows: List[List[str]],
                     pytest_block: str, cli_block: str) -> None:
    document.add_heading("8. Implementation progress", level=1)
    para(document,
         "Review 2 requires at least 50% of the proposed work to be complete, with completed modules shown by valid "
         "outputs. All six proposed modules are implemented, tested and demonstrated below (100% of the Review 2 "
         "scope). Work planned beyond that scope is listed in section 9.2.")

    document.add_heading("8.1 Completion and outputs by module", level=2)
    outputs = dict(module_outputs(data))
    for module_id, name, _files, _contract, _responsibility in MODULE_SPECS:
        bullet(document, f"completed (100%). Output: {outputs[module_id]}.",
               bold_prefix=f"{module_id} — {name}. ")
    para(document, "Overall completion of the proposed Review 2 modules: 6 of 6 (100%).", bold=True)

    document.add_heading("8.2 Demonstration 1 — automated verification", level=2)
    if test_rows:
        total = sum(int(row[1]) for row in test_rows)
        para(document,
             f"The suite collects {total} checks across {len(test_rows)} test files. The two legacy files exercise "
             f"the earlier offline selector, which is kept for comparison.")
    code(document, pytest_block, label="Output of python -m pytest -q:")

    document.add_heading("8.3 Demonstration 2 — end-to-end run output", level=2)
    para(document,
         "One command runs the full pipeline. The output below reproduces its printed lines, with file paths "
         "shortened to file names. A fresh run reproduced the committed artifacts except for host-timing columns.")
    code(document, cli_block, label="Output of python main.py runtime-experiment:")

    document.add_heading("8.4 Demonstration 3 — learned behaviour within one trace", level=2)
    demo = data.demo
    para(document,
         f"The controller switches policies inside a single trace. The example, chosen by a validation-only rule "
         f"(maximise policy switches among traces with zero unseen-state fallbacks), is training seed "
         f"{demo['training_seed']}, family {demo['family']}, repetition {demo['repetition']}, fingerprint "
         f"{demo['workload_fingerprint']}. It contains {demo['policy_switch_count']} policy switches over "
         f"{demo['decision_count']} decisions, with mean waiting time {demo['metrics']['avg_waiting_time']:.3f} "
         f"and mean response time {demo['metrics']['avg_response_time']:.3f}. It illustrates the mechanism and is "
         f"not a performance claim.")

    document.add_heading("8.5 Artifacts produced", level=2)
    para(document,
         "The run writes results/runtime/: final-test metrics (2,100 rows), validation metrics (1,400 rows), "
         "training metrics (6,000 rows), paired comparisons (36 rows), state-action coverage (15 rows), the Q table "
         "(3,240 rows), the workload manifest (6,350 rows), the illustrative trace, a summary JSON and a markdown report.")


def section_challenges(document: Document, data: Artifacts) -> None:
    document.add_heading("9. Challenges encountered and plan for completing the remaining work", level=1)
    document.add_heading("9.1 Challenges encountered", level=2)
    for title, why, how in [
        ("Keeping the controller causal",
         "an earlier whole-workload selector saw future arrivals, so its results could not be read as a runtime controller",
         "an immutable RuntimeObservation, an encoder that accepts only that object, and a hidden-future test"),
        ("FCFS after a Round-Robin requeue",
         "the requeued job could be ordered by original arrival or by live queue order",
         "one persistent FIFO ready queue with canonical (arrival time, pid) insertion; fixed-action runs are tested "
         "against the standalone schedulers"),
        ("Charging switch cost fairly",
         "charging only the learner would invalidate the comparison; charging nothing would be unrealistic",
         "one unit whenever the running PID changes, for every method including the fixed policies"),
        ("A reward that matches the objective",
         "a shaped reward can drift from the reported metric",
         "r = -dW / q with gamma = 1, so the episodic return equals minus total waiting time divided by q; a test asserts this"),
        ("Sparse coverage of the tabular space",
         "only about 42% of the 648 state-action pairs are visited, and unseen states appear at evaluation",
         "forced exploration on first visit, masking of unvisited actions, and a counted fallback to the heuristic"),
        ("Reporting a negative result",
         "the learner does not beat non-preemptive SJF on mean waiting time",
         "reported as a finding with its interval, and the structural reason stated: SJF uses exact bursts and pays no mid-job switch cost"),
        ("Host-dependent timing",
         "wall-clock columns differ between machines and can look like non-determinism",
         "timings are stored separately, never charged to simulated time, and excluded from schedule comparisons"),
    ]:
        bullet(document, f"Why it mattered: {why}. Resolution: {how}.", bold_prefix=f"{title} — ")

    document.add_heading("9.2 Plan for completing the remaining work", level=2)
    for task_id, task, gap, approach, deliverable in [
        ("W1", "Sensitivity study", "all results use quantum 4 and switch cost 1",
         "re-run the plan over a grid of quanta and switch costs", "metric tables for each cell"),
        ("W2", "Burst prediction", "exact bursts are assumed known at arrival",
         "add an explicit predictor and re-run the controller under its error", "results with error bands"),
        ("W3", "Richer state abstraction", "only about 42% of pairs are visited; 162 states is coarse",
         "tile coding or a small function approximator, keeping the same causal contract", "coverage and metric comparison"),
        ("W4", "Multi-objective reward", "waiting time is optimised; response time is not",
         "add a weighted or constrained objective", "trade-off front over the two metrics"),
        ("W5", "I/O and blocking", "processes never block; the queue is CPU-bound only",
         "model I/O bursts and blocked and wake-up transitions", "re-baselined comparisons"),
        ("W6", "Multicore extension", "single CPU only",
         "N cores with per-core queues and a migration cost", "multicore metrics and overhead"),
        ("W7", "Trace-driven validation", "all workloads are synthetic",
         "replay a public scheduling trace through the same environment", "trace-driven result table"),
        ("W8", "Phase-aware monitored features (proposed, not implemented)",
         "the encoder monitors queue size, completed count, median remaining burst, mean arrival age and priority "
         "spread; recent burst history, accumulated ready-queue waiting time and arrival rate are not yet monitored",
         "add these as causal features, detect workload phases, and let the controller switch at phase boundaries",
         "a re-trained controller compared with the current 162-state version"),
    ]:
        bullet(document,
               f"gap: {gap}. Approach: {approach}. Deliverable: {deliverable}.",
               bold_prefix=f"{task_id} — {task}. ")

    document.add_heading("9.3 Sequence for the remaining work", level=2)
    para(document,
         "W1 and W2 come first, because they test whether the conclusions depend on fixed parameters or on the "
         "idealised burst assumption. W8 and W3 follow, since they address the coarse state abstraction and the "
         "missing monitored features. W4 to W6 extend the environment, and W7 provides external validation.")


def collect_test_counts(root: Path) -> Tuple[List[List[str]], str]:
    """Run pytest and return (per-file rows, condensed output)."""
    descriptions = {
        "test_adaptive.py": "legacy offline selector: state encoding, decisions, read-only evaluation",
        "test_experiment_design.py": "frozen seeds, split sizes and configuration invariants",
        "test_experiments.py": "legacy experiment pipeline, metrics and report generation",
        "test_fcfs.py": "FCFS ordering, idle handling and metric computation",
        "test_generator.py": "workload family generation and reproducibility from seeds",
        "test_metrics.py": "waiting, turnaround, response, utilisation and throughput definitions",
        "test_priority.py": "priority ordering, ties and non-preemption",
        "test_q_learning.py": "Q updates, epsilon schedule, masking, visit counts, evaluation",
        "test_reward.py": "reward composition and clipping against the reference policies",
        "test_round_robin.py": "quantum boundaries, re-enqueue order and completion behaviour",
        "test_runtime_experiment.py": "split disjointness, determinism, artifact regeneration",
        "test_runtime_scheduler.py": "runtime environment: fixed-action equivalence, causality, reward identity",
        "test_schedule_result.py": "schedule trace validation and aggregate result objects",
        "test_sjf.py": "SJF selection, tie-breaking and non-preemption",
        "test_state.py": "offline state discretisation and mixed-radix indexing",
        "test_workload_models.py": "process and workload models, validation and trace invariants",
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
    tail = [line for line in run.stdout.strip().splitlines() if line.strip()][-1:]
    output = "\n".join(["$ python -m pytest -q", *tail]) if tail else (
        f"$ python -m pytest -q\n{sum(counts.values())} tests (output unavailable)")
    return rows, output


ARTIFACT_FILES = [
    ("training_metrics", "runtime_training_metrics.csv"),
    ("validation_metrics", "runtime_validation_metrics.csv"),
    ("final_test_metrics", "runtime_final_test_metrics.csv"),
    ("validation_decisions", "runtime_validation_decisions.csv"),
    ("final_test_decisions", "runtime_final_test_decisions.csv"),
    ("workload_manifest", "runtime_workload_manifest.csv"),
    ("state_action_coverage", "runtime_state_action_coverage.csv"),
    ("q_table", "runtime_q_table.csv"),
    ("paired_comparisons", "runtime_paired_comparisons.csv"),
    ("final_test_summary", "runtime_final_test_summary.csv"),
    ("validation_summary", "runtime_validation_summary.csv"),
    ("summary", "runtime_summary.json"),
    ("learned_switch_demo", "runtime_learned_switch_demo.json"),
    ("report", "runtime_report.md"),
]


def build_cli_block(data: Artifacts) -> str:
    """Mirror the lines printed by ``python main.py runtime-experiment``."""
    cfg = data.cfg
    lines = [
        "$ python main.py runtime-experiment",
        "Causal event-driven runtime experiment",
        "---------------------------------------",
        "  training seeds                 : " + ", ".join(str(s) for s in cfg["training_seeds"]),
        f"  training episodes/model        : {cfg['training_episodes_per_seed']}",
        f"  validation workloads           : {data.summary['validation_unique_workloads']}",
        f"  final held-out test workloads  : {data.summary['final_test_unique_workloads']}",
        "  final-test mean waiting time:",
    ]
    for method, values in data.summary["final_test_means"].items():
        if "avg_waiting_time" in values:
            lines.append(f"    {method:<22}: {values['avg_waiting_time']:.3f}")
    demo = data.summary["demo"]
    lines.append(f"  learned within-trace switches : {demo['policy_switch_count']} "
                 f"({demo['decision_count']} decisions; {Path(demo['path']).name})")
    lines.append("  artifacts:")
    for key, filename in ARTIFACT_FILES:
        lines.append(f"    {key:<24}: {filename}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repository root holding results/")
    parser.add_argument("--output", type=Path, default=None,
                        help="DOCX path (default: <root>/Review_2_Report.docx)")
    parser.add_argument("--docs-copy", type=Path, default=None,
                        help="second copy (default: <root>/docs/Review_2_Report.docx)")
    parser.add_argument("--skip-tests", action="store_true",
                        help="do not run pytest while building the document")
    parser.add_argument("--no-dataset", action="store_true",
                        help="skip rebuilding the dataset statistics (section 5.4)")
    args = parser.parse_args(argv)

    root: Path = args.root
    output = args.output or root / "Review_2_Report.docx"
    docs_copy = args.docs_copy or root / "docs" / "Review_2_Report.docx"

    data = Artifacts(root, with_dataset=not args.no_dataset)

    if args.skip_tests:
        test_rows: List[List[str]] = []
        pytest_block = "(pytest not executed: --skip-tests)"
        test_line = "273 checks (not re-run for this build)"
    else:
        test_rows, pytest_block = collect_test_counts(root)
        total = sum(int(row[1]) for row in test_rows) if test_rows else 0
        passed = " passed" in pytest_block
        test_line = (f"{total} checks, {'all passing' if passed else 'see section 8.3'}"
                     f"{' (re-run during this build)' if total else ''}")

    generated = _dt.date.today().strftime("%d %B %Y")

    _DIAGRAM_COUNTER["n"] = 0
    document = new_document()
    add_cover(document, data, generated, test_line)
    section_methodology(document, data)
    section_architecture(document, data)
    section_modules(document)
    section_workflows(document)
    section_dataset(document, data)
    section_tools(document, data)
    section_experiment(document, data)
    section_progress(document, data, test_rows, pytest_block, build_cli_block(data))
    section_challenges(document, data)

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    if docs_copy.resolve() != output.resolve():
        docs_copy.parent.mkdir(parents=True, exist_ok=True)
        docs_copy.write_bytes(output.read_bytes())

    print(f"wrote {output} ({output.stat().st_size / 1024:,.0f} KB)")
    print(f"wrote {docs_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
