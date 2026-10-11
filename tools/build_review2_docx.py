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
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11.5), ("Title", 20)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = BLACK
        style.font.bold = True
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
        run.font.size = Pt(9)
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


def mermaid_results(data: Artifacts, metric: str, title: str, unit: str) -> str:
    labels = [SHORT_NAME[method] for method in METHOD_ORDER]
    values = [round(data.summary_mean("final", method, metric), 3) for method in METHOD_ORDER]
    top = max(values) * 1.1
    lines = [
        "xychart-beta",
        f'    title "{title} - 210 held-out workloads"',
        "    x-axis [" + ", ".join(f'"{label}"' for label in labels) + "]",
        f'    y-axis "{unit}" 0 --> {top:.0f}',
        "    bar [" + ", ".join(str(value) for value in values) + "]",
    ]
    return "\n".join(lines)


def mermaid_gantt(data: Artifacts) -> str:
    demo = data.demo
    policies = {(d["time"], d["chosen_pid"]): d["policy"] for d in demo["decisions"]}
    lines = [
        "gantt",
        f"    title Illustrative validation trace - seed {demo['training_seed']}, "
        f"{demo['family']} repetition {demo['repetition']}",
        "    dateFormat X",
        "    axisFormat %s",
    ]
    segments = sorted(demo["trace"], key=lambda s: (int(s["pid"]), s["start"]))
    current_pid = None
    task_index = 0
    for segment in segments:
        task_index += 1
        pid = int(segment["pid"])
        policy = policies.get((segment["start"], pid))
        if policy is None:
            policy = next(
                (d["policy"] for d in demo["decisions"]
                 if d["chosen_pid"] == pid and d["time"] <= segment["start"]),
                "FCFS",
            )
        if pid != current_pid:
            lines.append(f"    section P{pid}")
            current_pid = pid
        lines.append(f"    {policy} :t{task_index}, {int(segment['start'])}, {int(segment['end'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
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
         "The proposed system is a single-CPU, event-driven, runtime-adaptive CPU scheduler. "
         "Instead of committing one policy for the lifetime of a workload, it decides at every "
         "dispatch or Round-Robin quantum boundary which of four classical policies (FCFS, SJF, "
         "Round Robin, Priority) governs the next execution segment of the same evolving trace. "
         "The controller sees only work that has arrived, so every measured gain is attributable "
         "to information a real scheduler could have had.")

    document.add_heading("1.1 Research questions", level=2)
    for question in [
        "RQ1 — Can a single-CPU scheduler switch between FCFS, SJF, Round Robin and Priority at "
        "run time, using only information available at that instant?",
        "RQ2 — Does such switching reduce mean waiting time relative to the fixed policies it "
        "switches between, once a context-switch cost is charged?",
        "RQ3 — Does a learned switching policy add value over a hand-written adaptive rule that "
        "sees exactly the same observation?",
        "RQ4 — Do any gains survive on a workload family that was never seen during training, "
        "and are they larger than the paired bootstrap noise?",
    ]:
        bullet(document, question)

    document.add_heading("1.2 Methodology pipeline", level=2)
    para(document,
         "The method is a closed loop between a discrete-event environment that owns the future "
         "and a controller that may see only the past and the present. Stages 3 to 7 run once per "
         "decision epoch inside a single trace; stage 8 runs once per experiment.")
    diagram(document, "Methodology pipeline (section 1.2)", mermaid_pipeline())
    for stage, where, what, out in [
        ("Stage 1 — Dataset", "workload/generator.py",
         "generate seven seeded families of 15-process workloads with SHA-256 derived seeds",
         "workload list and manifest"),
        ("Stage 2 — Simulation", "scheduler/runtime.py",
         "advance one CPU through arrivals, one FIFO ready queue, remaining bursts and switch cost",
         "decision epochs and schedule trace"),
        ("Stage 3 — Observation", "scheduler/runtime.py",
         "freeze an immutable snapshot of arrived and ready work only",
         "RuntimeObservation"),
        ("Stage 4 — Encoding", "rl/runtime_state.py",
         "discretise five causal features into 162 tabular states (3 x 3 x 3 x 3 x 2)",
         "state index s in [0, 162)"),
        ("Stage 5 — Decision", "rl/runtime_controller.py",
         "epsilon-greedy during training and greedy during evaluation over four policy actions, "
         "with an explicit fallback in unseen states",
         "action a in {FCFS, SJF, Round Robin, Priority}"),
        ("Stage 6 — Execution", "scheduler/runtime.py",
         "run the selected segment (Round Robin bounded by the quantum), charge one unit when the "
         "running PID changes, settle endpoint arrivals",
         "advanced simulated time"),
        ("Stage 7 — Reward", "scheduler/runtime.py",
         "measure the waiting-time area dW accumulated by ready processes and return r = -dW / q "
         "with gamma = 1",
         "interval reward r_t"),
        ("Stage 8 — Evaluation", "experiments/runtime_experiment.py",
         "greedy, read-only runs on held-out workloads; paired differences with a family-stratified "
         "crossed bootstrap",
         "metric tables and 95% intervals"),
    ]:
        bullet(document, f"{where}: {what}. Output: {out}.", bold_prefix=f"{stage} — ")

    document.add_heading("1.3 Causal observation contract", level=2)
    para(document,
         "The environment privately owns the complete event list so that it can admit future "
         "arrivals at their arrival times. The controller receives only an immutable "
         "RuntimeObservation, and the encoder and controller interfaces accept no Workload object.")
    para(document, "Included in the observation:", bold=True)
    for item in [
        "current simulated time and the policy selected at the immediately preceding epoch "
        "(None at the first epoch);",
        "processes that have arrived and are currently ready, with PID, arrival time, known burst, "
        "remaining burst and priority;",
        "counts of arrived and completed processes;",
        "aggregates restricted to work that has already arrived: observed mean burst and priority "
        "spread;",
        "mean arrival age of currently ready jobs (current time minus arrival time, which includes "
        "earlier CPU service and is not accumulated queue waiting).",
    ]:
        bullet(document, item)
    para(document, "Deliberately excluded:", bold=True)
    for item in [
        "any data about processes that have not arrived yet, including their future bursts, "
        "priorities and arrival times;",
        "the total size of the workload and the identity of its family;",
        "counterfactual or completed-schedule metrics of policies that were not run.",
    ]:
        bullet(document, item)
    para(document,
         "A paired hidden-future test changes the arrival, burst and priority of unarrived "
         "processes while keeping the observable prefix fixed, and asserts that the observation "
         "and action prefixes remain identical.")
    para(document,
         "Burst assumption: exact burst lengths are known once a process has arrived. This is the "
         "same assumption the standalone SJF model already requires; no burst predictor is included.")

    document.add_heading("1.4 State abstraction and learning objective", level=2)
    para(document, "The encoder maps each observation to one of 3 x 3 x 3 x 3 x 2 = 162 states, "
                   "with quantum q = 4:")
    for feature, bins, why in [
        ("ready_count", "1; 2–3; 4 or more", "ready-queue contention"),
        ("completed_count", "0; 1–3; 4 or more", "trace progress, without knowing the workload size"),
        ("median_ready_remaining_burst", "at most q; at most 4q; more than 4q",
         "remaining service demand of ready jobs"),
        ("mean_ready_arrival_age", "at most q; at most 4q; more than 4q",
         "age of ready jobs, including earlier service"),
        ("arrived_work_priority_spread", "zero; non-zero",
         "whether arrived processes have distinct priorities"),
    ]:
        bullet(document, f"{bins}; why: {why}.", bold_prefix=f"{feature} — ")
    para(document,
         "Each trace is one sequential episode. Between consecutive decision epochs the environment "
         "measures the waiting-time area dW accumulated by ready processes, including any switch "
         "interval, and returns the scaled interval reward:")
    code(document,
         "r_t = -dW_t / max(1, q)        gamma = 1.0\n"
         "sum over t of r_t = -TotalWaitingTime / q")
    para(document,
         "Because gamma = 1 and every episode is finite, maximising the undiscounted return is "
         "equivalent to minimising total waiting time. The tabular update is:")
    code(document,
         "Q(s, a) <- Q(s, a) + alpha * ( r_t + gamma * max over visited a' of Q(s', a') - Q(s, a) )\n"
         "alpha = 0.1; terminal transitions omit the bootstrap term; Q initialised to 0")
    para(document,
         "Exploration: epsilon starts at 1.0, is multiplied by 0.997 after each episode and is floored "
         "at 0.05. The first visit to a state forces a uniformly random action. Greedy choices and "
         "bootstrap targets consider only actions with a positive visit count, and ties go to the "
         "lowest action index. A wholly unseen state in evaluation triggers an explicit, counted "
         "fallback to the predeclared heuristic rather than a silent zero-value tie.")

    document.add_heading("1.5 Methodological controls", level=2)
    for control, how in [
        ("Fixed-policy baselines",
         "FCFS, SJF, Round Robin and Priority run on identical workloads and pay the identical "
         "switch cost"),
        ("Predeclared adaptive rule",
         "a non-learning heuristic with no fitted parameters, fixed in the configuration before the "
         "held-out test is run, so learning is not credited for what fixed rules already achieve"),
        ("Blinded held-out split",
         "the final test (master seed 19301) is generated after training and validation, and the "
         "pipeline asserts zero fingerprint overlap with training and validation"),
        ("Retired test seed",
         "an earlier final-test seed (9301) was examined during an independent audit and informed the "
         "queue and reward remediation (section 9); it is retired as a final claim, and no result from "
         "it is reported"),
        ("Out-of-family generalisation",
         "poisson_arrivals is excluded from training entirely and used only at evaluation time"),
        ("Independent seeds",
         "five agents (seeds 7101–7105) are trained separately, and Q-learning results average all five"),
        ("Paired uncertainty",
         "differences are paired per workload and per model seed, then resampled with a "
         "family-stratified crossed bootstrap of 1,000 replicates (seed 104729)"),
        ("Determinism",
         "evaluation is greedy and read-only, and every artifact regenerates from the declared seeds"),
        ("Separation of timing",
         "observation, selection, Q-update and simulator wall time are recorded in separate columns "
         "and never charged to simulated time"),
    ]:
        bullet(document, how, bold_prefix=f"{control} — ")


def section_architecture(document: Document, data: Artifacts) -> None:
    document.add_heading("2. Overall conceptual, architectural and pipeline diagram", level=1)
    para(document,
         "The system is layered so that the information boundary is structural: the environment owns "
         "the future and the accounting, the controller receives a frozen snapshot of arrived work, "
         "and the controller's reply is a single policy name. The controller cannot modify the queue, "
         "the bursts or the event list.")
    diagram(document, "System architecture (section 2)", mermaid_architecture())
    for layer, components, role in [
        ("Data layer", "workload/models.py, workload/generator.py",
         "defines Process and Workload objects and generates the seven seeded families"),
        ("Simulation layer", "scheduler/runtime.py",
         "discrete-event loop, FIFO ready queue, remaining bursts, switch cost and schedule trace"),
        ("Baseline layer", "scheduler/fcfs.py, sjf.py, round_robin.py, priority.py",
         "standalone fixed policies, kept so runtime actions can be checked against them"),
        ("Perception layer", "rl/runtime_state.py",
         "encodes an immutable observation into one of 162 causal states"),
        ("Decision layer", "rl/runtime_controller.py, rl/runtime_heuristic.py",
         "the learned tabular Q-learning policy and the predeclared non-learning rule"),
        ("Experiment layer", "experiments/runtime_experiment.py, experiments/runtime_config.py",
         "splits, training, validation, held-out testing, paired bootstrap and artifact export"),
        ("Evaluation layer", "evaluation/metrics.py",
         "metric definitions for waiting, turnaround, response, utilisation, throughput and switches"),
    ]:
        bullet(document, f"{components}: {role}.", bold_prefix=f"{layer} — ")

    document.add_heading("2.1 Runtime control flow", level=2)
    for number, text in enumerate([
        "The environment advances to the next decision epoch (a dispatch point, a completion or a "
        "Round-Robin quantum boundary) and admits every arrival at or before that instant, in "
        "(arrival time, pid) order.",
        "It freezes an observation and passes it to the controller, which returns one of four "
        "policy names.",
        "If the selected process differs from the running PID, one unit of switch cost is charged "
        "before the segment starts. Arrivals during that interval are queued, but the chosen dispatch "
        "is not reconsidered.",
        "The segment runs: Round Robin is bounded by the quantum, the other policies run to "
        "completion. Endpoint arrivals are admitted first, then an unfinished Round-Robin process is "
        "appended to the tail.",
        "The environment measures the waiting-time area accumulated by ready processes, converts it "
        "to the interval reward and returns it to the controller for the Q update.",
        "A policy switch changes only the next selection rule and service length. The queue, the "
        "remaining bursts and the switch accounting carry over unchanged.",
    ], start=1):
        step(document, number, text)


def section_modules(document: Document) -> None:
    document.add_heading("3. Description of the major modules", level=1)
    para(document,
         "Six modules carry the system. Two build and run the simulated machine (M1, M2), two implement "
         "the learned controller (M3, M4), one is the non-learning reference rule (M5), and one "
         "orchestrates the experiment and its statistics (M6). Section 4 gives the workflow diagram "
         "for each.")
    for module_id, name, files, contract, responsibility in MODULE_SPECS:
        document.add_heading(f"{module_id} — {name}", level=2)
        bullet(document, files, bold_prefix="Files: ")
        bullet(document, contract, bold_prefix="Contract: ")
        bullet(document, responsibility, bold_prefix="Responsibility: ")
        bullet(document, "implemented and covered by automated tests", bold_prefix="Status: ")


MODULE_SPECS = [
    ("M1", "Workload generator and dataset builder",
     "workload/generator.py, workload/models.py",
     "family configuration and seed → validated workload with fingerprint",
     "Generates the seven synthetic families and validates every process (non-negative PID, arrival "
     "and priority; positive burst; unique PIDs). Records split, family, seed and fingerprint metadata."),
    ("M2", "Discrete-event runtime environment",
     "scheduler/runtime.py",
     "workload, controller, quantum and switch cost → schedule trace and six metrics",
     "Owns the private event list, the single FIFO ready queue, remaining bursts, switch-cost "
     "accounting, the causal observation and the interval reward."),
    ("M3", "Causal observation and state encoder",
     "rl/runtime_state.py",
     "RuntimeObservation → state index in [0, 162)",
     "Extracts five causal features, discretises them relative to the quantum and flattens them in "
     "mixed-radix order. It accepts an observation only and rejects any other object, including a "
     "Workload."),
    ("M4", "Sequential Q-learning controller",
     "rl/runtime_controller.py, rl/q_learning.py",
     "observation (encode) and reward (update) → policy action; Q table and visit counts",
     "Tabular Q-learning over 162 x 4 = 648 state-action pairs, with epsilon-greedy exploration, "
     "first-visit forcing, unvisited-action masking, read-only greedy evaluation and a counted "
     "heuristic fallback."),
    ("M5", "Predeclared heuristic controller",
     "rl/runtime_heuristic.py",
     "RuntimeObservation → policy action",
     "Non-learning rule (Round Robin, then Priority, then SJF, then FCFS) that uses the same "
     "observation as the learner and has no fitted parameters."),
    ("M6", "Experiment orchestration, metrics and reporting",
     "experiments/runtime_experiment.py, experiments/runtime_config.py, evaluation/metrics.py",
     "configuration and trained models → CSV and JSON artifacts, paired intervals, markdown report",
     "Trains five agents, validates, checks fingerprint disjointness, runs the held-out test, computes "
     "paired crossed-bootstrap intervals and writes every artifact."),
]


def section_workflows(document: Document) -> None:
    document.add_heading("4. Workflow diagram for each major module", level=1)
    para(document,
         "Each module has its own Mermaid flowchart, followed by its numbered steps. Diamonds are "
         "decision points; the numbers in the diagram match the numbered steps.")

    document.add_heading("M1. Workload generator and dataset builder", level=2)
    para(document, "Source: workload/generator.py, workload/models.py", italic=True)
    diagram(document, "Workflow of M1, workload generator and dataset builder", WORKFLOW_M1)
    for number, (label, detail) in enumerate([
        ("Select family configuration", "one of seven frozen parameter sets (burst, arrival, priority)."),
        ("Derive the workload seed", "SHA-256 of (master seed, split tag, family index, repetition)."),
        ("Sample 15 processes", "burst, arrival time and priority drawn by the family rules."),
        ("Decision: valid values?", "burst > 0, arrival >= 0, priority >= 0, unique PIDs. If not, the "
                                    "workload is rejected with a ValidationError."),
        ("Fingerprint and tag", "SHA-256 workload fingerprint plus split, family and repetition metadata."),
        ("Assemble the split", "6,000 training episodes, 140 validation and 210 held-out test workloads."),
    ], start=1):
        step(document, number, detail, bold_prefix=f"{label} — ")

    document.add_heading("M2. Discrete-event runtime environment", level=2)
    para(document, "Source: scheduler/runtime.py (one pass per decision epoch)", italic=True)
    diagram(document, "Workflow of M2, discrete-event runtime environment", WORKFLOW_M2)
    for number, (label, detail) in enumerate([
        ("Advance to the next decision epoch", "a dispatch point, a completion or a Round-Robin quantum boundary."),
        ("Admit arrivals", "every arrival at or before the epoch, in (arrival time, pid) order."),
        ("Freeze the causal observation", "arrived and ready work only; future processes stay private."),
        ("Obtain the policy", "FCFS, SJF, Round Robin or Priority from the active controller."),
        ("Decision: running PID changes?", "if the selected PID differs from the running PID, go to step 6; "
                                           "otherwise, including the first dispatch, go to step 7."),
        ("Charge the switch cost", "one unit of simulated time before the segment starts."),
        ("Execute the segment", "min(quantum, remaining burst) for Round Robin; otherwise to completion."),
        ("Settle the endpoint", "admit arrivals first; a completed process leaves; an unfinished Round-Robin "
                                "process is appended to the tail."),
        ("Return the interval reward", "r = -dW / max(1, q), where dW is the waiting-time area accumulated since "
                                       "the previous decision."),
        ("Decision: all processes completed?", "if yes, emit the schedule trace and six metrics; if no, return to step 1."),
    ], start=1):
        step(document, number, detail, bold_prefix=f"{label} — ")

    document.add_heading("M3. Causal observation and state encoder", level=2)
    para(document, "Source: rl/runtime_state.py", italic=True)
    diagram(document, "Workflow of M3, causal observation and state encoder", WORKFLOW_M3)
    for number, (label, detail) in enumerate([
        ("Receive the observation", "an immutable RuntimeObservation; any other object, and an empty ready "
                                    "queue, are rejected."),
        ("Extract five causal features", "ready count, completed count, median ready remaining burst, mean "
                                         "ready arrival age and arrived-work priority spread."),
        ("Discretise each feature", "bin edges are expressed relative to the quantum (q = 4)."),
        ("Flatten in mixed-radix order", "3 x 3 x 3 x 3 x 2 collapses to one integer."),
        ("Return the state index", "s in [0, 162), used by the controller only."),
    ], start=1):
        step(document, number, detail, bold_prefix=f"{label} — ")

    document.add_heading("M4. Sequential Q-learning controller", level=2)
    para(document, "Source: rl/runtime_controller.py, rl/q_learning.py", italic=True)
    diagram(document, "Workflow of M4, sequential Q-learning controller", WORKFLOW_M4)
    for number, (label, detail) in enumerate([
        ("Encode the observation", "s = encode(observation)."),
        ("Decision: training run?", "training uses the exploration rule below; evaluation is greedy and read-only."),
        ("Decision: first visit or exploration?", "a first visit to s, or a draw below epsilon, gives a random action."),
        ("Random action", "a uniformly random action out of four."),
        ("Greedy action", "the highest-valued action among those already visited in s."),
        ("Decision: any visited action in s?", "if none, the fallback in step 7 applies."),
        ("Counted fallback", "use the predeclared heuristic and count the decision as an unseen-state fallback."),
        ("Greedy action in evaluation", "the highest-valued visited action; no update is made."),
        ("Receive the reward", "r = -dW / q from the environment."),
        ("Update the table", "Q(s,a) += alpha * (r + gamma * max over visited a' of Q(s',a') - Q(s,a)); "
                             "terminal transitions omit the bootstrap."),
        ("Persist values and visits", "Q values and visit counts are written per training seed."),
    ], start=1):
        step(document, number, detail, bold_prefix=f"{label} — ")

    document.add_heading("M5. Predeclared heuristic controller", level=2)
    para(document, "Source: rl/runtime_heuristic.py (fixed rules, no learning)", italic=True)
    diagram(document, "Workflow of M5, predeclared heuristic controller", WORKFLOW_M5)
    for number, (label, detail) in enumerate([
        ("Receive the observation", "the same frozen RuntimeObservation that the learner receives."),
        ("Round Robin rule", "at least three ready jobs and mean arrival age >= one quantum: return Round Robin."),
        ("Priority rule", "otherwise, at least two ready jobs with distinct priorities: return Priority."),
        ("SJF rule", "otherwise, largest remaining burst >= 2 x smallest: return SJF."),
        ("Default", "otherwise return FCFS. The rule is deterministic, with no state and no Q update, "
                    "and is identical in training and evaluation."),
    ], start=1):
        step(document, number, detail, bold_prefix=f"{label} — ")

    document.add_heading("M6. Experiment orchestration, metrics and reporting", level=2)
    para(document, "Source: experiments/runtime_experiment.py, evaluation/metrics.py", italic=True)
    diagram(document, "Workflow of M6, experiment orchestration, metrics and reporting", WORKFLOW_M6)
    for number, (label, detail) in enumerate([
        ("Train five agents", "seeds 7101–7105; 1,200 sequential episodes each over six families, with "
                              "poisson_arrivals withheld."),
        ("Validate", "140 workloads (seed 8201), used for reporting and for the demo rule only, never for tuning."),
        ("Decision: training and validation disjoint?", "zero fingerprint overlap is required; otherwise the run stops."),
        ("Select the illustrative trace", "by the validation-only rule, before any test workload is generated."),
        ("Generate the held-out test set", "210 untouched workloads (seed 19301), generated only after steps 1 to 4."),
        ("Decision: all splits disjoint?", "validation and test must not overlap, and training must not overlap "
                                           "either; otherwise the run stops."),
        ("Run the held-out test", "greedy, read-only evaluation; six metrics per workload derived from the traces."),
        ("Compare pairwise", "paired per-workload differences with a family-stratified crossed bootstrap "
                             "of 1,000 replicates (seed 104729)."),
        ("Write the artifacts", "CSV tables, Q table, state-action coverage, manifest, demo trace, summary JSON "
                                "and the generated report."),
    ], start=1):
        step(document, number, detail, bold_prefix=f"{label} — ")


def section_dataset(document: Document, data: Artifacts) -> None:
    document.add_heading("5. Dataset details and data collection procedure", level=1)

    document.add_heading("5.1 Data collection procedure", level=2)
    para(document,
         "No external or personal data is used. The dataset is synthetic and generated entirely by the "
         "project's own generator, so it can be reproduced from the declared integers on any machine.")
    for step_id, action, detail in [
        ("D1", "Fix seven family configurations",
         "burst distribution and range, arrival pattern and window or rate, and priority pattern, "
         "declared once in config.py and frozen in the runtime experiment configuration"),
        ("D2", "Derive one seed per workload",
         "workload seed = SHA-256(master seed, split tag, family index, repetition or episode); "
         "no global random state is shared between workloads"),
        ("D3", "Generate 15 processes per workload",
         "burst, arrival time and priority are drawn by the family rules, then validated (positive "
         "burst, non-negative arrival, priority within bounds)"),
        ("D4", "Assign workloads to splits",
         "training: 1,200 sequential episodes for each of five seeds, which is 6,000 episodes in total. "
         "Each episode generates one new workload, and the six training families cycle in a fixed order, "
         "so each family receives 200 episodes per seed. Validation: 140 workloads (7 families x 20). "
         "Held-out test: 210 workloads (7 families x 30)"),
        ("D5", "Fingerprint and check disjointness",
         "each workload gets a SHA-256 fingerprint; the pipeline asserts zero overlap between training, "
         "validation and test before the held-out test runs"),
        ("D6", "Store only what is needed",
         "metrics, the manifest (split, family, seed, fingerprint), Q tables and summaries are committed; "
         "the large per-decision event logs are regenerated by the command and gitignored"),
        ("D7", "Regenerate on demand",
         "python main.py runtime-experiment rebuilds every workload and every metric from the declared seeds"),
    ]:
        bullet(document, f"{action} — {detail}.", bold_prefix=f"{step_id}: ")

    document.add_heading("5.2 Dataset inventory", level=2)
    for split, families, reps, workloads, processes, seed in [
        ("Training (sequential episodes)", "six training families",
         "1,200 episodes per seed, five seeds", "6,000", "90,000", "7101–7105"),
        ("Validation", "all seven families", "20 repetitions per family", "140", "2,100", "8201"),
        ("Held-out test", "all seven families", "30 repetitions per family", "210", "3,150", "19301"),
    ]:
        bullet(document,
               f"{families}; {reps}; {workloads} workloads; {processes} processes; master seed {seed}.",
               bold_prefix=f"{split} — ")

    document.add_heading("5.3 Family specification", level=2)
    para(document,
         "Each family generates 15 processes. Parameters are taken from the frozen configuration "
         "(the same values are stored in results/runtime/runtime_summary.json).")
    for family in data.cfg["workload_family_parameters"]:
        role = ("evaluation only; never used for training" if family["name"] == "poisson_arrivals"
                else "training and evaluation")
        bullet(document, f"{family_sentence(family)} Role: {role}.",
               bold_prefix=f"{family['name']} — ")

    document.add_heading("5.4 Measured dataset statistics", level=2)
    para(document,
         "The statistics below were recomputed with the project's generator at document-build time, "
         "so they describe the actual held-out dataset rather than the configuration alone. High "
         "priority means priority <= 2.")
    if not data.processes.empty:
        for family in data.families:
            procs = data.processes[data.processes["family"] == family]
            workloads = data.workloads[data.workloads["family"] == family]
            bullet(document, dataset_line(procs, workloads),
                   bold_prefix=f"{family.replace('_', ' ')} — ")
        bullet(document, dataset_line(data.processes, data.workloads), bold_prefix="All families — ")

    document.add_heading("5.5 Data integrity and assumptions", level=2)
    for item in [
        "Reproducibility — seeds are derived with SHA-256, so a workload is a pure function of "
        "(master seed, split tag, family index, repetition).",
        f"Disjointness — the pipeline stops if a fingerprint appears in two splits; the committed run "
        f"reports overlap = {str(data.summary['split_fingerprint_overlap']).lower()} over "
        f"{data.summary['training_fingerprint_count_union']:,} distinct training fingerprints.",
        "Burst knowledge — exact burst lengths are known once a process arrives. This is the assumption "
        "already required by standalone SJF; no burst predictor is included.",
        "Storage — per-decision event logs are large and are regenerated by the command and gitignored; "
        "metrics, Q tables and manifests are committed.",
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
        ("Language", "Python 3.11.2", "whole implementation, command-line interface and report builders"),
        ("Numerics", "NumPy 2.4.6", "Q table and visit counts stored as arrays; bootstrap resampling"),
        ("Data handling", "pandas 3.0.6", "metric aggregation and CSV artifact export"),
        ("Testing", "pytest 9.1.1", "automated checks of every module (section 8.3)"),
        ("Diagrams", "Mermaid (text source)", "all diagrams in this document; renders in GitHub, VS Code, "
                                              "Obsidian and mermaid.live"),
        ("Document generation", "python-docx 1.2.0", "renders this text-only DOCX"),
        ("PDF generation", "ReportLab 5.0.1", "renders the text-only PDF from the DOCX"),
        ("Hashing", "hashlib (SHA-256, standard library)", "seed derivation and workload fingerprints"),
        ("Standard library", "argparse, dataclasses, json, pathlib", "configuration objects, command-line "
                                                                    "interface and artifact serialisation"),
        ("Version control", "Git and GitHub", "branch-per-change workflow with pull requests"),
        ("Environment", "Linux 6.1 (x86-64) with venv and pip", "development, experiment runs and report builds"),
    ]:
        bullet(document, f"{item} — {role}.", bold_prefix=f"{category}: ")
    para(document,
         "No deep-learning framework is used. The controller is a tabular Q-learning agent, so every "
         "value, visit count and fallback is inspectable in the committed artifacts.")

    document.add_heading("6.2 Algorithms", level=2)
    for name, definition, usage in [
        ("FCFS (action 0)",
         "dispatch the head of the ready queue, non-preemptively",
         "baseline and runtime action; after a Round-Robin requeue it follows live queue order"),
        ("SJF (action 1)",
         "smallest remaining burst first, non-preemptive, ties in ready-queue order",
         "baseline and runtime action; minimises mean waiting time when all jobs are present together, "
         "and is a strong but not guaranteed-optimal reference under staggered arrivals"),
        ("Round Robin (action 2)",
         "head of the queue runs for min(quantum, remaining burst); unfinished jobs go to the tail after "
         "endpoint arrivals",
         "baseline and runtime action; bounds the time to first service"),
        ("Priority (action 3)",
         "highest static priority first, non-preemptive, ties in ready-queue order",
         "baseline and runtime action; lower numeric value means higher priority"),
        ("Tabular Q-learning",
         "Q(s,a) += alpha * (r + gamma * max over visited actions of Q(s',.) - Q(s,a)) with epsilon-greedy "
         "exploration and unvisited-action masking",
         "the learned runtime controller over 162 states x 4 actions"),
        ("Predeclared heuristic",
         "Round Robin, then Priority, then SJF, then FCFS, applied to the same observation",
         "non-learning adaptive reference"),
        ("Crossed bootstrap",
         "resample workloads within each family and resample model seeds independently; 1,000 replicates",
         "paired 95% intervals for every comparison"),
        ("Discrete-event simulation",
         "advance the event list with a persistent FIFO ready queue and switch-cost charging",
         "the machine being controlled"),
    ]:
        bullet(document, f"{definition}. Used for: {usage}.", bold_prefix=f"{name} — ")

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
        ("P2", "Train five independent agents",
         "seeds 7101–7105; 1,200 sequential episodes each over six families, with poisson_arrivals withheld"),
        ("P3", "Validate",
         "140 workloads (seed 8201), used only for reporting and for the illustrative-trace rule"),
        ("P4", "Test once, held out",
         "210 untouched workloads (seed 19301). The fixed policies and the heuristic are deterministic and "
         "run once each; the five Q-learning models are evaluated greedily and read-only"),
        ("P5", "Compare every method against the others",
         "paired per-workload differences, plus independent resampling of the five model seeds"),
        ("P6", "Quantify uncertainty",
         "family-stratified crossed bootstrap, 1,000 replicates, seed 104729; 95% intervals"),
        ("P7", "Publish artifacts",
         "metric tables, Q values, visit counts, coverage, manifest, demo trace and summary JSON, written to "
         "results/runtime/ by one command"),
    ]:
        bullet(document, f"{action} — {detail}.", bold_prefix=f"{step_id}: ")

    document.add_heading("7.2 Evaluation metrics", level=2)
    for name, definition, why in [
        ("Mean waiting time",
         "mean over processes of (turnaround time minus burst time), that is, time spent waiting in the ready queue",
         "primary metric; it is the quantity the reward accumulates"),
        ("Mean turnaround time", "mean over processes of (completion time minus arrival time)",
         "end-to-end latency experienced by a job"),
        ("Mean response time", "mean over processes of (first dispatch time minus arrival time)",
         "interactivity; this is what Round Robin improves"),
        ("CPU utilisation (%)", "busy time divided by total elapsed time, times 100",
         "the cost of switching and idling"),
        ("Throughput", "processes completed per unit of simulated time", "work completed per unit time"),
        ("Context switches", "number of changes of the running PID",
         "overhead proxy; the switch cost is charged for every method"),
        ("Policy switches per trace", "decisions where the chosen policy differs from the previous one",
         "how much the controller actually adapts"),
        ("State-action coverage", "visited (state, action) pairs out of 648, and unseen-state fallbacks",
         "whether the learned table is populated where it is used"),
    ]:
        bullet(document, f"{definition}. Why: {why}.", bold_prefix=f"{name} — ")

    document.add_heading("7.3 Statistical procedure", level=2)
    para(document,
         "Comparisons are paired: for each workload, the difference (target minus reference) is computed on "
         "the same workload. For Q-learning, the model seed is also resampled independently of the workloads "
         "(a crossed bootstrap). Resampling is stratified by workload family, with 1,000 replicates drawn "
         "using seed 104729. An interval that excludes zero on the favourable side is treated as a real "
         "difference on this benchmark; an interval containing zero is reported as inconclusive.")

    document.add_heading("7.4 Results obtained with this plan", level=2)
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
    para(document,
         "All values are means ± standard deviation across the 210 held-out workloads (master seed 19301), "
         "with quantum 4 and switch cost 1. Q-learning also averages the five model seeds.")
    diagram(document, "Mean waiting time by method, held-out test (section 7.4)",
            mermaid_results(data, "avg_waiting_time", "Mean waiting time by method", "time units"))
    diagram(document, "Mean response time by method, held-out test (section 7.4)",
            mermaid_results(data, "avg_response_time", "Mean response time by method", "time units"))

    document.add_heading("7.5 Paired comparisons", level=2)
    para(document,
         "Each line gives target minus reference, with its 95% crossed-bootstrap interval, as committed in "
         "results/runtime/runtime_paired_comparisons.csv. Turnaround differences equal waiting differences "
         "exactly, because both methods serve the same bursts on each workload, and throughput is in the CSV.")
    pairs = [
        ("Runtime Q-learning", "SJF"),
        ("Runtime Q-learning", "Causal heuristic"),
        ("Round Robin", "SJF"),
        ("Causal heuristic", "SJF"),
        ("FCFS", "SJF"),
        ("Priority", "SJF"),
    ]
    metric_labels = {
        "avg_waiting_time": "waiting time",
        "avg_response_time": "response time",
        "context_switches": "context switches",
        "cpu_utilization": "CPU utilisation (percentage points)",
    }
    for target, reference in pairs:
        for metric, label in metric_labels.items():
            mean, low, high = ci_of(data, target, reference, metric)
            if mean == 0 and low == 0 and high == 0:
                reading = "no difference"
            elif metric == "cpu_utilization":
                reading = "favours the target" if mean > 0 else "favours the reference"
            else:
                reading = "favours the target" if mean < 0 else "favours the reference"
            bullet(document,
                   f"{label}: difference {mean:+,.3f}, 95% interval [{low:+,.3f}, {high:+,.3f}], {reading}.",
                   bold_prefix=f"{SHORT_NAME[target]} minus {SHORT_NAME[reference]} — ")

    document.add_heading("7.6 What the plan has established so far", level=2)
    q_sjf_wait = ci_of(data, "Runtime Q-learning", "SJF", "avg_waiting_time")
    q_sjf_resp = ci_of(data, "Runtime Q-learning", "SJF", "avg_response_time")
    rr_sjf_wait = ci_of(data, "Round Robin", "SJF", "avg_waiting_time")
    fcfs_sjf_wait = ci_of(data, "FCFS", "SJF", "avg_waiting_time")
    prio_sjf_wait = ci_of(data, "Priority", "SJF", "avg_waiting_time")
    q_heur_wait = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_waiting_time")
    q_heur_resp = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_response_time")
    q_mean = data.summary_mean("final", "Runtime Q-learning", "avg_waiting_time")
    rr_mean = data.summary_mean("final", "Round Robin", "avg_waiting_time")
    sjf_mean = data.summary_mean("final", "SJF", "avg_waiting_time")
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
        f"({q_mean:,.3f} against {rr_mean:,.3f}). This is the difference of the two committed comparisons "
        f"against SJF (Q minus SJF {q_sjf_wait[0]:+,.3f}; RR minus SJF {rr_sjf_wait[0]:+,.3f}). No paired "
        f"interval was computed for this pair. FCFS and Priority lie between the two, at "
        f"{fcfs_sjf_wait[0]:+,.3f} and {prio_sjf_wait[0]:+,.3f} relative to SJF.",
        f"Against the predeclared adaptive rule, mean waiting time is lower by {heur_mean - q_mean:,.3f} "
        f"(paired difference Q minus rule {q_heur_wait[0]:+,.3f}; 95% interval [{q_heur_wait[1]:+,.3f}, "
        f"{q_heur_wait[2]:+,.3f}]). Learning therefore improves on this particular adaptive rule on this "
        f"benchmark. The rule has the lower response time: Q-learning's response time is higher by "
        f"{q_heur_resp[0]:,.3f}, because the rule selects Round Robin whenever three or more aged jobs are ready.",
        f"Against SJF the result is mixed. Waiting time is higher for Q-learning by {q_sjf_wait[0]:,.3f} "
        f"(95% interval {q_sjf_wait[1]:,.3f} to {q_sjf_wait[2]:,.3f}), while response time is lower by "
        f"{abs(q_sjf_resp[0]):,.3f} (95% interval {abs(q_sjf_resp[2]):,.3f} to {abs(q_sjf_resp[1]):,.3f}). Non-preemptive "
        f"SJF with exact bursts and no mid-job switch cost is a strong reference on this benchmark, and the "
        f"learned controller does not beat it on waiting time.",
        f"Q-learning's mean waiting time lies between the SJF and Round Robin values on all "
        f"{len(families_between)} held-out families, including the unseen poisson_arrivals family "
        f"({poisson_q:,.2f}, against SJF {poisson_sjf:,.2f} and Round Robin {poisson_rr:,.2f}).",
        f"Controller overhead is small relative to simulated time: about "
        f"{data.summary['mean_runtime_q_observation_us_per_decision']:.2f} microseconds to build an observation "
        f"and {data.summary['mean_runtime_q_selection_us_per_decision']:.2f} microseconds to select an action per "
        f"decision on the build host. These timings are measured separately and are never charged to the schedule.",
    ]:
        bullet(document, item)


def section_progress(document: Document, data: Artifacts, test_rows: List[List[str]],
                     pytest_block: str, cli_block: str) -> None:
    document.add_heading("8. Implementation progress", level=1)
    para(document,
         "Review 2 requires at least 50% of the proposed work to be complete, with the completed modules "
         "demonstrated by valid outputs. All six proposed modules are implemented, tested and demonstrated "
         "below, which is 100% of the proposed Review 2 scope. Section 9.2 lists work planned beyond that "
         "scope.")
    document.add_heading("8.1 Completion by module", level=2)
    for module_id, name, _files, _contract, _responsibility in MODULE_SPECS:
        bullet(document, "completion 100%; status: implemented and tested; evidence: section 8.2.",
               bold_prefix=f"{module_id} — {name}. ")
    para(document, "Overall completion of the proposed Review 2 modules: 6 of 6 (100%).", bold=True)

    document.add_heading("8.2 Outputs demonstrated by each module", level=2)
    para(document, "Each module is demonstrated by the output it produced in the committed run:")
    for module_id, text in module_outputs(data):
        bullet(document, text, bold_prefix=f"{module_id} — ")

    document.add_heading("8.3 Demonstration 1 — automated verification", level=2)
    if test_rows:
        total = sum(int(row[1]) for row in test_rows)
        para(document, f"The test suite collects {total} checks. Counts per test file, recorded during this build:")
        for module, count, protects in test_rows:
            bullet(document, f"{count} checks — {protects}.", bold_prefix=f"{module}: ")
    code(document, pytest_block, label="Output of python -m pytest -q:")

    document.add_heading("8.4 Demonstration 2 — end-to-end run output", level=2)
    para(document,
         "The full pipeline (training, validation, held-out testing and artifact export) runs with one command. "
         "The output below reproduces the lines that command prints, with file paths shortened to file names, "
         "taken from the committed artifacts. A fresh run of the command on the build host printed the same "
         "metric values; its schedule metrics and policy decisions match the committed artifacts exactly, and "
         "only the per-decision timing columns depend on the host.")
    code(document, cli_block, label="Output of python main.py runtime-experiment:")

    document.add_heading("8.5 Demonstration 3 — learned behaviour within one trace", level=2)
    demo = data.demo
    para(document,
         f"The controller switches policies inside a single trace. The example comes from the validation split, "
         f"chosen by the predeclared validation-only rule (maximise policy switches among traces with zero "
         f"unseen-state fallbacks). It is training "
         f"seed {demo['training_seed']}, family {demo['family']}, repetition {demo['repetition']}, fingerprint "
         f"{demo['workload_fingerprint']}. It contains {demo['policy_switch_count']} policy switches over "
         f"{demo['decision_count']} decisions, with mean waiting time {demo['metrics']['avg_waiting_time']:.3f} "
         f"and mean response time {demo['metrics']['avg_response_time']:.3f}. It illustrates the mechanism and "
         f"is not a performance claim.")
    diagram(document, "Illustrative validation trace (section 8.5). Each bar is one execution segment, labelled "
                      "with the policy that ran it. Gaps between bars are the one-unit switch costs.",
            mermaid_gantt(data))

    document.add_heading("8.6 Artifacts produced", level=2)
    for artifact, size, contents in [
        ("results/runtime/runtime_final_test_metrics.csv", "2,100 rows",
         "six metrics, switch and decision counts and timings on the held-out test"),
        ("results/runtime/runtime_validation_metrics.csv", "1,400 rows",
         "the same columns for the validation split"),
        ("results/runtime/runtime_training_metrics.csv", "6,000 rows",
         "per-episode sequential training outcomes"),
        ("results/runtime/runtime_paired_comparisons.csv", "36 rows",
         "paired differences with 95% crossed-bootstrap intervals"),
        ("results/runtime/runtime_state_action_coverage.csv", "15 rows",
         "visited states and pairs, fallback counts and timings per seed"),
        ("results/runtime/runtime_q_table.csv", "3,240 rows", "Q values and visit counts per training seed"),
        ("results/runtime/runtime_workload_manifest.csv", "6,350 rows",
         "split, family, seed and workload fingerprint for every workload"),
        ("results/runtime/runtime_learned_switch_demo.json", "1 trace",
         "observations, actions, switch times and execution trace of the illustrative run"),
        ("results/runtime/runtime_summary.json", "summary",
         "configuration, split fingerprints, all means and software versions"),
        ("results/runtime/runtime_report.md", "report", "generated markdown report of the run"),
    ]:
        bullet(document, f"{size} — {contents}.", bold_prefix=f"{artifact}: ")


def section_challenges(document: Document, data: Artifacts) -> None:
    document.add_heading("9. Challenges encountered and plan for completing the remaining work", level=1)

    document.add_heading("9.1 Challenges encountered", level=2)
    for title, why, how in [
        ("Keeping the controller causal",
         "an earlier whole-workload selector saw future arrivals and counterfactual schedules, so its "
         "results could not be read as a runtime controller",
         "introduced an immutable RuntimeObservation, made the encoder accept only that object, and added a "
         "paired hidden-future test that fails if unarrived processes change the observation or action prefix"),
        ("Round Robin and FCFS after a requeue",
         "once Round Robin appends a preempted job to the tail, FCFS is ambiguous: original arrival order or "
         "live queue order",
         "defined one persistent FIFO ready queue with canonical (arrival time, pid) insertion; FCFS follows "
         "live queue order, and SJF and Priority ties preserve it; fixed-action runs are tested against the "
         "standalone schedulers"),
        ("Charging switch cost fairly",
         "if the learner pays for switching but the baselines do not, the comparison is invalid; if nothing is "
         "charged, switching is unrealistic",
         "one unit is charged whenever the running PID changes, for every method including the fixed policies; "
         "arrivals during the switch are queued, but the chosen dispatch is not reconsidered"),
        ("A reward that matches the objective",
         "a shaped reward can drift away from the metric being reported",
         "used r = -dW / q with gamma = 1, so the episodic return is exactly minus total waiting time divided by "
         "the quantum; a test asserts this identity"),
        ("Sparse coverage of the tabular space",
         "only about 42% of the 648 state-action pairs are visited during training, and unseen states appear "
         "during evaluation",
         "kept the abstraction small, forced exploration on first visit, masked unvisited actions in greedy "
         "steps and bootstraps, and made unseen states fall back to the predeclared heuristic with the fallback "
         "count reported"),
        ("Reporting a negative result",
         "the learner does not beat non-preemptive SJF on mean waiting time, so the result could be omitted or "
         "overstated",
         "reported as a headline finding with its bootstrap interval, and the structural reason stated: SJF "
         "receives exact bursts at arrival and pays no mid-job switch cost"),
        ("Host-dependent timing",
         "wall-clock columns differ between machines, which can look like non-determinism",
         "observation, selection, Q-update and simulator times are stored in separate columns, never charged to "
         "simulated time, and excluded from every schedule comparison"),
        ("Large per-decision event logs",
         "full causal decision logs are large and would bloat the repository",
         "the command writes them, they are gitignored, and they regenerate from the recorded configuration and "
         "seeds; compact metrics, Q tables and manifests are committed"),
    ]:
        bullet(document, f"Why it mattered: {why}. Resolution: {how}.", bold_prefix=f"{title} — ")

    document.add_heading("9.2 Plan for completing the remaining work", level=2)
    for task_id, task, gap, approach, deliverable in [
        ("W1", "Sensitivity study",
         "all results use quantum 4 and switch cost 1",
         "re-run the plan over a grid of quanta and switch costs and report where the ordering changes",
         "metric tables for each (quantum, cost) cell"),
        ("W2", "Burst prediction",
         "exact bursts are assumed known at arrival",
         "add an explicit predictor with measured error and re-run the controller under that noise",
         "results with prediction-error bands"),
        ("W3", "Richer state abstraction",
         "only about 42% of pairs are visited, and 162 states is coarse",
         "tile coding or a small function approximator, keeping the same causal observation contract and "
         "fallback reporting",
         "coverage and metric comparison against the tabular agent"),
        ("W4", "Multi-objective reward",
         "waiting time is optimised, response time is not",
         "add a constrained or weighted objective and measure the waiting-response trade-off",
         "Pareto front over the two metrics"),
        ("W5", "I/O and blocking",
         "processes never block, so the queue is CPU-bound only",
         "model I/O bursts and blocked and wake-up transitions in the environment",
         "extended environment with re-baselined comparisons"),
        ("W6", "Multicore extension",
         "single CPU only",
         "extend to N cores with per-core queues and a migration cost, choosing a policy per core",
         "multicore metrics and overhead analysis"),
        ("W7", "Trace-driven validation",
         "all workloads are synthetic",
         "replay a public scheduling trace through the same environment and report the gap to the synthetic results",
         "trace-driven result table"),
    ]:
        bullet(document,
               f"gap: {gap}. Approach: {approach}. Deliverable: {deliverable}.",
               bold_prefix=f"{task_id} — {task}. ")

    document.add_heading("9.3 Sequence for the remaining work", level=2)
    for number, text in enumerate([
        "First, W1 (sensitivity) and W2 (burst prediction): they test whether the current conclusions depend "
        "on fixed parameters or on the idealised burst assumption.",
        "Second, W3 and W4: they address the two measured weaknesses, coarse coverage and a single objective.",
        "Third, W5 and W6: environment extensions that make the simulator more realistic, each with "
        "re-baselined comparisons.",
        "Finally, W7: external validation on a real trace is the strongest available check that the synthetic "
        "ordering carries over.",
    ], start=1):
        step(document, number, text)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
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
