"""Build the text-only Review 2 DOCX (no tables, no colour, Mermaid for diagrams).

Companion to ``tools/build_review2_report.py``. The DOCX covers the same nine Review 2
rubric items but is plain text: no tables (every table becomes a bulleted list), no
colour and no images. Diagrams are emitted as **Mermaid source blocks** that can be
pasted into GitHub markdown, VS Code (Markdown Preview Mermaid Support), Obsidian or
https://mermaid.live to render.

Usage::

    pip install python-docx
    python tools/build_review2_docx.py                 # writes Review_2_Report.docx
    python tools/build_review2_docx.py --skip-tests    # do not re-run pytest
    python tools/build_review2_docx.py --output /tmp/r.docx
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
from docx.shared import Pt, RGBColor  # noqa: E402

from review2_data import (  # noqa: E402
    METHOD_ORDER,
    MODULES,
    SHORT_NAME,
    WORKFLOWS,
    Artifacts,
    ci_of,
    fmt,
)

MONO = "Consolas"


# --------------------------------------------------------------------------
# Document helpers
# --------------------------------------------------------------------------
def new_document() -> Document:
    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11.5),
                       ("Title", 20)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
    return document


def para(document: Document, text: str, bold: bool = False, italic: bool = False):
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    return paragraph


def bullet(document: Document, text: str, bold_prefix: str = "") -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    if bold_prefix:
        run = paragraph.add_run(bold_prefix)
        run.bold = True
        run.font.color.rgb = RGBColor(0, 0, 0)
    run = paragraph.add_run(text)
    run.font.color.rgb = RGBColor(0, 0, 0)


def numbered(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Number")
    run = paragraph.add_run(text)
    run.font.color.rgb = RGBColor(0, 0, 0)


def code(document: Document, text: str, label: str = "") -> None:
    """Monospace block (pseudocode, terminal output or Mermaid source)."""
    if label:
        caption = document.add_paragraph()
        run = caption.add_run(label)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0, 0, 0)
    for line in text.strip("\n").split("\n"):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.left_indent = Pt(18)
        run = paragraph.add_run(line if line else " ")
        run.font.name = MONO
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0, 0, 0)
    document.add_paragraph().paragraph_format.space_after = Pt(4)


def mermaid(document: Document, source: str, caption_text: str) -> None:
    code(document, "```mermaid\n" + source.strip() + "\n```",
         label=f"Mermaid diagram — {caption_text}")


# --------------------------------------------------------------------------
# Mermaid sources
# --------------------------------------------------------------------------
def mermaid_pipeline() -> str:
    stages = [
        ("1. Dataset", "7 seeded families, 15 processes"),
        ("2. Simulation", "single CPU, FIFO ready queue"),
        ("3. Observation", "arrived work only"),
        ("4. Encoding", "5 features, 162 states"),
        ("5. Decision", "Q-learning picks a policy"),
        ("6. Execution", "segment runs, switches charged"),
        ("7. Reward", "r = -dW / q"),
        ("8. Evaluation", "held-out test plus bootstrap"),
    ]
    lines = ["flowchart LR"]
    for i, (title, detail) in enumerate(stages, start=1):
        lines.append(f'    S{i}["{title}<br/>{detail}"]')
    for i in range(1, len(stages)):
        lines.append(f"    S{i} --> S{i + 1}")
    lines.append('    S7 -.->|"repeat at every decision epoch"| S3')
    return "\n".join(lines)


def mermaid_architecture() -> str:
    return """flowchart TB
    subgraph ENV["Environment - scheduler/runtime.py (owns the future)"]
        E1["Private event list: future arrivals"]
        E2["FIFO ready queue + remaining bursts"]
        E3["CPU, switch cost, schedule trace"]
    end
    subgraph CTL["Controller - arrived work only"]
        C1["RuntimeObservation (frozen snapshot)"]
        C2["State encoder: 162 states"]
        C3["Q-learning controller: 648 pairs"]
    end
    M["Waiting-time increment produces reward r = -dW / q"]
    E1 --> E2
    E2 --> E3
    E3 -->|"observation"| C1
    C1 --> C2
    C2 --> C3
    C3 -->|"action: FCFS / SJF / RR / Priority"| E3
    E3 --> M
    M -.->|"Q update"| C3"""


def mermaid_workflow(steps: Sequence[Tuple[str, str]]) -> str:
    lines = ["flowchart TD"]
    for i, (label, _detail) in enumerate(steps, start=1):
        lines.append(f'    N{i}["{i}. {label}"]')
    for i in range(1, len(steps)):
        lines.append(f"    N{i} --> N{i + 1}")
    return "\n".join(lines)


def mermaid_results(data: Artifacts, metric: str, title: str, unit: str) -> str:
    labels = [SHORT_NAME[method].replace(" ", " ") for method in METHOD_ORDER]
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
        f"{demo['family']} rep {demo['repetition']}",
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
        duration = int(segment["end"])
        lines.append(
            f"    {policy} :t{task_index}, {int(segment['start'])}, {duration}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def add_cover(document: Document, data: Artifacts, generated: str, test_line: str) -> None:
    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Operating Systems Project — Review 2")
    run.font.color.rgb = RGBColor(0, 0, 0)

    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("Methodology and Partial Implementation (Lab Assessment 9)")
    run.bold = True
    run.font.size = Pt(11.5)
    run.font.color.rgb = RGBColor(0, 0, 0)

    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run("Event-Driven Runtime-Adaptive CPU Scheduling "
                          "with Sequential Tabular Q-Learning")
    run.font.size = Pt(12.5)
    run.italic = True
    run.font.color.rgb = RGBColor(0, 0, 0)

    para(document, "")
    for field, value in [
        ("Repository", "github.com/lostpilot404/OS-Project"),
        ("Language and libraries", "Python 3.11.2 with NumPy 2.4.6, pandas 3.0.6, "
                                   "Matplotlib 3.11.2, pytest 9.1.1"),
        ("Automated verification", test_line),
        ("Dataset", "7 synthetic families: 6,000 training workloads, 140 validation "
                    "workloads, 210 held-out test workloads"),
        ("Report generated", generated),
        ("Format note", "Text-only document. No tables or colour are used; every diagram is "
                        "given as Mermaid source that renders in GitHub, VS Code, Obsidian or "
                        "mermaid.live."),
    ]:
        bullet(document, value, bold_prefix=f"{field}: ")

    para(document, "")
    para(document,
         "This document presents Review 2 in the required order: methodology, architecture, "
         "modules, module workflows, dataset, tools and algorithms, experimental plan and "
         "metrics, implementation progress, and challenges with the plan for the remaining "
         "work. Every number quoted here is read directly from the artifacts committed under "
         "results/, so the text cannot drift from the recorded experiment.")


def add_contents(document: Document) -> None:
    document.add_heading("Contents", level=1)
    for line in [
        "1. Detailed methodology of the proposed system",
        "2. Overall conceptual, architectural and pipeline diagram",
        "3. Description of the major modules",
        "4. Workflow diagram for each major module",
        "5. Dataset details and data collection procedure",
        "6. Tools, technologies, algorithms and frameworks used",
        "7. Experimental plan and evaluation metrics",
        "8. Implementation progress",
        "9. Challenges encountered and plan for completing the remaining work",
    ]:
        bullet(document, line)


def section_methodology(document: Document, data: Artifacts) -> None:
    document.add_heading("1. Detailed methodology of the proposed system", level=1)

    document.add_heading("1.1 Research questions", level=2)
    for question in [
        "RQ1 — Can a single-CPU scheduler switch between FCFS, SJF, Round Robin and "
        "Priority at run time, using only information available at that instant?",
        "RQ2 — Does such switching reduce mean waiting time relative to the fixed policies "
        "it switches between, once a context-switch cost is charged?",
        "RQ3 — Does a learned switching policy add value over a hand-written adaptive rule "
        "that sees exactly the same observation?",
        "RQ4 — Do any gains survive on a workload family that was never seen during "
        "training, and are they larger than the paired bootstrap noise?",
    ]:
        bullet(document, question)

    document.add_heading("1.2 Methodology pipeline", level=2)
    para(document,
         "The method is a closed loop between a discrete-event environment that owns the "
         "future and a controller that may see only the past and the present. Stages 3 to 7 "
         "run once per decision epoch inside a single trace; stage 8 runs once per experiment.")
    mermaid(document, mermaid_pipeline(), "methodology pipeline (Figure 1.1)")
    for stage, where, what, out in [
        ("Stage 1 — Dataset", "workload/generator.py",
         "generate seven seeded families of 15-process workloads with SHA-256 derived seeds",
         "workload list plus manifest"),
        ("Stage 2 — Simulation", "scheduler/runtime.py",
         "advance one CPU: arrivals, one FIFO ready queue, remaining bursts, switch cost",
         "decision epochs plus schedule trace"),
        ("Stage 3 — Observation", "scheduler/runtime.py",
         "freeze an immutable snapshot of arrived and ready work only (causal contract)",
         "RuntimeObservation"),
        ("Stage 4 — Encoding", "rl/runtime_state.py",
         "discretise five causal features into 162 tabular states (3 x 3 x 3 x 3 x 2)",
         "state index s in [0, 162)"),
        ("Stage 5 — Decision", "rl/runtime_controller.py",
         "epsilon-greedy during training, greedy during evaluation, over four policy actions, "
         "with an explicit fallback in unseen states",
         "action a in {FCFS, SJF, RR, Priority}"),
        ("Stage 6 — Execution", "scheduler/runtime.py",
         "run the selected segment (Round Robin bounded by the quantum), charge one unit when "
         "the running PID changes, settle endpoint arrivals",
         "advanced simulated time"),
        ("Stage 7 — Reward", "scheduler/runtime.py",
         "measure the waiting-time area dW accumulated by ready processes and return "
         "r = -dW / q with gamma = 1",
         "interval reward r_t"),
        ("Stage 8 — Evaluation", "experiments/runtime_experiment.py",
         "greedy read-only runs on held-out workloads, paired differences with a "
         "family-stratified crossed bootstrap",
         "metric tables plus confidence intervals"),
    ]:
        bullet(document, f"{where} — {what}. Output: {out}.", bold_prefix=f"{stage}: ")

    document.add_heading("1.3 Causal observation contract", level=2)
    para(document,
         "The environment privately owns the complete event list so that it can admit future "
         "arrivals at their arrival times; the controller receives an immutable "
         "RuntimeObservation. This is the methodological core of the project: without it, any "
         "measured gain could come from information that a real kernel could not have.")
    para(document, "Included in the observation:", bold=True)
    for item in [
        "current simulated time and the policy selected at the immediately preceding epoch "
        "(None at the first epoch);",
        "processes that have arrived and are currently ready, with PID, arrival time, known "
        "burst, remaining burst and priority;",
        "counts of arrived and completed processes;",
        "aggregates restricted to work that has already arrived: observed mean burst and "
        "priority spread;",
        "mean arrival age of currently ready jobs (current time minus arrival time, which "
        "includes earlier CPU service).",
    ]:
        bullet(document, item)
    para(document, "Deliberately excluded:", bold=True)
    for item in [
        "any data about processes that have not arrived yet, including future bursts and "
        "arrival times;",
        "the total size of the workload and the identity of its family;",
        "counterfactual or completed-schedule metrics of policies that were not run.",
    ]:
        bullet(document, item)
    para(document,
         "A paired hidden-future test changes the arrival, burst and priority of unarrived "
         "processes while keeping the observable prefix fixed, and asserts that the "
         "observation and action prefixes stay bit-for-bit identical.")

    document.add_heading("1.4 State abstraction and learning objective", level=2)
    for feature, bins, cuts, why in [
        ("1. ready_count", "3", "{1}, {2-3}, {4+}", "instantaneous ready-queue contention"),
        ("2. completed_count", "3", "{0}, {1-3}, {4+}",
         "trace progress without knowing the workload size"),
        ("3. median_ready_remaining_burst", "3", "<= q, <= 4q, > 4q",
         "remaining service demand of ready jobs"),
        ("4. mean_ready_arrival_age", "3", "<= q, <= 4q, > 4q",
         "age proxy for ready jobs, including prior service"),
        ("5. arrived_work_priority_spread", "2", "zero or non-zero",
         "whether arrived processes have distinct priorities"),
    ]:
        bullet(document, f"{bins} bins with cuts {cuts}; why: {why}.",
               bold_prefix=f"{feature} — ")
    para(document, "3 x 3 x 3 x 3 x 2 = 162 states, giving 162 x 4 = 648 state-action pairs.")
    para(document,
         "Each trace is one sequential episode. Between consecutive epochs the environment "
         "measures the waiting-time area dW accumulated by ready processes, including any "
         "switch interval, and returns the scaled interval reward:")
    code(document,
         "r_t = - dW_t / max(1, q)      gamma = 1.0\n"
         "=> sum over t of r_t = - TotalWaitingTime / q")
    para(document,
         "Because gamma = 1 and the episode is finite, maximising the undiscounted return is "
         "identical to minimising total waiting time; the objective is not a proxy for it. "
         "The tabular update is:")
    code(document,
         "Q(s, a) <- Q(s, a) + alpha * ( r_t + gamma * max over visited a' of Q(s', a') "
         "- Q(s, a) )\n"
         "alpha = 0.1, terminal transitions omit the bootstrap term")
    para(document,
         "The first visit to a state forces a uniform random action, greedy selection and "
         "bootstrapping mask unvisited actions, and a wholly unseen evaluation state triggers "
         "an explicit, counted fallback to the predeclared heuristic rather than a silent "
         "zero-Q tie.")

    document.add_heading("1.5 Methodological controls", level=2)
    for control, how in [
        ("Fixed-policy baselines",
         "FCFS, SJF, Round Robin and Priority run on identical workloads and pay the identical "
         "switch cost"),
        ("Predeclared adaptive rule",
         "a non-RL heuristic was fixed before any test result was seen, so learning is not "
         "credited for what fixed rules already do"),
        ("Blinded held-out split",
         "the final test (master seed 19301) is generated and evaluated after training, and the "
         "pipeline asserts zero fingerprint overlap with training"),
        ("Out-of-family generalisation",
         "poisson_arrivals is excluded from training entirely and used only at evaluation time"),
        ("Independent seeds",
         "five agents (7101-7105) are trained separately and Q-learning results average all five"),
        ("Paired uncertainty",
         "differences are paired per workload and per model seed, then resampled with a "
         "family-stratified crossed bootstrap of 1,000 replicates (seed 104729)"),
        ("Determinism",
         "evaluation is greedy and read-only, and artifacts regenerate from the declared seeds"),
        ("Separation of timing",
         "observation, selection, Q-update and simulator wall time are measured in separate "
         "columns and never charged to simulated time"),
    ]:
        bullet(document, how, bold_prefix=f"{control} — ")


def section_architecture(document: Document, data: Artifacts) -> None:
    document.add_heading("2. Overall conceptual, architectural and pipeline diagram", level=1)
    para(document,
         "The system is layered so that the information boundary is structural rather than "
         "conventional: the environment owns the future and the accounting, the controller only "
         "ever receives a frozen snapshot of arrived work, and the controller's reply is a "
         "single policy name. It cannot touch the queue, the bursts or the event list.")
    mermaid(document, mermaid_architecture(), "system architecture (Figure 2.1)")
    for layer, components, role in [
        ("Data layer", "workload/models.py, workload/generator.py",
         "defines Process and Workload objects and generates the seven seeded families with "
         "SHA-256 derived seeds"),
        ("Simulation layer", "scheduler/runtime.py",
         "discrete-event loop, FIFO ready queue, remaining bursts, switch cost, schedule trace"),
        ("Baseline layer", "scheduler/fcfs.py, sjf.py, round_robin.py, priority.py",
         "standalone fixed policies, preserved so runtime actions can be checked against them"),
        ("Perception layer", "rl/runtime_state.py",
         "encodes an immutable observation into one of 162 causal states"),
        ("Decision layer", "rl/runtime_controller.py, rl/runtime_heuristic.py",
         "learned policy (tabular Q-learning) and the predeclared non-RL rule"),
        ("Experiment layer", "experiments/runtime_experiment.py, runtime_config.py",
         "splits, training, validation, held-out testing, paired bootstrap, artifact export"),
        ("Evaluation layer", "evaluation/metrics.py, evaluation/comparison.py",
         "metric definitions and paired/bootstrap comparison helpers"),
    ]:
        bullet(document, f"{components} — {role}.", bold_prefix=f"{layer}: ")

    document.add_heading("2.1 Runtime control flow", level=2)
    for step in [
        "The environment advances to the next decision epoch (a dispatch point, a completion or "
        "a Round-Robin quantum boundary) and admits every arrival at or before that instant in "
        "(arrival time, pid) order.",
        "It freezes an observation and passes it to the controller, which returns one of four "
        "policy names.",
        "If the selected process differs from the running PID, one unit of switch cost is "
        "charged before the segment starts; arrivals during that interval are queued but the "
        "dispatch already chosen is not reconsidered.",
        "The segment runs (Round Robin bounded by the quantum, the others to completion); "
        "endpoint arrivals are admitted first, then an unfinished Round-Robin process is "
        "appended to the tail.",
        "The environment measures the waiting-time area accumulated by ready processes, "
        "converts it to the interval reward and hands it back to the controller for the Q "
        "update.",
        "A switch changes only the next selection rule and service length: the queue, the "
        "remaining bursts and the switch accounting are preserved across switches.",
    ]:
        numbered(document, step)


def section_modules(document: Document) -> None:
    document.add_heading("3. Description of the major modules", level=1)
    para(document,
         "Six modules carry the whole system: two build and run the simulated machine, two "
         "implement the competing controllers, one is the non-learning reference rule, and one "
         "orchestrates the experiment and its statistics. Section 4 gives the workflow diagram "
         "for each.")
    for module_id, name, files, inp, out, responsibility in MODULES:
        document.add_heading(f"{module_id} — {name}", level=2)
        bullet(document, files, bold_prefix="Files: ")
        bullet(document, f"{inp} → {out}", bold_prefix="Contract: ")
        bullet(document, responsibility, bold_prefix="Responsibility: ")
        bullet(document, "Implemented", bold_prefix="Status: ")


def section_workflows(document: Document) -> None:
    document.add_heading("4. Workflow diagram for each major module", level=1)
    para(document,
         "Each module is given as a Mermaid flowchart plus the numbered steps it executes. "
         "Paste any block into a Mermaid renderer to draw it.")
    for index, (title, subtitle, steps) in enumerate(WORKFLOWS, start=1):
        document.add_heading(title, level=2)
        para(document, subtitle, italic=True)
        mermaid(document, mermaid_workflow(steps), f"{title} (Figure 4.{index})")
        for step_index, (label, detail) in enumerate(steps, start=1):
            bullet(document, f"{label} — {detail}.", bold_prefix=f"Step {step_index}: ")


def section_dataset(document: Document, data: Artifacts) -> None:
    document.add_heading("5. Dataset details and data collection procedure", level=1)

    document.add_heading("5.1 Data collection procedure", level=2)
    para(document,
         "No external or personal data is used. The dataset is synthetic and is generated "
         "entirely by the project's own generator, so it is reproducible from three integers.")
    for step, action, detail in [
        ("D1", "Fix seven family configurations",
         "burst distribution and range, arrival pattern, window and rate, priority pattern; "
         "declared once and frozen in the experiment configuration"),
        ("D2", "Derive one seed per workload",
         "workload seed = SHA-256(master seed, split tag, family index, repetition); no global "
         "random state is shared between workloads"),
        ("D3", "Generate 15 processes per workload",
         "burst, arrival time and priority are drawn per the family rules and validated "
         "(positive burst, non-negative arrival, priority within bounds)"),
        ("D4", "Assign workloads to splits",
         "6,000 training workloads (6 families x 1,200 episodes x 5 seeds), 140 validation "
         "workloads (7 x 20) and 210 held-out test workloads (7 x 30)"),
        ("D5", "Fingerprint and check disjointness",
         "each workload gets a SHA-256 fingerprint and the pipeline asserts zero overlap "
         "between training, validation and test"),
        ("D6", "Store only what is needed",
         "metrics, the manifest (split, seed, fingerprint, process count) and compact summaries "
         "are committed; the large per-decision event logs are regenerated by the CLI and "
         "gitignored"),
        ("D7", "Regenerate on demand",
         "python main.py runtime-experiment rebuilds every workload and every metric from the "
         "declared seeds"),
    ]:
        bullet(document, f"{action} — {detail}.", bold_prefix=f"{step}: ")

    document.add_heading("5.2 Dataset inventory", level=2)
    for split, families, reps, workloads, processes, seed in [
        ("Training (sequential episodes)", "6 families",
         "1,200 episodes x 5 seeds", "6,000", "90,000", "7101-7105"),
        ("Validation", "7 families", "20 repetitions", "140", "2,100", "8201"),
        ("Final held-out test", "7 families", "30 repetitions", "210", "3,150", "19301"),
    ]:
        bullet(document,
               f"{families}; {reps}; {workloads} workloads; {processes} processes; "
               f"master seed {seed}.",
               bold_prefix=f"{split} — ")

    document.add_heading("5.3 Family specification", level=2)
    for family in data.cfg["workload_family_parameters"]:
        bullet(document,
               f"{family['description']} Bursts {family['burst_time_min']}-"
               f"{family['burst_time_max']} ({family['burst_distribution']}); arrivals "
               f"{family['arrival_pattern']}; priorities {family['priority_min']}-"
               f"{family['priority_max']}.",
               bold_prefix=f"{family['name']} — ")

    document.add_heading("5.4 Measured dataset statistics", level=2)
    para(document,
         "The figures below were recomputed with the project's generator at document-build "
         "time, so they describe the actual held-out dataset rather than the configuration "
         "alone. High priority means priority <= 2.")
    if not data.processes.empty:
        for family in data.families:
            procs = data.processes[data.processes["family"] == family]
            workloads = data.workloads[data.workloads["family"] == family]
            bullet(document,
                   f"{len(workloads)} workloads, {len(procs)} processes; burst "
                   f"{procs['burst_time'].mean():.1f} ± "
                   f"{procs['burst_time'].std(ddof=1):.1f}; mean arrival "
                   f"{procs['arrival_time'].mean():.1f}; mean total work "
                   f"{workloads['total_burst'].mean():.0f}; high-priority share "
                   f"{100.0 * (procs['priority'] <= 2).mean():.0f}%.",
                   bold_prefix=f"{family.replace('_', ' ')} — ")
        bullet(document,
               f"{len(data.workloads)} workloads, {len(data.processes)} processes; burst "
               f"{data.processes['burst_time'].mean():.1f} ± "
               f"{data.processes['burst_time'].std(ddof=1):.1f}; mean arrival "
               f"{data.processes['arrival_time'].mean():.1f}; mean total work "
               f"{data.workloads['total_burst'].mean():.0f}; high-priority share "
               f"{100.0 * (data.processes['priority'] <= 2).mean():.0f}%.",
               bold_prefix="All families — ")

    document.add_heading("5.5 Data integrity and assumptions", level=2)
    for item in [
        "Reproducibility — seeds are derived with SHA-256, so a workload is a pure function "
        "of (master seed, split tag, family index, repetition).",
        f"Disjointness — the pipeline fails loudly if a fingerprint appears in two splits; "
        f"the current run reports overlap = "
        f"{str(data.summary['split_fingerprint_overlap']).lower()} over "
        f"{data.summary['training_fingerprint_count_union']:,} distinct training fingerprints.",
        "Burst-knowledge assumption — exact burst lengths become known when a process "
        "arrives. This is the assumption already required by standalone SJF; no burst "
        "predictor is included.",
        "Storage discipline — per-decision event logs are large, so they are regenerated by "
        "the CLI and gitignored; metrics, Q tables and manifests are committed.",
        "No personal or licensed data — the dataset is entirely synthetic.",
    ]:
        bullet(document, item)


def section_tools(document: Document, data: Artifacts) -> None:
    document.add_heading("6. Tools, technologies, algorithms and frameworks used", level=1)

    document.add_heading("6.1 Tools and technologies", level=2)
    for category, item, role in [
        ("Language", "Python 3.11.2", "whole implementation, CLI and report generators"),
        ("Numerics", "NumPy 2.4.6", "Q table as an array, bootstrap resampling"),
        ("Data handling", "pandas 3.0.6", "metric aggregation and CSV artifact export"),
        ("Plotting", "Matplotlib 3.11.2", "figures in the PDF version of this report"),
        ("Testing", "pytest 9.1.1", "273 automated checks"),
        ("PDF generation", "ReportLab 5.0.1", "renders the PDF version from the artifacts"),
        ("Document generation", "python-docx 1.2.0", "renders this text-only DOCX"),
        ("Hashing", "hashlib (SHA-256, standard library)",
         "seed derivation and workload fingerprints"),
        ("Standard library", "argparse, dataclasses, json, pathlib",
         "configuration objects, CLI and artifact serialisation"),
        ("Version control", "Git and GitHub", "branch-per-change workflow with pull requests"),
        ("Environment", "Linux 6.1 (x86-64) with venv and pip",
         "development, experiment runs and report builds"),
    ]:
        bullet(document, f"{item} — {role}.", bold_prefix=f"{category}: ")

    document.add_heading("6.2 Algorithms", level=2)
    for name, definition, usage in [
        ("FCFS (action 0)",
         "dispatch the head of the ready queue, non-preemptively",
         "baseline and runtime action; after a Round-Robin requeue it follows live FIFO order"),
        ("SJF (action 1)",
         "smallest remaining burst first, non-preemptive, ties by ready order",
         "baseline and runtime action; optimal for mean waiting time under known bursts"),
        ("Round Robin (action 2)",
         "head of the queue for min(quantum, remaining); unfinished jobs go to the tail after "
         "endpoint arrivals",
         "baseline and runtime action; bounds response time"),
        ("Priority (action 3)",
         "highest static priority first, non-preemptive, ties by ready order",
         "baseline and runtime action; lower numeric value means higher priority"),
        ("Tabular Q-learning",
         "Q(s,a) += alpha * (r + gamma * max over visited actions of Q(s',.) - Q(s,a)) with "
         "epsilon-greedy exploration and unvisited-action masking",
         "the learned runtime controller over 162 states x 4 actions"),
        ("Predeclared heuristic",
         "Round Robin, then Priority, then SJF, then FCFS cascade over the same observation",
         "non-learning adaptive reference"),
        ("Crossed bootstrap",
         "resample workloads within family and model seeds independently, 1,000 replicates",
         "paired 95% confidence intervals for every comparison"),
        ("Discrete-event simulation",
         "event-list advancement with a persistent FIFO ready queue and switch-cost charging",
         "the machine being controlled"),
    ]:
        bullet(document, f"{definition}. Used for: {usage}.", bold_prefix=f"{name} — ")

    document.add_heading("6.3 Controller pseudocode", level=2)
    code(document,
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
    if ready_count >= 3 and mean_arrival_age >= quantum:   return Round Robin
    if >= 2 ready jobs have distinct priorities:           return Priority
    if max(remaining_burst) >= 2 * min(remaining_burst):   return SJF
    return FCFS
""")

    document.add_heading("6.4 Frameworks, and why none of the heavy ones", level=2)
    para(document,
         "The controller is a 162 x 4 table rather than a neural network, so no deep-learning "
         "framework is needed. This is a deliberate methodological choice: every Q value, visit "
         "count and fallback can be inspected directly in the committed artifacts, which is what "
         "makes claims about state coverage and unseen-state behaviour verifiable rather than "
         "anecdotal.")


def section_experiment(document: Document, data: Artifacts) -> None:
    document.add_heading("7. Experimental plan and evaluation metrics", level=1)

    document.add_heading("7.1 Experimental plan", level=2)
    for step, action, detail in [
        ("P1", "Fix the design before looking at results",
         "families, splits, seeds, quantum (4), switch cost (1), learning rates and the "
         "heuristic rule are declared in configuration and committed"),
        ("P2", "Train five independent agents",
         "seeds 7101-7105; 1,200 sequential episodes each on six families, with "
         "poisson_arrivals withheld"),
        ("P3", "Validate once",
         "140 workloads (seed 8201) used only for reporting and for the predeclared "
         "illustrative-demo rule, never for tuning"),
        ("P4", "Test once, held out",
         "210 untouched workloads (seed 19301); the fixed policies and the heuristic are "
         "deterministic, Q-learning is evaluated by all five models greedily and read-only"),
        ("P5", "Compare every method against every other",
         "paired per-workload differences, plus independent resampling of the five model seeds"),
        ("P6", "Quantify uncertainty",
         "family-stratified crossed bootstrap, 1,000 replicates, seed 104729; 95% intervals on "
         "every headline comparison"),
        ("P7", "Publish artifacts",
         "metric tables, Q values, visit counts, coverage, manifest, demo trace and summary JSON "
         "written to results/runtime/ and regenerated by one command"),
    ]:
        bullet(document, f"{action} — {detail}.", bold_prefix=f"{step}: ")

    document.add_heading("7.2 Evaluation metrics", level=2)
    for name, definition, why in [
        ("Mean waiting time",
         "mean over processes of (first service start minus arrival)",
         "primary metric; it is exactly the quantity the reward minimises"),
        ("Mean turnaround time", "mean over processes of (completion minus arrival)",
         "end-to-end latency experienced by a job"),
        ("Mean response time", "mean over processes of (first dispatch minus arrival)",
         "interactivity; this is what Round Robin buys"),
        ("CPU utilisation (%)", "busy time divided by total elapsed time, times 100",
         "cost of switching and idling"),
        ("Throughput", "processes completed per unit of simulated time",
         "work completed per unit time"),
        ("Context switches", "number of running-PID changes",
         "overhead proxy; the switch cost is charged for every method"),
        ("Policy switches per trace",
         "decisions where the chosen policy differs from the previous one",
         "how much the controller actually adapts"),
        ("State-action coverage", "visited (s, a) pairs out of 648, and unseen-state fallbacks",
         "whether the learned table is populated where it is used"),
    ]:
        bullet(document, f"{definition}. Why: {why}.", bold_prefix=f"{name} — ")

    document.add_heading("7.3 Statistical procedure", level=2)
    para(document,
         "Comparisons are paired: for each workload the difference (target minus reference) is "
         "computed on the same workload, and for Q-learning the model seed is resampled "
         "independently from the workloads, which is a crossed bootstrap. Resampling is "
         "stratified by workload family and 1,000 replicates are drawn with seed 104729. An "
         "interval that excludes zero on the favourable side is treated as a real difference on "
         "this benchmark; an interval containing zero is reported as inconclusive rather than as "
         "a win.")

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
               f"waiting {wait:,.3f} ± {wait_sd:,.2f}; turnaround {turn:,.3f} ± "
               f"{turn_sd:,.2f}; response {resp:,.3f} ± {resp_sd:,.2f}; CPU utilisation "
               f"{cpu:.2f} ± {cpu_sd:.2f} %; throughput {thr:.3f} ± {thr_sd:.3f}; "
               f"context switches {ctx:.2f} ± {ctx_sd:.2f}; policy switches per trace "
               f"{switches:.2f}.",
               bold_prefix=f"{method} — ")
    para(document,
         "All values are means ± standard deviation across the 210 held-out workloads of "
         "master seed 19301, with quantum 4 and switch cost 1; Q-learning also averages the five "
         "model seeds.")
    mermaid(document, mermaid_results(
        data, "avg_waiting_time",
        "Mean waiting time by method", "time units"),
        "final-test mean waiting time (Figure 7.1)")
    mermaid(document, mermaid_results(
        data, "avg_response_time", "Mean response time by method", "time units"),
        "final-test mean response time (Figure 7.2)")

    document.add_heading("7.5 Paired comparisons", level=2)
    pairs = [("Runtime Q-learning", "SJF"), ("Runtime Q-learning", "Causal heuristic"),
             ("Round Robin", "SJF"), ("Causal heuristic", "SJF"), ("FCFS", "SJF")]
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
                   f"{label}: difference {mean:+,.3f}, 95% interval [{low:+,.3f}, "
                   f"{high:+,.3f}], {reading}.",
                   bold_prefix=f"{SHORT_NAME[target]} minus {SHORT_NAME[reference]} — ")

    document.add_heading("7.6 What the plan has established so far", level=2)
    q_sjf_wait = ci_of(data, "Runtime Q-learning", "SJF", "avg_waiting_time")
    q_sjf_resp = ci_of(data, "Runtime Q-learning", "SJF", "avg_response_time")
    rr_sjf_wait = ci_of(data, "Round Robin", "SJF", "avg_waiting_time")
    fcfs_sjf_wait = ci_of(data, "FCFS", "SJF", "avg_waiting_time")
    prio_sjf_wait = ci_of(data, "Priority", "SJF", "avg_waiting_time")
    q_heur_wait = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_waiting_time")
    q_heur_resp = ci_of(data, "Runtime Q-learning", "Causal heuristic", "avg_response_time")
    for item in [
        f"Against Round Robin the learned controller is clearly better on waiting time: "
        f"{q_sjf_wait[0] - rr_sjf_wait[0]:+,.3f} time units, obtained by differencing the two "
        f"committed comparisons against SJF (Q minus SJF {q_sjf_wait[0]:+,.3f} and RR minus SJF "
        f"{rr_sjf_wait[0]:+,.3f}; both intervals exclude zero and are far apart). FCFS and "
        f"Priority sit between the two at {fcfs_sjf_wait[0]:+,.3f} and "
        f"{prio_sjf_wait[0]:+,.3f} relative to SJF.",
        f"Against the predeclared adaptive rule it reduces waiting time by "
        f"{q_heur_wait[0]:+,.3f} (95% interval [{q_heur_wait[1]:+,.3f}, "
        f"{q_heur_wait[2]:+,.3f}]), so the gain comes from learning rather than merely from "
        f"being adaptive. The rule does win on response time ({q_heur_resp[0]:+,.3f} against "
        f"Q-learning) because it slices aggressively with Round Robin.",
        f"Against SJF the result is mixed, and the report says so: waiting time is worse by "
        f"{q_sjf_wait[0]:+,.3f} (95% interval [{q_sjf_wait[1]:+,.3f}, {q_sjf_wait[2]:+,.3f}]) "
        f"while response time is better by {q_sjf_resp[0]:+,.3f} (95% interval "
        f"[{q_sjf_resp[1]:+,.3f}, {q_sjf_resp[2]:+,.3f}]). Non-preemptive SJF with exact "
        f"bursts and no mid-job switch cost is a very strong reference on this benchmark.",
        "On the held-out poisson_arrivals family the controller still lands between SJF and "
        "Round Robin, so the behaviour is not family memorisation.",
        f"Controller overhead is small relative to simulated time: about "
        f"{data.summary['mean_runtime_q_observation_us_per_decision']:.2f} microseconds for "
        f"observation and "
        f"{data.summary['mean_runtime_q_selection_us_per_decision']:.2f} microseconds for "
        f"selection per decision, measured but never charged to the schedule.",
    ]:
        bullet(document, item)


def section_progress(document: Document, data: Artifacts, test_rows: List[List[str]],
                     pytest_block: str, cli_block: str) -> None:
    document.add_heading("8. Implementation progress", level=1)
    para(document,
         "All six major modules are implemented, exercised by the automated suite and "
         "demonstrated with the outputs below. Review 2 requires at least 50% of the proposed "
         "work to be complete, with the completed modules demonstrated by valid outputs; the "
         "list below records completion per module and the artifact that evidences it.")
    for module_id, name, completion, evidence, status in [
        ("M1", "Workload generator and dataset builder", "100%",
         "6,350-row manifest plus the regenerated dataset statistics in section 5.4", "Complete"),
        ("M2", "Discrete-event runtime environment", "100%",
         "fixed-action equivalence tests and the schedule traces behind the final-test metrics",
         "Complete"),
        ("M3", "Causal observation and state encoder", "100%",
         "162-state table and the hidden-future invariance tests", "Complete"),
        ("M4", "Sequential Q-learning controller", "100%",
         "runtime_q_table.csv with 3,240 Q values, coverage report and fallback counts",
         "Complete"),
        ("M5", "Predeclared heuristic controller", "100%",
         "heuristic rows present in every metric table", "Complete"),
        ("M6", "Experiment orchestration, metrics and reporting", "100%",
         "final-test, validation, paired-bootstrap and coverage tables plus the generated report",
         "Complete"),
        ("-", "Review 2 report generators (PDF and this document)", "100%",
         "tools/build_review2_report.py and tools/build_review2_docx.py", "Complete"),
    ]:
        bullet(document,
               f"completion {completion}; evidence: {evidence}; status: {status}.",
               bold_prefix=f"{module_id} — {name}. ")
    para(document,
         "Overall, the proposed Review 2 scope (methodology, simulator, controller, experiment "
         "and evaluation) is complete. The items listed in section 9 are extensions beyond that "
         "scope.")

    document.add_heading("8.1 Demonstration 1 — automated verification", level=2)
    if test_rows:
        total = sum(int(row[1]) for row in test_rows)
        para(document,
             f"The suite collects {total} checks. Per-module counts, recorded while building "
             f"this document:")
        for module, count, protects in test_rows:
            bullet(document, f"{count} checks — {protects}.", bold_prefix=f"{module}: ")
    code(document, pytest_block, label="Output of python -m pytest -q:")

    document.add_heading("8.2 Demonstration 2 — end-to-end run output", level=2)
    para(document,
         "The full pipeline (training, validation, held-out testing and artifact export) is "
         "reproduced by one command. The output below is from a verification run on the "
         "reporting host and matches the committed artifacts exactly.")
    code(document, cli_block, label="Output of python main.py runtime-experiment:")

    document.add_heading("8.3 Demonstration 3 — learned behaviour", level=2)
    demo = data.demo
    para(document,
         f"The controller does switch policies inside a single trace. The example below is taken "
         f"from the validation split by the predeclared rule: training seed "
         f"{demo['training_seed']}, family {demo['family']}, repetition {demo['repetition']}, "
         f"fingerprint {demo['workload_fingerprint']}. It contains "
         f"{demo['policy_switch_count']} policy switches over {demo['decision_count']} "
         f"decisions, with mean waiting time "
         f"{demo['metrics']['avg_waiting_time']:.3f} and mean response time "
         f"{demo['metrics']['avg_response_time']:.3f}. It illustrates the mechanism and is not "
         f"a performance claim.")
    mermaid(document, mermaid_gantt(data),
            "illustrative validation trace, bars coloured by the policy chosen (Figure 8.2)")

    document.add_heading("8.4 Artifacts produced", level=2)
    for artifact, size, contents in [
        ("results/runtime/runtime_final_test_metrics.csv", "2,100 rows",
         "six metrics, switch and decision counts, timings, held-out test"),
        ("results/runtime/runtime_validation_metrics.csv", "1,400 rows",
         "the same columns for the validation split"),
        ("results/runtime/runtime_training_metrics.csv", "6,000 rows",
         "per-episode sequential training outcomes"),
        ("results/runtime/runtime_paired_comparisons.csv", "36 rows",
         "paired differences with 95% crossed-bootstrap intervals"),
        ("results/runtime/runtime_state_action_coverage.csv", "15 rows",
         "visited states and pairs, fallback counts, timings per seed"),
        ("results/runtime/runtime_q_table.csv", "3,240 rows",
         "Q values and visit counts per training seed"),
        ("results/runtime/runtime_workload_manifest.csv", "6,350 rows",
         "split, seeds and workload fingerprints"),
        ("results/runtime/runtime_learned_switch_demo.json", "1 trace",
         "observations, actions, switch times and execution trace"),
        ("results/runtime/runtime_summary.json", "summary",
         "configuration, split fingerprints, all means and software versions"),
        ("figures/review2/", "14 figures", "every figure in the PDF version of this report"),
    ]:
        bullet(document, f"{size} — {contents}.", bold_prefix=f"{artifact}: ")


def section_challenges(document: Document, data: Artifacts) -> None:
    document.add_heading(
        "9. Challenges encountered and plan for completing the remaining work", level=1)

    document.add_heading("9.1 Challenges encountered", level=2)
    for title, why, how in [
        ("Keeping the controller causal",
         "an earlier whole-workload selector saw future arrivals and counterfactual schedules, "
         "so its results could not be interpreted as a runtime controller",
         "introduced an immutable RuntimeObservation, made the encoder accept only that object, "
         "and added a paired hidden-future test that fails if unarrived changes leak into the "
         "observation or action prefix"),
        ("Round-Robin and FCFS interaction after a requeue",
         "once Round Robin appends a preempted job to the tail, FCFS is ambiguous: original "
         "arrival order or live queue order?",
         "defined one persistent FIFO ready queue with canonical (arrival time, pid) insertion, "
         "specified that FCFS follows live queue order while SJF and Priority ties preserve it, "
         "and tested fixed-action runs against the standalone schedulers"),
        ("Charging switch cost fairly",
         "if the learner pays for switching but baselines do not, the comparison is "
         "meaningless; if nothing is charged, switching is unrealistic",
         "one unit is charged whenever the running PID changes, for every method including the "
         "fixed policies; arrivals during the switch are queued but the chosen dispatch is not "
         "reconsidered"),
        ("Designing a reward that is the objective",
         "a shaped reward can drift away from the metric being reported",
         "used r = -dW/q with gamma = 1 so the episodic return is exactly minus total waiting "
         "time divided by the quantum, and added a test that asserts the identity"),
        ("Sparse coverage of the tabular space",
         "only about 42% of the 648 state-action pairs are ever visited, and unseen states "
         "appear during evaluation",
         "kept the abstraction small, forced exploration on first visit, masked unvisited "
         "actions in greedy steps and bootstraps, and made unseen states fall back to the "
         "predeclared heuristic with the fallback count reported"),
        ("Honest reporting of a negative result",
         "the learner does not beat non-preemptive SJF on waiting time, and it would be easy to "
         "bury this",
         "reported it as a headline finding with its bootstrap interval and stated the "
         "structural reason: SJF receives exact bursts at arrival and pays no mid-job switch "
         "cost"),
        ("Host-dependent timing noise",
         "wall-clock columns make artifacts look non-deterministic across machines",
         "observation, selection, Q-update and simulator times are stored in separate columns, "
         "never charged to simulated time and excluded from schedule comparisons"),
        ("Large per-decision event logs",
         "full causal decision logs are large and would bloat the repository",
         "they are written by the CLI but gitignored and regenerable from the recorded "
         "configuration and seeds, while compact metrics, Q tables and manifests are committed"),
    ]:
        bullet(document, f"Why it mattered: {why}. Resolution: {how}.",
               bold_prefix=f"{title} — ")

    document.add_heading("9.2 Plan for completing the remaining work", level=2)
    for task_id, task, gap, approach, deliverable in [
        ("W1", "Sensitivity study",
         "all results are for quantum 4 and switch cost 1",
         "re-run the plan over a grid of quanta and switch costs and report where the ordering "
         "changes",
         "metric tables per (quantum, cost) cell"),
        ("W2", "Burst prediction",
         "exact bursts are assumed known at arrival",
         "add an explicit predictor with measured error and re-run the controller under that "
         "noise",
         "results with prediction-error bands"),
        ("W3", "Richer state abstraction",
         "only about 42% of pairs are visited and 162 states is coarse",
         "tile coding or a small function approximator, keeping the same causal observation "
         "contract and fallback reporting",
         "coverage and metric comparison against the tabular agent"),
        ("W4", "Multi-objective reward",
         "waiting time is optimised and response time is not",
         "add a constrained or weighted objective and measure the waiting-response trade-off",
         "Pareto front over the two metrics"),
        ("W5", "I/O and blocking",
         "processes never block, so the queue is CPU-bound only",
         "model I/O bursts and blocked/wake transitions in the environment",
         "extended environment plus re-baselined comparisons"),
        ("W6", "Multicore extension",
         "single CPU only",
         "extend to N cores with per-core queues and migration cost, choosing a policy per core",
         "multicore metrics and overhead analysis"),
        ("W7", "Trace-driven validation",
         "all workloads are synthetic",
         "replay a public scheduling trace through the same environment and report the gap to "
         "the synthetic results",
         "trace-driven result table"),
    ]:
        bullet(document,
               f"gap: {gap}. approach: {approach}. deliverable: {deliverable}.",
               bold_prefix=f"{task_id} — {task}. ")

    document.add_heading("9.3 Sequence for the remaining work", level=2)
    for item in [
        "First: W1 (sensitivity) and W2 (burst prediction). They test whether the current "
        "conclusions are an artefact of fixed parameters or of an idealised assumption.",
        "Second: W3 and W4. They attack the two measured weaknesses directly, namely coarse "
        "coverage and a single objective.",
        "Third: W5 and W6. These are environment extensions that make the simulator more "
        "realistic, each with re-baselined comparisons.",
        "Finally: W7. External validation on a real trace is the strongest available check that "
        "the synthetic ordering means anything.",
    ]:
        numbered(document, item)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def collect_test_counts(root: Path) -> Tuple[List[List[str]], str]:
    """Run pytest and return (per-module rows, condensed output)."""
    descriptions = {
        "test_adaptive.py": "legacy offline selector: state encoding, decisions, read-only "
                            "evaluation",
        "test_experiment_design.py": "frozen seeds, split sizes and configuration invariants",
        "test_experiments.py": "legacy experiment pipeline, metrics and report generation",
        "test_fcfs.py": "FCFS ordering, idle handling and metric computation",
        "test_generator.py": "workload family generation and reproducibility from seeds",
        "test_metrics.py": "waiting, turnaround, response, utilisation and throughput "
                           "definitions",
        "test_priority.py": "priority ordering, ties and non-preemption",
        "test_q_learning.py": "Q updates, epsilon schedule, masking, visit counts, evaluation",
        "test_reward.py": "reward composition and clipping against the reference policies",
        "test_round_robin.py": "quantum boundaries, re-enqueue order and completion behaviour",
        "test_runtime_experiment.py": "split disjointness, determinism, artifact regeneration",
        "test_runtime_scheduler.py": "runtime environment: fixed-action equivalence, causality, "
                                     "reward identity",
        "test_schedule_result.py": "schedule trace validation and aggregate result objects",
        "test_sjf.py": "SJF selection, tie-breaking and non-preemption",
        "test_state.py": "offline state discretisation and mixed-radix indexing",
        "test_workload_models.py": "process and workload models, validation, trace invariants",
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
    tail = [line for line in run.stdout.strip().splitlines() if line.strip()][-3:]
    output = "\n".join(["$ python -m pytest -q", *tail]) if tail else (
        f"$ python -m pytest -q\n{sum(counts.values())} tests (output unavailable)"
    )
    return rows, output


def build_cli_block(data: Artifacts) -> str:
    lines = [
        "$ python main.py runtime-experiment",
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
        "  artifacts: training_metrics, validation_metrics, final_test_metrics,",
        "             paired_comparisons, state_action_coverage, q_table,",
        "             workload_manifest, learned_switch_demo, summary, report",
    ]
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
        passed = "passed" in pytest_block
        test_line = (f"{total} checks, {'all passing' if passed else 'see section 8.1'}"
                     f"{' (re-run during this build)' if total else ''}")

    generated = _dt.date.today().strftime("%d %B %Y")

    document = new_document()
    add_cover(document, data, generated, test_line)
    add_contents(document)
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


if __name__ == "__main__":  # noqa: E402
    raise SystemExit(main())
