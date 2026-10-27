#!/usr/bin/env python3
"""Fix Mermaid code placement to match section positions correctly.

Map: para position -> correct figure based on section heading."""

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
DOCX_PATH = __import__('os').path.join(BASE, "Final_Research_Paper.docx")

# CORRECT mapping: para index -> (figure label, mermaid code)
# Based on section headings, not figure numbers
CORRECT_MAPPING = {
    90: (
        "Figure 1. Experimental methodology pipeline",
        """flowchart LR
    A["Workload<br/>Families<br/>(7 families)"] --> B["Train / Valid /<br/>Test Split<br/>(seeds 7101-7105,<br/>8201, 23819)"]
    B --> C["Arms<br/>(10 arms:<br/>fixed, rule,<br/>Q-learning)"]
    C --> D["Evaluation<br/>(210 traces/arm,<br/>bootstrap 1000)"]
    D --> E["Counterfactual<br/>Audit &<br/>Report"]"""
    ),
    94: (
        "Figure 2. Overall system architecture of the adaptive CPU scheduling pipeline",
        """flowchart TB
    subgraph Workload Generation
        WG["Workload Generator<br/>(workload/)"] --> WF["Workload Fingerprint<br/>(fingerprint.py)"]
        WF --> DI["Drift Injector<br/>(drift.py)"]
        DI --> BE["Burst Estimator<br/>(burst_estimator.py)"]
    end
    subgraph Scheduling
        WF --> RS["Discrete-Event Scheduler<br/>(scheduler/runtime.py)"]
        RS --> RO["Runtime Observation<br/>(RuntimeObservation)"]
        DI --> RS
        RO --> PC["Policy Classes<br/>(FCFS/SJF/RR/Pri)"]
        BE --> RS
    end
    subgraph Self-Healing Controller
        RS --> CM["Causal Monitor<br/>(monitor.py)"]
        CM --> DD["Degradation Detector<br/>(detector.py)"]
        DD --> DX["Diagnosis<br/>(diagnosis.py)"]
        DX --> RP["Recovery Policy<br/>(policies.py)"]
    end
    subgraph Verification and Outcome
        RP --> VW["Verification Window<br/>(verifier.py)"]
        VW --> AR["Accept / Reject / Inconclusive"]
        AR --> RB["Rollback / Fallback<br/>(actions.py)"]
        RB --> IL["Incident Ledger"]
    end
    subgraph Evaluation
        VW --> MT["Metrics<br/>(evaluation/metrics.py)"]
        AR --> RL["Reliability<br/>(evaluation/reliability.py)"]
        RB --> BP["Paired Comparison<br/>(evaluation/paired.py)"]
    end
    RB -.->|"Feedback (action applied to scheduler)"| RS"""
    ),
    101: (
        "Figure 3. Workload generation and scheduling workflow",
        """flowchart TB
    FS["Family + Seed selection"] --> WG["WorkloadGenerator.generate()"]
    WG --> WO["Workload object (fingerprinted)"]
    DS["Optional: drift spec (drift.py)"] --> GD["generate_with_drift()"]
    GD --> DW["Drifted Workload (phase shift)"]
    WO --> RS["RuntimeSimulator.run(workload, ctrl)"]
    DW --> RS
    RS --> EL["Event loop: dispatch, switch, complete"]
    EL --> SR["ScheduleResult (per-process outcomes)"]"""
    ),
    105: (
        "Figure 4. Degradation detection and diagnosis workflow",
        """flowchart TD
    CN["Completion notice from scheduler"] --> RW["Update rolling window (EWMA ref + band)"]
    RW --> TH{"window_mean > ref + k * band ?"}
    TH -- "yes" --> HY["Hysteresis check (h consecutive)"]
    TH -- "no" --> SU["Suppress (reason logged)"]
    HY --> GA{"Gates pass? (cooldown, load, work)"}
    GA -- "yes" --> RE["Build ready evidence"]
    GA -- "no" --> SU
    RE --> DG["Diagnose: primary cause"]
    DG --> DI["Declare incident"]"""
    ),
    109: (
        "Figure 5. Verification, acceptance, and rollback workflow",
        """flowchart TD
    AA["Action applied; window opened"] --> CC["Collect completions over verify_decisions"]
    CC --> PC["Compute phase-adjusted closure residual"]
    PC --> AB{"residual >= accept boundary (+0.30)?"}
    AB -- "yes" --> AC["ACCEPTED (keep, learn)"]
    AB -- "no" --> RB{"residual <= reject boundary (-0.30)?"}
    RB -- "yes" --> RJ["REJECTED (rollback, fail +1)"]
    RB -- "no" --> IC["INCONCLUSIVE (rollback, no penalty)"]"""
    ),
    113: (
        "Figure 6. Self-healing controller workflow (one trace)",
        """flowchart TD
    GW["Generate workload"] --> SM["Simulate (dispatch slice)"]
    SM --> MN["Monitor (EWMA reference)"]
    MN --> DD{"Degradation detected?"}
    DD -- "no: continue simulating" --> SM
    DD -- "yes" --> DC["Diagnose cause"]
    DC --> SA["Select recovery action"]
    SA --> AA["Apply action to scheduler"]
    AA --> VW["Verification window"]
    VW -->|"good"| AC["Accepted (keep change)"]
    VW -->|"bad"| RJ["Rejected (rollback + failure)"]
    VW -->|"unclear"| IN["Inconclusive (rollback, no penalty)"]
    AC --> IL["Incident Ledger"]
    RJ --> IL
    IN --> IL
    AC --> FQ{"2 consecutive failures?"}
    FQ -- "yes" --> SF["Safe fallback"]
    FQ -- "no" --> SM"""
    ),
    117: (
        "Figure 7. Frontend-to-backend simulation execution flow",
        """flowchart LR
    UI["Browser UI (app.js)"] --> HR["HTTP Request POST /api/simulations"]
    HR --> FR["FastAPI Route (api.py)"]
    FR --> SL["Service Layer (service.py)"]
    SL --> SE["Simulation Engine (demo.run_fixture)"]
    SE --> MV["Model Validator (model.py)"]
    MV --> JR["JSON Response (workload, metrics, incidents)"]
    JR --> CH["Charts (inline SVG)"]
    CH --> RU["Render in UI (index.html)"]
    JR -.->|"JSON + SVG"| UI"""
    ),
}


def remove_all_content_from_paragraph(para):
    """Remove all runs and drawings from a paragraph."""
    from lxml import etree
    # Remove all child elements
    for child in list(para._element):
        para._element.remove(child)


doc = Document(DOCX_PATH)

# Fix each Mermaid block
for para_idx, (figure_label, mermaid_code) in CORRECT_MAPPING.items():
    para = doc.paragraphs[para_idx]

    # Clear the paragraph
    remove_all_content_from_paragraph(para)

    # Add figure label (centered, italic, bold)
    run = para.add_run(figure_label)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.bold = True
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add a spacer paragraph then the Mermaid code
    # We'll insert the code paragraphs AFTER the current one
    # Find insertion point
    parent = para._element.getparent()
    insert_after = para._element

    code_style_font = "Courier New"
    code_style_size = Pt(8)

    # Add separator
    sep_para = doc.add_paragraph()
    run = sep_para.add_run("─" * 70)
    run.font.name = code_style_font
    run.font.size = code_style_size
    # Move to correct position
    parent.insert(parent.index(insert_after) + 1, sep_para._element)

    # Add Mermaid code block
    code_para = doc.add_paragraph()
    code_para.paragraph_format.space_before = Pt(2)
    code_para.paragraph_format.space_after = Pt(2)

    for line in mermaid_code.split("\n"):
        run = code_para.add_run(line + "\n")
        run.font.name = code_style_font
        run.font.size = code_style_size

    # Move code para to correct position
    parent.insert(parent.index(sep_para._element) + 1, code_para._element)

    # Add another separator
    sep_para2 = doc.add_paragraph()
    run = sep_para2.add_run("─" * 70)
    run.font.name = code_style_font
    run.font.size = code_style_size
    parent.insert(parent.index(code_para._element) + 1, sep_para2._element)

    print(f"Fixed para {para_idx}: {figure_label[:50]}...")

doc.save(DOCX_PATH)
print(f"\nSaved {DOCX_PATH}")
