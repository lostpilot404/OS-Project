#!/usr/bin/env python3
"""Replace PNG diagrams in the DOCX with Mermaid source code blocks."""

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(BASE, "Final_Research_Paper.docx")
MERMAID_PATH = os.path.join(BASE, "diagrams", "figures_mermaid.md")

# Read Mermaid source from the markdown file
mermaid_blocks = {}
current_id = None
current_code = []
with open(MERMAID_PATH, 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        if line.startswith("## Figure "):
            if current_id:
                mermaid_blocks[current_id] = "\n".join(current_code)
            current_id = line
            current_code = []
        elif line.startswith("```mermaid"):
            continue
        elif line.startswith("```") and current_code:
            mermaid_blocks[current_id] = "\n".join(current_code)
            current_id = None
            current_code = []
        elif current_id is not None and line.strip():
            current_code.append(line)
    if current_id and current_code:
        mermaid_blocks[current_id] = "\n".join(current_code)

# Map paragraph indices to figure IDs based on CORRECT position-based numbering
# Para 90 = methodology (Fig 1), 94 = architecture (Fig 2), 101 = workflow m1 (Fig 3),
# 105 = workflow m2 (Fig 4), 109 = workflow m3 (Fig 5), 113 = healing loop (Fig 6), 117 = workflow m4 (Fig 7)
figure_mapping = {
    90: "Figure 1. Experimental methodology pipeline",
    94: "Figure 2. Overall system architecture of the adaptive CPU scheduling pipeline",
    101: "Figure 3. Workload generation and scheduling workflow",
    105: "Figure 4. Degradation detection and diagnosis workflow",
    109: "Figure 5. Verification, acceptance, and rollback workflow",
    113: "Figure 6. Self-healing controller workflow (one trace)",
    117: "Figure 7. Frontend-to-backend simulation execution flow",
}

# Read Mermaid code from file (hardcoded for reliability)
mermaid_code = {
    "Figure 1. Experimental methodology pipeline": """flowchart LR
    A["Workload<br/>Families<br/>(7 families)"] --> B["Train / Valid /<br/>Test Split<br/>(seeds 7101-7105,<br/>8201, 23819)"]
    B --> C["Arms<br/>(10 arms:<br/>fixed, rule,<br/>Q-learning)"]
    C --> D["Evaluation<br/>(210 traces/arm,<br/>bootstrap 1000)"]
    D --> E["Counterfactual<br/>Audit &<br/>Report"]""",

    "Figure 2. Overall system architecture of the adaptive CPU scheduling pipeline": """flowchart TB
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
    RB -.->|"Feedback (action applied to scheduler)"| RS""",

    "Figure 3. Workload generation and scheduling workflow": """flowchart TB
    FS["Family + Seed selection"] --> WG["WorkloadGenerator.generate()"]
    WG --> WO["Workload object (fingerprinted)"]
    DS["Optional: drift spec (drift.py)"] --> GD["generate_with_drift()"]
    GD --> DW["Drifted Workload (phase shift)"]
    WO --> RS["RuntimeSimulator.run(workload, ctrl)"]
    DW --> RS
    RS --> EL["Event loop: dispatch, switch, complete"]
    EL --> SR["ScheduleResult (per-process outcomes)"]""",

    "Figure 4. Degradation detection and diagnosis workflow": """flowchart TD
    CN["Completion notice from scheduler"] --> RW["Update rolling window (EWMA ref + band)"]
    RW --> TH{"window_mean > ref + k * band ?"}
    TH -- "yes" --> HY["Hysteresis check (h consecutive)"]
    TH -- "no" --> SU["Suppress (reason logged)"]
    HY --> GA{"Gates pass? (cooldown, load, work)"}
    GA -- "yes" --> RE["Build ready evidence"]
    GA -- "no" --> SU
    RE --> DG["Diagnose: primary cause"]
    DG --> DI["Declare incident"]""",

    "Figure 5. Verification, acceptance, and rollback workflow": """flowchart TD
    AA["Action applied; window opened"] --> CC["Collect completions over verify_decisions"]
    CC --> PC["Compute phase-adjusted closure residual"]
    PC --> AB{"residual >= accept boundary (+0.30)?"}
    AB -- "yes" --> AC["ACCEPTED (keep, learn)"]
    AB -- "no" --> RB{"residual <= reject boundary (-0.30)?"}
    RB -- "yes" --> RJ["REJECTED (rollback, fail +1)"]
    RB -- "no" --> IC["INCONCLUSIVE (rollback, no penalty)"]""",

    "Figure 6. Self-healing controller workflow (one trace)": """flowchart TD
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
    FQ -- "no" --> SM""",

    "Figure 7. Frontend-to-backend simulation execution flow": """flowchart LR
    UI["Browser UI (app.js)"] --> HR["HTTP Request POST /api/simulations"]
    HR --> FR["FastAPI Route (api.py)"]
    FR --> SL["Service Layer (service.py)"]
    SL --> SE["Simulation Engine (demo.run_fixture)"]
    SE --> MV["Model Validator (model.py)"]
    MV --> JR["JSON Response (workload, metrics, incidents)"]
    JR --> CH["Charts (inline SVG)"]
    CH --> RU["Render in UI (index.html)"]
    JR -.->|"JSON + SVG"| UI"""
}


def remove_inline_images_from_paragraph(para):
    """Remove all inline shape (image) elements from a paragraph's runs."""
    from lxml import etree
    nsmap = {
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    }
    for run in para.runs:
        for drawing in run._element.findall('.//w:drawing', nsmap):
            run._element.remove(drawing)
        for drawing in run._element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'):
            run._element.remove(drawing)


doc = Document(DOCX_PATH)

# For each image-bearing paragraph, remove the image and add Mermaid code
for para_idx, figure_label in figure_mapping.items():
    para = doc.paragraphs[para_idx]
    code = mermaid_code.get(figure_label, "")
    if not code:
        print(f"WARNING: No Mermaid code for {figure_label}")
        continue

    # Clear existing runs (which contain the image XML)
    remove_inline_images_from_paragraph(para)

    # Set paragraph style to monospace for code
    code_style = "Courier New"

    # Replace image paragraph with Mermaid code (keep the original caption paragraph intact)
    para.clear()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    
    # Add separator at top
    run = para.add_run("─" * 60 + "\n")
    run.font.name = code_style
    run.font.size = Pt(8)

    # Add Mermaid code block
    for line in code.split("\n"):
        if line.strip():
            run = para.add_run(line + "\n")
        else:
            run = para.add_run("\n")
        run.font.name = code_style
        run.font.size = Pt(8)

    # Add separator at bottom
    run = para.add_run("─" * 60)
    run.font.name = code_style
    run.font.size = Pt(8)

print(f"Replaced {len(figure_mapping)} PNG diagrams with Mermaid source code")

doc.save(DOCX_PATH)
print(f"Saved to {DOCX_PATH}")
