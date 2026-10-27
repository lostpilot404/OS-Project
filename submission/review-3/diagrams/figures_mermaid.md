# Mermaid Source Code for All Figures

This file contains the Mermaid source code for all 7 figures in the paper. You can render these diagrams using any Mermaid-compatible tool (e.g., https://mermaid.live, VS Code Mermaid preview, or GitHub markdown).

---

## Figure 1. Experimental methodology pipeline

```mermaid
flowchart LR
    A["Workload<br/>Families<br/>(7 families)"] --> B["Train / Valid /<br/>Test Split<br/>(seeds 7101-7105,<br/>8201, 23819)"]
    B --> C["Arms<br/>(10 arms:<br/>fixed, rule,<br/>Q-learning)"]
    C --> D["Evaluation<br/>(210 traces/arm,<br/>bootstrap 1000)"]
    D --> E["Counterfactual<br/>Audit &<br/>Report"]
```

---

## Figure 2. Overall system architecture of the adaptive CPU scheduling pipeline

```mermaid
flowchart TB
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
    RB -.->|"Feedback (action applied to scheduler)"| RS
```

---

## Figure 3. Workload generation and scheduling workflow

```mermaid
flowchart TB
    FS["Family + Seed selection"] --> WG["WorkloadGenerator.generate()"]
    WG --> WO["Workload object (fingerprinted)"]
    DS["Optional: drift spec (drift.py)"] --> GD["generate_with_drift()"]
    GD --> DW["Drifted Workload (phase shift)"]
    WO --> RS["RuntimeSimulator.run(workload, ctrl)"]
    DW --> RS
    RS --> EL["Event loop: dispatch, switch, complete"]
    EL --> SR["ScheduleResult (per-process outcomes)"]
```

---

## Figure 4. Degradation detection and diagnosis workflow

```mermaid
flowchart TD
    CN["Completion notice from scheduler"] --> RW["Update rolling window (EWMA ref + band)"]
    RW --> TH{"window_mean > ref + k * band ?"}
    TH -- "yes" --> HY["Hysteresis check (h consecutive)"]
    TH -- "no" --> SU["Suppress (reason logged)"]
    HY --> GA{"Gates pass? (cooldown, load, work)"}
    GA -- "yes" --> RE["Build ready evidence"]
    GA -- "no" --> SU
    RE --> DG["Diagnose: primary cause"]
    DG --> DI["Declare incident"]
```

---

## Figure 5. Verification, acceptance, and rollback workflow

```mermaid
flowchart TD
    AA["Action applied; window opened"] --> CC["Collect completions over verify_decisions"]
    CC --> PC["Compute phase-adjusted closure residual"]
    PC --> AB{"residual >= accept boundary (+0.30)?"}
    AB -- "yes" --> AC["ACCEPTED (keep, learn)"]
    AB -- "no" --> RB{"residual <= reject boundary (-0.30)?"}
    RB -- "yes" --> RJ["REJECTED (rollback, fail +1)"]
    RB -- "no" --> IC["INCONCLUSIVE (rollback, no penalty)"]
```

---

## Figure 6. Self-healing controller workflow (one trace)

```mermaid
flowchart TD
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
    FQ -- "no" --> SM
```

---

## Figure 7. Frontend-to-backend simulation execution flow

```mermaid
flowchart LR
    UI["Browser UI (app.js)"] --> HR["HTTP Request POST /api/simulations"]
    HR --> FR["FastAPI Route (api.py)"]
    FR --> SL["Service Layer (service.py)"]
    SL --> SE["Simulation Engine (demo.run_fixture)"]
    SE --> MV["Model Validator (model.py)"]
    MV --> JR["JSON Response (workload, metrics, incidents)"]
    JR --> CH["Charts (inline SVG)"]
    CH --> RU["Render in UI (index.html)"]
    JR -.->|"JSON + SVG"| UI
```

---

## Notes

- All 7 figures are now in correct sequential order (1-7)
- Each figure appears in its corresponding section:
  - Figure 1: §9.6 Experimental Design
  - Figure 2: §10 Overall Conceptual and Architectural Diagram
  - Figure 3: §12.1 Workload Generation
  - Figure 4: §12.2 Degradation Detection
  - Figure 5: §12.3 Verification and Acceptance
  - Figure 6: §12.4 Self-Healing Loop
  - Figure 7: §12.5 Frontend-to-Backend Execution Flow
- To render: Copy the Mermaid code blocks into any Mermaid renderer
- These diagrams replace the original PNG images in the DOCX and PDF
