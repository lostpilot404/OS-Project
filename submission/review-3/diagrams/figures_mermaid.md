# Mermaid Source for Review 3 Figures

Copy any of these diagrams into [mermaid.live](https://mermaid.live) or a Markdown viewer
that supports Mermaid to render them. All diagrams use black-and-white styling.

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
        WG["Workload<br/>Generator<br/>(workload/)"] --> WF["Workload<br/>Fingerprint<br/>(fingerprint.py)"]
        WF --> DI["Drift Injector<br/>(drift.py)"]
        DI --> BE["Burst<br/>Estimator<br/>(burst_estimator.py)"]
    end

    subgraph Scheduling
        WF --> RS["Discrete-Event<br/>Scheduler<br/>(scheduler/runtime.py)"]
        RS --> RO["Runtime<br/>Observation<br/>(RuntimeObservation)"]
        DI --> RS
        RO --> PC["Policy Classes<br/>(FCFS/SJF/RR/Pri)"]
        BE --> RS
    end

    subgraph Self-Healing Controller
        RS --> CM["Causal Monitor<br/>(monitor.py)"]
        CM --> DD["Degradation<br/>Detector<br/>(detector.py)"]
        DD --> DX["Diagnosis<br/>(diagnosis.py)"]
        DX --> RP["Recovery<br/>Policy<br/>(policies.py)"]
    end

    subgraph Verification and Outcome
        RP --> VW["Verification<br/>Window<br/>(verifier.py)"]
        VW --> AR["Accept / Reject<br/>/ Inconclusive"]
        AR --> RB["Rollback /<br/>Fallback<br/>(actions.py)"]
        RB --> IL["Incident<br/>Ledger"]
    end

    subgraph Evaluation
        VW --> MT["Metrics<br/>(evaluation/metrics.py)"]
        AR --> RL["Reliability<br/>(evaluation/reliability.py)"]
        RB --> BP["Paired Comparison<br/>(evaluation/paired.py)"]
    end

    RB -.->|"Feedback<br/>(action applied<br/>to scheduler)"| RS
```

---

## Figure 3. Self-healing controller workflow (one trace)

```mermaid
flowchart TD
    GW["Generate<br/>workload"] --> SM["Simulate<br/>(dispatch slice)"]
    SM --> MN["Monitor<br/>(EWMA reference)"]
    MN --> DD{"Degradation<br/>detected?"}
    DD -- "no: continue<br/>simulating" --> SM
    DD -- "yes" --> DC["Diagnose<br/>cause"]
    DC --> SA["Select recovery<br/>action"]
    SA --> AA["Apply action<br/>to scheduler"]
    AA --> VW["Verification<br/>window"]
    VW -->|"good"| AC["Accepted<br/>(keep change)"]
    VW -->|"bad"| RJ["Rejected<br/>(rollback + failure)"]
    VW -->|"unclear"| IN["Inconclusive<br/>(rollback, no penalty)"]
    AC --> IL["Incident<br/>Ledger"]
    RJ --> IL
    IN --> IL
    AC --> FQ{"2 consecutive<br/>failures?"}
    FQ -- "yes" --> SF["Safe<br/>fallback"]
    FQ -- "no" --> SM
    VW -.->|"record every<br/>decision"| IL
```

---

## Figure 4. Workload generation and scheduling workflow

```mermaid
flowchart TB
    FS["Family + Seed<br/>selection"] --> WG["WorkloadGenerator<br/>.generate()"]
    WG --> WO["Workload object<br/>(fingerprinted)"]
    DS["Optional: drift<br/>spec (drift.py)"] --> GD["generate_with_<br/>drift()"]
    GD --> DW["Drifted Workload<br/>(phase shift)"]
    WO --> RS["RuntimeSimulator<br/>.run(workload, ctrl)"]
    DW --> RS
    RS --> EL["Event loop:<br/>dispatch, switch, complete"]
    EL --> SR["ScheduleResult<br/>(per-process outcomes)"]
```

---

## Figure 5. Degradation detection and diagnosis workflow

```mermaid
flowchart TD
    CN["Completion notice<br/>from scheduler"] --> RW["Update rolling window<br/>(EWMA ref + band)"]
    RW --> TH{"window_mean > ref<br/>+ k * band ?"}
    TH -- "yes" --> HY["Hysteresis check<br/>(h consecutive)"]
    TH -- "no" --> SU["Suppress<br/>(reason logged)"]
    HY --> GA{"Gates pass?<br/>(cooldown, load, work)"}
    GA -- "yes" --> RE["Build ready<br/>evidence"]
    GA -- "no" --> SU
    RE --> DG["Diagnose:<br/>primary cause"]
    DG --> DI["Declare<br/>incident"]
```

---

## Figure 6. Verification, acceptance, and rollback workflow

```mermaid
flowchart TD
    AA["Action applied;<br/>window opened"] --> CC["Collect completions<br/>over verify_decisions"]
    CC --> PC["Compute phase-adjusted<br/>closure residual"]
    PC --> AB{"residual >= accept<br/>boundary (+0.30)?"}
    AB -- "yes" --> AC["ACCEPTED<br/>(keep, learn)"]
    AB -- "no" --> RB{"residual <= reject<br/>boundary (-0.30)?"}
    RB -- "yes" --> RJ["REJECTED<br/>(rollback, fail +1)"]
    RB -- "no" --> IC["INCONCLUSIVE<br/>(rollback, no penalty)"]
```

---

## Figure 7. Frontend-to-backend simulation execution flow

```mermaid
flowchart LR
    UI["Browser UI<br/>(app.js)"] --> HR["HTTP Request<br/>POST /api/simulations"]
    HR --> FR["FastAPI Route<br/>(api.py)"]
    FR --> SL["Service Layer<br/>(service.py)"]
    SL --> SE["Simulation Engine<br/>(demo.run_fixture)"]
    SE --> MV["Model Validator<br/>(model.py)"]
    MV --> JR["JSON Response<br/>(workload, metrics,<br/>incidents)"]
    JR --> CH["Charts<br/>(inline SVG)"]
    CH --> RU["Render in UI<br/>(index.html)"]
    JR -.->|"JSON +<br/>SVG"| UI
```
