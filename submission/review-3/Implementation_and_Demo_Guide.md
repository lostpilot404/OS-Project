# Implementation and Demonstration Guide

**Project:** Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads  
**Submission:** Review 3 — Complete Work and Final Paper Submission (Lab Assessment 10)

---

## 1. System Requirements

- **Operating System:** Windows 10/11 with WSL2 (Ubuntu 22.04 recommended), or native Linux
- **Python:** 3.11.x (recorded environment: CPython 3.11.2)
- **RAM:** 4 GB minimum (study completes in ~47 seconds)
- **Disk:** 500 MB for repository and dependencies

## 2. Environment Setup

### 2.1 Clone the Repository

```bash
git clone https://github.com/lostpilot404/OS-Project.git
cd OS-Project
```

### 2.2 Create Virtual Environment

```bash
# Linux / WSL
python3.11 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.3 Install Dependencies

```bash
# Core dependencies (for study and tests)
pip install -r requirements.lock

# Dashboard dependencies (for web interface)
pip install -r requirements-dashboard.txt

# Report generation (optional)
pip install -r requirements-report.txt
```

### 2.4 Verify Installation

```bash
# Run the full test suite
pytest tests/ -q

# Expected: ~500 tests pass, 1 warning (plotting-related)
```

## 3. Launch Commands

### 3.1 CLI Demonstration (Text Output)

```bash
python main.py demo
```

This runs two small deterministic self-healing rule fixtures and prints observations, scheduling segments, incidents, verification results, and metrics. Output is saved to `results/demo/demo.json`.

**Important:** Do not omit `demo` — the no-command default launches the full research workflow.

### 3.2 Interactive Web Dashboard

```bash
python tools/serve_dashboard.py
```

Open browser to **http://127.0.0.1:8000/**

Press Ctrl+C to stop the server.

### 3.3 Full Research Study (Optional)

```bash
python main.py healing-study
```

This runs the complete frozen study (~47 seconds) and writes 21 artifacts to `results/healing/`.

### 3.4 Static Classroom Dashboard

```bash
python tools/build_teacher_dashboard.py --input results/demo/demo.json --output results/demo/index.html
```

Opens from the file system — no server needed.

## 4. Demonstration Script for Live Assessment

### Step 1: Overview Page (30 seconds)
**What to show:** The dashboard overview page with scenario counts and study summary.  
**What to say:** "The system monitors CPU scheduling performance and applies recovery when degradation is detected. The overview shows the available scenarios and the committed research results."

### Step 2: Teaching Example — 3 Processes (1-2 minutes)
**What to show:** Simulation Studio → Teaching Example → Run.  
**What to demonstrate:** Run the 3-process teaching example under FCFS, then SJF. Show the Gantt chart.  
**What to say:** "This is a simple 3-process example. Under FCFS, process P2 waits for P1 to complete. Under SJF, the shortest job runs first, reducing total waiting time. The waiting times are calculated exactly as the textbook formulas specify."

### Step 3: Accepted Recovery — Seed 13 (1-2 minutes)
**What to show:** Simulation Studio → Accepted Recovery → Run.  
**What to demonstrate:** Show the incident with verdict "accepted."  
**What to say:** "The system detected degradation (waiting time exceeded the stable reference), diagnosed the cause as fairness starvation, applied a recovery action (switched to a different policy), and verified the improvement. The verification passed, so the change was accepted and kept."

### Step 4: Inconclusive Rollback — Seed 3 (1-2 minutes)
**What to show:** Simulation Studio → Inconclusive Recovery → Run.  
**What to demonstrate:** Show the incident with verdict "inconclusive."  
**What to say:** "Here the system detected degradation and applied a recovery, but the verification window did not provide enough evidence to confirm the improvement. The change was rolled back without penalty — the system prefers to undo an unverified change rather than risk keeping a harmful one."

### Step 5: Policy Comparison (1-2 minutes)
**What to show:** Policy Comparison page → Run comparison.  
**What to demonstrate:** Compare FCFS, SJF, Round Robin, and Priority on the same workload.  
**What to say:** "All four policies run on identical workloads, so the comparison is fair. SJF minimizes mean waiting time but requires knowing burst times in advance. Round Robin has good response time but high context-switch overhead. The adaptive system can switch between these policies based on observed performance."

### Step 6: Research Results (1-2 minutes)
**What to show:** Research Results page.  
**What to demonstrate:** Show the held-out comparison table, counterfactual audit, and verification behaviour.  
**What to say:** "These are the committed research results from 210 held-out test traces. The adaptive rule reduces mean waiting time by about 8.5 units compared to FCFS, but doesn't beat the SJF oracle. The Q-learning controller is statistically indistinguishable from the hand-written rule. Every accepted intervention was genuinely helpful, but only 8% of helpful interventions can be verified."

### Step 7: Custom Workload (1 minute)
**What to show:** Workload Explorer → Create custom workload.  
**What to demonstrate:** Enter arrival times, bursts, and priorities. Run under different policies.  
**What to say:** "The system supports custom workloads. You can specify any number of processes with their arrival times, burst times, and priorities, then compare how different scheduling policies handle them."

### Step 8: Explain the Self-Healing Loop (1-2 minutes)
**What to show:** Architecture diagram from the paper (Figure 1).  
**What to say:** "The self-healing loop has ten stages: generate workload → simulate dispatch → monitor performance → detect degradation → diagnose cause → select recovery → apply action → verify outcome → accept/rollback/fallback → record incident. The causal boundary ensures the controller never sees future arrivals or the finished schedule."

### Step 9: Explain Verification (1 minute)
**What to show:** Verification workflow (Figure 5 from the paper).  
**What to say:** "The tri-state verifier compares the observed improvement against a phase-adjusted baseline. If the residual exceeds +0.30, the change is accepted. If below -0.30, it is rejected as a confirmed failure. Between these boundaries, the verdict is inconclusive — the system rolls back without penalty. This ensures the system never keeps a change it cannot verify."

### Step 10: Honest Limitations (30 seconds)
**What to say:** "The Q-learning controller does not outperform the hand-written rule. The verifier can certify only 8% of genuinely helpful interventions. The system is a simulation, not a kernel modification. These are honest findings that the project reports transparently."

## 5. Expected Observable Outcomes

| Demonstration | Expected Outcome |
|---|---|
| CLI demo | Two fixture outputs with incidents, metrics, and JSON |
| Teaching example | Correct Gantt charts matching hand calculations |
| Accepted recovery | Incident with verdict "accepted", policy change kept |
| Inconclusive recovery | Incident with verdict "inconclusive", policy rolled back |
| Policy comparison | Bar chart showing FCFS > SJF for waiting time |
| Research results | Tables matching committed CSV artifacts |
| Custom workload | Schedule computed on demand, exportable as JSON |

## 6. Troubleshooting

| Problem | Solution |
|---|---|
| Port 8000 already in use | Kill the existing process or change the port in serve_dashboard.py |
| Module not found errors | Ensure virtual environment is activated and dependencies installed |
| pytest failures | Run `pip install -r requirements.lock` again; check Python version |
| Dashboard shows errors | Check browser console; ensure backend is running |
| Demo command runs full study | Use `python main.py demo` (not just `python main.py`) |
| No output after running demo | Check `results/demo/demo.json` for saved output |

## 7. Known Limitations

- Simulation only — no real kernel, no real workload traces
- Q-learning provides no measurable advantage over the rule-based approach
- Verifier certifies only 8% of beneficial interventions
- Fairness rule is empirically harmful on the tested workload families
- Detector has a measurable false-trigger rate (0.62-0.65 events per stable trace)

## 8. File Locations

| File | Description |
|---|---|
| `results/demo/demo.json` | Demo fixture output |
| `results/healing/` | 21 frozen study artifacts |
| `results/healing/healing_summary.json` | Summary of the complete study |
| `results/healing/healing_comparisons.csv` | Paired comparison results |
| `results/healing/healing_audit.csv` | Counterfactual audit |
| `docs/ARCHITECTURE.md` | Detailed architecture documentation |
| `docs/EVALUATION_RESULTS.md` | Every number traceable to artifacts |
| `docs/LIMITATIONS.md` | Complete limitations list |
| `docs/TEACHER_LIVE_DEMO.md` | Classroom demonstration guide |
