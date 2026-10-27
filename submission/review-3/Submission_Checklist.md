# Submission Checklist — Review 3 (Corrected)

**Project:** Adaptive CPU Scheduling Using Reinforcement Learning for Dynamic Workloads  
**Date:** 2026-10-27 (corrected version)

---

## Required Sections (22 total)

| # | Section | Status | Notes |
|---|---------|--------|-------|
| 1 | Final Project Title | ✅ Complete | Title matches project |
| 2 | Abstract | ✅ Complete | Self-contained, includes quantitative results, states limitation |
| 3 | Keywords | ✅ Complete | 10 keywords matching implementation |
| 4 | Introduction | ✅ Complete | Problem, motivation, scope, contribution |
| 5 | Related Work / Literature Survey | ✅ Complete | 31 papers 2021-2026, synthesised by theme |
| 6 | Inferences and Research Gaps | ✅ Complete | Synthesised from literature, identifies gap |
| 7 | Problem Statement | ✅ Complete | Two problems, assumptions, scope, constraints |
| 8 | Objectives and Novelty | ✅ Complete | 6 measurable objectives, honest novelty claim |
| 9 | Proposed Methodology | ✅ Complete | Pipeline from generation to evaluation; includes Figure 1 |
| 10 | Overall Architecture Diagram | ✅ Complete | Monochrome PNG and SVG, Figure 2 |
| 11 | Individual Module Descriptions | ✅ Complete | Table with 23 modules |
| 12 | Module-wise Workflow Diagrams | ✅ Complete | 5 diagrams: Figures 3-7 |
| 13 | Dataset Description | ✅ Complete | Synthetic data, 7 families, documented |
| 14 | Experimental Setup | ✅ Complete | Software, commands, parameters, outputs |
| 15 | Evaluation Metrics | ✅ Complete | 7 metrics with formulas, units, preference |
| 16 | Results and Discussion | ✅ Complete | Tables from actual artifacts, negative results |
| 17 | Comparison with Existing Methods | ✅ Complete | Comparison table, no incompatible numerical claims |
| 18 | Limitations | ✅ Complete | 10 specific limitations with measurements |
| 19 | Conclusion and Future Scope | ✅ Complete | Summary + 6 future work items (all unimplemented) |
| 20 | References | ✅ Complete | 34 entries, APA 7th, alphabetical; all DOIs verified |
| 21 | Source Code and Supporting Documents | ✅ Complete | Inventory table, installation instructions |
| 22 | Demonstration Guide | ✅ Complete | 10-step script, troubleshooting, limitations |

## Deliverables

| # | Deliverable | Status | File |
|---|-------------|--------|------|
| 1 | Final Research Paper (DOCX) | ✅ | `Final_Research_Paper.docx` |
| 2 | Final Research Paper (PDF) | ✅ | `Final_Research_Paper.pdf` (generated with reportlab) |
| 3 | Literature Verification Ledger | ✅ | `Literature_Verification_Ledger.csv` |
| 4 | Citation Audit | ✅ | `Citation_Audit.md` (rebuilt after corrections) |
| 5 | Architecture Diagram (PNG) | ✅ | `diagrams/fig_architecture.png` |
| 6 | Architecture Diagram (SVG) | ✅ | `diagrams/fig_architecture.svg` |
| 7 | Workflow Diagrams (PNG) | ✅ | `diagrams/fig_*.png` (6 files) |
| 8 | Workflow Diagrams (SVG) | ✅ | `diagrams/fig_*.svg` (6 files) |
| 9 | Methodology Diagram | ✅ | `diagrams/fig_methodology.png` |
| 10 | Mermaid Source | ✅ | `diagrams/figures_mermaid.md` (7 diagrams) |
| 11 | Implementation and Demo Guide | ✅ | `Implementation_and_Demo_Guide.md` |
| 12 | Source Code Inventory | ✅ | `Source_Code_Inventory.md` |
| 13 | Submission Checklist | ✅ | This file |
| 14 | Source Code ZIP | ✅ | `OS_Project_Source.zip` |
| 15 | PDF Generation Script | ✅ | `build_pdf.py` |

## Corrections Applied

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | Fu et al. (2024) ALPS title wrong ("heterogeneous functions") | Corrected to "serverless functions"; full authors added; USENIX URL corrected |
| 2 | Chen et al. (2023) DOI wrong (`s00586`) | Corrected to `s10586`; description corrected from HPC survey to DRL for edge computing |
| 3 | Sabek et al. (2022) DOI wrong (`3507913.3507913`) | Corrected to `3514221.3526158` per ACM DL |
| 4 | IADA attributed to "Martins et al." | Corrected to Meyer, da Silva, Kirchoff, & De Rose; all in-text citations updated |
| 5 | Qin & Lu (2021) cited for wrong topic | Citation removed; paragraph rewritten |
| 6 | SHML (2024) had no formal reference | Added Rauba et al. NeurIPS reference |
| 7 | Wei et al. (2024) title truncated, authors incomplete | Corrected title with subtitle; full authors (Wei, Liu, Ying) |
| 8 | Yahmed et al. (2023) had 8 authors | Corrected to 4 authors per DOI |
| 9 | Training count math error (`6 × 40 = 1200`) | Corrected to "1200 episodes per seed cycling through 6 families" |
| 10 | Figure numbering out of order | Renumbered: methodology=Fig 1, architecture=Fig 2, ..., frontend=Fig 7 |
| 11 | PNG diagrams took up space, hard to edit | Replaced all 7 PNG diagrams with Mermaid source code blocks (Courier New 8pt); PNG/SVG still in `diagrams/` for reference |
| 12 | Table columns had equal widths, squishing long content | PDF builder now uses proportional column widths based on max content length, capped at 45% per column |

## Quality Gate

| Check | Status | Notes |
|-------|--------|-------|
| All 22 sections present | ✅ | |
| ≥30 papers 2021-2026 | ✅ | 31 confirmed (2 preprints clearly labelled) |
| Every reference cited | ✅ | 34/34 |
| Every citation has reference | ✅ | 34/34 |
| Literature ledger complete | ✅ | 34 entries with verification sources |
| Diagrams monochrome | ✅ | Black lines, white background |
| Results match artifacts | ✅ | All numbers from results/healing/ |
| Q-learning described accurately | ✅ | No advantage claimed |
| Limitations included | ✅ | 10 specific limitations |
| Negative findings included | ✅ | No RL advantage, verifier power |
| DOCX readable | ✅ | Generated and saved |
| PDF generated | ✅ | reportlab; 799 KB |
| PDF layout inspected | ⚠️ | Generated but student should verify page breaks, table flow, figure placement |
| Demo guide functional | ✅ | Based on actual code and commands |
| Missing student info marked | ✅ | Placeholders on title page |
| No existing changes discarded | ✅ | Git status unchanged |
| No fabricated results | ✅ | All from committed artifacts |
| Training count correct | ✅ | 1200 episodes/seed, not 6×40 |
| Figure numbers sequential | ✅ | 1-7 in document order |

## Remaining Student Actions

1. **Open the PDF** in a viewer and verify:
   - Page breaks don't cut tables or figures
   - Figures are legible at print size
   - Table text doesn't overflow
   - Page numbers appear
   - Title page looks correct

2. **Fill student information** on the title page of both DOCX and PDF:
   - Replace `[Student Name(s)]`
   - Replace `[Roll Number(s)]`
   - Replace `[Department Name]`
   - Replace `[Institution Name]`
   - Replace `[Academic Year]`
   - Replace `[Date]`

3. **Add Table of Contents** in Word (References → Table of Contents)

4. **Re-export PDF** from the edited DOCX if student makes changes
