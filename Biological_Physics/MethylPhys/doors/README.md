# Physics of Methylation — start here

**What this field is called (2026-09-20): Physics of Methylation: Landauer Metrology** — measuring how far above the thermal noise quantum an information-writing process operates, against a fixed physical zero (H_min per cell class). Thermal noise is the unit (M = E_drive / k_B T), not the nuisance. Prior art: Sanchez & Mackenzie 2016 established that the methylome obeys Landauer's bound; Landauer metrology measures how far above it each cell class operates ([[Issue 003](../manual/IAMPerformance_GAPEIssue003_RC1.pdf)](../manual/IAMPerformance_GAPEIssue003_RC1.pdf) §0b).


This folder is the researcher-facing entry to the cellular track of the Informational Actualization Model: the claim, the current report, the kit that lets you verify the measurement chain on your own machine, the operating procedure, the papers, and the plates that show why the methylome is treated as a sky.

**The claim, in one paragraph.** A cell maintains its methylation pattern by irreversible information writing, and at body temperature that writing has a thermodynamic cost. For each of eight cellular architecture classes there is a floor entropy `H_min` below which a healthy cell of that class does not operate; it is calibrated once on healthy reference cells (an MCMC posterior, frozen before any disease sample is scored) and never re-fitted. A sample's reading is `A = H(β̄)/H_min` — the binary Shannon entropy of its mean methylation at the class's identity loci, over the class floor — read against an age-matched healthy band. Cohorts are used only to establish which *direction* a disease moves; they are never the baseline. This is the same measurement made in semiconductors (energy per switch over the Landauer floor) and in superconducting qubits, which is why the Mahaffey number `M = E_drive/(k_B T)` appears in all three.

| | file | what it is |
|---|---|---|
| **Report** | [`MethylPhys/manual/IAMPerformance_GAPEIssue003_RC1.pdf`](MethylPhys/manual/IAMPerformance_GAPEIssue003_RC1.pdf) | GAPE Issue 003 (September 2026, 263 pp). Supersedes Issue 002 (April 2026, pre-Atlas, in `Papers/`). Regenerate: `python MethylPhys/manual/build_gape_issue003.py out.pdf (historical path)` with `CPG_TRIAL` pointing at the runtime JSONs |
| **Verify it yourself** | [`MethylPhys/doors/RUNBOOK.md`](MethylPhys/doors/RUNBOOK.md) → [`RUNBOOK.md`](RUNBOOK.md) | five scripts, each printing input / operation / expected / observed / verdict. Every link of the chain from raw IDAT to the sealed anchors reproduces on a machine that had never seen the project |
| **Where every component lives** | [`MethylPhys/doors/COMPONENT_MAP.md`](MethylPhys/doors/COMPONENT_MAP.md) | repo vs. large local inputs vs. vault IP |
| **Operating procedure** | [`SOP/CPG_Chain_of_Custody_SOP_v2_0_0.md`](../sop/CPG_Chain_of_Custody_SOP_v2_0_0.md) | the chain stage by stage. **Read the SUPERSESSION LEDGER at the top first** — it maps every section superseded by the July 2026 commits to the file that is now authoritative. §105–§107 carry the scoring rulings and the July wiring |
| **Papers** | [`Papers/`](../papers/) | the cell-thermodynamics paper, Issue 002, the Hubble→GAPE derivation chain (`IAM_Hubble2GAPE_Alpha_Omega_4.tex`), and figures |
| **The translation map** | [`CMB_TO_METHYLOME_MAP.md`](CMB_TO_METHYLOME_MAP.md) | 79 CMB-analysis modules mapped to their methylome analogs by the author before the chain was built, with a 2026-09-19 status column: what got built, what was reversed (a second deconvolver; de-aging) and why |
| **The completion sprint, scored** | [`COMPLETION_SPRINT_scored.md`](COMPLETION_SPRINT_scored.md) | the spring-2026 plan (A: nulls → B: foregrounds → C: correlation → D: covariance → E: likelihood) against what was built, cut, refused, or never started — and why the order was wrong |
| **Plates** | [`Plates/`](../plates/) | Plate 01 Cosmic Methylome Background · 02 Breast anisotropy · 03 Methylome CMB vs microwave CMB · 04 Patterns · **05 Four skies** (Issue 003: Planck realization, Atlas immune posterior mean and sd, one patient's z-departure, one HEALPix grid) |

**Why a sky.** The atlas is a reference map with a per-pixel uncertainty (posterior mean and sd at every CpG); a patient is one observation whose residual against that map is read pixel by pixel. That is the Planck workflow, and it is why the CMB toolkit — HEALPix, Mollweide, matched filters, residual maps — transfers. Plate 05 shows the four skies side by side.

**What is derived and what is calibrated** is stated per quantity in Issue 003 §1 (reconciliation table) and §9 (the ledger). The floors are calibrated on healthy references, not fitted to disease data; the physical interpretation is stated as such.

Related: the atlas itself is in [`../MethylPhys/atlas/`](../atlas/), the running code in [`../MethylPhys/chain/`](../chain/), and every validation run in [`../Record/`](../../Record/).

*Research stage. Nothing here is clinical validation.*
