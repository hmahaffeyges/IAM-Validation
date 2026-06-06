# CPG Chain-of-Custody SOP — Part I (Foundations) (v1.3 — Stages 4.5 + 4.6 + 6-tier breakpoints + smoking/sex foregrounds)

> **CHANGELOG v1.2 → v1.3 (2026-06-06):**
> - **§36 Sex-axis foreground subtraction** — module rewrite: `sex_axis_foreground.py` now built; per-CpG β = α + ψ·indicator_male + ε with chrX/chrY/XCI flag handling; layer CSV pending v1.3 fit on n_hc=601 cohort. Stage 7 sex-stratified threshold tables remain as interim mitigation until layer fit complete.
> - **§39 Smoking-axis foreground subtraction** — module rewrite: `smoking_axis_foreground.py` now built; per-CpG β = α + δ·indicator_current + φ·recency_score + ε; recency mapped from smoking_bin (never=0.00 to current=1.00); layer CSV pending v1.3 fit on n_hc=601 cohort. Stage 7 smoking-bin threshold-stratification (in `tier_breakpoints.json v1.2`) remains as interim mitigation until layer fit complete.
> - **§46.5 NEW — Stage 4.5 Bidirectional decomposition** — mirrors the sealed VAL-051 `a_dir_score` formula at patient runtime. Sign-multiplied z-scores against frozen training-set HC mean/SD averaged across covered CpGs. Pooled-entropy comparator on parent panel. FLAG_BIDIRECTIONAL fires when pooled mute + directional loud. v1.0 panel coverage: immune class only (VAL-051 Rule A 7-CpG); other 7 classes return NO_PANEL honestly.
> - **§46.6 NEW — Stage 4.6 Patient brightness comparison** — per-class z-score departure of patient β from each frozen healthy class brightness reference, projected onto HEALPix NSIDE=128 grid for Mollweide rendering. Patient's personal Cosmic Microwave Methylome (customer-facing analog of CPG Plate 1). `iamatlas_cpg_to_healpix_nside128.npy` generated as one-time output of `generate_cpg_healpix_mapping.py` against the IAMAtlas REBUILD CSV + EPIC v1 B4 manifest.
> - **§59 Step 7.1 — Per-class tier call** — replaced 4-tier statistical-percentile schema (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH) with **6-tier physics-derived schema** (SUPPRESSED / NORMAL / ELEVATED / WARBURG_TRANSITION / SIGNIFICANTLY_ELEVATED / BREACH). The 1.07 Warburg line + 1.10 architectural-fidelity breach line are the framework's physics-defined inflection points, not statistical percentiles. Per-class structural ceiling table (1/H_min per class; stem_pluri structurally blind for BREACH at ceiling 1.0181). 7 covariate-override modes (EXPECTED_SUPPRESSION / TRAJECTORY_WATCH / TREATMENT_RESPONSE / CONTEXT_PREGNANCY / POSTPARTUM / HRT_BASELINE / WEIGHT_LOSS_INTERVENTION) triggered by patient intake covariates per BUILD_SPEC v1.2 §4.5. Smoking-bin override table. Bidirectional pattern handoff from §46.5 (directional composite drives tier when FLAG_BIDIRECTIONAL set). CI-based tier confidence propagation (BORDERLINE_TIER flag at 0.20 prob threshold from MCMC posteriors).
> - All other §-sections unchanged from v1.2.

> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them across all parts; this particular part (Foundations / L9 machinery)
> was already clean of fabricated paths in v1, so the disclaimer is included here
> only for version-consistency across the SOP. All real paths in v1.1 are documented
> in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be treated as
> not-yet-existing until verified against the repo.

---

# CPG Chain-of-Custody Standard Operating Procedure (SOP)

**Version:** v1 — 2026-05-31
**Author:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Working partner:** Walther (Claude)
**Repository:** https://github.com/hmahaffeyges/IAM-Validation
**Operative scope:** From IDAT-file-on-server to delivered-report. Every step. Every tool. Every reference. Every failure mode.

---

## How to use this document

This is the operator's manual for the Cellular Performance Gauge (CPG) chain of custody. It is granular to the quantum level — every step has its own section. Every step section follows the same template:

```
What this step does
Inputs (exact files/format)
Atlas reference (whether and how IAMAtlas is consulted)
Files invoked (modules + lookup tables)
The math (formulas, code, references)
CMB equivalent (what cosmologists call this; why it's the same operation)
How the methylome differs in implementation
How it's the same in principle
Outputs (exact files/format)
Decision points (yes/no routing to other steps)
Failure modes (what can go wrong; how to detect; what to do)
Canonical cross-references (Recipe §X, Roadmap §Y, related VALs)
CPG Plate references (where applicable)
Chain-link assignment (L1 through L9)
```

If a question can be answered by reading any single step section in isolation, this document is doing its job. If you have to read three sections to answer one question, that's a documentation bug — file it as an SOP-update VAL and we'll fix it.

A reader executing this document by hand should arrive at the same readout the engine produces. That's the test. Any deviation is either an SOP failure or an engine failure; both are findings.

---

## TABLE OF CONTENTS

### Part I — Foundations

- §1   The chain at a glance (visual + one-paragraph summary per stage)
- §2   The L1-L9 chain of custody — full grading table + what each link owes
- §3   The stages vs. the links — operational sequence × audit discipline
- §4   IAMAtlas — what it is, where it lives, when it's consulted
- §5   The Mahaffey Number (H_min) — the eight frozen anchors
- §6   The CMB → methylome translation principle (why the math is the same)
- §7   Bidirectional calibration — solved by entropy-space scoring (the structural physics win)
- §8   Vocabulary — what to say and what NOT to say (chain not pipeline, measure not classify, confirm not validate)
- §9   The two-deconvolver discipline — when each runs, why both, what disagreement means
- §10  Stage 6 reporting layer — the legal boundary between physics measurement and customer communication

### Part II — The step-by-step chain of custody

**Stage 0 — Sample intake (L1)**
- §11  Step 0.1 — IDAT file arrival on server
- §12  Step 0.2 — Sample manifest creation
- §13  Step 0.3 — IDAT integrity hash check
- §14  Step 0.4 — Control probe validation
- §15  Step 0.5 — Detection p-value QC per probe
- §16  Step 0.6 — Bead count QC
- §17  Step 0.7 — Sample-level call rate
- §18  Step 0.8 — Sex check vs metadata
- §19  Step 0.9 — Stage 0 decision gate (proceed / quarantine / reject)

**Stage 1 — Calibration & β computation (L2 + L3)**
- §20  Step 1.1 — Dye-bias correction
- §21  Step 1.2 — Probe-type normalization (funnorm vs noob vs SWAN vs BMIQ)
- §22  Step 1.3 — Batch correction (ComBat) when applicable
- §23  Step 1.4 — Bisulfite conversion efficiency check
- §24  Step 1.5 — β-value computation (β = M / (M + U + 100))
- §25  Step 1.6 — β-value sanity checks (range, distribution shape)
- §26  Step 1.7 — Probe response function (L3 grade improvement, currently provisional)
- §27  Step 1.8 — Stage 1 output: per-CpG β matrix

**Stage 2 — Deconvolution (L4 component separation, primary)**
- §28  Step 2.1 — IAMAtlas REBUILD load (the calibrated instrument enters)
- §29  Step 2.2 — Per-class marker pool extraction from `iamatlas_celltype_markers_v0_2.json`
- §30  Step 2.3 — Walther IAM Deconvolver — Path 1 (NNLS)
- §31  Step 2.4 — Walther per-class confidence + status codes
- §32  Step 2.5 — NILC v2 deconvolver — Path 2 (departure-from-consensus GLS)
- §33  Step 2.6 — Cross-method gate check (Walther vs NILC, biological-inference layer)
- §34  Step 2.7 — Stage 2 output: per-class fractions + cross-method gate verdict

**Stage 3 — Foreground subtraction (L4 component separation, secondary)**
- §35  Step 3.1 — Age-axis foreground subtraction (`age_axis_foreground.py`)
- §36  Step 3.2 — Sex-axis foreground (when present)
- §37  Step 3.3 — Batch/plate foreground (when present)
- §38  Step 3.4 — Ancestry foreground (when present)
- §39  Step 3.5 — Smoking foreground (when card requires it)
- §40  Step 3.6 — Stage 3 output: cleaned β matrix

**Stage 4 — A-score computation (entropy scoring)**
- §41  Step 4.1 — Per-class β_mean computation
- §42  Step 4.2 — Shannon entropy H(β_mean) calculation
- §43  Step 4.3 — Per-class A-score: A = H(β_mean) / H_min(class)
- §44  Step 4.4 — Per-cell-type A-score (115 cell types) via `iamatlas_a_scoring.py`
- §45  Step 4.5 — Disease panel A-score (when card has a curated panel)
- §46  Step 4.6 — Stage 4 output: 8 class A-scores + 115 cell-type A-scores

**Stage 5 — Multi-D departure (Mahalanobis hyper-volume)**
- §47  Step 5.1 — Patient 115-cell-type A-score vector assembly
- §48  Step 5.2 — HC centroid load (`mahalanobis_healthy_reference_v0_1.json`)
- §49  Step 5.3 — Inverse-covariance distance computation
- §50  Step 5.4 — Top-10 axis contribution decomposition
- §51  Step 5.5 — Stage 5 output: Mahalanobis distance + per-axis explainability

**Stage 6 — Cellular age inversion**
- §52  Step 6.1 — Per-class A-score input (from Stage 4)
- §53  Step 6.2 — Age reference matrix load (`age_reference_matrix.json`)
- §54  Step 6.3 — Per-class A inversion against the 80-cell baseline curve
- §55  Step 6.4 — Saturation handling (SAT_HIGH / SAT_LOW / OK / INSUFFICIENT_CPGS)
- §56  Step 6.5 — Eight per-class cellular ages — never collapsed by default
- §57  Step 6.6 — Percentile rank at patient's chronological age
- §58  Step 6.7 — Stage 6 output: per-class cellular age vector + optional summary

**Stage 7 — Tier breakpoint detection**
- §59  Step 7.1 — Per-class A-score tier call (`tier_breakpoints.json`)
- §60  Step 7.2 — Per-cell-type A-score tier call
- §61  Step 7.3 — cfDNA branch (when substrate is plasma) — `cfdna_weight.json`
- §62  Step 7.4 — FLOOR_BREACH detection
- §63  Step 7.5 — Engine-to-customer language mapping
- §64  Step 7.6 — Stage 7 output: per-class tier vector

**Stage 8 — Card-level pattern matching**
- §65  Step 8.1 — Disease signature matrix lookup (v1.5 binary)
- §66  Step 8.2 — Per-card residual map application
- §67  Step 8.3 — Multi-class pattern matching
- §68  Step 8.4 — Card-specific covariate adjustment (smoking inside lung-epic, etc.)
- §69  Step 8.5 — Stage 8 output: card-specific tier calls + confidence

**Stage 9 — Report assembly (Stage 6 reporting layer)**
- §70  Step 9.1 — Customer-facing language collapse
- §71  Step 9.2 — Literature anchors (`literature_anchors.json`) — reporting-layer translator
- §72  Step 9.3 — Cancer prior context (`cancer_prior.json`)
- §73  Step 9.4 — Family history multiplier (`family_history_multiplier.json`)
- §74  Step 9.5 — Sex-specific risk adjustment (engine-inline)
- §75  Step 9.6 — Report rendering pass
- §76  Step 9.7 — What CAN and CANNOT be said to a customer (legal boundary)

**Stage 10 — Delivery**
- §77  Step 10.1 — Report packaging
- §78  Step 10.2 — Delivery channel routing
- §79  Step 10.3 — Audit trail capture (every step's outputs hashed to repo)

### Part III — Chain-integrity scaffolding (runs ABOVE, not inside, the chain)

- §80  L9.0 — The 8-null suite framework (`cpg_null_runner.py`)
- §81  L9.1 — N1 HC label permutation
- §82  L9.2 — N2 Age-strata permutation
- §83  L9.3 — N3 Sex-strata permutation
- §84  L9.4 — N4 Cohort-split replication
- §85  L9.5 — N5 Plate-position null
- §86  L9.6 — N6 Injection-recovery null
- §87  L9.7 — N7 End-to-end synthetic-patient simulation
- §88  L9.8 — N8 Look-elsewhere correction
- §89  L9.9 — Synthetic patient generator (`synthetic_patient_generator.py`)
- §90  L9.10 — VAL sealing protocol (PREREG → OUTCOME → SEALED / RESTATE / RETRACT)
- §91  L9.11 — Null-suite invocation order (when to run; how to declare)

### Part IV — Failure modes & decision trees

- §92  Failure mode catalog by stage (what fails, how to detect, what to do)
- §93  Cross-stage decision tree (when to abort, when to flag, when to proceed with warning)
- §94  Cross-method disagreement protocol (Walther vs NILC at L4)
- §95  Out-of-calibration handling (saturation, low coverage, missing metadata)
- §96  Chain re-run protocol (when an upstream step changes downstream)

### Part V — Reference

- §97  File-to-stage mapping table
- §98  Glossary — CMB ↔ methylome term map
- §99  H_min values frozen 2026-04-06 — the eight Mahaffey Numbers
- §100 Canonical cross-reference index (Recipe, Roadmap, VAL inventory, Lessons Learned)
- §101 CPG Plates 1-4 cross-reference (which plate illustrates which step)
- §102 Change log

---
---

# Part I — Foundations

---

## §1. The chain at a glance

```
                    ┌─────────────────────────────────────────────────┐
                    │  IAMAtlas REBUILD (the calibrated instrument)  │
                    │  481,966 CpGs × 115 cell types × 8 classes     │
                    │  H_min frozen 2026-04-06                       │
                    └─────────────────────────────────────────────────┘
                                          │
                                consulted at every stage
                                          │
                                          ▼
                                          
  IDAT ──► Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4
  files    intake +    calibration  decon-      foreground   A-score
  arrive   QC          + β-compute  volution    subtraction  computation
  on       (L1)        (L2 + L3)    (L4)        (L4 cont.)  (entropy)
  server                                                          │
                                                                  ▼
                                                            
  ◄── Stage 10 ◄── Stage 9 ◄── Stage 8 ◄── Stage 7 ◄── Stage 6 ◄── Stage 5
   delivery        report        card          tier         cellular     Mahalanobis
   to              assembly      matching      breakpoint   age          hyper-volume
   customer        (Stage 6      (disease      detection    inversion    (L6 metric)
                    legal        signature
                    layer)       matrix)
                    
                    
       ╔═══════════════════════════════════════════════════════════╗
       ║  Running ABOVE the chain (not inside it):                ║
       ║  • L9 null suite (cpg_null_runner.py — 8 nulls N1–N8)    ║
       ║  • Synthetic patient generator (FFP10/NPIPE analog)      ║
       ║  • VAL sealing protocol (PREREG → OUTCOME)               ║
       ╚═══════════════════════════════════════════════════════════╝
```

**One-paragraph summary per stage:**

- **Stage 0 — Intake.** IDAT files arrive. We hash them, run probe-level QC, validate sample-level metadata, and decide whether to proceed. Quarantine on suspicion, reject on failure, proceed on clean. *(L1)*

- **Stage 1 — Calibration.** Raw fluorescence becomes β values via dye-bias correction, probe-type normalization, and the standard β = M/(M+U+100) computation. This is the methylome's calibration-to-flux step. *(L2 + L3)*

- **Stage 2 — Deconvolution.** Two independent deconvolvers run in parallel on the same β matrix. **Walther** (NNLS, production answer) and **NILC v2** (departure-from-consensus GLS, cross-method check). Both consult IAMAtlas. Their disagreement at the biological-inference layer is the cross-method gate. *(L4)*

- **Stage 3 — Foreground subtraction.** Age, sex, batch, ancestry, smoking — each is a foreground that contaminates the disease signal the way galactic dust contaminates the CMB. Each foreground has its own module that subtracts its component before downstream scoring. *(L4 continued)*

- **Stage 4 — A-score computation.** For each architectural class and cell type, compute β_mean across the marker CpGs, compute Shannon entropy H(β_mean), divide by the frozen H_min(class). The result is the A-score — a dimensionless ratio measuring departure from the architectural floor. *(Entropy scoring, formally between L3 and L4)*

- **Stage 5 — Multi-D departure.** The patient's 115-cell-type A-score vector is a single point in 115-dimensional space. The Mahalanobis distance from the HC centroid in that space, weighted by the IAMAtlas covariance structure, gives one calibrated headline number for "how far is this patient from healthy." *(L6 — Ledoit-Wolf shrinkage covariance applied as a distance metric. NOT L7 likelihood and NOT L8 parameter inference, which remain empty until Phase E.)*

- **Stage 6 — Cellular age inversion.** For each class, invert the patient's A-score against the 80-cell age reference matrix to find the age at which a typical HC person has that A-score. Eight per-class cellular ages, never collapsed by default. *(Scoring step that consumes Stage 4 A-scores against a healthy-reference baseline. L7 likelihood construction remains empty in V1 — cellular age inversion is not a likelihood evaluation, it is a reference-curve readback.)*

- **Stage 7 — Tier breakpoint detection.** Per-class A-scores get mapped to engine tiers (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH) and customer-facing labels (NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED). cfDNA branch activates when substrate is plasma. *(Thresholding on Stage 4 outputs — neither L7 nor L8. Tier breakpoints are operational rules, not likelihood evaluation or parameter inference.)*

- **Stage 8 — Card-level pattern matching.** Per-class and per-cell-type readouts get matched against the disease signature matrix (v1.5 — 77 rows × 131 columns × 354 populated signature cells). The patient's pattern gets a card-specific tier call. *(Rule-based card evaluation + matrix Mahalanobis-style match — runs Path A and Path B in parallel. L7 proper Bayesian per-card likelihood and L8 MCMC posteriors per card both remain empty until Phase E.)*

- **Stage 9 — Report assembly.** The Stage 6 reporting layer translates physics measurements into customer language. Literature anchors, cancer priors, family history multipliers wrap the biology in clinically meaningful context. This is the layer where we say "in studies, people with your A-score had outcome X" rather than "you have disease Y" — because the latter is medical advice we cannot legally provide. *(Legal boundary layer)*

- **Stage 10 — Delivery.** Report packaging, channel routing, audit trail capture. Every step's output is hashed to the repo so any subsequent question about how a number was produced is traceable. *(L1 closes the loop)*

---

## §2. The L1-L9 chain of custody — full grading table

The chain of custody is the audit framework borrowed directly from CMB cosmology. Every CPG-VAL must trace through these nine links. **The current grades reflect what each link has TODAY (2026-05-31)** — not what it will have at engine completion.

| Link | Name | What CPG has today | What CPG needs to be complete | Current Grade |
|------|------|---------------------|--------------------------------|---------------|
| **L1** | Detector timestreams | IDAT raw intensities + control probes ingested via standard pipeline | Formal IDAT integrity check protocol (cross-array hash, missing-channel detection, contamination flags) — documented | **B+** |
| **L2** | Calibration | Dye-bias correction, probe-type normalization, batch correction (funnorm/BMIQ/ComBat) — done implicitly | Declare formally as named module. Add bisulfite-conversion-efficiency nuisance parameter | **C+** |
| **L3** | Map-making | β = M/(M+U+100) per CpG per sample, standard | Probe-response-function nuisance treatment | **B** |
| **L4** | Component separation | Walther IAM Deconvolver + NILC v2 cross-check + age-axis foreground module (Phase B3) | Sex, batch, ancestry, smoking foreground modules formalized; foreground_registry.py to chain them | **C** (was D before Phase B2.1 + B3) |
| **L5** | Correlation structure | **EMPTY in V1.** Genomic-distance correlation C(d), bispectrum, and cross-substrate cross-correlations are Phase C deliverables (Roadmap TODO 2.1 / 2.2 / 2.3). Stages 4–8 currently operate on independent CpGs / cell types without exploiting correlation structure between them. | C(d) genomic-distance correlation. Bispectrum. Banana-degeneracy mapping. Cross-substrate cross-correlations. | **F** (declared empty — Phase C) |
| **L6** | Covariance modeling | **FILLED via Mahalanobis hyper-volume (Stage 5).** Ledoit-Wolf shrinkage covariance built from the n_hc=601 pooled-healthy cohort, applied as a distance metric: Mahalanobis_d² = (x − μ_HC)ᵀ Σ⁻¹ (x − μ_HC). This IS L6 — the covariance is constructed and consulted. Mahalanobis distance is L6 applied as a metric, not L7 likelihood and not L8 inference. | Sim-based covariance from synthetic patient ensemble (Phase D). Per-CpG covariance. Cross-cohort covariance. Nuisance-parameter covariance. | **B** (functional with Ledoit-Wolf; upgrades to sim-based in Phase D) |
| **L7** | Likelihood construction | **EMPTY in V1.** Cellular age inversion (Stage 6) is a scoring step — a reference-curve readback, not a likelihood function. Card threshold breakpoints (Stage 8 Path A) are rule-based, not log-L evaluation. Matrix Mahalanobis-style match (Stage 8 Path B) is L6 applied as match magnitude, not L7. A proper Bayesian per-card log L(data \| params, Σ) with nuisance-parameter marginalization is a Phase E deliverable. | Proper Bayesian per-card likelihood. Nuisance parameter marginalization. Profile vs marginalized decomposition. | **F** (declared empty — Phase E) |
| **L8** | Parameter inference / posterior | **EMPTY in V1.** A-scores (Stage 4) return point estimates, not posteriors. Mahalanobis distance (Stage 5) returns a scalar metric, not a posterior. The atlas is built via MCMC (per-CpG posteriors over the 8 architecture classes, frozen 2026-04-06), but the orchestrator does NOT run MCMC per-patient — that is a Phase E per-card deliverable. Credible intervals propagate from the atlas posterior SDs forward into Stage 2 fraction CIs (see §31), but per-card posterior inference is not yet built. | MCMC posterior for per-card parameters. Banana-degeneracy mapping. Posterior-predictive checks. | **F** (declared empty — Phase E) |
| **L9** | Null suite + end-to-end sims | Phase A done 2026-05-30: unified 8-null framework + synthetic patient generator + 5 of 7 Family A VALs sealed, 2 restated | Phase A2.1: production-precision N7 recovery; signed-direction injection. Phase A3.1: per-cohort bimodality recomputation for VAL-004 | **A-** |

**Overall grade as of 2026-06-02: C (honestly declared).** Filled links: L1 (B+), L2 (C+), L3 (B), L4 (C with Walther+NILC v2+age-axis foreground), L6 (B with Ledoit-Wolf via Mahalanobis), L9 (A- with sealed-VAL discipline). Empty links: L5 (Phase C), L7 (Phase E), L8 (Phase E). The chain has L4 → L6 → Stage 8 dual matching → report, skipping over the empty L5/L7/L8 links. This is honest — not pretending to do likelihood evaluation or parameter inference when we are not. v1.1 of this SOP overstated L7 and L8 progress by assigning Mahalanobis to L8 and cellular age to L7; v1.2 corrects this per the v2 walkthrough §0 mapping.

**What each link owes the next:**

- **L1 → L2:** A clean IDAT with documented control probe state. If L1's integrity check fails, L2 gets a quarantine flag and refuses to calibrate.
- **L2 → L3:** Calibrated intensities ready for β-computation. If L2's dye-bias or normalization fails, L3 returns NULL for that sample.
- **L3 → L4:** A per-CpG β matrix in (0,1). If any β value is out of range, L4 throws an exception and the sample gets re-routed to Stage 0 quarantine.
- **L4 → L5/L6/L7:** A foreground-cleaned β matrix + per-class fractions + per-class confidence + cross-method gate verdict. If the cross-method gate fires, L5+ analyses pause until reviewed.
- **L5 → L7:** Correlation structure for likelihood construction. *Both empty in V1 — L5 is Phase C, L7 is Phase E. The chain currently leaps directly from L4 component separation to L6 covariance-applied-as-metric (Mahalanobis), skipping correlation-structure exploitation and likelihood-evaluation entirely.*
- **L6 → L7:** Covariance model for likelihood. *L6 is filled (Ledoit-Wolf shrinkage applied as Mahalanobis distance metric); L7 is empty (Phase E). The covariance feeds the Stage 5 metric and the Stage 8 matrix match-magnitude function — neither of which is a likelihood evaluation.*
- **L7 → L8:** Likelihood evaluation per card. *Both empty in V1. V1 ships rule-based tier-threshold cards (Path A) + Mahalanobis-style matrix match (Path B); neither is a likelihood, neither produces a posterior. Phase E builds proper Bayesian per-card L and MCMC over per-card parameters.*
- **L8 → L9:** Posterior estimate for nulls to challenge. L9 runs the 8 nulls against the posterior and either seals the VAL or downgrades to RESTATE / RETRACT.
- **L9 → (closes the loop):** Audit trail back to L1.

---

## §1.5. The three-component architectural separation (walkthrough §6)

The CPG operational system is **three independently updatable components**, even when V1 ships components 1 and 2 together as a single CLI script. The discipline is real; the file count is a deployment choice.

| Component | Role | Independently updatable | V1 status |
|---|---|---|---|
| **(1) Orchestration runtime** — `walther_clinical.py` | Loads the startup artifacts. Runs Stages 0–8 of this SOP. Calls real modules in `walther_clinical_runtime/`. Outputs structured Stage-8 result. | ✓ Restart picks up new card JSONs without code change. Deconvolver / scoring modules swappable. H_min anchors recalibrate without touching orchestrator code. | Built in V1 as a single CLI script. |
| **(2) Doctor report builder** — internal module within V1 | Reads Stage-8 outputs + the lookup JSONs + literature anchors. Assembles doctor-facing Markdown → PDF (Stage 9). Owns the engine-tier → clinical-language translation. | ✓ Report template changes are content updates, not code structure changes. | Built in V1 as an internal module of `walther_clinical.py` with a clear internal boundary, ready to lift into a standalone `walther_report_builder.py` in V2. |
| **(3) Patient-facing destination** — iamperformance.net + a future `walther_patient_report_builder.py` | Customer report; per-class pages; per-cell pages with cell-function descriptions; vigilance content per tier; deep links from card `educational_page_url` + matrix `organ_pages_to_link` fields. | ✓ Website content team updates pages without touching code. Per-cell descriptions are content, not algorithm. | **NOT in V1.** Deferred to V2 after doctor feedback informs the framing. |

**Why the discipline matters for the SOP:** Stages 0–8 produce structured outputs. Stage 9 reads those outputs and assembles a report. Stage 10 delivers. The boundary between components 1 and 2 lives at the Stage 8 → Stage 9 interface — Stage 8 produces `stage_8_outputs` as a single structured object; Stage 9 reads it without re-running any chain step. Operators editing this SOP should preserve that boundary. Stage 9 is permitted to call lookup JSONs (literature anchors, cancer prior, family history, cfDNA weight) but is forbidden from re-invoking deconvolvers, A-score scoring, Mahalanobis, or cellular age scoring — those are component-1 work.

### §1.5.1 Conditional consumption discipline (silent pass-by)

Several lookup matrices are consumed only when their input data exists. **The engine passes by silently — no error, no degraded output, just no enrichment from that layer.**

| Lookup | Consumed when | Otherwise |
|---|---|---|
| `cfdna_weight.json` | `substrate == "plasma_cfdna"` | Silently skipped at Stage 7 (see §61) |
| `family_history_multiplier.json` | `patient_metadata.family_history` present and non-empty | Silently skipped at Stage 9 (§73) — Stage 9 falls back to overall-population prior with the absence noted in audit trail |
| Per-card residual maps | The card's gating criteria met (eligibility passes) | Card returns NOT_FIRED at Stage 8 before residual maps are loaded |

Every conditional-consumption file ships in the production folder ahead of the data that activates it. The orchestrator never errors on a missing optional input — it logs the absence in the audit trail and moves on.



---

## §3. The stages vs. the links — operational sequence × audit discipline

The chain has two axes that get conflated easily. Pinning them down:

**Operational stages (0-10)** = WHAT happens in WHAT ORDER as IDAT moves through the system. An operator follows the stages.

**Chain links (L1-L9)** = WHAT LEVEL of audit discipline each step satisfies. An auditor tests the links.

They are not identical. Stage 2 (deconvolution) and Stage 3 (foreground subtraction) both belong to L4 (component separation). Stage 5 (Mahalanobis) is partly L8 and partly L6 (covariance modeling). The mapping:

| Operational Stage | Chain Link(s) involved |
|-------------------|------------------------|
| Stage 0 — Intake | L1 |
| Stage 1 — Calibration & β | L2 + L3 |
| Stage 2 — Deconvolution | L4 (primary) |
| Stage 3 — Foreground subtraction | L4 (secondary) |
| Stage 4 — A-score computation | between L3 and L4 (entropy scoring on cleaned β) |
| Stage 5 — Mahalanobis | L6 (covariance) + L8 (parameter inference, partial) |
| Stage 6 — Cellular age | L7 (likelihood, partial) |
| Stage 7 — Tier breakpoints | L8 (partial) |
| Stage 8 — Card matching | L7 + L8 |
| Stage 9 — Report assembly | Legal boundary layer (above L9) |
| Stage 10 — Delivery | L1 closes (audit hash) |
| **Above all stages:** L9 null suite | L9 (cross-cutting) |

A reader executing Part II step-by-step is following the operational stages. A Planck postdoc auditing the chain is checking the L-links. Both views must align — that alignment is the discipline.

---

## §4. IAMAtlas — what it is, where it lives, when it's consulted

**What IAMAtlas IS:**

The Informational Actualization Model Atlas — IAMAtlas — is the calibrated reference for everything CPG measures. It is the methylome's analog of Planck's beam profile and frequency response calibration: every measurement traces back to it, every interpretation is anchored against it.

The current production version is **IAMAtlas REBUILD** (frozen 2026-04-06), comprising:

- **481,966 CpGs** with posterior β distributions per architectural class
- **8 architectural classes** (stem_pluri, stem_adult, progenitor, cycling, secretory, immune, terminal, stromal)
- **115 cell types** mapped to those 8 classes
- **8 frozen H_min values** (the Mahaffey Numbers — see §5)
- **MCMC chains** to convergence for all 8 classes (Rh̄ < 1.001)

**Where IAMAtlas LIVES:**

The full atlas posterior is at `IAMAtlasREBUILD.csv` (605 MB, 483,093 CpGs × 8 classes × {mean, sd, ci_lo, ci_hi}). The full file is NOT in the public repository — it's the proprietary core. What IS in the repo:

- `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` — per-cell-type marker CpGs derived from the atlas (115 × 100 markers)
- `IAMAtlasREBUILD_provenance.json` — the H_min values and convergence diagnostics
- `Biological_Physics/atlas_vault/pipeline_runtime_matrices/mahalanobis_healthy_reference_v0_1.json` — HC centroid + covariance in 115-cell-type A-score space
- `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.{json,csv,py}` — 80-cell age × class baseline

The atlas is the SOURCE for all runtime artifacts but is itself proprietary IP. The Recipe (§9 of `v1_CPG_Recipe.md`) is the vault. The runtime artifacts are the public surfaces derived from the vault.

**When IAMAtlas is CONSULTED (and how):**

The atlas is consulted at every stage that requires a reference comparison or a calibrated constant:

| Stage | What consults the atlas | What it pulls |
|-------|--------------------------|---------------|
| Stage 2 (deconvolution) | Walther + NILC | Per-class marker CpG posteriors → reference matrix for NNLS / GLS |
| Stage 3 (age foreground) | `age_axis_foreground.py` | Per-CpG age-drift slopes (`IAMAtlas_age_layer.csv`) |
| Stage 4 (A-score) | `iamatlas_a_scoring.py` | Per-class marker CpGs + H_min per class |
| Stage 4 (per-cell-type A-score) | `iamatlas_a_scoring.py` | Per-cell-type marker CpGs from `iamatlas_celltype_markers_v0_2.json` |
| Stage 5 (Mahalanobis) | `iamatlas_mahalanobis_scoring.py` | HC centroid + covariance from `mahalanobis_healthy_reference_v0_1.json` |
| Stage 6 (cellular age) | `iam_cellular_age_scoring.py` | 80-cell age × class baseline curve |

**Every step section in Part II carries an explicit "Atlas reference" line.** If a step does NOT consult the atlas, that line says so. If it does, the line names which slice of the atlas, in which file, at which path.

---

## §5. The Mahaffey Number (H_min) — the eight frozen anchors

The H_min values are the architectural floor entropies — the minimum Shannon entropy each architectural class must maintain to remain that class. They are derived from the IAMAtlas REBUILD MCMC posteriors via the Jacobson → virial → Landauer derivation chain. They are **frozen 2026-04-06** and are the only "constants" in the entire framework. Everything else is computed.

| Class | H_min | Source |
|-------|-------|--------|
| terminal | 0.7728 | IAMAtlas REBUILD MCMC posterior (2026-04-06) |
| immune | 0.838889 | IAMAtlas REBUILD MCMC posterior |
| secretory | 0.843264 | IAMAtlas REBUILD MCMC posterior |
| cycling | 0.856055 | IAMAtlas REBUILD MCMC posterior |
| progenitor | 0.852216 | IAMAtlas REBUILD MCMC posterior |
| stromal | 0.86295 | IAMAtlas REBUILD MCMC posterior |
| stem_adult | 0.873718 | IAMAtlas REBUILD MCMC posterior |
| stem_pluri | 0.982166 | IAMAtlas REBUILD MCMC posterior |

**Where they live in the codebase:**

`IAMAtlasREBUILD_provenance.json` under the key `h_min_values_frozen_2026_04_06`. ONE source of truth. Any runtime artifact that needs an H_min reads it from here. If the value ever needs to change (e.g., new MCMC convergence with denser cohort), the change happens in `IAMAtlasREBUILD_provenance.json` first and all downstream artifacts are re-derived from it.

**Why "Mahaffey Number" matters:**

This is the dimensionless physics constant the framework rests on. Like Boltzmann's constant or Avogadro's number, it sets the scale at which a physical phenomenon (cellular architecture) is measured. Unlike Boltzmann or Avogadro, it is empirically derived from biological data (the MCMC posteriors) anchored to the IAM physics chain. **It is not a population statistic.** No patient cohort entered the H_min calibration. It is the architectural floor — the minimum entropy a cell of that class must carry by virtue of being that class.

When a patient's A-score = H(β_mean) / H_min(class) exceeds 1, the cell has paid more than the floor — it has departed from the architectural minimum. When it falls below 1, the cell is operating below the architectural floor, which is itself a measurement (suppressed below baseline, the opposite direction of departure).

---

## §6. The CMB → methylome translation principle (why the math is the same)

Heath has stated this repeatedly across sessions, and it deserves the canonical statement:

> "Math doesn't care if it's a CMB or a methylome."

The principle: the mathematical operations used to extract signal from the cosmic microwave background — Bayesian inference, multi-frequency component separation, MCMC posteriors, null tests, joint posterior ellipsoids, banana degeneracies, power spectrum decomposition, bispectrum estimation — were not invented for cosmology. They are general statistical-physics tools. Cosmology happens to be the field that has driven those tools to their highest precision because the CMB is a single all-sky map of an inflationary remnant with rich angular structure and well-characterized noise.

The methylome is, in a deep structural sense, the same kind of object:

- A signal field defined over a fixed substrate (the genome instead of the sky)
- Sampled by a measurement apparatus (methylation arrays instead of bolometers)
- Carrying both cosmological-equivalent signal (the cell architecture, the IAM physics) and foreground contamination (cell composition, age, sex, batch — like galactic dust)
- Reducible to spherical-pixelization (HEALPix at NSIDE=128 across the genomic axis — see Plates 1-4)
- Subject to the same statistical disciplines (cross-method consensus, null testing, banana degeneracy mapping)

**The Cosmic Microwave Methylome (CMM) plates** (CPG_Plate_01 through CPG_Plate_04) make this visually irrefutable. They render IAMAtlas REBUILD posteriors using Planck visualization conventions (Mollweide projection, Planck colormap, HEALPix NSIDE=128). The methylome literally LOOKS like a CMB sky map — same angular structure, same large-scale anisotropy with embedded small-scale fluctuation. The visible difference (methylome bimodal, CMB Gaussian) is itself the biological signature.

Every step section in Part II carries an explicit "CMB equivalent" subsection. The point is not metaphor. The point is that the SAME ALGORITHM, applied to the methylome, does the same thing it does on the CMB — because the math doesn't care which substrate it's running on.

---

## §7. Bidirectional calibration — solved by entropy-space scoring

This is one of the deepest structural physics wins in the framework, and it needs to be canonical.

**The old problem (β-space panels like Xu-538):**

In disease, some CpGs go UP and others go DOWN. If you average β across a panel, hypermethylation signal at some CpGs and hypomethylation signal at others cancel out. To avoid cancellation, traditional methylation panels required:

- Per-CpG directionality vectors (you had to know in advance which way each CpG would move in disease)
- Separate panels for "hypermethylation markers" vs "hypomethylation markers"
- OR multiple atlas references each calibrated for one direction

This was the multi-atlas calibration nightmare. Get one direction wrong on one CpG in one panel and the signal collapses to noise.

**The IAM solution — entropy space:**

CPG works in entropy space, not β space. The A-score formula is:

> **A_class = H(β_mean_class) / H_min(class)**

The Shannon entropy function `H(β) = −β·log₂(β) − (1−β)·log₂(1−β)` is **symmetric around β = 0.5** and **concave with maximum at β = 0.5**. So:

- β = 0.3 and β = 0.7 produce the **same entropy** (H ≈ 0.881)
- β = 0.2 and β = 0.8 produce the **same entropy** (H ≈ 0.722)
- β = 0.5 produces **maximum entropy** (H = 1.0)

Because of this symmetry, when disease drives β values **away from the class equilibrium toward 0.5** — regardless of which direction in β space — entropy goes up monotonically. When disease drives β values **toward the extremes (0 or 1)**, entropy goes down. The A-score captures **departure from H_min in entropy space**, which is direction-agnostic in β space by construction.

**Why IAMAtlas makes this work without external calibration:**

Each architectural class has its own H_min anchor. The framework knows:

- "terminal class operates at H_min = 0.7728"
- "immune class operates at H_min = 0.838889"
- ... etc.

When a patient's class-level β_mean produces H(β_mean) > H_min(class), the patient's cellular architecture has DEPARTED from the floor — in whichever β-space direction. Hypermethylation and hypomethylation BOTH register as departure-from-floor in entropy space, contributing to A in the same direction.

**What this kills:**

- ❌ Per-CpG directionality vectors needed
- ❌ Separate "hyper" vs "hypo" panels needed
- ❌ Multi-atlas reconciliation for opposing directions needed
- ❌ Cancellation when pooling across CpGs

**What replaces it:**

- ✓ Single class-level H_min anchor per architectural class (8 numbers, MCMC-derived, frozen 2026-04-06)
- ✓ Entropy-space A-score that is direction-agnostic by construction
- ✓ One atlas (IAMAtlas REBUILD) carrying all per-class and per-cell-type information
- ✓ Bidirectional changes ADD signal in entropy space rather than cancel

**The plain-language form (from The Cellular Margin):**

> EDEAR measures how much margin a cell has between the order it is maintaining and the minimum cost of maintaining any order at all. That margin is the reading. A healthy human cell runs at about 21 times that floor. The framework reads each architectural class against its own floor — same physics, eight different thresholds — and reports the departure regardless of which direction in β space the disease drives.

---

## §8. Vocabulary — what to say and what NOT to say

Borrowed verbatim from v2 Capability Translator + v4 Roadmap §11. The shift is **not cosmetic.** It affects how every step section is described, how outcomes are reported, how any future paper would be referenced.

| Don't say | Say instead | Why |
|------------|-------------|-----|
| pipeline | chain of custody | A pipeline is a one-way conveyor. A chain of custody is a link-by-link audit trail where every link declares what it owes the next and what it accepts from the previous. CMB cosmology runs on chains of custody, not pipelines. |
| classifier / threshold | translator / readout | CPG does not learn case-vs-HC discrimination on a training set. It measures β, computes H(β)/H_min against a physics-derived constant, and reads what the cell wrote. No classifier is trained anywhere in production. |
| validate / validation | confirm / confirmation | "Validate" implies pass/fail against a ground truth. "Confirm" implies the framework predicted a value and the data are consistent with it. The latter is what's actually happening. |
| discriminate / separation | departure from floor | CPG does not separate cases from controls by drawing a hyperplane. It measures how far each patient sits from H_min for each architectural class. |
| pipeline rehearsal | chain rehearsal | A rehearsal of the chain of custody, not of a pipeline. |
| test for [disease] | measure [physiology] consistent with [disease pattern] | CPG measures; the card interprets the departure pattern. We do not diagnose. |

A card that says "the immune A-score classifies pre-dx breast cancer at AUC 0.74" sounds like CPG learned to separate two groups on training data. A card that says "the immune A-score departs from H_min by 1.78 SD ten years before clinical breast cancer detection, consistent with the framework's prediction of immunosurveillance failure preceding disease" tells the actual story.

The Euclid framing rule from the project memory applies here: **feathers, not verdicts.** No decisive tests. No load-bearing single results. No "this confirms" / "this validates" / "this proves" language. The data are consistent with predictions within the framework. That's the discipline.

---

## §9. The two-deconvolver discipline — when each runs, why both, what disagreement means

**Both deconvolvers run on every patient. They run in parallel, not in sequence.**

- **Walther IAM Deconvolver** (NNLS) is the production answer. Its per-class fractions feed Stage 3, Stage 4, Stage 8 directly.
- **NILC v2** (departure-from-consensus GLS) is the cross-method check. Its per-class fractions are compared to Walther's at the L4 cross-method gate (Step 2.6).

This is direct Planck discipline: Planck's CMB component separation runs **Commander + NILC + SMICA + SEVEM** in parallel on the same input. The four methods are not redundant — they are independent algorithms that should agree at the inference layer (cosmological parameters) even when they disagree at the substrate layer (per-pixel temperature).

**What disagreement at the substrate layer means:**

Walther and NILC have published agreement of ~12 percentage points in fraction-level disagreement on EPIC-Italy (median L1 = 0.23 on per-class fractions; Phase B2.1 finding). That's expected — NNLS and GLS have different inductive biases for borderline mass. NNLS pushes ambiguous mass to the single best-fitting class. GLS distributes ambiguous mass across all classes with non-zero atlas posteriors.

**Substrate-level disagreement is documented as a Planck-style systematic and does NOT block the chain.**

**What disagreement at the inference layer means:**

The Phase B2.1 finding: across the disease-relevant biological inference (case-vs-HC immune class direction), Walther and NILC AGREE on 4 of 5 non-zero effects, including agreement on the disease-relevant immune signal direction. **This is partial cross-method confirmation at the layer that matters.**

If the two methods ever DISAGREE on the disease-relevant direction (e.g., Walther says immune A-score is elevated in cases, NILC says it's suppressed), the cross-method gate FIRES. The patient's record is flagged. The card's tier call pauses. Manual review is required before the report ships.

**When each runs:**

| Deconvolver | When | What it consumes | What it outputs |
|-------------|------|------------------|------------------|
| Walther | Every patient, Stage 2 step 2.3 | β matrix at IAMAtlas marker CpGs | 8 per-class fractions (NNLS-constrained) + per-class confidence + status codes |
| NILC v2 | Every patient, Stage 2 step 2.5 | Same β matrix at same CpGs | 8 per-class fractions (departure-from-consensus GLS, simplex-projected) + per-class residuals |
| Cross-method gate | Every patient, Stage 2 step 2.6 | Walther fractions + NILC fractions | PASS / FLAG / FAIL verdict |

**Why both:**

The framework is not "use Walther because Walther is right." The framework is "use Walther's answer, but require NILC's answer to corroborate it at the inference layer." Two methods agreeing at the inference layer is stronger evidence than one method's answer alone. Two methods disagreeing at the inference layer is a flag that something is wrong upstream — a sample QC issue, a calibration issue, a real biological anomaly worth a closer look. **The disagreement is a signal, not a failure.**

---

## §10. Stage 6 reporting layer — the legal boundary between physics measurement and customer communication

CPG measures physics. The customer report communicates that measurement to a non-clinician. Between those two activities is a legal boundary: **CPG cannot tell a customer they have a disease, or will get a disease, or should take a medical action.** That is medical advice, which requires a license CPG does not have.

What CPG CAN say to a customer at Stage 6:

- "Your immune class A-score is 1.07."
- "This places you at the [P_age] percentile of the healthy reference population at your age."
- "In published studies (citation), patients with A-scores in this range had outcome X." [literature anchor]
- "US lifetime incidence of [class-relevant condition] is [N]%; family history of [condition] in a first-degree relative raises that to [N × multiplier]%." [cancer_prior + family_history_multiplier]
- "Consider discussing this reading with your physician."

What CPG CANNOT say:

- "You have [disease]."
- "You will get [disease]."
- "You should take [medication]."
- "You should not [activity]."

This boundary is **enforced at Stage 9 (report assembly).** Every word that goes into the rendered report is checked against the Stage 6 language collapse (`tier_breakpoints.json` engine → customer mapping) and the literature anchors. No engine-internal language ("FLOOR_BREACH", "URGENT") appears in the customer report. The customer sees "SIGNIFICANTLY_ELEVATED" with literature context.

**The literature_anchors.json, cancer_prior.json, and family_history_multiplier.json artifacts are NOT measurement.** They are reporting-layer translators that wrap the IAM physics measurement in language a non-clinician can act on. The measurement is the A-score and the cellular age. The report is the legally-permissible translation of that measurement into clinical context.

When a card author writes a report template, the rule is: **the physics is the truth, the literature is the language.** The physics measurement is fixed once Stage 4 completes. The literature anchoring is fixed once the card's source citations are vetted. The report assembly is the composition of those two — never a re-interpretation.

This is the layer where the entire chain of custody ends and the customer's understanding begins. Everything upstream is physics. Everything downstream is communication. The boundary must hold.

---

## End of Part I

Parts II-V follow in the next document delivery (sections §11-§102). Reading order: foundations → step-by-step → audit machinery → failure modes → reference. Random-access by section number is fully supported — every step section is self-contained.

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*


---

## §10.5. The 13 non-negotiable rules (walkthrough §8)

The walkthrough names 13 disciplines the CPG operational system must obey without exception. Each is summarized here for cross-reference; the SOP body enforces each at the stage that owns it.

1. **Wellness-first positioning.** Cellular health and cellular age are the lead. Disease detection is secondary. (V1 doctor report applies this — wellness panel first, disease findings labeled secondary.)
2. **Single IAMAtlas at runtime — only IAMAtlas.** No external atlases queried at runtime, ever. No Moss-NNLS, no Loyfer-NNLS, no EpiDISH-via-rpy2, no Salas-QC calls at runtime. Source atlases were ingested at IAMAtlas BUILD time; the orchestrator queries only IAMAtlas REBUILD.
3. **No customer-facing physics terminology in commercial code.** Boltzmann, Landauer, Arrhenius, Bose-Einstein, decoherence, k_B, ln2, coth, "thermal", "activation energy", "Mahaffey Number" — none of these in `walther_clinical.py` source code, docstrings, comments, API outputs, or HTML. Internal variable names neutral. This protects the Recipe.
4. **Recipe stays in the vault forever.** Never disclosed under any NDA. Full acquisition is the only scenario where the Recipe transfers. The orchestrator consumes only operational artifacts (atlas, H_min anchors, deconvolver source, cards, schema), never Recipe content.
5. **Screening-language rule.** EDEAR never specifies screening tests, ages, intervals, or follow-up workups. Vigilance language defers to "the screening recommendations your clinician has discussed with you." (V1 doctor report has slightly more flexibility — but even there, no specific test names are prescribed.)
6. **Per-class A-score aggregation discipline.** Each class has its own H_min anchor. Customer report shows class-level + cell-level scores separately (never silently combined).
7. **Class assignments non-negotiable.** Megakaryocyte → progenitor (NOT immune). Cortical neurons → terminal (NOT stromal). The orchestrator consumes `IAMAtlasREBUILD_celltype_to_class.json` as the single source of truth; no per-cell-type re-classification anywhere.
8. **Empty cells in the disease matrix mean "no documented signature" — NOT zero.** The Stage 8 match function treats blank cells as missing data, not as zero signal. Honest research-in-progress state.
9. **Run-everything doctrine.** Every IDAT runs Stages 0–10 with the FULL deconvolution and the FULL per-class + per-cell-type A-score computation, regardless of any single stage's result. No gating on "first signal that crosses threshold" — compute every measurement, let the anomaly stack tell the story.
10. **Disease severity class describes the row's phase, NOT the customer match.** The matrix's `disease_severity_class` column is a property of the (disease, phase, substrate) tuple — not a property of the customer. The customer's tier is computed at runtime from match × phase × severity × evidence.
11. **No retroactive flags on prior VAL readings when interpretation re-frames are issued.** Re-frames apply only to forward-looking card preregs. Prior VAL findings remain valid under the documented historical interpretation.
12. **Bidirectional grouping rule.** Cells that move in opposite directions across diseases get separated in output structure (Stage 5 surfaces this so Stage 9 and the website present it correctly).
13. **Follow the biology, not the class** (website discipline). The orchestrator returns per-cell-type output in the natural biological grouping the website expects — by cell-type group for immune/progenitor/terminal/stromal; by organ for cycling/secretory.

These rules are stable; they do not change between SOP versions. They live here in Part I because the rest of the document refers to them by number.

