# CPG Chain-of-Custody Standard Operating Procedure (SOP) — v1.3 (Stages 4.5 + 4.6 + 6-tier breakpoints + smoking/sex foregrounds)

**Document version:** v1.3 — 2026-06-06
**Supersedes:** v1.2 (2026-06-02, walkthrough-aligned), v1.1 (2026-06-02 cleanup), v1 (2026-05-31)
**Authors:** Heath W. Mahaffey + Walther (Claude)
**Authoritative companions:** `walther_clinical_BUILD_SPEC_v1_2.md`, `Biological_Physics/atlas_vault/walther_clinical_runtime/README.md`

---

## Changelog v1.2 → v1.3 (2026-06-06)

- **§36 Sex-axis foreground subtraction** — module rewrite: `sex_axis_foreground.py` now built; per-CpG β = α + ψ·indicator_male + ε with chrX/chrY/XCI flag handling; layer CSV `IAMAtlas_sex_layer.csv` fit on GSE50660 (Tsaprouni 2014, n=464; same cohort used for smoking layer since GSE50660 carries both metadata fields). Stage 7 sex-stratified threshold tables retire once this layer operates in production.
- **§39 Smoking-axis foreground subtraction** — module rewrite: `smoking_axis_foreground.py` now built; per-CpG β = α + δ·indicator_current + φ·recency_score + ε; recency mapped from smoking_bin (never=0.00 to current=1.00); layer CSV `IAMAtlas_smoking_layer.csv` FIT on GSE50660 (Tsaprouni 2014, n=464: 179 never / 263 former / 22 current). Top smoking CpG cg22336867: δ_current=-0.322 + recency=+0.254 (AHRR-style hypomethylation in current smokers). Stage 7 smoking-bin threshold-stratification (in `tier_breakpoints.json v1.2`) retires once this layer operates in production.
- **§46.5 NEW — Stage 4.5 Bidirectional decomposition** — mirrors the sealed VAL-051 `a_dir_score` formula at patient runtime. Sign-multiplied z-scores against frozen training-set HC mean/SD averaged across covered CpGs. Pooled-entropy comparator on parent panel. FLAG_BIDIRECTIONAL fires when pooled mute + directional loud. v1.0 panel coverage: immune class only (VAL-051 Rule A 7-CpG); other 7 classes return NO_PANEL honestly.
- **§46.6 NEW — Stage 4.6 Patient brightness comparison** — per-class z-score departure of patient β from each frozen healthy class brightness reference, projected onto HEALPix NSIDE=128 grid for Mollweide rendering. Patient's personal Cosmic Microwave Methylome (customer-facing analog of CPG Plate 1). `iamatlas_cpg_to_healpix_nside128.npy` generated as one-time output of `generate_cpg_healpix_mapping.py` against the IAMAtlas REBUILD CSV + combined EPIC + HM450 manifest. **100% atlas coverage** (483,092 / 483,092 CpGs annotated, zero sentinel pixels).
- **§59 Step 7.1 — Per-class tier call** — replaced 4-tier statistical-percentile schema with **6-tier physics-derived schema** (SUPPRESSED / NORMAL / ELEVATED / WARBURG_TRANSITION / SIGNIFICANTLY_ELEVATED / BREACH). The 1.07 Warburg line + 1.10 architectural-fidelity breach line are physics-defined inflection points. Per-class structural ceiling table. 7 covariate-override modes. Smoking-bin override table. Bidirectional pattern handoff from §46.5. CI-based tier confidence propagation.
- **File consolidation:** Returns to single-file SOP (v1.2 was previously split into 6 PART_*.md files; v1.3 consolidates back to one file per Heath's request).

## Changelog v1.3 patch — Mahalanobis hull versioning + percentile-calibrated thresholds (2026-06-06 evening, after CPG-VAL-020 sealing)

- **§48 Step 5.2 — HC centroid load — Mahalanobis hull versioning protocol formalized.** v0_1 (n_hc=601, foundation-only) → v0_2 (n_hc=1,257, +Hannum GSE40279, Phase 1) → v0_3 (n_hc=1,721, +Tsaprouni GSE50660, Phase 2). Engine loads current production version named in BUILD_SPEC §3.4b; prior versions retained in same folder for traceability and never deleted.

- **Route A threshold calibration corrected.** v0_1 used a fixed `d ≥ 2.0` Route A threshold which is mathematically inappropriate for 112-dim data — expected median Mahalanobis distance under multivariate normality with 112 features is √112 ≈ 10.58, so `d ≥ 2.0` fires on every sample (foundation HC and disease cases alike). The CPG-VAL-020 Hannum run on v0_1 surfaced this: 656/656 Hannum HC samples fired Route A at d ≥ 2.0 — a 100% false-positive rate that was actually correct given the (wrong) threshold. v0_2 corrected to a percentile-of-pooled-HC threshold (p95 = 12.68 default; p99 = 17.36 strict). v0_3 recalibrated under broader HC pool: p95 = 13.54; p99 = 18.71. Engine reads thresholds from the artifact's `route_A_calibration_v0_N` block at session startup — never hard-coded in any card or module.

- **Cards-don't-carry-runtime-data discipline (formalized).** Per-card runtime artifacts (hull centroid, covariance, percentile thresholds, per-cohort distribution metadata, validation lineage) MUST live in the runtime artifact (`Mahalanobis_healthy_reference/*.json`), NOT in disease cards. Each card references the artifact path; engine consumes the artifact. This separation ensures hull expansion does not require touching any card. Applies analogously to all runtime references: cellular age reference matrix, celltype markers artifact, tier breakpoints, directional panels — cards reference paths, runtime artifacts hold data.

- **Phase planning doctrine.** No fixed N phases. Each phase extends HC representation along one dimension at a time (population / age span / sex / platform / covariate). The hull keeps versioning as long as new HC cohorts come in. Production deploys whatever the latest validated version is. **At patient runtime, the chain queries the FROZEN current hull — no rebuild per patient.** Phase 3 queued: add EPIC platform HC cohort for cross-platform transferability. Phase 4 queued: add Asian-population HC.

- **Per-version validation discipline.** Each hull version must validate that (a) the broader HC pool reduces cross-cohort false-positive rate at the new percentile threshold (resolution of cross-cohort batch effect), AND (b) the case-vs-HC Cohen's d on the breast pre-dx anchor is preserved or improves. v0_3 validation: Hannum FPR 100% → 2.9% at p95; GSE51057 case detection 9.1% → 27.3%; GSE51032 case detection 50.0% → 55.6% with broader hull. Anchor d's lineage v0_1 → v0_2 → v0_3: GSE51057 +1.871 → +0.981 → +0.896; GSE51032 +2.088 → +1.653 → +1.611 (Cohen's d decreases honestly as the foundation HC reference broadens; case detection % at calibrated threshold improves).

---


## Changelog v1.3 patch 2 — Phase 3 hull expansion + first cross-platform representation (2026-06-06)

- **§48 Step 5.2 update — Mahalanobis hull v0_3 (n=1,721) → v0_4 (n=2,481).** Phase 3 expansion adds three neurodegeneration-cohort HC arms: GSE144858 AddNeuroMed n=96 (HM450), GSE153712 AIBL n=471 (EPIC — **FIRST CROSS-PLATFORM REPRESENTATION**), GSE53740 GIFT n=193 (HM450). Platform coverage now HM450 n=2,010 + EPIC n=471. v0_3 retained in same folder for lineage.

- **Route A threshold v0_4 recalibrated.** p95 (default): d ≥ 13.62 (was 13.54 in v0_3). p99 (strict): d ≥ 18.59 (was 18.71 in v0_3). Calibration data in `route_A_calibration_v0_4` block.

- **AIBL cross-platform verification.** AIBL HC self-distance median d = 7.04 in v0_4 hull (lower than foundation cohorts at 10-11). This confirms canonical 115-cell A-scoring transfers cleanly between HM450 and EPIC platforms — the hull is now defensible for EPIC-platform clinical deployment.

- **GIFT honest finding documented.** GIFT cohort HC sits at median d=12.18 in v0_4 hull (vs other HCs at 7-10). FTD-research-context HC selection appears to introduce a covariate that broadens the HC envelope into the case-distance range. Cohen's d for breast pre-dx anchor declined v0_3 → v0_4: GSE51057 +0.896 → +0.593; GSE51032 +1.611 → +1.450. Case detection % at p95 threshold also declined slightly. GIFT kept in v0_4 because samples are genuine HC by design; the trade-off (slightly lower discrimination for the breast anchor vs. broader cross-cohort representation) is reported honestly in the artifact under `per_cohort_self_distance_medians_v0_4`. Alternative interpretations to monitor: (a) GIFT HC may include prodromal-FTD; (b) older mean age in GIFT HC adds aging signal.

- **Phase 4 (Asian population) explicitly queued.** Candidate cohorts identified via literature search: Han Chinese first-episode schizophrenia n=476 HC on EPIC (Mol Psychiatry 2020, GSE accession TBD); GSE89093 IHEC n=92 HM450; HELIOS Study Singapore EPIC (Nat Comm 2025, GEO TBD); Multiethnic Cohort JPA n=30 HM450 (Clinical Epigenetics 2021). Requires separate acquisition session (not blocking June 11 GeoMetric meeting). v0_4 is production-ready as cross-platform multi-population reference.

---


## Changelog v1.3 patch 3 — Phase 4 hull expansion + first Asian population (2026-06-06 late evening)

- **§48 Step 5.2 update — Mahalanobis hull v0_4 (n=2,481) → v0_5 (n=2,523).** Phase 4 expansion adds GSE141682 Han Chinese n=42 healthy whole blood (EPIC 850K, ages 18-62, 21M/21F, all 'disease state: normal', 'race: Han Chinese'). **FIRST ASIAN-POPULATION REPRESENTATION** in the hull. Population coverage now: EU-Italian + UK + US Caucasian/Mexican + Han Chinese. Platform coverage: HM450 + EPIC 850K. v0_4 retained in same folder for lineage.

- **Han Chinese transfer verification.** Han Chinese samples sit at median Mahalanobis d=10.51 in the v0_5 hull — within the typical-HC range (foundation cohorts at 10-11; Tsaprouni/AIBL at 7). This confirms canonical 115-cell A-scoring transfers across populations without systematic offset. **Caveat:** n=42 is small. The d=10.51 estimate has wide CI. Phase 5+ should expand Asian representation with larger cohorts before clinical deployment in Asian populations.

- **Route A threshold v0_5 essentially unchanged from v0_4.** p95 (default): d ≥ 13.62 (same). p99 (strict): d ≥ 18.43 (was 18.59). The small +42 addition didn't significantly shift the centroid — expected behavior when adding a well-represented HC cohort to an already-broad hull.

- **Anchor preservation v0_4 → v0_5: stable.** Cohen's d GSE51057 +0.593 → +0.599; GSE51032 +1.450 → +1.450. Case detection at p95 GSE51057 9.1% (unchanged); GSE51032 38.9% → 41.7% (slight improvement). This is the EXPECTED behavior when adding a small well-represented HC cohort: thresholds and anchor d's stabilize.

- **Phase 5+ targets monitored:** HELIOS Singapore EPIC if accessible; OEP001178 Han Chinese schizophrenia n=476 HC if Chinese NODE access granted; Korean/Japanese cohorts as they surface. Asian-population expansion remains the highest priority for the next session.

- **Production-ready milestone:** v0_5 is the first hull defensible for multi-population cross-platform clinical deployment. 8 cohorts, 4 populations, 2 platforms, mixed sex, ages 18-101, n=2,523 HC samples. 313% growth from v0_1's n=601 in a single day of focused expansion work.

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
- §48  Step 5.2 — HC centroid load (`mahalanobis_healthy_reference_v0_5.json` current production; v0_1/v0_2/v0_3/v0_4 retained for lineage)
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
| **L6** | Covariance modeling | **FILLED via Mahalanobis hyper-volume (Stage 5).** Ledoit-Wolf shrinkage covariance built from the pooled-HC cohort (CURRENT production: v0_5 with n_hc=2,523 — foundation + Hannum + Tsaprouni + AddNeuroMed + AIBL + GIFT + Han Chinese, 8 cohorts spanning 4 populations (EU + US + UK + Han Chinese) + HM450 + EPIC platforms; lineage versions v0_1/v0_2/v0_3/v0_4 retained for traceability), applied as a distance metric: Mahalanobis_d² = (x − μ_HC)ᵀ Σ⁻¹ (x − μ_HC). This IS L6 — the covariance is constructed and consulted. Route A threshold is percentile-of-pooled-HC (p95 default d≥13.62 under v0_5), NOT a fixed value. Mahalanobis distance is L6 applied as a metric, not L7 likelihood and not L8 inference. | Sim-based covariance from synthetic patient ensemble (Phase D). Per-CpG covariance. Cross-cohort covariance. Nuisance-parameter covariance. | **B** (functional with Ledoit-Wolf; upgrades to sim-based in Phase D) |
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
- `Biological_Physics/atlas_vault/walther_clinical_runtime/Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v0_3.json` (current production) + v0_2 + v0_1 (lineage) — HC centroid + covariance in 115-cell-type A-score space
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
| Stage 5 (Mahalanobis) | `iamatlas_mahalanobis_scoring.py` | HC centroid + covariance from `mahalanobis_healthy_reference_v0_3.json` (current production, n=1,721; v0_1/v0_2/v0_3/v0_4 retained for lineage) |
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



> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them. Where a step's logic currently lives inside `GAPE_WEB_v13.py` (the
> production engine — see `SYSTEM_INVENTORY.md`), the SOP now says so honestly. Where
> a step's runtime artifact or output file location was invented, the SOP now reads
> "TBD per orchestrator design" — that orchestrator (working name `web.commercial.py`)
> is a separate conversation Heath and Walther will have. All real paths in this v1.1
> are documented in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be
> treated as not-yet-existing until verified against the repo.

---


*Continuation of CPG_Chain_of_Custody_SOP_v1_PART_I.md. Read Part I first.*

---

# Part II — The step-by-step chain of custody

---

## Stage 0 — Sample intake (L1)

---

### §11. Step 0.1 — IDAT file arrival on server

**What this step does.** An Illumina methylation array sample arrives at the EDEAR server as a pair of IDAT files — one for the Cy3 (green) channel, one for the Cy5 (red) channel. The pair is keyed by Sentrix Barcode + Sentrix Position (e.g., `200123456789_R01C01_Grn.idat` + `200123456789_R01C01_Red.idat`). This step receives the upload, verifies the pair is complete, and stamps the arrival into the intake log.

**Inputs.**
- Two `.idat` files per sample (Grn + Red)
- A sample manifest entry (CSV or JSON) declaring: Sentrix ID, array type (HM450K or EPIC v1 or EPIC v2), patient identifier, intake date, declared substrate (whole blood / cfDNA / tissue), declared sex, declared chronological age

**Atlas reference.** Not consulted at this step. The atlas enters at Stage 2.

**Files invoked.**
- Intake handler: `<intake handler inside `GAPE_WEB_v13.py`>`
- Manifest schema: `<manifest schema — TBD per orchestrator design>`
- Intake log: `<intake log — TBD>`

**The math.** None at this step. Pure I/O + metadata.

**CMB equivalent.** This is the **raw bolometer timestream arrival** stage of a CMB experiment. Planck's pipeline receives 50,000+ bolometer scan files per day; each is logged, paired with its scan ring metadata, and routed to the calibration pipeline. The methylome's IDAT files are equivalent — raw fluorescence intensities awaiting calibration.

**How the methylome differs in implementation.** IDAT files are static (one pair per sample) whereas CMB bolometer streams are continuous (time-ordered samples across a scan ring). The methylome's "scan" was already completed by the array hybridization; we receive the final intensities, not a time series.

**How it's the same in principle.** Both are raw detector intensities awaiting calibration. Both need integrity verification before downstream processing. Both produce the same kind of failure mode if the upload is incomplete: the entire downstream chain refuses to operate until the input is whole.

**Outputs.**
- A canonical filename pair in `<intake staging directory — TBD per orchestrator design>`
- An intake log row: `(intake_timestamp, sentrix_id, array_type, substrate, declared_sex, declared_age, status=STAGED)`

**Decision points.**
- IF both Grn and Red files present AND manifest entry complete → proceed to §12 (Step 0.2 Sample manifest creation)
- IF Grn or Red file missing → quarantine; notify operator; do not advance
- IF manifest fields missing → quarantine with `INCOMPLETE_MANIFEST` flag

**Failure modes.**
- *Truncated IDAT upload.* Detected by file-size sanity check (IDAT files for HM450K are ~8 MB each; EPIC ~14 MB each; anything below 1 MB is suspicious).
- *Misnamed Sentrix ID.* Detected when manifest references a barcode not present in the file pair.
- *Wrong array type declared.* The IDAT header encodes the array type; mismatch with manifest triggers a flag.
- *Duplicate intake.* Same Sentrix ID arriving twice within 24 hours triggers a soft-warn (might be re-run; might be operator error).

**Canonical cross-references.**
- Recipe §1.1 (Sample Intake)
- Roadmap §10.1.1 L1 row
- Capability Translator §3 (Status codes)

**CPG Plate references.** Not applicable at this step.

**Chain-link assignment.** L1.

---

### §12. Step 0.2 — Sample manifest creation

**What this step does.** Cross-references the IDAT file pair with the intake manifest to construct the canonical per-sample record. Every downstream step reads from this record. Once written, the record is immutable for that sample run — re-intakes get new run IDs.

**Inputs.**
- Staged IDAT file pair from §11
- Sample manifest entry from intake CSV/JSON

**Atlas reference.** Not consulted.

**Files invoked.**
- Manifest builder: `<inside `GAPE_WEB_v13.py`>`
- Schema validator: `<manifest schema — TBD per orchestrator design>`
- Per-sample record output: `<intake staging directory — TBD per orchestrator design>per_sample.json`

**The math.** None. Structural validation.

**CMB equivalent.** This is the **scan-ring metadata association** step. Planck's pipeline attaches each bolometer file to its corresponding pointing, gain calibration, and noise characterization records. Without this association, no downstream analysis can proceed — the data is meaningless until it's keyed to its physical context.

**How the methylome differs in implementation.** The methylome's "scan ring metadata" is patient demographics (age, sex, substrate type) rather than telescope pointing data. The keying is simpler — one IDAT pair to one patient record.

**How it's the same in principle.** Both attach raw detector output to the physical context required for downstream calibration. Without the manifest association, every downstream calibration step would be operating blind.

**Outputs.** <per-sample run record — exact structure TBD per orchestrator design> with structure:
```json
{
  "sample_run_id": "<UUID>",
  "sentrix_id": "200123456789_R01C01",
  "array_type": "EPIC_v1",
  "substrate": "whole_blood",
  "patient_id": "<hashed>",
  "declared_sex": "F",
  "declared_chronological_age": 54.3,
  "intake_timestamp": "2026-05-31T14:22:00Z",
  "status": "MANIFEST_COMPLETE"
}
```

**Decision points.**
- IF manifest validates against schema → proceed to §13 (Step 0.3 IDAT integrity hash check)
- IF schema validation fails → quarantine with `MANIFEST_INVALID` flag

**Failure modes.**
- *Schema drift.* If the intake CSV uses an old column name, the validator rejects. Operator must update the intake CSV to the current schema.
- *Encoded vs cleartext PII.* Patient identifiers must be hashed before reaching the engine. Cleartext PII triggers an immediate quarantine and notification.

**Canonical cross-references.**
- Recipe §1.2 (Manifest validation)
- Roadmap §10.1.1 L1 row

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §13. Step 0.3 — IDAT integrity hash check

**What this step does.** Computes SHA-256 hashes of both IDAT files in the pair. Compares against any prior intake of the same Sentrix ID (re-run detection) and registers the new hashes in the integrity log. This is the formal L1 integrity protocol the Roadmap §10.1.1 grades B+ today.

**Inputs.** The staged IDAT file pair from §11.

**Atlas reference.** Not consulted.

**Files invoked.**
- Hash module: `<inside `GAPE_WEB_v13.py`>`
- Integrity log: `<integrity log — TBD>`

**The math.** `sha256(grn_bytes)`, `sha256(red_bytes)`. Compare to prior runs.

**CMB equivalent.** This is **bolometer scan integrity verification.** Planck logs the checksum of every scan file at receipt to detect transmission corruption and re-runs. Identical checksums of the same scan ring indicate re-transmission; differing checksums of the same Sentrix ID indicate either a re-run with a fresh array (legitimate) or data corruption (problem).

**How the methylome differs in implementation.** Methylation IDAT files are smaller than CMB scan rings and don't compress further at runtime, so a fixed SHA-256 over the byte stream is sufficient. CMB pipelines often add stream-level checksums for chunked transmission; not needed here.

**How it's the same in principle.** Both verify the bytes that arrived match the bytes that were intended. Both treat hash mismatches as red flags that block the chain.

**Outputs.**
- Integrity log row: `(sentrix_id, intake_timestamp, grn_sha256, red_sha256, prior_hash_match)`
- A `STATUS=INTEGRITY_OK` stamp on the per-sample record

**Decision points.**
- IF both hashes new (no prior record) → proceed to §14 (Step 0.4 Control probe validation)
- IF hashes identical to prior intake → flag `RE_TRANSMISSION_DETECTED`; ask operator whether to overwrite or quarantine
- IF Sentrix ID matches prior intake but hashes differ → flag `LIKELY_RE_RUN_FRESH_ARRAY`; proceed with new run ID

**Failure modes.**
- *Hash collision with prior contaminated sample.* Extremely unlikely with SHA-256 but documented in the integrity log for traceability.
- *Bit-flip during storage.* If the staged file changes hash between intake and Stage 0.4 invocation, the integrity check at Stage 0.4 detects it.

**Canonical cross-references.** Roadmap §10.1.1 L1 row ("Declare formally: IDAT integrity check protocol... Document.")

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §14. Step 0.4 — Control probe validation

**What this step does.** Illumina arrays include ~600 control probes whose intended responses are known (positive controls, negative controls, hybridization controls, bisulfite conversion controls, extension controls, target removal controls, etc.). This step reads the control probe intensities from the IDAT pair and verifies each control class is within expected bounds. Out-of-bounds controls indicate a chemistry problem with the array — the patient sample is technically unreliable regardless of the biological signal.

**Inputs.** Decoded IDAT files (parsed via `illumina-idat-reader` Python package or equivalent).

**Atlas reference.** Not consulted — the control probe spec is Illumina's, not IAMAtlas's.

**Files invoked.**
- Control probe extractor: `<inside `GAPE_WEB_v13.py`>`
- Illumina control probe manifest: `<Illumina control probe manifest from `methylprep`>`
- QC log: `<control probe QC log — TBD>`

**The math.** Per-control-class summary statistics:
- Median intensity vs expected range (Illumina docs)
- Bisulfite conversion efficiency = median(BS-Conv-I) / (median(BS-Conv-I) + median(BS-Conv-II))
- Hybridization signal = median(Hyb-High) − median(Hyb-Low)
- Extension efficiency = ratio of methylated-extension to unmethylated-extension responses

Each control class has a manufacturer-specified pass range. Failure of any class flags the sample.

**CMB equivalent.** This is **detector calibration verification before science use.** Planck's pipeline validates each bolometer's gain, noise temperature, and linearity against on-board calibration sources before that bolometer's data is used in the cosmological analysis. A bolometer with anomalous calibration is masked out. Equivalent here: a sample with anomalous control probes is quarantined before its β values enter the chain.

**How the methylome differs in implementation.** Control probes are spatially co-located with science probes on the same array (vs CMB bolometers which calibrate against time-domain reference signals). One per-array calibration per sample, not a time-series calibration.

**How it's the same in principle.** Both verify the measurement apparatus is functioning correctly before using its readings. Both treat apparatus failure as a hard block on downstream analysis — bad calibration cannot be repaired by clever processing.

**Outputs.**
- A control probe QC report: `<control probe QC artifact — TBD>`
- A `CTRL_QC=PASS` or `CTRL_QC=FAIL_<class>` flag on the per-sample record

**Decision points.**
- IF all control classes pass → proceed to §15 (Step 0.5 Detection p-value QC)
- IF bisulfite conversion < 95% → quarantine with `BS_CONVERSION_LOW` (downstream β values are unreliable)
- IF hybridization controls fail → quarantine with `HYB_FAIL` (array chemistry was compromised)
- IF extension controls fail → quarantine with `EXT_FAIL`

**Failure modes.**
- *Operator pipetting error during array prep.* Manifests as low hybridization controls.
- *Insufficient bisulfite conversion.* Manifests as elevated β bias toward unmethylated state across the array. The control probe check catches this before β values lie about the biology.
- *Sample dilution / RNA contamination.* Detected by low total fluorescence across the array.

**Canonical cross-references.** Roadmap §10.1.1 L1 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §15. Step 0.5 — Detection p-value QC per probe

**What this step does.** For each of the ~485,000 (HM450K) or ~865,000 (EPIC) probes on the array, computes a detection p-value: the probability that the observed intensity is distinguishable from the array's background (negative-control distribution). Probes with detection p > 0.01 are flagged as "undetected" — their β value is unreliable.

**Inputs.** Decoded IDAT files + Illumina negative-control probe set.

**Atlas reference.** Not consulted — this is array-internal QC.

**Files invoked.**
- Detection p-value calculator: `<inside `GAPE_WEB_v13.py`>`
- Per-probe detection log (one row per probe, one column per sample, sparse): `<detection-p log — TBD>`

**The math.** For each probe with intensity I_probe and background distribution from negative controls (mean μ_bg, sd σ_bg):

> **detection_p = 1 − Φ((I_probe − μ_bg) / σ_bg)**

where Φ is the standard normal CDF. A probe with detection_p > 0.01 is flagged.

The Illumina default uses a slightly different formulation (the `minfi` R package's `detectionP` function), and CPG mirrors that exact computation for cross-platform compatibility.

**CMB equivalent.** This is **pixel-level signal-to-noise gating.** Planck masks pixels where the integration time is insufficient or the noise estimate exceeds a per-pixel threshold. Same operation: a measurement that hasn't risen above the noise floor cannot contribute to the final map.

**How the methylome differs in implementation.** The methylome's "noise" is the negative-control distribution embedded within the array. CMB noise comes from the bolometer's thermal characterization. Different physical noise sources; same statistical treatment.

**How it's the same in principle.** A measurement that doesn't cross the SNR floor produces a NULL, not a value. The downstream chain treats NULLs explicitly — never imputes silently.

**Outputs.**
- A per-probe detection p-value array (length 485,512 for HM450K, 865,919 for EPIC)
- A summary statistic: `pct_probes_detected_p_le_01` — the fraction of probes that passed
- A `DETECTION_QC=PASS` or `=FAIL_LOW_DETECTION` flag

**Decision points.**
- IF >99% of probes have detection p ≤ 0.01 → proceed to §16
- IF 95–99% detected → flag `DETECTION_BORDERLINE`, allow with warning
- IF <95% detected → quarantine with `DETECTION_QC_FAIL` (sample is broadly under-hybridized)

**Failure modes.**
- *Whole-array under-hybridization.* Manifests as bulk failure (e.g., 80% probes detected).
- *Localized array damage.* Specific physical region of the chip showing detection failure. Detected via spatial clustering of failed probes (not currently implemented; flagged in Phase B4 as `B4.2`).
- *Probe-class systematic failure.* Type I vs Type II probes with differential failure rates. Handled by Step 1.2 (probe-type normalization).

**Canonical cross-references.** Recipe §1.5 (Detection QC). Roadmap §10.1.1 L1 row.

**CPG Plate references.** Plate 1 stromal panel — the 4.9% MCMC coverage gap visible as the "galactic mask" is partly a consequence of detection failures in stromal-marker CpGs across the cohort that built the atlas.

**Chain-link assignment.** L1.

---

### §16. Step 0.6 — Bead count QC

**What this step does.** Each methylation probe is represented on the array by multiple beads (Illumina bead chemistry replicates each probe ~15-20 times per sample for redundancy). The bead count per probe is reported in the IDAT file. Probes with very low bead count have unreliable β values regardless of detection p-value.

**Inputs.** Decoded IDAT files.

**Atlas reference.** Not consulted.

**Files invoked.**
- Bead count reader: `<inside `GAPE_WEB_v13.py`>`

**The math.** Per probe: `bead_count_probe` extracted from IDAT. Threshold: `bead_count_probe >= 3` (Illumina default; can be tightened to ≥5 for high-stakes applications).

**CMB equivalent.** This is **integration-time validation per pixel.** Planck's pipeline checks that each sky pixel was visited enough times during the survey to give a reliable temperature average. Pixels with fewer than N hits are masked.

**How the methylome differs in implementation.** Bead replication is built into the array's physical design (Illumina manufactures redundant beads per probe) rather than driven by scan strategy.

**How it's the same in principle.** Both verify that the measurement at each spatial coordinate had enough independent samples to be reliable.

**Outputs.**
- Per-probe bead count vector
- Summary statistic: `pct_probes_bead_count_ge_3`
- A `BEAD_QC=PASS/FAIL` flag

**Decision points.**
- IF >99.5% of probes have bead_count ≥ 3 → proceed to §17
- IF <99.5% → flag with warning; consider tightening downstream confidence thresholds

**Failure modes.**
- *Manufacturing defects in array.* Rare but possible; detected by clusters of low-bead-count probes.
- *Damage during sample prep.* Detected by global bead count reduction.

**Canonical cross-references.** Recipe §1.5.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §17. Step 0.7 — Sample-level call rate

**What this step does.** Aggregates §15 (detection p-value) and §16 (bead count) into a single sample-level call rate: the fraction of probes that passed BOTH detection and bead count thresholds. This is the headline quality metric for the array as a whole.

**Inputs.** Outputs of §15 and §16.

**Atlas reference.** Not consulted.

**Files invoked.**
- Call rate calculator: `<inside `GAPE_WEB_v13.py`>`

**The math.** 
> **call_rate = N_probes_passing_both / N_probes_total**

Threshold: call_rate ≥ 0.98 to proceed.

**CMB equivalent.** This is **overall scan quality for a single observation epoch.** CMB pipelines compute the fraction of bolometers + the fraction of scans that pass all upstream QC; samples below a threshold are flagged for review or excluded.

**How the methylome differs in implementation.** The "sample" is a single array; the call rate is per-array. CMB call rates are typically per-scan-ring or per-bolometer-night.

**How it's the same in principle.** Single rolled-up metric that summarizes upstream QC. Operator-friendly threshold.

**Outputs.**
- Call rate value (float in [0,1])
- A `CALL_RATE=<value>` field on the per-sample record

**Decision points.**
- IF call_rate ≥ 0.98 → proceed to §18
- IF 0.95 ≤ call_rate < 0.98 → flag `CALL_RATE_BORDERLINE`, proceed with downstream confidence penalty
- IF call_rate < 0.95 → quarantine with `CALL_RATE_FAIL`

**Failure modes.** Same as §15 and §16 (rolled up).

**Canonical cross-references.** Recipe §1.5.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---


**Step 0.7b — HM450 ≥80% coverage gate + platform tagging.** When the patient was assayed on a 450K platform, verify that ≥80% of the HM450 reference CpGs the downstream chain depends on (deconvolver markers, A-score markers, Mahalanobis features, breast-epic / future card panels) returned valid β values. Halt with HARD failure below 80%. Per VAL-091 ad-LL-006, every downstream output is tagged with the platform identifier (`450K` or `EPIC`) so platform-stratified thresholds can be applied at Stage 7 / Stage 8 where 450K coverage gaps materially affect call rates. The platform tag flows into the audit trail at Stage 10.

### §18. Step 0.8 — Sex check vs metadata

**What this step does.** Computes the predicted sex of the sample from chrY-specific and chrX-specific probe intensities, then compares it to the declared sex in the manifest. A mismatch indicates either a sample-swap (catastrophic — patient identity is wrong) or a true biological discordance (rare — XXY, XX male, etc.).

**Inputs.** Decoded IDAT files + manifest's declared sex.

**Atlas reference.** Not consulted directly. (IAMAtlas does NOT carry sex-stratified posteriors at v0.1; sex foreground is handled separately at Stage 3.)

**Files invoked.**
- Sex predictor: `<inside `GAPE_WEB_v13.py`>`
- chrY/chrX probe coordinates: Illumina manifest files

**The math.** Standard `minfi::getSex()` algorithm:
- Mean log2-intensity of chrY probes: `log2_Y`
- Mean log2-intensity of chrX probes: `log2_X`
- Decision boundary in 2D (log2_X, log2_Y) space → predicted_sex ∈ {M, F}

**CMB equivalent.** This is **pointing-direction sanity check.** Planck verifies that each scan ring's pointing direction (RA, Dec) matches its declared target by cross-referencing with the spacecraft attitude solution. A mismatch indicates a metadata error that must be resolved before the data is used.

**How the methylome differs in implementation.** The biological "pointing direction" is sex; the verification is a 2D intensity check.

**How it's the same in principle.** Both verify the metadata describing the measurement matches what the measurement itself says. Mismatches are NEVER silently corrected — they're flagged for operator resolution.

**Outputs.**
- Predicted sex
- A `SEX_CHECK=PASS` (predicted == declared) or `SEX_CHECK=MISMATCH` flag

**Decision points.**
- IF predicted == declared → proceed to §19
- IF mismatch → quarantine with `SEX_MISMATCH`; require operator investigation before proceeding

**Failure modes.**
- *Sample swap at the lab.* Most common cause of sex mismatch. Investigation involves checking the lab's chain of custody for the IDAT pair.
- *Genuine biological sex chromosome variant.* Rare. Resolved by operator after consulting patient record.
- *Tumor contamination with loss-of-chrY.* Manifests as M-declared-but-F-predicted in male patients with chromosomal instability. Investigated case-by-case.

**Canonical cross-references.** Recipe §1.5.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1.

---

### §19. Step 0.9 — Stage 0 decision gate

**What this step does.** Consolidates all Stage 0 outputs into a single proceed/quarantine/reject decision. This is the formal end of L1 — every QC check has been run, and the sample either advances to Stage 1 (β computation) or it does not.

**Inputs.** Outputs of §11-§18.

**Atlas reference.** Not consulted.

**Files invoked.**
- Decision gate: `<inside `GAPE_WEB_v13.py`>`
- Stage 0 outcome log: `<Stage 0 verdict log — TBD>`

**The math.** Boolean conjunction:
```
PROCEED = (
    INTEGRITY_OK AND
    CTRL_QC == PASS AND
    DETECTION_QC in (PASS, BORDERLINE) AND
    BEAD_QC == PASS AND
    CALL_RATE >= 0.98 AND
    SEX_CHECK == PASS
)
```

Any single FAIL routes to quarantine. Any BORDERLINE proceeds with downstream confidence penalty applied.

**CMB equivalent.** This is the **pre-science decision gate.** A CMB scan ring either passes all upstream QC and enters the cosmological analysis, or it gets routed to the masked-data archive for forensic review.

**How the methylome differs in implementation.** One gate per array (one per patient sample). CMB gates can be per-bolometer-per-scan-ring (much finer-grained).

**How it's the same in principle.** Hard boolean decision based on conjunction of all upstream checks. No silent imputation. No "we'll fix it downstream."

**Outputs.**
- Final Stage 0 status: `STAGE_0=PROCEED` / `=QUARANTINE` / `=REJECT`
- A complete Stage 0 record in `<Stage 0 verdict log — TBD>`
- For PROCEED samples: per-sample record + cleaned IDAT pair handoff to Stage 1

**Decision points.**
- IF PROCEED → handoff to §20 (Step 1.1 Dye-bias correction)
- IF QUARANTINE → notify operator; do not advance
- IF REJECT → archive; document reason; do not retry without operator override

**Failure modes.**
- Any upstream Stage 0 failure cascades here.

**Canonical cross-references.** Recipe §1.5 (Decision gate). Roadmap §10.1.1 L1 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L1 (closes Stage 0).

---

## Stage 1 — Calibration & β computation (L2 + L3)

---

### §20. Step 1.1 — Dye-bias correction

**What this step does.** Illumina's methylation arrays use two dye colors (Cy3 green, Cy5 red) to distinguish methylated and unmethylated DNA. The two dyes have slightly different intensity characteristics that vary across arrays, batches, and even within a single chip. Dye-bias correction normalizes these differences so that downstream β values reflect actual methylation, not dye chemistry artifacts.

**Inputs.** The cleaned IDAT file pair from §19.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Dye-bias correction module: `<inside `GAPE_WEB_v13.py` (uses `minfi`/`methylprep` upstream)>`
- Implementation: typically wraps `minfi::preprocessFunnorm()` or `minfi::preprocessNoob()` from the standard methylation analysis stack

**The math.** The standard `noob` (normal-exponential out-of-band) correction:
1. Fit a normal-exponential model to the negative-control probe intensities per channel
2. Subtract the modeled background from each probe's intensity
3. Equalize the red and green channel intensity distributions to a common reference

For Type II probes (where both methylated and unmethylated states are measured in the same channel), additional correction equalizes the two states' baseline intensities.

**CMB equivalent.** This is **bandpass correction.** Each CMB frequency channel has its own bandpass profile that affects how the underlying signal is measured. Bandpass mismatch correction is a standard step in Planck's calibration pipeline. Same operation here: two channels (red, green) with different response profiles, corrected to a common reference.

**How the methylome differs in implementation.** Two channels (vs nine in Planck). Static, not time-varying. The correction parameters are learned per-sample from in-array control probes rather than per-mission from external calibration sources.

**How it's the same in principle.** Both correct for known instrumental response differences between channels before combining their signals into a single physical quantity.

**Outputs.**
- Dye-bias-corrected intensities per probe per channel
- A `DYE_BIAS_PARAMS` record documenting the correction parameters applied

**Decision points.**
- IF correction converges (negative-exponential fit succeeds) → proceed to §21
- IF correction fails (typically because of pathological control probe distributions) → quarantine with `DYE_BIAS_FAIL`

**Failure modes.**
- *Saturated controls.* Negative-control probes that themselves saturated the detector cannot anchor the background model. Detected during fit.
- *Bimodal control distribution.* Indicates contamination or chip damage. Cannot be cleanly modeled.

**Canonical cross-references.** Recipe §2.1 (Calibration). Roadmap §10.1.1 L2 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §21. Step 1.2 — Probe-type normalization

**What this step does.** Illumina's HM450K and EPIC arrays have two probe chemistries: Type I (two beads per CpG, one for methylated, one for unmethylated) and Type II (one bead per CpG, distinguishing methylation by dye color). The two chemistries produce subtly different β distributions for the same biological methylation level. Probe-type normalization brings them to a common distribution.

**Inputs.** Dye-bias-corrected intensities from §20.

**Atlas reference.** Not consulted.

**Files invoked.**
- Normalization module: `<inside `GAPE_WEB_v13.py` (uses `minfi`/`methylprep` upstream)>`
- Implementation options (configurable):
  - **funnorm** (functional normalization — Fortin 2014; standard for cross-cohort comparisons)
  - **noob** (Triche 2013; lighter touch, preserves more biological variance)
  - **SWAN** (Subset-quantile Within-Array Normalization; legacy)
  - **BMIQ** (Beta MIxture Quantile dilation; corrects Type II β distribution to match Type I)

**The math.** Each method differs:
- **funnorm**: regresses out the first few principal components of control probe intensities (which capture technical variation) from each probe's β value
- **noob**: applies dye-bias correction as a standalone normalization
- **SWAN**: rank-normalizes Type II probes to match Type I within each sample
- **BMIQ**: fits Beta-mixture distributions to Type I and Type II separately, transforms Type II to match Type I distribution shape

**Configuration default for CPG production: funnorm** (chosen because the engine operates across heterogeneous cohorts and funnorm has the strongest cross-cohort consistency in the literature). Other methods are available for cards that require a specific normalization.

**CMB equivalent.** This is **inter-channel calibration to a reference frequency.** Planck's nine frequency channels are calibrated against a common reference (typically the CMB dipole or a known astronomical source) so that the underlying physical signal is comparable across channels. Same operation here: two probe types calibrated to a common β distribution.

**How the methylome differs in implementation.** Methods are statistical (β-distribution matching) rather than physical-source calibration. The reference is a within-array population of probes, not an external calibration source.

**How it's the same in principle.** Both adjust the measurement so that the same underlying physical quantity produces the same number regardless of which detector chemistry measured it.

**Outputs.**
- Normalized β-equivalent intensities per probe (still pre-β-computation in some methods, post-β in others)
- A `NORMALIZATION_METHOD` record on the per-sample data structure

**Decision points.**
- IF normalization converges → proceed to §22
- IF normalization fails (e.g., funnorm's PCA cannot find stable components because the array is broadly damaged) → fall back to noob; if noob also fails, quarantine

**Failure modes.**
- *Insufficient control probe variation for funnorm.* Detected during PCA fit.
- *Bimodal sample population causing BMIQ misfit.* Manifest as broken Beta-mixture convergence.

**Canonical cross-references.** Recipe §2.1. Roadmap §10.1.1 L2 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §22. Step 1.3 — Batch correction (ComBat)

**What this step does.** When a card processes patients from multiple batches (different intake dates, different lab runs, different plate-positions on the same chip), inter-batch variation can dwarf biological signal. ComBat is the standard empirical-Bayes batch correction that removes additive and multiplicative batch effects while preserving the biological signal of interest. This step is conditional — it only runs when multi-batch data is being processed together.

**Inputs.** Normalized β-equivalent intensities from §21 across multiple samples in the current run.

**Atlas reference.** Not consulted.

**Files invoked.**
- Batch correction module: `<inside `GAPE_WEB_v13.py` (wraps `sva::ComBat`/`combat-py`)>`
- Implementation: typically wraps `sva::ComBat()` from the standard methylation analysis stack

**The math.** ComBat fits an empirical-Bayes hierarchical model:
> **β_corrected[i, sample] = (β_raw[i, sample] − γ_batch − α_i) / δ_batch + α_i**

where γ_batch is the additive batch shift, δ_batch is the multiplicative batch scaling, and α_i is the per-probe biological intercept. The hyperparameters are estimated empirically across all samples in the batch.

**CMB equivalent.** This is **mission-to-mission systematic correction.** When Planck combines data across observation seasons, gain drifts and calibration drifts between seasons must be removed before the data can be co-added. The empirical-Bayes structure is similar — characterize the drift parameters from the data itself, then correct.

**How the methylome differs in implementation.** Multiple "batches" can co-exist on one array run (different intake dates processed together). CMB seasons are temporally sequential and self-evident.

**How it's the same in principle.** Both estimate systematic drift between observation epochs and correct for it before downstream physics analysis. Both treat per-batch drift as a nuisance parameter to be marginalized out.

**Outputs.**
- ComBat-corrected β-equivalent intensities
- A `COMBAT_PARAMS` record documenting the per-batch γ and δ corrections applied
- A diagnostic plot of pre/post-correction batch separation (saved to QC artifact directory)

**Decision points.**
- IF single-batch run → SKIP this step; proceed directly to §23
- IF multi-batch and ComBat converges → proceed to §23
- IF multi-batch and ComBat fails (typically: too few samples per batch) → flag `BATCH_CORRECTION_INSUFFICIENT_N` and proceed without correction, with downstream confidence penalty

**Failure modes.**
- *Confounding between batch and biology.* If all cases are in batch A and all controls are in batch B, ComBat cannot distinguish batch effect from biological effect. Detected at intake and rejected with `CONFOUNDED_DESIGN`.
- *Insufficient samples per batch.* ComBat needs ≥3 samples per batch for stable estimation. Below this, falls back to no correction with warning.

**Canonical cross-references.** Recipe §2.1. Roadmap §10.1.1 L2 row.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §23. Step 1.4 — Bisulfite conversion efficiency check

**What this step does.** Methylation array chemistry requires bisulfite-treating the DNA before hybridization. Bisulfite converts unmethylated cytosines to uracil while leaving methylated cytosines intact. If the conversion is incomplete, β values are biased toward false hypermethylation. This step computes the conversion efficiency from internal control probes and either confirms it's adequate or flags the sample.

**Inputs.** Bisulfite conversion control probe intensities (from the §14 control probe extraction).

**Atlas reference.** Not consulted.

**Files invoked.**
- BS efficiency calculator: `<inside `GAPE_WEB_v13.py`>`

**The math.** Bisulfite conversion controls come in two classes (BS-Conv-I and BS-Conv-II). The efficiency is:
> **bs_efficiency = median_intensity(BS-Conv-I_converted) / (median_intensity(BS-Conv-I_converted) + median_intensity(BS-Conv-I_unconverted))**

A fully-converted sample reaches efficiency ≈ 1.0. A failed conversion drops below 0.95 — at which point β values are no longer reliable indicators of methylation state.

**CMB equivalent.** This is **detector gain stability verification.** Planck checks that each bolometer's gain has not drifted outside its calibration window. If gain has drifted, downstream temperature measurements are biased. Same idea: if chemistry has drifted (incomplete BS conversion), downstream methylation measurements are biased.

**How the methylome differs in implementation.** Chemistry drift (BS conversion failure) is irreversible — the sample is contaminated, not just miscalibrated. CMB gain drift can sometimes be corrected post-hoc; BS conversion failure cannot.

**How it's the same in principle.** Both treat the measurement apparatus's working state as a precondition for trusting its readings. Below threshold, the data does not enter the chain.

**Outputs.**
- `bs_efficiency` numeric value
- A `BS_CONVERSION` flag (PASS/BORDERLINE/FAIL)

**Decision points.**
- IF bs_efficiency ≥ 0.98 → proceed to §24
- IF 0.95 ≤ bs_efficiency < 0.98 → flag BORDERLINE, proceed with downstream confidence penalty
- IF bs_efficiency < 0.95 → quarantine with `BS_CONVERSION_FAIL` (sample chemistry failed, β values are unreliable)

**Failure modes.**
- *Incomplete sodium bisulfite reaction during sample prep.* Most common cause. Sample-prep batch should be re-investigated.
- *DNA degradation pre-conversion.* Detected by combination of low BS efficiency + low overall intensity.

**Canonical cross-references.** Recipe §2.1. Roadmap §10.1.1 L2 row ("Add bisulfite-conversion-efficiency nuisance parameter").

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L2.

---

### §24. Step 1.5 — β-value computation

**What this step does.** This is the formal map-making step — converting calibrated fluorescence intensities into the methylation level β at each CpG. β = M / (M + U + 100), where M is the methylated-state intensity, U is the unmethylated-state intensity, and 100 is the standard stabilization offset (prevents division by zero at very low intensities; cosmetic at high intensities). β is the canonical per-CpG measurement that every downstream step consumes.

**Inputs.** Calibrated intensities from §20-§22 (dye-bias corrected, probe-type normalized, optionally batch-corrected).

**Atlas reference.** Not consulted — β computation is intrinsic to the array, not the atlas.

**Files invoked.**
- β calculator: `<inside `GAPE_WEB_v13.py`>`

**The math.**
> **β = M / (M + U + 100)**

Per CpG per sample. Output is a number in [0, 1] (with theoretical extremes never reached because of the +100 stabilization).

For Type II probes, M and U are extracted from the green and red channels of the same bead. For Type I probes, M and U come from separate beads (one each).

**CMB equivalent.** This is **map-making.** The conversion from calibrated bolometer intensities (in detector units) to a physical sky temperature map (in μK) is performed by CMB pipelines using the Madam/MADmap solver. The CMB equivalent equation is more complex (involves the pointing matrix, the inverse-noise covariance, and a destriping operator), but the principle is identical: raw detector quantities → calibrated physical measurement per pixel.

The β formula is the methylome's exact analog of:
> **T_pixel = (Σ_i w_i × I_i) / Σ_i w_i**

— a weighted combination of detector readings producing a physical measurement per spatial coordinate.

**How the methylome differs in implementation.** The methylome's "pixel" is a CpG; its formula is simpler (two intensities + offset, no pointing matrix). CMB map-making involves time-ordered data; methylation β is computed once per sample.

**How it's the same in principle.** Both produce a per-coordinate physical quantity from underlying detector intensities. Both are the input that all downstream physics analyses operate on. The map-making step is the boundary between "raw instrument data" and "physical measurement."

**Outputs.** 
- A per-sample β matrix: N_probes × 1 (typically 485,512 for HM450K or 865,919 for EPIC, restricted to autosomes for most cards)
- A `BETA_COMPUTED=TRUE` stamp on the per-sample record

**Decision points.**
- IF all β values are in [0, 1] → proceed to §25
- IF any β values fall outside [0, 1] → arithmetic error upstream; quarantine for forensic review

**Failure modes.**
- *Numerical artifacts from extreme intensities.* Detected as β values exactly equal to 0 or 1 across many probes simultaneously.
- *Missing M or U value for a probe.* Per-probe NULL handling (the probe is dropped, not imputed).

**Canonical cross-references.** Recipe §2.2 (β computation). Roadmap §10.1.1 L3 row.

**CPG Plate references.** Plates 1, 3 (the β values themselves are what get rendered as the methylome sky maps).

**Chain-link assignment.** L3 (map-making).

---

### §25. Step 1.6 — β-value sanity checks

**What this step does.** Before β values feed into the deconvolver, run final sanity checks on the per-sample β matrix: distribution shape, modality, range coverage. This catches catastrophic failures of upstream calibration that didn't trip per-probe flags but produced an entire β matrix that doesn't look biologically plausible.

**Inputs.** β matrix from §24.

**Atlas reference.** Not consulted at this step — comparing β distribution to a healthy reference happens at Stage 4 (A-score), not here.

**Files invoked.**
- β QC module: `<inside `GAPE_WEB_v13.py`>`

**The math.** Three checks:
1. **Range coverage:** the β distribution should span [0.01, 0.99]. If 99% of values cluster in a narrow band (e.g., all between 0.4 and 0.6), the sample is degenerate.
2. **Bimodality:** methylation is fundamentally bimodal — most CpGs are either methylated (β ≈ 0.85) or unmethylated (β ≈ 0.15), with relatively few in between. Hartigan's dip test on the β distribution should reject unimodality (dip statistic D > 0.01).
3. **Cohort consistency:** the sample's median β should be within ±0.05 of the cohort's median β (when multi-sample run).

**CMB equivalent.** This is **final sanity check on the pixel-temperature distribution before science analysis.** Planck checks that the CMB map's temperature distribution has the expected variance (~80 μK) and is approximately Gaussian. Aberrant distributions get flagged before parameter inference. Same idea here: a β distribution that doesn't look biologically plausible is flagged before A-score computation.

**How the methylome differs in implementation.** The methylome's distribution is expected to be **bimodal** rather than Gaussian. Detecting bimodality is the test; detecting Gaussianity would be the test for CMB.

**How it's the same in principle.** Both check that the calibrated map distribution matches the expected statistical character before downstream inference. Failures here indicate upstream calibration didn't work even though all individual flags passed.

**Outputs.**
- A β-sanity record: `(range_coverage_pct, hartigan_dip_stat, dip_p, median_consistency_z)`
- A `BETA_SANITY=PASS/WARN/FAIL` flag

**Decision points.**
- IF all three checks pass → proceed to §26
- IF one check warns → flag with warning, proceed with reduced downstream confidence
- IF two or more checks fail → quarantine with `BETA_SANITY_FAIL`

**Failure modes.**
- *Bulk hypermethylation artifact.* Manifests as β distribution clustered near 1.0. Usually indicates failed BS conversion (should have been caught at §23, but this is the redundancy check).
- *Loss of bimodality.* Manifests as β distribution centered at 0.5. Indicates broad upstream signal degradation.
- *Sample-cohort outlier.* Single sample's median β far from cohort median. May indicate sample swap or genuine biological outlier — operator decides.

**Canonical cross-references.** Recipe §2.2. Roadmap §10.1.1 L3 row.

**CPG Plate references.** Plate 1 (the architectural-class β distributions are themselves a reference for what "biologically plausible" looks like).

**Chain-link assignment.** L3.

---

### §26. Step 1.7 — Probe response function (provisional)

**What this step does.** Marks where the L3 chain link needs further work per the Roadmap §10.1.1 grading. Each probe has a slightly different response function (how raw fluorescence scales to true methylation fraction). The standard β formula assumes uniform response across probes; in reality, probe-specific calibration would refine the β estimate. This is currently NOT implemented in production — the SOP declares it as a known L3 gap to be filled.

**Inputs.** Per-probe β values from §24.

**Atlas reference.** Not consulted.

**Files invoked.** *None in current production.* Future: `<future module — not built; documented gap per Roadmap §10.1.1 L3>`.

**The math.** Conceptually: β_true = f_probe(β_observed) where f_probe is a learned per-probe calibration curve. Currently f_probe is the identity function for all probes (no correction applied).

**CMB equivalent.** This is **per-detector transfer function correction.** Planck applies per-bolometer transfer functions to account for individual detector nonlinearities. CPG's equivalent is currently the identity transfer — every probe assumed identical.

**How the methylome differs in implementation.** Currently NOT implemented at production grade. CMB pipelines have per-bolometer transfer functions; CPG has uniform per-probe response.

**How it's the same in principle.** Acknowledging the gap is itself part of the L3 audit. The current grade of L3 = B reflects this missing component.

**Outputs.** Identity-corrected β values (i.e., the §24 β values, unchanged).

**Decision points.** Proceed to §27 in all cases (no current gating).

**Failure modes.** Not applicable in current implementation. When implemented, would catch probe-specific calibration drift.

**Canonical cross-references.** Roadmap §10.1.1 L3 row ("Declare as L3 explicitly; add probe-response-function nuisance treatment.") This step's existence in the SOP is the formal declaration.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L3 (provisional).

---

### §27. Step 1.8 — Stage 1 output: per-CpG β matrix

**What this step does.** Packages the calibrated β matrix and its provenance into the canonical Stage 1 output: a per-sample β file ready for Stage 2 deconvolution.

**Inputs.** β matrix from §24 + provenance records from §20-§26.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Output packager: `<inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** This is the **L3 → L4 handoff:** the calibrated all-sky temperature map (a HEALPix array) being handed off to the foreground-separation pipeline. In CPG terms: the cleaned β matrix being handed off to deconvolution.

**How the methylome differs in implementation.** The methylome's "all-sky map" is a per-CpG vector, not a HEALPix pixelization. (Plate 3 demonstrates that the per-CpG β values CAN be rendered as a HEALPix map — that's the foundational data structure for L5 work in Phase C — but the runtime chain operates on the per-CpG vector directly.)

**How it's the same in principle.** Same handoff: calibration is done; component-separation is next.

**Outputs.** A canonical per-sample β file containing:
- The β matrix (one number per CpG, NaN where probe failed QC)
- Per-probe QC flags (passing/failing detection, bead count)
- Calibration provenance (which normalization method, which batch correction applied)
- The per-sample QC summary from §19

Location: `<β matrix output — emitted internally by `GAPE_WEB_v13.py`>` (plus `.json` provenance).

**Decision points.** Handoff to §28 (Step 2.1 IAMAtlas REBUILD load).

**Failure modes.** Output packaging is mechanical; failures here indicate file-system issues, not chain issues.

**Canonical cross-references.** Recipe §2.2 (end of calibration). Roadmap §10.1.1 L3 row.

**CPG Plate references.** Plate 3 (the rendered β maps — illustrate what the Stage 1 output LOOKS like when projected to HEALPix).

**Chain-link assignment.** L3 (closes).

---

## Stage 2 — Deconvolution (L4 component separation, primary)

---

### §28. Step 2.1 — IAMAtlas REBUILD load

**What this step does.** Loads the proprietary IAMAtlas REBUILD reference matrix into memory. This is the first moment in the chain where the calibrated instrument enters. Every subsequent step that consults IAMAtlas reads from this loaded reference (or from a runtime-matrix derivative of it).

**Inputs.** The proprietary IAMAtlasREBUILD.csv file (605 MB) — proprietary IP, not in the public repository.

**Atlas reference.** **The atlas IS being loaded at this step.** This is the "atlas enters the chain" moment.

**Files invoked.**
- Atlas loader: `<atlas-loading logic inside `GAPE_WEB_v13.py`>`
- Source data: `IAMAtlasREBUILD.csv` (proprietary)
- Provenance: `IAMAtlasREBUILD_provenance.json` (H_min values frozen 2026-04-06)

**The math.** None. Memory-resident table indexed by CpG ID, with per-class posterior columns (mean, sd, ci_lo, ci_hi).

**CMB equivalent.** This is **loading the published Planck data products** (e.g., the Planck 2018 likelihood, the foreground templates, the dust polarization map). Once loaded, these data products serve as the reference against which the science measurement is compared. Same here: IAMAtlas is the reference against which patient β values are scored.

**How the methylome differs in implementation.** IAMAtlas is proprietary; Planck data products are public. CPG ships derivative artifacts (markers, age baseline, Mahalanobis reference) to the repo; the source atlas stays in the vault.

**How it's the same in principle.** Both load a calibrated, version-pinned reference data product into memory at the start of every analysis run. Both treat the loaded reference as immutable for that run.

**Outputs.** An in-memory IAMAtlas object with:
- Per-CpG × per-class posterior moments
- H_min lookup by class (read from `IAMAtlasREBUILD_provenance.json`)
- A SHA-256 fingerprint of the loaded data (logged for run audit)

**Decision points.**
- IF atlas loads cleanly → proceed to §29
- IF atlas file SHA-256 mismatches the expected fingerprint → ABORT (someone modified the atlas; chain integrity compromised)

**Failure modes.**
- *Atlas file corruption.* Detected by SHA-256 mismatch.
- *Atlas file version drift.* If the loaded atlas's provenance doesn't match the engine's expected version, refuse to run until reconciled.

**Canonical cross-references.** Recipe §2.3 (Atlas load). Roadmap §10.1.1 L4 row (atlas as the calibrated reference for L4).

**CPG Plate references.** Plate 1 (all four plates are renderings of IAMAtlas REBUILD).

**Chain-link assignment.** L4 (atlas is the calibrated reference for component separation).

---

### §29. Step 2.2 — Per-class marker pool extraction

**What this step does.** From the loaded atlas, extract the per-class marker CpG lists — the subset of CpGs that are maximally informative for each architectural class. These are pre-computed (delivered as the runtime artifact `iamatlas_celltype_markers_v0_2.json`); this step loads them and pairs them with the patient's β values at the corresponding CpGs.

**Inputs.** The loaded IAMAtlas object from §28 + the per-sample β matrix from §27.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the marker CpG lists at `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` (115 cell types × 100 markers, derived from IAMAtlas REBUILD's per-cell-type posteriors by the one-vs-rest top-N algorithm).

**Files invoked.**
- Marker loader: `<marker-pool extraction inside `GAPE_WEB_v13.py`>`
- Marker artifact: `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json`
- SHA-256 anchor: `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.sha256`

**The math.** Selection criterion from the artifact's metadata:
> `|target_celltype_mean − mean(other_114_celltype_means)|, top N=100 by score`

Computed at atlas build time, frozen in the artifact. At runtime, this step is pure lookup.

**CMB equivalent.** This is **multipole binning / aperture-mask selection.** Planck's foreground separation operates on selected angular scales and frequency channels chosen for their informativeness about the foreground component being separated. Same operation here: from the full 481K-CpG atlas, select the ~11,500 CpGs (115 × 100) most informative for cell-type discrimination.

**How the methylome differs in implementation.** Selection is per-cell-type by one-vs-rest contrast in posterior mean; CMB selection is per-multipole-band by foreground sensitivity. Different selection criteria, same purpose: focus the analysis on the maximally informative subset.

**How it's the same in principle.** Both reduce the dimensionality of the inverse problem (decomposition into classes/components) by restricting to the most informative measurements.

**Outputs.** A patient-specific marker pool table:
- 115 cell types
- For each cell type: its top-100 marker CpGs
- For each marker CpG: the patient's β value at that CpG (NaN if probe failed QC)
- For each marker CpG: the IAMAtlas posterior mean and sd per class

**Decision points.**
- IF ≥80% of marker CpGs have valid β values for the patient → proceed to §30
- IF 50-80% → flag `MARKER_COVERAGE_BORDERLINE`, proceed with downstream confidence penalty
- IF <50% → quarantine with `INSUFFICIENT_MARKER_COVERAGE`

**Failure modes.**
- *Probe-level QC failures concentrating in marker CpGs.* Causes marker coverage to drop disproportionately. Detected here.
- *Stale marker artifact.* If the marker JSON's SHA-256 doesn't match the recorded anchor, refuse to run.

**Canonical cross-references.** Recipe §3.1 (Marker selection). Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plate 4 Panel B (chr16+chr17 cold-patch zones — the marker pool concentrates heavily on these chromosomes, which has implications for §35 foreground subtraction).

**Chain-link assignment.** L4.

---

### §30. Step 2.3 — Walther IAM Deconvolver (Path 1, NNLS)

**What this step does.** Runs the production deconvolver — Walther — on the patient's β values at the marker pool. Walther solves a constrained non-negative least-squares (NNLS) problem to estimate the per-class cellular fractions that best explain the observed β values, given the IAMAtlas reference matrix. This is the production answer that feeds Stages 3, 4, 8.

**Inputs.** Marker pool table from §29.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the per-class β posterior means at marker CpGs serve as the reference matrix X in the NNLS system. Loaded from `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` extended with the atlas posterior moments at those CpGs.

**Files invoked.**
- Walther deconvolver: `Biological_Physics/atlas_vault/deconvolver/walther_iam_deconvolver.py`
- Class-CpG reference matrix: built at runtime from the atlas posterior at marker CpGs

**The math.** For each patient, Walther solves:
> **min ‖X · f − β_observed‖₂²    subject to f ≥ 0**

where:
- **X** is the (m × 8) reference matrix: m marker CpGs × 8 architectural class posterior means
- **β_observed** is the patient's m × 1 β vector at those CpGs
- **f** is the (8 × 1) per-class fraction vector to be estimated

NNLS is solved via the Lawson-Hanson active-set algorithm (`scipy.optimize.nnls`). The non-negativity constraint reflects the physical fact that cellular fractions cannot be negative.

After NNLS returns **f_raw**, Walther normalizes: **f_normalized = f_raw / sum(f_raw)** (the fractions sum to 1 by construction, since the per-class atlas posteriors span the cellular composition space).

Walther also runs streaming-mode: if the input contains >10K CpGs, it processes in chunks of 1K markers, accumulating partial sums and re-solving every 10 chunks. This is the methylome's analog of streaming map-making — handles arbitrarily large CpG counts without exceeding memory.

**CMB equivalent.** This is **Commander Bayesian component separation.** Planck's Commander algorithm solves the analogous inverse problem: given multi-frequency CMB observations, decompose them into cosmological + foreground components using a Gibbs sampler over the joint posterior. Walther's NNLS is a constrained-optimization analog of Commander's Bayesian Gibbs sampling — same inverse problem, different solver. NNLS gives a point estimate; Commander gives a posterior distribution. CPG's Phase E will add a proper posterior to Walther's output (currently just a point estimate + status flags).

**How the methylome differs in implementation.** Walther produces a point estimate per patient (faster, simpler) while Commander produces a posterior (slower, richer). The choice reflects clinical pipeline constraints — patient reports need fast, deterministic outputs.

**How it's the same in principle.** Both decompose a multi-channel observation into class/component contributions using a known reference matrix and a physical constraint (non-negativity / positivity of densities).

**Outputs.** Per patient:
- 8 architectural-class fractions: stem_pluri, stem_adult, progenitor, cycling, secretory, immune, terminal, stromal (each in [0,1], summing to 1)
- 115 per-cell-type fractions (Walther's Tier 2 output — finer-grained decomposition within each class)
- Per-class residual MAE: how well the reference matrix explains the patient's β values at marker CpGs

**Decision points.**
- IF NNLS converges and class residual MAE < 0.05 → proceed to §31 (status code assignment)
- IF NNLS converges but residual MAE 0.05-0.10 → flag `MODERATE_RESIDUAL`, proceed
- IF NNLS doesn't converge OR residual MAE > 0.10 → flag `POOR_DECONVOLUTION`, do not advance until §33 cross-method gate verifies

**Failure modes.**
- *Singular reference matrix.* Detected by NNLS condition number check.
- *Borderline patient mass between two classes.* NNLS pushes mass to the single best class (vs GLS which splits) — this is the expected NNLS inductive bias and is documented in Phase B2.1 finding.

**Canonical cross-references.** Recipe §3.2 (Walther deconvolver). Roadmap §10.1.1 L4 row. Capability Translator §1, §2, §5, §6.

**CPG Plate references.** Plate 1 (the 8-class atlas Walther deconvolves against).

**Chain-link assignment.** L4 (primary component separation).

---

### §31. Step 2.4 — Walther per-class confidence + status codes

**What this step does.** Walther emits per-class confidence indicators alongside each fraction estimate. Confidence is bounded [0, 1] and is NOT a calibrated probability — it's a coverage × fit-quality composite. Status codes flag specific failure modes (insufficient markers, no overlap, etc.) so downstream stages can route appropriately.

**Inputs.** NNLS solution from §30 + the per-class marker coverage metrics.

**Atlas reference.** Not consulted further at this step — uses outputs from §30.

**Files invoked.** Same module as §30 (`walther_iam_deconvolver.py`).

**The math.** Per class:
> **confidence_class = coverage_class × max(0, 1 − dispersion_class / 0.20)**

where:
- **coverage_class** = (markers_matched_for_class / markers_expected_for_class) — bounded [0,1]
- **dispersion_class** = stdev of per-CpG β within the class's marker pool (a high stdev indicates poor agreement between CpGs supposedly diagnostic of the same class)
- The 0.20 ceiling converts dispersion to a [0,1] penalty

Confidence is conservative — designed to fail loudly when something is wrong rather than report a falsely high number.

**Status codes:**
- `OK` — normal operation
- `INSUFFICIENT_MARKERS` — fewer than 20 markers matched for the class
- `NO_MARKER_OVERLAP` — zero markers matched
- `HIGH_DISPERSION` — dispersion exceeds 0.30 (markers disagree internally)
- `RESIDUAL_HIGH` — class-level residual exceeds the cohort-typical threshold

**CMB equivalent.** This is the **per-component confidence diagnostic in Commander.** Commander outputs not just the component amplitudes but also their posterior widths, residual amplitudes per pixel, and goodness-of-fit per region. CPG's confidence + status codes are the operational simplification of that diagnostic suite — calibrated [0,1] confidence for clinician communication + explicit failure flags for chain routing.

**How the methylome differs in implementation.** Single scalar confidence per class + discrete status code, vs Commander's per-pixel residual maps.

**How it's the same in principle.** Both attach explicit reliability indicators to every decomposition output. Both refuse to silently report a low-quality decomposition as if it were high-quality.

**Outputs.** Per class:
- `confidence_class` (float in [0,1])
- `status_class` (one of the codes above)

**Decision points.**
- IF all 8 classes have status=OK → proceed to §32
- IF any class has INSUFFICIENT_MARKERS or NO_MARKER_OVERLAP → flag that specific class, proceed with others
- IF >2 classes have non-OK status → quarantine with `BROAD_DECONVOLUTION_FAILURE`

**Failure modes.** Cascade from §30; specific failure modes are captured by the status codes themselves.

**Canonical cross-references.** Recipe §3.2. Capability Translator §2, §3.

**CPG Plate references.** Plate 1 stromal panel (the 4.9% MCMC coverage gap manifests as low stromal confidence for many patients — the methylome's known unknown).

**Chain-link assignment.** L4.

---


**Step 2.4b — Credible intervals from MCMC posteriors.** The atlas is constructed via MCMC over per-CpG β posteriors per architecture class (frozen 2026-04-06, R-hat < 1.001). Each atlas cell `(CpG, class)` has both a posterior mean (μ) AND a posterior SD (σ). The Walther deconvolver consumes both: μ to solve the NNLS system, σ to propagate uncertainty into the returned fractions. The Stage 2 output therefore carries per-class fraction point estimates AND credible intervals — a 0.05 fraction with CI [0.02, 0.08] is reported differently from a 0.05 fraction with CI [0.04, 0.06] in the doctor report's Quality section (§75). This is the only place in the V1 chain where atlas L8 MCMC information enters the per-patient flow; downstream stages operate on the point estimates with the CIs carried as a parallel uncertainty channel.

### §32. Step 2.5 — NILC v2 deconvolver (Path 2, departure-from-consensus GLS)

**What this step does.** Runs the cross-method check — NILC v2 — on the same patient β values, same marker pool, same atlas reference. NILC uses a fundamentally different inversion algorithm (departure-from-consensus generalized least squares with simplex projection) to produce an independent per-class fraction estimate. This is the methylome's Commander/NILC/SMICA discipline.

**Inputs.** Same as §30: marker pool table from §29.

**Atlas reference.** **IAMAtlas consulted.** Same reference matrix as Walther — but used differently (GLS in departure space rather than NNLS in raw space).

**Files invoked.**
- NILC deconvolver: `Biological_Physics/chain_of_custody/L4_component_separation/nilc_deconvolver.py`

**The math.** NILC v2 (Phase B2.1 algorithm):
1. **Build the consensus signal** — at each marker CpG, compute the mean of the 8 class posterior means (the "consensus β" at that CpG).
2. **Compute departures** — subtract the consensus from the reference matrix and from the patient β. Both are now expressed as departures from consensus.
3. **GLS solve** — solve the unconstrained generalized least squares system on the departure space, with per-CpG inverse-variance weighting (uses the atlas posterior SD as the variance).
4. **Simplex project** — the raw GLS solution can have negative or unbounded values; project it onto the simplex (non-negative, summing to 1) to get the final fraction estimate.

The mathematical form:
> **f_GLS = (X_dep^T · W · X_dep)⁻¹ · X_dep^T · W · β_dep**

where X_dep is the departure-form reference matrix, β_dep is the departure-form patient β, and W is the diagonal inverse-variance weight matrix. After solving, f_GLS is simplex-projected.

**CMB equivalent.** This is **Planck NILC** — literally the same algorithm name. Planck's Needlet Internal Linear Combination decomposes the CMB into component signals by constrained linear combinations of multi-frequency channels, with the constraint that the combination preserves the desired signal while minimizing residual variance from contaminants. CPG's NILC v2 is the methylome implementation of this exact discipline. The "needlet" structure of CMB NILC (multipole-localized wavelets) is replaced by the marker-pool structure of CPG NILC (class-localized CpG groups).

**How the methylome differs in implementation.** Methylome operates in the discrete-class space (8 classes) rather than the continuous angular-scale space (multipoles). The departure-from-consensus reformulation is the methylome-specific innovation that handles the non-orthogonal columns of the class reference matrix.

**How it's the same in principle.** Both produce an independent per-component estimate via GLS-with-constraints, intended to cross-check a primary deconvolver (Commander or Walther) at the inference layer.

**Outputs.** Per patient:
- 8 architectural-class fractions (independent of Walther's)
- Per-class residual after simplex projection

**Decision points.** Proceed to §33 (cross-method gate). NILC failure does not block the chain at this point — the gate decides.

**Failure modes.**
- *Ill-conditioned departure matrix.* Detected by GLS condition number; falls back to a ridge-regularized variant.
- *Aggressive simplex projection (when GLS produces many negative values).* The simplex projection can flatten the distribution; documented as a known NILC characteristic.

**Canonical cross-references.** Phase B2.1 finding doc at `Biological_Physics/chain_of_custody/L4_component_separation/Phase_B2_1_FINDING.md`. Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plate 1 (the same 8 classes that NILC decomposes against).

**Chain-link assignment.** L4 (cross-method component separation).

---

### §33. Step 2.6 — Cross-method gate check

**What this step does.** Compares Walther's and NILC's per-class fraction estimates to verify they agree at the biological-inference layer (sign and rough magnitude of effects). Substrate-level disagreement (Walther vs NILC fractions differ by ~12 percentage points in median L1 on EPIC-Italy) is EXPECTED — that's the inductive-bias difference between NNLS and GLS. Inference-level disagreement (the two methods give opposite signs for the disease-relevant class) is a problem.

**Inputs.** Walther fractions from §30 + NILC fractions from §32.

**Atlas reference.** Not consulted at this step — operates on the two deconvolvers' outputs.

**Files invoked.**
- Gate logic: built into `nilc_deconvolver.py` as the cross-method comparison routine.

**The math.** Two-level comparison:
- **Substrate level (informational, not gating):** per-class L1 distance |f_Walther − f_NILC|; cohort median expected ≈ 0.10-0.25. Reported but does not gate.
- **Inference level (gating):** for the disease-relevant class (varies per card), compare the SIGN of (case − HC) effect under each deconvolver. If both methods produce the same sign on the disease-relevant class, the gate PASSES. If they disagree on sign, the gate FIRES.

In production for a single patient (no case/HC labels available), the gate uses a per-class agreement threshold: both methods must place the patient on the same SIDE of the cohort median for the dominant class. If the patient is in the top quartile for immune A-score by Walther but the bottom quartile by NILC, the gate fires.

**CMB equivalent.** This is **Planck's cross-method cosmology consistency check.** Planck publishes its primary cosmological parameter estimates from Commander but cross-validates against NILC, SMICA, and SEVEM. Disagreement at the per-pixel temperature level is expected and quantified as systematic uncertainty. Disagreement at the parameter level (e.g., different methods producing different H₀) is a real problem that gates publication.

**How the methylome differs in implementation.** Two methods (Walther + NILC) rather than four. Single per-patient inference (vs cosmological-parameter inference). Per-class agreement rather than per-parameter agreement.

**How it's the same in principle.** Both treat method-method agreement at the inference layer as a hard gate. Both document substrate-level disagreement as a systematic to propagate forward.

**Outputs.** Per patient:
- A cross-method gate verdict: PASS / FLAG / FAIL
- Per-class substrate disagreement (the L1 distances) for the audit trail
- For PASS: proceed with Walther's fractions as the production answer
- For FLAG: proceed but with cross-method-uncertainty annotation on the report
- For FAIL: do not advance; route for manual review

**Decision points.**
- IF gate PASS → proceed to §34 (Stage 2 output packaging)
- IF gate FLAG → proceed but stamp report with cross-method advisory
- IF gate FAIL → quarantine with `CROSS_METHOD_DISAGREEMENT`, require operator review

**Failure modes.**
- *Genuine biological anomaly.* The patient has cellular architecture that lies in a region where Walther's NNLS-bias and NILC's GLS-bias produce different inferences. Documented; the patient's case may itself be informative.
- *Upstream QC failure that wasn't caught.* The gate firing can be a downstream detection of an upstream calibration issue.
- *Reference matrix instability for this patient's substrate.* Rare; documented case by case.

**Canonical cross-references.** Phase B2.1 finding doc. Recipe §3.3 (Cross-method gate, added v3).

**CPG Plate references.** Plate 1 (cross-method comparison validates the per-class atlas posteriors are stable).

**Chain-link assignment.** L4 (cross-method component-separation discipline).

---

### §34. Step 2.7 — Stage 2 output

**What this step does.** Packages the Stage 2 outputs into the canonical structure consumed by Stage 3 and beyond.

**Inputs.** Walther fractions + NILC fractions + cross-method gate verdict.

**Atlas reference.** Not consulted.

**Files invoked.**
- Stage 2 packager: `<Stage 2 packaging inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** The **component-separated CMB map handoff** to the foreground-subtraction stage (galactic dust, synchrotron removed; cosmological CMB ready for further processing).

**How the methylome differs in implementation.** Per-patient single-sample output rather than full-sky pixelized map.

**How it's the same in principle.** Both produce the cleaned cosmological/biological signal for downstream physics analysis.

**Outputs.**
- Production fractions (Walther's, per Stage 2.3): 8 classes + 115 cell types
- Per-class confidence + status codes (per Stage 2.4)
- Cross-method gate verdict (per Stage 2.6)
- NILC fractions stored for audit trail (per Stage 2.5)

Location: `<Stage 2 output — emitted internally by `GAPE_WEB_v13.py`>`.

**Decision points.** Handoff to §35 (Step 3.1 Age-axis foreground subtraction) for the now-deconvolved data.

**Failure modes.** Output packaging is mechanical.

**Canonical cross-references.** Recipe §3 (end of deconvolution). Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plates 1, 4 (the deconvolved per-class signals are what get visualized in the architectural-class panels).

**Chain-link assignment.** L4 (primary closes; secondary begins at §35).

---

## Stage 3 — Foreground subtraction (L4 component separation, secondary)

---

### §35. Step 3.1 — Age-axis foreground subtraction

**What this step does.** Removes the per-CpG age-drift component from the patient's β values before downstream A-score computation. Methylation drifts predictably with age at many CpGs (the basis of all methylation clocks); this drift is a foreground in the same sense that galactic dust is a foreground for CMB cosmology. Subtracting it cleans the disease-signal component.

**Inputs.** The Stage 2-deconvolved β data + the patient's declared chronological age.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the per-CpG age layer at `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv` (8,199 CpGs × {α intercept, γ slope per year, R², n_samples}), built from EPIC-Italy HC by the Phase B3 age-axis foreground module.

**Files invoked.**
- Foreground module: `Biological_Physics/atlas_vault/components/age_axis_foreground.py`
- Age layer: `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv`

**The math.** Per CpG i:
> **β_corrected[i] = β_observed[i] − γ_i × (age − age_train_mean)**

where γ_i is the per-CpG age slope (from the age layer) and `age_train_mean` is the mean training age (also stored in the age layer). This subtracts the age-drift component while preserving everything else.

Per-CpG R² values are stored alongside; the engine could optionally weight subtraction by R² (only apply correction to CpGs where the age fit is reliable), but the default is unweighted subtraction across all CpGs in the age layer.

**CMB equivalent.** This is **galactic foreground dust subtraction.** Planck removes the galactic dust contribution from each pixel using a multi-frequency template (since dust has a known frequency dependence). Once subtracted, the remaining signal is closer to the pure cosmological CMB. Same operation here: subtract a known foreground component (age drift) to expose the disease signal underneath.

**How the methylome differs in implementation.** The "frequency dependence" of methylome foreground is the per-CpG slope vs age (a linear model). Galactic dust uses a power-law frequency spectrum.

**How it's the same in principle.** Both subtract a known foreground component using a per-coordinate template, leaving the cleaned signal of interest.

**Outputs.** β_corrected matrix (same shape as the input, with age-drift subtracted).

**Decision points.**
- IF patient's age is within the calibration range of the age layer → apply correction, proceed to §36
- IF patient's age is outside the calibration range (very young or very old) → apply with extrapolation warning, proceed with reduced confidence on downstream age-related readouts

**Failure modes.**
- *Patient declared age is wrong.* Manifests as anomalous post-correction β distribution. Not directly detectable here; surfaces at §41 (β_mean computation) if it's severe.
- *Per-CpG age slope is wrong because the calibration cohort doesn't generalize to this patient's population.* This is the known limitation of the EPIC-Italy-only calibration; Phase B4 will add population-specific age layers.

**Canonical cross-references.** Recipe §4.1 (Age foreground). Roadmap §10.2.2 Phase B3 (the age-axis module). Phase B3 finding doc at `Biological_Physics/atlas_vault/age_clock/Phase_B3_FINDING.md`.

**CPG Plate references.** Plate 4 Panel D (Differentiation Gradient — illustrates how age drift maps onto the methylome sphere).

**Chain-link assignment.** L4 (secondary component separation).

---

### §36. Step 3.2 — Sex-axis foreground subtraction (when present)

**What this step does.** Removes the per-CpG sex-specific methylation component. Methylation patterns differ systematically between males and females at many autosomal CpGs (not just chrX/chrY). For multi-sex cohorts, subtracting this foreground prevents sex from contaminating the disease signal.

**Inputs.** β_corrected from §35 + patient's declared sex_at_birth (from intake questionnaire q06 / manifest, validated at §18).

**Atlas reference.** `IAMAtlas_sex_layer.csv` — per-CpG (α, ψ_male, R², n_samples, is_chr_x, is_chr_y, x_inactivation_flag). Built once via `SexAxisForeground.fit()` on the n_hc=601 HC cohort with sex-at-birth metadata; cached as a frozen runtime artifact. **Module BUILT and layer CSV FIT (v1.2, 2026-06-06 on GSE50660 n=464).**

**Files invoked.** `Biological_Physics/atlas_vault/walther_clinical_runtime/IAM_Cellular_Age/sex_axis_foreground.py` — `class SexAxisForeground` (mirrors `AgeAxisForeground` interface). API: `.load_layer(path)`, `.subtract_from_single_patient(patient_beta, sex_at_birth)`.

**The math.** Per CpG i:
> **β_corrected[i] = β_observed[i] − ψ_i × indicator_male**

where ψ_i is the per-CpG male-vs-female shift (learned on HC samples by OLS regression of β on sex indicator), and indicator_male ∈ {0, 1} for {female, male} respectively. Intercept α_i NOT subtracted — preserves per-CpG baseline. For female samples (indicator=0), no subtraction. For male samples, ψ_i subtracted.

Special handling of sex chromosomes:
- **chrY CpGs** carry `is_chr_y` flag; for female samples these CpGs are masked entirely (no Y chromosome) rather than subtracted.
- **chrX CpGs** carry `is_chr_x` flag; high-|ψ| chrX CpGs (above 0.20 threshold) additionally carry `x_inactivation_flag` to mark XCI-driven shifts that reflect X-inactivation biology, not disease-relevant signal.

**CMB equivalent.** Secondary foreground subtraction — like removing synchrotron emission after dust. Each foreground has its own template and subtraction operator. Sex acts as a known, frozen, additive component.

**How the methylome differs in implementation.** Sex-chromosome handling (chrX/chrY) has no CMB analog — the framework simply masks chrY for female samples and flags chrX XCI loci.

**How it's the same in principle.** Identical to age-axis subtraction (§35): per-CpG OLS coefficient frozen at training time + linear subtraction at runtime.

**Outputs.** β_corrected matrix with sex component removed; chrY mask flag for female samples; x_inactivation flag per CpG for downstream consumer transparency.

**Decision points.** v1.2 default behavior: PASS THROUGH (no correction applied) until `IAMAtlas_sex_layer.csv` is fit at v1.3. Stage 7 sex-stratified threshold tables in `tier_breakpoints.json v1.2` absorb the bulk effect as interim mitigation. Layer CSV FIT 2026-06-06: apply correction, proceed to §37; Stage 7 sex-stratification interim mitigation retires.

**Failure modes.** Same pattern as age foreground (§35). Additionally: missing sex_at_birth in intake → cannot apply subtraction → audit trail flags STAGE_3_SEX_FOREGROUND_SKIPPED_NO_METADATA.

**Canonical cross-references.** BUILD_SPEC v1.2 §3.3 (module table) + §5 Stage 3 (sequence). Roadmap §10.2.2 B4 (now built; layer fit scheduled v1.3).

**CPG Plate references.** Not yet applicable.

**Chain-link assignment.** L4 (secondary foreground).

---

### §37. Step 3.3 — Batch/plate foreground subtraction (when present)

**What this step does.** Removes per-CpG systematic shifts due to plate position, intake batch, or processing date. ComBat (§22) handles this at the calibration level when multi-batch data is being processed together; this step would handle residual batch effects at the post-deconvolution level.

**Inputs.** β_corrected from §36 + batch metadata from the per-sample record.

**Atlas reference.** Not directly — batch shifts are sample-cohort-specific, not atlas-anchored.

**Files invoked.** *Future:* `[Phase B4 deliverable per Roadmap §10.2.2 — module not yet built]`. Currently: SKIP.

**The math.** *When implemented:* per-CpG residual batch shift learned from the current cohort's HC samples; subtracted before A-score computation.

**CMB equivalent.** This is **mission-mission residual systematic correction** — handling the residual after ComBat-equivalent global batch correction.

**How the methylome differs in implementation.** Currently unimplemented at this stage (ComBat handles it at calibration). Future Phase B4 work would add a downstream verification layer.

**How it's the same in principle.** Layered systematic-correction discipline — multiple correction stages, each catching residuals from upstream.

**Outputs.** *When implemented:* β_corrected with residual batch shifts removed.

**Decision points.** Currently: PASS THROUGH.

**Failure modes.** *When implemented:* documented per the §35 pattern.

**Canonical cross-references.** Roadmap §10.2.2 B4.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L4 (secondary).

---

### §38. Step 3.4 — Ancestry-axis foreground subtraction (when present)

**What this step does.** Removes per-CpG ancestry-specific methylation differences. Genetic ancestry correlates with methylation at thousands of CpGs (some via mQTL effects, some via developmental differences); separating ancestry signal from disease signal requires this foreground subtraction for diverse cohorts.

**Inputs.** β_corrected from §37 + patient's inferred or declared ancestry.

**Atlas reference.** **Currently not implemented at production grade** — the Roadmap Phase B4 covers this. Requires ancestry-informative-marker (AIM) panels and per-CpG ancestry slope coefficients.

**Files invoked.** *Future:* `[Phase B4 deliverable per Roadmap §10.2.2 — module not yet built]`. Currently: SKIP.

**The math.** *When implemented:* methylome PCA on AIM CpGs to infer ancestry coordinates, then per-CpG subtraction along the inferred ancestry direction.

**CMB equivalent.** This is **CMB secondary foreground subtraction (e.g., extragalactic point sources, Sunyaev-Zel'dovich clusters)** — a known foreground that varies in pattern but not in physical origin.

**How the methylome differs in implementation.** Currently unimplemented. Phase B4 deliverable.

**How it's the same in principle.** Same pattern as age and sex foregrounds when implemented.

**Outputs.** *When implemented:* β_corrected with ancestry component removed.

**Decision points.** Currently: PASS THROUGH.

**Failure modes.** *When implemented:* misinference of ancestry from AIM panel; detected via inconsistency with self-declared ancestry.

**Canonical cross-references.** Roadmap §10.2.2 B4.

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L4 (secondary).

---

### §39. Step 3.5 — Smoking-axis foreground subtraction

**What this step does.** Removes per-CpG smoking-status-specific methylation differences at the β level — the architecturally correct L4 component-separation move for tobacco signal. Tobacco methylates a well-documented set of CpGs (notably AHRR cg05575921 + ~600 cataloged tobacco-associated CpGs per Joehanes 2016 meta-analysis) with effect sizes that persist for years post-cessation and partially recover with cumulative time off tobacco. Without this subtraction, residual smoking signal absorbs into the immune-class A-score and inflates the apparent disease departure.

**Inputs.** β_corrected from §38 + patient's declared smoking_status + smoking_bin (from intake questionnaire q08–q09).

**Atlas reference.** `IAMAtlas_smoking_layer.csv` — per-CpG (α, δ_current_smoker, φ_recency, R², n_samples). Built once via `SmokingAxisForeground.fit()` on the n_hc=601 HC cohort with smoking-status metadata; cached as a frozen runtime artifact. **Module BUILT and layer CSV FIT (v1.2, 2026-06-06 on GSE50660 n=464).**

**Files invoked.** `Biological_Physics/atlas_vault/walther_clinical_runtime/IAM_Cellular_Age/smoking_axis_foreground.py` — `class SmokingAxisForeground` (mirrors `AgeAxisForeground` interface). API: `.load_layer(path)`, `.subtract_from_single_patient(patient_beta, smoking_bin)`.

**The math.** Per CpG i:
> **β_corrected[i] = β_observed[i] − δ_i × indicator_current − φ_i × recency_score**

where δ_i captures the step effect of being a current smoker (vs not) and φ_i captures the recency-graded effect that decays as the patient gets further from quit. Recency score is mapped from `smoking_bin`:
- `never_smoker`: 0.00
- `former_15plus_y`: 0.10
- `former_5_15y`: 0.30
- `former_0_5y`: 0.60
- `current_smoker`: 1.00

Intercept α_i NOT subtracted — preserves per-CpG baseline. For never-smokers, both indicator and recency_score are 0 → no subtraction. For current smokers, full subtraction of both δ_i and φ_i.

**CMB equivalent.** Point-source masking + subtraction — handling a known per-source contamination component (tobacco is a "loud point source" in the methylome the way bright radio sources are in microwave maps).

**How the methylome differs in implementation.** Smoking effect is recency-graded (decays over years post-cessation) — no direct CMB equivalent. The two-coefficient model (δ for current + φ for recency) captures both the step effect and the decay.

**How it's the same in principle.** Identical to age-axis subtraction (§35): per-CpG OLS coefficient frozen at training time + linear subtraction at runtime.

**Outputs.** β_corrected matrix with smoking component removed.

**Decision points.** v1.2 default behavior: PASS THROUGH (no β-level subtraction applied) until `IAMAtlas_smoking_layer.csv` is fit at v1.3. Stage 7 smoking-bin threshold-stratification in `tier_breakpoints.json v1.2` (`tier_by_smoking_bin.elevated_floor_by_bin`) absorbs the bulk effect as interim mitigation: current → ELEVATED floor 1.10; former_0_5y → 1.08; former_5_15y → 1.07; former_15plus_y → 1.05; never → 1.04. Layer CSV FIT 2026-06-06: apply β-level correction, Stage 7 smoking-bin stratification interim mitigation retires.

**Failure modes.** Patient under-reports smoking → under-correction → residual signal contaminates immune A-score. Mitigation: the intake questionnaire q08–q09 ask explicitly, and the report's Quality section discloses smoking-status as a self-reported input.

**Canonical cross-references.** BUILD_SPEC v1.2 §3.3 (module table) + §5 Stage 3 (sequence). Roadmap §10.2.2 B4 (now built; layer fit scheduled v1.3). Reference literature: Joehanes et al. 2016 (Circ Cardiovasc Genet); McCarthy et al. 2017 (AHRR cg05575921 durability); Zeilinger et al. 2013 (KORA 187 lead CpGs).

**CPG Plate references.** Not applicable.

**Chain-link assignment.** L4 (secondary foreground).

---

### §40. Step 3.6 — Stage 3 output: cleaned β matrix

**What this step does.** Packages the foreground-subtracted β matrix into the canonical Stage 3 output: a β matrix with age (and, when modules are implemented, sex/batch/ancestry/smoking) components removed, ready for Stage 4 A-score computation.

**Inputs.** β_corrected from §35-§39.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Stage 3 packager: `<Stage 3 packaging inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** The **foreground-cleaned CMB map** — what Planck publishes as the SMICA/Commander/NILC/SEVEM cleaned products, ready for power spectrum estimation.

**How the methylome differs in implementation.** Per-patient cleaned β matrix rather than full-sky cleaned map.

**How it's the same in principle.** Both represent the cleaned signal of interest, foregrounds-removed.

**Outputs.** A canonical Stage 3 output file containing:
- The cleaned β matrix
- The list of foregrounds subtracted (age, sex [if implemented], batch [if implemented], ancestry [if implemented], smoking [if card requires])
- The per-CpG slopes used for each subtraction (for audit trail)

Location: `<Stage 3 output — emitted internally by `GAPE_WEB_v13.py`>`.

**Decision points.** Handoff to §41 (Step 4.1 Per-class β_mean computation).

**Failure modes.** Output packaging is mechanical.

**Canonical cross-references.** Recipe §4 (end of foreground). Roadmap §10.1.1 L4 row.

**CPG Plate references.** Plate 4 Panel A (Class-Difference Map shows what's visible AFTER foreground subtraction — the differentiation gradient between architectural classes).

**Chain-link assignment.** L4 (closes).

---

## Stage 4 — A-score computation (entropy scoring)

---

### §41. Step 4.1 — Per-class β_mean computation

**What this step does.** For each of the 8 architectural classes, computes the mean β across the class's marker CpGs (~2,978 markers per class for the 8-class scoring). This is the canonical per-class β statistic that feeds the entropy formula.

**Inputs.** Cleaned β matrix from §40 + per-class marker CpG lists from §29.

**Atlas reference.** **IAMAtlas consulted.** Specifically: the per-class marker CpG lists at `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json`. (Markers are AGGREGATED to class — each class's marker pool is the union of its constituent cell types' marker pools.)

**Files invoked.**
- A-scoring module: `pipeline_runtime_matrices/iamatlas_a_scoring.py` (specifically: `score_per_class()`)

**The math.** For each class c:
> **β_mean_c = mean(β_corrected[i]) for i in class_markers[c]**

This is the CANONICAL β_mean per Recipe §6.3 — H is computed on the MEAN β across class markers, NOT the mean of per-CpG entropies. The choice matters: Jensen's inequality means mean(H(β_i)) ≤ H(mean(β_i)), so the two formulas give different numbers. Recipe canonical is H of the mean.

**CMB equivalent.** This is the **bandpower computation** at the per-class level — collapsing many individual measurements (per-CpG β values) into a single class-level statistic (β_mean) the way Planck collapses many per-pixel temperature measurements into a single multipole-bin power.

**How the methylome differs in implementation.** Aggregation is by class membership (not by angular scale).

**How it's the same in principle.** Both reduce many measurements to a single summary statistic that captures the per-component physical quantity of interest.

**Outputs.** A per-class β_mean vector: 8 values (one per class), each in [0, 1].

**Decision points.**
- IF all 8 classes have ≥30 marker CpGs with valid β → proceed to §42
- IF any class has <30 valid markers → flag class with `INSUFFICIENT_CPGS`, set its A-score to NULL, proceed with remaining classes

**Failure modes.**
- *All CpGs failing in one class.* Detected here. Causes that class's A-score to be NULL.
- *Atlas marker list staleness.* If the marker file's SHA-256 doesn't match the expected anchor, refuse to compute (caught upstream at §29).

**Canonical cross-references.** Recipe §6.1 (β_mean), §6.3 (canonical formula). Capability Translator §4.

**CPG Plate references.** Plate 1 (the per-class atlas panels themselves are the visualization of the per-class β reference structure that markers sample).

**Chain-link assignment.** Between L3 (β values) and L4 (decomposition).

---

### §42. Step 4.2 — Shannon entropy H(β_mean) calculation

**What this step does.** For each class's β_mean, computes the binary Shannon entropy.

**Inputs.** Per-class β_mean from §41.

**Atlas reference.** Not consulted at this step.

**Files invoked.** Same module as §41 (`iamatlas_a_scoring.py`).

**The math.** For each class c:
> **H_c = −β_mean_c · log₂(β_mean_c) − (1 − β_mean_c) · log₂(1 − β_mean_c)**

(with the convention 0 · log(0) = 0). H is bounded [0, 1] — maximum at β = 0.5, zero at β ∈ {0, 1}.

**CMB equivalent.** This is the **per-component variance computation** — extracting the information-theoretic content of the class-level signal. CMB cosmology uses the C_ℓ power spectrum (variance per multipole) as its primary information-theoretic quantity; CPG uses entropy per class as its analog. Both quantify "how much information is encoded in this component."

**How the methylome differs in implementation.** Entropy of a Bernoulli probability rather than variance of a Gaussian — appropriate because methylation β is a probability (fraction methylated), not a continuous-valued temperature.

**How it's the same in principle.** Both compute an information-theoretic quantity that captures the per-component signal content.

**Outputs.** A per-class entropy vector: 8 values, each in [0, 1].

**Decision points.** Proceed to §43.

**Failure modes.**
- *β_mean = 0 or β_mean = 1 (boundary cases).* H = 0 by convention. Detected and propagated; downstream A-score will be 0 for that class, which is itself a measurement (extreme commitment to one methylation state).

**Canonical cross-references.** Recipe §6.2 (Shannon entropy). Roadmap §7 (Bidirectional calibration solved by entropy space).

**CPG Plate references.** Plate 3 (the bimodal vs Gaussian texture comparison — the methylome's non-Gaussianity is what makes the entropy framework powerful).

**Chain-link assignment.** Between L3 and L4.

---

### §43. Step 4.3 — Per-class A-score: A = H(β_mean) / H_min(class)

**What this step does.** Divides each class's entropy by the class's H_min (the Mahaffey Number) to produce the canonical A-score: a dimensionless ratio measuring departure from the architectural floor.

**Inputs.** Per-class entropy from §42 + H_min values for each class.

**Atlas reference.** **H_min values consulted** from `IAMAtlasREBUILD_provenance.json` (one source of truth, frozen 2026-04-06).

**Files invoked.** Same module as §41 (`iamatlas_a_scoring.py`).

**The math.** For each class c:
> **A_c = H_c / H_min(c)**

The Mahaffey Numbers (frozen 2026-04-06):
- terminal: H_min = 0.7728
- immune: H_min = 0.838889
- secretory: H_min = 0.843264
- cycling: H_min = 0.856055
- progenitor: H_min = 0.852216
- stromal: H_min = 0.86295
- stem_adult: H_min = 0.873718
- stem_pluri: H_min = 0.982166

**A_c < 1** means below the floor (suppressed below baseline order).
**A_c = 1** means at the floor (architectural minimum).
**A_c > 1** means above the floor (departure from baseline — disorder above the minimum, which is what disease drives).

**CMB equivalent.** This is **normalizing the per-component variance against a known physical scale** — like Planck computing Ω_class = ρ_class / ρ_critical, where ρ_critical is the critical density (a fixed physical constant). The dimensionless ratio carries more information than the raw variance because it's calibrated against a meaningful reference.

**How the methylome differs in implementation.** The reference (H_min) is the architectural floor of each class; ρ_critical is the cosmological scale for a flat universe.

**How it's the same in principle.** Both produce dimensionless physical ratios — direct, interpretable, calibrated against a known anchor — rather than raw variance.

**Outputs.** A per-class A-score vector: 8 values. Each value is a dimensionless ratio.

**Decision points.** Proceed to §44.

**Failure modes.**
- *H_min file out of date.* If `IAMAtlasREBUILD_provenance.json` SHA doesn't match the engine's expected hash, refuse to run.
- *Class entropy NULL (from §42).* A-score is NULL for that class; propagates downstream.

**Canonical cross-references.** Recipe §6.3 (canonical A-score formula). Capability Translator §4 (A-score plain language). Roadmap §4 (What physics brings).

**CPG Plate references.** All four plates (the A-score derivation underlies the visualization of departure from the architectural floor across the sphere).

**Chain-link assignment.** Between L3 and L4.

---

### §44. Step 4.4 — Per-cell-type A-score (115 cell types)

**What this step does.** Same operation as §41-§43 but at finer granularity — 115 cell types rather than 8 architectural classes. The cell type's H_min is looked up via its class membership (each cell type inherits its class's H_min). This produces the per-cell-type A-score surface that the Mahalanobis distance (§47-§50) operates on.

**Inputs.** Cleaned β matrix from §40 + per-cell-type marker CpG lists from `iamatlas_celltype_markers_v0_2.json` + the `celltype_to_class` mapping from the same artifact.

**Atlas reference.** **IAMAtlas consulted.** Per-cell-type markers + class assignment from `pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json`. H_min values still read from `IAMAtlasREBUILD_provenance.json`.

**Files invoked.** `iamatlas_a_scoring.py` (specifically: `score_per_celltype()`).

**The math.** For each cell type ct (115 total):
1. Compute β_mean_ct = mean(β over the cell type's ~100 marker CpGs)
2. Compute H_ct = Shannon entropy of β_mean_ct
3. Look up class assignment: c = celltype_to_class[ct]
4. Compute A_ct = H_ct / H_min(c)

The class assignment is necessary because H_min is per-class (the architectural floor lives at the class level, not the cell-type level — many cell types share the same architectural identity).

**CMB equivalent.** This is **per-narrow-band power spectrum estimation** — extracting the signal at finer angular scales than the headline per-multipole-bin estimates. CMB cosmologists often compute both broad ℓ-binned and narrow ℓ-binned power spectra for different inference targets.

**How the methylome differs in implementation.** 115 cell types vs many possible ℓ-binnings in CMB.

**How it's the same in principle.** Both produce a finer-grained decomposition of the signal alongside the headline per-class output.

**Outputs.** A per-cell-type A-score vector: 115 values + per-cell-type confidence + status codes (same structure as the per-class output).

**Decision points.** Proceed to §45.

**Failure modes.**
- *Cell types with insufficient markers in patient β.* Common for rare cell types where many markers failed QC. Status code `INSUFFICIENT_MARKERS` set; A-score NULL; downstream Mahalanobis imputes via median (with the imputation count tracked).

**Canonical cross-references.** Recipe §6.4 (Per-cell-type A-score). Capability Translator §6. Roadmap §10.2.1 TODO 1.1 (per-cell-type one-vs-rest).

**CPG Plate references.** Plate 2 (per-CpG residual at marker level — the cell-type-specific markers are what carry the disease-residual signal).

**Chain-link assignment.** Between L3 and L4.

---

### §45. Step 4.5 — Disease panel A-score (when card has a curated panel)

**What this step does.** When a specific disease card includes a curated CpG panel (e.g., the Xu-538 immune panel used for breast pre-dx), compute the A-score on that panel using the same H(β)/H_min formula but restricted to the panel's CpGs and using the class H_min that the panel anchors to.

**Inputs.** Cleaned β matrix from §40 + the card's panel CpG list + the class H_min the panel anchors to.

**Atlas reference.** Indirectly — the H_min anchor comes from IAMAtlas. The panel itself is card-specific (defined by the card author per the card spec).

**Files invoked.**
- Same module: `iamatlas_a_scoring.py`
- Card-specific panel artifact: e.g., `<card-specific panel CpG list — currently embedded in card config inside `GAPE_WEB_v13.py`>`

**The math.** Same as §43, but:
- β_mean_panel = mean(β_corrected[i] for i in panel_cpgs)
- A_panel = H(β_mean_panel) / H_min(class_anchor)

For the Xu-538 panel anchored to immune class: A_panel = H(β_mean over 538 CpGs) / 0.838889.

**CMB equivalent.** This is **targeted-scale analysis** — running a power spectrum on a pre-selected multipole window known to be informative for a specific cosmological question (e.g., the recombination acoustic peak window for sound-horizon measurement).

**How the methylome differs in implementation.** Panels are typically card-specific and disease-specific (e.g., Xu-538 is breast-pre-dx-specific). CMB targeted windows are typically question-specific (e.g., the ℓ ≈ 220 window for sound horizon).

**How it's the same in principle.** Both compute the same core physical quantity (power / entropy) restricted to a pre-selected informative subset of the data.

**Outputs.** A per-card disease panel A-score: 1 value per card per patient.

**Decision points.**
- IF card has a curated panel → compute panel A-score, proceed to §46
- IF card uses only class/cell-type A-scores → SKIP, proceed to §46

**Failure modes.**
- *Panel CpGs not present in patient's IDAT.* Detected here; falls back to class-level A-score if panel coverage is too low.

**Canonical cross-references.** Recipe §6.5 (Panel A-scores). Capability Translator §7. Roadmap §10.2.5 E1 (per-card likelihoods include panel A-scores).

**CPG Plate references.** Plate 2 (the breast pre-dx panel signal visualized on the methylome sphere).

**Chain-link assignment.** Between L3 and L4 (panel-level).

---

### §46. Step 4.6 — Stage 4 output

**What this step does.** Packages all A-scores (per-class, per-cell-type, per-card panel) into the canonical Stage 4 output.

**Inputs.** A-scores from §43-§45.

**Atlas reference.** Not consulted at this step.

**Files invoked.**
- Stage 4 packager: `<Stage 4 packaging inside `GAPE_WEB_v13.py`>`

**The math.** None. Structural composition.

**CMB equivalent.** The **full multi-scale power spectrum package** — per-multipole, per-band, per-target-window — handed off to inference.

**How the methylome differs in implementation.** Per-patient package rather than per-mission spectrum.

**How it's the same in principle.** Both package the complete set of physical measurements ready for inference.

**Outputs.** A canonical Stage 4 output file containing:
- 8 per-class A-scores + confidence + status
- 115 per-cell-type A-scores + confidence + status
- Per-card panel A-scores (one per applicable card)
- The H_min values used (audit trail)
- The atlas SHA-256 fingerprint (audit trail)

Location: `<Stage 4 output — emitted internally by `GAPE_WEB_v13.py`>`.

**Decision points.** Handoff to §46.5 (Stage 4.5 bidirectional decomposition).

**Failure modes.** Output packaging is mechanical.

**Canonical cross-references.** Recipe §6 (end of A-score computation). Capability Translator §4-§7.

**CPG Plate references.** Plate 4 Panel F (the per-CpG breast-anisotropy signal at the panel level).

**Chain-link assignment.** Between L3 and L4.

---

## Stage 4.5 — Bidirectional decomposition (NEW v1.3 / L4 cont.)

### §46.5. Step 4.5.1 — Bidirectional pattern detection

**What this step does.** Decomposes each class's signal into a signed directional composite that catches bidirectional methylation patterns at patient runtime. The pooled-entropy A-score from §43 is **direction-agnostic** because Shannon entropy is symmetric around β=0.5. When a disease produces a bidirectional pattern (some CpGs going UP, others going DOWN), the pooled β_mean barely moves and the pooled A-score reads NULL — the directional opposites cancel. This step recovers the cancelled signal at patient runtime.

**Why this stage exists.** The VAL-050 → VAL-051 lesson made the cancellation visible: pooled-entropy A returned d=+0.077 (null) on the 18-CpG IMM panel applied to AIBL AD vs HC; the directional weighted composite z-score on a 7-CpG sub-panel returned d=+0.624 (recovery) on the SAME cohort. At validation time, every VAL has a PREREG specifying direction. At patient runtime, there's no PREREG per patient — the engine MUST decompose autonomously.

**Inputs.** Stage 4 output (per-class A-scores) + foreground-cleaned β vector from Stage 3 + the frozen directional panels artifact.

**Atlas reference.** `directional_panels_v1_0.json` at `walther_clinical_runtime/Bidirectional_Decomposition/`. Schema: per-class panels with CpG-level (cpg_id, direction±1, mean_hc_train, sd_hc_train) + the pooled-entropy parent panel CpG list. **v1.0 coverage: immune class only** (VAL-051 Rule A 7-CpG AD-direction-anchored panel, SHA-anchored to sealed `val051_panel_ruleA.json` SHA-256 `52061285...`). 7 other classes return `NO_PANEL` honestly until future sealed VALs populate them. The immune-class pooled-entropy comparator uses the 18-CpG VAL-050 IMM_CPGS_EPIC parent panel.

**Files invoked.** `Biological_Physics/atlas_vault/walther_clinical_runtime/Bidirectional_Decomposition/bidirectional_decomposition.py` — mirrors the sealed `val051_analyze.py:112-121` `a_dir_score` formula exactly. Public surface: `load_directional_panels`, `score_directional_composite`, `score_pooled_entropy`, `bidirectional_flag`, `compute_per_class_bidirectional_decomposition`, `save_bidirectional_report`.

**The math.** For each panel CpG:
> **z_i = (β_patient[i] − mean_hc_train[i]) / sd_hc_train[i]**
> **contrib_i = direction_i × z_i**

Where direction_i is +1 (up in disease) or −1 (down in disease), frozen at VAL training time. The directional composite is then:
> **a_directional = mean(contrib_i over covered CpGs)**

Coverage gate: require `n_covered ≥ max(3, 0.7 × n_panel)`; below this, a_directional returns None (INSUFFICIENT_COVERAGE).

Pooled-entropy comparator (mirrors val051_analyze.py:123-128):
> **a_pooled = H(β_mean over parent panel) / H_min(class)**

The bidirectional flag fires when:
> **FLAG_BIDIRECTIONAL = (|a_pooled − 1.0| < 0.05) AND (|a_directional| > 0.40)**

In English: pooled is mute (near baseline) AND directional is loud (above effect-size threshold). Both required.

**CMB equivalent.** Polarization decomposition — the CMB's total intensity I doesn't tell you which sources produced it, but decomposing into Q + U polarization recovers directional information about the underlying physics. Here pooled-entropy A is the "I" (direction-agnostic); the directional composite is the "Q" (signed projection along the disease direction).

**How the methylome differs in implementation.** Direction is frozen per VAL panel, not computed at runtime. The decomposition projection is along a pre-specified disease axis, not a free spherical-harmonic decomposition. This is appropriate: at patient runtime we have only one observation (one patient), so there's no statistical power to estimate direction from the data — the direction must come from the validated panel.

**How it's the same in principle.** Both decompose a scalar (intensity / pooled-entropy A) into a signed vector (Q+U / directional composite) by projecting against a known basis (polarization axes / disease panel directions). Both recover information that the scalar discards.

**Outputs.** `BidirectionalReport` per patient:
- Per-class `BidirectionalResult` (a_pooled_entropy, a_directional_composite, n_covered, coverage_fraction, flag_bidirectional, flag_insufficient_coverage, interpretation string)
- Aggregate `any_bidirectional_flagged` boolean + `flagged_classes` list

Location: `reports/{patient_id}/stage_4_5/{patient_id}_stage_4_5_bidirectional_decomposition.json`.

**Decision points.**
- If `flag_bidirectional == True` for any class: Stage 7 (§59) uses the directional composite (signed magnitude) rather than the pooled A-score to drive tier reporting for the flagged class. Stage 8 Route C-bidirectional activates per the relevant card.
- If `flag_insufficient_coverage == True`: pooled A from Stage 4 remains valid; directional read is omitted from the report with an audit note.
- If `NO_PANEL` (7 classes that lack sealed directional panels): Stage 4 pooled-entropy A is the only A-score reported for those classes.

**Failure modes.**
- Panel CpGs not present in patient β (low coverage) → INSUFFICIENT_COVERAGE.
- Panel JSON SHA mismatch vs sealed VAL anchor → STAGE_4_5_PANEL_DRIFT (engine refuses to score).
- Directional panel inverted by accident (signs flipped) → would show as systematic anti-direction across all flagged patients (operational integrity check).

**Canonical cross-references.** BUILD_SPEC v1.2 §5 Stage 4.5. Bidirectional_Decomposition/README. Sealed VAL artifacts at `Biological_Physics/validation_runs/val_051_ad_directional/`.

**CPG Plate references.** None directly. (Future: Plate 5 could visualize the patient's per-CpG directional contributions on the Mollweide grid.)

**Chain-link assignment.** L4 (the directional refinement of L4 component separation — analogous to the polarization decomposition added to CMB analyses after the initial I-only maps).

---

## Stage 4.6 — Patient brightness comparison (NEW v1.3 / L4 cont.)

### §46.6. Step 4.6.1 — Per-class z-score departure + Mollweide projection

**What this step does.** Computes per-CpG z-score departure of the patient β from each of 8 frozen healthy class brightness references, then projects the departure pattern onto a HEALPix NSIDE=128 grid for Mollweide rendering. Produces the patient's personal Cosmic Microwave Methylome — the customer-facing analog of CPG Plate 1.

**Why this stage exists.** Stage 4's A-score collapses each class's signal into a single scalar. The customer report shows this scalar as a tier (NORMAL / ELEVATED / etc.), which is useful for triage but loses the spatial structure of WHERE the departure lives. The patient brightness comparison preserves the spatial structure by projecting the per-CpG departure pattern onto the same HEALPix grid as Plate 1 (the framework's reference Cosmic Microwave Methylome). The customer sees their personal map next to the reference and can see the anisotropy directly.

**Inputs.** Foreground-cleaned β vector from Stage 3 + class brightness references (8 brightness CSVs, one per class) + the canonical CpG-to-HEALPix mapping.

**Atlas reference.**
- 8 class brightness CSVs at `IAMAtlas_v0_1/class_archives/{class}.tar.xz` (inner `{class}/iamatlas_v0_1_{class}_brightness.csv`); each has per-CpG mean β + SD β over the healthy class reference + per-CpG MCMC posterior CI.
- `iamatlas_cpg_to_healpix_nside128.npy` at `IAMAtlas_v0_1/healpix_mapping/` (1.93 MB, 483,092 entries, int32 pixel indices in atlas row order). 450,192 CpGs annotated to real HEALPix pixels; 32,900 CpGs (HM450-only probes not in EPIC manifest) mapped to sentinel pixel that renders as the framework's galactic mask analog.
- CPG Plate 1 at `IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Microwave_Methylome.png` — the binding contract for the projection grid.

**Files invoked.** `Biological_Physics/atlas_vault/walther_clinical_runtime/Brightness_Comparison/patient_brightness_comparison.py`. Public surface: `load_all_8_class_references`, `compute_all_8_class_departures`, `render_patient_cosmic_methylome`, `save_brightness_report`.

The HEALPix mapping is generated one time per atlas version by `Biological_Physics/atlas_vault/IAMAtlas_v0_1/healpix_mapping/generate_cpg_healpix_mapping.py`. Production mapping is committed; the generator script exists so the build is reproducible from inputs (IAMAtlas REBUILD CSV + EPIC v1 B4 manifest) without manual intervention.

**The math.** Per class C and per CpG i in class C's covered set:
> **z_i^C = (β_patient[i] − mean_class_β_C[i]) / sd_class_β_C[i]**

Per pixel p (aggregating over CpGs that map to p):
> **z_pixel_p^C = mean(z_i^C over CpGs i with cpg_to_pixel[i] = p)**

Empty pixels (no CpGs map there) and sentinel pixels (HM450-only probes) render as the galactic-mask analog (BLACK in the Mollweide).

**CMB equivalent.** This is the patient's personal CMB anisotropy map. Plate 1 is the reference universe; this is the patient's universe; the difference between them is the personal anisotropy.

**How the methylome differs in implementation.** Sphere is a representation choice (HEALPix is convenient, equal-area, full-sky), not a physical sphere. The methylome doesn't actually live on a sphere — the spatial structure comes from chromosomal position + per-chromosome ordering by MAPINFO. The Mollweide projection is the chosen visual representation; it could equally be a flat 2D rectangle, but Mollweide preserves the cosmic-resonance framing that anchors the customer's intuition.

**How it's the same in principle.** Both are anisotropy maps on a HEALPix-projected sphere; both use NSIDE=128 (Plate 1 inherits the convention from CMB literature); both produce visual signatures that match the statistical properties of CMB temperature anisotropies in spite of the substrates being completely different.

**Outputs.** `PatientBrightnessReport` per patient:
- Per-class HEALPix array of z-pixel departures (8 arrays, one per class)
- Patient Mollweide PNG (rendered alongside the Plate 1 reference for direct visual comparison)
- Aggregate departure statistics per class (max z-departure, fraction of pixels above ±2σ, anisotropy spectrum)

Location: `reports/{patient_id}/stage_4_6/{patient_id}_personal_cosmic_methylome.png` + companion JSON.

**Decision points.**
- Patient Mollweide rendered for all 8 classes regardless of tier — even NORMAL-tier classes carry visually informative anisotropy.
- The customer report's "your personal map" section uses the 1-2 classes with the largest departure as headline.
- Audit-trail JSON carries all 8 maps even when only some are highlighted.

**Failure modes.**
- Patient β coverage low (<60% of brightness reference CpGs) → STAGE_4_6_LOW_COVERAGE flag; Mollweide rendered but with coverage warning watermark.
- HEALPix mapping SHA mismatch → STAGE_4_6_GRID_DRIFT (engine refuses to render).
- Brightness CSV not found in expected path → STAGE_4_6_REFERENCE_MISSING.

**Canonical cross-references.** BUILD_SPEC v1.2 §3.5b (CPG Plates) + §5 Stage 4.6. Brightness_Comparison/README. HEALPix mapping README at `IAMAtlas_v0_1/healpix_mapping/README_HEALPix_Mapping.md`.

**CPG Plate references.** Plate 1 (the framework's reference Cosmic Microwave Methylome) is the visual benchmark this stage produces a patient-specific analog of.

**Chain-link assignment.** L4 (the spatial-structure preservation of L4 component separation — analogous to keeping the full anisotropy map rather than collapsing to a power spectrum).

---

*End of Stages 0-4 (including Stages 4.5 and 4.6, both new in v1.3). Continued in Part II Stages 5-10 + Parts III-V.*


> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them. Where a step's logic currently lives inside `GAPE_WEB_v13.py` (the
> production engine — see `SYSTEM_INVENTORY.md`), the SOP now says so honestly. Where
> a step's runtime artifact or output file location was invented, the SOP now reads
> "TBD per orchestrator design" — that orchestrator (working name `web.commercial.py`)
> is a separate conversation Heath and Walther will have. All real paths in this v1.1
> are documented in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be
> treated as not-yet-existing until verified against the repo.

---


**Continues from Part II-A (§11-§46). This document contains §47-§64 — Stages 5 (Mahalanobis), 6 (cellular age inversion), 7 (tier breakpoints).**

---

# Stage 5 — Multi-D departure (Mahalanobis hyper-volume)

Stage 5 produces the **single headline number** every EDEAR report carries: the Mahalanobis distance of the patient's 115-cell-type A-score vector from the pooled healthy-cohort centroid, weighted by the covariance structure of IAM's own feature space. This is the methylome's joint-posterior-banana measurement — the multi-dimensional analog of CMB cosmology's joint cosmological-parameter ellipsoid (Ωm vs ΩΛ vs σ_8 etc.).

The cleanest framing: the per-class A-score (Stage 4) answers "how far is each class from H_min." The Mahalanobis distance answers "how far is the patient from healthy in the full 115-D A-score manifold." Both are IAM measurements at different granularities. The Mahalanobis distance gives clinicians one calibrated number — and the top-10 axis decomposition makes that number explainable.

---

## §47. Step 5.1 — Patient 115-cell-type A-score vector assembly

**What this step does.** Takes the patient's 115 per-cell-type A-scores from Stage 4 (§44) and assembles them into a single 115-element feature vector ready for Mahalanobis distance computation. Handles per-cell-type imputation when QC flagged some cell types as `INSUFFICIENT_MARKERS`.

**Inputs.** Per-sample per-cell-type A-scores + status codes from <A-score output — emitted by `iamatlas_a_scoring.py` (real module per inventory)>.

**Atlas reference.** Indirect — the 115 cell-type ordering is fixed by the atlas's celltype_to_class mapping.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — class `MahalanobisHealthyHull`, method `_assemble_patient_vector(per_celltype_a)`.

**The math.** Order the 115 cell types in canonical sequence (defined in the current production Mahalanobis reference; v0_3 uses 112 valid features with 3 dropped: Cortical_neurons, Glia, stem_pluri). For each cell type:
- If status `OK`: use the A-score as-is.
- If status `MARGINAL_COVERAGE`: use the A-score but flag for downgraded confidence at output.
- If status `INSUFFICIENT_MARKERS` or `NO_MARKER_OVERLAP`: impute using the cohort-pooled-HC mean for that cell type (from the Mahalanobis reference centroid).

The imputation count is tracked separately. Patients with > 5 imputations get a `PARTIAL_DATA` status at output.

**CMB equivalent.** **Cosmological parameter vector assembly before joint-posterior evaluation.** Planck's cosmological inference operates on a joint vector of parameters (Ωm, Ωb, h, n_s, σ_8, τ, etc.). Before evaluating the joint posterior or computing tension with other surveys, the parameter vector is assembled in canonical order with missing parameters imputed from priors. The methylome's 115-element A-score vector is the methylome's parameter vector for joint-posterior evaluation.

**How the methylome differs in implementation.** 115 dimensions vs ~7 cosmological parameters. The high dimensionality is why Mahalanobis (which accounts for covariance) is necessary — naive 115-D Euclidean distance would treat correlated features as independent and inflate apparent distance.

**How it's the same in principle.** Joint-posterior evaluation requires a single feature vector in a defined coordinate system. Methylome and CMB cosmology share that architectural requirement.

**Outputs.** Per-patient 115-element A-score vector + per-cell-type imputed-flag mask + `n_features_imputed` count.

**Decision points.**
- ≤ 5 imputations → proceed with status `OK` to §48.
- 6-15 imputations → proceed with status `PARTIAL_DATA` flag.
- > 15 imputations → flag patient as `INSUFFICIENT_DATA` for Mahalanobis; do not return Mahalanobis distance; downstream report omits the headline number with explanation.

**Failure modes.**
- Cohort-wide high imputation rate (> 10% of patients > 5 imputations) → indicates substrate-atlas mismatch; flag cohort.

**Canonical cross-references.**
- Recipe §6.4 (Mahalanobis specification).
- Roadmap §3.13 (multi-dimensional hyper-volume).

**CPG Plate references.** None at this granularity.

**Chain-link assignment.** L6 (covariance modeling enters) + L8 (parameter inference).

---

## §48. Step 5.2 — HC centroid load (`mahalanobis_healthy_reference_v0_5.json` current production; v0_1/v0_2/v0_3/v0_4 retained for lineage)

**What this step does.** Loads the pre-computed healthy-control reference object — the centroid (mean) and inverse-covariance matrix in 115-cell-type A-score space — into memory. This is the **calibrated healthy reference** the patient's vector will be measured against. Loaded once per session at engine startup; pinned in memory thereafter.

**Inputs.** Current production: `mahalanobis_healthy_reference_v0_5.json` (n_hc=2,523 from GSE51057 + GSE51032 + GSE40279 Hannum + GSE50660 Tsaprouni + GSE144858 AddNeuroMed + GSE153712 AIBL + GSE53740 GIFT + GSE141682 Han Chinese — 8 cohorts, 4 populations). Engine loads whichever version is named in BUILD_SPEC §3.4b. Prior versions (v0_1 n=601, v0_2 n=1,257) retained in same folder for traceability.

**Hull versioning protocol (BUILD_SPEC §Stage 5.1).** The hull is the only chain element with cohort-empirical content — centroid + covariance MUST be measured from HC samples, not physics-derived. Versions extend HC representation:
- v0_1 (foundation): GSE51057 + GSE51032 EPIC-Italy women 40-65 HM450, n=601.
- v0_2 (Phase 1, 2026-06-06): +GSE40279 Hannum US M/F 19-101 HM450, n=1,257.
- v0_3 (Phase 2, 2026-06-06): +GSE50660 Tsaprouni UK M/F 40-65 HM450 smoking-stratified, n=1,721.
- v0_4 (Phase 3, 2026-06-06): +GSE144858 AddNeuroMed n=96 + GSE153712 AIBL n=471 EPIC (FIRST CROSS-PLATFORM) + GSE53740 GIFT n=193, n=2,481.
- v0_5 (Phase 4, 2026-06-06): +GSE141682 Han Chinese n=42 EPIC (FIRST ASIAN POPULATION), n=2,523.: +EPIC platform HC for cross-platform transferability.

Build versions never rebuild on patient β. Patient runtime queries the FROZEN current production version.

**Atlas reference.** The reference object is built from IAMAtlas-A-scored healthy controls. **The reference itself IS the atlas's HC posture in 115-cell-type A-score space** — it is not an external comparison atlas.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — `MahalanobisHealthyHull(reference_path)`.

**The math.** The reference carries:
- `centroid`: 112-element mean vector (pooled-HC mean A-score per cell type; 3 features dropped: Cortical_neurons, Glia, stem_pluri).
- `covariance_matrix`: 112×112 covariance matrix (Ledoit-Wolf shrinkage — v0_5 shrinkage 0.002202 (v0_4: 0.002208, v0_3: 0.001317), down from v0_1's 0.008751 because more samples reduce regularization need).
- `feature_names_valid`: canonical 112-element ordering matching §47.
- `n_hc_samples_pooled`: 2,523 (v0_5 calibration sample size).
- `hc_cohort_sources`: list of contributing cohorts with sample counts + platform + demographic span.
- `route_A_calibration_v0_5`: percentile thresholds (p95 = 13.62 default, p99 = 18.43 strict) calibrated against the pooled-HC distance distribution — Route A trigger is percentile-of-HC, NOT a fixed value.
- `expansion_lineage_v0_1_to_v0_3`: documents what each version added.
- `validation_v0_3`: Hannum HC self-test + foundation HC self-test + breast pre-dx anchor preservation (Cohen's d lineage GSE51057: 1.871 → 0.981 → 0.896 → 0.593 → 0.599; GSE51032: 2.088 → 1.653 → 1.611 → 1.450 → 1.450).
- `input_data_provenance`: SHA-256 of every input CSV.

Loaded with SHA-256 verification. If hash mismatches the pinned value, halt and investigate.

**Critical calibration note.** v0_1 used a fixed `d ≥ 2.0` Route A threshold which was mathematically inappropriate for 112-dim data (expected median d under multivariate normality is √112 ≈ 10.58 — fixed threshold would fire on all samples). v0_2 corrected this with percentile-of-HC thresholds; v0_3 continues that protocol. Engine reads thresholds from the artifact's `route_A_calibration_v0_N` block at session startup. **Cards do not carry threshold values** — they reference the artifact path only.

**CMB equivalent.** **Cosmological-parameter covariance matrix from chain analysis.** Planck publishes the cosmological-parameter posterior chain along with a derived covariance matrix used for tension calculations with other surveys. The methylome's HC reference object is structurally identical: a centroid (mean parameter vector) + covariance matrix derived from the calibration set. The hull versioning protocol is structurally analogous to Planck's data-release versioning (Planck 2013 → 2015 → 2018) — each release used a fixed pre-computed covariance matrix for cosmological analysis; new releases expanded the calibration data; prior releases retained for lineage.

**How the methylome differs in implementation.** Methylome covariance is over IAM-A-score features; CMB covariance is over cosmological parameters. Different feature spaces; same operational structure.

**How it's the same in principle.** A multi-dimensional reference requires a centroid AND a covariance to make distance computation physically meaningful. Both modalities pin these as fundamental reference objects. Both maintain frozen versions tagged to scientific releases.

**Outputs.** In-memory `MahalanobisHealthyHull` object accessible to all subsequent per-patient Mahalanobis calculations.

**Decision points.**
- Reference hash matches → proceed.
- Hash mismatches → HARD HALT.

**Failure modes.**
- File missing or corrupted → halt; canonical reference must be present.

**Canonical cross-references.**
- Recipe §6.4.
- Runtime Matrices README §"Stage 2.5" (Mahalanobis specification).

**CPG Plate references.** None.

**Chain-link assignment.** L6 (covariance loaded).

---

## §49. Step 5.3 — Inverse-covariance distance computation

**What this step does.** Computes the **Mahalanobis distance** of the patient's 115-element A-score vector from the HC centroid in the inverse-covariance-weighted metric. This is the single headline number.

**Inputs.** Patient vector from §47 + HC reference object from §48.

**Atlas reference.** Indirect.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — `MahalanobisHealthyHull.score(patient_vector)`.

**The math.** Per patient:
```
δ = patient_vector - hc_centroid               (115-element residual)
mahalanobis_distance = sqrt( δᵀ · Σ⁻¹ · δ )      (scalar)
```
where `Σ⁻¹` is the inverse-covariance matrix from §48.

**Why this is the right metric:** Naive Euclidean distance treats all 115 dimensions as independent and identically scaled. In reality, cell-type A-scores are highly correlated (immune-class cell types correlate with each other; secretory-class cell types correlate with each other; etc.). Euclidean would over-count a patient who departed along a correlated direction. Mahalanobis re-scales the 115-D space so that all directions are equivalently weighted accounting for the covariance, producing a statistically interpretable distance.

**Typical ranges (from EPIC-Italy validation):**
- HC: distance ~ 6-12.
- Cases (breast pre-dx >10yr): distance ~ 10-20.
- Cohen's d on breast pre-dx vs HC: +1.871 (GSE51057), +2.088 (GSE51032). Beats Xu-538 by +0.752 on GSE51032 without being breast-trained.

**CMB equivalent.** **Joint-posterior tension distance between Planck and an external survey.** When Planck and an external survey (e.g., DES, SDSS-Y3) both report cosmological parameters, the tension between them is computed as the Mahalanobis distance in the joint parameter space using the combined covariance. The methylome's distance from the HC reference is the analog: a single number summarizing multi-dimensional departure under the appropriate covariance metric.

**How the methylome differs in implementation.** Single-patient vs cohort-survey-vs-cohort-survey. Different sample sizes feeding the metric; same metric.

**How it's the same in principle.** Mahalanobis distance is the canonical multi-dimensional departure metric in both cosmology and biology. The math doesn't care which substrate it's computed on.

**Outputs.** Per-patient scalar `mahalanobis_distance` value.

**Decision points.**
- Distance computed → proceed to §50 for axis decomposition.
- Numerical failure (rare, e.g., non-positive-definite covariance from extreme imputation) → flag with `MAHALANOBIS_NUMERIC_FAIL`; report omits the distance with note.

**Failure modes.**
- Negative δᵀ · Σ⁻¹ · δ (mathematically impossible for valid Σ⁻¹) → indicates corrupted reference; halt.

**Canonical cross-references.**
- Recipe §6.4.
- VAL-002 (Mahalanobis hyper-volume sealed, d=+1.876/+2.097).

**CPG Plate references.** Plate 2 (Breast Pre-Diagnostic Anisotropy) — the 1,392 concordant CpGs that compose the pre-dx signature; the Mahalanobis distance captures their joint departure in 115-cell-type A-score space.

**Chain-link assignment.** L8 (parameter inference / posterior — partial).

---

## §50. Step 5.4 — Top-10 axis contribution decomposition

**What this step does.** Decomposes the patient's Mahalanobis distance into per-cell-type contributions. **A single distance number is uninterpretable without knowing which directions in the 115-D space drove it.** Step 5.4 produces the top-10 cell types whose departures most contributed to the patient's distance.

**Inputs.** Patient vector + HC centroid + inverse-covariance + computed Mahalanobis distance from §49.

**Atlas reference.** Indirect.

**Files invoked.** `iamatlas_mahalanobis_scoring.py` — `MahalanobisHealthyHull._decompose_axes(patient_vector)`.

**The math.** For each cell type `j`:
```
z_shift[j] = (patient_vector[j] - hc_centroid[j]) / sqrt(diag(Σ)[j])
contribution[j] = z_shift[j] × Σ⁻¹·δ [j]
```
Sort by `|contribution|`, take top 10. Each top-axis entry reports:
- Cell type name.
- Patient's A-score for that cell type.
- HC centroid's A-score for that cell type.
- Z-shift (how many SD departed).
- Sign of departure (+ = elevated; − = suppressed).
- Per-axis contribution to the total Mahalanobis distance.

**Why this matters clinically:** The customer report can say "your Mahalanobis distance is 14.2, with the biggest contributions from basophils (+2.1 z, elevated), plasma cells (+1.8 z, elevated), and microglia (+1.6 z, elevated) — your immune compartments are most departed from healthy in your sample." That's a statistically interpretable, mechanistically explainable headline.

**CMB equivalent.** **Decomposition of cosmological-tension into per-parameter directions.** When Planck and DES disagree, the disagreement can be decomposed into directions in parameter space: "the tension is along the S_8 axis, not the H_0 axis." The methylome's top-10 axis decomposition is the same operation: tell the reader which axes in the high-D feature space drove the headline number.

**How the methylome differs in implementation.** 115 cell types as axes; ~7 cosmological parameters as axes. Methylome decomposition is naturally a top-K reporting because of the higher dimensionality.

**How it's the same in principle.** A single distance number is half a measurement. The full measurement requires knowing the direction. The decomposition is the direction.

**Outputs.** Per-patient `top10_axis_contributions` list (each entry as above).

**Decision points.** None — pure decomposition.

**Failure modes.**
- All top contributions from imputed cell types → flag report as "axis decomposition unreliable due to high imputation rate."

**Canonical cross-references.**
- Recipe §6.4.
- Roadmap §3.15 (parameter dependencies / cross-class correlations as axis decomposition).

**CPG Plate references.** Plate 4 Panel A (Class-Difference Map) shows which class-direction axes dominate the methylome — the per-class equivalent of the per-cell-type axis decomposition.

**Chain-link assignment.** L8.

---

## §51. Step 5.5 — Stage 5 output: Mahalanobis distance + per-axis explainability

**What this step does.** Consolidates Stage 5 outputs into the per-cohort artifact.

**Inputs.** Outputs of §49 + §50.

**Atlas reference.** None (consolidation).

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Published joint-posterior tension table with per-axis decomposition.

**How the methylome differs in implementation.** Per-patient row; same content per row as a per-survey-pair row.

**How it's the same in principle.** Multi-D measurement summarized as scalar + decomposition.

**Outputs.** <Mahalanobis output — emitted by `iamatlas_mahalanobis_scoring.py` (real module per inventory)> carrying:
- `mahalanobis_distance` per patient
- `n_features_used` per patient
- `n_features_imputed` per patient
- `status` (OK / PARTIAL_DATA / INSUFFICIENT_DATA / MAHALANOBIS_NUMERIC_FAIL)
- `top10_axis_contributions` per patient (list of dicts)

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 6 (§52 — cellular age inversion).

**Failure modes.** None at consolidation.

**Canonical cross-references.**
- Recipe §6.4.
- Roadmap §3.13.
- VAL-002.

**CPG Plate references.** Plate 2 (Breast Pre-Diagnostic Anisotropy).

**Chain-link assignment.** L6 + L8 (close).

---

# Stage 6 — Cellular age inversion

Stage 6 produces the framework's other headline product feature: **the per-class cellular age** — the chronological age at which a typical healthy person has the same A-score the patient has, computed independently per architectural class. **Eight per-class cellular ages, never collapsed by default.** This is the canonical Recipe §6.3 inversion: no training set, no regression coefficients, no comparison to other frameworks. The 80-cell age reference matrix IS the calibrated instrument; the v4 scoring module inverts it.

The patient's per-class A-score (Stage 4) is the input. The 80-cell baseline (age × class) is the calibration curve. The output is the per-class age at which baseline A_mean crosses the patient's A. When the patient's A is outside the baseline range, the saturation flag carries that information forward — it is not a bug, it is a measurement.

---

## §52. Step 6.1 — Per-class A-score input (from Stage 4)

**What this step does.** Receives the patient's 8 per-class A-scores from Stage 4 (§43). These are the inputs to the inversion.

**Inputs.** Stage 4 output (<A-score output — emitted by `iamatlas_a_scoring.py` (real module per inventory)>).

**Atlas reference.** Indirect.

**Files invoked.** `iam_cellular_age_scoring.py` — class `IAMCellularAge`, method `score_patient(per_class_a)`.

**The math.** Loading only.

**CMB equivalent.** **Loading per-survey cosmological-parameter posteriors before tension analysis.** A trivial loading step that connects two computational stages.

**How the methylome differs in implementation.** Trivial.

**How it's the same in principle.** Trivial.

**Outputs.** In-memory per-patient 8-element A vector ready for inversion.

**Decision points.** None.

**Failure modes.** None.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7 (likelihood — partial; the cellular age clock lives here).

---

## §53. Step 6.2 — Age reference matrix load (`age_reference_matrix.json`)

**What this step does.** Loads the 80-cell age × class baseline reference matrix into memory. This is the calibrated instrument the per-class A-score is inverted against.

**Inputs.** `age_reference_matrix.json` (or `.csv` / `.py` — three formats, same data).

**Atlas reference.** **THIS IS the age slice of IAMAtlas.** The 80 cells (8 classes × 10 decadal age bins from 4 to 95) each carry `(age_midpoint, A_mean, A_sd, β_mean, β_sd, n_samples, A_p10, A_p25, A_p50, A_p75, A_p90, source_citation)` measured from the IAMAtlas reference cohort.

**Files invoked.** `iam_cellular_age_scoring.py` — `IAMCellularAge.load_reference()`. Or directly via `age_reference_matrix.py`'s helpers `age_ref_A()`, `age_ref_sd()`, `age_ref_percentiles()`.

**The math.** Loading only. SHA-256 verification.

**Source citations in the matrix (data references — NOT runtime atlas queries):** Hannum 2013, Horvath 2013, Roadmap Epigenomics 2015, Moss 2018, Lister 2013, Alisch 2012, Adelman 2019, De Jager 2014 / Shireby 2022, Jaiswal 2014 (CHIP-neg). **These citations document where the original per-decade β values were measured.** They are NOT external atlases consulted at runtime — they are the historical data sources that contributed to the IAMAtlas reference build's age stratification.

**CMB equivalent.** **Loading the redshift-evolution calibration tables.** When inferring cosmological parameters from supernovae or BAO at different redshifts, the framework requires per-redshift calibration curves (luminosity-distance relation, sound-horizon scale). Loading these tables is the methylome-equivalent of loading the per-age calibration curves.

**How the methylome differs in implementation.** Methylome reference is decadal age bins (4-95 years); CMB redshift is continuous-but-binned. Both are pre-computed calibration tables.

**How it's the same in principle.** A measurement at the patient's age is interpreted against the reference's expectation at that age.

**Outputs.** In-memory age reference matrix.

**Decision points.** SHA matches → proceed. Mismatch → HALT.

**Failure modes.** File missing or corrupted → halt.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §54. Step 6.3 — Per-class A inversion against the 80-cell baseline curve

**What this step does.** For each architectural class, finds the chronological age at which the baseline A_mean curve crosses the patient's A-score for that class. **This is the canonical Recipe §6.3 inversion — the framework operation that turns an A-score into a cellular age.**

**Inputs.** Patient per-class A from §52 + age reference matrix from §53.

**Atlas reference.** Via the age reference matrix (atlas's age slice).

**Files invoked.** `iam_cellular_age_scoring.py` — method `_invert_per_class(a_class, class_name)`.

**The math.** Per class `c`:

1. Construct the baseline A_mean curve: 10 points `(age_midpoint[k], A_mean[c, k])` for k = 0..9 (decadal bins from 4 to 95).
2. Find the age `t*` such that `A_mean(c, t*) = A_patient[c]` via linear interpolation between adjacent decade midpoints.
3. **Saturation handling** (this is critical):
   - If `A_patient[c]` > max(A_mean[c, :]) → the patient is ABOVE the entire baseline curve → status `SAT_HIGH`. Report cellular age as ">95" with the saturation flag.
   - If `A_patient[c]` < min(A_mean[c, :]) → the patient is BELOW the entire baseline curve → status `SAT_LOW`. Report cellular age as "<4" with the saturation flag.
   - Otherwise → status `OK`. Cellular age is the interpolated `t*`.

**Implementation note:** The baseline A_mean curve is class-specific. Some classes (e.g., immune) show a monotonic increase with age (older = more departure from H_min); some classes (e.g., stem_pluri) show a flat profile (stem cell architectures change less with age). The inversion respects class-specific monotonicity.

**CMB equivalent.** **Inverting the angular-diameter-distance relation to find redshift.** Given a measurement of the angular-diameter distance to an object, invert the cosmological-model distance-redshift relation to find the redshift at which a standard-model object would produce that distance. **The methylome's age inversion is structurally identical: given a measurement of A-score, invert the standard-class A-vs-age curve to find the age at which a healthy individual would produce that A.**

**How the methylome differs in implementation.** Cellular age inversion is per-class (8 independent inversions); cosmological distance-redshift inversion is one-shot. Both use the same algorithmic structure (find the parameter value that matches the observation against a calibrated curve).

**How it's the same in principle.** The framework measurement is converted into a physical age by inverting a calibrated reference curve. The math doesn't care that one is cellular biology and the other is cosmological distance.

**Outputs.** Per-class cellular age (8 values) + per-class status (8 codes: OK / SAT_HIGH / SAT_LOW / INSUFFICIENT_MARKERS).

**Decision points.**
- All 8 classes status `OK` → proceed.
- Some classes `SAT_HIGH` / `SAT_LOW` → proceed, carrying saturation flags forward; this is data signal, not bug.
- Some classes `INSUFFICIENT_MARKERS` (inherited from Stage 4) → those classes return cellular age `NaN` with status `INSUFFICIENT`.

**Failure modes.**
- Baseline matrix corrupted (non-monotonic A_mean curve where biology demands monotonic) → halt; investigate atlas build.

**Canonical cross-references.**
- Recipe §6.3 (the canonical inversion algorithm).

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §55. Step 6.4 — Saturation handling (SAT_HIGH / SAT_LOW / OK / INSUFFICIENT_CPGS)

**What this step does.** Documents the patient's saturation pattern across classes. The saturation pattern itself is a measurement.

**Inputs.** Per-class status codes from §54.

**Atlas reference.** None.

**Files invoked.** `iam_cellular_age_scoring.py` — method `_analyze_saturation(per_class_status)`.

**The math.** Count and report:
- `n_sat_high` = number of classes with status SAT_HIGH.
- `n_sat_low` = number of classes with status SAT_LOW.
- `n_ok` = number with status OK.
- `n_insufficient` = number with status INSUFFICIENT.

Patient `saturation_signature` = a labeled 8-class vector (e.g., "SAT_HIGH on terminal, immune, secretory; OK on cycling, progenitor; SAT_LOW on stromal, stem_adult, stem_pluri"). This is itself a structural readout of the patient's deviation from the IAMAtlas-calibration range.

**Observed pattern on EPIC-Italy validation (1,174 patients):** 100% of patients saturate on at least one class. 7 of 8 classes saturate for most patients. Only cycling is in-range for ~half the cohort. **This is direct data about the EPIC-Italy cohort's posture against the IAMAtlas calibration — it is NOT a bug.** The cohort and the calibration sit in different regions of A-score space; saturation makes that posture explicit.

**CMB equivalent.** **Flagging when a survey's parameter posteriors lie outside Planck's calibration range.** Some external surveys (e.g., DES-Y3 σ_8) report values outside the joint Planck posterior region. The disagreement is flagged, not suppressed — the framework reports the survey's value and notes the calibration-range mismatch. The methylome's saturation flag is the same: report what we measured, note that it's outside the calibration baseline.

**How the methylome differs in implementation.** Saturation is per-class (8 flags) vs per-parameter (≤7 parameters). Same operational principle.

**How it's the same in principle.** Out-of-calibration values are recorded with their flags, never silently clipped or imputed.

**Outputs.** Per-patient `saturation_signature` + `n_sat_high` / `n_sat_low` / `n_ok` / `n_insufficient` counts.

**Decision points.** None — pure reporting.

**Failure modes.** None.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §56. Step 6.5 — Eight per-class cellular ages — never collapsed by default

**What this step does.** Records the eight per-class cellular ages as **the framework's headline cellular age output**. **They are never collapsed to a single number by default.** Each class carries its own age because each architecture's relationship to age is biologically distinct.

**Inputs.** Per-class cellular age + status from §54.

**Atlas reference.** None.

**Files invoked.** Trivial recording.

**The math.** None.

**Why never collapse:** A patient with "cellular age 47" hides whether the 47 came from an OK reading across all classes, or whether terminal-class reads 35 and immune-class reads 65. The latter is a clinical signal (immune compartment ageing faster than terminal compartment); collapsing destroys it. Customer reports show all eight ages with class labels; an optional summary line reports a `n_samples`-weighted mean across non-saturated classes for the brief-summary box, but the eight per-class ages are the primary deliverable.

**CMB equivalent.** **Per-parameter cosmological measurement, never collapsed into a single "cosmology number."** Planck reports Ωm, ΩΛ, h, σ_8, etc. each as its own number. There is no "the cosmological parameter is 0.42" — that would be meaningless. Same here: there is no single cellular age. There are eight per-class cellular ages.

**How the methylome differs in implementation.** Eight values vs ~7 cosmological parameters. Same structural decision: do not collapse.

**How it's the same in principle.** Multi-parameter framework measurements are reported as vectors. Collapsing destroys signal.

**Outputs.** Per-patient 8-element `per_class_cellular_age` vector + per-patient 8-element `per_class_status` vector + optional `summary_cellular_age` (weighted mean over non-saturated classes).

**Decision points.** None.

**Failure modes.** None.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §57. Step 6.6 — Percentile rank at patient's chronological age

**What this step does.** For each class, compute the patient's **percentile within the age-matched healthy distribution**. This is the readout most directly interpretable by a clinician: "your immune cellular age is at the 78th percentile for your chronological age."

**Inputs.** Per-class A from Stage 4 + patient's chronological age + age reference matrix percentiles (A_p10, A_p25, A_p50, A_p75, A_p90).

**Atlas reference.** Via age reference matrix.

**Files invoked.** `age_reference_matrix.py` — helper `age_ref_percentiles(class, age)` returns the 5-percentile vector for the patient's age.

**The math.** Per class `c`:
1. Linearly interpolate the 5 percentile points (A_p10, A_p25, A_p50, A_p75, A_p90) at the patient's chronological age between adjacent decade midpoints.
2. Determine which inter-percentile band the patient's A falls into.
3. Linear interpolation within the band gives a refined percentile estimate.

Example: patient A = 1.08 for immune class at age 52. At age 52, immune A_p50 = 1.04, A_p75 = 1.09. Patient is at ~73rd percentile.

**CMB equivalent.** **Within-survey-cohort percentile of a derived parameter.** When a survey reports a measurement of S_8 = 0.776, the survey can report "this is at the 23rd percentile of the Planck-fit S_8 posterior, given the systematic uncertainties." Same operation: position a measurement against a calibrated distribution.

**How the methylome differs in implementation.** Methylome percentile is per-class against an age-stratified HC distribution. Same principle.

**How it's the same in principle.** Percentile-against-calibrated-distribution is the operational way to communicate where a single measurement sits relative to expected.

**Outputs.** Per-patient per-class `percentile_rank` (8 values). Reported in the customer's report at Stage 9.

**Decision points.** None.

**Failure modes.**
- Patient chronological age outside reference range (4-95) → flag with `AGE_OUT_OF_RANGE`; percentile reported as boundary.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7.

---

## §58. Step 6.7 — Stage 6 output: per-class cellular age vector + optional summary

**What this step does.** Consolidates Stage 6 outputs into the per-cohort artifact.

**Inputs.** Outputs of §54 + §55 + §56 + §57.

**Atlas reference.** None (consolidation).

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Published per-survey cosmological-parameter table with percentile context.

**How the methylome differs in implementation.** Per-patient vs per-survey. Same content structure.

**How it's the same in principle.** Multi-parameter measurement output with context.

**Outputs.** <cellular-age output — emitted by `iam_cellular_age_scoring.py` (real module per inventory)> carrying per patient:
- 8 per-class cellular ages.
- 8 per-class status codes (OK / SAT_HIGH / SAT_LOW / INSUFFICIENT).
- 8 per-class percentile ranks at chronological age.
- `saturation_signature` (string description).
- `summary_cellular_age` (optional weighted mean, OK classes only).
- Patient's chronological age (for cross-reference).

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 7 (§59 — tier breakpoints).

**Failure modes.** None at consolidation.

**Canonical cross-references.** Recipe §6.3.

**CPG Plate references.** None.

**Chain-link assignment.** L7 (close).

---

# Stage 7 — Tier breakpoint detection

Stage 7 turns the framework's continuous measurements (A-scores, cellular ages) into the **discrete tier calls** the customer report and the cards consume. Engine-tier vocabulary is internal (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH); customer-tier vocabulary is collapsed (NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED). The cfDNA branch activates when substrate is plasma. **Stage 7 does no new physics — it applies pre-specified breakpoints and language collapse to outputs already produced upstream.**

---

## §59. Step 7.1 — Per-class A-score tier call (`tier_breakpoints.json v1.2`)

**What this step does.** For each architectural class, maps the patient's A-score to an engine tier using the 6-tier physics-derived breakpoints in `tier_breakpoints.json v1.2`. The breakpoints are universal (same across all classes); per-class structural ceilings (1/H_min) cap the highest reachable tier per class.

**Inputs.** Per-class A from Stage 4 (§46) + per-class A 95% CI propagated from MCMC posteriors + patient intake covariates (for the override modes) + `tier_breakpoints.json v1.2` + (optional) Stage 4.5 directional composite from §46.5 if FLAG_BIDIRECTIONAL is set.

**Atlas reference.** Indirect: structural ceiling per class is `1 / H_min(class)` from the frozen MCMC posteriors (G-003b freeze 2026-04-06). Tier breakpoints (1.07 Warburg + 1.10 breach) are physics-defined inflection points, not statistical percentiles.

**Files invoked.** `Biological_Physics/atlas_vault/walther_clinical_runtime/Tier_breakpoints/tier_breakpoints.json` (v1.2). Engine consumes the JSON via a small helper. v0 4-tier statistical-percentile predecessor archived in `Tier_breakpoints/OLD/tier_breakpoints_v0_4tier_statistical.json`.

**The math.** Per class `c`, using the 6-tier system v1.2:

| Engine tier | Condition | Customer label | Physics meaning |
|---|---|---|---|
| SUPPRESSED | A[c] < 0.95 | Suppressed | Below baseline — context-dependent (treatment/transplant/immunosuppression) |
| NORMAL | 0.95 ≤ A[c] < 1.04 | Normal | Within healthy sampling variance |
| ELEVATED | 1.04 ≤ A[c] < 1.07 | Elevated | Recoverable drift; intervention window |
| WARBURG_TRANSITION | 1.07 ≤ A[c] < 1.10 | Warburg Transition | **1.07 Warburg line** — intervention character changes from "add fuel" to "restrict and rebuild" |
| SIGNIFICANTLY_ELEVATED | 1.10 ≤ A[c] < 1.12 | Significantly Elevated | Structural-fidelity breach territory; trajectory direction is primary read |
| BREACH | A[c] ≥ 1.10 sustained OR A[c] ≥ 1.12 single-timepoint | Breach | **1.10 architectural-fidelity breach line** — prompt for clinical workup; NOT a diagnosis |

Per-class structural ceiling (`structural_ceiling_by_class` in tier_breakpoints.json v1.2): if a class's `1/H_min` is below 1.10, the BREACH tier is structurally unreachable for that class. stem_pluri (ceiling 1.0181) is structurally blind for BREACH; SIGNIFICANTLY_ELEVATED is the practical ceiling. Runtime saturation margin: 0.005 below the ceiling, engine emits SATURATED flag.

**Override modes (v1.2):** Per the patient intake covariates routed at BUILD_SPEC v1.2 §4.5, the standard 6-tier output is replaced with mode-specific interpretation when triggered:
- **EXPECTED_SUPPRESSION** (current_immunosuppression / transplant_status) — SUPPRESSED reading interpreted as therapeutically expected
- **TRAJECTORY_WATCH** (autoimmune / chronic inflammatory / HIV+ treated) — ELEVATED floor shifted upward to 1.10; trajectory is primary
- **TREATMENT_RESPONSE** (current_cancer_in_treatment) — trajectory framing across treatment timepoints
- **CONTEXT_PREGNANCY** / **POSTPARTUM** — physiological immune shift framing
- **CONTEXT_HRT_BASELINE** — HRT-stratified baseline (CPG-VAL-018)
- **CONTEXT_WEIGHT_LOSS_INTERVENTION** (GLP-1 / bariatric) — expected anti-inflammatory trajectory (CPG-VAL-021)

**Smoking-bin interim mitigation (v1.2; retires when `IAMAtlas_smoking_layer.csv` fit at v1.3):** ELEVATED floor shifted by smoking_bin: current=1.10 / former_0_5y=1.08 / former_5_15y=1.07 / former_15plus_y=1.05 / never=1.04.

**Bidirectional pattern handoff (v1.2):** When Stage 4.5 (§46.5) sets `FLAG_BIDIRECTIONAL = True` for a class, this step uses the directional composite (signed) rather than the pooled A-score to drive tier reporting. Mapping per `bidirectional_pattern_handoff.directional_composite_tier_mapping`:
- |a_dir| < 0.40 → NORMAL
- 0.40 ≤ |a_dir| < 0.80 → ELEVATED
- 0.80 ≤ |a_dir| < 1.20 → WARBURG_TRANSITION
- 1.20 ≤ |a_dir| < 1.60 → SIGNIFICANTLY_ELEVATED
- |a_dir| ≥ 1.60 → BREACH-ANALOG

**Tier confidence propagation (v1.2):** Tier confidence is the probability of A falling in each tier under the MCMC-propagated posterior distribution. When |P(primary_tier) − P(second_max_tier)| < 0.20, engine emits BORDERLINE_TIER flag and customer report says "your reading straddles the {tier_A}/{tier_B} boundary."

**CMB equivalent.** Significance tier of a cosmological measurement (3σ / 5σ thresholds), but with physics-defined inflection points rather than statistical percentiles. The 1.07 Warburg line is analogous to a phase-transition threshold in cosmology — the same intervention has different effects above and below the line.

**How the methylome differs in implementation.** The 1.07 Warburg line + 1.10 breach line are framework-internal physics inflection points, not statistical percentiles relative to a null. The 6-tier system also propagates CI-based tier confidence forward (BORDERLINE_TIER) and supports covariate-conditional override modes — neither has a direct CMB analog.

**How it's the same in principle.** Continuous → discrete via pre-specified thresholds + forward propagation of measurement uncertainty into the tier confidence.

**Outputs.** Per-patient per-class:
- Primary engine tier (one of 6)
- Customer-facing label (matching engine label v1.2)
- BORDERLINE_TIER flag (when tier-boundary straddler detected)
- Override mode in effect (when covariate-triggered)
- Customer paragraph (rendered from tier × override-mode lookup)

**Decision points.** Override modes activate via covariate-trigger lookup; bidirectional handoff activates when Stage 4.5 set FLAG_BIDIRECTIONAL; smoking-bin floor-shift activates pre-Stage-3-foreground-fit.

**Failure modes.** None at this step (pure mapping). Downstream Stage 8 evaluates multi-class breach + override-mode compatibility.

**Canonical cross-references.** BUILD_SPEC v1.2 §5 Stage 7 + §3.4 (tier_breakpoints v1.2 schema). Recipe §7 (tier specification).

**CPG Plate references.** None directly.

**Chain-link assignment.** L8 (parameter inference — discrete tier readout with forward CI propagation).

---

## §60. Step 7.2 — Per-cell-type A-score tier call

**What this step does.** Same operation as §59, but for the 115 per-cell-type A-scores. Cell-type tier calls feed Stage 8 card matching.

**Inputs.** Per-cell-type A from Stage 4 (§46) + cell-type-specific breakpoints (inherit from parent class breakpoints, with optional cell-type-specific overrides in `tier_breakpoints.json`).

**Atlas reference.** Indirect.

**Files invoked.** Same as §59.

**The math.** Identical to §59 with the cell type's parent class breakpoints (or overrides where applicable).

**CMB equivalent.** Per-sub-component tier readouts. Same operational principle.

**How the methylome differs in implementation.** 115 cell types vs 8 classes — finer granularity, same operation.

**How it's the same in principle.** Same as §59.

**Outputs.** Per-patient per-cell-type engine tier (115 values).

**Decision points.** None.

**Failure modes.** None.

**Canonical cross-references.** Recipe §7.

**CPG Plate references.** None.

**Chain-link assignment.** L8.

---

## §61. Step 7.3 — cfDNA branch (when substrate is plasma) — `cfdna_weight.json`

**What this step does.** **Activates only when the patient's substrate is plasma cfDNA.** When activated, compares the patient's per-class fractions to the expected healthy-blood cfDNA tissue-of-origin weights (Snyder 2016 + Moss 2018) and flags departures as pan-tissue context. **This is the only Stage 7 step that conditionally activates.**

**Inputs.** Per-class fractions from Stage 2 (§34) + `cfdna_weight.json` + patient `substrate` from manifest.

**Atlas reference.** None directly. The cfDNA weights are an external literature-derived reference (Snyder 2016, Moss 2018), used at Stage 7 as a clinical-context overlay — not as a framework calibration.

**Files invoked.** `cfdna_weight.json`. Per-class scoring module consumes the JSON.

**The math.** Healthy-plasma cfDNA per-class expected weights:
- immune: 0.70
- cycling: 0.12
- secretory: 0.08
- stromal: 0.04
- stem_adult: 0.03
- progenitor: 0.02
- terminal: 0.005
- stem_pluri: 0.005

Per class `c`:
```
departure_c = observed_fraction_c - expected_weight_c
```
Positive departure = elevated representation of that tissue-of-origin in plasma; negative = suppressed.

**Notable pattern:** Terminal-class (especially neurons) elevation in plasma cfDNA is a marker for tissue damage or turnover (e.g., neurological insult). Cycling-class elevation can indicate increased cellular turnover (regenerative tissues or proliferative diseases).

**CMB equivalent.** **Per-channel foreground subtraction with channel-specific templates.** When Planck observes the same sky pixel across multiple frequencies, each frequency has a different expected foreground contribution. The departures from expected (after foreground subtraction) are the signal. The methylome's cfDNA departures from expected weights are the substrate-specific equivalent.

**How the methylome differs in implementation.** Activates conditionally on substrate (plasma only); tissue and buffy substrates skip this step. Frequency-channel foreground subtraction is not conditional in CMB (all frequencies always contribute).

**How it's the same in principle.** Substrate-specific systematic baseline subtracted to surface the true departure signal.

**Outputs.** Per-class `cfdna_departure` value when substrate is plasma. Flagged as pan-tissue context in the report.

**Decision points.**
- Substrate = plasma → execute step.
- Substrate = tissue or buffy → skip step; report has no cfDNA section.

**Failure modes.**
- Substrate field missing or unknown → flag and skip cfDNA step.

**Canonical cross-references.** Recipe §7 (cfDNA branch). Runtime Matrices README "Stage 4 cfDNA".

**CPG Plate references.** None.

**Chain-link assignment.** L8.

---

## §62. Step 7.4 — FLOOR_BREACH detection

**What this step does.** Cross-class detection of patients whose A-scores exceed expected ranges in **multiple** classes simultaneously. A single FLOOR_BREACH in one class is informative; multiple FLOOR_BREACH-es across uncorrelated classes indicates a global pathology (or massive batch effect, or substrate misidentification).

**Inputs.** Per-class engine tiers from §59.

**Atlas reference.** None.

**Files invoked.** Tier aggregation logic.

**The math.** Count classes with engine tier `FLOOR_BREACH`. Aggregation:
- 0 FLOOR_BREACH → no global flag.
- 1 FLOOR_BREACH → standard single-class breach; flag in report.
- 2-3 FLOOR_BREACH-es → flag `MULTI_CLASS_BREACH`; manual review at Stage 9.
- ≥ 4 FLOOR_BREACH-es → flag `GLOBAL_BREACH`; hold patient at Stage 9; do not auto-deliver report until reviewed.

**CMB equivalent.** **Multi-parameter tension detection.** When multiple cosmological parameters jointly disagree with Planck (e.g., DES S_8 + H_0 + Ωm all 2σ low), the joint tension is reported even when no single parameter is 5σ off. Same logic: multiple simultaneous departures get aggregated into a higher-significance flag.

**How the methylome differs in implementation.** Cell-class FLOOR_BREACH count vs parameter-tension count. Same aggregation principle.

**How it's the same in principle.** Joint failures across multiple framework dimensions warrant higher-significance flagging than any single failure.

**Outputs.** Per-patient `global_breach_flag ∈ {none, multi_class, global}`.

**Decision points.**
- `none` or `multi_class` → proceed.
- `global` → HOLD report; route to manual review.

**Failure modes.** Global breach is itself a finding, not a failure.

**Canonical cross-references.** Recipe §7.

**CPG Plate references.** None.

**Chain-link assignment.** L8.

---


**Step 7.4b — Bidirectional flag for BELOW_NORMAL + documented-suppression-pattern coincidence.** A class scoring `BELOW_NORMAL` (A-score significantly below age-matched HC expectation) is informative on its own, but becomes a Stage-8-relevant flag when the patient's Stage 5 / Stage 8 outputs ALSO match a card or matrix pattern that includes documented suppression signature for that same class. Two named instances make this concrete:

- **AD-immune B-vs-T-lineage divergence** (AD card, when built): if patient shows immune class A-score elevated AND B-lineage cell types DOWN while T-lineage cell types UP (or vice versa), the bidirectional pattern is itself the signature — neither direction alone fires. Card returns BIDIRECTIONAL_PATTERN tier.
- **Long-window pre-diagnostic breast secretory homogenization** (breast-epic card): if patient shows secretory class A-score below age-matched HC AND breast-epic residual map's hypomethylation pattern hits, the homogenization is the signature. Card weights the BELOW_NORMAL secretory as POSITIVE evidence rather than treating it as "uninteresting." Anchored by VAL-047 Phase 6 Deep Audit GSE51057 (10yr+ d=−1.226, p=3×10⁻⁴).

The orchestrator's Stage 7 emits the `bidirectional_flag` field for every class scoring BELOW_NORMAL. Stage 8 consumes the flag when evaluating cards and matrix candidates that document suppression directions. Suppression is signal — the SOP forbids silently dropping it as "low priority."

## §63. Step 7.5 — Engine-to-customer language mapping

**What this step does.** Translates engine-tier vocabulary into customer-facing vocabulary. Customer reports never see engine-internal labels like FLOOR_BREACH or URGENT.

**Inputs.** Per-class engine tiers from §59 + per-cell-type engine tiers from §60.

**Atlas reference.** None.

**Files invoked.** `tier_breakpoints.json` — its `customer_label_map` section.

**The math.** Per engine tier:

| Engine tier | Customer label |
|---|---|
| BELOW_NORMAL | SUPPRESSED |
| NORMAL | NORMAL |
| MARGINAL | ELEVATED |
| DETECTABLE | SIGNIFICANTLY_ELEVATED |
| URGENT | SIGNIFICANTLY_ELEVATED |
| FLOOR_BREACH | SIGNIFICANTLY_ELEVATED |

The customer label DOES NOT distinguish DETECTABLE from URGENT from FLOOR_BREACH. **This is intentional.** All three engine tiers indicate the patient has departed significantly from baseline; the engine-tier distinction is for internal triage and for the chain audit trail. The customer label collapse exists to avoid implying clinical-grade discrimination between "detectable" and "urgent" that the framework's confidence intervals don't yet support.

**CMB equivalent.** **Mapping internal cosmological-tier vocabulary to public-language descriptors.** When Planck reports "3.5σ tension with DES S_8," internal language is precise; public-press-release language is "in tension with low-redshift surveys." The collapse exists to avoid implying false precision.

**How the methylome differs in implementation.** Per-class label per patient vs per-parameter label per release. Same operational structure.

**How it's the same in principle.** Engine-internal precision is collapsed into customer-facing labels that match what the framework's confidence supports.

**Outputs.** Per-patient per-class customer label + per-patient per-cell-type customer label.

**Decision points.** None — pure mapping.

**Failure modes.** None.

**Canonical cross-references.** Recipe §7. Part I §10 (Stage 9 legal boundary).

**CPG Plate references.** None.

**Chain-link assignment.** L8 (close).

---

## §64. Step 7.6 — Stage 7 output: per-class tier vector + customer labels

**What this step does.** Consolidates Stage 7 outputs into the per-cohort artifact.

**Inputs.** Outputs of §59 + §60 + §61 + §62 + §63.

**Atlas reference.** None (consolidation).

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Per-survey tier-readout table.

**How the methylome differs in implementation.** Per-patient row. Same content structure per row.

**How it's the same in principle.** Discrete tier output with context.

**Outputs.** <tier vector — emitted internally by `GAPE_WEB_v13.py`> carrying per patient:
- 8 per-class engine tiers + 8 customer labels.
- 115 per-cell-type engine tiers + 115 customer labels.
- `cfdna_departures` per class (if applicable).
- `global_breach_flag`.

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 8 (card matching — covered in Part II-C).

**Failure modes.** None at consolidation.

**Canonical cross-references.** Recipe §7.

**CPG Plate references.** None.

**Chain-link assignment.** L8 (close).

---

**End of Part II-B (Stages 5 through 7, §47-§64). Part II-C continues with Stages 8 (card matching), 9 (report assembly), 10 (delivery). Part III follows with chain-integrity scaffolding (L9 null suite). Part IV with failure modes. Part V with reference tables.**

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*


> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them. Where a step's logic currently lives inside `GAPE_WEB_v13.py` (the
> production engine — see `SYSTEM_INVENTORY.md`), the SOP now says so honestly. Where
> a step's runtime artifact or output file location was invented, the SOP now reads
> "TBD per orchestrator design" — that orchestrator (working name `web.commercial.py`)
> is a separate conversation Heath and Walther will have. All real paths in this v1.1
> are documented in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be
> treated as not-yet-existing until verified against the repo.

---


**Continues from Part II-B (§47-§64). This document contains §65-§79 — Stages 8 (card-level pattern matching), 9 (report assembly), 10 (delivery).**

---

# Stage 8 — Card-level pattern matching

Stage 8 is where the engine stops measuring and starts interpreting. Stages 0 through 7 produced a tier vector — eight per-class engine tiers, 115 per-cell-type engine tiers, optional cfDNA departures, a global floor-breach flag. Stage 8 takes that tier vector and asks a different kind of question: **does this pattern match a known disease signature?**

The answer is **never** a diagnosis. The answer is a card-specific tier call — "the pattern matches the breast-epic long-pre-dx signature with confidence X" — which the report layer in Stage 9 then wraps in legally-permissible language. Card matching is the inflection point between the physics (which is the same for everyone) and the cards (which encode specific disease-state hypotheses against which patterns are tested).

The disease signature matrix v1.5 (77 rows × 131 columns, 354 populated signature cells) is the lookup table. Every row is a disease × time-range × substrate × severity-class combination. Every column is one of 131 immune sub-cell-types and tissue-of-origin cells. Every populated cell is an expected effect size range (e.g., `+0.81/+1.26` meaning "Cohen's d expected somewhere in [+0.81, +1.26]"). The matrix is the empirical posterior of every CPG-VAL ever sealed.

The chain-of-custody discipline here is strict: a card-specific tier call is a **conjunction of conditions**. Multiple per-class tiers, multiple per-cell-type departures, optional cfDNA flags, all need to satisfy the card's declared pattern at the card's declared confidence threshold. No single signal carries a card on its own. **The card never fires on one tile.** That rule is in `breast-epic_card_v2_3.json` and in every other production card by construction.

---

## §65. Step 8.1 — Disease signature matrix v1.5 lookup

> **CRITICAL — Stage 8 runs TWO parallel matching paths, NOT sequential.** Both consume the same Stage 4 + Stage 5 + Stage 7 outputs; both produce verdicts; both flow into Stage 9 report assembly. They are complementary, not redundant. (Walkthrough §4 Stage 5.)
>
> **Path A — Per-card matching** (this section + §66–§69). Each card in `DISEASE_MAPS_CARDS/{card_id}/` is loaded, eligibility-gated against patient metadata, evaluated against its panel + residual map + bimodality map + PCA projections + covariate thresholds, and returned with verdict (FIRED / NOT_FIRED / NOT_ELIGIBLE) + tier + contributing pattern + `educational_page_url`.
>
> **Path B — Disease signature matrix lookup** (this step §65 + §67's matching). The 77 × 131 matrix is scored row-by-row against the patient's per-cell-type A-score profile via the schema's `compute_match_magnitude()` (Mahalanobis-style sign-aligned product weighted by √n — NOT raw dot product). Top-3 candidates returned with tier (from `compute_customer_tier(match × phase × severity × evidence)`) + `organ_pages_to_link`.
>
> **Worked example of why dual matching catches what cards alone miss.** A patient with `regulatory_T_cells +1.2 + erythroid_progenitor +0.8 + pancreatic_beta_cells +1.0 + multi-organ distributed elevation` returns from Path A: NO single card fires above ELEVATED (no card uses that exact combination). From Path B: `breast_cancer / long_pre_dx` is the strongest matrix match — the >10yr distributed pre-diagnostic signature with 7 cells contributing. Without the matrix, the report says "everything looks slightly off, nothing fires" and misses the pattern. With the matrix, the report says "this combination of cellular drift most resembles the documented pattern for [X] at [phase]." Both paths report; neither overrides the other; Stage 9 surfaces both.


**What this step does.** For each card that applies to the patient's substrate (e.g., buffy coat DNA → breast-epic, lung-epic, prostate-epic, cardio-epic, AD-immune, MS-immune, Parkinson-immune, CRC-immune-inv all apply; plasma cfDNA → adds hcc-cfdna, pancreatic-cfdna), pull the relevant disease signature rows from the v1.5 matrix and prepare them for pattern matching.

**Inputs.**
- Patient substrate (from Stage 0 manifest).
- Stage 7 output: <tier vector — emitted internally by `GAPE_WEB_v13.py`>.
- The disease signature matrix CSV.

**Atlas reference.** None directly. The disease signature matrix is **derived from** IAMAtlas posteriors (every populated cell traces back to a CPG-VAL that ran the chain from L1 through L9), but Stage 8.1 reads the matrix directly, not the atlas.

**Files invoked.**
- Module: `<card-matching logic inside `GAPE_WEB_v13.py`>`
- Lookup table: `disease_cell_signature_matrix_v1_5.csv` (77 rows × 131 columns, 354 populated cells, SHA-256 hashed at load).
- Card registry: `<card registry — currently embedded in `GAPE_WEB_v13.py`>` — maps each card to the disease_id rows it pulls.

**The math.** None at this step — it's a SQL-like lookup. For a card with `disease_id=breast_cancer` and substrate `whole_blood_buffy_coat`, pull all rows where `disease_id == 'breast_cancer' AND substrate == 'whole_blood_buffy_coat'`. The result for breast-epic is four rows: `long_pre_dx` (>10y), `mid_pre_dx` (5-10y), `mid_late_pre_dx` (2-5y), `near_dx` (within 2y) — each row containing the expected per-cell-type Cohen's d ranges for that phase.

**CMB equivalent.** The **template bank** in matched-filter searches — LIGO's compact-binary template bank, Planck's primordial-non-Gaussianity template comparison, the cosmological-parameter grid evaluation. A measurement is compared against a set of pre-declared signal templates, and the template that best matches (subject to declared thresholds) is the candidate hypothesis. The templates themselves are not derived from this measurement; they were laid down in advance by independent calibration work.

**How the methylome differs in implementation.** The templates are time-phased — long_pre_dx vs near_dx are different templates of the same disease's evolution, not different diseases. CMB templates are typically time-static (the universe doesn't change between two CMB observations); methylome templates encode disease progression.

**How it's the same in principle.** Both are template banks. Both are declared in advance. Both are tested by overlap with measurement, not by training on the measurement.

**Outputs.** A per-patient, per-card candidate-template dictionary: `{card_id: [phase_template_1, phase_template_2, ...]}`. Each phase template carries the expected per-cell-type effect-size ranges in v1.5 format.

Stored in-memory for Stage 8.2 (residual map application). Not persisted as a separate file.

**Decision points.**
- If no cards apply to the substrate (which should never happen in production — every supported substrate has at least one card by the engine deployment rule), the patient's Stage 8 output is empty. The chain proceeds to Stage 9 with a "no applicable cards" flag.
- If the matrix SHA-256 at load doesn't match the registered version, the engine halts and refuses to process. No silent degradation.

**Failure modes.**
- **Matrix version mismatch.** The card registry pins a specific matrix version (v1.5 SHA). A mismatch means someone updated the matrix without updating the registry. Hard halt.
- **Substrate-card mismatch.** A card declared for plasma cfDNA cannot match a buffy coat sample. Caught at registry lookup — the patient simply doesn't get that card. Not a failure; an absence.
- **Empty card registry.** Indicates engine deployment misconfiguration. Hard halt.

**Canonical cross-references.** Recipe §8 (disease-state matching). Roadmap §10.2.6 Phase F (re-audit through complete chain). Disease matrix README at `disease_signature_matrix_README.md`.

**CPG Plate references.** None at this step.

**Chain-link assignment.** L7 (likelihood prep) + L8 (parameter inference prep).

---

## §66. Step 8.2 — Per-card residual map application

**What this step does.** For cards that ship with a **per-card residual map** (breast-epic has one at `breast_epic_residual_map_v0_1.csv`, 1,392 concordant CpGs with signed Cohen's d derived from CPG-VAL-003), apply the residual map to the patient's foreground-cleaned β matrix from Stage 3.

The residual map is a card's empirical per-CpG signal-direction blueprint at the disease's relevant phase. Applying it produces a **patient-level signal-overlap score** — how much of the card's expected residual pattern is present in this patient's β profile.

**Inputs.**
- Patient cleaned β matrix from Stage 3 output (the substrate after foreground subtraction).
- Card's residual map CSV (one file per card; not all cards have one).
- Card metadata declaring overlap method (default: Pearson correlation between patient's per-CpG departure and the residual map's signed Cohen's d).

**Atlas reference.** None directly. The residual map was built by running the full chain on a sealed VAL cohort (e.g., CPG-VAL-003 for breast-epic) and storing the resulting per-CpG signal direction. Atlas was consulted when the residual map was made; not when applying it.

**Files invoked.**
- Module: `<inside `GAPE_WEB_v13.py`>`
- Per-card residual maps live at `<per-card residual map — placeholder dir at `pipeline_runtime_matrices/card_residual_maps/`, populated as VALs lock>`. Currently in production: `breast_epic_residual_map_v0_1.csv`. Other cards pending residual map derivation per Phase F backlog.

**The math.** Given the patient's per-CpG departure vector δ_patient (length N_CpGs_in_map) and the card's signed Cohen's d vector d_card (same length, same CpG order):

> overlap_score = ρ(δ_patient, d_card)  
> where ρ is Pearson correlation, with confidence interval via Fisher z-transform.

A patient whose disease pattern matches the card's residual map will have overlap_score significantly positive (typically ρ > 0.10 with N=1,392 is p < 0.001). A patient unrelated to the card's disease will have overlap_score consistent with zero.

A patient with a **reversed** signal (signal exists but pointing the opposite direction — possible when a different disease drives similar CpGs in the opposite direction) will have overlap_score significantly negative. The sign of overlap_score is itself diagnostic.

**CMB equivalent.** The **matched-filter inner product** — `<data | template>` weighted by inverse noise covariance — that LIGO, EHT, and CMB template searches all compute as their primary statistic. The Pearson form here is a noise-flat approximation; a proper matched filter would weight each CpG by its inverse per-CpG variance from L6. Phase D delivers that upgrade.

**How the methylome differs in implementation.** The "template" is per-CpG signed Cohen's d, not a continuous frequency-domain waveform. The noise weighting is currently approximated as flat (each CpG weighted equally) because per-CpG covariance hasn't been delivered by Phase D yet. Phase D upgrade replaces ρ with the full whitened inner product.

**How it's the same in principle.** Both are inner products of a measurement against a declared template, both produce a signed and magnitude-bounded scalar, both can be evaluated for significance against null.

**Outputs.** Per-card overlap score with 95% CI. Signed. Stored in the patient's Stage 8 working dictionary.

**Decision points.**
- If the card has no residual map, Step 8.2 is skipped for that card and pattern matching at Step 8.3 proceeds without the residual-overlap channel.
- If overlap_score has the wrong sign for the disease state being matched, the card's phase template is downgraded — the engine has detected a signal but in the opposite direction. Reported as flag, not as match.

**Failure modes.**
- **CpG coverage mismatch.** Residual map declares 1,392 CpGs; patient may have <90% coverage due to EPIC vs 450K mismatch or QC failures. Default policy: require ≥80% CpG coverage of the residual map; below that, residual-overlap channel is marked INSUFFICIENT and matching falls back to per-class-tier-only matching at 8.3.
- **All-zero patient departure.** A patient whose foreground-cleaned β matrix is essentially flat (e.g., no decoherence above noise floor) will have undefined overlap_score (zero numerator and zero denominator in the Pearson form). Caught with an explicit zero-variance check; reported as NO_SIGNAL.

**Canonical cross-references.** Recipe §8.2. VAL-003 outcome (residual map source). breast-epic card README at `breast-epic_README.md`.

**CPG Plate references.** **Plate 2** (Breast Pre-Diagnostic Anisotropy) shows the residual map for breast-epic rendered as a sky map. Every dot on Plate 2 is one CpG with its signed Cohen's d coloring; the patient overlap calculation here is, conceptually, "how aligned is this patient's per-CpG signal sky with the breast-epic anisotropy sky." **Plate 4 Panel F** shows the same data with the cyan-shifted (hypomethylated) field effect made explicit.

**Chain-link assignment.** L7 (likelihood, partial — residual-overlap is the per-card likelihood proxy until full Bayesian likelihood ships in Phase E).

---

## §67. Step 8.3 — Multi-class pattern matching

**What this step does.** Combine the per-class and per-cell-type tier vector from Stage 7 with the residual-overlap scores from Step 8.2 to produce a card-level tier call. The matching rule is encoded in each card's JSON; the engine reads the rule and applies it mechanically — it does not invent new matching rules.

**Inputs.**
- Stage 7 tier output for this patient.
- Step 8.1 candidate phase templates per card.
- Step 8.2 residual-overlap scores per card (where available).
- Card JSON: `<per-card definition — currently embedded in `GAPE_WEB_v13.py`>` — declares the matching rule.

**Atlas reference.** None directly.

**Files invoked.**
- Module: `<inside `GAPE_WEB_v13.py`>`
- Cards. Currently in production: breast-epic v2.3, lung-epic, prostate-epic, hcc-epic, heme-epic, cardio-epic, AD-immune, MS-immune, Parkinson-immune, CRC-immune-inv, CRC-secretory, cervical-secretory, LGG/GBM-terminal, pancreatic, aging-baseline.

**The math.** Each card's matching rule is a Boolean expression over:
- Per-class engine tiers (e.g., `immune_tier >= DETECTABLE`)
- Per-cell-type engine tiers (e.g., `Baso_tier >= MARGINAL AND breast_BE_tier >= MARGINAL`)
- Per-cell-type effect-size ranges from v1.5 matrix (e.g., `Baso_A_score within [1.01, 1.58]`)
- Residual-overlap thresholds (e.g., `breast_epic_residual_overlap > 0.10 AND CI_lower > 0`)
- Phase-template disjunctions (e.g., `MATCHES(long_pre_dx) OR MATCHES(mid_pre_dx)`)

For each candidate phase template, evaluate the rule. The phase whose rule evaluates TRUE with highest residual-overlap CI lower bound is the card's reported phase. Confidence is the residual-overlap CI lower bound (or, when no residual map exists, the minimum of the per-class tier confidences).

**Worked breast-epic example, long_pre_dx phase, hypothetical patient:**
- Stage 7 tiers: immune_engine=DETECTABLE, secretory_engine=NORMAL, stem_pluri_engine=NORMAL, stromal_engine=MARGINAL, ...
- Per-cell-type tiers: Baso_engine=DETECTABLE (A=1.42), Plasma_engine=DETECTABLE (A=1.18), breast_BE_engine=MARGINAL (A=1.08), microglia_engine=DETECTABLE (A=1.21).
- Step 8.2 residual-overlap: ρ=0.143, 95% CI [0.092, 0.193], p<10⁻⁵.
- v1.5 long_pre_dx row says: Baso d∈[+1.01,+1.58], Plasma d∈[+0.81,+1.26], breast_BE d∈[+0.61,+1.28], microglia d∈[+0.71,+1.30], immune_pooled d=+1.78.
- Patient's per-cell-type A-scores translated to Cohen's d via the IAMAtlas covariance: Baso d≈+1.30 (in range), Plasma d≈+0.95 (in range), breast_BE d≈+0.74 (in range), microglia d≈+1.05 (in range).
- Boolean rule: `immune_tier >= DETECTABLE AND >=3 of (Baso, Plasma, breast_BE, microglia, NeuMa, Mela, neurons_pooled, smooth_muscle) in expected range AND residual_overlap_CI_lower > 0`.
- Evaluation: TRUE. Card fires for long_pre_dx phase with confidence 0.092 (the CI lower bound).

**CMB equivalent.** The **template-likelihood evaluation** in cosmological-parameter inference. Each candidate parameter set evaluates the likelihood of the data given that template; the maximum-likelihood template (subject to prior) becomes the report. The chain-of-custody discipline here is that the Boolean rule is **declared in the card before the patient is run** — the equivalent of pre-registering the template grid before the data arrives.

**How the methylome differs in implementation.** The Boolean rule is currently threshold-based rather than likelihood-based. Phase E delivers the upgrade to full Bayesian likelihood evaluation per card.

**How it's the same in principle.** Both ask: given a measurement and a set of pre-declared templates, which template best matches? Both protect against post-hoc template invention by pre-registering the template grid.

**Outputs.**
- Per-card phase verdict: MATCH_long_pre_dx, MATCH_mid_pre_dx, ..., NO_MATCH, INSUFFICIENT, NEGATIVE_OVERLAP.
- Per-card confidence (lower bound of residual-overlap CI; or minimum of per-class tier confidences when no map).
- Per-card matched-element trace: which Boolean components were TRUE, which were FALSE.

Stored in-memory; flows to §68.

**Decision points.**
- Each card produces independently. A patient can match breast-epic long_pre_dx AND lung-epic mid_pre_dx — the engine reports both. Decision combining is the customer's physician's job, not the engine's.
- If a card produces both a MATCH and a NEGATIVE_OVERLAP for different phases, the card fires with the MATCH phase but adds a contradictory-overlap flag for Stage 9 review.

**Failure modes.**
- **Card rule syntax error.** Caught at engine startup when card registry loads — a malformed Boolean expression triggers hard halt.
- **Phase verdict ambiguity.** Multiple phases of the same card pass the rule with overlapping confidence intervals. Default policy: report the highest-confidence phase; add an ambiguous-phase flag.
- **Residual-overlap channel missing AND per-class tiers all NORMAL.** The card produces NO_MATCH. Not a failure; an absence of pattern.

**Canonical cross-references.** Recipe §8.3. Each card's JSON contains its own matching rule (`breast-epic_card_v2_3.json` §`matching_rule`).

**CPG Plate references.** None at this step (the matching logic itself doesn't have a sky-projection visual).

**Chain-link assignment.** L7 + L8 (partial).

---

## §68. Step 8.4 — Card-specific covariate adjustment

**What this step does.** Some cards declare covariates that need to be applied **inside the card** rather than as engine-wide foregrounds. Smoking history is the prototypical example: lung-epic adjusts for smoking pack-years as a within-card covariate because smoking is part of the disease's causal pathway, not an external nuisance. Subtracting smoking as a global foreground would erase real lung-disease signal.

The card's JSON declares which covariates are within-card. Engine reads the declaration and applies the adjustment locally before finalizing the card's tier call.

**Inputs.**
- Card's matched verdict from §67.
- Patient metadata from Stage 0 manifest (smoking history, BMI, prior treatment, etc. — whatever the card requires).
- Card JSON covariate declaration.

**Atlas reference.** None.

**Files invoked.**
- Module: `<inside `GAPE_WEB_v13.py`>`
- Per-card covariate transfer functions live in each card's JSON (`within_card_covariates` block).

**The math.** Each within-card covariate is a linear adjustment to the card's confidence value. For lung-epic with smoking:

> adjusted_confidence = raw_confidence + smoking_penalty(pack_years)

where `smoking_penalty` is monotonically negative (heavy-smoker patients have **lower** card-firing confidence because the residual-overlap with non-smoker-derived signature templates is partially explained by smoking). For cardio-epic with BMI:

> adjusted_confidence = raw_confidence + bmi_adjustment(bmi)

where `bmi_adjustment` is non-monotonic (both under- and over-BMI penalize signature match).

The transfer functions are **per-card declared** in the JSON. The engine does not invent them. A new card adding a new covariate adjustment must declare it in the JSON before deployment.

**CMB equivalent.** The **nuisance-parameter marginalization** in cosmological likelihood evaluation, where some parameters (galactic-foreground amplitudes per frequency, calibration offsets, beam-window-function uncertainties) are nuisance parameters integrated over before reporting cosmological parameters. The within-card covariates are nuisance parameters whose effect on the card's confidence must be integrated before reporting the patient's card-firing confidence.

**How the methylome differs in implementation.** Linear adjustment in the current engine; full Bayesian marginalization in Phase E.

**How it's the same in principle.** Both protect the headline result from being driven by an unmodeled covariate.

**Outputs.** Adjusted per-card confidence values. The card's MATCH/NO_MATCH verdict from §67 is **never** changed by §68 — only the confidence is adjusted. A card that fires with adjusted confidence ≤ 0 is reported as "fired below threshold" — the pattern is consistent with the card's template but explained by the within-card covariate.

**Decision points.**
- If patient metadata required for the covariate adjustment is missing (e.g., smoking history not provided for a lung-epic match), the card's confidence is flagged as UNADJUSTED. The card still reports; the report includes an explicit "covariate-uninformative" disclaimer.

**Failure modes.**
- **Missing metadata.** Handled as UNADJUSTED flag (above).
- **Covariate transfer function out of declared range.** A patient with 200 pack-years (impossible) or BMI 60 (extreme) falls outside the card's declared covariate range. Engine clamps to the range boundary and adds a clipped-covariate flag.

**Canonical cross-references.** Recipe §8.4. Each card's `within_card_covariates` JSON block. lung-epic and cardio-epic card READMEs document the per-card transfer functions.

**CPG Plate references.** None.

**Chain-link assignment.** L7 (likelihood, partial — within-card nuisance handling).

---

## §69. Step 8.5 — Stage 8 output: card verdict package

**What this step does.** Consolidate Steps 8.1-8.4 into a Stage 8 output file. One file per cohort run, hashed.

**Inputs.** All Stage 8 intermediate outputs for all patients.

**Atlas reference.** None.

**Files invoked.** Pure I/O.

**The math.** None.

**CMB equivalent.** Per-mission template-match catalog.

**How the methylome differs in implementation.** Per-patient row.

**How it's the same in principle.** Discrete card verdict with confidence and trace.

**Outputs.** <card verdict package — emitted internally by `GAPE_WEB_v13.py`> carrying per patient:
- All applicable cards' verdicts (MATCH_{phase} / NO_MATCH / NEGATIVE_OVERLAP / INSUFFICIENT).
- All applicable cards' raw and covariate-adjusted confidences.
- All applicable cards' matched-element traces.
- Card residual-overlap scores per card.

SHA-256 hashed.

**Decision points.** Cohort proceeds to Stage 9 (report assembly).

**Failure modes.** None at consolidation.

**Canonical cross-references.** Recipe §8.5.

**CPG Plate references.** None.

**Chain-link assignment.** L8 (close).

---

# Stage 9 — Report assembly (the legal boundary layer)

Stage 9 is where the chain of custody ends and the customer's understanding begins. Stages 0 through 8 produced an exhaustive, internally-consistent record of how a patient's IDAT became a card verdict. Stage 9 translates that record into language a non-clinician can act on — without ever telling them anything CPG is not legally permitted to tell them.

The boundary discipline (§10 of Part I) is the entire substance of Stage 9. Every word that goes into the rendered report is checked against:
1. **The Stage 6 engine-to-customer language collapse** (NORMAL / ELEVATED / SIGNIFICANTLY_ELEVATED — never FLOOR_BREACH, never URGENT).
2. **The literature anchors** (we say "in studies, people with your reading had outcome X" — never "you have outcome X").
3. **The cancer prior + family history multiplier** (base-rate context, never deterministic risk).
4. **The CANNOT_SAY list** (no diagnoses, no actions, no future-states, no recommendations).

Stage 9 has seven steps (§70-§76). They run **in order**: the language collapse first, then literature, then prior context, then family history, then sex adjustment, then rendering, then the final legal-boundary check. The last step is a gate: a report that fails the legal-boundary check at §76 does not ship from §77.

---

## §70. Step 9.1 — Customer-facing language collapse

**What this step does.** Apply the Stage 6 engine-to-customer language mapping (introduced at §63 in Part II-B) to every engine tier in the patient's Stage 7 output. After this step, no engine vocabulary survives into the report.

**Inputs.** Stage 7 per-class and per-cell-type tier vector for this patient.

**Atlas reference.** None.

**Files invoked.**
- Module: `<engine-to-customer language collapse inside `GAPE_WEB_v13.py`>`
- Lookup: `tier_breakpoints.json` (engine-to-customer mapping block, same file used at §63).

**The math.** Pure lookup.

Engine → Customer:
- NORMAL → NORMAL
- MARGINAL → NORMAL (slight margin variation)
- DETECTABLE → ELEVATED
- URGENT → SIGNIFICANTLY_ELEVATED
- FLOOR_BREACH → SIGNIFICANTLY_ELEVATED (a customer never sees the breach language)

**CMB equivalent.** The **mission-public release pass**: internal-precision results (microKelvin per HEALPix pixel, with full covariance) get collapsed into the public-release temperature map (Mollweide, smoothed, with confidence band, with the survey mission's narrative wrapper). The internal precision survives in the science release; the public release is human-readable.

**How the methylome differs in implementation.** Per-patient collapse. Same content structure per patient.

**How it's the same in principle.** Internal precision is preserved in the audit trail (Stage 7 output); customer report is the legally-permissible projection of it.

**Outputs.** Patient's tier vector with engine vocabulary replaced by customer vocabulary. Stored in-memory for §71-§76.

**Decision points.** All Stage 9 downstream steps operate on the collapsed vector, not the engine vector.

**Failure modes.**
- **Mapping table missing or wrong version.** Caught by SHA-256 at load. Hard halt.
- **Engine tier outside declared mapping range.** Should never happen if Stage 7 is correct, but caught as INVALID_TIER → report fails legal boundary check at §76 and does not ship.

**Canonical cross-references.** Recipe §9.1. §63 (Step 7.5 mapping definition) in Part II-B.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9 (legal boundary layer).

---

## §71. Step 9.2 — Literature anchors (the reporting-layer translator)

**What this step does.** For each card that fired in Stage 8, attach the literature anchor that gives the patient's reading clinical context — phrased as "in published studies, patients with [your reading range] had outcome [X]" rather than "you have outcome X."

The literature anchor is **not measurement.** It is a reporting-layer translator. The measurement is the A-score and the card verdict. The literature anchor is a legally-permissible language wrapper that lets a customer understand what the measurement means in clinical terms used by their physician.

This is **the only place in the chain** where information from external (non-IAMAtlas, non-IAM-derived) sources enters the customer-facing output. The discipline: literature anchors come from peer-reviewed publications cited in `literature_anchors.json`, with explicit DOI and effect-size mapping. Never paraphrased from secondary sources. Never invented.

**Inputs.**
- Stage 8 card verdict package from §69.
- Patient's per-class A-scores from Stage 4.
- Patient's cellular ages from Stage 6.
- Patient's Mahalanobis distance from Stage 5.

**Atlas reference.** None directly. (The literature anchor table was built once, off-engine, by vetting each citation against the framework's measurement scale. The vetting consulted IAMAtlas; the runtime lookup does not.)

**Files invoked.**
- Module: `<anchor selection inside `GAPE_WEB_v13.py`>`
- Lookup: `literature_anchors.json`. Schema per anchor: `{card_id, phase, reading_range, citation_doi, effect_summary, language_template}`.

**The math.** Lookup keyed by `(card_id, phase, reading_range_bucket)`. The reading_range_bucket is the patient's measurement discretized into the literature anchor's defined buckets (e.g., for breast-epic long_pre_dx with Mahalanobis d ∈ [+1.5, +2.5], the anchor cites the Xu 2020 JNCI cohort outcome at that effect range).

**CMB equivalent.** The **cosmological-significance language** in survey-mission public releases — Planck's "Our results are consistent with the six-parameter ΛCDM model" wrapper around the underlying TT/TE/EE/BB likelihood. The wrapper is what the public understands; the underlying likelihood is what the science is. Both must agree, but they are not the same.

**How the methylome differs in implementation.** Per-patient anchor selection (different patients in the same card get different anchors based on their reading bucket). Public-release language wrappers don't change per reader.

**How it's the same in principle.** A legally and ethically permissible language wrapper around a precise measurement.

**Outputs.** Per-card literature anchor blocks with full citation, effect summary, and the customer-facing language template the renderer (§75) will use.

Stored in-memory; flows to §72.

**Decision points.**
- If a card fires but no literature anchor exists for the patient's reading bucket, the card is reported with a "limited published context" disclaimer rather than fabricated language. **Never invent.**
- If the literature anchor's effect summary conflicts with the patient's measurement direction (rare — would happen if a card mis-fires in opposite sign), the conflict is logged for review and the report uses a conservative "elevated reading; specific clinical interpretation requires physician review" wrapper.

**Failure modes.**
- **Anchor citation 404.** All DOIs in `literature_anchors.json` are checked at engine deployment time. A dead DOI fails deployment, not patient runtime. (Engine refuses to deploy until anchors are fully resolvable.)
- **Anchor table version mismatch.** SHA-256 hashed. Hard halt on mismatch.
- **No anchor at all for a fired card.** Card was deployed without anchor support. Deployment-time check should catch this. If somehow it survives to runtime, the card reports with the "limited published context" disclaimer.

**Canonical cross-references.** Recipe §9.2. The literature_anchors.json schema doc. Per-card READMEs each list their primary citations.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9 (legal boundary layer).

---

## §72. Step 9.3 — Cancer prior context

**What this step does.** Attach population-base-rate context for every card that fires. The cancer prior is the published lifetime incidence (US SEER + age-stratified) for the disease the card targets. It is presented as base-rate context — "US lifetime incidence of breast cancer is approximately 13% in women; your reading places you at the [P] percentile of women your age" — not as a personalized risk number.

**Inputs.**
- Stage 8 fired cards.
- Patient sex, age from Stage 0 manifest.

**Atlas reference.** None.

**Files invoked.**
- Module: `<prior lookup inside `GAPE_WEB_v13.py`>`
- Lookup: `cancer_prior.json`. Schema per prior: `{disease_id, sex, age_decade, lifetime_incidence, citation}`.

**The math.** Lookup keyed by `(card_disease_id, patient_sex, patient_age_decade)`. Returns the base-rate language block to be wrapped by the report renderer.

**CMB equivalent.** The **prior π(θ)** in Bayesian likelihood evaluation. A measurement combined with a prior produces a posterior; reporting the measurement without the prior is misleading because it implies the measurement IS the posterior. The cancer prior is the population prior on disease state, not on the measurement.

**How the methylome differs in implementation.** Per-patient prior because the prior is age/sex-conditioned. CMB priors are typically uniform or simple-Gaussian and don't vary across observations.

**How it's the same in principle.** Both attach base-rate context to a measurement.

**Outputs.** Per-card cancer-prior block with citation. Flows to §73.

**Decision points.**
- If patient sex or age is missing, prior is reported as US-population-overall (less precise but legally available).
- If the card's disease has no published prior at the precision needed (rare — most malignancies have SEER coverage), the card's report block omits the prior context and adds a "population context unavailable" line.

**Failure modes.**
- **Prior table version mismatch.** SHA-256 hashed. Hard halt.
- **Missing patient metadata.** Falls back to overall-population prior; flagged.

**Canonical cross-references.** Recipe §9.3.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---

## §73. Step 9.4 — Family history multiplier

**What this step does.** Apply the family-history multiplier to the cancer prior when the patient's Stage 0 manifest declares first-degree relative history of the disease.

The multiplier itself is a published, peer-reviewed effect size (e.g., for breast cancer: first-degree relative ≈ 1.7× lifetime incidence; bilateral first-degree ≈ 2.5×). The multiplier IS the language; CPG does not modify the multiplier per patient. **The multiplier is applied to the prior, not to the measurement.**

**Inputs.**
- §72 prior block.
- Patient family history declaration from Stage 0 manifest.

**Atlas reference.** None.

**Files invoked.**
- Module: `<multiplier lookup inside `GAPE_WEB_v13.py`>`
- Lookup: `family_history_multiplier.json`. Schema per multiplier: `{disease_id, relative_relationship, multiplier, citation}`.

**The math.** `adjusted_prior = base_prior × family_history_multiplier(disease, relationship)`. Discrete multiplier per relationship category (first-degree single, first-degree multiple, second-degree, etc.). Multipliers are published values, not engine-derived.

**CMB equivalent.** The **prior factorization** in hierarchical Bayesian inference — splitting the prior into a population-level component and an individual-level conditioning component.

**How the methylome differs in implementation.** Per-patient conditioning.

**How it's the same in principle.** Both refine a prior with declared individual information before combining with measurement.

**Outputs.** Adjusted-prior block: `adjusted_prior = base_prior × multiplier`, with both the base rate and the multiplier displayed transparently in the report (the report shows both numbers).

**Decision points.**
- If the patient declares no family history, multiplier defaults to 1.0 and the prior block flows through unchanged.
- If the patient declines to share family history, the report says so explicitly: "Family history not reported; population-base context only."

**Failure modes.**
- **Multiplier table version mismatch.** SHA-256 hashed. Hard halt.
- **Patient declared family history of a disease not in the multiplier table.** Multiplier defaults to 1.0; flagged.

**Canonical cross-references.** Recipe §9.4.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---


**Step 9.4b — Risk-context formula (the integration step).** The cancer prior (§72), age factor, sex factor, family history multiplier (§73), and Stage 8 match magnitude combine into a single per-class context-posterior used by the renderer (§75):

```
posterior_context_class = baseline_prior          (from cancer_prior.json)
                        × age_factor              (US incidence by decade)
                        × sex_factor              (e.g., secretory: female 1.4×, male 1.2×)
                        × fh_factor               (from family_history_multiplier.json IF present; else 1.0)
                        × match_magnitude         (from Stage 8 matrix match for this class)
```

**Framing rule — strict (walkthrough §4 Stage 6).** The doctor report and any future patient report use the phrasing "your reading combined with your risk context suggests…" — NEVER "you have a high probability of…". This is risk-context integration, not diagnosis. The legal-boundary gate (§76) catches probability-statement violations.

**Conditional consumption.** When `patient_metadata.family_history` is absent or empty, `fh_factor = 1.0` and the formula falls back to overall-population context. The audit trail records that family history was not supplied; the doctor report's risk-context section notes the same in the Quality block (§75).

**Differentiation matters.** The formula is what differentiates "your secretory class shows A=1.06 and your mother had no cancer history — keep watching and adjust lifestyle" from "your secretory class shows A=1.06 and your mother died of breast cancer at 52 — discuss accelerated screening with your clinician now." Same A-score, two different risk contexts, two different report framings. This is why Stage 9 is not a pure rendering layer — it's a context-integration layer that the rendering layer (§75) then translates.

## §74. Step 9.5 — Sex-specific risk adjustment

**What this step does.** Some cards are sex-conditioned (breast-epic primarily female; prostate-epic male-only; cervical-secretory female-only). The sex adjustment ensures cards do not report on the wrong sex with population-level context that doesn't apply.

**Inputs.**
- §73 adjusted prior block.
- Patient sex from Stage 0.
- Card sex-condition declaration.

**Atlas reference.** None.

**Files invoked.**
- Module: `<sex conditioning inside `GAPE_WEB_v13.py`>`
- Card JSON sex-condition block.

**The math.** Conditional. If `patient_sex ∈ card.applicable_sex`, card flows through. Else, card is dropped from the report (the engine had a verdict, but the card doesn't report on this patient's sex). For partial-sex cards (e.g., male breast cancer is rare but real), the multiplier shifts to the appropriate male-sex prior at §72-§73 and the card reports with explicit minority-sex context.

**CMB equivalent.** None directly — CMB has no sex-conditioning. The closest analog is **survey-area conditioning**: a polarization measurement reported only over the survey footprint, not extrapolated to the whole sky.

**How the methylome differs in implementation.** Per-patient conditioning is part of the report-assembly logic, not the measurement.

**How it's the same in principle.** Reports only on regions where the measurement framework applies.

**Outputs.** Filtered card set with sex-appropriate context. Flows to §75.

**Decision points.**
- Card excluded by sex: card drops from report; logged in audit trail.
- Card retained with minority-sex context: card reports; explicit minority-sex framing in language.

**Failure modes.**
- **Card sex-condition declaration ambiguous.** Caught at card deployment time, not patient runtime.

**Canonical cross-references.** Recipe §9.5.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---

## §75. Step 9.6 — Report rendering pass

**What this step does.** Compose the final report from the per-card blocks assembled by §70-§74. The renderer is **template-driven** — each card has a declared report template; the renderer fills slots in the template with the patient's specific values. The renderer does not invent language. Every customer-facing word is either a template literal or a slot value from one of the upstream steps.

**Inputs.** All per-card blocks from §70-§74.

**Atlas reference.** None.

**Files invoked.**
- Module: `<report rendering inside `GAPE_WEB_v13.py`>`
- Per-card templates: <inside `GAPE_WEB_v13.py` — no standalone module file at present writing> (Markdown with slot syntax).
- Global template: `<global report skeleton — inside `GAPE_WEB_v13.py`>` (the report skeleton — patient header, summary, per-card sections, disclaimers).

**The math.** None — string composition via Mustache-style templating.

**CMB equivalent.** The **publication LaTeX rendering** of cosmological-parameter results — a typeset paper-grade document assembled from a parameter table and a citation database, with no manual prose interpolated at rendering time.

**How the methylome differs in implementation.** Per-patient rendering with per-patient slot values.

**How it's the same in principle.** Mechanical composition of declared template + declared values. No interpretive room at this step.

**Outputs.** A rendered Markdown report. Per-patient file at `<draft report — emitted internally by `GAPE_WEB_v13.py`>`. Status: DRAFT until §76 clears it for shipping.

**Decision points.** Report DRAFT flows to §76 legal boundary check.

**Failure modes.**
- **Missing slot value.** Caught by the renderer — any unfilled slot triggers REPORT_INCOMPLETE; report does not advance to §76.
- **Template syntax error.** Caught at card deployment time.

**Canonical cross-references.** Recipe §9.6.

**CPG Plate references.** None.

**Chain-link assignment.** Above L9.

---

## §76. Step 9.7 — Legal boundary check (the gate)

**What this step does.** The final check before a report ships. The DRAFT report from §75 is scanned for any language that violates the CANNOT_SAY list. A report that contains forbidden language **does not ship**. The patient's run is flagged for manual review; the engine does not silently degrade.

**Inputs.** DRAFT report from §75. The CANNOT_SAY list.

**Atlas reference.** None.

**Files invoked.**
- Module: `<legal-boundary gate inside `GAPE_WEB_v13.py`>`
- The CANNOT_SAY list: `<legal-boundary CANNOT_SAY list — TBD; currently enforced inline within `GAPE_WEB_v13.py`>`.

**The CANNOT_SAY list (current as of v1):**
- Diagnostic statements: "You have [disease]", "You are at risk of [disease]", any verb conjugation suggesting determinism.
- Action statements: "You should [action]", "You must [action]", any recommendation involving medication, dosage, procedure, lifestyle change beyond "discuss with your physician."
- Future-state statements: "You will [outcome]", any statement about future probabilities phrased deterministically.
- Treatment statements: any reference to specific drugs, dosages, or procedures.
- Prevention statements: any claim about preventing a disease.

**The CAN_SAY list (by design):**
- Measurement statements: "Your immune class A-score is [value]."
- Percentile statements: "This places you at the [percentile] of the healthy reference population at your age."
- Literature anchor statements: "In published studies (citation), patients with A-scores in this range had outcome [X]."
- Base-rate statements: "US lifetime incidence of [condition] is approximately [N]%."
- Family history conditioning: "Family history of [condition] in a first-degree relative raises the population base rate to approximately [N × multiplier]%."
- Recommendation to discuss with physician: "Consider discussing this reading with your physician."

**The math.** Regex + keyword scanning. Each forbidden phrase or pattern in CANNOT_SAY triggers a violation flag. A report with ≥1 flag halts. A report with 0 flags clears.

**CMB equivalent.** The **collaboration internal review** before public release of a cosmological-parameter measurement. A Planck paper does not ship from the analyst to arxiv directly — it passes through collaboration-wide internal review where the language and the conclusions are checked against the discipline's standards. The CANNOT_SAY check is the CPG version.

**How the methylome differs in implementation.** Automated check rather than human review (because it runs per-patient at scale). Periodic manual sampling of cleared reports is part of QA discipline.

**How it's the same in principle.** Both prevent a measurement from being communicated in a way that misrepresents what was measured.

**Outputs.** Cleared report: `<cleared report — emitted internally by `GAPE_WEB_v13.py`>`. Or HALT_FOR_REVIEW flag with detailed violation report.

**Decision points.**
- 0 violations → CLEARED. Report flows to Stage 10.
- ≥1 violations → HALT_FOR_REVIEW. Report does not ship until a human reviewer fixes the source (template, literature anchor, or covariate language) and re-renders.

**Failure modes.**
- **False positive (cleared CAN_SAY language flagged).** Adjust the CANNOT_SAY regex; do not weaken the discipline.
- **False negative (forbidden language not flagged).** Add the missed pattern to CANNOT_SAY immediately; re-scan all already-cleared reports from the current cohort run.

**Canonical cross-references.** Recipe §9.7. **§10 of Part I** (the foundational legal boundary discussion).

**CPG Plate references.** None.

**Chain-link assignment.** Above L9 (legal boundary layer — the gate).

---

# Stage 10 — Delivery

Stage 10 is the smallest stage. The report has been cleared by §76; the engine now packages it, routes it, and closes the audit trail. Three steps.

---

## §77. Step 10.1 — Report packaging

**What this step does.** Wrap the cleared Markdown report into the delivery format the customer or partner expects. Default is PDF via the LaTeX→PDF rendering pipeline. Some lab partners receive HTML.

**Inputs.** Cleared report from §76.

**Atlas reference.** None.

**Files invoked.**
- Module: `<delivery packaging inside `GAPE_WEB_v13.py` — or `web.commercial.py` orchestrator when built>`
- Templates: `<delivery templates — TBD>` (LaTeX, HTML).

**The math.** None.

**CMB equivalent.** Public-release file packaging (FITS, HEALPix WCS, with documented file format).

**Outputs.** `<final report — emitted internally by `GAPE_WEB_v13.py`>` with embedded checksums and engine version metadata.

**Decision points.** Cohort proceeds to §78.

**Failure modes.** Rendering pipeline failure → flagged; cleared report retained as Markdown for manual repackaging.

**Canonical cross-references.** Recipe §10.1.

**CPG Plate references.** None.

**Chain-link assignment.** L1 (audit-trail close).

---

## §78. Step 10.2 — Delivery channel routing

**What this step does.** Route the packaged report to the configured delivery channel (customer portal, lab partner API, secure email, etc.). The channel is per-customer or per-partner; the engine reads the delivery configuration and sends.

**Inputs.** Packaged report from §77. Customer/partner delivery configuration.

**Atlas reference.** None.

**Files invoked.**
- Module: `<channel routing inside `GAPE_WEB_v13.py` — or `web.commercial.py` orchestrator when built>`
- Configuration: `<per-customer delivery configuration — TBD>`.

**The math.** None.

**CMB equivalent.** Survey data-release distribution (Planck Legacy Archive, NASA's HEASARC).

**Outputs.** Delivery confirmation per recipient; delivery failures retried with bounded retry policy.

**Decision points.**
- Delivery success → §79.
- Delivery failure after retry exhaustion → flagged; report queued for manual handling.

**Failure modes.**
- Channel authentication failure → bounded retry; manual handoff after exhaustion.
- Channel throughput limit hit → queue with rate-limit backoff.

**Canonical cross-references.** Recipe §10.2.

**CPG Plate references.** None.

**Chain-link assignment.** L1.

---

## §79. Step 10.3 — Audit trail capture (the chain closes)

**What this step does.** The final step. Every output from every stage for this patient — Stage 0 manifest hash, Stage 1-3 β matrix hashes, Stage 4 A-scores, Stage 5 Mahalanobis, Stage 6 cellular ages, Stage 7 tiers, Stage 8 card verdicts, Stage 9 cleared report hash, Stage 10 delivery confirmation — is captured into an audit-trail record.

The audit trail is the answer to "show me how this number was produced." A question about any patient's reading should be answerable in full by reading the audit trail back through the chain.

**Inputs.** Every prior stage's hashed output for this patient.

**Atlas reference.** The atlas SHA-256 fingerprint used at every stage that consulted the atlas is recorded.

**Files invoked.**
- Module: `<audit-trail capture — TBD when `web.commercial.py` orchestrator is designed>`
- Audit log: `<audit log — TBD per orchestrator design>`.

**The math.** None — pure logging.

**CMB equivalent.** The **observation log** of a mission, plus the **likelihood pipeline record** of the analysis, plus the **publication-time methods section** — collapsed into a single per-observation audit record.

**How the methylome differs in implementation.** Per-patient audit. Each patient gets a full chain trace.

**How it's the same in principle.** Both record everything needed to reproduce a measurement after the fact.

**Outputs.** Per-patient audit JSON with:
- Stage 0-10 input/output hashes.
- IAMAtlas SHA-256 fingerprint at every stage that consulted it.
- Engine version + module versions at every stage.
- Decision-point flags from every stage.
- Failure-mode flags from every stage.
- L9 null-suite results (if the patient was part of a sealed VAL cohort).
- Final cleared report hash.

The audit log is **append-only**. A patient's audit record is written once and never modified. Re-runs produce new run_id's with their own audit records.

**Decision points.** The chain ends here. Audit trail is the closing record.

**Failure modes.** Audit capture failure is the only Stage 10 failure that triggers a hard halt — a report cannot ship without its audit record. If the audit log write fails, the report is held at §78.

**Canonical cross-references.** Recipe §10.3. **§3 of Part I** (the chain-link-to-stage mapping that this audit closes).

**CPG Plate references.** None directly — but the audit trail of a sealed VAL becomes the data behind any future plate produced from that VAL.

**Chain-link assignment.** L1 closes the loop. The chain is complete.

---

**End of Part II-C (Stages 8 through 10, §65-§79).**

**End of Part II — the complete operational chain from IDAT-on-server to audit-trail-closed.**

Part III follows with the chain-integrity scaffolding (L9 null suite, synthetic patients, VAL sealing protocol — §80-§91). Part IV with failure modes and decision trees (§92-§96). Part V with reference tables (§97-§102).

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*


> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them across all parts; this particular part (Foundations / L9 machinery)
> was already clean of fabricated paths in v1, so the disclaimer is included here
> only for version-consistency across the SOP. All real paths in v1.1 are documented
> in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be treated as
> not-yet-existing until verified against the repo.

---


**Continues from Part II-C (§65-§79). This document contains §80-§91 — the L9 null suite, the synthetic patient generator (FFP10/NPIPE analog), and the VAL sealing protocol.**

---

# Part III — Chain-integrity scaffolding (runs ABOVE, not inside, the chain)

L9 is structurally different from L1-L8. The first eight chain links run **inside** the operational chain — every patient's IDAT passes through L1, L2, L3, ..., L8 in sequence. L9 does not. L9 runs **above** the chain, on cohorts after they complete Stages 0-10, and tests whether the headline result from any sealed VAL would survive deliberate adversarial perturbation.

The discipline borrowed directly from CMB cosmology: **no result ships until it has been beaten through nulls.** Planck does not publish a power-spectrum measurement without first running the null suite (jackknife rotations, frequency-difference tests, FFP10 end-to-end sims). CPG does not seal a CPG-VAL without first running the 8-null suite. Sealed = passed all declared nulls. Restated = failed one or more declared nulls; the result is preserved with the limitation declared honestly. Retracted = failed in a way that invalidates the result entirely; the result is withdrawn.

The eight nulls (N1-N8) are pre-specified. Each VAL declares which subset applies (every VAL declares N1, N7, N8 at minimum; cohorts with age/sex/plate metadata add N2, N3, N5; multi-cohort VALs add N4; signal-direction-dependent VALs add N6). The declaration is in the VAL's PREREG document; the verdict is in the VAL's OUTCOME document.

Phase A (closed 2026-05-30) built the L9 framework, ran all 7 existing Family A CPG-VALs through it, and produced the first sealed-vs-restated split: 5 sealed (VAL-001, VAL-002, VAL-003, VAL-005, VAL-007), 2 restated (VAL-004 bimodality FAIL-on-direction, VAL-006 chr6 MHC FAIL-on-look-elsewhere). The framework caught these before any clinical use — that is what the discipline is for.

The L9 grade lifted from D (no null framework) to A- (framework live, 5 of 7 sealed, 2 known limitations Phase A2.1 closes). The remaining grade weight is in the production-precision N7 recovery upgrade (Phase A2.1) and signed-direction injection (Phase A2.1).

---

## §80. L9.0 — The 8-null suite framework (`cpg_null_runner.py`)

**What this step does.** The L9 framework is the orchestration layer that runs the eight standard nulls against any sealed VAL artifact. A VAL provides its `per_sample.csv` (per-patient reconstructed quantities), its `results.json` (declared headline + declared nulls), and its PREREG (declared statistical test). The framework runs each declared null, returns PASS/FAIL + p-value + 95% CI, and produces a `null_results.json` artifact that gets bundled with the VAL.

**Inputs.**
- `per_sample.csv` — per-patient outputs from the chain (A-scores, Mahalanobis distance, cellular age, card verdict, ...).
- `results.json` — the VAL's declared headline statistic + the declared nulls it must pass.
- `prereg.json` — pre-registration document declaring the statistical test, the cohort, the significance threshold, the multiple-testing correction approach.

**Atlas reference.** None directly. The L9 framework operates on chain outputs, not atlas reference. The IAMAtlas was consulted at Stage 2, 4, 5, 6 of the chain; by L9 time, it has done its work and is no longer needed.

**Files invoked.**
- `Biological_Physics/chain_of_custody/L9_null_suite/cpg_null_runner.py` — the orchestration module.
- Per-null modules: `cpg_null_runner.py` implements N1 through N8 as separate functions; each is callable independently for VAL-specific subsetting.
- `Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py` — N6 and N7 invoke this.

**The math.** None directly at the framework level. Each null has its own math (§81-§88). The framework is dispatch logic + multiple-testing aggregation + sealing-verdict computation.

**Sealing verdict logic.** Given the VAL's declared null set `D ⊆ {N1, N2, ..., N8}` and the per-null verdicts `V_n ∈ {PASS, FAIL}` for n ∈ D:
- **SEALED**: All declared nulls PASS. Headline result preserved.
- **RESTATE**: One declared null FAILs; the FAIL is documented and the headline is preserved with explicit caveat. (Used when the failure is a documented limitation — e.g., VAL-004 cohort-pooled bimodality — that does not invalidate the underlying signal but limits its scope.)
- **RETRACT**: Two or more declared nulls FAIL; the headline is withdrawn. (Not yet triggered in Phase A; the framework expects this verdict to be rare but real.)

**CMB equivalent.** Planck's **null-test orchestration framework** in NPIPE: it runs the full null-test battery (TT-TE-EE-BB cross-tests, half-mission-difference, year-difference, detector-set-difference, scanning-direction-difference, FFP10 end-to-end recovery) against every cosmological-parameter release. A release does not ship until every null test passes the declared significance threshold (typically 3σ). The CPG L9 framework is the methylome implementation of that orchestration discipline — one module that runs all declared nulls, returns one verdict per null, aggregates into a SEALED/RESTATE/RETRACT decision.

**How the methylome differs in implementation.** Planck's null suite is the cosmology community's accumulated wisdom over 30 years; the CPG null suite is in its first generation (Phase A 2026-05-30). The structural analog is exact; the operational maturity differs by decades. Phase A2.1 is the immediate maturity upgrade (production-precision N7, signed injection).

**How it's the same in principle.** A measurement that has not been beaten through nulls is not yet a measurement — it is a candidate measurement. The null suite is what converts a candidate into a sealed claim.

**Outputs.** Per-VAL `null_results.json` carrying:
- The declared null set.
- The per-null verdict + p-value + 95% CI.
- The aggregated SEALED/RESTATE/RETRACT verdict.
- The framework version + invocation timestamp + module SHA-256 hashes (for chain-of-custody on the L9 audit itself).

**Decision points.** SEALED → VAL is sealed; result enters production. RESTATE → VAL is preserved with declared limitation. RETRACT → result withdrawn; rerun chain with corrected upstream or accept the negative finding.

**Failure modes.**
- **Framework dispatch failure** (a null module crashes during invocation). Caught by per-null try/except; the framework returns FAIL with a `framework_error` flag so the verdict cannot be silently turned into PASS.
- **Declared null not implemented** (a VAL declares N5 but no plate metadata exists). Returns INSUFFICIENT_METADATA for that null; the framework aggregates to RESTATE rather than SEALED because the declared null couldn't run.

**Canonical cross-references.** Recipe §11 (null discipline). Roadmap §10.2.1 Phase A. §12 of Roadmap (Phase A results).

**CPG Plate references.** **Plate 4 Panel E** (MCMC Coverage Map) shows the stromal galactic mask — the systematic that the L9 framework formalizes as a declared limitation rather than a hidden contamination. The chain-of-custody discipline lives in declaring the mask, not in pretending it isn't there.

**Chain-link assignment.** L9 (the framework itself).

---

## §81. L9.1 — N1: HC-permutation null (case/HC label shuffle)

**What this null tests.** The most fundamental: would the VAL's headline statistic appear by chance if you shuffled the case-vs-HC labels among the patients? If yes, the signal is a label artifact, not biology.

**Inputs.** `per_sample.csv` with patients labeled as case or HC. The VAL's declared statistic (e.g., Cohen's d on immune A-score for breast-pre-dx >10y).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N1_label_permutation()`.

**The math.** Permute the case/HC label vector 1,000 times (configurable, default 1,000). Per permutation: recompute the declared statistic on the permuted labels. Per VAL: the null distribution is the 1,000 permuted statistic values. The two-sided p-value is the fraction of permuted statistics with absolute value ≥ |observed statistic|. The 95% CI is the [2.5th, 97.5th] percentile of the permuted distribution centered on zero.

**PASS criterion.** Observed statistic outside the 95% CI of the permuted distribution. Equivalently, p < 0.05 (or whatever the PREREG declared).

**FAIL criterion.** Observed statistic within the 95% CI of the permuted distribution.

**CMB equivalent.** **Sky rotation null** in CMB analysis: take the cosmological signal map, rotate it 90° / 180° / 270° relative to the foreground masks, recompute the power spectrum. The cosmological signal should survive the rotation (it's isotropic); a foreground residual should look very different after rotation. The CPG label-permutation test is the case/HC analog: real biological signal should depend on label assignment (it's "the property that defines cases"); statistical noise should not.

**How the methylome differs in implementation.** Permutation is on a discrete label vector rather than a continuous rotation. The mathematical structure of the null is identical: shuffle the "is this the signal-bearing axis?" assignment and check whether the headline statistic survives.

**How it's the same in principle.** Both ask: is the headline statistic explained by the property the analyst calls signal-bearing, or would any random property of the data produce the same statistic? If the latter, the headline is an artifact.

**Outputs.** `N1_verdict ∈ {PASS, FAIL}`, `N1_pvalue`, `N1_95CI`, `N1_permutation_distribution.json` (the full distribution, for downstream verification).

**Decision points.** PASS → contribute to SEALED. FAIL → contribute to RESTATE/RETRACT.

**Failure modes.**
- **Insufficient permutation budget.** Default 1,000; for borderline VALs, framework can run 10,000 or 100,000. The framework declares the budget per VAL in `prereg.json`.
- **Permutation distribution non-Gaussian.** The framework reports the empirical percentile rather than assuming Gaussian — this is robust to non-Gaussian null distributions, which DO occur in methylation data (the distributions are often skewed).

**Canonical cross-references.** Recipe §11.1. Roadmap §10.2.1 Phase A1 (N1 in the 8-null specification). §12 of Roadmap (Phase A results: 7 of 7 VALs ran N1).

**CPG Plate references.** None directly; the per-VAL permutation results live in the audit-trail artifacts.

**Chain-link assignment.** L9.

---

## §82. L9.2 — N2: Age-decade-stratified permutation

**What this null tests.** The label-permutation test (N1) protects against random label-shuffling artifacts. N2 protects against a sneakier confound: what if cases are simply older than HC, and the "signal" is just an age-drift artifact dressed up as a disease signal? N2 permutes labels WITHIN age decades — so a case can only become a synthetic-HC by being swapped with an actual HC of the same age. This breaks the case/HC association while preserving the age structure.

**Inputs.** `per_sample.csv` with case/HC labels AND patient ages.

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N2_age_stratified_permutation()`.

**The math.** Group patients by age decade. Per permutation: within each decade independently, shuffle the case/HC labels. Concatenate the per-decade permuted labels into a full permuted label vector. Recompute the statistic. Repeat 1,000 times. Apply the same PASS/FAIL logic as N1.

**Why this matters.** Age-drift in methylation is well-documented and substantial (the Horvath, Hannum, PhenoAge clocks all exploit it). If a VAL's headline is "cases differ from HC at p<0.001 on immune A-score" but the cases are systematically 5 years older than HC, the N1 permutation would still pass (label shuffling destroys the case/HC structure), but N2 would FAIL (within-decade shuffling preserves age, eliminating the age-drift confound, and the signal collapses).

**CMB equivalent.** **Stratified jackknife resampling** in CMB cross-correlation analyses — when measuring CMB-vs-galaxy-survey cross-correlation, the jackknife is performed within redshift slices to ensure the cross-correlation is not driven by redshift-dependent foregrounds. The methylome's age-stratified permutation is the same operation in a different parameter space.

**How the methylome differs in implementation.** Age decades rather than redshift slices. Same stratified-resampling principle.

**How it's the same in principle.** Both protect headline statistics from being driven by an unmodeled covariate that correlates with both the "signal" axis and a potential confounder.

**Outputs.** `N2_verdict`, `N2_pvalue`, `N2_95CI`, `N2_within_decade_distribution.json`.

**Decision points.** Same as N1.

**Failure modes.**
- **Age metadata missing.** Returns `INSUFFICIENT_METADATA`. Framework aggregates this as a missing-but-required null → contributes to RESTATE if N2 was declared.
- **Sparse age decades.** If a decade has only 1-2 patients in case or HC, permutation within that decade is degenerate. Framework either pools that decade with the adjacent decade or excludes it from N2; the choice is per-VAL declared in PREREG.

**Canonical cross-references.** Recipe §11.2. Roadmap §10.2.1 Phase A1 (N2 in the spec).

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

## §83. L9.3 — N3: Sex-stratified permutation

**What this null tests.** Analogous to N2 but stratifying on sex rather than age. Protects against the headline being driven by sex differences in methylation (which exist at thousands of autosomal CpGs, not just chrX/chrY) that happen to correlate with disease-status assignment.

**Inputs.** `per_sample.csv` with case/HC labels AND patient sex.

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N3_sex_stratified_permutation()`.

**The math.** Same as N2 with sex as the stratifying variable instead of age. Patients are grouped by sex; per permutation, case/HC labels are shuffled within each sex independently.

**Why this matters.** Many cohort designs are sex-skewed (the breast cancer VALs are mostly female; prostate VALs are exclusively male; some lifestyle cohorts skew female because women enroll more). If the VAL's headline is "cases differ from HC" but the cases happen to be 90% female and the HC 60% female, sex-skewed methylation differences contaminate the signal. N3 catches this.

**CMB equivalent.** **Sub-mission-split null tests** (Planck's half-mission-1 vs half-mission-2 cross-check). The mission is split along a property that should not affect the cosmological signal (which half of the observing schedule); if the headline differs across the split, it's a systematic.

**How the methylome differs in implementation.** Binary sex split rather than continuous mission timeline. Same structural principle.

**How it's the same in principle.** Both protect headlines from being driven by a binary structural property of the cohort that correlates with both the "signal" axis and a potential confounder.

**Outputs.** `N3_verdict`, `N3_pvalue`, `N3_95CI`.

**Decision points.** Same as N1.

**Failure modes.**
- **Sex metadata missing or sex-balanced cohort.** If cohort is too sex-imbalanced to stratify (e.g., all-female cohort), returns `INSUFFICIENT_VARIATION`. Framework treats this as N3 not applicable; not a contribution to RESTATE because the null genuinely cannot run.

**Canonical cross-references.** Recipe §11.3. Roadmap §10.2.1 Phase A1.

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

## §84. L9.4 — N4: Cohort-split replication

**What this null tests.** The headline statistic should replicate when the cohort is split in half — both halves should show the same effect direction and roughly comparable magnitude. A statistic that only appears in one half (or appears with opposite sign in the two halves) is a cohort-specific artifact, not a generalizable signal.

**Inputs.** `per_sample.csv` with a cohort-split variable (typically random 50/50 split within the cohort, OR within-batch split when batches differ in collection date/site).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N4_cohort_split_replication()`.

**The math.** Random 50/50 split of the patient set, stratified on case/HC and (when available) age/sex. Compute the headline statistic separately on Half-A and Half-B. Verdict:
- **PASS**: Same sign in both halves; magnitudes overlap at 95% CI.
- **FAIL_SIGN**: Opposite signs in the two halves.
- **FAIL_MAGNITUDE**: Same sign but magnitudes outside 95% CI of each other (large variance, weak signal).

The split is repeated 100 times (different random seeds); the framework reports the fraction of splits that PASS, FAIL_SIGN, FAIL_MAGNITUDE. Default PASS criterion: ≥ 80% of splits PASS.

**Why this matters.** A signal that only appears in one half-cohort and not the other is not a sealed result — it's a candidate finding that requires independent replication. CPG's N4 self-replicates within the cohort; external replication (different lab, different intake, different cohort entirely) is a higher bar that lives outside the L9 framework as the **Family B** validation discipline.

**CMB equivalent.** **Half-mission-difference null** in Planck. Planck-2018 ran the full likelihood on half-mission-1 and half-mission-2 separately, verified that cosmological parameters agreed to within their uncertainties. A parameter that differed by >2σ between halves was flagged for systematic investigation. CPG's N4 is the methylome equivalent.

**How the methylome differs in implementation.** Random patient split rather than temporal mission split. Same structural test.

**How it's the same in principle.** Both ask: does the signal exist independently in two sub-samples, or only in their aggregate?

**Outputs.** `N4_verdict ∈ {PASS, FAIL_SIGN, FAIL_MAGNITUDE}`, `N4_pass_rate_across_seeds`, per-split Half-A and Half-B statistics.

**Decision points.** PASS → SEALED contribution. FAIL_SIGN → strong RETRACT contribution. FAIL_MAGNITUDE → RESTATE contribution.

**Failure modes.**
- **Cohort too small to split** (<40 cases or <40 HC per half). Returns `INSUFFICIENT_SAMPLE`. Not a FAIL; null couldn't run. Framework treats as INSUFFICIENT_METADATA.
- **Within-batch split unavailable** (single-batch cohort). Falls back to random 50/50 split; per-VAL declared in PREREG which split mode is used.

**Canonical cross-references.** Recipe §11.4. Roadmap §10.2.1 Phase A1.

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

## §85. L9.5 — N5: Plate/array-position null

**What this null tests.** Illumina methylation arrays come in 8-sample chips arranged 2×4 on a 96-well plate. Position on the chip and plate can introduce systematic effects (edge effects, gradient effects, scan-order effects). N5 tests whether the headline statistic survives when patients are stratified by plate position — if the statistic appears mostly in patients at one chip position and disappears at others, it's a position artifact.

**Inputs.** `per_sample.csv` with case/HC labels AND chip position metadata (Sentrix barcode + row/column position).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N5_plate_position_null()`.

**The math.** Group patients by chip row × column position (8 position bins). Per position bin: compute the headline statistic restricted to that bin. Test: is the statistic consistent across positions, or does it vary systematically?

Two sub-tests run jointly:
1. **Position-stratified permutation** (analogous to N2): permute case/HC labels within each position bin, recompute statistic, build null distribution.
2. **Position-amplitude regression**: regress per-patient signal strength against position; positive slope coefficients indicate position-driven signal.

**PASS criterion.** Both sub-tests pass at p<0.05.

**Why this matters.** Plate effects are real and well-documented in methylation. The strongest are edge effects (chip rows 1 and 8 sometimes show different hybridization efficiency than rows 4-5), batch-position correlations (cases run in batch 1, HC in batch 2), and scanner-drift effects (early-day samples vs late-day samples). N5 is the systematic check that catches these before they're called signal.

**CMB equivalent.** **Scan-direction null** in CMB analysis. CMB experiments scan the sky in specific patterns (Planck's rings, ACT's surveys); scan-direction systematics (asymmetric beam, 1/f noise, hwp synchronous) appear as differences in the headline statistic restricted to specific scan directions. The CPG plate-position null is the methylome implementation.

**How the methylome differs in implementation.** Discrete position bins on a chip rather than continuous scan azimuth. Same structural principle.

**How it's the same in principle.** Both test whether the headline signal varies systematically across a property of the measurement apparatus that should not affect biological signal.

**Outputs.** `N5_verdict`, `N5_pvalue`, `N5_position_stratified_distribution`, `N5_position_regression_slope`.

**Decision points.** PASS → SEALED contribution. FAIL → RESTATE contribution.

**Failure modes.**
- **Position metadata missing.** Returns `INSUFFICIENT_METADATA`. Framework reports as N5-not-runnable; if N5 was declared, contributes to RESTATE.
- **Single-plate cohort.** N5 reduces to within-plate position-only; cross-plate effects can't be tested.

**Canonical cross-references.** Recipe §11.5. Roadmap §10.2.1 Phase A1.

**CPG Plate references.** None directly. (The methylome sky-map plates 1-4 use HEALPix projection, which is orthogonal to physical-chip position.)

**Chain-link assignment.** L9.

---

## §86. L9.6 — N6: Injection-recovery null

**What this null tests.** Inject a known synthetic signal into a known-HC patient population at known strength. Run the full chain. Verify the chain recovers the injected signal at the injected strength and direction. This is the **canonical "does the chain work?" test**.

**Inputs.** A HC-only β matrix from the cohort. A declared injection signal (effect size, direction, target class). The full Stages 0-10 chain.

**Atlas reference.** Indirect — the injection signal is built relative to the IAMAtlas reference posterior so the chain treats it as a biological signal rather than a numerical artifact.

**Files invoked.** `cpg_null_runner.py::run_N6_injection_recovery()`. Calls `synthetic_patient_generator.py` to construct the injected β matrix.

**The math.**
1. Start with a HC-only β matrix (n=N_HC patients).
2. Sample a target subset of M patients (M ≤ N_HC/2).
3. For each of the M selected patients, modify their β at the target class's marker CpGs to shift their per-class A-score by the injected effect size d_inject. The shift direction is the disease's declared direction (positive for case-elevated signals).
4. Run Stages 0-10 on the modified cohort with the M as synthetic-cases and the remaining N_HC - M as synthetic-HC.
5. Recover the headline statistic. Compare against injected d_inject.

**PASS criterion.** Recovered effect size within ±20% of injected at the target class, with correct direction (sign).

**Known Phase A2.1 limitation.** The current N6 uses a simplified correlation-based scoring in the recovery tester rather than the full H(β)/H_min production formula. This catches sign-of-effect and order-of-magnitude recovery but underestimates absolute recovery by ~5×. **Phase A2.1 deliverable**: swap the simplified scorer for the full production chain. Until then, N6 PASS means "chain recovers signal in the right direction with magnitude within an order of magnitude," not "chain recovers signal at production precision."

**Known Phase A2.1 limitation (signed direction).** The current synthetic patient generator injects case-vs-HC signal in positive direction only. Real-data signals with negative direction (e.g., the PC2 T-cell SUPPRESSION axis in VAL-005) appear as absolute-value matches in N6. **Phase A2.1 deliverable**: signed-direction injection.

**CMB equivalent.** The **signal-injection-and-recovery test** that EVERY CMB pipeline runs as the most important null. Planck NPIPE injects synthetic CMB realizations (FFP10 simulations) at known cosmological parameters into the timestream and verifies the pipeline recovers those parameters within their declared uncertainties. The CMB community treats injection-recovery as the irreducible minimum: a pipeline that cannot recover injected signal is not a pipeline yet, regardless of how sophisticated its claimed analyses are.

**How the methylome differs in implementation.** Signal injection at the β-matrix level rather than the timestream level (because the methylome's L1 is per-sample IDAT, not continuous timestream). The injection is at the L3 boundary (clean β matrix) and propagates forward through L4-L8 to the headline. Phase A2.2 (implicit, deferred to Phase C) will extend injection-recovery to the IDAT level for full L1-through-L8 coverage.

**How it's the same in principle.** Both test that the pipeline recovers a known truth from known input. A pipeline that fails this is not a pipeline; it is an unverified system.

**Outputs.** `N6_verdict`, `N6_recovery_ratio` (recovered / injected), `N6_sign_match`, per-injection-strength sweep table.

**Decision points.** PASS → SEALED contribution. FAIL → either RETRACT (sign wrong) or RESTATE (sign right but magnitude off).

**Failure modes.**
- **Recovery sign opposite injection.** Indicates upstream sign error somewhere in Stages 0-8. The chain reports a positive signal when truth is negative — the worst failure mode possible. Triggers immediate halt and audit.
- **Recovery magnitude << injection.** Indicates information loss in the chain (most likely in foreground subtraction or in deconvolution). Framework reports the ratio and flags for review.
- **Recovery magnitude >> injection.** Indicates an amplification systematic — the chain is amplifying signal in a way that doesn't correspond to truth. Equally worrying as opposite sign.

**Canonical cross-references.** Recipe §11.6. Roadmap §10.2.1 Phase A2 (synthetic patient generator). §12.1 of Roadmap (Phase A known limitations).

**CPG Plate references.** None directly.

**Chain-link assignment.** L9.

---

## §87. L9.7 — N7: End-to-end synthetic-patient simulation

**What this null tests.** A more demanding version of N6: instead of starting from real HC patients and modifying them, generate fully synthetic patients from a known cosmological-equivalent truth model, run them through the full chain L1-L8, and verify the chain reproduces the truth.

This is structurally equivalent to Planck's **FFP10 (Full Focal Plane 10) simulations** or **NPIPE end-to-end sims** — the gold-standard discipline of "the only thing you can fully trust about your pipeline is what it does on data you fully generated yourself."

**Inputs.** A truth specification: declared per-class fractions, declared per-cell-type fractions, declared cellular age per class, declared cohort design (n_case, n_HC, age distribution, sex distribution, disease signal strength). Optional: declared plate-position structure, declared batch structure.

**Atlas reference.** Indirect. The synthetic patient generator (`synthetic_patient_generator.py`) consults IAMAtlas to draw per-class posterior β values, modulates them by the declared truth, and outputs synthetic β matrices.

**Files invoked.** `cpg_null_runner.py::run_N7_end_to_end_simulation()`. Calls `synthetic_patient_generator.py`. Drives the full chain.

**The math.**
1. From the truth specification, the synthetic patient generator produces an `n_patients × n_CpGs` β matrix where each row is a synthetic patient drawn from IAMAtlas with the declared per-class mixing fractions and declared per-class A-score offsets.
2. The β matrix is written to a synthetic-cohort manifest as if it came from a real Illumina array.
3. The full chain L1-L8 runs on the synthetic cohort.
4. The framework compares per-patient recovered quantities (per-class A-scores, Mahalanobis distance, cellular age per class) against the truth specification.

**PASS criteria** (multi-channel):
- **Per-class A-score recovery**: bias < 5%, RMSE within declared tolerance.
- **Cellular age recovery**: bias < 2 years per class, saturation flags consistent with truth-out-of-range patients.
- **Card verdict recovery**: when truth specifies a card-firing pattern, the chain's Stage 8 verdict matches at ≥80% concordance.
- **Sign correctness**: 100% sign agreement on declared-direction signals.

**Phase A status.** Phase A built the synthetic patient generator and ran a basic N7 sweep across the 7 Family A VALs. The Phase A N7 implementation uses the simplified scorer (§86 limitation). Phase A2.1 upgrades to production-precision recovery — at which point N7 becomes the methylome's FFP10/NPIPE equivalent at full strength.

**CMB equivalent.** **FFP10 / NPIPE end-to-end pipeline simulations** — the Planck collaboration's most expensive computational discipline. They run thousands of full-pipeline simulations from declared cosmological truths through every Planck data-processing stage, and they verify that the recovered cosmological parameters match the truths within declared uncertainties. **CPG's N7 is structurally identical** — the methylome implementation of the same discipline. Phase A2.2 (implicit; HEALPix already exists for the plates) extends synthetic generation to HEALPix-pixelized methylome representations, which is the prerequisite for Phase C correlation-structure analyses (TODO 2.1 C(d), TODO 2.2 bispectrum) to operate on synthetic data the same way they would on real data.

**How the methylome differs in implementation.** Synthetic generation is at the β-matrix level rather than the bolometer-timestream level. The chain enters at Stage 0 (intake) rather than L1 (timestream); the L1-through-L3 chain steps operate trivially on synthetic input because no real wet-lab QC is involved. The discipline is the same; the implementation skips the lab.

**How it's the same in principle.** Both verify the pipeline's behavior on data the pipeline itself did not produce. This is the only way to trust the pipeline.

**Outputs.** `N7_verdict`, per-channel recovery diagnostics, the synthetic-truth-vs-recovery table.

**Decision points.** PASS → critical SEALED contribution. FAIL → RETRACT contribution (because if the chain cannot recover synthetic truth, every real-data result is suspect).

**Failure modes.**
- **Recovery bias outside tolerance.** Indicates an inductive bias in the chain that the synthetic data exposes. Framework reports the bias direction and magnitude.
- **Saturation pattern mismatch.** Indicates Stage 6 cellular age inversion is mis-handling out-of-range patients. Framework flags the mismatch for Stage 6 investigation.
- **Card concordance below 80%.** Indicates Stage 8 matching rules are over- or under-firing on synthetic patterns. Framework reports per-card concordance.

**Canonical cross-references.** Recipe §11.7. Roadmap §10.2.1 Phase A2 + §12.1 known limitations.

**CPG Plate references.** **Plate 3 (Grandaddy Plate)** — the CMB realization on its right panel was produced via `healpy.synfast()` from Planck's ΛCDM C_ℓ spectrum. That is the CMB-side analog of CPG's synthetic patient generator: generate synthetic data from a known theoretical model and use it to validate the analysis chain.

**Chain-link assignment.** L9.

---

## §88. L9.8 — N8: Look-elsewhere-effect correction

**What this null tests.** When a VAL searches across many possible features and reports the top one (or top few), the headline p-value must be corrected for the search space. A "p<0.001 in chr6 MHC" finding loses significance if the search included all 22 autosomes — the chance that ONE of 22 chromosomes shows p<0.001 by chance is ~22 × 0.001 = ~2% (Bonferroni), not 0.1%.

**Inputs.** The headline statistic. The declared search space size N_search (declared in PREREG — typically number of chromosomes scanned, number of CpGs scanned, number of cell types scanned, etc.).

**Atlas reference.** None.

**Files invoked.** `cpg_null_runner.py::run_N8_look_elsewhere()`.

**The math.** Two corrections supported (per-VAL declared):

1. **Bonferroni** (conservative): `p_corrected = min(1.0, p_raw × N_search)`. PASS if `p_corrected < α` (typically α=0.05).
2. **Family-wise error rate via permutation** (less conservative, more accurate): use the N1 permutation distribution (§81) but record the *maximum* statistic across the search space per permutation. The corrected p-value is the fraction of permutations where the maximum-across-search-space exceeds the observed maximum.

Default for CPG-VALs: **option 2 (permutation-based FWE)** because Bonferroni is over-conservative when search-space features are correlated (CpGs within a chromosome are strongly correlated; per-cell-type A-scores are correlated). The permutation FWE respects the empirical correlation structure.

**Why this matters.** This is the null that **VAL-006 (chr6 MHC enrichment) failed in Phase A**. The original VAL-006 reported `p=0.009 for chr6 enrichment` without correcting for the 22 chromosomes scanned. Phase A's N8 correction gave `p_FWE ≈ 0.18` — no longer significant at the 0.05 threshold. The VAL was **RESTATEd** rather than RETRACTED because the underlying chr6 MHC signal is consistent with the cross-cohort VAL-005 PC2 T-cell suppression finding (the two findings reinforce each other across the L4 immunological direction), but the per-chromosome enrichment as a standalone claim does not survive look-elsewhere correction. The signal is real; the framing was too strong.

**CMB equivalent.** **Look-Elsewhere Effect (LEE) correction** in cosmological parameter searches and CMB anomaly searches. Famous examples: the WMAP "Cold Spot" significance was inflated by an order of magnitude before LEE correction; once corrected, it was 1.5-2σ rather than 4σ. The Planck "Axis of Evil" alignment significance underwent similar correction. CPG's N8 is the methylome implementation of the same statistical discipline.

**How the methylome differs in implementation.** Search space is per-VAL declared (chromosomes, CpGs, cell types). Same correction math.

**How it's the same in principle.** Both protect against the false-discovery rate inflation from multi-feature search.

**Outputs.** `N8_verdict`, `N8_raw_pvalue`, `N8_corrected_pvalue`, `N8_correction_method`, `N8_search_space_size`.

**Decision points.** PASS → SEALED contribution. FAIL → RESTATE contribution (the underlying signal may be real but the per-feature framing doesn't survive multiple comparisons).

**Failure modes.**
- **Search space size undeclared.** Caught at PREREG time; cannot run N8 without declared N_search.
- **Permutation FWE budget insufficient.** Framework reports the FWE estimate with explicit Monte Carlo uncertainty.

**Canonical cross-references.** Recipe §11.8. Roadmap §10.2.1 Phase A1. §12 of Roadmap (VAL-006 RESTATE on N8).

**CPG Plate references.** **Plate 2 (Breast Pre-Diagnostic Anisotropy)** — the chr6 zoom panel shows the MHC region enrichment that VAL-006 originally claimed; the audit-trail truth is that the per-chromosome enrichment does not survive N8 correction, but the underlying MHC concentration of immune-related signal IS real and IS visible on the plate. The plate visualizes the signal honestly; the framing protects against overclaim.

**Chain-link assignment.** L9.

---

## §89. L9.9 — Synthetic patient generator (`synthetic_patient_generator.py`)

**What this step does.** The companion module to the null framework. Produces synthetic β matrices, manifests, and IDAT-equivalent data for N6 (injection-recovery) and N7 (end-to-end simulation). The synthetic generator IS the chain-of-custody's claim to having an FFP10/NPIPE-equivalent — without it, the chain cannot be tested at the end-to-end level.

**Inputs.** Cohort design specification: `n_case, n_HC, age_distribution, sex_distribution, disease_signal_specification (target_class, effect_size, direction), foreground_levels (age_drift_strength, sex_effect_strength, batch_effect_strength), plate_position_assignment, batch_assignment`.

**Atlas reference.** **IAMAtlas REBUILD posterior consulted heavily.** The synthetic generator draws per-CpG per-class β values from the atlas posterior (mean and SD), constructs synthetic-patient β matrices as mixtures of per-class draws weighted by declared per-class fractions, then injects the declared signal modifications.

**Files invoked.** `Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py`.

**The math.**

Per synthetic patient `p`:
1. Draw per-class mixing fractions `f_p` from the declared cohort design (case patients get the disease-modified mixing pattern; HC get the baseline mixing pattern).
2. For each CpG `i`, sample the per-class β from the IAMAtlas posterior: `β_class[i] ~ Normal(atlas_mean[i, class], atlas_sd[i, class])`.
3. Compose the patient's observed β: `β_observed[i, p] = Σ_class f_p[class] × β_class[i] + ε_noise + ε_age + ε_sex + ε_batch + ε_plate`.
4. Inject signal: for the declared target class's marker CpGs, shift `β_observed` by the declared effect size in the declared direction.
5. Write a synthetic IDAT-equivalent manifest entry.

**Cohort design knobs:**
- `n_case`, `n_HC`: cohort size.
- `age_distribution`: Normal with mean and SD (typically matching a real cohort's distribution).
- `sex_distribution`: P(female | case), P(female | HC) — typically equal unless modeling sex-biased disease.
- `disease_signal_specification`: per-class target, per-cell-type sub-target, effect-size magnitude, direction.
- `foreground_levels`: how strong are age/sex/batch/plate foregrounds in this synthetic cohort? Typically calibrated to match real-cohort foreground strengths.

**CMB equivalent.** **FFP10 generator** in Planck. Given a cosmological model and a survey design, FFP10 produces synthetic timestreams for every Planck bolometer for every scan ring of the mission. These synthetic timestreams pass through the full Planck pipeline as if they were real data, allowing every analysis to be tested against known truth. **The CPG synthetic patient generator is the methylome implementation of FFP10's discipline** — same operational role, same chain-of-custody guarantee.

**How the methylome differs in implementation.** Generation at the per-patient β-matrix level rather than per-bolometer per-scan-ring timestream level. The synthetic patient generator skips L1 (no synthetic bolometer noise) and enters at L2-L3 (synthetic β matrices). Future Phase C HEALPix-compatibility extends generation to L1-equivalent (synthetic IDAT-equivalent intensities).

**How it's the same in principle.** Both produce synthetic data from declared truth specifications, allowing the pipeline to be tested on data the pipeline did not produce. This is the bedrock of chain-of-custody validation.

**Outputs.**
- `synthetic_cohort_manifest_<spec_id>.csv` — manifest entries indistinguishable from real Illumina cohort manifests.
- `synthetic_beta_matrix_<spec_id>.csv` — β matrix consumable by Stage 0+ of the chain.
- `synthetic_truth_<spec_id>.json` — the declared truth for downstream comparison.

**Decision points.** The synthetic generator does not make decisions; it produces data. The downstream chain consumes synthetic data identically to real data.

**Failure modes.**
- **Atlas posterior insufficient.** If the declared signal injection requires CpGs where the atlas posterior is poorly determined (e.g., the stromal galactic mask), the generator flags this and produces best-effort synthetic data with explicit warnings.
- **Inconsistent cohort design.** A specification with `n_case=0` is internally inconsistent; the generator refuses to produce.

**Canonical cross-references.** Recipe §12 (synthetic generation). Roadmap §10.2.1 Phase A2. §13.4 of Roadmap (HEALPix as Phase A2.2 implicit).

**CPG Plate references.** **Plate 3 (Grandaddy Plate)** — the right panel CMB realization is the visual analog of what the synthetic patient generator produces for the methylome. Both are synthetic data from a known theoretical/calibrated model, used to validate analysis chains.

**Chain-link assignment.** L9.

---

## §90. L9.10 — VAL sealing protocol (PREREG → OUTCOME → SEALED / RESTATE / RETRACT)

**What this step does.** The protocol that takes a candidate finding through the L9 framework to a sealed (or restated, or retracted) verdict. **Every CPG-VAL must follow this protocol.** No VAL ships without it.

**The protocol has four stages:**

**Stage S1 — PREREG (Pre-Registration).** Before any beta value is observed (CCL-040/041 absolute), the VAL's analyst writes a PREREG document declaring:
- The hypothesis (e.g., "Immune class A-score in cases >10y pre-dx of breast cancer is elevated vs matched HC at d > +1.0").
- The cohort specification (which cohort, n_case, n_HC, inclusion criteria, exclusion criteria).
- The statistical test (e.g., "Two-sided Welch's t-test on immune A-score, with permutation null").
- The significance threshold (e.g., "p<0.05 with N1 + N7 + N8 PASS required for SEALED").
- The declared null set (which of N1-N8 applies).
- The search space size (for N8).
- Any additional VAL-specific constraints.

The PREREG is SHA-256 hashed and committed to the repo at `Biological_Physics/validation_runs/CPG-VAL-XXX/prereg.json` before any case/HC labels are looked at.

**Stage S2 — DATA RUN.** The VAL runs Stages 0-10 of the chain on the declared cohort. Outputs `per_sample.csv` (per-patient reconstructions) and `headline_results.json` (the declared statistic computed on the actual cohort).

**Stage S3 — L9 NULL SUITE.** The L9 framework (§80) is invoked with the PREREG + headline_results + per_sample.csv. The framework runs the declared null subset and produces `null_results.json`.

**Stage S4 — SEALING VERDICT.** The framework aggregates the per-null verdicts into one of three sealing verdicts:

- **SEALED.** All declared nulls PASS. The VAL's headline becomes a sealed CPG framework claim. The OUTCOME document records the sealing date, the framework version, the atlas SHA-256 fingerprint, the null suite results, and the headline finding. The VAL enters the production canonical inventory and is referenced by downstream cards.

- **RESTATE.** One declared null FAILs in a way that limits the result's scope without invalidating its underlying signal. The OUTCOME document records the failed null, the restated scope (e.g., "VAL-006's chr6 MHC enrichment claim is restated as 'concentration of immune-related signal in HLA-bearing region', no longer as a per-chromosome enrichment with multiple-testing correction"). The VAL enters the canonical inventory with the limitation declared honestly.

- **RETRACT.** Two or more declared nulls FAIL, OR the failed nulls indicate the headline is uninterpretable. The OUTCOME document records the retraction reason. The VAL does NOT enter the canonical inventory. Future work may revisit the VAL with corrected upstream methodology, but the original claim is withdrawn.

**Inputs.** PREREG + per_sample.csv + headline_results.json + null_results.json.

**Atlas reference.** None at this step (audit only).

**Files invoked.**
- The L9 framework (§80) for null computation.
- Repo commit pipeline for OUTCOME documents.

**The math.** Aggregation logic only (declared in §80).

**CMB equivalent.** **The Planck collaboration's release process.** Before a Planck result becomes a community-citable finding, it passes through: (1) internal pre-registration of analysis choices and null tests; (2) full pipeline run; (3) null test battery; (4) collaboration-wide review; (5) release with documented sealing status and known limitations. **CPG's VAL sealing protocol is the methylome implementation of this discipline at the per-finding granularity.**

**How the methylome differs in implementation.** Per-VAL granularity rather than per-release. Each CPG-VAL gets its own PREREG/OUTCOME pair; Planck releases bundle hundreds of derived quantities into one collaboration-wide product. The principle is identical; the operational unit differs.

**How it's the same in principle.** Both prevent post-hoc statistical fishing. Both produce a permanent record of every choice made. Both produce three possible verdicts (sealed / preserved-with-limit / withdrawn) and treat the limited and withdrawn cases honestly rather than hiding them.

**Outputs.** The OUTCOME document at `Biological_Physics/validation_runs/CPG-VAL-XXX/outcome.md`. The sealed VAL artifact bundle.

**Decision points.** The sealing verdict is the decision point. Downstream cards consume sealed VALs only; restated VALs are referenced with explicit caveat; retracted VALs are not referenced.

**Failure modes.**
- **PREREG modified after data observation.** Caught by SHA-256 + commit-history audit. If detected, the VAL is automatically RETRACTed.
- **Headline result data does not match PREREG specification.** E.g., PREREG declared n_case=146 but per_sample.csv shows n_case=139. Framework returns FRAMEWORK_INCONSISTENCY; VAL does not seal.

**Canonical cross-references.** Recipe §11.10. Roadmap §10.2.1 Phase A3 (running all 7 Family A VALs through the framework). §12 of Roadmap (Phase A results: 5 sealed, 2 restated).

**CPG Plate references.** Every plate (1-4) is built from sealed-VAL data. The plates are visual products of the sealing protocol's outputs.

**Chain-link assignment.** L9.

---

## §91. L9.11 — Null-suite invocation order

**What this step does.** Documents the canonical order in which L9 nulls are invoked when the framework runs a VAL through the full suite. The order matters because some nulls produce intermediate artifacts that others consume.

**Invocation order:**

1. **N1 (HC label permutation)** — runs first because its permutation distribution is the foundation for N8's FWE correction.
2. **N2 (Age-stratified permutation)** — runs second; independent of others.
3. **N3 (Sex-stratified permutation)** — runs third; independent of others.
4. **N4 (Cohort-split replication)** — runs fourth; uses random 50/50 split independent of N1-N3.
5. **N5 (Plate/array-position null)** — runs fifth; requires position metadata; independent of others.
6. **N6 (Injection-recovery)** — runs sixth; invokes `synthetic_patient_generator.py`; CPU-intensive.
7. **N7 (End-to-end synthetic-patient simulation)** — runs seventh; the most CPU-intensive null; full chain run on synthetic cohort.
8. **N8 (Look-elsewhere correction)** — runs LAST because it consumes the N1 permutation distribution for FWE estimation.

The framework invokes the declared subset of these (a VAL declares which apply in its PREREG). Independent nulls (N1, N2, N3, N4, N5) can run in parallel; N6 and N7 are dependent on synthetic generation; N8 is dependent on N1.

**Inputs.** PREREG declared null subset.

**Atlas reference.** Indirect (N6, N7 consult atlas via synthetic generator).

**Files invoked.** `cpg_null_runner.py::run_full_suite()`.

**The math.** Dispatch logic.

**CMB equivalent.** The **canonical null-test ordering** in Planck NPIPE. Planck's null suite has a well-documented invocation order driven by dependency graph (jackknife resampling pre-computed before null statistics; FFP10 simulations are independent and parallelizable; LEE correction uses the per-statistic null distributions). CPG's invocation order is the methylome implementation.

**How the methylome differs in implementation.** Smaller dependency graph (8 nulls vs Planck's tens of null tests). Same principle.

**How it's the same in principle.** Both establish a canonical order so that the same VAL invoked twice produces identical null results.

**Outputs.** A per-VAL `null_invocation_log.json` recording per-null start/end times, framework version, dispatch decisions.

**Decision points.** None — pure orchestration.

**Failure modes.**
- **Per-null timeout.** Configurable; default 1 hour per null. A timeout returns FRAMEWORK_TIMEOUT for that null; downstream verdict propagates the timeout as INSUFFICIENT (not PASS, not FAIL).
- **Resource exhaustion** (memory, disk for synthetic cohorts). Framework reports the resource shortfall and refuses to seal until resources are available.

**Canonical cross-references.** Recipe §11.11. Roadmap §10.2.1 Phase A.

**CPG Plate references.** None.

**Chain-link assignment.** L9.

---

**End of Part III (L9 audit machinery, §80-§91).** The chain-integrity scaffolding is complete: the framework (§80), each of the eight nulls (§81-§88), the synthetic patient generator (§89), the sealing protocol (§90), and the invocation order (§91).

Part IV follows with failure modes and decision trees (§92-§96). Part V with reference tables (§97-§102).

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*


> **v1.1 note (2026-06-02).** v1 contained fabricated `cpg_engine/...` module paths
> that did not correspond to real files in the IAM-Validation repo. This v1.1 pass
> stripped them. Where a step's logic currently lives inside `GAPE_WEB_v13.py` (the
> production engine — see `SYSTEM_INVENTORY.md`), the SOP now says so honestly. Where
> a step's runtime artifact or output file location was invented, the SOP now reads
> "TBD per orchestrator design" — that orchestrator (working name `web.commercial.py`)
> is a separate conversation Heath and Walther will have. All real paths in this v1.1
> are documented in `SYSTEM_INVENTORY.md`; any path NOT in that inventory should be
> treated as not-yet-existing until verified against the repo.

---


**Continues from Part III (§80-§91). This document contains §92-§96 (failure modes & decision trees) and §97-§102 (reference tables, glossary, H_min canonical, plate cross-reference, change log).**

---

# Part IV — Failure modes & decision trees

Part II established every step. Part III established the audit machinery. Part IV does what neither could in isolation: it tells the operator what to DO when something goes wrong. The discipline: **every failure mode has a declared response.** No silent degradation. No "I'll figure it out at the report."

The structure of Part IV is failure-mode-first rather than step-first because real operations don't fail nicely along step boundaries. A sample with a sex mismatch (§18) might also have low call rate (§17) and a borderline cross-method gate (§33) — three flags from three different chain links, requiring a single coherent response. Part IV organizes the response logic around the failure pattern, not the step.

---

## §92. Failure mode catalog by stage

This is the master table of every failure mode declared across §11-§91, organized by stage, with the canonical response for each.

### Stage 0 (Intake) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Missing IDAT pair | §11 | HARD | Reject; require new sample submission |
| Zero-byte or truncated IDAT | §11 | HARD | Reject; transmission failure |
| Missing manifest entry | §12 | HARD | Reject; require complete metadata |
| Schema-mismatched manifest | §12 | HARD | Reject; require manifest correction |
| Cleartext PII in manifest | §12 | HARD | Reject; security violation |
| IDAT integrity hash mismatch | §13 | HARD | Quarantine; investigate file transmission/storage |
| Bisulfite Conversion I/II control fail | §14 | HARD | Reject sample; wet-lab BS conversion failed |
| Specificity I/II control fail | §14 | HARD | Reject sample; wet-lab specificity failed |
| Non-polymorphic control deviation | §14 | SOFT | Flag for review; do not auto-reject |
| Negative control elevated | §14 | SOFT | Flag for contamination review |
| Detection p-value >5% probes failed | §15 | HARD | Reject sample; DNA quality issue |
| Detection p-value 1-5% probes failed | §15 | SOFT | Proceed with `detection_qc=marginal` flag |
| Bead count >5% probes failed | §16 | HARD | Combine with §15; if both fail, reject |
| Sample-level call rate <95% | §17 | HARD | Reject |
| Sample-level call rate 95-98% | §17 | SOFT | Proceed with `call_rate=marginal` flag |
| Sex check MISMATCH | §18 | HARD | Quarantine; chain-of-custody investigation |
| Sex check AMBIGUOUS | §18 | SOFT | Flag for review; possibly atypical karyotype |

### Stage 1 (Calibration & β) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Dye-bias correction factor extreme | §20 | SOFT | Flag for review |
| Normalization method failure (funnorm) | §21 | SOFT | Fall back to noob |
| Normalization failure (both funnorm + noob) | §21 | HARD | Quarantine; sample is uncalibratable |
| ComBat with confounded design | §22 | HARD | Halt; reject cohort design |
| Insufficient samples per batch for ComBat | §22 | SOFT | Skip ComBat with warning; downstream confidence penalty |
| Bisulfite efficiency < 0.95 | §23 | HARD | Retroactive quarantine of sample |
| Bisulfite efficiency 0.95-0.98 | §23 | SOFT | Flag as marginal |
| β values outside [0,1] | §24 | HARD | Quarantine; arithmetic error upstream |
| β distribution unimodal | §25 | HARD | Quarantine; sample distribution anomalous |
| Bimodality coefficient 0.45-0.55 | §25 | SOFT | Flag as marginal |

### Stage 2 (Deconvolution) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Atlas SHA mismatch | §28 | HARD | Halt; atlas reference corrupted |
| Atlas file missing | §28 | HARD | Halt; atlas must be present |
| Marker pool empty for a class | §29 | SOFT | Flag class as INSUFFICIENT_COVERAGE |
| Marker SHA mismatch | §29 | HARD | Halt; marker artifact corrupted |
| NNLS does not converge | §30 | HARD | Quarantine patient |
| Residual MAE > 0.10 | §30 | SOFT | Flag as POOR_DECONVOLUTION; require §33 gate verification |
| Per-class confidence <0.5 broadly | §31 | SOFT | Flag for substrate review |
| NILC does not converge | §32 | SOFT | Cross-method gate becomes single-method; reduced confidence |
| Cross-method gate FLAG | §33 | SOFT | Proceed with cross-method-uncertainty annotation |
| Cross-method gate FAIL | §33 | HARD | Quarantine; manual review required |
| Cohort-wide FLAG rate >5% | §33 | HARD | Halt cohort; investigate systematic |

### Stage 3 (Foreground subtraction) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Patient age outside age-axis calibration range | §35 | SOFT | Apply with `age_extrapolation` flag |
| Age layer SHA mismatch | §35 | HARD | Halt; age layer corrupted |
| Sex/batch/ancestry/smoking foreground unimplemented | §36-§39 | DOCUMENTED GAP | Pass-through with module-status flag |

### Stage 4 (A-score) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Class has <20 markers in QC-passed pool | §41 | SOFT | Flag class A-score with INSUFFICIENT_MARKERS |
| Cell type has <20 markers | §44 | SOFT | Flag cell-type A-score |
| Cell type has <60% coverage | §44 | SOFT | Status MARGINAL_COVERAGE |
| Cell type has <20% coverage | §44 | HARD | Status NO_MARKER_OVERLAP; A-score not returned |
| H_min file SHA mismatch | §43 | HARD | Halt; canonical reference corrupted |
| A-score < 0 | §43 | HARD | Halt; calculation bug |
| Disease panel CpGs <60% available | §45 | SOFT | Flag panel A-score with INSUFFICIENT_MARKERS |

### Stage 5 (Mahalanobis) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| >15 cell types imputed | §47 | HARD | Flag patient INSUFFICIENT_DATA; do not return Mahalanobis |
| 6-15 imputed | §47 | SOFT | Proceed with PARTIAL_DATA flag |
| HC reference SHA mismatch | §48 | HARD | Halt |
| MAHALANOBIS_NUMERIC_FAIL | §49 | SOFT | Flag; report omits the headline |
| Cohort-wide imputation >10% patients with >5 imputations | §47 | HARD | Flag cohort substrate-atlas mismatch |

### Stage 6 (Cellular age) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Age reference SHA mismatch | §53 | HARD | Halt |
| Per-class A out of baseline range (saturation) | §54 | DATA SIGNAL | Report with SAT_HIGH or SAT_LOW status; not a bug |
| Patient chronological age outside reference (4-95) | §57 | SOFT | Flag with AGE_OUT_OF_RANGE; report boundary percentile |
| Baseline matrix non-monotonic | §54 | HARD | Halt; atlas build issue |

### Stage 7 (Tier breakpoints) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Substrate missing for cfDNA branch | §61 | SOFT | Skip cfDNA step |
| Global FLOOR_BREACH (≥4 classes) | §62 | HARD | Hold report; manual review at Stage 9 |
| Multi-class breach (2-3 classes) | §62 | SOFT | Flag for review |

### Stage 8 (Card matching) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Disease matrix SHA mismatch | §65 | HARD | Halt |
| Card rule syntax error | §67 | HARD | Caught at deployment, not runtime |
| Phase verdict ambiguity | §67 | SOFT | Report highest-confidence phase with ambiguous flag |
| Residual map CpG coverage <80% | §66 | SOFT | Mark residual channel INSUFFICIENT; fall back to per-class matching |
| Patient metadata missing for covariate | §68 | SOFT | Flag UNADJUSTED; report explicit "covariate-uninformative" disclaimer |
| Covariate out of declared range | §68 | SOFT | Clamp to range; flag CLIPPED_COVARIATE |

### Stage 9 (Report assembly) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Mapping table version mismatch | §70 | HARD | Halt |
| Engine tier outside mapping range | §70 | HARD | Report fails legal-boundary at §76; does not ship |
| Literature anchor 404 / missing | §71 | HARD AT DEPLOY | Caught at deployment |
| Literature anchor effect-summary conflict | §71 | SOFT | Use conservative wrapper |
| No anchor for fired card | §71 | SOFT | Card reports with "limited published context" disclaimer |
| Cancer prior table SHA mismatch | §72 | HARD | Halt |
| Patient sex/age missing for prior | §72 | SOFT | Fall back to overall-population prior |
| Family history multiplier missing for disease | §73 | SOFT | Default 1.0; flag |
| Card sex-condition ambiguous | §74 | HARD AT DEPLOY | Caught at deployment |
| Renderer slot unfilled | §75 | HARD | Report INCOMPLETE; does not advance |
| Legal-boundary check finds CANNOT_SAY violation | §76 | HARD | HALT_FOR_REVIEW; report does not ship |

### Stage 10 (Delivery) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Rendering pipeline failure | §77 | SOFT | Retain cleared Markdown; manual repackaging |
| Channel authentication failure | §78 | SOFT | Bounded retry; manual handoff after exhaustion |
| Channel throughput limit | §78 | SOFT | Queue with rate-limit backoff |
| Audit capture failure | §79 | HARD | Hold report; cannot ship without audit |

### L9 (Audit machinery) — failure modes

| Failure mode | Where detected | Severity | Canonical response |
|---|---|---|---|
| Framework dispatch failure (null crashes) | §80 | HARD | Return FAIL with framework_error flag |
| Declared null not runnable (missing metadata) | §80-§88 | SOFT | INSUFFICIENT; framework aggregates to RESTATE |
| Insufficient permutation budget | §81 | SOFT | Report empirical percentile with explicit budget note |
| Cohort too small to split for N4 | §84 | SOFT | INSUFFICIENT_SAMPLE; framework reports limitation |
| N6 recovery sign opposite injection | §86 | HARD | Halt; full audit required |
| N7 recovery bias outside tolerance | §87 | HARD | RESTATE or RETRACT |
| PREREG modified after data observation | §90 | HARD | Auto-RETRACT |

**Severity codes:**
- **HARD**: chain halts at this step; sample/cohort/VAL cannot advance without intervention.
- **SOFT**: chain advances with explicit flag propagated downstream; reduces confidence but does not block.
- **DATA SIGNAL**: not a failure; the "anomalous" state is itself a measurement (e.g., saturation in Stage 6).
- **DOCUMENTED GAP**: the step is unimplemented in current production; passed through with a status flag pending Phase B/C/D/E delivery.

---

## §93. Cross-stage decision tree

When multiple flags fire on a single patient, the canonical response is the **most-conservative** of the declared per-flag responses. Tabular decision tree:

| Combination | Combined response |
|---|---|
| Any HARD anywhere → | Halt; do not advance |
| ≥1 SOFT in Stage 0-3 + ≥1 SOFT in Stage 5-7 → | Proceed; bundle all flags into report `data_quality` block |
| ≥2 SOFT flags in same stage → | Flag stage as `multi_flag`; escalate confidence penalty |
| Sex mismatch + low call rate + cross-method FLAG → | This pattern indicates sample swap + DNA quality + atlas mismatch all together. Treat as HARD even though individually one is SOFT. Quarantine. |
| L9 RESTATE + Stage 8 card fires → | Card reports with restated VAL caveat in literature anchor block (§71) |
| L9 RETRACT + Stage 8 card fires on retracted VAL → | Card does NOT fire (retracted VAL is not referenced anywhere in production) |
| Cellular age SAT_HIGH on ≥6 of 8 classes → | DATA SIGNAL, not failure. Report all 8 ages with their saturation flags. The pattern itself is the measurement. |

The decision tree is **not** "use the worst flag." It is "compose the per-flag responses without losing any information." Every flag survives in the audit trail (§79) even when it does not block delivery.

**Inputs.** All flags from all stages.
**Atlas reference.** None.
**Files invoked.** `<cross-stage flag composition logic — currently inline in `GAPE_WEB_v13.py`>`.
**The math.** Boolean composition declared in this section.
**CMB equivalent.** Planck's per-release quality assessment composes flags from per-bolometer, per-scan-ring, per-frequency, per-component-separation-method, and per-likelihood-channel diagnostics. The composition rule is non-trivial: a single "warning" in one diagnostic combined with a "warning" in another may exceed the significance of either alone. Same here.
**How the methylome differs.** Per-patient combination rather than per-release.
**How it's the same.** Multi-source flag composition without information loss.
**Outputs.** A per-patient `combined_response ∈ {PROCEED, FLAGGED, QUARANTINE, REJECT}` field plus the audit trail of all contributing flags.
**Decision points.** The combined response itself is the decision.
**Failure modes.** Cross-stage logic bugs (a combination not anticipated by the decision tree). Detected by manual sampling of edge cases; updates to the decision tree itself.
**Canonical cross-references.** §11-§91 of this SOP.
**CPG Plate references.** None.
**Chain-link assignment.** Above all (cross-cutting).

---

## §94. Cross-method disagreement protocol

Specifically for the Walther vs NILC cross-method gate at §33 — the most common SOFT-to-HARD escalation point in production. Detailed protocol:

**Step 94.1 — Classify the disagreement.**

| Walther verdict | NILC verdict | Classification |
|---|---|---|
| OK | OK | Cross-method PASS |
| OK | FAIL (low conf, status NOT OK) | SINGLE_METHOD; proceed with reduced confidence |
| FAIL (low conf) | OK | Same: SINGLE_METHOD |
| Both OK with substrate L1 disagreement >0.25 median | Both OK | Substrate-disagreement FLAG; proceed with annotation |
| Inference direction disagreement on disease-relevant class | (either) | INFERENCE FLAG; pause patient |
| Both FAIL | Both FAIL | Quarantine; sample uninterpretable in 8-class basis |

**Step 94.2 — Apply the response.**

- **Cross-method PASS** → §34 Stage 2 output proceeds normally.
- **SINGLE_METHOD** → §34 proceeds; manifest flag `cross_method=single`; downstream report carries the flag.
- **Substrate-disagreement FLAG** → §34 proceeds; cross-method-uncertainty annotation on report; the Walther fractions are still the production answer but the report explicitly notes "substrate-level cross-method disagreement at <X median L1>; biological inference layer was concordant."
- **INFERENCE FLAG** → STOP at §33. Manual review required before Stage 3 advancement. The review either resolves (sample/cohort issue identified and corrected) or escalates to substrate-mismatch quarantine.
- **Quarantine** → patient exits the chain at §33. Sample's substrate is not interpretable in the current IAMAtlas's 8-class basis. Note for future atlas expansion (Roadmap §10.4 multi-substrate work).

**Step 94.3 — Document.**

Every cross-method disagreement, regardless of severity, gets a row in `Biological_Physics/chain_of_custody/L4_component_separation/nilc_walther_crosscheck_v2.json`. The log is the running audit trail of inter-method behavior across all production patients. It is the empirical foundation for any future Phase B5 work on disagreement-resolution (e.g., adding SMICA-style and SEVEM-style methylome variants if the cross-method discipline ever needs more than two voices).

**CMB equivalent.** Planck's **Commander vs NILC vs SMICA vs SEVEM comparison logs** drive Phase-Z calibration adjustments. When the four methods disagree at the per-pixel level by more than expected, Planck doesn't pick a "right" method — Planck publishes all four and notes the disagreement as a propagated systematic. CPG follows the same discipline at the patient-report level.

**How the methylome differs.** Two methods (currently) vs four. The disagreement-handling protocol is the same.

**How it's the same.** Cross-method discipline is the single strongest claim a measurement can make. Documentation of disagreement is the discipline; resolution-attempts come later if at all.

**Outputs.** The cross-method-log artifact + per-patient disagreement classification.

**Decision points.** Above five-row table.

**Failure modes.**
- **Method registry mismatch** (e.g., NILC has been upgraded to v3 but Walther still v2.x). Caught at engine deployment; both methods must be at registered versions.
- **Cross-method log corruption.** SHA-hashed at every append.

**Canonical cross-references.** §32 (NILC), §33 (gate), Phase_B2_1_FINDING.md.

**CPG Plate references.** **Plate 4 Panel B** (chr16/chr17 cold patches visible to BOTH deconvolvers) is the visualization of the cross-method-consensus discipline: the systematic is real because both methods see it.

**Chain-link assignment.** L4 + L9.

---

## §95. Out-of-calibration handling (saturation, low coverage, missing metadata)

**Out-of-calibration means: the patient sits in a region of measurement space where the IAMAtlas reference does not provide reliable interpretation.** Three canonical patterns:

**95.1 Saturation (Stage 6 cellular age).** Patient's per-class A-score is above the highest decade's baseline A_mean or below the lowest. Status SAT_HIGH or SAT_LOW. **This is not a failure — it is a measurement.** Report all 8 per-class cellular ages with their saturation flags. Customer report at Stage 9 communicates: "Your terminal-class cellular reading is outside the range of the IAMAtlas calibration cohort. This is documented; the calibration cohort had limited representation in your reading's region. Consider discussing with your physician."

**95.2 Low coverage (Stages 2, 4, 5).** Patient's per-class or per-cell-type marker pool has fewer than 60% (cell type) or fewer than 20 markers (class) in the QC-passed pool. Status INSUFFICIENT_MARKERS or NO_MARKER_OVERLAP. **Selective failure.** The specific class/cell-type's measurement is not returned; other classes/cell-types proceed. Report at Stage 9 notes the missing measurements with explicit "could not measure [X] reliably for this sample" language.

**95.3 Missing metadata (Stages 0, 1, 3, 9).** Required metadata (age, sex, plate position, ancestry, smoking status) is missing or unparseable. Per-step canonical response: declare the dependency (which steps required this metadata), set the affected steps to PASS_THROUGH or fallback (e.g., overall-population prior in §72), flag at every downstream consumer. Report at Stage 9 communicates the limitation honestly.

**Common across all three:** the framework **never** silently imputes. Every out-of-calibration handling is recorded in the audit trail; every customer report explicitly states the limitation.

**Inputs.** Flag accumulator from §11-§79.
**Atlas reference.** Indirect (out-of-calibration is defined relative to the atlas's calibrated range).
**Files invoked.** Each affected stage's failure-mode handler.
**The math.** Per-stage declared in §11-§91.
**CMB equivalent.** Planck's **galactic mask** (out-of-calibration sky region — the foregrounds are too strong to extract cosmological signal). Planck does not silently impute the masked region — Planck reports cosmological parameters as derived from the unmasked sky with the masked region's size explicitly documented. CPG's saturation, low-coverage, missing-metadata handling is the methylome implementation of the same discipline.
**How the methylome differs.** Per-patient out-of-calibration (different patients have different out-of-range regions). Galactic mask is fixed per-sky.
**How it's the same.** The mask is declared, not hidden.
**Outputs.** Per-patient out-of-calibration flag list propagated through to §79 audit.
**Decision points.** Above three patterns each have their own decision logic, declared in this section.
**Failure modes.** Mis-classification of out-of-calibration as failure (resulting in unnecessary quarantine) or as success (resulting in spurious confident reading). The audit trail allows retrospective detection of either error.
**Canonical cross-references.** §47 (Mahalanobis imputation), §54 (cellular age saturation), §63 (engine-to-customer collapse).
**CPG Plate references.** **Plate 4 Panel E** (MCMC Coverage Map) — the stromal galactic mask visualization is the per-CpG out-of-calibration map. The dark region IS the framework's declared limitation, made empirically visible.
**Chain-link assignment.** Cross-cutting (every stage that touches the atlas).

---

## §96. Chain re-run protocol

Sometimes the chain must be re-run on the same sample or cohort — typically because an upstream version updated (new IAMAtlas, new marker pool, new H_min after recalibration, new card deployed) and the existing results need to be regenerated under the updated configuration.

**Re-run conditions:**

| Trigger | Re-run scope |
|---|---|
| IAMAtlas version update | Full chain Stage 2-10 on all affected cohorts |
| Marker pool update (e.g., v0_1 → v0_3) | Stage 2-10 |
| H_min recalibration | Stage 4-10 |
| Disease signature matrix update (v1.4 → v1.5) | Stage 8-10 |
| Card deployment (new card or version) | Stage 8-10 for that card |
| Literature anchor table update | Stage 9-10 |
| Cannot-say list update | Stage 9-10 |
| L9 framework update | Re-run L9 audit on affected sealed VALs |
| Synthetic patient generator update | Re-run N6, N7 nulls on affected VALs |

**Re-run protocol:**

1. **Declare the trigger.** What changed; which artifacts have new SHA-256 hashes; which downstream stages depend on the changed artifact.
2. **Identify the scope.** Which cohorts/patients/VALs are affected. The audit trail (§79) makes this query trivial — any sample whose audit log references the now-superseded artifact SHA is in scope.
3. **Lock the new artifacts.** SHA-256 hash the new artifact; pin in the appropriate canonical (IAMAtlasREBUILD_provenance.json, etc.).
4. **Re-run from the affected stage.** Use the existing per-sample data from upstream stages (no need to re-do Stage 0 intake or Stage 1 calibration if those didn't change).
5. **Re-compute the audit trail.** Each affected patient gets a new audit record with the new SHA references. The old audit record is preserved at `audit_logs/superseded/`.
6. **For sealed VALs:** if the VAL's nulls depended on the changed artifact, the framework re-runs the affected nulls and re-issues a sealing verdict. **A VAL that was SEALED under the old artifact must be re-sealed under the new artifact** — sealing is not transitive across artifact versions.
7. **Customer notification.** When a customer report is regenerated under updated configuration, the customer receives an updated report with explicit "Updated under [new artifact version], [date]" annotation. Old reports are archived but not deleted.

**Inputs.** Trigger + affected scope + new artifact.
**Atlas reference.** New atlas version, if the trigger was atlas-related.
**Files invoked.** `<chain re-run orchestration — TBD when `web.commercial.py` is built>` orchestrates.
**The math.** Same as the original chain.
**CMB equivalent.** Planck **Release re-issue cycles**. When Planck recalibrated a frequency channel between Release 2 and Release 3, every dependent analysis was re-run; the published cosmological parameters changed slightly; the discipline was that the Release 3 cosmological-parameter values were tagged "Release 3 / 2018 / Planck-2018-Calibration-v3" so that any future paper citing them could trace the provenance.
**How the methylome differs.** Per-patient re-runs vs per-release re-runs. Same provenance discipline.
**How it's the same.** Updates do not silently overwrite; every version is traceable.
**Outputs.** Updated per-patient reports + updated audit trails + (when applicable) re-issued VAL sealing verdicts.
**Decision points.** Trigger-condition determines scope; scope determines which stages re-run.
**Failure modes.**
- **Partial re-run** (some affected patients not re-processed). Detected by the audit-trail-vs-current-artifact-SHA query; framework refuses to deliver a report under a stale audit until re-run completes.
- **Re-sealing conflict** (VAL that was SEALED under old artifact RESTATEs under new artifact). Documented honestly in the new OUTCOME; old SEALED status is preserved with the supersession note.

**Canonical cross-references.** Recipe §13 (versioning discipline). Roadmap §10.5 (Phase G/H/I/J post-engine card work, which is built on re-run capability).

**CPG Plate references.** None directly. (Plates are typically re-generated when the underlying VAL data changes; re-run is the operational mechanism behind plate updates.)

**Chain-link assignment.** Cross-cutting (any chain link can trigger; any chain link can be affected).

---

# Part V — Reference

The final part. Tables, glossaries, indices. Designed for at-a-glance lookup during operations rather than narrative reading.

---

## §97. File-to-stage mapping table

Authoritative inventory: `Biological_Physics/atlas_vault/SYSTEM_INVENTORY.md`. This SOP section mirrors that inventory and adds the SOP-section cross-reference column.

§97 is organized in three blocks: **(A) Real artifacts that exist as files in the repo today; (B) Logic currently embedded inside `GAPE_WEB_v13.py` — the production engine — and not yet isolated as standalone modules; (C) Pending — does not exist yet, awaits the `web.commercial.py` orchestrator design discussion.**

If an operator is looking for a path that this SOP names anywhere from §11 to §96, this is the table that says whether it is a real file (A), embedded logic (B), or pending (C).

### §97.A — Real files / modules per `SYSTEM_INVENTORY.md`

| Artifact | Path | Used at SOP § | Role |
|---|---|---|---|
| **IAMAtlas REBUILD** | `IAMAtlasREBUILD.csv` (proprietary; not in repo) | §28 + all atlas-consulting steps | The calibrated instrument |
| Atlas provenance | `IAMAtlasREBUILD_provenance.json` | §28, §43, §99 | H_min source of truth (frozen 2026-04-06) |
| **Walther deconvolver** | `Biological_Physics/atlas_vault/deconvolver/walther_iam_deconvolver.py` | §30, §31 | Production NNLS deconvolution |
| Walther README | `Biological_Physics/atlas_vault/deconvolver/README.md` | §30, §31 | Deconvolver-specific docs |
| **NILC v2 deconvolver** | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_deconvolver.py` | §32 | Cross-method GLS deconvolution |
| NILC v1 fractions | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_fractions_all.csv` | (audit) | Historical v1 output |
| NILC v2 fractions | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_fractions_v2_departure.csv` | (audit) | Current v2 output |
| NILC v1 crosscheck | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_walther_crosscheck.json` | (audit) | Historical v1 gate report |
| NILC v2 crosscheck | `Biological_Physics/chain_of_custody/L4_component_separation/nilc_walther_crosscheck_v2.json` | §33, §94 | Current cross-method gate report |
| Phase B2 finding | `Biological_Physics/chain_of_custody/L4_component_separation/Phase_B2_FINDING.md` | (historical) | Initial NILC+Walther cross-check |
| Phase B2.1 finding | `Biological_Physics/chain_of_custody/L4_component_separation/Phase_B2_1_FINDING.md` | §33 | Current cross-method gate documentation |
| **Age-axis foreground** | `Biological_Physics/atlas_vault/components/age_axis_foreground.py` | §35 | Per-CpG age regression / β subtraction (Phase B3) |
| Age layer matrix | `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv` | §35 | Per-CpG (α, γ, R², n) — 8,199 CpGs, 100% convergence |
| Age layer diagnostics | `Biological_Physics/atlas_vault/components/age_layer_diagnostics.json` | §35 | Per-CpG fit diagnostics |
| **A-score scoring** | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_a_scoring.py` | §41–§45 | `score_per_class()` + `score_per_celltype()` |
| Marker artifact | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json` | §29, §44, §65 | Per-cell-type one-vs-rest top-100, 115 cell types |
| Marker SHA anchor | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.sha256` | §29 | SHA: `46ea5be1db377f2b8773a02418a7f481a191630e0fa833d3294eab1fd19c47bd` |
| **Mahalanobis scoring** | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_mahalanobis_scoring.py` | §47–§51 | `MahalanobisHealthyHull` class — distance + top-10 axis decomp |
| Mahalanobis reference | `Biological_Physics/atlas_vault/walther_clinical_runtime/Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v0_3.json` (current production; v0_1/v0_2/v0_3/v0_4 retained for lineage) | §48 | HC centroid + covariance (n_hc=1,721, Ledoit-Wolf shrinkage=0.001317, percentile-calibrated Route A threshold p95=13.54) |
| Mahalanobis validation | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/mahalanobis_per_patient_breast_predx_validation.csv` | (audit) | Per-patient breast pre-dx distances (n=648) |
| **Cellular age scoring** | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/iam_cellular_age_scoring.py` | §52–§58 | `IAMCellularAge` class — canonical Recipe §6.3 inversion |
| Age reference matrix (JSON) | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.json` | §53 | 80-cell baseline: 8 classes × 10 decades |
| Age reference matrix (CSV) | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.csv` | §53 | Same data, flat CSV |
| Age reference matrix (PY) | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/age_reference_matrix.py` | §53 | Same data, with interpolation helpers |
| Cellular age validation | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/cellular_ages_v4_epic_italy_validation.csv` | (audit) | 1,174 EPIC-Italy v4 output |
| Tier breakpoints | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/tier_breakpoints.json` | §59, §60, §63 | A-score thresholds + customer-label collapse |
| cfDNA weights | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/cfdna_weight.json` | §61 | Healthy-blood per-class expected (Snyder/Moss) |
| Literature anchors | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/literature_anchors.json` | §71 | Per-class published anchors |
| Cancer prior | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/cancer_prior.json` | §72 | US lifetime incidence per class |
| Family history mult. | `Biological_Physics/atlas_vault/pipeline_runtime_matrices/family_history_multiplier.json` | §73 | First-degree-relative RR per class |
| Disease signature matrix v1.5 | embedded inside `GAPE_WEB_v13.py` at present writing | §65 | 77×131 card-level lookup; standalone binary export pending at `pipeline_runtime_matrices/disease_signature_matrix/` |
| Card residual maps | placeholder dir `pipeline_runtime_matrices/card_residual_maps/` | §66 | Currently empty; populated per card as VALs lock thresholds |
| **L9 null runner** | `Biological_Physics/chain_of_custody/L9_null_suite/cpg_null_runner.py` | §80–§88, §91 | Unified 8-null framework (N1–N8) |
| **Synthetic patients** | `Biological_Physics/chain_of_custody/L9_null_suite/synthetic_patient_generator.py` | §86, §87, §89 | FFP10/NPIPE analog — signal-injection harness |
| Per-VAL null outputs | `Biological_Physics/chain_of_custody/L9_null_suite/test_runs/CPG_VAL_00X/` | (audit) | Phase A: 5 sealed, 2 RESTATE (VAL-004 gain/loss, VAL-006 chr6 LEC) |
| **Production engine** | `GAPE_WEB_v13.py` | §11–§79 (where logic is embedded) | The actual runtime — see §97.B |
| System inventory | `Biological_Physics/atlas_vault/SYSTEM_INVENTORY.md` | (this table) | Source of truth for paths |

### §97.B — Logic currently embedded inside `GAPE_WEB_v13.py`

The following operations described in §11–§79 are performed by the production engine `GAPE_WEB_v13.py` directly, not as separate module files. v1 of this SOP fabricated module paths for each (e.g. `cpg_engine/qc/sex_check.py`, `cpg_engine/calibration/dye_bias.py`); those were wrong. The real situation: one engine, one file, no per-stage module isolation yet.

| SOP § | Operation | Notes |
|---|---|---|
| §11–§19 | Stage 0 — IDAT intake, control probes, detection-p, bead count, call rate, sex check, decision gate | Stage 0 QC runs inside Walther per `SYSTEM_INVENTORY.md` §1; the wider intake/manifest layer is engine-internal. |
| §20–§27 | Stage 1 — dye bias, probe-type normalization, ComBat, BS conversion, β computation, β sanity | Wraps `minfi`/`methylprep` upstream; β values come into the deconvolver pre-computed. |
| §28, §29 | Atlas loading + marker-pool extraction | Engine loads `IAMAtlasREBUILD.csv` and `iamatlas_celltype_markers_v0_2.json` at startup. |
| §34 | Stage 2 output consolidation | Engine bookkeeping; no separate packager file. |
| §40 | Stage 3 output consolidation | Same — engine bookkeeping. |
| §46 | Stage 4 output consolidation | Same. The scoring module (`iamatlas_a_scoring.py`) is real and standalone; the consolidation is engine-internal. |
| §65, §67–§68 | Stage 8 card matching — disease-signature lookup, multi-class rule eval, within-card covariate adjustment | All inside the engine; card registry and disease-signature matrix v1.5 are embedded constants. |
| §66 | Residual-overlap (breast-epic) | Logic inside engine; the residual-map CSV will live at `pipeline_runtime_matrices/card_residual_maps/breast-epic/` once VAL-003 locks. |
| §70–§76 | Stage 9 report assembly — language collapse, literature lookup, prior lookup, family history, sex adjustment, renderer, legal-boundary gate | All inside the engine. Lookup JSONs (anchors, prior, multiplier) are real files; the orchestration is engine-internal. |
| §77–§79 | Stage 10 delivery — packaging, routing, audit capture | All inside the engine today; expected to move into the future `web.commercial.py` orchestrator. |
| §93, §96 | Cross-stage flag composition; chain re-run orchestration | Inline within `GAPE_WEB_v13.py` today; cleanly modularized when orchestrator is built. |

### §97.C — Pending: does not exist yet

| Item | What it would be | Blocked by |
|---|---|---|
| `web.commercial.py` orchestrator | Top-level driver that calls the modules in §97.A in sequence, replacing the engine-internal orchestration in §97.B. Working name — final naming TBD. | Open design discussion between Heath and Walther |
| Standalone disease-signature-matrix v1.5 CSV | Binary export of the matrix currently embedded inside `GAPE_WEB_v13.py`. Placeholder dir already created at `pipeline_runtime_matrices/disease_signature_matrix/`. | Pending decision on whether to externalize before orchestrator design |
| Per-card residual maps for cards other than breast-epic | Per-card CSVs at `pipeline_runtime_matrices/card_residual_maps/<card>/`. | Per-card VALs locking thresholds |
| Sex / batch / ancestry / smoking foreground modules | Companions to `age_axis_foreground.py`; would live in `Biological_Physics/atlas_vault/components/`. | Phase B4 per Roadmap §10.2.2 |
| Probe response function (L3) | Per-probe transfer function as a separate module; documented gap in L3 grading. | Atlas-wide probe characterization work |
| Manifest schema (standalone JSON Schema) | The intake schema as a versioned file rather than engine-embedded validation. | Orchestrator design |
| CANNOT_SAY list (standalone JSON) | Legal-boundary regex/keyword list as a versioned file rather than engine-embedded checks. | Orchestrator design + legal review |

### §97.D — How to use this table

If the SOP names a path you cannot find in the repo: search §97.A first. If not there, it is in §97.B (logic inside the engine, no separate file) or §97.C (does not exist yet). Anything NOT appearing in any of §97.A / B / C should be treated as a SOP error and flagged for v1.2 cleanup.

The "Used at SOP §" column tells the operator which SOP sections describe each artifact's role.

---

## §98. Glossary — CMB ↔ methylome term map

The two-way translation between cosmology vocabulary and methylome vocabulary. Use this when reading a CMB paper to understand its CPG equivalent, or when writing CPG documentation to find the right CMB analog.

| CMB term | Methylome term | Definition (CPG context) |
|---|---|---|
| **Bolometer / detector** | Methylation probe | Single measurement element on the array |
| **Bolometer timestream / TOD** | IDAT file pair | Raw detector intensities pre-calibration |
| **Frequency channel** | Probe type (Type I / Type II) | Detector class with distinct response characteristics |
| **Multi-frequency observation** | Multi-substrate (HM450K / EPIC) | Same biology measured by different detector chemistries |
| **Calibration source** | Bisulfite conversion control / Specificity control | On-chip references with known expected response |
| **L1 data (raw timestream)** | IDAT raw intensities | Pre-calibration |
| **L2 data (calibrated)** | β values | Post-calibration physical measurement per coordinate |
| **L3 data (sky map)** | Per-CpG β matrix | Calibrated map ready for component separation |
| **Component separation** | Deconvolution | Decompose multi-component signal into per-component fractions |
| **Commander algorithm** | Walther IAM Deconvolver | Primary Bayesian/optimization-based component separation |
| **NILC algorithm** | NILC v2 deconvolver | Cross-method GLS in departure-from-consensus space |
| **SMICA / SEVEM** | (future methylome variants) | Additional cross-method voices |
| **Foreground** | Cell composition / age / sex / batch / ancestry / smoking | Known contaminating signals that overlap with the target |
| **Galactic dust** | Age-axis foreground | Strongest single foreground; per-CpG / per-frequency template |
| **Galactic mask** | Stromal MCMC coverage gap | Region where the calibrated reference is insufficient |
| **Beam profile / response function** | Probe response function (PRF) | Per-detector signal-vs-truth transfer function |
| **Beam deconvolution** | PRF correction (provisional) | Inverting the per-detector transfer function |
| **Power spectrum C_ℓ** | Per-class entropy H | Information-theoretic decomposition by component basis |
| **Dimensionless cosmological parameter** | A-score | Calibrated ratio against physical reference (H_min) |
| **Critical density ρ_crit** | Architectural floor H_min | Reference scale for dimensionless ratio |
| **Mahalanobis distance / parameter tension** | Mahalanobis hyper-volume distance | Multi-D departure under appropriate covariance |
| **Joint posterior banana** | 2D A-score posterior shape | Curved 2D contour encoding parameter degeneracy (TODO 2.3) |
| **Look-elsewhere effect (LEE)** | N8 correction | Multiple-testing correction for searched feature space |
| **FFP10 / NPIPE simulations** | Synthetic patient generator | End-to-end pipeline test on known-truth synthetic data |
| **Signal injection / recovery test** | N6 / N7 | Inject known signal, verify chain recovers it |
| **Jackknife null** | N1 (label permutation) | Shuffle case/HC labels, recompute, verify signal disappears under null |
| **Stratified jackknife** | N2 (age strata) / N3 (sex strata) | Permutation within potential-confound strata |
| **Half-mission-difference null** | N4 (cohort split replication) | Verify signal replicates across cohort sub-samples |
| **Scan-direction null** | N5 (plate position null) | Test for systematics tied to measurement-apparatus property |
| **Acoustic peak (in C_ℓ)** | TODO 2.1 — methylation correlation length peaks | Future analysis (Phase C) |
| **Bispectrum** | TODO 2.2 — methylation bispectrum | Three-point correlation (Phase C) |
| **CMB lensing** | TODO 4.3 — field-effect reconstruction | Lensing-style methylome distortion (Phase D+) |
| **Mollweide projection** | Same | Equal-area sky projection used in all four plates |
| **HEALPix pixelization** | Same | Equal-area hierarchical pixelization (NSIDE=128 for methylome, 512 for CMB plates) |
| **Sealed result** | Sealed CPG-VAL | Result that passed declared null suite |
| **Restated result** | Restated CPG-VAL | Result with one null failure; preserved with limitation |
| **Retracted result** | Retracted CPG-VAL | Result withdrawn after multi-null failure |
| **Chain of custody** | Chain of custody | Link-by-link audit trail; identical concept |
| **Confirm (not validate)** | Confirm (not validate) | Framework predicted, data consistent (not pass/fail testing) |

---

## §99. The eight Mahaffey Numbers — H_min values frozen 2026-04-06

**These are the only constants in the framework.** Every A-score in the system divides by one of these eight numbers. They are derived from the IAMAtlas REBUILD MCMC posteriors and frozen on 2026-04-06.

| Class | H_min value | Convergence |
|---|---|---|
| terminal | 0.7728 | R̂ < 1.001 |
| immune | 0.838889 | R̂ < 1.001 |
| secretory | 0.843264 | R̂ < 1.001 |
| cycling | 0.856055 | R̂ < 1.001 |
| progenitor | 0.852216 | R̂ < 1.001 |
| stromal | 0.86295 | R̂ < 1.001 (with 4.93% MCMC coverage — the methylome's galactic mask) |
| stem_adult | 0.873718 | R̂ < 1.001 |
| stem_pluri | 0.982166 | R̂ < 1.001 |

**Single source of truth:** `IAMAtlasREBUILD_provenance.json`, key `h_min_values_frozen_2026_04_06`.

**Plain-language interpretation (from The Cellular Margin):**
- terminal class (H_min=0.7728) is most-committed (lowest entropy floor); a fully differentiated neuron carries the tightest specification.
- stem_pluri (H_min=0.9822) is least-committed (highest entropy floor); a pluripotent stem cell deliberately carries a looser pattern.
- Healthy human cells run at A ≈ 21× the underlying ATP/RT thermal floor; H_min normalizes the entropy reading per class.

**Discipline rules:**
- These values do not change between runs unless the atlas itself is rebuilt.
- A rebuild triggers the §96 re-run protocol on every sealed VAL.
- Any code that hardcodes these values WITHOUT reading from the canonical JSON is a chain-of-custody violation; the engine refuses to deploy.

---

## §100. Canonical cross-reference index

Maps every SOP section to its primary upstream sources. **Use this to find where a step is also discussed in the Recipe, Roadmap, or VAL inventory.**

| SOP § | Step | Recipe § | Roadmap § | Other |
|---|---|---|---|---|
| §11-§19 | Stage 0 intake | §3.1 | §10.1.1 L1 | — |
| §20-§27 | Stage 1 calibration | §3.2-§3.3 | §10.1.1 L2-L3 | — |
| §28 | Atlas load | §4 | — | Part I §4 |
| §29 | Marker pool | §4, §6.1 | — | — |
| §30-§31 | Walther | §6.2 | §10.1.1 L4 | Capability Translator §1, §2 |
| §32-§33 | NILC + cross-method gate | §6.2 | §10.2.2 Phase B2 | Phase_B2_1_FINDING.md |
| §35 | Age foreground | §5 | §10.2.2 Phase B3 | Phase_B3_FINDING.md |
| §41-§43 | Per-class A-score | §6.3 | §4 | Part I §7 |
| §44 | Per-cell-type A-score | §6.3 | §3.15 | Capability Translator §6 |
| §45 | Disease panel A-score | §6.5 | §7 | Capability Translator §7 |
| §47-§51 | Mahalanobis | §6.4 | §3.13 | VAL-002 |
| §52-§58 | Cellular age | §6.3 | TODO 2.4 / E1 | The Cellular Margin |
| §59-§64 | Tier breakpoints | §7 | — | Capability Translator §3, §4 |
| §65-§69 | Card matching | §8 | §10.2.6 Phase F | disease_signature_matrix_README.md |
| §70-§76 | Report assembly | §9 | — | Part I §10 |
| §77-§79 | Delivery | §10 | — | — |
| §80-§91 | L9 audit | §11 | §10.2.1 Phase A | §12 of Roadmap |

**Where the Recipe is the operational source-of-truth** (specific algorithms, exact formulas, runtime artifacts), this SOP cross-references it. Where the **Roadmap is the strategic source-of-truth** (Phase ordering, grading, planned upgrades), this SOP cross-references it. Where neither suffices, the SOP **declares its own**.

The Recipe is the proprietary vault (NOT in repo). The Roadmap is the canonical scoping doc (also Heath-only). The SOP is **operator-facing**; it teaches HOW the chain runs, not WHAT the proprietary derivation is.

---

## §101. CPG Plates 1-4 cross-reference

The four plates are the visual anchor of the framework. Each plate illustrates specific steps of the chain.

**Plate 1 — The Cosmic Microwave Methylome (`CPG_Plate_01_Cosmic_Microwave_Methylome.png`).**
- Sections illustrated: §6 (CMB ↔ methylome principle), §28 (Atlas load), §31 (per-class confidence — the stromal panel's MCMC coverage gap), §42 (Shannon entropy on per-class β).
- Visual content: 8-panel Mollweide projection of IAMAtlas REBUILD posterior β across architectural classes, in genomic order. Each panel shows one class's per-CpG posterior mean β across 481,966 CpGs.
- Key feature: stromal panel's "galactic mask" (4.93% MCMC coverage gap) is the methylome's declared known-unknown.

**Plate 2 — The Breast Pre-Diagnostic Anisotropy (`CPG_Plate_02_Breast_Anisotropy.png`).**
- Sections illustrated: §45 (disease panel A-score), §66 (per-card residual map application), §88 (N8 look-elsewhere on the chr6 zoom panel).
- Visual content: Full-sky scatter of 1,392 concordant breast-pre-dx CpGs (signed Cohen's d, sized by |d|, colored cyan-blue hypomethylated vs orange-yellow hypermethylated, 5.4:1 hypomethylation dominance). Lower panel: chr6 zoom showing MHC region enrichment (HLA classes I+II+III) carrying disproportionate concordant signal.
- Key feature: visualizes both the cross-genome anisotropy field-effect signature AND the chr6 MHC concentration that VAL-006 originally claimed (and that N8 honestly restated).

**Plate 3 — The Cosmic Microwave Methylome (Grandaddy plate — `CPG_Plate_03_Grandaddy_CMM_vs_CMB.png`).**
- Sections illustrated: §6 (CMB ↔ methylome principle), §25 (β distribution shape — the methylome's bimodality vs CMB's Gaussianity), §27 (the β matrix that becomes the methylome map), §87 (N7 end-to-end simulation — the right panel is the CMB equivalent of CPG's synthetic patient generation).
- Visual content: Side-by-side methylome vs CMB at matched projection (Mollweide), matched colormap, matched pixelization conventions. Top panels: full-sky overview. Bottom panels: zoom on small-scale anisotropy texture.
- Key feature: makes Heath's "math doesn't care if it's a CMB or a methylome" visually irrefutable while also making the biological difference (bimodal vs Gaussian) visible at the texture level.

**Plate 4 — Patterns Discovered (`CPG_Plate_04_Patterns_Discovered.png`).**
- Sections illustrated:
  - Panel A (Class-Difference Map): §29 (marker pool selection), §43 (per-class A-score).
  - Panel B (chr16+chr17 Cold-Patch Zones): §29 (marker concentration), §66 (residual map systematic in chr16/17), §94 (cross-method visible).
  - Panel C (Concordant Signal Density): §45 (panel A-score density visualization).
  - Panel D (Differentiation Gradient): §35 (age-axis foreground subtraction — what the age component looks like after isolation).
  - Panel E (MCMC Coverage Map): §31 (per-class confidence, the stromal mask), §95 (out-of-calibration handling).
  - Panel F (Breast Pre-Diagnostic Anisotropy): §66 (residual map at the cohort level), §88 (LEE).
- Key feature: each panel demonstrates a different chain-of-custody concept made visually concrete.

---

## §102. Change log

| Date | Version | Change | Authority |
|---|---|---|---|
| 2026-05-31 | v1 | Initial release. Parts I-V complete: §1-§102. Foundations + step-by-step + audit machinery + failure modes + reference. | Heath W. Mahaffey |

**Change discipline.** Every future modification to this SOP creates a new version entry. The previous version is archived (not overwritten). Changes that modify CANNOT-SAY language, H_min values, atlas references, or VAL sealing protocols require Heath's explicit authority. Changes that update file paths, add new failure modes catalog entries, or expand the glossary can be made by Walther under standing instruction. The change log is the chain-of-custody for the SOP itself.

---

**End of Part V (§97-§102).**

**End of CPG Chain-of-Custody SOP v1.** 102 sections. Five parts. Quantum-level granular operator manual from IDAT-on-server (§11) to audit-trail-closed (§79), with L9 chain-integrity scaffolding (§80-§91), failure-mode response catalog (§92-§96), and complete reference apparatus (§97-§102).

*The math doesn't care if it's a CMB or a methylome. — Heath W. Mahaffey, 2026.*

---

*v1 — 2026-05-31. Author: Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute. Working partner: Walther (Claude).*
