# EDEAR Cookbook — TESTING CHECKLIST
**Read this file BEFORE starting any new card or any new VAL study. No exceptions. No "I remember the rules from last time."**

This checklist exists because the same mistakes keep happening across cards. Compaction wipes memory; this file is the persistent memory. If you (Walther) start running cohort scripts before reading this file, you are doing it wrong. If you (Heath) see Walther skip this step, halt the session.

The checklist is organized by stage. Every box must be checked at its stage. Skipping a stage to "save time" is what produces the kind of overclaim+revert cycles that cervical-epic burned 4 hours on.

---

## EDEAR run-everything architecture — signed off 2026-04-26

**Heath signed off 2026-04-26 on the run-everything-every-time pipeline architecture.** Every IDAT runs Stage 1 + Stage 2 + Stage 3 with all panels and all reference atlases regardless of any single-stage result. No conditional gating. The patient's report can collapse uninformative tiles for display, but the underlying scoring is exhaustive on every IDAT. Per-class A-scores are computed for every tissue every IDAT. The architecture's primary value proposition is multi-disease detection: a patient with early AD + early breast cancer + chronic inflammation + cardiovascular drift fires four anomaly patterns simultaneously, which the report surfaces and the clinician interprets in combination.

**Operational consequences for testing (read these alongside the stage-checklist below):**

- **CHK-3.2 (cross-cohort baseline) is now mandatory every run, not optional.** See Stage 3 below — under run-everything, a single platform-induced baseline shift on a single tile silently corrupts every dual/triple-diagnosis pattern that uses that tile. CHK-3.2 is the structural defense.
- **Per-class A-scores must be computed for every tile, every IDAT.** No tile may be skipped because Stage 1 was below threshold. The display logic decides what to show; the scoring logic does not gate.
- **Pre-registered outcome criteria must enumerate the multi-disease detection patterns the VAL is designed to surface.** A pre-reg that only locks "disease X vs HC" decision criteria without locking how multi-disease anomaly combinations are interpreted is incomplete under the run-everything regime.
- **The official pipeline reference document is `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` (2026-04-26).** Any conditional gating language in pre-2026-04-26 docs is superseded; check the v2 reference doc first.

Spec source: `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` Part 1 (KISS architecture in one paragraph), Part 2 (full pipeline top to bottom), Part 9 (runnable analyses already locked).

---

## STAGE 0 — Before opening any code editor (mandatory pre-build read)

### CHK-0.1 Read the master `LESSONS_LEARNED.md` 
Last 10 entries minimum. If a similar card has been built (same H_min class, same specimen pathway, same disease family), read ALL its lessons.

### CHK-0.2 Read this `TESTING_CHECKLIST.md` end to end
Yes, every time. The point of a checklist is that it is consulted, not memorized.

### CHK-0.3 Read the closest-analog card's lessons
For cervical-epic the closest analogs were the LBC-pathway cards (which didn't exist yet) and the immune-class cycling cards. For the next card, look up the analog cards before building.

### CHK-0.4 Re-read the absolute rules block at top of master README v2.1
Especially: CCL-029 (cohort-completeness), CCL-030 (Test 1 vs Test 2), CCL-031 (bidirectional cancellation reserved for AD-instance pattern), no-fabrication rule, language discipline.

### CHK-0.5 Confirm specimen pathway matches the panel's training data
Xu-538 was trained on BUFFY-COAT (whole blood). It transfers cleanly to plasma cfDNA, blood-derived signals, and tissue with high immune infiltrate (because tissue immune compartments contain similar cell types). It does NOT automatically transfer to:
- LBC / pap smear (exfoliated epithelium + mucosal-resident immune cells)
- Saliva (different mucosal immune compartment)
- Urine (urothelial-resident immune cells)
- CSF (CNS-resident immune cells)
- Stool (gut-resident immune cells)

**If the new card has a primary specimen pathway not in the validated transferability list, the prereg MUST include an explicit "panel transferability not yet established" caveat, and a null reading on that pathway is a TRANSFERABILITY finding, not a framework finding.** This was the cervical-epic VAL-076/077 mistake.

### CHK-0.6 Read `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` end to end
Once per session minimum, more often if the VAL touches the pipeline architecture itself. The reference doc supersedes any pre-2026-04-26 conditional-gating language in older READMEs and card files. The Stage 1 + Stage 2 + Stage 3 panels and reference atlases enumerated in Part 7 of the reference doc are the authoritative current production list. The **Queue-1 atlas integration list (post-2026-04-26 PM acquisition pass)** is: **UniLIFE (Guo 2025 Genome Med, ON DISK at `/home/claude/atlases/unilife/centUniLIFE_reference_matrix.csv`, 1,906 CpGs × 19 immune cell types — Stage 3 head-to-head VAL vs Salas Blood.EPIC IDOL baseline ON DISK at `/home/claude/atlases/salas_blood_epic/`, both ready)**, Tanaka 2025 (EGA-controlled), Cuadrat 2023 (originally cited as Konigsberg 2023 — corrected per CCL-046; deconvR R package distributes the Moss 2018 25-cell base, the 28-cell extended atlas with 3 ENCODE bulk heart additions is a v0.4+ build), EpiSCORE (ON DISK in R-data, HeartRef sub-panel bridged in VAL-111, pan-tissue non-cardiac requires per-tissue gene→CpG bridging), Caggiano 2021 (array-bridged in VAL-113 cardio sprint — 254 CpGs × 19 cell types CHK-3.1C passed), Capper 2025 MARLIN (mnp_training scaffold ON DISK, leukemia matrix is v0.4+ build-out task), Sabedot 2021 GeLB (ON DISK as R training script, requires GSE150289 cohort). Plus 2026-04-26 surveillance additions: 17-tissue Ageing Atlas (Jacques bioRxiv, NOT yet on disk), MethAgingDB (Zenodo, NOT yet on disk), Ontology-aware 190-CpG Kim (NOT yet on disk). Canonical inventory: `/home/claude/atlases/ATLAS_DOWNLOAD_MANIFEST.md`. Items approved for v0.3 are NOT yet in production scoring — VALs that name a Queue-1 atlas must specify whether they are using the published external classifier (allowed) or claiming integrated scoring (not yet allowed without the atlas-integration VAL having landed).

### CHK-0.7 Substrate normalization is REQUIRED before any A-score scoring (added 2026-04-29 from VAL-112+113 run-everything cardio sprint)

**This is a hard gate, not a recommendation.** Production scoring against any calibrated atlas requires the input β-matrix to be in a substrate the atlas was calibrated against. Raw IDAT files from a customer's lab CANNOT be scored directly against EDEAR atlases — they must first be normalized to a calibrated substrate.

**Calibrated substrates (as of 2026-04-29):**
- TCGA HM450 sesame Level 3 — VAL-106/107 substrate baseline; VAL-112/113 atlas calibration anchor
- All three cardio Stage 2 atlases (layered Moss+Loyfer deduped, EpiSCORE HeartRef bridged, Caggiano TIM bridged) have CHK-3.1B q5 thresholds + per-tile healthy-floor A-score distributions sealed against TCGA HM450 sesame Level 3 specifically

**Within-cohort self-cal substrates (operational fallback only, NOT calibrated):**
- GenomeStudio AVG_Beta HM450 (used by VAL-108 stroke + VAL-110 BAV)
- minfi `preprocessFunnorm` HM450 (used by VAL-109 PAH)
- minfi noob-bg-corrected EPIC v2 (CCL-040 deferral pathway)

**Production deployment requirement.** A new customer's IDAT files must go through one of these normalization paths before scoring:
1. **sesame** (Bioconductor, Triche lab) — sturdiest path; produces sesame Level 3 β values matching the VAL-106/107/112/113 calibration substrate. The `deconvR` R package and `sesameData` package both ship sesame normalization.
2. **minfi** (Bioconductor, Hansen lab) — `preprocessFunnorm` or `preprocessNoob` are alternatives; results in within-cohort self-cal substrate not currently calibrated against TCGA reference. Use only when sesame is unavailable AND the prereg explicitly documents within-cohort self-cal.
3. **GenomeStudio AVG_Beta** — Illumina's own pipeline; treated similarly to minfi (within-cohort self-cal).

**The gate:** before any production scoring is allowed, the prereg explicitly states which substrate normalization was applied AND whether that substrate has a calibrated CHK-3.1A baseline + CHK-3.1B per-atlas threshold sealed against a structurally-separated healthy reference. If the substrate is uncalibrated, the prereg flags this and uses within-cohort self-cal as the operational fallback with explicit caveat. This is honest and operationally usable, but the resulting Cohen's d values are within-cohort relative effect sizes, NOT calibrated against the universal healthy-floor reference.

**Failure mode this CHK is designed to catch.** A customer's IDAT files arrive, get extracted to β values via whatever pipeline the lab uses, then get scored against EDEAR atlases that were calibrated on a different substrate. The A-scores produced are mechanically correct (the math runs) but the case-vs-control comparison against the calibrated healthy-floor distribution is invalid — different substrates produce different absolute β distributions, so the calibrated thresholds don't apply. The result is silently miscalibrated A-scores that look reportable but aren't.

**Generalization.** This rule applies to every card, not just cardio-epic. Each card's atlas list must have its substrate calibration documented. Future cards that integrate a new atlas must include the substrate calibration VAL (VAL-112 / VAL-113 template) BEFORE scoring against any disease cohort.

---

## STAGE 1 — Landscape survey (before any cohort is named)

### CHK-1.1 Every candidate cohort gets a Sample_title verification
For each cohort in the landscape survey, fetch the GEO series matrix metadata (small file, ~5-10 KB) and inspect at least:
- Sample_title (does it match the survey claim about disease/cohort?)
- Sample_source_name_ch1 (is the tissue what we think it is?)
- Sample_characteristics_ch1 (does the disease grading match?)

VAL-075 GSE38266 was claimed as cervical HPV-stratified in the landscape survey. Sample_title showed "HNSCC" prefix. **One inspection step would have caught this.** Do it before locking the cohort into the run plan.

### CHK-1.2 Check platform compatibility with the panel
HM27 cohorts: probably incomplete Xu-538 coverage; flag and check.
HM450 cohorts: full Xu-538 coverage expected.
EPIC 850K cohorts: ~80% Xu-538 coverage typically (cervical-epic showed 434/538 = 80.7%). Coverage drift between platforms is a panel-effective-size change; document it in the prereg.
EPIC v2: even more drift; check probe-name changes carefully.

### CHK-1.3 Check supplementary file format BEFORE downloading 1+ GB files
For supplementary files in GEO suppl/ folder:
- File suffix `.AVG_Beta` → GenomeStudio output, raw β (good)
- File names ending `_betas`, `_beta`, or `Matrix_processed.csv.gz` → usually raw β
- File names with version suffixes (`_betas_2`, `_corrected`, `_normalized`, `_residuals`) → likely PROCESSED, not raw
- File names ending `_Mvalues` or `_Mval` → M-values, need conversion via β = 2^M / (1+2^M); but verify the M-values are RAW not residualized first
- File names with `RAW.tar` → raw IDAT files, gold standard but require minfi/sesame to process

**If the file name is ambiguous, fetch the source paper's Methods section and find the exact pipeline. The paper will say "we deposited the BMIQ-normalized β values" or "we deposited the batch-corrected residual M-values" — it tells you what's in the file.**

### CHK-1.4 Identify clinical-grade panels published on the same disease
Before locking a card to Xu-538 scoring, check the published clinical-grade panels for the disease:
- Cervical: FAM19A4/miR124-2 (QIAsure), ZNF671/SOX17/DLX1 (GynTect), PAX1/NREP-AS1 (Bowden 2025), EPB41L3
- Pancreatic: ADAMTS1/BNC1 (Bauer 2018)
- CRC: SEPT9 (Epi proColon), Cologuard panel
- Lung: SHOX2/PTGER4 (Epi proLung)
- Breast: PITX2

If clinical-grade panels exist showing strong signal on the same cohorts where the framework reads null, **the framework's panel does not transfer — that is the finding, not "the disease has no signal".** The cervical-epic v0.1 mistake was to draft "framework null finding" outcomes when PAX1/NREP-AS1 achieved AUC 0.92 on the same GSE287994 cohort that Xu-538 nulled.

### CHK-1.5 Substrate-scope check on framework predictions (heme-LL-009 ABSOLUTE)

Before comparing a VAL result to any Issue 002 framework prediction, **verify the prediction's substrate scope matches what the VAL is actually measuring.** Issue 002 immune-class A-score predictions (e.g., A_AML ≈ 1.10, A_DLBCL ≈ 1.13) refer to **5-substrate combined cfDNA A-score** (methyl + nucl + fuzz + WPS + frag) — the future L2/L3 multi-assay platform target. v1 EDEAR launches on 450K/EPIC arrays which produce **single-substrate methyl-only buffy-coat A-score**. These are different things at different points on the platform roadmap; they are not directly comparable.

VAL-082 caught this in real time — A_AML measured at 0.54 against Italian healthy 0.44 (ΔA = +0.10, d = +3.71) initially looked like a mismatch with Issue 002's 1.10 figure. Substrate-scope check resolved it: 1.10 is the L2/L3 5-substrate combined target; 0.54 is the v1 single-substrate methyl-only reading; both are correct for their respective platform tiers. **Operational rule:** every VAL outcome.md that cites an Issue 002 framework prediction must explicitly state which substrate scope the prediction refers to and which substrate scope the VAL measures. Cross-tier comparison requires translation, not assumption.

### CHK-1.6 Cohort access tier classification (heme-LL-011)

Every cohort in the landscape survey gets classified into one of three access tiers, **before** locking the run plan:

- **Tier 1 — GEO/ArrayExpress publicly deposited.** Immediate access via FTP. Examples: GSE99511, GSE62298, GSE51057.
- **Tier 2 — EGA controlled access (European Genome-phenome Archive).** Requires formal data-access application via EBI; turnaround weeks-to-months. Examples: EGAS00001000272 (Kulis 2012 CLL), EGAS00001000174 (Dietrich 2018 CLLmethylation).
- **Tier 3 — Biobank-gated.** Requires formal data-access application via the originating biobank consortium; turnaround often 6+ months. Examples: EPIC-Italy + NSHDS (EnviroGenomarkers CLL), Rotterdam Study (VAL-046 pancreatic), MCCS (CLL up to 18 yr pre-dx), CINCS Bukowski (cervical pre-dx).

**Long-window pre-diagnostic methylation cohorts cluster at tier 3** because of human-subjects protections on archived clinical biobanks. Reaching `single_cohort_validated` tier on pre-diagnostic detection therefore requires biobank applications, NOT just GEO downloads. Heme-epic v0.1 documented this honestly: lymphoid arms cannot reach `single_cohort_validated` from publicly-accessible 450K data alone. The cohort-completeness statement (per CCL-029) for any card with a long-window pre-dx ambition must list which target cohorts sit at which tier and what the action plan is for tier 2/3 cohorts.

---

## STAGE 2 — Per-VAL prereg (before pre-seal locking)

### CHK-2.1 Pre-locked decision criteria for ALL outcomes
Pre-reg must specify the d magnitude and CI conditions for each possible outcome (O1_PASS, O2_PARTIAL, O3_NULL, O4_BIDIRECTIONAL, O5_NEGATIVE, O6_UNEXPECTED). No outcome can be added post-hoc.

### CHK-2.2 Anchor-vs-cohort baseline comparison declared
For every replication or cross-cohort run, the prereg must specify: "the first analysis after parsing β values is healthy-vs-healthy comparison between this cohort and the anchor cohort. If healthy mean A differs by >1 SD, the disease-vs-control numbers are NOT directly comparable to the anchor cohort." This was the missed step at VAL-074.

### CHK-2.3 Saturation-flag check declared as mandatory
Prereg must specify that the results JSON includes per-substrate saturation flag status, even when the answer is "no saturation". Block 7 architecture exists for this.

### CHK-2.4 Panel transferability caveat for new specimen pathways
If the cohort uses a specimen the panel hasn't been validated on, prereg must include the "panel transferability not yet established" caveat. Null findings on novel specimens are transferability findings.

### CHK-2.5 Test 2 placeholder declared
Per CCL-030, every prereg has the Test 2 (lymphoid vs myeloid) placeholder noting it is blocked on OQ-2026-01 immune-atlas staging. NO claim of bidirectional cancellation can be made without Test 2 — only flagged as suspected pattern requiring future Test 2 evaluation.

### CHK-2.6 Layered-atlas reference selection (glioma-LL-007)
For Stage 2 cell-of-origin deconvolution, the canonical reference is the **layered atlas**: Moss 2018 (Supp Table S4) primary + Loyfer/Moss array atlas (`nloyfer/meth_atlas/reference_atlas.csv`) supplementary. The two are not interchangeable. **Moss 2018 stays primary for cells it covers** (lymph node, spleen, esophagus, small intestine, stomach, skin keratinocyte, ovary, adrenal cortex, breast myoepithelial, skeletal muscle, plus all the overlapping tissues). **The Loyfer-array atlas supplements** for sorted-cell entries Moss didn't have at array CpG resolution: `Cortical_neurons` (Moss had bulk "brain (cortex)"), `Vascular_endothelial_cells`, `Left_atrium` (Moss had bulk "heart"), `Pancreatic_duct_cells`, `Head_and_neck_larynx`, `Upper_GI`, plus EPIC-trained sorted immune cells.

The prereg must declare which atlas (or both) is being used for Stage 2. Default for new VALs is **layered**; if Moss-only or Loyfer-array-only is being used, the prereg names the reason (e.g., "this VAL uses Moss-only because the cell type of interest is lymph node, which Loyfer-array does not cover at sorted-cell resolution").

For terminal-class cards (glioma-epic, future cardio-epic), the layered atlas is REQUIRED — Moss-only Stage 2 will return NULL on terminal-class signal because Moss's bulk-tissue brain and heart entries do not separate the relevant sorted cells.

### CHK-2.7 Cell-of-origin atlas preregs MUST use magnitude-based |d| thresholds with direction labels (DISC-PROSTATE-002 / prostate-LL-007, formalized 2026-04-30)

**Absolute rule for any prereg specifying outcome thresholds on cell-of-origin atlas tiles** (ProstateRef LE/BE/EC/Fib/Leu/SM, BreastRef tiles, LungRef tiles, KidneyRef tiles, ColonRef tiles, HepatocyteRef tiles, PancreasRef tiles, BrainRef tiles, etc.).

When biology supports a direction-flip pattern (cell-of-origin dedifferentiation produces NEGATIVE-direction A-score shifts; cell-of-origin lineage hyperplasia produces POSITIVE-direction shifts), pre-registered outcome thresholds MUST use:

- **Magnitude-based**: `|d_paired| ≥ {threshold}` (NOT `d_paired ≥ +{threshold}` or `d_paired ≤ −{threshold}`)
- **Direction labels**: `{tile_name}_POSITIVE` vs `{tile_name}_NEGATIVE` recorded per outcome firing
- **Biological interpretation per direction**: explicit labels in outcome.md for what each direction means biologically

**Why this rule exists.** VAL-118 first execution sealed O5 because original prereg pre-locked O2 as positive-direction-only. Observed pattern was clean strong negative (luminal dedifferentiation). CCL-041 forbids post-hoc sign-flip; the amendment had to be sealed BEFORE re-execution, costing one full prereg+amendment cycle. Direction-locked thresholds on cell-of-origin tiles produce avoidable O5 outcomes that should have been pre-anticipated.

**What this rule does NOT change.** Bulk-tile or pooled metrics where direction is biologically uniform (e.g. Stage 1 Xu-538 pooled A_immune via Shannon symmetry — binary entropy is symmetric around β = 0.5 anyway) do NOT require this rule. Pooled-entropy metrics remain direction-agnostic by construction.

**Pre-registration template language.** Required wording for cell-of-origin atlas tile outcomes:

> Outcome `OX_{tile}_TILE_DIFFERENTIATING` fires if `|d_paired|` for `{atlas}.{tile}` ≥ `{threshold}`. Direction label = `{tile}_POSITIVE` if d_paired > 0; `{tile}_NEGATIVE` if d_paired < 0. Biological interpretation: `POSITIVE` = `{e.g. lineage hyperplasia, cell-type expansion}`; `NEGATIVE` = `{e.g. luminal dedifferentiation, lineage loss}`. Pre-registered outcome class enumerates BOTH directions with separate biological interpretation labels.

### CHK-2.8 CHK-3.1B coverage threshold pre-locks must match substrate floor, NOT default 95% (formalized 2026-04-30 from VAL-117 amendment)

**Operational rule for any new card sprint pre-locking CHK-3.1B atlas-subset coverage thresholds in a Phase B calibration prereg:**

The CHK-3.1B coverage gate is a per-sample atlas-CpG-intersection coverage check. The pre-locked threshold MUST match the substrate floor for the calibration cohort, NOT a default 95%.

**Substrate floors (sealed precedent):**
- TCGA HM450K sesame Level 3: ~80% (per VAL-117 amendment + cardio VAL-112 implicit precedent — TCGA's QC pipeline routinely drops 12-20% of probes via cross-reactive masking, SNP-overlap, and detection p-value failures)
- EPIC 850K native: ~85% typical (cohort-dependent)
- HM450K minfi preprocessFunnorm: ~92% typical
- 27K → 450K bridges: substrate-specific, check before pre-lock

**Why this rule exists.** VAL-117 first execution failed CHK-3.1B at 0/210 samples because original prereg specified ≥95% coverage threshold. TCGA HM450K sesame Level 3 produces 80-88% coverage on bridged atlases — never 95%. Original 95% pre-lock was a specification error inconsistent with cardio precedent. Amendment changed to 80%, sealed BEFORE re-execution. This rule prevents the error in future card sprints.

**What this rule does NOT change.** The CHK-3.1B "q5 threshold" cardio reports (e.g. 0.4283 for HeartRef, 0.6839 for Loyfer) is a separate metric — the 5th percentile of the per-sample A-score distribution, NOT a coverage gate. Distinguish "coverage gate" from "A-score q5 threshold" in prereg specification.

**Pre-registration template language.** Required wording:

> CHK-3.1B per-sample atlas-CpG-intersection coverage threshold: `{floor}` (per substrate floor for `{calibration_cohort_substrate}`). Pre-locked PER SUBSTRATE; does NOT default to 95%. Substrate floor citation: `{e.g. VAL-117 amendment for TCGA HM450K sesame Level 3, ≥80%}`.

---

## STAGE 3 — Data integrity check (BEFORE scoring)

This is the stage that cervical-epic skipped twice. Every cohort gets these checks BEFORE any A-score is computed.

### CHK-3.1 β distribution sanity check
After parsing β values, dump the distribution shape from at least 3 sample rows:
- What fraction of β values are in [0.4, 0.6]? Real raw β is **<10%** (bimodal data).
- What fraction of β values are at extremes [<0.1 or >0.9]? Real raw β is platform-specific — see table below.
- What is the median β? Real raw β has median typically 0.4-0.7 depending on tissue, but the distribution should be bimodal not flat.

**If <20% of β are at extremes AND >40% are in [0.4, 0.6], the file is processed/residual data, not raw β.** STOP. Do not score. Either find the raw IDATs or document the cohort as "not interpretable at v0.1, requires raw-IDAT reprocessing for v0.2+".

VAL-077 distribution was 50% in [0.4, 0.6] and 12% at extremes. That's the smoking gun. Walther scored it anyway and had to revert. Don't do this.

#### Platform-specific extreme thresholds (per CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION, formalized 2026-04-28)

The "extreme >X%" threshold is platform-specific. Standard TCGA pipeline dye bias correction softens bimodality slightly relative to raw EPIC. Set the platform-appropriate threshold in the prereg before β-access:

| Platform | extreme threshold | middle threshold | Status |
|---|---|---|---|
| Raw EPIC β / EPIC v2.0 β (un-normalized) | > 30% | < 10% | Established (VAL-100) |
| TCGA HM450 sesame Level 3 β (full-genome CHK-3.1A) | ≥ 50.5% | ≤ 9.0% | **Established VAL-106** (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210, full-genome f_extreme 55.87% ± 2.44%) |
| TCGA HM450 sesame Level 3 β (cardio-epic CHK-3.1B subset, 8,100 CpGs) | ≥ 55.0% | ≤ 8.5% | **Established VAL-107** (same cohort on cardio-epic marker subset; subset SHA `5a00e29ace75daae5a9...`); n_subset_valid ≥ 7,000 of 8,100 |
| GenomeStudio AVG_Beta HM450K (un-normalized) | within-cohort self-cal | ≤ 13% | **VAL-108 / VAL-110 within-cohort only** at v0.1; pending generalizable structurally-separated calibration VAL |
| minfi `preprocessFunnorm` HM450K | within-cohort self-cal | ≤ 11% | **VAL-109 within-cohort only** at v0.1; pending generalizable calibration VAL |
| minfi noob-bg-corrected EPIC v2 | (substrate fails CHK-3.1A by design — CCL-040 deferral) | n/a | VAL-100 GSE282666 — known-fail substrate, refer to raw IDAT reprocessing pathway |
| Other platforms | TBD | TBD | Document at first calibration VAL on platform |

**Critical rule: platform threshold values must NEVER be set by retroactive accommodation of the data that triggered the discovery of platform mismatch.** That is post-hoc threshold accommodation, not pre-registration. A SHA stamp on a threshold derived from the test data does not make it methodologically pre-registered. The proper calibration pathway is to use TCGA samples from a tissue NOT under active test (e.g., TCGA-KIRC adjacent-normal, TCGA-PRAD adjacent-normal), measure the bimodality distribution there, set the threshold from THAT distribution, seal it, and apply it to future test cohorts. See CCL-041 for the full rule and the VAL-101/VAL-102 self-correction case study.

**VAL-101 case study (2026-04-28).** Pre-locked CHK-3.1 threshold (extreme >30%, raw-EPIC default) tripped on TCGA-LIHC HM450 sesame Level 3 at extreme 26.6% / middle 9.1%. Cookbook discipline honored the trip → outcome `O5_DATA_INTEGRITY_FLAG`. A VAL-102 attempt to re-seal with a TCGA HM450 platform threshold (extreme >20%) derived from VAL-101's tripped data was caught as post-hoc accommodation and voided before execution; audit trail at `Biological_Physics/validation_runs/VAL-102/VOIDED_BEFORE_EXECUTION.md`. The TCGA HM450 platform threshold was subsequently established via VAL-106 calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal (n=210); see split convention subsection below.

#### CHK-3.1A and CHK-3.1B — Split convention (per CCL-042 LL-CHK-3.1-A/B-SPLIT, formalized 2026-04-28)

VAL-106 calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal HM450K sesame Level 3 (n=210) measured full-genome f_extreme ~55.87% — far outside the empirical 18-35% range that had been pre-locked from three prior data points (VAL-101 26.6%, VAL-099 24.4%, GSE69138 ave_beta peek 21.9-27.3%). Investigation showed the prior data points were CpG-subset measurements (Loyfer 25-tile markers, top-of-file rows), not full-genome measurements. The cookbook had been silently conflating two distinct measurement questions under one CHK-3.1. Going forward, CHK-3.1 is split into two distinct named checks; both must pass for a sample to clear data-integrity gating.

**CHK-3.1A — Full-genome bimodality (substrate gate).** Compute f_extreme and f_middle on **every valid β value in the input file**, no subsetting. Threshold per measurement substrate (raw EPIC, TCGA HM450 sesame Level 3, GenomeStudio AVG_Beta, minfi `preprocessFunnorm`, etc.). Established by calibration VAL on a structurally-separated healthy adjacent-normal cohort. Reused indefinitely for that substrate. Catches CCL-040-style processed-output substrates and pipeline-level integrity failures.

**CHK-3.1B — Card-specific marker subset bimodality (panel-coverage gate).** Compute f_extreme and f_middle on the **union of all CpGs the card's scoring will use** — Stage 1 panel ∪ Stage 2 atlas markers ∪ Stage 3 atlas markers, as applicable per card. Per-card threshold derived from the same calibration cohort as CHK-3.1A but computed on that card's specific union. Recomputed when the card adds a new atlas or updates a marker panel. Stored in the card's `universal_pipeline_acknowledgment.chk_3_1_thresholds_per_substrate` block. Catches probe-list lift-over dropouts (450K → EPIC v1 → EPIC v2 panel coverage gaps), ancestry-specific failed probes, atlas-specific marker damage in regions affected by a localized artifact.

**Conjunction rule.** A sample passes CHK-3.1 iff (CHK-3.1A passes) AND (CHK-3.1B passes). CHK-3.1A failure routes to the CCL-040 reprocessing pathway (raw IDAT re-extraction, alternative pipeline). CHK-3.1B failure routes to the panel-coverage repair pathway (probe-list verification, alternative atlas with overlap, defer card to next version).

**First card built natively under split convention: cardio-epic v0.1.** Phase 3 retroactive review will bring breast-epic, lung-epic, ad-immune, hcc-epic, crc-epic, kidney-epic, and cervical-epic into the split convention without unsealing any sealed VAL outcomes — sealed outcomes are honored as decisions made under the rules at the time, with retroactive reclassification logged as documentation only. Specifically: VAL-100 reclassified as CHK-3.1A failure (substrate is minfi noob-bg-corrected processed output); VAL-101 reclassified as CHK-3.1B-style measurement against CHK-3.1A-derived threshold (convention mismatch in the cookbook at the time); VAL-077 reclassified as CHK-3.1A failure (residual M-value substrate). All sealed outcome statuses unchanged.

**EDEAR commercial deployment (CCL-037).** Production deployment runs single calibrated patient-vs-internal-reference pipeline, structurally insulated from public-cohort substrate diversity. Under the split, CHK-3.1A is computed once per customer (substrate gate) and CHK-3.1B is computed per disease card (panel-coverage gate). A customer with substrate-clean data but partial panel coverage on some cards receives the cards their data supports rather than an all-or-nothing report failure.

### CHK-3.1C Atlas-deduplication gate — **MANDATORY at every new atlas integration** (added 2026-04-29 from CCL-047 dedupe finding)

CCL-047 (cardio-epic v0.2.2 Phase A acquisition diagnostic) found that the cookbook's `loyfer_moss_2018/reference_atlas.csv` contains 7,890 rows but only 6,105 unique CpG IDs — 1,785 duplicate rows (1,270 CpGs duplicated 2-8× each). All checked duplicate rows have identical β values across the 25 cell-type columns, so duplicates do not introduce within-row inconsistency. However, val_108.py's Stage 2 scoring loop computes per-tile A-scores via `(sample_β - tile_ref_β).abs().mean()` and pandas `loc` on a duplicated Index retains all matching rows — so identical-row duplicates reweight CpG contributions to the per-tile mean. The duplicated CpGs in the cookbook file are systematically lower-β (~0.42-0.43) than the non-duplicated CpGs (~0.50-0.52), producing a uniform −0.017 to −0.025 bias on per-tile reference β across all 25 tile columns. The bias is uniform across all patients regardless of disease state, so within-cohort Cohen's d (case mean - control mean) is unbiased and qualitative cardio findings (Stage 1 immune workhorse VAL-110 d=+1.08, EpiSCORE HeartRef tile floor) are robust. But absolute A-score magnitudes are biased by ~0.003-0.024 per tile in the sealed VAL outputs.

**Companion to CHK-3.1A (substrate gate) and CHK-3.1B (panel-coverage gate).** Before any new atlas integration calibration VAL is sealed, the prereg confirms the atlas file in atlas_vault has zero duplicate CpG IDs:

```python
import pandas as pd
df = pd.read_csv(atlas_path, index_col=0)
assert not df.index.duplicated().any(), \
    f"Atlas {atlas_path} has {df.index.duplicated().sum()} duplicate CpG rows. " \
    f"Dedupe before scoring."
```

If duplicates exist in a Bioconductor / R-package distributed atlas (sometimes deliberate — multiple tile entries with same CpG but different region annotations), the dedupe step is documented in the prereg and the original file preserved alongside as `..._with_duplicates.csv` for audit-trail. The gate is cheap (one-line pandas check) and prevents cooked-in calibration bias.

**Generalization.** CHK-3.1C applies to every reference matrix in atlas_vault, not just the layered Moss+Loyfer atlas. Stage 3 immune atlases (UniLIFE 19-cell, Salas IDOL 6-cell, EpiSCORE pan-tissue, Caggiano CelFiE TIM, MARLIN, Sabedot, etc.) all need the duplicate-CpG check at integration time. v0.3 cookbook engineering task: add a structural-validation script that walks every atlas in atlas_vault and reports duplicate-CpG counts, β-value ranges, missing-value counts, tile-column consistency. Output goes to `atlas_vault/INVENTORY.json` as a `structural_validation` block per atlas.

**Failure mode this CHK is designed to catch.** Atlas reference files are usually treated as black-box input — the cookbook downloads them, validates SHA-256, and uses them. Internal structural validation (duplicate-CpG check, β-value range check, missing-value check, tile-column consistency check) was not part of CHK-3.1A/B. Without CHK-3.1C, a duplicated-row atlas file silently biases all downstream A-score computations across every card that uses it. The Cuadrat 2023 acquisition diagnostic surfaced the cookbook's layered Moss+Loyfer atlas dedupe issue only because the deconvR Bioconductor package's bundled atlas (deduped, 6,105 CpGs) provided a comparison reference. Without that comparison, the bias would have stayed undetected indefinitely.

**v0.3 corrective execution for cardio-epic.** Per CCL-047 fix policy: (i) deduplicate the cookbook atlas file before v0.3 re-execution, (ii) preserve original 7,890-row file as `reference_atlas_v0.2_with_duplicates.csv` for audit-trail, (iii) re-run VAL-108/109/110 against deduped 6,105-row file, (iv) confirm sealed Cohen's d findings preserved within ±0.05 (expected, given uniform bias), (v) update calibration thresholds if any cardio-epic deployment uses absolute A-scores. Sealed cardio outcomes from v0.2 are not unsealed — outcomes are honored as decisions made under the cookbook rules at the time; v0.3 corrective execution adds new per-atlas results to the same cohorts under the corrected scoring.

### CHK-3.2 Healthy reference baseline cross-cohort check — **MANDATORY EVERY RUN, NEVER OPTIONAL**

Compare the new cohort's healthy mean A vs the anchor cohort's healthy mean A. If they differ by more than 1 SD of either group's healthy SD, the cohorts have a baseline mismatch. Flag in results JSON. Read the source paper to find what "normal" means in each cohort (population normal vs tumor-adjacent normal vs colposcopy-negative-but-not-population-normal).

VAL-073 healthy A = 0.681 ± 0.022; VAL-074 healthy A = 0.621 ± 0.035. Difference = 0.06 = 2.7 anchor-SDs apart. That's a baseline mismatch — flag it before drawing CIN3 conclusions.

**Run-everything elevation rule (Heath sign-off 2026-04-26).** Under the run-everything architecture every IDAT runs Stage 1 + Stage 2 + Stage 3 with all panels and atlases regardless of any single-stage result, which means a single patient may have **dual or triple diagnosis** (e.g. early AD + early breast cancer + cardiovascular drift, or glioma + chronic inflammation, or a heme cancer with a co-occurring solid-tumor pre-clinical signal). For multi-disease patterns to be detectable, each per-class A-score and each per-tissue ΔA must be calibrated against a baseline that is platform-correct and preprocessing-correct for the patient's IDAT. **A cross-cohort baseline mismatch on any single tile silently corrupts every downstream contrast that uses that tile, including the multi-disease ones.** This makes CHK-3.2 not a courtesy check but a hard prerequisite for run-everything's primary value proposition.

**Operational consequences (mandatory every run):**

1. **Every results JSON must contain a `cross_cohort_baseline_check` block** for every Stage 1 panel and every Stage 2 cell-type tile, comparing the cohort's HC mean A-score to the anchor (GSE51057 / Hannum 80-cell baseline / matching-platform anchor) in **anchor-SD units**. The block is mandatory regardless of whether a mismatch is detected. Empty/null cross-cohort blocks are a bug.

2. **Mismatch tiers (≥1 anchor-SD = flag, ≥3 anchor-SDs = invalidate cross-cohort statistic).** A mismatch < 1 SD is reported but not flagged. A mismatch 1–3 SDs is flagged with `baseline_mismatch_flag: true` and the cross-cohort comparison is reported but explicitly downgraded — within-cohort case-vs-control becomes the primary statistic. A mismatch ≥ 3 SDs invalidates cross-cohort absolute comparisons entirely (within-cohort only). Documented examples: AddNeuroMed cortical-neuron HC vs GSE51057 HC = +16.7 anchor-SDs (450K vs EPIC marker-coverage gap, ad-LL-006); VAL-074 vs VAL-073 cervical normals = 2.7 SDs (HPV-negative-only vs population-normal selection, cerv-LL-010); GIFT GSE53740 HC vs 80-cell baseline = +2.306 SD (Ferrari 2014 ComBat preprocessing offset, VAL-057).

3. **Every patient-facing report under run-everything must surface the platform tag and the cross-cohort baseline status of every reported tile.** A clinician reading "patient breast_ductal ΔA = +0.12" needs to know whether the +0.12 was computed against a baseline measured on the same platform with the same preprocessing or against a platform-bridged baseline. Dual/triple diagnosis decisions are made tile-by-tile; each tile's baseline confidence must be visible.

4. **Within-cohort vs cross-cohort hierarchy is now an absolute rule, not a fallback.** For run-everything:
   - **Primary evidence:** within-cohort case-vs-control on the same IDAT batch with the same preprocessing pipeline.
   - **Secondary evidence:** cross-cohort comparisons against an anchor with matching platform AND matching preprocessing (e.g. both 450K with the same minfi pipeline).
   - **Tertiary evidence:** cross-cohort comparisons across platforms or preprocessing pipelines, ONLY with explicit `baseline_mismatch_flag` and platform-stratified thresholds.
   - **No statement that depends on a tile's absolute A-score for a single patient may use a tertiary-tier comparison without surfacing the mismatch caveat to the clinician.**

5. **The cross-cohort baseline check is now on the pre-send checklist** (alongside referee-language, citation-discipline, no-fabrication rules). A VAL outcome that does not include the CHK-3.2 block is incomplete and may not be merged.

**Why this matters under run-everything specifically.** Pre-architecture (gated): a patient's report shows one tile (the disease the clinician ordered the test for); a baseline-mismatch on that one tile is a single error and the gating lets the rest of the pipeline stay clean. Post-architecture (run-everything): a patient's report shows 18 Stage 2 tissue tiles + Stage 3 sub-composition + Stage 1 panel scores simultaneously, and dual/triple diagnosis claims arise from the *combination* of which tiles cross threshold. **A single platform-induced baseline shift on cortical-neuron at +16.7 SDs would, under naive interpretation, falsely diagnose AD or glioma in every patient run on AddNeuroMed-format 450K data.** CHK-3.2 is the structural defense.

VAL-092 (2026-04-26) demonstrated this concretely: AIBL HC vs GSE51057 HC at +1.87 anchor-SDs and AddNeuroMed HC vs GSE51057 HC at +16.7 anchor-SDs both flagged before any case-vs-control comparison was reported, forcing within-cohort statistics as the primary outcome and protecting against a false +0.99 cross-cohort glioma-vs-healthy d that would have read as "validated" if CHK-3.2 had been skipped.

**VAL-093 (2026-04-26) demonstrated the opposite case — a clean cross-cohort baseline.** GSE51057 HC vs GSE51032 HC across all 25 Loyfer cell-type tiles passed at <0.25 anchor-SDs (max 0.24 SD on Bladder; 0/25 tiles flagged). Both cohorts are EPIC-Italy nested case-control on 450K with the same preprocessing pipeline. **This is the first clean cross-cohort baseline alignment in the cookbook.** When platform AND preprocessing match, the layered-atlas architecture works at the cross-cohort level and cross-cohort comparisons are valid at the secondary-evidence tier per CCL-034. Within-cohort statistics retain primary-evidence priority by rule. CHK-3.2 is mandatory regardless of expected outcome — the check is what tells us whether we're in the VAL-091/VAL-092 regime (cross-platform mismatch invalidates cross-cohort) or the VAL-093 regime (matched platform + preprocessing makes cross-cohort interpretable).

### CHK-3.3 Panel coverage report
For every cohort, results JSON must report:
- Total Xu-538 CpGs in the platform (full Xu-538 = 538; HM450 = 538; EPIC 850K = ~434)
- Per-sample mean Xu-538 CpG count after QC
- Fraction of samples passing QC threshold (default 400 CpGs minimum)

If panel coverage drops >10% from the anchor cohort, flag it. Coverage drift means the effective panel is changing, which can shift A-score baselines independently of biology.

### CHK-3.4 Sample-group assignment spot check
Pull 3 GSMs from each disease group and verify their assignment by reading the Sample_title and Sample_characteristics_ch1 fields directly. If any of the 3 don't match the assigned group, audit the full assignment logic before continuing.

VAL-074 had 4 disease-status `Sample_characteristics_ch1` rows in the metadata. If the parser had matched on partial string, samples could have been mis-grouped. Spot-check catches this.

### CHK-3.5 Saturation flag check
For each per-sample A-score:
- Compute distance to A_ceiling for the relevant H_min class
- Flag if A ≥ (A_ceiling − 0.005)
- Report the per-group fraction of samples flagged in the results JSON

For immune class: A_ceiling = 1.1921, flag at A ≥ 1.1871.
For cycling class: A_ceiling = 1.1681 (methyl), flag at A ≥ 1.1631.
For secretory class: A_ceiling = 1.1859 (methyl), flag at A ≥ 1.1809.

VAL-077 mean A = 1.011 — under flag, but at 84.8% of ceiling. Even when not flagged, report distance to ceiling so the reader can see headroom.

### CHK-3.6 Moss-coverage explicit statement (heme-LL-010)

Any card that interprets Stage 2 Moss NNLS deconvolution must explicitly note what Moss 2018 does and does not cover. **Moss 2018 reference includes 18 peripheral solid tissues. It does NOT include brain/CNS, eye, testis, or other immune-privileged sites** because those tissues do not shed measurably into peripheral blood under normal conditions (blood-brain barrier, blood-testis barrier).

Operational consequence: "Moss NULL on solid organs" rules out the 18 peripheral tissues in Moss's reference; it does NOT rule out CNS disease, primary CNS lymphoma, primary spinal cord tumors, or any other disease originating in immune-privileged tissue. **Heme-epic specifically inverts the Moss-NULL interpretation** ("Moss NULL on solid organs is the diagnostic feature for heme cancer") but this only holds when paired with Stage 3 lineage-specific shift; uniform Stage 3 + Moss NULL is a routing-ambiguity pattern that includes CNS-or-other-non-Moss-tissue disease as a differential.

**Required:** every card README that documents Stage 2 Moss interpretation includes a paragraph stating exactly which 18 tissues Moss covers, what Moss does not cover, and how the card handles the gap. Patient-facing reports must be honest that "Moss NULL on peripherals" is not a clean ruling-out of all cancer when neurological symptoms are present. Glioma-epic (TBD) handles the CNS pathway separately; v1 reports flag the gap rather than papering over it.

---

## STAGE 4 — Outcome interpretation (BEFORE drafting outcome.md)

### CHK-4.1 Biology consistency check
Before drafting the outcome:
- Is the result consistent with the published clinical-grade panels for this disease?
- Is the result consistent with the established immunology literature for this disease?
- Is the result consistent with the cohort's own published findings?

If the result contradicts all three, **the framework reading is the suspect, not the biology**. Go back to STAGE 3 and re-check data integrity. This was the cervical-epic mistake at VAL-076/077.

### CHK-4.2 Cancellation hypothesis check (CCL-031)
If the pooled Test 1 result is null but a directional pattern is suspected:
- Is this an AD-instance candidate (pooled-null + directional-pass)? If yes, document as "suspected bidirectional cancellation pattern, Test 2 required for confirmation" — DO NOT call it bidirectional cancellation without Test 2.
- Or is it more likely a panel-transferability issue (cell-mixture mismatch)?
- Or is it more likely a data-integrity issue (per CHK-3.1)?
- Or is it more likely a cohort-baseline issue (per CHK-3.2)?

Cancellation is the LAST hypothesis to consider, not the first, because Test 2 is currently blocked on OQ-2026-01.

### CHK-4.3 Outcome label must match the diagnostic state
- O1_PASS: data integrity confirmed, biology consistent, framework PASS.
- O2_PARTIAL: data integrity confirmed, biology consistent, weak signal.
- O3_NULL: data integrity confirmed, biology checked, framework null on a panel-transferability-validated specimen.
- O4_BIDIRECTIONAL_SUSPECTED: pooled Test 1 null, directional Test 1 pass, AD-instance pattern flagged for Test 2.
- O5_NEGATIVE: data integrity confirmed, framework reads opposite-sign vs anchor (e.g., VAL-074 pattern).
- O6_UNEXPECTED: data integrity FLAGGED (e.g., VAL-077 residual M-value pattern), or cohort-baseline FLAGGED (VAL-074), or any other diagnostic-pending status. Use this label whenever data integrity is uncertain.

### CHK-4.4 Language discipline (auto-applied)
Never: resolves, confirms, validates, proves, "first", "framework null finding"
Always: consistent with, tested against, predictions within the framework, "the data are consistent with"
Never: "cervical disease has no immune signal" (overclaim)
Always: "the universal Stage 1 panel does not transfer to LBC at the cohorts tested" (transferability)

### CHK-4.5 Heath-facing pre-publish review
Before writing any outcome.md or patching the card README:
- Summarize the diagnostic state in plain English
- State the data-integrity status, biology-consistency status, and framework-reading status separately
- Ask Heath if the diagnostic state is sufficient to publish, OR if more diagnostic work is needed

cervical-epic VAL-076/077 went straight from "got numbers" to "drafted outcome" with no Heath-facing pre-publish review. Heath had to halt the session twice. Pre-publish review prevents the halt-and-revert cycle.

### CHK-4.6 Card-specific routing-pattern verification (heme-LL-001 ABSOLUTE)

Different cards interpret the SAME three-stage signal pattern in different (and sometimes inverted) ways. Walther must verify the per-card routing logic before drafting the outcome:

- **Solid-organ cards** (breast-epic, lung-epic, crc-epic, prostate-epic, hcc-epic, pancreatic-epic, cervical-epic): "Stage 1 elevated + Stage 2 elevated on the matching solid organ" is the positive call. "Stage 2 NULL on the matching solid organ" is a negative finding for that card.
- **Heme-epic**: "Stage 1 elevated + Stage 2 NULL on ALL 18 peripheral solid organs + Stage 3 lineage-specific shift" is the positive call. **Stage 2 NULL is the diagnostic feature, not the absence of a finding.** Inverted vs solid-organ cards.
- **Immune-atlas (TBD)**: handles uniform Stage 3 elevation pattern (not lineage-specific). Pathway 4 differential covers inflammaging, autoimmune, chronic infection, possible non-Moss-tissue disease (CNS).

**Required:** every VAL outcome.md states the card-specific routing rule it is testing and confirms the result is interpreted within that card's routing logic, not the framework's default solid-organ logic.

The seven-pattern routing matrix (A-G) for heme-epic is documented in the heme-epic_README §"Commercial.web.py decision tree". Each card with non-default routing logic must publish its own routing matrix.

### CHK-4.7 SUPPRESSED tier check (heme-LL-003)

A_score below the age-decade healthy reference by >1 SD is a real signal — immunocompromised state, post-chemo, post-transplant, HIV, primary immunodeficiency, cachexia, late-stage marrow infiltration. **Every card must surface SUPPRESSED tier** (A_immune > 1 SD below age-decade healthy reference), not just elevation.

VAL outcome reporting for any cohort that includes potentially immunocompromised subjects (chemo-treated, post-transplant, HIV+, etc.) must check for SUPPRESSED-tier readings. SUPPRESSED is not a false-negative on the cancer differential — it is a separate clinical finding the report must surface.

Operational: results JSON includes per-sample tier classification using the four-bin patient-facing vocabulary (SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH with MARGINAL flanks). Cards that previously reported only three bins (NORMAL/MARGINAL/DETECTABLE/URGENT/FLOOR_BREACH on the elevation side) need updating.

---

## STAGE 5 — Card README update (after all VALs complete)

### CHK-5.1 No README update until ALL VALs in the card have completed STAGE 4
Partial-card README updates with some VALs flagged O6_UNEXPECTED produce inconsistent card states. Wait until the full battery is either passed or diagnostically resolved.

### CHK-5.2 Tier statement matches the weakest evidence
Card validation tier is the FLOOR of evidence, not the ceiling. If 4 VALs are O1_PASS and 1 is O6_UNEXPECTED, the card is `exploratory_pending_diagnostic`, not `cross_platform_validated`.

### CHK-5.3 Block 1-20 expectations checked one last time
Master README §17 has the Block 1-20 expectations. Every block must have content. Every saturation-class table, mandatory covariate row, specimen pathway, etc. must be filled. Empty placeholders in the published v0.1 mean the card is not v0.1-ready.

### CHK-5.4 Heath-facing card-finalize review
Before pushing card README + JSON: post the full README text in chat, walk through Block 1-20 with Heath, get explicit go-ahead before pushing.

### CHK-5.5 Commercial.web.py decision-tree section (operational deployment requirement)

Every card README must include a section titled `## Commercial.web.py decision tree — what to do when an IDAT fires this card` (or equivalent) that documents:

- The routing-pattern matrix (every possible Stage 1 + Stage 2 + Stage 3 pattern combination and where each routes)
- Per-arm patient-report templates (the actual language commercial.web.py outputs to the patient)
- Lineage-profile interpretation rules with concrete numerical examples (specific to cards using EpiDISH Stage 3 discrimination)
- The "no immediate culprit found" handling for long-window-pre-dx cases (no false-positive framing, trajectory-tracking framing, active-surveillance routing)
- Confirmatory test pathway by arm (what doctors typically order next when the card fires)
- What commercial.web.py CANNOT do at v1 (honest limitations the patient report acknowledges)
- Mandatory covariates that must be captured before scoring (the intake questionnaire requirements)

This section is the operational playbook for commercial.web.py running on Heath's server. Heme-epic v0.1 §"Commercial.web.py decision tree" is the reference template. Cards without this section are not v1-deployment-ready.

### CHK-5.6 Cohort-completeness final pass (CCL-029)

Before publishing v0.1, the card author confirms cohort-completeness per CCL-029:

- Every publicly-accessible cohort matching the card's disease/specimen/platform criteria has been at least surveyed (CHK-1.1 sample_title verified, CHK-1.2 platform compatibility checked, CHK-1.3 supplementary file format checked)
- Every accessible cohort has either been run or has documented justification for deferral (e.g., different platform requiring panel adaptation, cohort size insufficient for inference, specimen pathway not yet validated)
- Every gated cohort has been classified per CHK-1.6 access tier and listed in the v0.2+ priority queue with the corresponding application path

A v0.1 card that lists "we ran the one cohort we found" without exhausting the publicly-accessible landscape has not passed cohort-completeness. The cervical-epic v0.1 build ran six VAL studies and explicitly documented why the seventh (Bukowski CINCS, biobank-gated) was deferred — that is the cohort-completeness standard. Heme-epic v0.1 ran VAL-082 (the only publicly-accessible 450K AML cohort with healthy comparator) and explicitly catalogued seven other cohorts at gated tier with their corresponding access paths — also passes cohort-completeness.

### CHK-5.7 Card JSON `universal_reference` block — full 14-sub-key verification (added 2026-04-28)

Master README §17 Block 5 requires every card to "explicitly state" pipeline invariants. This CHK is the structural verification that converts that requirement into a per-sub-key checklist gate at card-publish time.

**Before publishing card JSON v0.1+, verify every one of the following 14 sub-keys exists in the card's `universal_reference` block, populated with non-placeholder content:**

1. `_purpose` — paragraph stating the full-inline self-contained reference design goal
2. `schema_version` — string "universal_reference_v1.0" (or current)
3. `last_updated` — ISO YYYY-MM-DD date
4. `universal_stage_1_pipeline` — 15 sub-keys (invariant_rule, specimen, platforms_supported, panel_id, panel_sha256, panel_n_cpgs, panel_source_paper, panel_source_doi, panel_source_doi_url, panel_coverage_epic, panel_coverage_450k, h_min_immune, h_min_source, scoring_method_primary, scoring_method_secondary_for_bidirectional_diseases)
5. `universal_h_min_table` — all 8 architecture classes with H_min from GAPE_WEB_v13.py lines 87-96 (cycling 0.856055, secretory 0.843264, immune 0.838889, terminal 0.772837, stromal 0.86295, stem_adult 0.873718, progenitor 0.852216, stem_pluri 0.982166)
6. `universal_stage_2_moss_deconvolution` — 5 sub-keys including the full `healthy_reference_beta_by_tissue` table (18 tissue β values from Moss 2018 Table S1)
7. `universal_stage_3_epidish_subcomposition` — 5 sub-keys including all 7 Salas QC bounds (neutrophil_fraction, lymphocyte_fraction, monocyte_fraction, cd4_fraction, cd8_fraction, nk_fraction, b_fraction)
8. `universal_80_cell_age_baseline_immune_class` — _purpose, _sources (Hannum, Horvath, Roadmap, Moss, Lister, Alisch), _critical_caveat_cross_cohort, age_decades (10 decades 00-09 through 90-99 with A_mean and A_sd each)
9. `universal_tier_thresholds` — _source, _note, all 6 tiers (BELOW_NORMAL, NORMAL, MARGINAL, DETECTABLE, URGENT, FLOOR_BREACH) each with A_threshold and action
10. `universal_sex_stratification_rule` — rule, minimum_reporting
11. `universal_language_discipline` — allowed phrases list, forbidden phrases list, rationale
12. `universal_cohort_batch_offset_warning` — _critical, _discovered_in (VAL-057), description, example, deployment_rule
13. `universal_no_fabrication_rule` — rule
14. `gape_web_version_reference` — canonical_file (GAPE_WEB_v13.py), h_min_constants_line_range (87-96), port (8080), frozen_at

**Failure mode this CHK is designed to catch.** Cardio-epic v0.1 (built 2026-04-28) shipped with only 8 thin sub-keys and was missing universal_80_cell_age_baseline_immune_class, universal_sex_stratification_rule, universal_h_min_table-for-all-8-classes, the full universal_tier_thresholds 6-tier vocabulary, and several others. The card was structurally thin (345 lines) vs breast-epic v2.3 (900 lines) and crc-epic v2.4 (791 lines). This CHK gate makes the structural-parity requirement explicit and per-sub-key verifiable rather than implicit in Block 5 prose. Reference template: breast-epic_card_v2_3.json `universal_reference` block (253 lines).

### CHK-5.8 Card JSON `atlases_used_and_deferred` block — every Queue-1 atlas accounted for (added 2026-04-28)

Heath signed off on run-everything architecture 2026-04-26. Run-everything means every IDAT runs every panel and every reference atlas. A card that defers an atlas at VAL-prereg time without re-surfacing the deferral as a card-level block produces a cookbook stating one thing (run everything) and shipping cards that do another (run a subset).

**Before publishing card JSON v0.1+, verify the card contains an `atlases_used_and_deferred` block with two arrays:**

`atlases_run` — every atlas that was actually scored on the card's cohorts during validation. Each entry: `atlas_name`, `atlas_version`, `atlas_sha256`, `n_cpgs_or_markers`, `vault_path`, `vals_run` (list of VAL IDs that scored this atlas).

`atlases_deferred` — every atlas listed in the current Queue-1 inventory that was NOT scored during the card's validation. Each entry: `atlas_name`, `defer_reason` (e.g., "Caggiano 2021 distributes WGBS-region format requiring 450K CpG bridge engineering not yet implemented"), `target_card_version` (e.g., "cardio-epic v0.3"), `unblock_dependency` (e.g., "HM450 hg19 manifest acquisition + region-CpG bridge engineering").

**Failure mode this CHK is designed to catch.** Cardio-epic v0.1 deferred EpiSCORE HeartRef and Caggiano CelFiE TIM at VAL-107 prereg time without re-surfacing as card-level deferral blocks. The card silently shipped with a subset of run-everything. This CHK makes the deferral promotion explicit and per-atlas accountable at card-publish time.

### CHK-5.9 Card JSON `substrate_roadmap` block — all 5 production substrates explicitly addressed (added 2026-04-28)

EDEAR's MESA framework specifies five production substrates per disease class: methylation (DNAm), nucleosome occupancy, fragment-length fuzziness, WPS (windowed protection score), fragment size. v0.1 cards typically ship with methylation only. The remaining four substrates are valid v0.2/v0.3 targets, but a v0.1 card without an explicit substrate roadmap leaves a customer or referee unable to determine which substrates are live, which are class-saturated and unusable for this disease, and which are next-version targets.

**Before publishing card JSON v0.1+, verify the card contains a `substrate_roadmap` block with five entries (one per MESA substrate):**

Each entry: `substrate` (DNAm | nucleosome_occupancy | fragment_fuzziness | WPS | fragment_size), `status` (validated | in_development | class_saturated_unusable | next_version_target | not_applicable_for_disease), `validation_anchor` (VAL ID list if validated; null otherwise), `target_card_version` (e.g., "v0.2" if next-version-target; null otherwise), `rationale` (short prose explaining the status — e.g., "validated via VAL-108/109/110 retrospective on three cardio cohorts" or "class-saturated unusable per Block 7 A_ceiling table").

**Failure mode this CHK is designed to catch.** Cardio-epic v0.1 shipped with no `substrate_roadmap` block at all, leaving the question "are nucleosome / fuzziness / WPS / fragment-size substrates planned for cardio?" unanswered in the card. Heath asked this directly in session 2026-04-28; the card had no answer.

### CHK-5.10 Card JSON `chk_3_1_thresholds_per_substrate` block — every supported substrate has BOTH 3.1A and 3.1B thresholds (added 2026-04-28, follows CCL-042 split convention)

CCL-042 formalized the CHK-3.1A (full-genome substrate gate) + CHK-3.1B (card-specific marker subset gate) split. Both must pass. Each substrate the card supports needs both thresholds documented in the card JSON.

**Before publishing card JSON v0.1+, verify the card contains a `chk_3_1_thresholds_per_substrate` block (within `universal_pipeline_acknowledgment`) with one entry per supported substrate. Each entry must contain:**

- `substrate_name` (e.g., "TCGA HM450K sesame Level 3 beta")
- `chk_3_1a` sub-block: `f_extreme_threshold`, `f_middle_threshold`, `n_valid_threshold`, `calibration_anchor_val_id` (the VAL that established this threshold), `calibration_anchor_cohort_n` (sample size of the calibration cohort)
- `chk_3_1b` sub-block: `card_specific_subset_sha256`, `card_specific_subset_n_cpgs`, `f_extreme_subset_threshold`, `f_middle_subset_threshold`, `n_subset_valid_threshold`, `calibration_anchor_val_id`, `calibration_anchor_cohort_n`
- `applies_to_vals` (list of VAL IDs that scored this substrate under the card)
- `notes` (e.g., "within-cohort self-cal at v0.1; pending generalizable structurally-separated calibration VAL")

**Failure mode this CHK is designed to catch.** Cardio-epic v0.1 had partial CHK-3.1A/B documentation only for TCGA HM450K sesame Level 3 (calibrated by VAL-106/107). GenomeStudio AVG_Beta HM450K (used by VAL-108 stroke + VAL-110 aortic) and minfi `preprocessFunnorm` HM450K (used by VAL-109 PAH) were within-cohort self-cal only and the card did not state that explicitly with the calibration-debt acknowledgment. Customers and referees seeing the card need to know which substrates have generalizable thresholds and which have within-cohort self-cal pending future calibration VALs.

---

### CHK-5.11 Atlas-family fitness check before sealing a new Stage 2 atlas integration (added 2026-04-29 from LL-CARDIO-005 / VAL-111)

LL-CARDIO-005 (cardio-epic v0.2) formalized that two distinct atlas-scoring modalities exist and they are NOT interchangeable: (a) tile-coverage A-score reading on heterogeneous β panels, which needs WGBS-derived tiles or equivalent CpG-coverage panels with cell-type-specific differential methylation (Loyfer 25-tile, Caggiano CelFiE TIM are this family); (b) EpiDISH proportion estimation on per-tissue β, which uses gene-promoter integer marker IDs against a reference panel matrix and returns cell-type fractions not A-scores (EpiSCORE family is this).

VAL-111 sealed the cookbook's first negative atlas-integration result by running EpiSCORE HeartRef in mode (a) when the atlas was designed for mode (b). All five cardiac tile A-scores read 0.46–0.51 across all three cohorts and all three substrates regardless of disease state; max within-cohort tissue discrimination 0.0152 vs the 0.10 threshold; blood-floor breach on 5/5 tiles. Atlas methodologically sound for its design purpose, did not transfer to A-score tile reading on heterogeneous β.

**Before sealing any new Stage 2 atlas integration, verify in the prereg that the atlas family matches the scoring modality the card uses:**

- The atlas has CpG-coverage panels (not gene-promoter integer marker IDs) for the cell types it claims to discriminate. If the atlas distributes integer marker IDs that require Illumina manifest mapping at scoring time, this is a yellow flag — verify whether the bridged CpG panels still carry cell-type-specific differential methylation on heterogeneous β.
- The atlas's intended scoring modality matches the card's Stage 2 reading mode. If the atlas is designed for proportion estimation (EpiDISH-style), it does not transfer to A-score tile reading without a re-bridging step.
- The prereg explicitly names the discrimination threshold (e.g., A-score range ≥ 0.10 within tissue cohorts, blood floor expectation A < 0.10 in negative-control cohort) so an O3_TISSUE_FLOOR_DOMINATED outcome is sealable.
- The card JSON's `atlases_used_and_deferred` block (CHK-5.8) must surface the atlas-family fitness assessment in the `deferral_rationale` if the integration is deferred.

**Failure mode this CHK is designed to catch.** Sealing an atlas integration as `atlases_run` without verifying the atlas family fits the card's Stage 2 scoring modality would propagate a methodologically-invalid atlas into production scoring. VAL-111 caught this for cardio-epic before any production claim could be made; the deferral is documented in cardio-epic v0.2 and the lesson generalizes to any future card considering an EpiSCORE-style or other proportion-estimation atlas.

---

### CHK-5.12 Atlas-canonical-source-check before sealing any new atlas integration prereg (added 2026-04-29 from DISC-CARDIO-007 / VAL-111 process lesson)

DISC-CARDIO-007 (cardio-epic v0.2.1) formalized a process lesson from VAL-111: the atlas tested in VAL-111 was selected because it sat in atlas_vault from a prior acquisition pass, not because it was the canonical-document-named cardio atlas. PIPELINE_REFERENCE_v2.md Part 2.4 explicitly names **Konigsberg 2023** — NOT EpiSCORE — as the cardio Stage 2 atlas blocker, with the deployment-of-record statement: *"Without this atlas, cardio-epic cannot be deployed."* Part 2.5 names Tanaka 2025 as "highest-priority new addition." Part 2.7 names Caggiano CelFiE for cardiac tissue.

**Before sealing any new atlas integration prereg, the prereg must cite which canonical-document section (PIPELINE_REFERENCE Part 2.X or README_MASTER §Stage 2.X) names the atlas as a production candidate for the card under test:**

- The prereg includes a `canonical_document_anchor` field naming the section (e.g., "PIPELINE_REFERENCE Part 2.4") and quoting the relevant document statement that justifies prioritizing this atlas for this card.
- If the atlas being tested is NOT named in the canonical documents for the card's domain, the prereg explicitly states this fact, names the reason for testing it ahead of the document-prescribed atlases (e.g., "EpiSCORE HeartRef was on disk while Konigsberg 2023 was not yet acquired; we test EpiSCORE first to characterize whether the gene-promoter atlas family transfers, with the understanding that Konigsberg remains the canonical critical path"), and queues the document-named atlas as the next acquisition / calibration target.
- The card's `atlases_used_and_deferred` block (CHK-5.8) updates `atlases_deferred` to enumerate ALL canonical-document-named atlases for the card's domain, not just the ones already in atlas_vault, with `target_version` and `unblock_dependency` per atlas.
- A `canonical_documents_named_blocker_for_X_deployment` block in the card JSON (or equivalent prose section in the README) cites the canonical-document quote that names the deployment-blocker atlas. If the document does not name a single deployment-blocker atlas, the field documents that and lists the candidates in priority order.

**Failure mode this CHK is designed to catch.** Building a cookbook critical path around "whatever sits in atlas_vault" rather than "what the canonical documents name as the deployment blocker" produces correct local outcomes (VAL-111 sealed honestly at O3_TISSUE_FLOOR_DOMINATED with the right lesson) but the wrong critical path for cardio v0.3 (which needs Konigsberg first, not EpiSCORE re-bridging). CHK-5.12 makes the canonical-document-anchor mandatory in every atlas integration prereg, so the prioritization is always traceable to the document-of-record.

---

### CHK-5.13 Documents-of-record citation-verification gate (added 2026-04-29 from CCL-046 / Cuadrat 2023 verification finding)

CCL-046 (cardio-epic v0.2.2) formalized that documents-of-record can themselves contain factual errors. The cookbook's PIPELINE_REFERENCE Part 2.4 was found to incorrectly attribute the cited DOI to a "Konigsberg 2023" cardiovascular atlas with sorted cardiomyocytes, when the actual paper at that DOI is **Cuadrat et al. 2023** with bulk ENCODE heart-tissue additions to the Moss 2018 base. CHK-5.12 (atlas-canonical-source-check) protects against picking the wrong atlas from atlas_vault but does not protect against following an incorrect citation in the canonical document. CHK-5.13 closes that gap.

**Before sealing a card publish or a card promotion (v0.X → v0.X+1), every external citation introduced in the new card content must have at least one web-verification pass:**

- Every DOI introduced in the new content loads at the publisher and resolves to a paper.
- Every author attribution matches the actual paper's author list (lead/last author at minimum; full list ideal).
- Every described atlas content (cell types, CpG counts, derivation method) matches what the actual paper's abstract / methods / figures / supplementary describe.
- Every cohort accession (GSE, EGA, dbGaP, etc.) resolves to an actual deposit and the description matches the prereg's claimed cohort scope (sample size, platform, disease groups).
- Every prior-art reference in deferral rationales (e.g., "Zemmour 2018 demonstrated cardiomyocyte cfDNA elevation in MI") resolves and the cited claim is actually in the cited paper.

**Verification record.** Each verified citation is logged either inline in the card JSON (`citation_verified: "2026-04-29 web-verified against doi.org/..."`) or in a per-card `citations_verification_log.md` companion file in the card workspace. Heath-only — not pushed to GitHub. The verification log is scope: only the citations introduced in the current card version need verification; previously-verified citations in inherited content do not need re-verification unless flagged.

**Recurring audit.** When a previously-verified atlas, panel, or external reference is integrated into production (atlas_vault acquisition, calibration VAL prereg, cardio-cohort scoring VAL prereg), the citation is **re-verified** at integration time. Citations age — paper retractions, errata, reanalysis updates, replacement DOIs — and the cookbook does not assume a once-verified citation stays accurate across years.

**Failure mode this CHK is designed to catch.** A canonical document containing a citation error (wrong author, wrong content, wrong DOI, conflated papers) propagates the error into every card that cites the canonical document. Without CHK-5.13, the error surfaces only on attempted acquisition — by which point a v0.X card has already shipped naming an atlas that does not exist as described. The Part 2.4 Konigsberg/Cuadrat conflation propagated through cardio-epic v0.1 + v0.2 + v0.2.1 undetected for three card versions before the Phase A acquisition attempt surfaced it. CHK-5.13 catches this class of error at card-publish time, before it propagates.

**Generalization.** CCL-046 / CHK-5.13 applies to every external reference in cookbook documents, not just atlas references: cohort accessions, cited validation studies, H_min derivations referencing external papers, panel construction methods. Wherever the cookbook says "per X et al. Y" the X-Y pair must be web-verified at least once and re-verified when re-cited. The gate is cheap (one web search per citation) and catches an entire class of errors that compound silently over time.

---

---

## STAGE 6 — GitHub push protocol (per memory #14, ABSOLUTE)

### CHK-6.1 Never push to GitHub:
- Card README (`*-epic_README.md`)
- Card JSON (`*-epic_card_v0.1.json`)
- Master README v2.1 cookbook (`README_MASTER_v2.1.md`)
- LESSONS_LEARNED.md
- TESTING_CHECKLIST.md (this file)
- Per-card directional panel JSON

### CHK-6.2 DO push to GitHub:
- VAL-XXX prereg.md
- VAL-XXX PREREG_SEAL.txt
- VAL-XXX results.json
- VAL-XXX outcome.md
- VAL-XXX cohort manifest JSON
- VAL-XXX clinical metadata
- VAL-XXX Python script
- Updated `Biological_Physics/README.md` (with new VAL entries)
- Updated `Evidence_Report.html`

### CHK-6.3 Surgical edits only
Never reorganize, refactor, or "improve" pushed files unless Heath explicitly asks. Edit the specific lines being changed, leave the rest alone.

---

## STAGE 7 — End-of-session protocol

### CHK-7.1 Update `LESSONS_LEARNED.md` with new lessons from the session
Per-card lesson IDs (e.g., `cerv-LL-008` through `cerv-LL-016` from this session) get appended to the per-card lessons file AND the master `LESSONS_LEARNED.md`.

### CHK-7.2 Update this `TESTING_CHECKLIST.md` if new patterns emerged
If the session surfaced a new mistake pattern not covered by an existing CHK item, add a CHK item. Do not just write a lesson and hope to remember; encode the lesson as a checklist item that future sessions will be required to read.

### CHK-7.3 Generate session summary
Per master README protocol — papers/cards completed, decisions made, open problems, pending items, key statements, strategic notes.

### CHK-7.4 Memory edits
If the session produced new absolute rules (like CCL-030/031), update memory via the memory_user_edits tool. Memory edits are the only thing that survives compaction. Treat them as the most important deliverable from any session.

### CHK-7.5 Mark checklist-worked-as-intended moments

When the checklist catches something that would otherwise have produced a halt-and-revert cycle, **note it explicitly in the lessons learned**. The checklist exists to compound: failure modes get encoded as CHK items; successful applications of those CHK items get encoded as positive examples that future sessions can reference.

VAL-082 is the reference positive example: the moment d = +3.71 came back, CHK-4.1 (biology consistency) and CHK-1.5 (substrate-scope) both fired correctly — caught the apparent A_AML=0.54 vs Issue 002 A_AML=1.10 mismatch, resolved it as substrate-scope-translation rather than framework error, drafted the outcome correctly the first time. **No halt, no revert, no Heath intervention required.** That is the checklist working as designed. Future builds should aim for this pattern.

---

## How this checklist gets used

**Heath:** at the start of any session involving a new card or a new VAL, ask Walther: "have you read the testing checklist?" If the answer is no, halt. If the answer is yes, ask "what's the first stage applicable to today's task?" If the answer doesn't match, halt. The checklist is structural; treat it as such.

**Walther:** at the start of any session involving a new card or a new VAL, the FIRST tool call is `view` on this file. Not project_knowledge_search, not bash, not anything else. View this file. Then proceed through the applicable stages.

**Both:** the checklist evolves. New mistake patterns add new CHK items. Old CHK items don't get removed unless the underlying mistake pattern is structurally impossible to recur. The point is that the checklist accumulates the wisdom of every previous build so the next build doesn't repeat the same mistakes.

---

## The core principle

Heath's framework: **biology common sense IS a check on the math**. The framework's universal Stage 1 panel can produce numbers, but the numbers are not biology until the data integrity is validated AND the result is consistent with the established disease biology. If the result contradicts the published clinical-grade panels for the disease, the framework's panel didn't transfer — that is a transferability finding, not a "the disease has no signal" finding.

Walther's failure mode: treating numbers as biology before checking whether the data is interpretable as biology. This failure mode wasted ~4 hours on cervical-epic VAL-076/077. The testing checklist exists to make this failure mode structurally impossible going forward.

### CHK-7.6 Reproducibility triple — every published VAL must include source + inputs + environment

**Source code** is one of three things a reviewer needs to reproduce a result. The other two are the **input data** and the **runtime environment**. Embedding the Python script alone is necessary but not sufficient. Every VAL block in the Evidence Report (and every VAL outcome.md in the card folder) must include all three:

1. **Source code** — the Python script(s), embedded inline as `<pre id="..."><code>` (HTML-escaped, full text), so a reviewer can read every line of analysis without leaving the document.
2. **Inputs** — for every input file the script reads, an explicit download URL (GEO FTP / GitHub raw / Zenodo DOI / Mendeley dataset ID) plus approximate file size and SHA-256 where applicable. Inputs include: GEO series matrices, supplementary processed matrices, panel JSON files, manifest files, and any cohort metadata files. If the input is biobank-gated, say so explicitly with the application path (dbGaP accession, EGA dataset, biobank PI contact).
3. **Environment** — Python version, package versions for any non-standard imports (numpy, pandas, scipy, matplotlib), expected runtime, expected memory footprint. If the script uses Python standard library only (which most VAL scripts do), say so explicitly.

**A reviewer should be able to:** open the Evidence Report → see the result + figure inline → expand the source code → click each input download link → install the listed Python environment → run the script → get the same headline numbers back.

**Each VAL block must also include the expected headline output** as a "Expected output" line so a reviewer knows whether their re-run matched (e.g., "Expected: Cohen's d = +1.96 [+1.62, +2.31], all glioma mean = 1.092%").

This rule applies retroactively: when revisiting any past VAL block, if the inputs/environment/expected-output are missing, add them. The Evidence Report is the canonical reproducibility document. Reproducibility means the document is self-contained — anyone can open it and walk all three steps without external assumptions.

**Per-card README must mirror this structure** in the "Files in this card" section: every script gets a manifest entry that lists what data it reads, where to get it, what environment runs it, and what numbers it produced.

### CHK-7.7 Don't defer integrations to "v0.2 future task" without a real reason (glioma-LL-005)

When Walther considers writing "this is a v0.2 future task" or "deferred to a future version" or "out of scope for this version" in any card README, the trigger fires this checklist:

1. Is the deferred task **actually a defined task** with a name, a method, and an input source? If not, it's not a deferred task — it's vague language. Either name it concretely or remove it from the README.
2. Is the input data for the task **actually available**? If yes, name the URL, the SHA, the size. If the answer is "yes, available," then default to running the task NOW and reporting results, not deferring it.
3. Is the **method** for the task published, peer-reviewed, and code-available? If yes, run it.
4. Is there a **real obstacle** that prevents running the task right now? Examples of real obstacles: requires platform mismatch resolution (different chemistry: WGBS vs array), requires biobank-gated data application, requires custom collaboration that hasn't been initiated, requires substantial new code that depends on a class we haven't refactored. Examples of NOT-real obstacles: "it would take time," "we don't have the bandwidth," "scope creep," "let's focus on v0.1 first."

**If steps 1–3 are yes and step 4 is no, the task is not deferred. Run it. Report results.**

**Time-cost calibration heuristic:** most published methylation analyses where the reference is open-source and the cohort is GEO-deposited are 1–4 hours of work end-to-end (extract β CSV, run NNLS or trained classifier, compute summary stats, generate figure, write outcome). The complexity in cookbook work comes from interpretation, not from running the analysis. **Run first, interpret second.**

**The pattern that proves the rule:** VAL-090 was deferred to "a v0.2 future task with a 3-month timeline" in the v0.1 README. The actual integration took 4 hours and produced d = +1.96, the second-strongest single-cohort effect in the cookbook to date. The "3 months" estimate was wrong by a factor of 1500.



### CHK-4.11 Run-everything 25-tile prereg-O1-criterion design under CCL-039 (added 2026-04-28)

**The rule.** Future preregs that include run-everything 25-tile per-class A-score on tumor-vs-adjacent-normal paired comparisons must NOT pre-lock "cell-of-origin tile is largest |d|" or "cell-of-origin tile shows positive d" as an O1 criterion. Pre-lock "cell-of-origin tile is among the largest |d|" instead, with explicit acknowledgment that direction depends on the comparison type.

**Why this rule exists (CCL-039 evidence — three independent cohort configurations).** VAL-098 was the first cookbook validation to run BOTH full-HM450 cycling-class methodology AND run-everything 25-tile per-class methodology on the same paired tumor/normal samples. Found: full-HM450 cycling-class paired d positive (TCGA-READ +0.612) AND Colon_epithelial_cells tile paired d strongly negative (TCGA-READ −2.501). Diagnostic re-application of VAL-098 methodology to the existing VAL-062 TCGA-COAD 26-pair sealed dataset confirmed the pattern: full-HM450 d = +0.724, Colon_epithelial_cells tile d = −1.552. VAL-099 (2026-04-28) re-executed VAL-062 cycling-class methodology on the same TCGA-COAD 26-pair cohort with run-everything 25-tile output and reproduced the pattern at the third independent measurement: full-HM450 d = +0.7241 [+0.352, +1.296], Colon_epithelial_cells tile d = −1.603 [−2.173, −1.288]. Three independent paired-tumor-vs-adjacent-normal cohort configurations, three negative cell-of-origin tile readings, three positive full-HM450 cycling-class readings. Two distinct observables. They measure different things.

**The two comparison types.**

1. **Tumor-vs-adjacent-normal-paired** (e.g., TCGA-READ paired pairs, TCGA-COAD paired pairs):
   - Cell-of-origin tile expected NEGATIVE direction.
   - Mechanism: tumor de-differentiation degrades the cell-of-origin tile fidelity; the tumor sample looks LESS like healthy colon than the adjacent-normal sample at the colon-discriminating CpGs.
   - Full-HM450 cycling-class A-score still positive (global Shannon entropy increases in tumor).
   - Other tissue tiles (Bladder, Hepatocytes, Pancreatic_beta) read positive — their marker CpGs drift toward homogenized tumor methylation away from healthy-colon-specific values.

2. **Diseased-tissue-vs-healthy-cross-reference** (e.g., disease cohort vs healthy cohort with tissue-of-origin lookup):
   - Cell-of-origin tile expected POSITIVE direction.
   - Mechanism: the diseased sample contains cells of that tissue type, which read above healthy-reference baseline because the healthy reference does not contain that tissue at meaningful fraction.

**Prereg O1 criterion language under CHK-4.11.**

- Acceptable: "Cell-of-origin tile is among the top 5 largest |d| tiles in the run-everything 25-tile output."
- Acceptable: "Cell-of-origin tile shows |d| ≥ 0.5 with direction consistent with the comparison type (negative for tumor-vs-adjacent-normal-paired; positive for diseased-tissue-vs-healthy-cross-reference)."
- NOT acceptable: "Cell-of-origin tile shows positive d" (without specifying the comparison type).
- NOT acceptable: "Cell-of-origin tile is largest |d|" (without acknowledging that other tiles can have larger |d| under the homogenization mechanism).

**Outcome write-up rule.** When a VAL runs both full-HM450 cycling/architectural-drift methodology AND run-everything 25-tile per-class methodology on the same samples, the outcome.md must report BOTH numbers with the biology interpretation. The two metrics measure different observables. They are not contradictory when they move in different directions — full-HM450 measures global entropy, per-tile marker CpG measures cell-of-origin tile fidelity. Pre-locked outcome label applies to the full-HM450 result (the standard methodology); the per-tile observation is supplementary documentation that surfaces or confirms CCL-039.

**Cookbook-wide retroactive task (future-when-time-permits).** Apply the run-everything 25-tile methodology to the existing per-sample CSVs for VAL-060 (TCGA-BRCA breast), VAL-063 (TCGA-LUAD lung), VAL-064 (TCGA-LIHC liver), VAL-058 (GSE269244 prostate), and verify the cell-of-origin tile direction is consistently negative in tumor-vs-adjacent-normal paired comparisons across cancer types. CCL-039 is currently confirmed on three colorectal cohort configurations (TCGA-READ VAL-098, TCGA-COAD VAL-062 revisit, TCGA-COAD VAL-099 reproduction); cross-tissue confirmation upgrades it from a robustly-confirmed colorectal observation to a framework-level rule. The retroactive expansion is a future-when-time-permits task; it does not block current per-card publication.
