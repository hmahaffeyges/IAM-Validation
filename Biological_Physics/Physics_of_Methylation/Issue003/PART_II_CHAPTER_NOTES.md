# Part II — The Chain, Stage by Stage (working notes for Issue 003 / 004)

Purpose: one chapter per stage, written for a researcher who knows bootstrapping but not MCMC and has never met a
Mahalanobis hull. Each chapter: purpose · the cosmology it borrows · what was tried first and why it failed · the runtime
files it reads · the empirical confirmation (PROC) · what to know before touching it. Written after the chain is sealed;
these notes collect the material as it is learned so nothing is lost.

## Three kinds of file — the distinction every chapter leans on
- **FLOOR**: `H_min` (40 values, 8 classes × 5 substrates). Physics. One number. Frozen 2026-04-06. Never a cohort.
- **RULER**: `iamatlas_gauge_identity_loci_v1_0.json` — per class, the CpGs where the healthy class sits within ±0.05 of
  `H_min_β`; the gauge reads β̄ here. Derived from the Atlas, frozen. Counts: terminal 57,247 … stromal 2,294.
- **BAND**: `age_reference_matrix.json` — per class × 10 age bins, healthy A and β (mean, sd, p10–p90) with n and source.
  A cohort statistic compiled 2026-05-28 from nine papers; provenance is uneven (immune n=102/bin Hannum; stem_adult n=28 Adelman).
  Every "healthy reads wrong" case of 2026-09-19 traced to a BAND, never a FLOOR (RECON B1). Phase 1 rebuilds all eight bands
  from one cohort (GSE87571) through one pipeline (Stage 1).

## Stage 0 — intake. Lesson 2026-09-19: the decision gate did not read Step 0.1's quarantine status (fail-open); fixed.
   Array type is verified from the IDAT header. Intensity QC (detection-p, call rate, sex) awaits the Stage 0↔1 hand-off.
## Stage 1 — calibration. PROC-CAL-01: 11/11 bit-identical to the cache. methylprep noob; needs pandas < 2. The +0.05–0.09
   identity-loci β offset vs the Atlas is a normalization gain (6–7 % multiplicative, zero at β=0, max at 0.7–0.8) — absorbed by the band.
## Stage 2 — Walther deconvolver. PROC-DECON-01 MAE 0.0004. NILC cut 2026-07-02 (collapsed on correlated blood mixtures). Gates nothing.
   Presence: DETECT_FLOOR 0.01 (conductor) vs 3 % (adjudicator) — two floors, RECON D2.
## Stage 3 — foreground. Built (age/sex/smoking layers, `RETIRED/…/IAM_Cellular_Age/`), deliberately NOT wired (SOP §104):
   a galactic foreground is a separate source; the methylome "foreground" is the patient's own biology — annotate, don't subtract.
## Stage 4 — the gauge. `A = H(β̄)/H_min` on identity loci, placed in the band. Two surfaces (§106): gauge vs separation.
   Lesson: four formula/loci combinations shipped in three weeks (2026-06-11 … 07-01); settled by measurement (PROC-FORMULA-01, PROC-ANCHOR-01).
## Stage 4.5 — bidirectional. H is symmetric about 0.5, so up-and-down disease patterns cancel in β̄ (VAL-050 d=+0.08); VAL-051's
   directional composite recovered d=+0.62. `bidirectional_decomposition.py` runs that sealed formula per patient.
## Stage 4.6 — patient CMB. `cpg_patient_cmb.py`; the four-skies plate. Assessability by median|z| wrongly admits absent classes — the
   deconvolver presence gate is the right gate.
## Stage 5 — Mahalanobis Option A. Eight class-gauge A's against the band, n-adaptive χ². Defect: `run_full` emits {distance, beyond}
   while the report reads mahalanobis_distance — key mismatch (PROC-CHAIN-01). Distance on healthy donors driven by the stem_adult band.
## Stage 6 — cellular age. `iam_cellular_age_scoring.py` v3: invert the band's β_mean(age) curve per class. v1 was a trained clock
   (wrong), v2 inverted the wrong formula (Jensen). Reads 4 yr for adults today — the band curves are too flat/thin to invert. Not reportable.
## Stage 7 — tiers. NORMAL 0.95–1.01, ELEVATED 1.01–1.07, SIGNIFICANTLY_ELEVATED 1.07–1.10, BREACH ≥1.10 (onset moved 1.04→1.01, 66f37fe).
## The Atlas — several chapters: build (per-class MCMC, 115 cells, 262 columns), the flatness lesson, the brightness posterior
   (mean/sd per CpG = a reference map with per-pixel uncertainty), HEALPix and why a sky.

## Opening chapter — the translation map. `CMB_TO_METHYLOME_MAP.md`: 78 rows written before the build; scored. Two reversals are the finding:
   row 20 (second deconvolver: NILC built then cut — the one Planck principle the chain knowingly does not follow) and row 47 (de-aging built then refused, §104 —
   the methylome's foreground is the patient). Row 44 (aging as lensing) is the unwritten explanation of the age band.
