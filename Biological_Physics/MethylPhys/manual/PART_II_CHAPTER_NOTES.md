# Part II — The Chain, Stage by Stage (working notes for [[Issue 003](MethylPhys_CPG_Operations_Manual.pdf)](MethylPhys_CPG_Operations_Manual.pdf) / 004)

Purpose: one chapter per stage, written for a researcher who knows bootstrapping but not MCMC and has never met a
Mahalanobis hull. Each chapter: purpose · the cosmology it borrows · what was tried first and why it failed · the runtime
files it reads · the empirical confirmation (PROC) · what to know before touching it. Written after the chain is sealed;
these notes collect the material as it is learned so nothing is lost.

## Three kinds of file — the distinction every chapter leans on
- **FLOOR**: `H_min` (40 values, 8 classes × 5 substrates). Physics. One number. Frozen 2026-04-06. Never a cohort.
- **RULER**: [`iamatlas_gauge_identity_loci_v1_0.json`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json) — per class, the CpGs where the healthy class sits within ±0.05 of
  `H_min_β`; the gauge reads β̄ here. Derived from the Atlas, frozen. Counts: terminal 57,247 … stromal 2,294.
- **BAND**: [`age_reference_matrix.json`](../chain/Runtime%20Matrices/A_Scoring_Module/age_reference_matrix.json) — per class × 10 age bins, healthy A and β (mean, sd, p10–p90) with n and source.
  A cohort statistic compiled 2026-05-28 from nine papers; provenance is uneven (immune n=102/bin Hannum; stem_adult n=28 Adelman).
  Every "healthy reads wrong" case of 2026-09-19 traced to a BAND, never a FLOOR (RECON B1). Phase 1 rebuilds all eight bands
  from one cohort (GSE87571) through one pipeline (Stage 1).

## Stage 0 — intake. The gate that decides whether a specimen is measured at all: arrival, manifest, integrity hash, control probes, negative controls, bead counts, call rate, platform coverage, sex from chrX/chrY, and the decision. Two lessons worth the chapter: a gate that cannot read a gzipped file never fires, and a gate whose failure is caught and logged as deferred is worse than no gate at all.
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
   directional composite recovered d=+0.62. [`bidirectional_decomposition.py`](../chain/Runtime%20Matrices/Directional%20Panel/bidirectional_decomposition.py) runs that sealed formula per patient.
## Stage 4.6 — patient CMB. [`cpg_patient_cmb.py`](../chain/cpg_patient_cmb.py); the four-skies plate. Assessability by median|z| wrongly admits absent classes — the
   deconvolver presence gate is the right gate.
## Stage 5 — Mahalanobis Option A. Eight class-gauge A's against the band, n-adaptive χ². Defect: `run_full` emits {distance, beyond}
   while the report reads mahalanobis_distance — key mismatch (PROC-CHAIN-01). Distance on healthy donors driven by the stem_adult band.
## Stage 6 — cellular age. [`iam_cellular_age_scoring.py`](../chain/Runtime%20Matrices/Cellular_Age/iam_cellular_age_scoring.py) v3: invert the band's β_mean(age) curve per class. v1 was a trained clock
   (wrong), v2 inverted the wrong formula (Jensen). Reads 4 yr for adults today — the band curves are too flat/thin to invert. Not reportable.
## Stage 7 — tiers. NORMAL 0.95–1.01, ELEVATED 1.01–1.07, SIGNIFICANTLY_ELEVATED 1.07–1.10, BREACH ≥1.10 (onset moved 1.04→1.01, 66f37fe).
## The Atlas — several chapters: build (per-class MCMC, 115 cells, 262 columns), the flatness lesson, the brightness posterior
   (mean/sd per CpG = a reference map with per-pixel uncertainty), HEALPix and why a sky.

## Opening chapter — the translation map. [`CMB_TO_METHYLOME_MAP.md`](../doors/CMB_TO_METHYLOME_MAP.md): 79 rows written before the build; scored. Two reversals are the finding (row 1, the 115-cell basis as the harmonic basis, is the frame):
   row 20 (second deconvolver: NILC built then cut — the one Planck principle the chain knowingly does not follow) and row 47 (de-aging built then refused, §104 —
   the methylome's foreground is the patient). Row 44 (aging as lensing) is the unwritten explanation of the age band.

## Part II page — what we built before the bones were trusted. [`COMPLETION_SPRINT_scored.md`](../doors/COMPLETION_SPRINT_scored.md): Phase A (nulls) held; B built then torn out (NILC cut, de-aging refused); C–E never started. Order should have been A → trust the bones → C/E. Banana degeneracy (C3), C(d) (C1), per-card likelihood (E2/E3) are the correct next layer AFTER Phase 1 rebuilds the bands.

## Stage 4 addendum — the lesson N7 taught (2026-09-19). The production gauge read the marker union, not the identity loci, for eleven weeks; real blood never showed it because the band was compiled the same way; a synthetic healthy patient showed it in one run. Two morals for the chapter: (1) a statistic that agrees with its own band is not thereby measuring anything; (2) end-to-end simulation is not optional - it is the only test that knows the truth. Also the Jensen gap as a diagnostic: 0.026 on identity loci, 0.257 on the marker union.

## Stage 2 addendum — the marker-pool lesson, told three times. Gauge (beta_mean on the marker union, PROC-N7-01), NILC (GLS basis on the marker pool, RECON D3), the 2026-06-11 all-BREACH bug: one surface, one rule - a cell-type marker panel answers 'which cell is this' and nothing else. Walther works because it built its own class-level ruler. For the chapter: show the three failure modes side by side, then the Jensen gap and the condition number as the two numbers that would have predicted them.

## Stage 2 addendum 2 — NILC was right (PROC-NILC-01). kappa(blood)=30.6, r(prog,stem_adult)=+0.99: the Atlas cannot split them in blood; Walther's stem_adult is constraint-chosen; the false BREACH of PROC-CHAIN-01 follows. Chapter moral: a second method's disagreement is the diagnostic, not the defect - Planck compared four methods to find WHERE the sky was uncertain, not to vote. Fix: joint haematopoietic-progenitor component for whole blood.

## Part II opening for the Stage 2 / Stage 4 chapters — 'the safeguards did their job and were switched off for it'. NILC (June) and the synthetic generator (retired unused) both surfaced the same class of problem; both were read as broken because the chain had been trusted longer than they had existed. Planck's answer: methods that cannot be individually disabled; sims that run on every release. RUNBOOK s11.

## Stage 2 addendum 3 - PROC-SEP-01. The HSC/progenitor split IS in the Atlas (1,290 CpGs) and the deconvolver uses 129 of them: field-ranked markers answer 'haematopoietic or not', not 'HSC or MPP'. Class boundary drawn one step down a lineage (stem_adult = 1 cell type, progenitor = 11). Remedy ladder: pairwise-forced markers -> coarse-to-fine composition -> independent HSC reference. Chapter point: 'more CpGs' is rarely the answer; the RIGHT CpGs for the contrast that is ill-conditioned is.


## Stage 2 addendum 4 - PROC-SEP-03, the two-tool design. Tool A: what is in the tube. Tool B: how one compartment divides, on the contrast CpGs only, with its own kappa check. kappa 30.6 -> 5.5; stem_adult = 0 in 7/7 blood; COMPARTMENT_ONLY where there is nothing to split. Chapter point (the author's question): 'can we refine the tools into two tools with each their own speciality?' - yes, and it is Planck's multi-scale habit. Three routes (NILC, contrast Walther, splitter) now agree stem_adult ~0 in healthy blood.


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Record/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

## Part II opening image — the stadium (author, 2026-09-20; Kelvin correction by analyst)

Walk into a football stadium with a temperature gun, point it at one person's forehead, read 102.0 °F. The instrument was calibrated before you walked in; the number means the same thing in any stadium, on any day. That is the single-patient CPG read: one draw, one array, one pass through the chain, an absolute A against a fixed floor.

The field of methylation research walks into the same stadium and feels foreheads with the back of a hand — grouping people by who feels normal and who feels warmer *than the others present*. The result depends on the crowd, the season, the ambient temperature in the building; it has no unit that survives leaving. That is a cohort comparison, and it is why batch effects are the field's chronic disease.

**The correction that makes the image fair:** the field does have thermometers — β is a real fraction, a clock returns years. What it lacks is a **fixed zero**. Every study re-etches its "normal" mark at the mean of the crowd in the room: Celsius-of-this-crowd. IAM's claim is a **Kelvin** claim — H_min is a zero set by physics that does not move between stadiums. Not "they have no instrument"; "their instrument has no absolute zero, and ours does."

Two honesties travel with the image. We still walked into a crowd of *healthy* people to learn where normal reads on our gun (the band) and to calibrate the gun against a known standard (the scale map) — those are the reference range and the instrument calibration every clinical measurement carries. And whether H_min is a true physical zero or a very good frozen constant is the question peer review will press; the empirical claim — one frozen reference, absolute readings, transfer across labs (Phase 1c) — stands either way.

Author's rule for future readers and future AIs: *not "no cohorts" — "no cohorts where the physics belongs."* Floor and score: physics. Reference range: healthy people only, never with a disease label in the room.

## Prior art as the door (author, 2026-09-20)
Sanchez & Mackenzie (PLoS ONE 2016; IJMS 2019; MethylIT) put Landauer under methylation a decade ago, peer-reviewed. Enter the conversation on what is agreed - the bound, the thermal background, information divergences, the clinic gap - then carry it forward: they had no fixed zero (centroid of controls), no single-sample reading, no composition step, no calibration toolkit. "One filters, one calibrates." Cite, do not credit: the author arrived from cosmology without them and first read the paper 2026-09-20. Their thermal-background filter is complementary and is a Future Goal on the identity loci.

## Title adopted 2026-09-20: *Physics of Methylation: Landauer Metrology*
The field is Landauer metrology; the methylome is its third application after silicon (SCAPE) and qubits (QAPE). Part II chapter added: **'Landauer Metrology and the Physics of the Biological Write-Head'** — the cell as a write-head that writes and holds a state against thermal noise; the Mahaffey number as energy per bit above the noise floor (M1 ~117, cell 20.94, transmon 1); why the cell cannot cool and the qubit must.
