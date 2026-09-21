# CPG / GAPE Issue 003 — Reproduction Kit

**See also `COMPONENT_MAP.md`** — what lives in the repo, what lives in the kit, what lives only in the author's folder, and the repo commits this kit implies.

**Purpose.** Everything used to produce GAPE Issue 003 and to verify the measurement core of the CPG chain,
in one directory, so that any person or any AI can run it without being taught the project first.
Every script prints its own input / operation / expected / observed / verdict block.

**Provenance.** Engine and runtime files are from `github.com/hmahaffeyges/IAM-Validation` at commit
`66f37fe3aa7a5302d77cbe418aba15fe4a11b471` (2026-07-03), except where RULING M1b (below) adopts the author's
corrected copy. `CHECKSUMS.sha256` covers every file in this kit; verify with `sha256sum -c CHECKSUMS.sha256`.

---

## 0. Read this first — the four rules that took a day to learn

1. **The physics measures; cohorts only point.** Never compute a group statistic (Cohen's d, AUC, Mann-Whitney)
   as the primary result. Each sample is read as an absolute A against the fixed reference (H_min, the age band).
   Cohorts enter only to establish the *direction* a disease moves a class.
2. **Two surfaces, two aggregations, fixed by their references (RULING A3).**
   - Class **gauge** (8 classes, identity loci): `A = H(mean β) / H_min(class)`. H_min is itself H of a mean β; the age band is compiled this way.
   - **Separation** (115 cell types, discriminative markers): `A = mean_i H(β_i) / H_min(class)`. Markers are bimodal; averaging β first is wrong here (SOP v1.4.0 §105). The sealed anchors are this statistic.
   - Never read a statistic against a band compiled the other way.
3. **Presence before score.** A class absent from the substrate (deconvolved fraction < `DETECT_FLOOR` = 0.01) is not scored.
   Its identity loci in a sample that contains none of its DNA read as a uniform offset, not as a finding.
   The gauge in `cpg_kit.gauge_A` refuses absent classes and bimodal panels.
4. **Substrate decides what can be read.** Whole blood carries immune architecture only (epithelial fraction ≈ 0 in
   healthy donors, by biology). Plasma cfDNA carries shed tissue. Bulk tissue is a mixture and inflates `H(mean β)`.
   Every H_min is per (class, substrate): the 40-cell table is in `engine/cpg_gauge_engine.py::H_MIN_TABLE`.

---

## 1. Environments

| env | purpose | pins |
|---|---|---|
| `python` (default) | everything except Stage 1 | numpy ≥ 1.26, pandas ≥ 2, scipy, reportlab (for the Issue build), pypdfium2 (page checks) |
| `methylprep` | Stage 1 only (`PROC_CAL_01.py`) | **python 3.11, methylprep==1.7.1, numpy==1.26.4, pandas==1.5.3, pytz, python-dateutil** — methylprep calls `DataFrame.append`, removed in pandas 2 |

Stage 1 also needs: `HOME` pointing at a writable directory (methylprep writes `$HOME/.methylprep_manifest_files/`), and
network access to `https://array-manifest-files.s3.amazonaws.com/` for the Illumina manifests on first use
(`HumanMethylation450k_15017482_v3.csv.gz`, `HumanMethylationEPIC_manifest_v2.csv.gz`). Offline: place both files in that directory.

---

## 2. Large inputs (not shipped) — put them in `data/` or point `CPG_KIT_DATA` at them

| file | source | notes |
|---|---|---|
| `IAMAtlasREBUILD.csv` | repo `Biological_Physics/IAM_Atlas/IAMAtlasREBUILD.csv.xz` → `xz -d` | 605 MB, 483,092 rows, build 2026-05-28 |
| `betas_cache.pkl` | `10_TEST_DATA.zip` (author) | 140 MB; Stage-1 output for the 11 test samples |
| `idats/*_Grn.idat.gz, *_Red.idat.gz` | `10_TEST_DATA.zip` | 11 pairs: 7 × 450K whole blood, 4 × EPIC colorectal tissue |
| `GSE51032_series_matrix.txt.gz` | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE51nnn/GSE51032/matrix/` | 3.0 GB |
| `GSE51057_series_matrix.txt.gz` | same path, GSE51057 | 1.2 GB |
| `GSE122126-GPL21145_series_matrix.txt.gz` | `.../GSE122nnn/GSE122126/matrix/` | plasma cfDNA + in-vitro mixes (Moss 2018) |

---

## 3. Procedures — run in this order; each is independent

| script | env | what it proves | expected | observed 2026-09-19 |
|---|---|---|---|---|
| `PROC_CAL_01.py` | methylprep | raw IDAT → β reproduces the project's Stage-1 cache | bit-identical | **11/11, r=1.000000, max diff 0.000000** |
| `PROC_DECON_01.py` | python | deconvolver reproduces `TEST_DATA_MANIFEST.md`; whole blood reads epithelial ≈ 0; gauge read with presence gate and age band | MAE ≤ 0.001; epi < 0.02 | **MAE 0.0004 / 0.0002 / 0.0002; WB epi 0.000–0.011** |
| `PROC_ANCHOR_01.py` | python | sealed 115-cell anchors reproduce from raw GEO | r ≥ 0.9999 | **GSE51032 r=1.00000 (112/115); GSE51057 r=1.00000 (115/115)** |
| `PROC_FORMULA_01.py` | python | the measurement behind RULING A3 | see docstring | WB immune Spearman +1.000, offset +0.029; tissue +0.24; band = H(mean β) 80/80 |
| `PROC_PLASMA_MIX_01.py` | python | deconvolver vs real known mixtures (Moss Table 6) | per-tissue r ≥ 0.9 | **terminal PASS (r=0.945); secretory FAIL (0.19); cycling FAIL (0.10)** |

Results land in `results/PROC_*.json`. `results/VAL_INDEX.{csv,json}` is the mechanical index of all 175 validation records (G, VAL-001..128, T1..T15, CPG-VAL-001..022, hull, N7, September PROCs; unique keys by series) in the repo (Issue 003 Appendix V).

`PROC_SKIES_01` (not a script yet): Issue 003 Fig. 5A-1 'Four skies' was produced by the repo's own `cpg_patient_cmb.py` on GSM1051533 + `IAMAtlasREBUILD.csv` immune_mean/immune_sd + CAMB Planck-2018 → `healpy.synfast`; needs `healpy`, `camb`, and `HOME`/`XDG_CONFIG_HOME` pointed at a writable dir (astropy config). A PROC that prints FAIL is a finding, not an error — record it.

---

## 4. Rulings recorded in this kit (Issue 003 §1.5)

- **A3** — one aggregation per surface (rule 2 above). `SOP v1.4.0 §105` is amended to scope its "never H(β_mean)" to marker panels.
- **M1b** — `runtime/iamatlas_celltype_markers_v0_2.json` is the **chrX-removed** file (131 chrX markers dropped 2026-06-11 for
  derived sex-invariance). The repo HEAD copy is kept as `..._REPO_HEAD_prechrX.json` because the 2026-05-29 seal (`anchors_v1/`)
  was made with it. `anchors_v2/` are the re-sealed values under the canonical file: 32/115 cells shift, max 0.059 (Mela),
  r = 0.9996 to v1. **Author action:** commit the chrX-removed file to the repo; mark `anchors_v1` SUPERSEDED.

---

## 5. Building Issue 003

```
cd issue003_build
CPG_TRIAL=../runtime python build_gape_issue003.py IAMPerformance_GAPEIssue003_RC1.pdf
```
`data003.py` holds every number printed in the document; change a value there and rebuild. `gape002_lib.py` is the Issue 002
script with its `build()` cut into page functions — every 002 primitive, card and section reused verbatim.
Note: `data003.py` reads `handoff/*.json` relative to its parent for the live-run tables; in this kit those live in `results/` —
set `CPG_HANDOFF=../results` or copy them.

---

## 6. What is still open (do not assume it is settled)

- **Healthy whole blood reads below the age band** (43M z −2.07, 58M −4.03, 67F −3.49) on the project's own Stage-1 betas. Not a
  Stage-1 artifact (PROC-CAL-01). Leading hypothesis: the band was compiled on a different IDAT→β pipeline (Xu-538 relativity, 0.38–0.62).
  Test: locate the pipeline the band was compiled from, or re-derive the band from ≥30 public healthy blood samples through Stage 1.
- **Deconvolver tissue-of-origin.** Passes on synthetic linear mixes and on the terminal class; does not recover hepatocyte or colon
  spikes into their classes; routes shed epithelium to gastric references. Atlas work, not solver work (LESSON-DECONV-01).
- **Two presence floors** (1% conductor, 3% adjudicator) and **two tier vocabularies** in the corpus — see Issue 003 RECON D2, T3.
- **Stage 3 is not wired by decision** (SOP §104 foreground firewall) — do not add age/sex/smoking subtraction.
- Recipe §6.3 (vault) still states the pre-§105 ruling; the author records the correction there.

---

*Nothing in this kit is clinical validation. Public retrospective data, small n, no prospective testing.*

### 6a. The below-band mechanism, quantified (added after the guard test)
The immune identity loci are selected where the atlas immune reference sits at β ≈ 0.73 (`H_min_beta` 0.7318, band ±0.05).
In all seven Stage-1 whole-blood samples those same loci read at **β_mean 0.786–0.822, i.e. +0.054 to +0.090 above the atlas
reference** — outside the ±0.05 selection band in 7 of 7 (smallest shift +0.0544). Higher β at these loci means lower entropy, hence A ≈ 0.81–0.89 against
a band centred near 0.95. So "below band" is a reference-β offset between the atlas's source pipelines and noob Stage-1 output,
not a Stage-1 error and not a property of the donors. The test that closes it: re-derive `H_min_beta` and the age band from ≥30
public healthy whole-blood IDATs run through Stage 1 (`PROC_CAL_01.py`), then re-read the seven.

## 7. PROC-CHAIN-01 — the conductor end to end (added 2026-09-19)

`cpg_conductor.run_full(beta_dict, atlas_csv, cfg={"age": N})` from `CPG_Engine/` on the eleven Stage-1 betas. First run from the repository layout; two things had to be fixed to make it run at all: the conductor resolved every file as `HERE/<name>` (flat working-folder layout) and `iam_cellular_age_scoring.py` was not in the repo. Both fixed in the same commit.

Results (immune gauge): all seven whole-blood samples IN_BAND / NORMAL (0.954–1.006) — the shipped chain absorbs the Stage-1 β offset through the age band. Three defects recorded in Issue 003 RECON/PROC-CHAIN-01: stem_adult false BREACH on 6 of 7 blood samples — every one where its fraction cleared the 1% presence floor (its band is n = 28, one source), a Mahalanobis key mismatch between `run_full` and the report builder, and cellular age pinned at the curve floor. **Do not report stem_adult from blood, Stage 5 distance, or Stage 6 age until those are fixed.**

## 8. PROC-STAGE0-01 — Stage 0 intake on raw IDATs (added 2026-09-19)

`stage_0_intake.py` steps 0.1–0.9 on the eleven raw IDAT pairs (methylprep env). Array type is verified from the IDAT header, files are SHA-256 hashed with re-transmission detection, bead counts pass. Detection-p, call rate and sex check need Stage 1's decoded intensities and report DEFERRED until that hand-off is wired. **Defect fixed:** the decision gate ignored Step 0.1 quarantines (fail-open) — guard added.

## 9. PROC-N7-01 — end-to-end synthetic simulation (added 2026-09-19)

`CPG_Engine/Synthetic_Patient_Generator/synthetic_patient_generator.py` (restored; now reads the repo atlas; `composition_alpha=WHOLE_BLOOD_ALPHA`) → 24 synthetic patients → `run_full`. Composition recovery PASS (MAE ≤ 0.015). **Gauge FAIL:** every synthetic healthy reads BREACH, because the conductor's class gauge is H(β̄) over the *marker union* (bimodal), not the identity loci — despite its docstring. Real blood masked this because the age band was compiled on the same statistic. Identity-loci H(β̄) reads the synthetic healthy at 0.99 and real adenoma at 1.10 (correct both times) but has no band until Phase 1. **Do not read a conductor gauge value as a class measurement until `gauge_surface` says `identity_loci`.** Section 7's 'healthy IN_BAND' is withdrawn as conformance.

## 10. RULE — the cosmology-evidence ledger (added 2026-09-19)

Every time a CMB-derived method (end-to-end simulation, injection-recovery, split-half cross-check, convergence/distinctness test, look-elsewhere correction, sealed pre-registration, transfer-function decomposition) surfaces something a cohort comparison could not have, add a row to `COSMO_EVIDENCE` in `Issue003/data003.py` the same day: date · method · why a cohort is blind to it · what was found · PROC/VAL. Reversals and withdrawals go in too. This ledger is Issue 003 §1.6 and is the pre-built answer to "your reasoning is circular".

## 11. RULE — a safeguard that can be switched off when it disagrees is not a safeguard (added 2026-09-19)

> **Trust your equipment.** Cutting a check because it trips is ripping the methane detector out of the house because it went off. When the alarm sounds, the first question is not "is the detector broken?" but "what does the detector know that I don't?" — H. Mahaffey, senior grid operator, 2026-09-19.

Two safeguards built in spring 2026 were switched off because their first real finding looked like a defect in the safeguard: NILC (disagreed with Walther on every blood sample → cut; it was reporting that the Atlas cannot split immune/progenitor/stem_adult in blood, PROC-NILC-01) and the synthetic patient generator (retired unused; on its first run it exposed the production gauge reading the marker union, PROC-N7-01). Rule: **N7 (synthetic cohort through the full chain) and the cross-method comparison run on every chain release.** A disagreement gets a row in the §1.6 ledger and a RECON entry; it is never resolved by disabling the check.


## 12. RULE — map β onto the floor's scale before any absolute reading (added 2026-09-20)

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

Pre-flight check: for every cohort, compute healthy-immune β̄ on the identity loci and compare to 0.737 (Roadmap). A departure > 0.01 without a matching entry in `beta_scale_maps_v1.json` halts absolute reporting for that cohort.

## 13. Getting raw IDATs fast (added 2026-09-20)

Do not download `GSExxxxx_RAW.tar`. GEO serves it single-stream at ~0.7 MB/s and it carries every sample in the study; a 5.7 GB tar for 210 controls out of 699 took 2.5 h. Instead:

```
python3 Biological_Physics/CPG_Engine/tools/geo_fetch_idats.py GSE125105 idats/GSE125105 --field diagnosis --value control --workers 8
```

reads the series-matrix header by range request, selects samples on a characteristics field, and fetches only their `_Grn/_Red.idat.gz` from `geo/samples/GSMnnn/GSM/suppl/` with 8 threads: 210 controls, 1.7 GB, **6 minutes at ~5–7 MB/s**. Idempotent; writes `selected.json`. Then run Stage 1 with a process pool (6 workers on 8 cores, ~6×) — `band_v2_test_run.py::calibrate_all` is the template. The per-sample code path is identical to PROC-CAL-01; only the scheduling changes.

Streaming a series matrix (author-processed β) is still the fastest route for **within-pipeline** work (anchor reproduction, PROC-ANCHOR-01) and must NOT be used for absolute gauge readings or reference bands (LESSON-SCALE-01).

## 14. THE FINDING PROTOCOL — one procedure, every finding, in this order (added 2026-09-20)

Written because the pipeline-scale offset was known in April and lost by June, and because on 2026-09-20 three day-one passages in Issue 003 were still stating overturned claims until the author read them. Information updated in different places at different times in different manners is how a record rots. This is the one way.

**When a finding lands (a PROC, PHASE, VAL outcome, or a correction):**

| step | do | where | check |
|---|---|---|---|
| 1 | **Seal the record.** OUTCOME.md with the sealed PREREG it answers, sha256 at the foot. Post-seal changes are labelled ADDENDUM/CORRECTION, never edits. | `Testing_and_Code/VAL_PostAtlas/<ID>/` | file exists, checksum line present |
| 2 | **Kill what it overturns.** List the phrases the finding makes false ("retires"). Grep `Issue003/build_gape_issue003.py`, `Issue003/data003.py`, `SOP/*.md`, `Reproduction_Kit/*.md`, `HANDOFF.md`. Fix, or mark WITHDRAWN with the ID. | everywhere | none of the retired phrases render in the PDF (record-of-correction sentences excepted — they must contain the word WITHDRAWN/CORRECTED/SUPERSEDED) |
| 3 | **Register it.** One row in each register it touches: RECON (a constant/rule changed), FALSIFICATION (a claim withdrawn), §1.6 COSMO_EVIDENCE (a CMB tool found it), CHAIN_COMMISSIONING.md (the stage's status), switching_order.py (the stage's lessons/procedures), FUTURE_GOALS (opened or closed). | `data003.py`, `CHAIN_COMMISSIONING.md`, `switching_order.py` | the ID appears in each register the finding touches |
| 4 | **Close it in code** if it is a lesson. A label, a guard, a refusal to report (`scale=UNMAPPED → reportable=False` is the model). A lesson that lives only in prose is re-learned. | `CPG_Engine/` | the guard has a test in the kit |
| 5 | **Teach it.** ONE canonical paragraph, identical text, in every door a reader opens first: HANDOFF.md, root/Engine/Testing/Atlas READMEs, SOP (new §), RUNBOOK (new § or pre-flight), CPG_Lessons_Learned, README_FOR_FUTURE_AI, and the module docstring it bites. | the door list | the ID appears in every door |
| 6 | **Rebuild and READ.** Page 1, page 2, §11 (coverage), and every touched section — by eye. Assertions catch strings; only reading catches a stale sentence that uses new words. | PDF | page count, ID rendered, retired phrases absent, visual check of touched pages |
| 7 | **Push with copies.** Commit names the ID. `push_copies_<date>_<from>_to_HEAD.zip` + the OUTCOME as a plain file + the RC PDF saved as artifacts. Update `STATUS_*.md` in place. | repo + artifacts | the user has the copies |

`Reproduction_Kit/finding_check.py <ID> --retires "phrase" ...` runs steps 2, 3, 5 and 6's string checks as assertions and exits non-zero on any miss. It does not replace step 6's reading.

**What is NOT a finding:** a typo, a layout fix, a renamed variable. Those get a commit and copies, nothing else.

**Two rules that keep the count down** (author, 2026-09-20 — "that is exactly how I ended up with 50 versions of an SOP and 10 other documents"):

- **No new document for a finding.** The document set is CLOSED: Issue 003 (the one book), the SOP (one live file), this RUNBOOK, HANDOFF.md, CHAIN_COMMISSIONING.md, switching_order.py, one STATUS sheet, and one OUTCOME folder per sealed procedure. A finding goes into a register or a door that already exists. Wanting a new roadmap / lessons / status file is the signal that its place has not yet been found — find it. New files are for new *code* and new *sealed procedures*, never for new *prose*.
- **One live version per document, edited in place; git is the history.** No `_v2_1`, `_final`, `_FINAL2`. The SOP's supersession ledger says what changed and why; `git log -p` holds every prior state. Retired documents go to `RETIRED/` with a date and are never edited again.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Testing_and_Code/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.

**H_min CROSS-CHECK (PROC-HMIN-BOOT-01, 2026-09-20).** The eight methylation floors were calibrated by G-002 MCMC (37 reference cells, R-hat < 1.001). The April bootstrap cross-check (0.168 %, 24/32 in CI) covered the 32 non-methylation floors only; the methylation eight were bootstrapped for the first time on 2026-09-20 — 8/8 inside the 95 % CI, 0.060 % mean / 0.095 % max relative difference. Values unchanged. The calibration scripts are not at HEAD (removed 2026-04-19 as commercial) but are in public history at 22749f0 — a disclosure decision for the author. Record: `Testing_and_Code/PROC_data/PROC-HMIN-BOOT-01/`.

**LAB ZERO — PANEL SPECIFICATION (PROC-PANEL-01 → PROC-PANEL-03, 2026-09-20; supersedes the '20–30 arrays' wording above).** A laboratory's zero is measured once on **40** healthy arrays of any age mix through the same Stage 1 and map: z = median[A − c(decade)] − 1, where c is the reference healthy age curve (`reference_age_curve_v1.json`; healthy immune A rises ≈0.045 from the teens to the eighties within a lab, while between-lab offsets are parallel). A patient reads A″ = A − c(decade) − z. Tested leave-one-lab-out on four labs: a band built on three holds 75–84 % of the fourth's healthy donors. In code: `CPG_Engine/lab_zero.py` — panels under 40 are refused and `lab_zero=UNSET` is not reportable. Record: `Testing_and_Code/PROC_data/PROC-PANEL-01 → PROC-PANEL-03/`.

**THE VALIDATION COUNT, CORRECTED (PROC-HISTORY-01, 2026-09-21).** The record, not the tree: 3 G-series calibrations; **119 pre-Atlas VALs (VAL-001..128; 107 executed)** incl. the **T1–T15** cross-population series of VAL-049 (12 executed, 6 populations); **22 post-Atlas CPG-VALs (001..022; 21 executed)**; the Mahalanobis hull v0_1→v0_5 (n=2,523, four populations incl. Han Chinese n=42); L9 N7; the September PROCs. The 2026-09-19 index said 103 — it keyed on bare numbers (VAL-001 collided with CPG-VAL-001) and counted folders. `Testing_and_Code/VAL_INDEX.csv` is rebuilt (175 rows, unique keys by series); AD folders CPG-VAL-008..014 moved to `VAL_PostAtlas/`. Record: `Testing_and_Code/PROC_data/PROC-HISTORY-01/`.

**THE GAUGE SWITCH (PROC-SWITCH-01 → PROC-SWITCH-02, 2026-09-21; row B COMMISSIONED).** `cpg_conductor.run_full` now REPORTS the identity-loci gauge: A = H(β̄)/H_min on `iamatlas_gauge_identity_loci_v1_0.json`, on mapped β, minus c(decade) (`reference_age_curve_v1.json`), minus the laboratory zero (`lab_zero.py`), placed in `identity_band_v3.json` (four zeroed labs, n = 1,379, pooled p10–p90 0.9724–1.0248). The marker-union statistic is `diagnostic_marker_union` — never the reported A. Stages 5 and 6 carry `pending_recalibration=True`. Test: `Reproduction_Kit/test_gauge_switch.py`. **Finding:** the atlas posterior is a fifth laboratory (z = −0.0146) — SWITCH-01's S4 assumed zero and failed as sealed; every β source, including a simulator, is zeroed before it is read absolutely.
