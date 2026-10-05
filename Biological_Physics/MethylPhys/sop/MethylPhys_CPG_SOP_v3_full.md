# MethylPhys CPG SOP — chain v3, neutrophils (full procedure)

**Build:** DEVELOPMENT - not commissioned (chain v3, neutrophils only). Not a diagnostic test. No tier beyond Normal is printed.
**Round 2 (2026-10-04).** The intake, noise gate, identifiers and development flags changed in round 2: sex and age are optional (`NOT_DECLARED`); only blood specimens are accepted (others stop with `SPECIMEN_REFUSED`); identifiers are hashed in the bundle and ledger; if fewer than 90 % of the noise sites are read the gauge state is withheld with the reason; Stage Q prints the IAM-A C-score; development stages run only behind `--dev-*` flags. `MethylPhys_CPG_SOP_v3.md` (sections 2, 2b, 5) and the code are current where this file differs; the line numbers below are from the earlier build.
**Stage T (2026-10-04).** Stage T is self-tare II, then the median tare, adopted by the author on 2026-10-04 (`boxruns/run1/JOBS.md` job A; `doors/DEV_SELFTARE_02.md` reading (iv); development log 2026-10-04, DEV-PAIRED-01). Section 1 (tare rows), 4.1, 4.3, Stage T in section 5, section 6 item 6, the tare rows of section 7 and section 9 items 1 and 9 are updated for it. Self-tare II was wired into Stage T on 2026-10-04 (development log DEV-SELFTARE-03); `--dev-selftare-ii` is now a no-op alias. The noise-corrected tare described in this file before was removed on 2026-10-02 (DEV-TARE-02) and is kept here only as history.
**Scope:** one cell type (neutrophils); Illumina EPIC v1 arrays for Met-A; single-molecule reads (pipeline `loyfer_pat_v1`) for IAM-A. 450K neutrophil floor: pending (canon `Met_A_floor_450K_neutrophil` = null).
**Written from:** repository `hmahaffeyges/IAM-Validation`, `main` at `7cdbbf8` (floor v1.3) **plus the audit-fix patch** (`chain_fix_patch.zip`, branch `audit-fixes`, not yet pushed). Line numbers refer to the patched files. Paths are relative to `Biological_Physics/MethylPhys/`. Every number is read from a frozen file or the canon; the file and key are given beside it.
**Readings:** Met-A (arrays), Met-A C-score (arrays), A_rel (Met-A after the same-run tare), IAM-A (sequencing).

---

## 1. Physics stated once

| quantity | formula | where it is computed |
|---|---|---|
| per-site entropy | H(β) = −β log₂β − (1−β) log₂(1−β), bits; β clipped to [1e-6, 1−1e-6] | `chain/stage_m_met_a.py:33` (`_H`) |
| Met-A, isolated neutrophils | A = mean over the measured identity sites of H(β) ÷ floor | `chain/stage_m_met_a.py:71` |
| Met-A, whole blood | A = mean_i H(β_i) ÷ mean_i H(e_i), e_i = Σ_g f_g μ_g,i (f: this specimen's Stage A fractions; μ: purified EPIC group profiles) | `chain/conductor_v3.py:106-110` |
| shift per 1 % loss (whole blood) | β′_i = β_i + f_NEU × 0.01 × (0.5 − μ_NEU,i); shift = mean H(β′)/mean H(e) − A | `chain/conductor_v3.py:111-112` |
| shift per 1 % loss (isolated) | β′_i = β_i + 0.01 × (0.5 − μ_NEU,i); shift = A × mean H(β′)/mean H(β) − A | `chain/conductor_v3.py:136` |
| entropy ceiling flag | m = mean β at the sites where μ_NEU > 0.5; `past_entropy_ceiling` = (m < 0.5) | `chain/conductor_v3.py:120-126` |
| residual map | z_i = (H(β_i) − H(ref_i)) ÷ s_i; ref_i = healthy neutrophil mean H (isolated) or H(e_i) (whole blood); s_i = shrunk healthy SD of H | `chain/conductor_v3.py:117, 138` |
| Met-A C-score | c = var(means of consecutive blocks of `clustering_block` sites of z, × √block) ÷ var(z); C = c ÷ healthy median clustering | `chain/conductor_v3.py:84-88, 141-148` |
| noise index | N = mean H(β) over the noise sites measured on this array (≥ 90 % of 48,528, else None) | `chain/conductor_v3.py:52-57` |
| tare, step 1: self-tare II (adopted 2026-10-04) | per probe design (type I, type II): L, U = mean β over this array's low and high fixed sites; β′ = Lr + (β − L)(Ur − Lr) ÷ (U − L), Lr, Ur = the same anchors averaged over the six reference arrays; Met-A formed from β′; nothing fitted; a design with anchors missing or U − L ≤ 0.1 is left unmapped | `chain/conductor_v3.py` (`stage_t_selftare_ii`) with `chain/dev_stages.py` (`anchors`, `selftare_map`), wired into Stage T 2026-10-04 |
| tare, step 2: median (≥ 3 references) | A_rel = A ÷ median(A of the references) | `chain/conductor_v3.py: stage_t_tare` |
| reference spread | SD (ddof 1) of reference A ÷ median | `chain/conductor_v3.py: stage_t_tare` |
| history: tare, noise-corrected (removed 2026-10-02, DEV-TARE-02) | whole blood: A = a + b f_NEU + c N fitted by least squares on the references; isolated: A = a + c N; A_rel = A ÷ prediction(this specimen's f_NEU, N); spread = SD of each reference's leave-one-out A ÷ prediction | was `chain/conductor_v3.py:183-190` |
| detection limit | 2 × reference_spread_sd ÷ shift_per_1pct_loss, in % loss of the neutrophil pattern | `chain/conductor_v3.py:195` |
| IAM-A | A = H(ε) ÷ (P_cell × H(ε₀)), ε = isolated copy errors ÷ opportunities | `chain/stage_q_iam_a.py:79-81` |
| holding energy | E = ln((1−ε)/ε), in kT | `chain/stage_q_iam_a.py:83` |

**Normal band:** 0.95–1.05 (canon `constants.Normal_band` = [0.95, 1.05]; code `chain/stage_m_met_a.py:23` `NORMAL`, used by Stage M, T and Q). State words: `Normal` (0.95 ≤ A ≤ 1.05), `above Normal`, `below Normal`.

Loss of a held pattern pulls β toward 0.5 and raises H, so Met-A rises with loss **while the cell's methylated sites stay above β = 0.5**. Past that point H falls again and A is no longer monotone in loss (flag `past_entropy_ceiling`).

---

## 2. Install

The run is one Python process: Stage 0, Stage 1 (methylprep) and the v3 stages run in the same interpreter.

| package | version | source |
|---|---|---|
| Python | 3.11 | `chain/requirements.txt` header; `atlas/v2/environment/box_env_requirements.txt` (3.11.16) |
| methylprep | 1.7.1 | `chain/requirements.txt` |
| numpy | 1.26.4 | same |
| pandas | 1.5.3 (methylprep 1.7.1 calls `DataFrame.append`, removed in pandas 2) | same |
| scipy | 1.17.1 | same |
| pytz, python-dateutil | unpinned | same |
| matplotlib, pyarrow | toolkit only (`chain/TOOLKIT.md`); not needed by v3 readings | same |

```
python3.11 -m venv cpg_v3 && . cpg_v3/bin/activate
git clone https://github.com/hmahaffeyges/IAM-Validation.git
pip install -r IAM-Validation/Biological_Physics/MethylPhys/chain/requirements.txt
```
Stage 1 setup (`doors/RUNBOOK.md:67-69`): `HOME` must point at a writable directory (methylprep writes `$HOME/.methylprep_manifest_files/`). First use needs network access to `https://array-manifest-files.s3.amazonaws.com/` for the Illumina manifests; offline, place `HumanMethylation450k_15017482_v3.csv.gz` and `HumanMethylationEPIC_manifest_v2.csv.gz` in that directory. If methylprep is missing, Stage 1 tries a one-time `pip install methylprep`; install it by hand instead, with the pins above.

---

## 3. Frozen files and the values the chain reads

All under `chain/Runtime Matrices/`.

| file | key | value | read by |
|---|---|---|---|
| `../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json` | `version` | `metA_floors_v1_3` (bundle `floors_version`) | `stage_m_met_a.py:28` |
| | `platforms` | EPIC, cell `neutrophils` only | `stage_m_met_a.py:57-71` |
| | `platforms.EPIC.neutrophils.floor` | 0.33026279581151297 bits (canon `Met_A_floor_EPIC_neutrophil` 0.330263) | `stage_m_met_a.py:71` |
| | `…n_sites` / `…sites` | 6000 identity sites (3,000 methylated, 3,000 unmethylated) | `stage_m_met_a.py:68-69` |
| | `…n_ref` / `…refs` | 6 physical arrays (GSE110554; GSE167998 re-deposits the same 6, listed in `…duplicates_removed`) | record only |
| | `…precision_heldout` | n 6, SD 0.019756, 0.98264–1.04472 (sites re-chosen on the other 5 arrays) | record only |
| `../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv` | rows `platform=EPIC, cell=neutrophils`, column `A_loo` | 6 rows; printed as `n_ref` 6, `normal_fraction` 1.0, `sd` 0.0198, `min` 0.983, `max` 1.045 | `stage_m_met_a.py:50-55` (`floor_precision`) |
| `../chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json` | `groups` | B, BASO, CD4T, CD8T, EOS, MONO, NEU, NK | `conductor_v3.py:81` |
| | `markers` / `mu_markers` | 963 composition markers (none is a neutrophil identity site); ≥ 867 must be measured | `conductor_v3.py:65-82` |
| | `neutrophil_sites` | the same 6000 sites as the floor file; ≥ 5400 must be measured in whole blood | `conductor_v3.py:97, 107` |
| | `profiles_at_neutrophil_sites` | group mean β at those sites; 6 sites carry a missing value in ≥ 1 group and drop out of the whole-blood reading (at most 5994 sites) | `conductor_v3.py:97, 106` |
| `../chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json` | `version` | `neutrophil_reference_v1_1` (bundle `reference_version`) | `conductor_v3.py:37-40` |
| | `sites_ordered` | the 6000 sites in genome order | `conductor_v3.py:116, 130` |
| | `neutrophil_H_mean`, `neutrophil_H_sd_shrunk` | healthy neutrophil mean and shrunk SD of H per site (6 arrays) | `conductor_v3.py:117, 138` |
| | `clustering_block` | 50 sites | `conductor_v3.py:85` |
| | `healthy_clustering_median` | 1.1104 | `conductor_v3.py:144-145` |
| | `healthy_clustering_LOO` | 6 values, 0.7763–1.3632 (÷ median: 0.6991–1.2277) | `conductor_v3.py:147` |
| | `profiles_mean_beta`, `profile_map` | record only (not read) | — |
| `../chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json` | `sites` (`n` 48,528) | EPIC sites every purified blood group holds fixed (`rule`: every group mean ≤ 0.03 or ≥ 0.97, every group SD ≤ 0.02; not neutrophil sites); copied from `doors/data/noise_sites_EPIC_v1.json` unchanged | `conductor_v3.py:45-57` |
| `../chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` | `eps0` | 0.032 (canon `eps0_meth`) | `stage_q_iam_a.py:70` |
| | `cells.neutrophils.P` | 1.099 (canon `P_neutrophil_IAM_A`) | `stage_q_iam_a.py:81` |
| | `cells.neutrophils.pipeline` | `loyfer_pat_v1` | `stage_q_iam_a.py:75` |
| | `…P_range`, `…cv_across_donors`, `…n_donors` | [1.0841, 1.1079], 0.0118, 3 | record only |
| `../chain/Runtime Matrices/Intake/intake_thresholds_v1.json` | `detection.pass_fraction` / `detection.borderline_fraction` | 0.99 / 0.93 | `stage_0_intake.py:580-581` |
| | `call_rate.proceed_at_or_above` / `call_rate.quarantine_below` | 0.98 / 0.93 | `stage_0_intake.py:682-683` |
| | `bead.pass_fraction` | 0.995 | `stage_0_intake.py:649` |
| | `bisulfite_conversion.min` | 0.95 (provisional: recorded, not refused) | `stage_0_intake.py:490` |

Code constants (not in a frozen file): `MIN_READ_FRACTION` 0.20, `MIN_MARKER_FRACTION` 0.9, `MIN_REFS` 3, `MIN_REFS_NOISE` 20, `MIN_NOISE_FRACTION` 0.9 (`conductor_v3.py:30-34`), `ACCEPTED_ARRAY_TYPES` (EPIC_v1) (`:35`); `SITE_COVERAGE_MIN` 0.9 (`stage_m_met_a.py:24`); C-score ≥ 10 blocks (`conductor_v3.py:87`); IAM-A ≥ 100,000 opportunities, half-readings above 50,000 (`stage_q_iam_a.py:17-18`); hybridisation ratio ≥ 2.0, extension ratio 0.2–5.0, detection p ≤ 0.01, ≥ 3 beads, sex cut −2.0, 450K coverage ≥ 0.80 (`stage_0_intake.py`).

---

## 4. Run order

### 4.1 One specimen (arrays)
Run from `chain/MethylPhys_Interface/`.
```
python run_sample.py --grn S_Grn.idat.gz --red S_Red.idat.gz --specimen "whole blood" --sex F --age 52 --id S001 --out S001.html
```
- `--engine v3` is the default.
- `--sex` (F/M) and `--age` are required by Stage 0 (manifest fields).
- `--array-type`: omit it and the IDAT header decides. If the header is unreadable and the flag is omitted, Stage 0 quarantines (`QUARANTINE_INCOMPLETE_MANIFEST`, array_type missing). Only EPIC_v1 is read.
- `--specimen`: `isolated neutrophils`, `sorted neutrophils`, `purified neutrophils` or `neutrophils` select the isolated path; **any other string is read as whole blood**.
- Custody: `--patient-id <hashed id>`, `--intake-log <path.jsonl>`, `--manifest-dir <dir>`.
- References (pass 2, Stage T step 2): `--slide-ref-table refs.csv` (column `A`, optional `id`) or `--slide-ref-A a1,a2,a3`; ≥ 3 → median tare; not both. (Before 2026-10-02, ≥ 20 rows with `f_neu,N` gave the noise-corrected tare; removed, DEV-TARE-02.)
- Stage T step 1 (self-tare II, adopted 2026-10-04): wired into Stage T on 2026-10-04 and always runs; no references; recorded under `tare.selftare_ii`; `tare.A_rel` is the median tare of the self-tared A. `--dev-selftare-ii` is a no-op alias.
- Record: `--covariate KEY=VALUE` (repeatable) or `--covariates file.json`; `--ledger <path.jsonl>` (default `evidence_ledger.jsonl` beside the report); `--no-bundle` writes neither bundle nor ledger row.
- Already-calibrated β: `--betas S.csv` (two columns `cpg_id,beta`; Stage 0 and Stage 1 do not run).
- IDAT pair without a custody record (lab-made DNA mixtures): `--no-intake`.

Outputs: `S001.html`, `S001_bundle.json` (or `--bundle <path>`), one ledger row. Console: `S001: neutrophil Met-A <A> (<state or reason>) | C <C> | tare <A_rel>`. Exit code 2 = QUARANTINE (no report, no bundle).

### 4.2 One specimen (sequencing, IAM-A)
```
python run_sample.py --pat S.pat.gz --id S001 --out S001.html                       # extractor loyfer_pat_v1
python run_sample.py --pat S.pat.gz --pat-max-bytes 60000000 --id S001 --out S001.html  # the byte range P was measured on
python run_sample.py --site-table S_sites.csv --seq-pipeline loyfer_pat_v1 --id S001 --out S001.html
```
`--seq-cell` defaults to `neutrophils`. `--site-table` without `--seq-pipeline` is a usage error; `--pat` with a `--seq-pipeline` other than `loyfer_pat_v1` is refused. Sequencing input may be combined with `--betas` or an IDAT pair in one run; Stage 0 runs only for IDAT input.

### 4.3 A batch with references (two-pass tare)
The gauge state of every Met-A reading comes from the tare: whole blood always; isolated neutrophils when ≥ 3 same-run references are supplied (untared otherwise, and so labelled).
1. **Plan the run.** ≥ 3 healthy reference specimens of the same specimen type on the same slide as the specimens (else in the same batch), processed the same way (extraction, bisulfite batch, scanner, Stage 1).
2. **Pass 1.** Run every array (references and specimens) without references. From each bundle take `met_a.A`, `met_a.fraction` (f_neu; null for isolated) and `met_a.noise_index` (N), or the ledger columns `A`, `f_neu`, `noise_index`.
3. **Choose each specimen's references.** Healthy references of the same specimen type, excluding the specimen itself, with `A` a number: the same slide (≥ 3), else the same batch (median tare, Stage T step 2). (Before 2026-10-02 a batch of ≥ 20 references with f_neu and N gave the noise-corrected tare; removed, DEV-TARE-02.) The code does not check that a reference is healthy, in the same run or of the same specimen type: the operator is responsible. Put the specimen's id in the table's `id` column only for references; a row whose `id` equals the specimen's `--id` is dropped.
4. **Pass 2.** Re-run each specimen with `--slide-ref-table refs.csv` (columns `A[,id]`), or with `--slide-ref-A a1,a2,a3,…`, for the median tare. Stage T step 1, self-tare II, always runs (wired 2026-10-04; it needs no references); the median tare runs on the self-tared A (`met_a.A`), and the references' A values are their `met_a.A` from pass 1. Pass 2 repeats Stage 0 and Stage 1. **Use a different `--intake-log` (or none) for pass 2**: the same bytes logged twice in one intake log stop the run before calibration (`RE_TRANSMISSION_DETECTED`).
5. Read `tare.A_rel`, `tare.state`, `tare.method`, `tare.detection_limit_pct_loss`, and `tare.selftare_ii` (`status`, `anchors`, `maps`).

`chain_tests/chain_batch.py` implements steps 2–4 (pass 2 writes `reports/<gsm>_refs.csv` and passes `--slide-ref-table`); it and `run_chain_acceptance.py` carry the author's machine paths and are worked examples, not the operator tool.

---

## 5. Stage by stage

### Stage 0 — intake (`chain/stage_0_intake.py`, driven by `run_sample.py:294-389`)
**Purpose:** refuse a specimen the chain cannot vouch for, before calibration.
**Inputs:** the Grn/Red IDAT pair; the manifest entry `run_sample.py` builds: `sentrix_id` (file name, pattern `(\d{9,12})[_-](R0\dC0\d)`), `array_type` (`--array-type`, else the header), `patient_id` (`--patient-id`, else `--id`, else the file-name prefix; hashed to 32 hex characters of SHA-256 unless already a ≥ 16-character alphanumeric token), `intake_date` (today), `substrate` (`--specimen`, spaces → `_`), `declared_sex`, `declared_chronological_age`.
**Order and stops (all before Stage 1):** 0.1 → 0.2 → 0.3; any `QUARANTINE_*` status or an integrity status other than `INTEGRITY_OK` stops the run (`run_sample.py:329`). Then the QC hand-off and 0.4–0.8, then 0.7b (EPIC: `NA_EPIC`) and the 0.9 gate; a `QUARANTINE` verdict stops the run (`run_sample.py:374-380`). Exit code 2 in every case. After Stage 1, 0.7b and 0.9 run once more (they can only add the 450K coverage check).

| step | what it decides (as coded) | threshold / rule | state written | effect |
|---|---|---|---|---|
| 0.1 arrival (`:110`) | 7 manifest fields present: sentrix_id, array_type, patient_id, intake_date, substrate, declared_sex, declared_chronological_age | non-empty | `QUARANTINE_INCOMPLETE_MANIFEST` (flag `INCOMPLETE_MANIFEST:<fields>`) | stop |
| | array type token | HM450K, EPIC_v1, EPIC_v2 | `QUARANTINE_INCOMPLETE_MANIFEST` (flag `UNKNOWN_ARRAY_TYPE`) | stop |
| | both files exist | — | `QUARANTINE_MISSING_CHANNEL` | stop |
| | file size | ≥ 1,000,000 bytes each | `QUARANTINE_TRUNCATED_UPLOAD` | stop |
| | header vs declared (nSNPsRead < 800,000 HM450K; < 1,080,000 EPIC_v1; else EPIC_v2) | 450K vs EPIC family must agree | `QUARANTINE_ARRAY_TYPE_MISMATCH`; EPIC_v1/v2 difference → flag `ARRAY_SUBTYPE_NOTE`; unreadable header → flag `IDAT_HEADER_UNREADABLE` | stop / note |
| | same Sentrix ID in the intake log within 24 h | — | flag `DUPLICATE_INTAKE_24H_SOFTWARN` | continue |
| 0.2 manifest (`:358`) | patient_id not cleartext (no space, no `@`, ≥ 16 alphanumeric) | — | `QUARANTINE_MANIFEST_INVALID` (flag `CLEARTEXT_PII`) | stop |
| | core fields present; writes `patient_manifest_<sample_run_id>.json` to `--manifest-dir` | — | `MANIFEST_COMPLETE` | continue |
| 0.3 integrity (`:438`) | SHA-256 of both files vs earlier rows for this Sentrix ID in `--intake-log` | identical pair | `integrity_status` = `RE_TRANSMISSION_DETECTED` | stop |
| | | different hashes | flag `LIKELY_RE_RUN_FRESH_ARRAY`; `INTEGRITY_OK` | continue |
| hand-off (`stage_0_1_qc_handoff.decode_qc_inputs`) | decodes controls, design-aware per-probe intensity (Type I in its own colour, Type II both channels), negative-control background (median, MAD × 1.4826), bead counts, chrX/chrY intensity, probe IDs | decoder exception | `QUARANTINE_CORRUPT_IDAT` (flag `IDAT_DECODE_FAILED:<error>`) | stop |
| | | module missing | 0.4–0.8 `DEFERRED_PENDING_STAGE1_DECODER` | quarantine at 0.9 (`intake_deferred:detection+call_rate`) |
| 0.4 controls (`:505, :545`) | bisulfite conversion per matched pair C/(C+U); hybridisation high/low; extension meth/unmeth | BS ≥ 0.95; hyb ratio ≥ 2.0; extension in [0.2, 5.0] | `ctrl_qc` = `PASS`, `FAIL_<flags>` (hard), or `PROVISIONAL_BS_THRESHOLD_UNCALIBRATED` when BS is the only one below (recorded deferred, not refused) | — |
| 0.5 detection (`:579`) | detection p = 1 − Φ((I − μ_bg)/σ_bg); detected = p ≤ 0.01 | fraction > 0.99 `PASS`; ≥ 0.93 `DETECTION_BORDERLINE`; else `FAIL_LOW_DETECTION` | `detection_qc`, `pct_probes_detected_p_le_01` | — |
| 0.6 beads (`:648`) | fraction of probes with ≥ 3 beads | ≥ 0.995 `PASS`, else `WARN_LOW_BEAD_COUNT` (borderline) | `bead_qc`, `pct_probes_bead_count_ge_3` | — |
| 0.7 call rate (`:682`) | fraction passing detection and beads | ≥ 0.98 `PASS`; ≥ 0.93 `CALL_RATE_BORDERLINE`; else `CALL_RATE_FAIL` | `call_rate`, `call_rate_status` | — |
| 0.8 sex (`:757`) | predicted F if log2(Y median) − log2(X median) < −2.0, else M | must equal declared F/M | `sex_check` = `PASS` / `MISMATCH` (hard) | — |
| 0.7b platform (`:728`) | platform tag; 450K coverage of the neutrophil identity sites | EPIC → `NA_EPIC`; 450K ≥ 0.80 | `platform_tag`, `hm450_coverage_gate` | — |
| 0.9 gate (`:794`) | hard: any `QUARANTINE_*`, integrity ≠ OK, `ctrl_qc` FAIL, `FAIL_LOW_DETECTION`, `CALL_RATE_FAIL`, coverage FAIL, sex `MISMATCH`, deferred detection or call rate. Borderline: detection, bead, call rate. Recorded deferred: provisional bisulfite, sex or controls not decoded | any hard → `QUARANTINE`; else borderline → `PROCEED_WITH_PENALTY`; else `PROCEED` | `stage0_verdict`, `stage0_hard_fail`, `stage0_borderline`, `stage0_deferred_qc` | `QUARANTINE` → stop |

**Stage-1 values recorded beside the Stage 0 record (not gated; `run_sample.py:402-436`):** `intake.stage1_qc` = {`ctrl_qc`, `ctrl_metrics`, `ctrl_flags` (from Stage 1 control medians), `detection_statistic` "poobah p <= 0.05", `detection_qc`, `pct_probes_detected`, `n_probes`, `n_probes_bead_aligned`, `call_rate`, `call_rate_status` (poobah × the extracted bead mask, aligned by probe ID)}; `intake.controls` = Stage 1 control medians incl. `signal_to_background_G/R`. A Stage-1 FAIL adds a flag `STAGE1_<field>:<value> (recorded, not gated)`.
**Outputs:** bundle `intake` (whole record); rows appended to `--intake-log` (arrival, integrity, verdict).

### Stage 1 — IDAT calibration (`chain/stage_1_idat_calibration.py:88`, called at `run_sample.py:400`)
**Purpose:** turn this array's raw intensities into calibrated β using only its own controls.
**Method:** `methylprep.run_pipeline(betas=True, export=True, save_control=True, poobah=True)`: noob background, dye-bias and probe-type normalisation, per sample. Keeps `cg` probes with poobah p ≤ 0.05; probes at background are removed before any stage reads β. Returns the per-probe poobah mask to the runner (`return_mask=True`).
**Outputs:** β vector; bundle `stage1` = {`detection` (`detection_available`, `n_probes`, `n_detected`, `pct_detected`, `n_masked`), `n_cpgs`, `pipeline`}.

### Platform check (`conductor_v3.platform_refusal`, `:201-210`)
**Rule (first that applies):** Stage 0 array type (header, else declared) not `EPIC_v1` → `array type <type>: chain v3 reads EPIC v1 arrays only (no frozen neutrophil floor for this platform)`; any probe name with the EPIC v2 design suffix (`_TC21`, `_BC11`, …) → `EPIC v2 probe names (design suffix, e.g. cg..._TC21): chain v3 reads EPIC v1 arrays only (…)`; ≤ 700,000 probes → `<n> probes (450K or incomplete vector): chain v3 reads EPIC v1 arrays only (450K neutrophil floor pending)`.
**Output:** bundle `refusal`; no Stage A/M/MC/T. **Operator:** none for 450K or EPIC v2 (no floor); for `--betas`, supply the full calibrated EPIC v1 vector.

### Stage A — composition (whole blood only; `conductor_v3.py:68-82`)
**Purpose:** this specimen's blood-cell fractions, on the same platform and reference as the expectation profiles.
**Formula:** NNLS of β at the measured markers on `mu_markers` (8 groups), f ← f/Σf.
**Rule:** measured markers < 867 (0.9 × 963) → `fractions` null, `reason` = `only <n> of 963 composition markers measured (>= 867 required): composition not solved`; Stage M then withholds A with the same reason.
**Outputs:** bundle `composition` = {`stage`, `method`, `n_markers_used`, `n_markers_required`, `n_markers_total`, `fractions` {B, BASO, CD4T, CD8T, EOS, MONO, NEU, NK}, `residual_mae`} or `reason`. Isolated specimens: {`stage`, `note` "isolated neutrophils: composition not solved"}.

### Stage M — Met-A, isolated neutrophils (`stage_m_met_a.read`; `conductor_v3.py:128-139`)
**Inputs:** β at the 6000 identity sites; floor 0.33026279581151297; `../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`; `profiles_at_neutrophil_sites.NEU`.
**Rules:** cell must be `neutrophils`; measured identity sites ≥ 5400 (0.9 × 6000), else `only <n> of 6000 identity sites measured` (A withheld, no C-score).
**Outputs:** bundle `met_a` = {`stage`, `reading`, `cell`, `platform`, `specimen`, `fraction` null, `A`, `band` "Normal 0.95-1.05", `build`, `floors_version`, `n_sites`, `floor`, `state_own_floor` (Normal / above / below), `state`, `floor_precision` {`n_ref` 6, `normal_fraction`, `sd` 0.0198, `min` 0.983, `max` 1.045}, `methylated_sites_mean_beta`, `past_entropy_ceiling`, `shift_per_1pct_loss`, `noise_index`, `noise_sites_measured`, `noise_sites_total`}. `state` = `untared (own-floor state: <state>): read A_rel (Stage T)` without references; `tared: read A_rel (Stage T)` after a tare.
**Operator:** supply ≥ 3 same-run references and read `tare.A_rel`. If only the own floor is available, read `state_own_floor` knowing the floor's held-out spread (SD 0.020) and that array noise in another laboratory moves this reading (section 8). If `past_entropy_ceiling` is true, read `methylated_sites_mean_beta`, not A.

### Stage M — Met-A, whole blood (`conductor_v3.py:93-118`)
**Inputs:** β at the 6000 `neutrophil_sites`; Stage A fractions; `profiles_at_neutrophil_sites` (8 groups).
**Rules (in order):** composition not solved → A withheld (Stage A reason); f_NEU < 0.20 → `neutrophil fraction <f> < 0.2: fraction reported, A withheld`; sites with both β and e < 5400 → `only <n> of 6000 neutrophil sites measured (>= 5400 required): A withheld`. In each case C-score and tare return no value. A read value is printed with `state` = `untared: read A_rel (Stage T)` until tared.
**Outputs:** bundle `met_a` = {`stage`, `reading`, `cell`, `specimen` "whole blood", `fraction`, `build`, `band` "Normal 0.95-1.05 (after tare)", `A`, `shift_per_1pct_loss`, `methylated_sites_mean_beta`, `past_entropy_ceiling`, `n_sites`, `expectation`, `state`, `noise_index`, `noise_sites_measured`, `noise_sites_total`} or `reason` (the noise fields are recorded in every case).
**Operator:** run pass 2. A low fraction is the result, not a fault. `shift_per_1pct_loss` falls with f_NEU; read it through the detection limit.

### Stage MC — Met-A C-score (`conductor_v3.py:141-148`)
**Inputs:** residual z at `sites_ordered`; `clustering_block` 50; `healthy_clustering_median` 1.1104; `healthy_clustering_LOO` (6).
**Rule:** needs ≥ 10 blocks of measured sites and a Stage M reading; otherwise `C` null, `reason` `no residual map`. No band.
**Outputs:** bundle `met_a_cscore` = {`stage`, `reading`, `C`, `clustering`, `healthy_baseline` 1.1104, `n_healthy_baseline` 6, `block_sites` 50, `healthy_range` [0.6991, 1.2277], `status` "development: healthy band not yet set", `frac_abs_z_gt3`}.
**Operator:** record C; no interpretation against a band. In whole blood the residual includes composition error.

### Stage T — same-run tare: self-tare II, then the median tare (adopted 2026-10-04)
**Purpose:** read the specimen on the reference arrays' scale and against healthy references run the same way, removing the composition and laboratory offset; state the smallest loss this specimen could show. Applies to whole blood and isolated neutrophils.
**Adopted:** by the author on 2026-10-04 (`boxruns/run1/JOBS.md` job A); the method is `doors/DEV_SELFTARE_02.md` reading (iv), which met every replicate and other-laboratory bar (replicate within-person SD 0.0164, 62/63 Normal; other laboratories 49/49; floor 6/6); development log 2026-10-04, DEV-PAIRED-01.
**Step 1, self-tare II (`chain/conductor_v3.py`: `stage_t_selftare_ii`; `chain/dev_stages.py`: `anchors`, `selftare_map`):**
- Inputs: this array's β; `chain/Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json` (development file): the fixed-site sets `I_low`, `I_high`, `II_low`, `II_high`, the design of each probe, and `ref_anchors` (the six reference arrays: type I 0.0186 / 0.9820, type II 0.0555 / 0.9486).
- Fixed sites: type I = the noise sites of the same state (DEV-NOISE-01); type II = EPIC type II probes, not a neutrophil identity site and not a composition marker, with every purified GSE110554 group mean ≤ 0.15 (low) or ≥ 0.85 (high), group SD ≤ 0.02, largest difference between group means ≤ 0.03.
- Method, as coded: per design d, L = mean β over this array's `d_low` sites, U = mean β over its `d_high` sites; β′ = Lr + (β − L)(Ur − Lr) ÷ (U − L) at the sites of design d, clipped to [1e-6, 1 − 1e-6]; other sites unchanged. A design with L or U missing or U − L ≤ 0.1 is left unmapped. Met-A is then formed from β′ as Stage A and Stage M form it. Nothing is fitted; no references are needed.
- Assumption (conjecture, DEV-SELFTARE-02 step 3): the fixed sites hold the same true state on every array of healthy blood.
- Output (wired 2026-10-04): β′ goes to Stage A, Stage M and Stage MC; the noise index and the noise gate read β before step 1. Bundle `tare.selftare_ii` = {`step`, `status`, `anchors`, `maps` {per design: `L`, `U`, `L_ref`, `U_ref`, `slope`, `n_sites_mapped`}, `note`}; status `NOT_RUN` with the `reason` when the file is missing (the reading then uses β). `tare.A_rel` below is the median tare of the self-tared A. `--dev-selftare-ii` is a no-op alias: it copies this record to `development.selftare_ii` (`A_selftared` = `met_a.A`).
**Step 2, median tare (`conductor_v3.py: stage_t_tare`):**
- Inputs: `met_a.A`, `met_a.shift_per_1pct_loss`; references as plain A values (`--slide-ref-A`) or records {A[, id]} (`--slide-ref-table`). A record whose id equals the specimen's id is dropped (`n_self_excluded`).
- Method, as coded: fewer than 3 references with A → no tare; otherwise A_rel = A ÷ median(reference A); spread = SD (ddof 1) of reference A ÷ median; detection limit = 2 × spread ÷ shift per 1 % loss (null when the shift is missing or ≤ 0). Nothing is fitted.
**Rules:** no A → `reason` `no A`; fewer than 3 references → `A_rel` null, `reason` `untared: <n> same-run reference arrays (>= 3 required)`. With fewer than 3 references the median step cannot run; DEV-PAIRED-01 (two arrays on one slide) read self-tare II alone (A 1.0437 / 1.0426) and the gauge state was withheld on the noise index.
**Outputs:** bundle `tare` = {`stage`, `A_rel`, `state`, `method` "median tare (same-run healthy references)", `reference_median`, `n_refs`, `n_self_excluded`, `reference_spread_sd`, `detection_limit_pct_loss`, `detection_note`}.
**Operator:** with 3–19 references the spread is itself imprecise. A detection limit above the change you need to see means this specimen cannot show it. Self-tare II rests on the fixed-site assumption; the physical control DNA route (fully methylated and fully unmethylated control DNA and a 50 % mix on every slide) stays the check on it.
**History (kept):** until 2026-10-04 Stage T was the median tare alone (step 2 above; DEV-TARE-02, 2026-10-02). Before 2026-10-02 (`conductor_v3.py:150-199` of the earlier build): with ≥ 20 records with A, N (and f_neu for whole blood) and a noise index for this specimen, a noise-corrected tare, least squares A = a + b f_neu + c N (isolated: A = a + c N) on the references, A_rel = A ÷ prediction, spread = SD of each reference's leave-one-out A ÷ prediction, bundle fields `fit` and `prediction`; removed on 2026-10-02 (DEV-TARE-02).

### Noise gate (`conductor_v3.py`: `noise_gate`, applied in `run_neutrophil` after Stage T)
**Inputs:** the noise index N recorded by Stage M (mean H(β) over the noise sites, read on β before Stage T step 1); `chain/Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json`: N_max = 0.149, the top of the noise range of the six reference arrays (0.1223–0.1489; DEV-NOISE-01).
**Rule, as coded:** `met_a.noise_gate` = `pass` when N ≤ N_max, `above the reference arrays' range` when N > N_max. With N > N_max, no same-run tare (`A_rel` null) and an A, the state is `withheld: noise index N > N_max and no same-run tare; A printed as a number only`; a tared reading keeps its state, and a reading without A keeps its own reason. With fewer than 90 % of the noise sites measured, N is null, `noise_gate` is `not measured: noise-site coverage below 90 %` and the state is withheld with the reason, tared or not (author decision A, 2026-10-04). Nothing is fitted.
**Outputs:** bundle `met_a.noise_gate`, `met_a.noise_gate_N_max`; `met_a.state` when withheld.

### Report (`chain/MethylPhys_Interface/report_v3.py`)
**Outputs:** `<out>.html`; `<out stem>_bundle.json` unless `--no-bundle`; bundle top level: `build`, `specimen`, `platform`, `array_type`, `scope`, `floors_version`, `reference_version`, [`refusal`], `composition`, `met_a`, `met_a_cscore`, `tare`, `withheld`, [`iam_a`], `intake`, `intake_skipped`, `sample_id`, `covariates`, `run_id`, [`stage1`]. Ledger row (`run_sample.py:506-517`): run_id, engine, sample_id, utc, report, bundle, specimen, platform, array_type, refusal, floors/reference versions, stage0_verdict, call_rate_status, f_neu, A, state, n_sites, shift_per_1pct_loss, past_entropy_ceiling, C, A_rel, tare, n_refs, tare_method, noise_index, detection_limit_pct_loss, iam_a, iam_a_pipeline, covariates.

### Stage Q — IAM-A, sequencing (`chain/stage_q_iam_a.py`; called by `run_sample.py:482-492`)
**Purpose:** the neutrophils' per-molecule copy error against the physics floor at the neutrophil's frozen position.
**Extractor (`pat_site_table`, `:44-65`, pipeline `loyfer_pat_v1`):** reads a wgbstools `.pat`/`.pat.gz` (chrom, first CpG index, pattern of C/T/., molecule count), optionally only the first `max_bytes` bytes (multi-member gzip; a cut tail is dropped). `.` calls are dropped; a molecule qualifies with ≥ 6 calls and ≥ 80 % methylated; each interior call is an opportunity at its CpG; an unmethylated interior call with both neighbours methylated is an isolated error; halves A/B = odd/even molecule ordinal over the file. Output: per-site table `pos` ("chrom:CpG index"), `opp_A`, `err_A`, `opp_B`, `err_B`; totals equal `chain_tests/iama_floor.py`'s counts on the same file and byte range. P was measured on the first 60,000,000 bytes of each granulocyte file (`LOYFER_PAT_V1_HEAD_BYTES`).
**Reading (`read(site_table, cell, pipeline, mask=None)`, `:67-85`; pipeline required):** ε = Σerr ÷ Σopp; IAM-A = H(ε) ÷ (1.099 × H(0.032)); halves per half with > 50,000 opportunities; E_kT = ln((1−ε)/ε).
**Refusals (`A` null):** `no frozen IAM-A position for <cell>`; `pipeline not stated: name the read-level pipeline that produced the table`; `position for neutrophils was measured on loyfer_pat_v1, not <pipeline>: measure P on healthy neutrophils with this pipeline first`; `too few opportunities (<n> < 100000)`; `copy error <ε> outside (0, 1): no reading`.
**Outputs:** bundle `iam_a` = {`stage` "Q", `reading`, `cell`, `pipeline`, `build`, `A`, `eps0`, `eps`, `P`, `E_kT`, `halves` {A, B}, `state`, `opportunities`, `n_sites`, `input` {`pat` or `site_table`, `max_bytes`, `n_lines`, `n_qualifying_lines`, `n_molecules`}} or `refusal`.
**Operator:** for a site table, name the pipeline truthfully (the code cannot check what produced it). Halves should agree; a large difference is a run artefact.

---

## 6. Reading the report (top to bottom)

1. **Banner:** build string, "Not a diagnostic test."
2. **Header:** specimen, platform, array type, floors version (`metA_floors_v1_3`), reference version (`neutrophil_reference_v1_1`); a red **Refused** line when the platform check refused.
3. **Stage 0 intake:** verdict (PROCEED / PROCEED_WITH_PENALTY; `not run` for `--betas`, `--no-intake` or sequencing-only), call rate status and value, flags (first 300 characters); a second line with the Stage-1 values recorded beside the record (poobah detection, poobah × bead call rate, controls).
4. **Stage A composition:** groups ≥ 1 %, highest first; or the refusal reason; isolated: the note.
5. **Stage M Met-A — neutrophils:** gauge 0.80–1.30 with Normal 0.95–1.05 shaded. Marker = `A_rel` whenever the specimen was tared (label "tared: A_rel …"); untared isolated neutrophils: own-floor A (label "untared: … against the own floor"); untared whole blood: no marker, text "no gauge position until Stage T". Then A with state or reason, fraction, sites, expectation, shift per 1 % loss.
6. **Stage T same-run tare:** A_rel with state or reason; number of references and their median; detection limit (% loss of the neutrophil pattern) and reference spread; tare method (step 1 self-tare II with its status, then the median tare); self-tare II (adopted 2026-10-04, wired 2026-10-04) is recorded in the bundle under `tare.selftare_ii` (anchors and the map per design); (the noise-corrected method with a, b, c was removed on 2026-10-02, DEV-TARE-02); noise index N and the noise sites measured; methylated-site mean β with **"past the entropy ceiling: … read beta, not A"** when flagged.
7. **Stage MC Met-A C-score:** C, the healthy held-out range, status.
8. **Stage Q IAM-A** (when sequencing input was given): gauge, A with state or refusal, pipeline, ε, P, ε₀, halves, opportunities, E in kT.
9. **Withheld:** tier lines beyond Normal; other cell types.
10. **bundle** (collapsed): the bundle without `intake`, first 20,000 characters.

What to read: `A_rel` and its state with the detection limit; for untared isolated neutrophils the own-floor A with its untared label; always the ceiling flag; IAM-A with its pipeline. An untared whole-blood A is a number, not a state.

---

## 7. Fault table — every state and what the operator does

| printed | cause | action |
|---|---|---|
| `QUARANTINE_INCOMPLETE_MANIFEST` | a manifest field missing (`--sex`, `--age`, or array type when the header is unreadable) or an unknown array-type token | supply the field; array type HM450K, EPIC_v1 or EPIC_v2 |
| `QUARANTINE_MANIFEST_INVALID` (`CLEARTEXT_PII`) | `--patient-id` contains a space or `@` | pass a hashed id, or omit it and let the runner hash `--id` |
| `QUARANTINE_MISSING_CHANNEL` | Grn or Red file absent | supply both files |
| `QUARANTINE_TRUNCATED_UPLOAD` | a file < 1,000,000 bytes | re-fetch |
| `QUARANTINE_ARRAY_TYPE_MISMATCH` | declared type and header disagree (450K vs EPIC) | omit `--array-type` and let the header decide, or correct it |
| `QUARANTINE_CORRUPT_IDAT` | the decoder failed on the file | re-fetch; a file can pass the size floor and still be truncated inside its gzip stream |
| `RE_TRANSMISSION_DETECTED` → QUARANTINE (`integrity`) | these exact bytes are already in this intake log | for a planned re-run (tare pass 2) use another `--intake-log` or none; otherwise find who submitted the first copy |
| `FAIL_HYB_FAIL` / `FAIL_EXT_FAIL` → QUARANTINE | hybridisation or extension controls outside the line | re-hybridise / re-run the array |
| `PROVISIONAL_BS_THRESHOLD_UNCALIBRATED` | bisulfite value below 0.95, threshold not calibrated | none; recorded, not refused |
| `FAIL_LOW_DETECTION` / `CALL_RATE_FAIL` → QUARANTINE | too many probes at background | specimen or hybridisation problem; re-run the specimen |
| `DETECTION_BORDERLINE` / `CALL_RATE_BORDERLINE` / `WARN_LOW_BEAD_COUNT` → PROCEED_WITH_PENALTY | between the lines | reading proceeds; note the penalty with the result; many warnings on one plate → raise with the core facility |
| `intake_deferred:detection+call_rate` → QUARANTINE | hand-off decoder not installed | install methylprep (section 2) and re-run |
| sex `MISMATCH` → QUARANTINE | chrX/chrY call ≠ declared sex, or declared sex not F/M | check the paperwork and the sample identity; do not override |
| `STAGE1_…: … (recorded, not gated)` flag | Stage 1's poobah-based values fall below the lines | none at the gate; report it with the reading |
| `refusal: array type …` / `EPIC v2 probe names …` / `… probes (450K or incomplete vector) …` | not EPIC v1 | none (no floor); for `--betas` supply the full EPIC v1 vector |
| `only <n> of 963 composition markers measured (>= 867 required)` | markers missing after detection masking | low-quality array; re-run |
| `only <n> of 6000 neutrophil sites measured (>= 5400 required)` / `only <n> of 6000 identity sites measured` | identity sites missing | low-quality array; re-run |
| `neutrophil fraction <f> < 0.2: fraction reported, A withheld` | neutrophils < 20 % of the specimen | none; the fraction is the result |
| `untared: <n> same-run reference arrays (>= 3 required)` | fewer than 3 references | run pass 2 with `--slide-ref-table` or `--slide-ref-A` |
| history: `method: median tare: noise not corrected (<20 references)` | printed until 2026-10-02, when the noise-corrected tare was removed (DEV-TARE-02) | none |
| `noise_index` null (until 2026-10-02 also `method: median tare: noise not corrected (this specimen has no noise index)`) | < 90 % of the 48,528 noise sites measured on the specimen | low-quality array; re-run |
| `tare.selftare_ii` status `NOT_RUN`, `dev_selftare_typeII_EPIC_v1.json not found` (the reading then uses β without self-tare II) | the self-tare II file is not in `chain/Runtime Matrices/Development/` | restore the file from the repository |
| `tare.selftare_ii` `maps.<design>` null | this array's anchors for that design are missing or U − L ≤ 0.1 | none: that design is left unmapped; report it with the reading |
| `give --slide-ref-A or --slide-ref-table, not both` / `--slide-ref-table …: needs a column A` | usage | supply one reference input, with columns A, f_neu, N |
| `C` null, `no residual map` | A withheld, or < 10 blocks | none |
| `past_entropy_ceiling: true` | methylated sites average below β 0.5 | read `methylated_sites_mean_beta`; A understates loss |
| Stage Q refusals | section 5, Stage Q | name the pipeline; measure P for a new pipeline; supply ≥ 100,000 opportunities |
| methylprep import error / manifest download hang | environment or `HOME` / network | section 2 |
| `ModuleNotFoundError` on a stage module | script moved | run `run_sample.py` from `chain/MethylPhys_Interface/` |

---

## 8. Validation record (development measurements)

Measurements made on the chain while it is in development; none is a commissioning result.

### 8.1 Records in `chain_tests/`
| record | what was measured | what it showed |
|---|---|---|
| `freeze_v13.py` → `metA_floors_v1_3.json`, `../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv`, `neutrophil_reference_v1_1.json` | the neutrophil floor rebuilt with each physical array counted once (6), held-out readings with the identity sites re-chosen on the other 5 arrays | Floor unchanged (0.33026279581151297). Held-out A 0.983–1.045, SD 0.020 (on the frozen sites: 0.993–1.008). C-score baseline median 1.1104 on 6 arrays. |
| `CHAIN_V3_ACCEPTANCE_RUN3.md` (+ `chain_acceptance.csv`, `run_chain_acceptance.py`) | 22 IDAT pairs through `run_sample.py --engine v3`, two passes, on the build before the audit fixes | 22/22 processed. Purified neutrophils (the floor's own arrays): A 0.994–1.006. Six DNA mixtures (neutrophils 58–70 %): untared 0.943–0.968, tared 0.987–1.021. Ten remission bloods from a second lab: untared 1.073–1.115 on 5, tared 0.986–1.032 against the other bloods of that batch; 5 had A withheld at fractions 0.058–0.473 under that build's 0.50 line; the four at ≥ 0.20 are read by the current code and need a re-run. C: isolated 0.69–1.21, whole blood 0.78–1.49. |
| `WHOLE_BLOOD_COMPOSITION_DEV.md` (+ `blood_comp.py`, `selfconsist.py`) | 6 DNA mixtures (≥ 50 % neutrophils), composition by EPIC NNLS with profiles from the other study | Untared A 0.943–0.968 (offset about −0.05, SD 0.009); tared A_rel 0.992–1.021 (SD 0.010); a simulated 2 % neutrophil pattern loss read 1.052–1.090 tared (shift +0.061); re-fitting the fraction on the neutrophil sites absorbs the loss. |
| `IAMA_FLOOR_COMPARISON.md` (+ `iama_floor.py`, `iama_floor_granulocytes.csv`) | 3 granulocyte donors, read-level .pat, 16–20 M opportunities each; three floors compared | On ε₀ alone healthy donors read 1.084–1.127; with the frozen position P (1.099): 0.978–1.040 (P measured on the same 3 donors, leave-one-out). Odd/even repeat ≤ 0.002. Simulated 2 % copy damage: 1.285–1.346. The bare floor reads healthy blood cells 0.70–0.79 on a second pipeline: one P per pipeline. |
| `neut_ref.py`, `blood_comp.py` | builders of the neutrophil reference (v1, superseded by `freeze_v13.py`) and `blood_composition_EPIC_v1.json` | — |
| `chain_batch.py` | batch runner for the neutrophil test series below | — |

### 8.2 Synthetic exercise of the fixed code (audit-fix patch, `_audit_fix_tests/`)
Every code path above was run on synthetic β vectors, synthetic .pat files and site tables, and through `run_sample.py` (IDAT cases with the two decoder modules replaced by stubs; Stage 0 real). 41 checks: tared and untared whole blood and isolated readings, noise index present/absent, the noise-corrected tare (fit recovers a known A = a + b f + c N on 25 synthetic references; A_rel within 0.001 of the known value), median tare below 20 references or without N, self-exclusion, isolated fit without the fraction term, `--slide-ref-table`, every withhold and refusal path, report gauge source, ledger and `--no-bundle`, Stage Q extractor totals and halves equal to `iama_floor.py`'s on the same file, required pipeline, and every hard intake failure stopping before calibration. No real IDAT was run through methylprep in this exercise.

### 8.3 Development records in `doors/` that bear on v3
| record | what was measured | what it showed |
|---|---|---|
| `DEV_LOWFRAC_01_OUTCOME.md` | 656 whole bloods, Met-A at every neutrophil fraction; healthy spread and the shift of a simulated 2 % loss by fraction | Healthy SD 0.020–0.024 from 0.40 to 1.00; the shift falls with fraction (0.033 at 0.40–0.50, 0.064 at 0.70–1.00). Below 0.40: 5 healthy arrays. Basis of the per-specimen detection limit. |
| `PROC_DNMT_01_PARTA_OUTCOME.md` | 51 EPIC arrays, 3 cell lines, DNMT1 inhibitor dose and time series, each line against its own 4 vehicle arrays | Vehicle 0.968–1.048, inactive analog 1.002–1.032; active drug ≥ 80 nM 1.16–1.85 (day-4 dose series; all ≥ 80 nM arrays 1.16–1.87), second active compound 1.73–1.81 (day 4; days 2 and 4 1.60–1.85); change in the methylated channel. A saturates near 1/H(floor) once methylated sites reach β ≈ 0.5 and falls past it. Basis of the ceiling flag. |
| `DEV_NOISE_01_OUTCOME.md` | noise index N (mean H at 48,528 invariant EPIC sites) on floor arrays and a second lab | Floor arrays N 0.1223–0.1489; second-lab arrays up to 0.2434; Met-A follows N (ρ 0.79–0.83). |
| `DEV_NOISE_02_OUTCOME.md` | 495 whole bloods from one lab | Untared A tracks neutrophil fraction and N; dividing by the expectation for each array's own (fraction, N), fitted on the other healthy arrays, gives healthy SD 0.022. Fitted after looking; needs a held-out lab. |
| `PROC_NEUT_TEST_01_OUTCOME.md`, `…_T2_OUTCOME.md` | FACS-counted bloods, isolated neutrophils from a second lab, technical replicates, 570 whole bloods | Neutrophil fraction within 0.035 of FACS (median). Isolated second-lab neutrophils read 0.86–1.26 against the floor; the spread follows array noise, not purity. Tared healthy bloods in one lab spread SD 0.052; the tared reading still rises about +0.12 A per unit fraction. Replicate set: neutrophils 0.30–0.56, not read on that build. |
| `PROC_WB_NEUT_01_OUTCOME.md` | 6 mixtures with known fractions | Known-fraction expectation 0.982–1.016; simulated 2 % loss 1.049–1.079; the neutrophil floor alone reads the same mixtures 1.062–1.118. |
| `PROC_PREDX_NEUT_01_OUTCOME.md` | 845 450K arrays on a development 450K neutrophil floor | Outside current scope (EPIC v1 only). The reading differed by donor sex by 0.045, not explained by purified-neutrophil differences (0.008); centre/plate and composition residual remain open. |

---

## 9. Pending changes (not yet in the code)

1. A held-out laboratory with ≥ 20 healthy references to measure the noise-corrected tare (its coefficients were fitted after looking, DEV-NOISE-02). History: the noise-corrected tare was removed on 2026-10-02 (DEV-TARE-02); Stage T is self-tare II then the median tare since 2026-10-04.
2. Fraction-dependent precision rule: report A only where the shift at the specimen's own fraction is ≥ 2 × the healthy spread, in place of the fixed 0.20 line.
3. Bisulfite-conversion threshold calibrated at intake (`BS_THRESHOLD_CALIBRATED` set from healthy arrays).
4. Intake detection statistic: one statistic for the gate and the thresholds file (the gate uses p ≤ 0.01 against the negative-control background; `intake_thresholds_v1.json` provenance and the Stage-1 record use poobah p ≤ 0.05).
5. Tare required, not optional, for isolated neutrophils.
6. Molecule-assignment IAM-A (per-molecule cell assignment before the copy-error count).
7. IAM-A C-score (per-region copy-error map against the floor).
8. 450K neutrophil floor (purified 450K neutrophils, GSE88824), with donor sex stated.
9. Self-tare II wired into Stage T, so that A_rel is the median tare of the self-tared A: done on 2026-10-04 (`conductor_v3.py: stage_t_selftare_ii`, before `stage_t_tare`; DEV-SELFTARE-03) (adopted 2026-10-04; DEV-SELFTARE-02; DEV-PAIRED-01).
