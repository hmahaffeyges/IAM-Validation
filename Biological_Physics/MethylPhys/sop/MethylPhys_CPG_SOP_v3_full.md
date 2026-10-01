# MethylPhys CPG SOP — chain v3, neutrophils (full procedure)

**Build:** development v3. Not commissioned. Not a diagnostic test. No tier beyond Normal is printed.
**Scope:** one cell type (neutrophils), Illumina EPIC v1 arrays for Met-A; single-molecule reads (pipeline `loyfer_pat_v1`) for IAM-A. 450K neutrophil floor: pending (canon `Met_A_floor_450K_neutrophil` = null).
**Written from:** repository `hmahaffeyges/IAM-Validation`, `main` at commit `d5873bd` (2026-10-01 15:23 -0700). Code paths below are relative to `Biological_Physics/MethylPhys/`. Every number is read from a frozen file or the canon; the file and key are given beside it.
**Readings:** Met-A (arrays), Met-A C-score (arrays), A_rel (Met-A after same-run tare), IAM-A (sequencing).

---

## 1. Physics stated once

| quantity | formula | where it is computed |
|---|---|---|
| per-site entropy | H(β) = −β log₂β − (1−β) log₂(1−β), bits; β clipped to [1e-6, 1−1e-6] | `chain/stage_m_met_a.py:32-34` (`_H`) |
| Met-A, isolated neutrophils | A = mean over the identity sites of H(β) ÷ floor | `chain/stage_m_met_a.py:67` |
| Met-A, whole blood | A = mean_i H(β_i) ÷ mean_i H(e_i), e_i = Σ_g f_g μ_g,i (f: this specimen's Stage A fractions; μ: purified EPIC group profiles) | `chain/conductor_v3.py:61-62` |
| shift per 1 % loss (whole blood) | β′_i = β_i + f_NEU × 0.01 × (0.5 − μ_NEU,i); shift = mean H(β′)/mean H(e) − A | `chain/conductor_v3.py:63-64` |
| shift per 1 % loss (isolated) | β′_i = β_i + 0.01 × (0.5 − μ_NEU,i); shift = A × mean H(β′)/mean H(β) − A | `chain/conductor_v3.py:87-88` |
| entropy ceiling flag | m = mean β at the sites where μ_NEU > 0.5; `past_entropy_ceiling` = (m < 0.5) | `chain/conductor_v3.py:74-80` |
| residual map | z_i = (H(β_i) − H(ref_i)) ÷ s_i; ref_i = healthy neutrophil mean H (isolated) or H(e_i) (whole blood); s_i = shrunk healthy SD of H | `chain/conductor_v3.py:69, 89` |
| Met-A C-score | c = var(means of consecutive 50-site blocks of z, × √50) ÷ var(z); C = c ÷ healthy median clustering | `chain/conductor_v3.py:46-49, 92-98` |
| tare | A_rel = A ÷ median(A of ≥ 3 references) | `chain/conductor_v3.py:100-109` |
| reference spread | SD (ddof 1) of (reference A ÷ reference median) | `chain/conductor_v3.py:105` |
| detection limit | 2 × reference_spread_sd ÷ shift_per_1pct_loss, in % loss of the neutrophil pattern | `chain/conductor_v3.py:105` |
| IAM-A | A = H(ε) ÷ (P_cell × H(ε₀)), ε = isolated copy errors ÷ opportunities | `chain/stage_q_iam_a.py:18` |
| holding energy | E = ln((1−ε)/ε), in kT | `chain/stage_q_iam_a.py:20` |

**Normal band:** 0.95–1.05 (canon `constants.Normal_band` = [0.95, 1.05]; code `chain/stage_m_met_a.py:22` `NORMAL`; `chain/stage_q_iam_a.py:21`). State words: `Normal` (0.95 ≤ A ≤ 1.05), `above Normal`, `below Normal`.

Loss of a held pattern pulls β toward 0.5 and raises H, so Met-A rises with loss **while the cell's methylated sites stay above β = 0.5**. Past that point H falls again and A is no longer monotone in loss (flag `past_entropy_ceiling`).

---

## 2. Install

The run is one Python process: Stage 0, Stage 1 (methylprep) and the v3 stages run in the same interpreter, so one environment must satisfy methylprep.

| package | version | source |
|---|---|---|
| Python | 3.11 (3.11.16 in the environment the v3 runs used) | `atlas/v2/environment/box_env_requirements.txt` header; `doors/RUNBOOK.md:65` |
| methylprep | 1.7.1 | same |
| numpy | 1.26.4 | same |
| pandas | 1.5.3 (methylprep 1.7.1 calls `DataFrame.append`, removed in pandas 2) | same |
| scipy | 1.17.1 in that environment (NNLS, normal survival function) | `box_env_requirements.txt` |
| pytz, python-dateutil | as pinned there | `doors/RUNBOOK.md:65` |

Install (new environment):
```
python3.11 -m venv cpg_v3 && . cpg_v3/bin/activate
pip install methylprep==1.7.1 numpy==1.26.4 pandas==1.5.3 scipy pytz python-dateutil
git clone https://github.com/hmahaffeyges/IAM-Validation.git
```
Stage 1 setup (`doors/RUNBOOK.md:67-69`): `HOME` must point at a writable directory (methylprep writes `$HOME/.methylprep_manifest_files/`). First use needs network access to `https://array-manifest-files.s3.amazonaws.com/` for the Illumina manifests; offline, place `HumanMethylation450k_15017482_v3.csv.gz` and `HumanMethylationEPIC_manifest_v2.csv.gz` in that directory. If methylprep is missing, Stage 1 tries a one-time `pip install methylprep` (`chain/stage_1_idat_calibration.py:31-73`); install it by hand instead, with the pins above.

Not needed by the v3 path: matplotlib, pyarrow, the atlas files.

---

## 3. Frozen files and the values the chain reads

All under `chain/Runtime Matrices/`.

| file | key | value | read by |
|---|---|---|---|
| `Met_A_Floors/metA_floors_v1_2.json` | `version` | `1.2` | `stage_m_met_a.py:57`, bundle `floors_version` |
| | `platforms` | EPIC only, cell `neutrophils` only | `stage_m_met_a.py:55` |
| | `platforms.EPIC.neutrophils.floor` | 0.33026279581151297 bits (canon `Met_A_floor_EPIC_neutrophil` 0.330263) | `stage_m_met_a.py:67` |
| | `…n_sites` / `…sites` | 6000 identity sites (3,000 methylated, 3,000 unmethylated) | `stage_m_met_a.py:64-65` |
| | `…n_ref` / `…refs` | 12 entries = 6 physical arrays, each deposited twice (GSE110554 and GSE167998; same Sentrix IDs) | record only |
| | `rule` | SD ≤ 0.05; mean 0.75–0.95 or 0.05–0.25; ≤ 3,000 per channel; floor = mean of per-array mean H over sites (canon `Met_A_site_rule`) | record only |
| `Met_A_Floors/metA_floors_v1_2_loo.csv` | rows `platform=EPIC, cell=neutrophils`, column `A_loo` | 12 rows; printed as `n_ref` 12, `normal_fraction` 1.0, `sd` 0.0073, `min` 0.99, `max` 1.009 | `stage_m_met_a.py:44-49` (`floor_precision`) |
| `Met_A_Floors/blood_composition_EPIC_v1.json` | `version` | `blood_composition_EPIC_v1` | — |
| | `groups` | B, BASO, CD4T, CD8T, EOS, MONO, NEU, NK | `conductor_v3.py:42` |
| | `markers` / `mu_markers` | 963 composition markers (none is a neutrophil identity site) and their group mean β | `conductor_v3.py:40` |
| | `neutrophil_sites` | the same 6000 sites as the floor file | `conductor_v3.py:55, 85` |
| | `profiles_at_neutrophil_sites` | group mean β at those sites; 6 sites carry a missing value in ≥ 1 group (BASO 5, NK 2, CD4T 1, CD8T 1) and drop out of the whole-blood reading | `conductor_v3.py:55, 61` |
| | `rule` | markers: not neutrophil sites; within-group SD ≤ 0.05; margin ≥ 0.25 vs every other group; top 150/group; NNLS, sum 1 | record only |
| `Met_A_Floors/neutrophil_reference_v1.json` | `version` | `neutrophil_reference_v1`, bundle `reference_version` | `conductor_v3.py:115` |
| | `sites_ordered` | the 6000 sites in genome order | `conductor_v3.py:68, 84` |
| | `neutrophil_H_mean` | healthy neutrophil mean H per site | `conductor_v3.py:89` |
| | `neutrophil_H_sd_shrunk` | healthy per-site SD of H (shrunk) | `conductor_v3.py:69, 89` |
| | `healthy_clustering_median` | 1.1236 | `conductor_v3.py:95` |
| | `healthy_clustering_LOO` | 12 values, 0.7769–1.3585 (pairs: 6 physical arrays) | `conductor_v3.py:96-97` |
| `IAM_A_Positions/iama_positions_v1.json` | `cells.neutrophils.P` | 1.099 (canon `P_neutrophil_IAM_A`) | `stage_q_iam_a.py:12, 18` |
| | `cells.neutrophils.pipeline` | `loyfer_pat_v1` | `stage_q_iam_a.py:14` |
| | `…P_range`, `…cv_across_donors`, `…n_donors` | [1.0841, 1.1079], 0.0118, 3 | record only |
| `Intake/intake_thresholds_v1.json` | `detection.pass_fraction` | 0.99 | `stage_0_intake.py:580` |
| | `detection.borderline_fraction` | 0.93 | `stage_0_intake.py:581` |
| | `call_rate.proceed_at_or_above` | 0.98 | `stage_0_intake.py:682` |
| | `call_rate.quarantine_below` | 0.93 | `stage_0_intake.py:683` |
| | `bead.pass_fraction`, `bisulfite_conversion.min` | 0.995, 0.95 (the code uses its own equal constants, `stage_0_intake.py:649, 490`) | — |

Code constants (not in a frozen file): `MIN_READ_FRACTION` = 0.20 (`conductor_v3.py:72`); identity-site coverage ≥ 0.9 × 6000 for isolated readings (`stage_m_met_a.py:65`); C-score block 50 sites, ≥ 10 blocks (`conductor_v3.py:46-48`); ≥ 3 tare references (`conductor_v3.py:103`); ε₀ = 0.0320 (`stage_q_iam_a.py:8`; canon `eps0_meth` 0.032 = 1/(1+exp(φM)), φ 0.1628, M 20.94); IAM-A opportunities ≥ 100,000, half-readings when a half has > 50,000 (`stage_q_iam_a.py:17, 19`); EPIC platform = calibrated β vector longer than 700,000 probes (`stage_m_met_a.py:38`).

---

## 4. Run order

### 4.1 One specimen
Run from `chain/MethylPhys_Interface/` (the script puts `chain/` on the path itself, `run_sample.py:37-38`).
```
python run_sample.py --grn S_Grn.idat.gz --red S_Red.idat.gz --engine v3 \
  --specimen "whole blood" --array-type EPIC_v1 --sex F --age 52 --id S001 --out S001.html
```
- `--engine v3` is the default (`run_sample.py:268`).
- `--sex` (F/M) and `--age` are required by Stage 0 (manifest fields). Without them the run quarantines.
- `--array-type`: if omitted, the chain uses the type read from the IDAT header (`run_sample.py:293`). Give it only when you know it; a family mismatch quarantines.
- `--specimen`: `isolated neutrophils`, `sorted neutrophils`, `purified neutrophils` or `neutrophils` select the isolated path (`conductor_v3.py:29`); **any other string is read as whole blood**.
- Optional custody: `--patient-id <hashed id>`, `--intake-log <path.jsonl>`, `--manifest-dir <dir>`.
- Already-calibrated β (no IDAT): `--betas S.csv` (two columns, `cpg_id,beta`). Stage 0 and Stage 1 do not run; the report prints intake `not run`.
- IDAT pair without custody record (lab-made DNA mixtures): add `--no-intake`; the bundle records `intake_skipped: true`.

Outputs: `S001.html` and `S001_bundle.json` (or the path given by `--bundle`). The console ends with `S001: neutrophil Met-A <A> (<state or reason>) | C <C> | tare <A_rel>`. Exit code 2 = QUARANTINE (no report, no bundle).

### 4.2 A batch with references (two-pass tare)
Whole-blood Met-A is printed as a gauge state only after the tare. Isolated readings may also be tared (Stage T runs whenever references are supplied).
1. **Plan the run.** Put ≥ 3 healthy reference specimens of the same specimen type on the same slide as the specimens (else in the same batch), processed the same way (same extraction, bisulfite batch, scanner, Stage 1).
2. **Pass 1.** Run every array (references and specimens) as in 4.1 without `--slide-ref-A`. Read `met_a.A` from each bundle.
3. **Choose the reference set per specimen.** The healthy references on the same slide (≥ 3), excluding the specimen itself; if fewer than 3, the healthy references of the same batch. Use only references whose `met_a.A` is a number. The code does not check that a reference is healthy, on the same slide or of the same specimen type: the operator is responsible.
4. **Pass 2.** Re-run each specimen with `--slide-ref-A a1,a2,a3,…` (the references' untared A, comma-separated). Pass 2 repeats Stage 0 and Stage 1. **Use a different `--intake-log` (or none) for pass 2**: the same bytes logged twice in one intake log stop the run as `RE_TRANSMISSION_DETECTED`.
5. Read `tare.A_rel`, `tare.state`, `tare.detection_limit_pct_loss` from the pass-2 bundle.

Batch scripts in the repository (`chain_tests/run_chain_acceptance.py`, `chain_tests/chain_batch.py`) carry the author's machine paths (`/home/ubuntu/...`) and their own reference selection; use them as worked examples of the two passes, not as the operator tool.

---

## 5. Stage by stage

### Stage 0 — intake (`chain/stage_0_intake.py`, called from `run_sample.py:274-413`)
**Purpose:** refuse a specimen the chain cannot vouch for, before any reading.
**Inputs:** the Grn/Red IDAT pair; the manifest entry `run_sample.py` builds: `sentrix_id` (from the file name, pattern `(\d{9,12})[_-](R0\dC0\d)`, `run_sample.py:283`), `array_type`, `patient_id` (`--patient-id`, else `--id`, else the file-name prefix; hashed to 32 hex characters of SHA-256 when it is not already a ≥ 16-character alphanumeric token, `run_sample.py:285-287`), `intake_date` (today), `substrate` (`--specimen` with spaces → `_`), `declared_sex` (`--sex`), `declared_chronological_age` (`--age`).
**Order:** 0.1 → 0.2 → 0.3; a status beginning `QUARANTINE` stops the run here (exit 2). Then the QC hand-off and 0.4–0.8; then Stage 1; then the post-Stage-1 re-check; then 0.7b and the 0.9 gate; a `QUARANTINE` verdict stops the run before any v3 stage (exit 2).

| step | what it decides (as coded) | threshold / rule | state written | effect |
|---|---|---|---|---|
| 0.1 arrival (`:110`) | 7 manifest fields present: sentrix_id, array_type, patient_id, intake_date, substrate, declared_sex, declared_chronological_age | non-empty | `QUARANTINE_INCOMPLETE_MANIFEST` (flag `INCOMPLETE_MANIFEST:<fields>`) | stop |
| | array type token | HM450K, EPIC_v1, EPIC_v2 | `QUARANTINE_INCOMPLETE_MANIFEST` (flag `UNKNOWN_ARRAY_TYPE`) | stop |
| | both files exist | — | `QUARANTINE_MISSING_CHANNEL` | stop |
| | file size | ≥ 1,000,000 bytes each | `QUARANTINE_TRUNCATED_UPLOAD` | stop |
| | header type vs declared (nSNPsRead: < 800,000 HM450K; < 1,080,000 EPIC_v1; else EPIC_v2) | 450K vs EPIC family must agree | `QUARANTINE_ARRAY_TYPE_MISMATCH`; EPIC_v1/v2 difference → flag `ARRAY_SUBTYPE_NOTE` only; unreadable header → flag `IDAT_HEADER_UNREADABLE` | stop / note |
| | same Sentrix ID in the intake log within 24 h | — | flag `DUPLICATE_INTAKE_24H_SOFTWARN` | continue |
| | all passed | — | `STAGED` | continue |
| 0.2 manifest (`:358`) | patient_id is not cleartext (no space, no `@`, ≥ 16 alphanumeric after removing `_`/`-`) | — | `QUARANTINE_MANIFEST_INVALID` (flag `CLEARTEXT_PII`) | stop |
| | core fields present | — | `QUARANTINE_MANIFEST_INVALID` (flag `MANIFEST_INVALID:<fields>`) | stop |
| | writes `patient_manifest_<sample_run_id>.json` to `--manifest-dir` | — | `MANIFEST_COMPLETE` | continue |
| 0.3 integrity (`:438`) | SHA-256 of both files vs earlier rows for this Sentrix ID in `--intake-log` | identical pair → hold | `integrity_status` = `RE_TRANSMISSION_DETECTED` (hard fail at 0.9) | quarantined at 0.9 |
| | | different hashes | flag `LIKELY_RE_RUN_FRESH_ARRAY`; `INTEGRITY_OK` | continue |
| hand-off (`stage_0_1_qc_handoff.decode_qc_inputs`) | decodes controls, per-probe intensity, negative-control background (median, MAD × 1.4826), bead counts, chrX/chrY intensity | decoder exception | `QUARANTINE_CORRUPT_IDAT` (flag `IDAT_DECODE_FAILED:<error>`) | stop (`run_sample.py:333-349`) |
| | | module missing (ImportError) | 0.4–0.8 left unmeasured | detection/call rate then deferred → quarantine at 0.9 |
| 0.4 controls (`:505`, `:545`) | bisulfite conversion I/(I+II); hybridisation high/low; extension meth/unmeth | BS ≥ 0.95; hyb ratio ≥ 2.0; extension ratio in [0.2, 5.0] | `ctrl_qc` = `PASS`, `FAIL_<flags>` (hard), or `PROVISIONAL_BS_THRESHOLD_UNCALIBRATED` when BS is the only failing check (recorded deferred, not refused; `BS_THRESHOLD_CALIBRATED = False`) | — |
| 0.5 detection (`:579`) | detection p = 1 − Φ((I − μ_bg)/σ_bg) per probe; detected = p ≤ 0.01 | detected fraction > 0.99 `PASS`; ≥ 0.93 `DETECTION_BORDERLINE`; else `FAIL_LOW_DETECTION` | `detection_qc`, `pct_probes_detected_p_le_01` | — |
| 0.6 beads (`:648`) | fraction of probes with ≥ 3 beads | ≥ 0.995 `PASS`, else `WARN_LOW_BEAD_COUNT` (borderline) | `bead_qc`, `pct_probes_bead_count_ge_3` | — |
| 0.7 call rate (`:682`) | fraction passing detection and beads | ≥ 0.98 `PASS`; ≥ 0.93 `CALL_RATE_BORDERLINE`; else `CALL_RATE_FAIL` | `call_rate`, `call_rate_status` | — |
| 0.8 sex (`:757`) | predicted F if log2(Y median) − log2(X median) < −2.0, else M | must equal declared F/M | `sex_check` = `PASS` / `MISMATCH` (hard), `predicted_sex` | — |
| post-Stage-1 re-check (`run_sample.py:372-384`) | **overwrites** 0.4, 0.5, 0.7 with Stage 1's numbers: controls from methylprep control medians; detection = poobah p ≤ 0.05 against the array's own negatives; call rate = poobah mask × beads all-pass | same 0.99/0.93 and 0.98/0.93 lines | `ctrl_qc`, `ctrl_metrics`, `detection_qc`, `call_rate_status`; flag `BEAD_COUNT_NOT_EXTRACTED`; no poobah column → call rate `DEFERRED_PENDING_STAGE1_DECODER` | — |
| 0.7b platform (`:728`) | platform tag; 450K coverage gate | EPIC → `NA_EPIC`; 450K: coverage of the neutrophil identity sites ≥ 0.80 | `platform_tag`, `hm450_coverage_gate` | — |
| 0.9 gate (`:794`) | hard: any `QUARANTINE_*` status, integrity ≠ OK, `ctrl_qc` FAIL, `FAIL_LOW_DETECTION`, `CALL_RATE_FAIL`, coverage FAIL, sex `MISMATCH`, deferred detection or call rate (`intake_deferred:…`). Borderline: detection, bead, call rate. Deferred (recorded): provisional bisulfite, bead, sex, controls when not decoded | any hard → `QUARANTINE`; else any borderline → `PROCEED_WITH_PENALTY`; else `PROCEED` | `stage0_verdict`, `stage0_hard_fail`, `stage0_borderline`, `stage0_deferred_qc` | `QUARANTINE` → stop, exit 2 |

**Outputs:** bundle `intake` (the whole record, including `flags`, `grn_sha256`, `red_sha256`, `sample_run_id`, the QC fields above); rows appended to `--intake-log` (arrival, integrity, verdict) when given.
**Operator:** see the fault table, section 7.

### Stage 1 — IDAT calibration (`chain/stage_1_idat_calibration.py:88-186`, called at `run_sample.py:365`)
**Purpose:** turn this array's raw intensities into calibrated β using only its own controls.
**Inputs:** the IDAT pair. Array type from the bead-address count.
**Method as implemented:** `methylprep.run_pipeline(betas=True, export=True, save_control=True, poobah=True)`: noob background, dye-bias and probe-type normalisation, per sample. Keeps `cg` probes with poobah p ≤ 0.05 (`:142-144`); probes at background are removed before any stage reads β. Also extracts control-probe medians and the SNP-probe noise model.
**Frozen inputs:** Illumina manifests via methylprep (section 2).
**Outputs:** β vector (cg probes); bundle `stage1` = {`detection` (`detection_available`, `n_probes`, `n_detected`, `pct_detected`, `n_masked`), `n_cpgs`, `pipeline`}. Console: `Stage 1: <n_detected> of <n_probes> probes detected (<%>); <n_masked> at background removed`.
**Failure:** methylprep import or manifest download failure stops the run with the error (see 3b items in section 7).

### Platform check (`conductor_v3.py:114-117`)
**Rule:** calibrated β vector > 700,000 probes → `EPIC`; otherwise `450K` → bundle `refusal` = `no frozen neutrophil floor for this platform yet (EPIC only; 450K pending)`; no reading. The report prints the header and the refusal; Stage M/MC/T are absent.
**Operator:** none for 450K (floor pending). A `--betas` table with ≤ 700,000 rows is classed 450K: supply the full calibrated vector.

### Stage A — composition (whole blood only; `conductor_v3.py:36-44`)
**Purpose:** this specimen's blood-cell fractions, on the same platform and reference as the expectation profiles.
**Inputs:** β at the 963 `markers` of `blood_composition_EPIC_v1.json`.
**Formula:** NNLS of β(markers) on `mu_markers` (8 groups, measured markers only), f ← f/Σf.
**Outputs:** bundle `composition` = {`stage` "A", `method` "EPIC blood NNLS (blood_composition_EPIC_v1)", `fractions` {B, BASO, CD4T, CD8T, EOS, MONO, NEU, NK}, `n_markers_used`, `residual_mae`}. Isolated specimens: `composition` = {`stage` "A", `note` "isolated neutrophils: composition not solved"}.
**Rules:** none in code (no minimum marker count). **Operator:** check `n_markers_used` = 963 for an EPIC v1 array; a value near 0 means the probe names do not match EPIC v1 (e.g. EPIC v2 suffixed names) and every fraction reads 0.

### Stage M — Met-A, isolated neutrophils (`stage_m_met_a.read`, `conductor_v3.py:82-90`)
**Purpose:** the neutrophils' held pattern against their own healthy floor.
**Inputs:** β at the 6000 identity sites; `floor` 0.33026279581151297; `metA_floors_v1_2_loo.csv`; `profiles_at_neutrophil_sites.NEU`.
**Formula:** A = mean H(β at measured identity sites) ÷ floor.
**Rules (in order):** cell must be `neutrophils` (scope); platform floor must exist; measured identity sites ≥ 0.9 × 6000 = 5400, else A withheld with `only <n> of 6000 identity sites measured`.
**Outputs:** bundle `met_a` = {`stage` "M", `reading` "Met-A", `cell`, `platform`, `specimen`, `fraction` null, `A`, `band` "Normal 0.95-1.05", `build`, `floors_version`, `n_sites`, `floor` (5 decimals), `state` (Normal / above Normal / below Normal), `floor_precision` {`n_ref`, `normal_fraction`, `sd`, `min`, `max`}, `methylated_sites_mean_beta`, `past_entropy_ceiling`, `shift_per_1pct_loss`} or `reason` when withheld.
**Operator:** read `state` from A unless the array's lab is not known to be as clean as the floor arrays; then run same-run references and read `tare.A_rel` (Stage T runs on isolated readings too). If `past_entropy_ceiling` is true, read `methylated_sites_mean_beta`, not A.

### Stage M — Met-A, whole blood (`conductor_v3.py:51-70`)
**Purpose:** the neutrophils' held pattern inside whole blood, against the healthy expectation for this specimen's own composition.
**Inputs:** β at the 6000 `neutrophil_sites`; Stage A fractions; `profiles_at_neutrophil_sites` (8 groups).
**Formula:** e_i = Σ_g f_g μ_g,i; A = mean H(β_i) ÷ mean H(e_i) over sites where both β and e exist (≤ 5994 sites).
**Rules:** neutrophil fraction f_NEU < 0.20 (`MIN_READ_FRACTION`) → A withheld, `reason` = `neutrophil fraction <f> < 0.2: fraction reported, A withheld`; Met-A C-score and tare then return no value. The untared value carries a composition and laboratory offset; it is printed as a number with `state` = `untared: read A_rel (Stage T)`; the gauge state comes only from Stage T.
**Outputs:** bundle `met_a` = {`stage`, `reading`, `cell`, `specimen` "whole blood", `fraction`, `build`, `band` "Normal 0.95-1.05 (after tare)", `A`, `shift_per_1pct_loss`, `methylated_sites_mean_beta`, `past_entropy_ceiling`, `n_sites`, `expectation`, `state`} or `reason`.
**Operator:** run pass 2 (section 4.2). Low fraction: nothing to fix; the fraction is the result. `shift_per_1pct_loss` shrinks with f_NEU; read it with the detection limit.

### Stage MC — Met-A C-score (`conductor_v3.py:92-98`)
**Purpose:** how clustered the per-site departures are along the genome, healthy = 1.
**Inputs:** residual z at `sites_ordered` (genome order), `neutrophil_H_sd_shrunk`, `neutrophil_H_mean` (isolated) or H(e) (blood); `healthy_clustering_median` 1.1236; `healthy_clustering_LOO`.
**Formula:** section 1. Needs ≥ 10 blocks of 50 measured sites; otherwise, or when Stage M withheld A, `C` = null with `reason` = `no residual map`.
**Outputs:** bundle `met_a_cscore` = {`stage` "MC", `reading` "Met-A C-score", `C`, `clustering`, `healthy_baseline`, `n_healthy_baseline` (12), `healthy_range` [0.6914, 1.2091] (= LOO min and max ÷ median), `status` "development: healthy band not yet set", `frac_abs_z_gt3`}.
**Rules:** no band. **Operator:** record C; do not interpret against a band. In whole blood the residual includes composition error.

### Stage T — tare (`conductor_v3.py:100-109`)
**Purpose:** remove the composition and laboratory offset by reading the specimen against healthy references run the same way; state the smallest loss this specimen could show.
**Inputs:** `met_a.A`; `--slide-ref-A` list (untared A of the references); `met_a.shift_per_1pct_loss`.
**Formula:** section 1.
**Rules:** A null → `reason` `no A`; fewer than 3 numeric references → `A_rel` null, `reason` `untared: <n> same-slide reference arrays (>= 3 required)`. Detection limit null when the shift is missing or ≤ 0.
**Outputs:** bundle `tare` = {`stage` "T", `A_rel`, `slide_reference_median`, `n_refs`, `reference_spread_sd`, `detection_limit_pct_loss`, `detection_note`, `state`}.
**Operator:** with 3 references the spread is itself imprecise; use more references where the slide allows. A detection limit above the change you need to see means this specimen cannot show it.

### Report (`chain/MethylPhys_Interface/report_v3.py`, written by `run_sample.py:415-428`)
**Purpose:** one HTML page plus the JSON bundle.
**Outputs:** `<out>.html`; `<out stem>_bundle.json` (always written on the v3 path); bundle top level: `build`, `specimen`, `platform`, `scope`, `floors_version`, `reference_version`, [`refusal`], `composition`, `met_a`, `met_a_cscore`, `tare`, `withheld`, `intake`, `intake_skipped`, `sample_id`, [`stage1`].
On the v3 path `--ledger`, `--no-bundle`, `--covariate(s)`, `--lab`, `--lab-zero`, `--pipeline` and `--atlas-v2` have no effect.

### Stage Q — IAM-A, sequencing (`chain/stage_q_iam_a.py`; development; not called by `run_sample.py`)
**Purpose:** the neutrophils' per-molecule copy error against the physics floor at the neutrophil's frozen position.
**Inputs:** a per-site table with columns `pos, opp_A, err_A, opp_B, err_B` (A/B = the two run halves); the cell (`neutrophils`); the read-level pipeline name; optional `mask` (positions to drop). Opportunity = an interior CpG call on a qualifying molecule (≥ 6 CpG calls, ≥ 80 % methylated); error = an unmethylated interior call with both neighbouring calls methylated (`stage_q_iam_a.py:4-5`; extractor definition `Biological_Physics/Salmonid/PROC_SALMON_01/extract.py:3-4`).
**Formula:** ε = (err_A + err_B) ÷ (opp_A + opp_B); IAM-A = H(ε) ÷ (P × H(ε₀)), P = 1.099 (`iama_positions_v1.json` `cells.neutrophils.P`), ε₀ = 0.0320; half-readings per half with > 50,000 opportunities; E_kT = ln((1−ε)/ε).
**Rules (refusals, `A` null):** no frozen position for the cell → `no frozen IAM-A position for <cell>`; pipeline ≠ `loyfer_pat_v1` → `position for neutrophils was measured on loyfer_pat_v1, not <pipeline>: measure P on healthy neutrophils with this pipeline first`; opportunities < 100,000 → `too few opportunities (<n> < 100000)`.
**Outputs:** {`stage` "Q", `reading` "IAM-A", `cell`, `pipeline`, `build`, `A`, `eps`, `P`, `E_kT`, `halves` {A, B}, `state`, `opportunities`} or `refusal`.
**Run:**
```
cd chain && python -c "import pandas as pd, json, stage_q_iam_a as Q; \
print(json.dumps(Q.read(pd.read_csv('sites.csv'), cell='neutrophils', pipeline='loyfer_pat_v1'), indent=1))"
```
**Operator:** always pass `pipeline=` explicitly and truthfully (the default is `loyfer_pat_v1`; the code cannot check what produced the table). Halves should agree; a large difference is a run artefact, not a reading.

---

## 6. Reading the report (top to bottom)

1. **Banner:** build string, "Not a diagnostic test."
2. **Header:** specimen, platform, `floors` version (1.2), `reference` version (neutrophil_reference_v1).
3. **Stage 0 intake:** `verdict` (PROCEED / PROCEED_WITH_PENALTY; `not run` for `--betas` or `--no-intake`), call rate status, flags (first 300 characters; the full list is in the bundle `intake.flags`).
4. **Stage A composition:** groups ≥ 1 %, highest first. Isolated: the note.
5. **Stage M Met-A — neutrophils:** gauge bar 0.80–1.30 with Normal 0.95–1.05 shaded; the marker is `A_rel` for whole blood and `A` for isolated specimens (values outside 0.80–1.30 sit at the edge). Then `A` with state or reason, neutrophil fraction, sites, expectation (`own floor` for isolated).
6. **Stage T slide tare:** `A_rel` with state or reason; detection limit (% loss of the neutrophil pattern) and reference spread; methylated-site mean β, with **"past the entropy ceiling: … read beta, not A"** when flagged.
7. **Stage MC Met-A C-score:** C, the healthy held-out range, status.
8. **Withheld:** tier lines beyond Normal; other cell types.
9. **bundle** (collapsed): the bundle without `intake`, first 20,000 characters.

What to read: whole blood → `A_rel` and its `state`, with `detection_limit_pct_loss`; isolated → `A` and `state` (or `A_rel` when tared); always the ceiling flag. An untared whole-blood A is a number, not a state.

---

## 7. Fault table — every state and what the operator does

| printed | cause | action |
|---|---|---|
| `QUARANTINE_INCOMPLETE_MANIFEST` | a manifest field missing (most often `--sex` or `--age`) or an unknown array-type token | supply `--sex F|M` and `--age`; array type must be HM450K, EPIC_v1 or EPIC_v2 |
| `QUARANTINE_MANIFEST_INVALID` (`CLEARTEXT_PII`) | `--patient-id` contains a space or `@` | pass a hashed id, or omit it and let the runner hash `--id` |
| `QUARANTINE_MISSING_CHANNEL` | Grn or Red file absent | supply both files |
| `QUARANTINE_TRUNCATED_UPLOAD` | a file < 1,000,000 bytes | re-fetch the file |
| `QUARANTINE_ARRAY_TYPE_MISMATCH` | declared type and header disagree (450K vs EPIC) | omit `--array-type` and let the header decide, or correct it |
| `QUARANTINE_CORRUPT_IDAT` | the decoder failed on the file (e.g. truncated inside the gzip stream) | re-fetch; a file can pass the size floor and still be truncated |
| `RE_TRANSMISSION_DETECTED` → QUARANTINE (`integrity`) | these exact bytes are already in this intake log | for a planned re-run (tare pass 2) use another `--intake-log` or none; if not planned, find who submitted the first copy |
| `FAIL_HYB_FAIL` / `FAIL_EXT_FAIL` → QUARANTINE (`ctrl_qc`) | hybridisation or extension controls outside the line | re-hybridise / re-run the array |
| `PROVISIONAL_BS_THRESHOLD_UNCALIBRATED` | bisulfite value below 0.95, threshold not calibrated | none; value recorded, not refused |
| `FAIL_LOW_DETECTION` / `CALL_RATE_FAIL` → QUARANTINE | too many probes at background | specimen or hybridisation problem; re-run the specimen |
| `DETECTION_BORDERLINE` / `CALL_RATE_BORDERLINE` / `WARN_LOW_BEAD_COUNT` → PROCEED_WITH_PENALTY | between the lines | reading proceeds; note the penalty with the result; many warnings on one plate → raise with the core facility |
| `intake_deferred:detection` / `call_rate` → QUARANTINE | detection or call rate could not be measured (decoder module missing, no poobah column) | fix the installation (section 2) and re-run |
| sex `MISMATCH` → QUARANTINE | chrX/chrY call ≠ declared sex, or declared sex not F/M | check the paperwork and the sample identity; do not override |
| `refusal: no frozen neutrophil floor for this platform yet (EPIC only; 450K pending)` | not EPIC, or β vector ≤ 700,000 probes | none for 450K; for `--betas` supply the full vector |
| `only <n> of 6000 identity sites measured` | isolated specimen, < 5400 identity sites after detection masking | low-quality or wrong-platform array; re-run |
| `neutrophil fraction <f> < 0.2: fraction reported, A withheld` | neutrophils < 20 % of the specimen (or probe names not EPIC v1: check `n_markers_used`) | none; the fraction is the result |
| `untared: <n> same-slide reference arrays (>= 3 required)` | fewer than 3 references supplied | run pass 2 with `--slide-ref-A` |
| `C` null, `no residual map` | A withheld, or < 500 residual sites | none |
| `past_entropy_ceiling: true` | methylated sites average below β 0.5 | read `methylated_sites_mean_beta`; A understates loss |
| methylprep import error / manifest download hang | environment or `HOME` not writable / no network | section 2 |
| `ModuleNotFoundError` on a stage module | script moved | run `run_sample.py` from `chain/MethylPhys_Interface/` |
| Stage Q `refusal` lines | section 5, Stage Q | measure P for the pipeline, or supply ≥ 100,000 opportunities |

---

## 8. Validation record (development measurements)

Measurements made on the chain while it is in development. Nothing here is a commissioning result.

### 8.1 Records in `chain_tests/`
| record | what was measured | what it showed |
|---|---|---|
| `CHAIN_V3_ACCEPTANCE_RUN3.md` (+ `chain_acceptance.csv`, `run_chain_acceptance.py`) | 22 IDAT pairs through `run_sample.py --engine v3`, two passes | 22/22 processed. Isolated purified neutrophils (these 6 arrays are in the floor): A 0.994–1.006. Six DNA mixtures (neutrophils 58–70 %): untared 0.943–0.968, tared 0.987–1.021. Ten remission bloods from a second lab: untared 1.073–1.115 on 5, tared 0.986–1.032 against the other bloods of that batch; 5 had A withheld at fractions 0.058–0.473 under the read-fraction line of that build (0.50); the four at ≥ 0.20 are read by the current code and need a re-run. C: isolated 0.69–1.21, whole blood 0.78–1.49. |
| `WHOLE_BLOOD_COMPOSITION_DEV.md` (+ `blood_comp.py`, `selfconsist.py`) | 6 DNA mixtures (≥ 50 % neutrophils), composition by EPIC NNLS with profiles from the other study | Untared A 0.943–0.968 (offset about −0.05, SD 0.009); tared A_rel 0.992–1.021 (SD 0.010); a simulated 2 % neutrophil pattern loss read 1.052–1.090 tared (shift +0.061); re-fitting the fraction on the neutrophil sites absorbs the loss. |
| `IAMA_FLOOR_COMPARISON.md` (+ `iama_floor.py`, `iama_floor_granulocytes.csv`) | 3 granulocyte donors, read-level .pat, 16–20 M opportunities each; three floors compared | On ε₀ alone healthy donors read 1.084–1.127; with the frozen position P (1.099): 0.978–1.040 (P measured on the same 3 donors, leave-one-out). Odd/even repeat ≤ 0.002. Simulated 2 % copy damage: 1.285–1.346. The bare floor reads healthy blood cells 0.70–0.79 on a second pipeline: the reading depends on the pipeline, hence one P per pipeline. |
| `neut_ref.py`, `blood_comp.py` | builders of `neutrophil_reference_v1.json` and `blood_composition_EPIC_v1.json` | — |
| `chain_batch.py` | batch runner for the neutrophil test series below | — |

### 8.2 Development records in `doors/` that bear on v3
| record | what was measured | what it showed |
|---|---|---|
| `DEV_LOWFRAC_01_OUTCOME.md` | 656 whole bloods, Met-A at every neutrophil fraction; healthy spread and the shift of a simulated 2 % loss by fraction | Healthy SD stays 0.020–0.024 from 0.40 to 1.00; the shift falls with fraction (0.033 at 0.40–0.50, 0.064 at 0.70–1.00). Below 0.40: 5 healthy arrays. Led to the per-specimen detection limit. |
| `PROC_DNMT_01_PARTA_OUTCOME.md` | 51 EPIC arrays, 3 cell lines, DNMT1 inhibitor dose and time series, each line against its own 4 vehicle arrays | Vehicle 0.968–1.048, inactive analog 1.002–1.032; active drug ≥ 80 nM 1.16–1.85, second active compound 1.73–1.81; change in the methylated channel. A saturates near 1/H(floor) once methylated sites reach β ≈ 0.5 and falls past it. Led to the ceiling flag. |
| `DEV_NOISE_01_OUTCOME.md` | noise index N (mean H at 48,528 invariant EPIC sites) on floor arrays and a second lab | Floor arrays N 0.1223–0.1489; second-lab arrays up to 0.2434; Met-A follows N (ρ 0.79–0.83). |
| `DEV_NOISE_02_OUTCOME.md` | 495 whole bloods from one lab | Untared A tracks neutrophil fraction and N; dividing by the expectation for each array's own (fraction, N), fitted on the other healthy arrays, gives healthy SD 0.022. Fitted after looking; needs a held-out lab. |
| `PROC_NEUT_TEST_01_OUTCOME.md`, `…_T2_OUTCOME.md` | FACS-counted bloods, isolated neutrophils from a second lab, technical replicates, 570 whole bloods | Neutrophil fraction within 0.035 of FACS (median). Isolated second-lab neutrophils read 0.86–1.26 against the floor; the spread follows array noise, not purity. Tared healthy bloods in one lab spread SD 0.052; the tared reading still rises about +0.12 A per unit fraction. Replicate set: neutrophils 0.30–0.56, not read on that build. |
| `PROC_WB_NEUT_01_OUTCOME.md` | 6 Salas mixtures with known fractions | Known-fraction expectation: 0.982–1.016; simulated 2 % loss 1.049–1.079; the neutrophil floor alone reads the same mixtures 1.062–1.118. |
| `PROC_PREDX_NEUT_01_OUTCOME.md` | 845 450K arrays on a development 450K neutrophil floor | Outside current scope (EPIC only). The reading differed by donor sex by 0.045, not explained by purified-neutrophil differences (0.008); centre/plate and composition residual remain open. |

---

## 9. Pending changes (not yet in the code)

1. Array noise index N from 48,528 blood-invariant EPIC sites (`doors/data/noise_sites_EPIC_v1.json`), reported at Stage 0, with a gate (proposed: state withheld above the floor arrays' range unless tared).
2. Batch calibration of the whole-blood expectation on (fraction, N) from same-run healthy references (needs ≥ 20 references).
3. Fraction-dependent precision rule: report A only where the shift at the specimen's own fraction is ≥ 2 × the healthy spread, in place of the fixed 0.20 line.
4. Bisulfite-conversion threshold calibrated at intake (`BS_THRESHOLD_CALIBRATED` set from healthy arrays).
5. Tare required for isolated neutrophils as well as whole blood; report gauge then drawn from A_rel for both.
6. Molecule-assignment IAM-A (per-molecule cell assignment before the copy-error count).
7. IAM-A C-score (per-region copy-error map against the floor).
8. A per-site extractor for the `loyfer_pat_v1` pipeline inside `chain/`, and a Stage Q entry in the runner and report.
9. 450K neutrophil floor (purified 450K neutrophils, GSE88824), with donor sex stated.
10. Floor and C-score baseline from independent physical arrays (the 12 entries are 6 arrays deposited twice).
