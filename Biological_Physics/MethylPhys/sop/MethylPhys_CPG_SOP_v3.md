# MethylPhys CPG SOP — chain v3 (neutrophils)

**Build:** development v3, 2026-10-01; frozen inputs re-checked against the code 2026-10-02. Not commissioned. Not a diagnostic test.
**Scope:** one cell, neutrophils, on Illumina EPIC v1 arrays. Other cells are added one at a time after each passes the three tests
(pure-cell precision, mixture recovery, known damage). 450K neutrophil floor: pending.
**Readings:** Met-A (arrays) and its C-score. IAM-A (sequencing) and its C-score run through a separate stage, which is in development.

## 1. Physics stated once

- Per-site entropy: H(β) = −β log₂β − (1−β) log₂(1−β), in bits.
- **Met-A** = mean over the cell's identity sites of H(β), divided by the healthy reference for that same cell type on the same platform.
  A healthy cell holds its pattern at its reference, so Met-A = 1. Loss of pattern pulls β toward 0.5, which raises H, so Met-A rises.
- **Normal = 0.95–1.05.** One gauge for every reading. No other tier is printed until it has been measured on this scale.
- **C-score** = how clustered the per-site departures are along the genome, relative to healthy (healthy = 1).

## 2. Stages and code

| stage | what it does | code | frozen input |
|---|---|---|---|
| 0 Intake | manifest, hash, controls, detection p, bead count, call rate, sex check, decision gate | `chain/stage_0_intake.py` | `Runtime Matrices/Intake/intake_thresholds_v1.json` |
| 1 Calibration | IDAT → noob β | `chain/stage_1_idat_calibration.py` | Illumina manifest |
| A Composition (whole blood only) | 8 blood groups by NNLS on 963 markers; the markers exclude the neutrophil sites | `chain/conductor_v3.py: stage_a_composition` | `../chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json` |
| M Met-A | isolated neutrophils: H̄ / own floor. Whole blood: H̄ / H̄(e), where e = Σ f_g μ_g from the purified EPIC profiles | `chain/stage_m_met_a.py`, `conductor_v3.py: stage_m_*` | `../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `../chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv` |
| MC C-score | residual z map in genomic order; variance of 50-site block means ÷ site variance ÷ healthy median | `conductor_v3.py: stage_mc_cscore` | `../chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json` |
| T Tare | same-run healthy references (same slide, else same batch): with ≥ 20 references carrying {A, f_neu, N}, A is corrected by a least-squares fit A = a + b f_neu + c N on those references (DEV-NOISE-02; isolated cells: A = a + c N); with 3–19, A_rel = A ÷ median reference A; with none, the reading is untared | `conductor_v3.py: stage_t_tare` | `../chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json` (noise index N = mean H over 48528 sites every purified blood group holds fixed) |
| Report | one HTML page plus a JSON bundle | `MethylPhys_Interface/report_v3.py` | — |

Frozen values (read from the files, never typed):
- EPIC neutrophil healthy reference 0.330263 bits (6 physical arrays, Salas GSE110554; GSE167998 re-deposits the same 6; 6000 sites; our Stage 1).
- Healthy clustering median 1.1104 (6 physical arrays, leave-one-out, block 50).

## 3. Rules the chain enforces

1. **Dominant cell only.** In whole blood, A is computed only when neutrophils are ≥ 50 %. Below that, the fraction is printed and A is withheld.
2. **Whole blood must be tared.** Untared whole-blood A carries a composition and laboratory offset, measured at −0.04 in one lab and +0.09 in another.
   The gauge state is printed only from A_rel. Isolated neutrophils are read against their own floor and need no tare.
3. **Platform match.** The floor, the profiles and the specimen must be on the same platform. Without a frozen floor for the platform, the chain refuses.
4. **Quarantine stops the run.** A Stage 0 QUARANTINE produces no reading.
5. **No population term.** Nothing in a reading depends on a cohort, a classifier or a disease label.

## 4. Running it (operator)

One specimen:
```
cd Biological_Physics/MethylPhys/chain/MethylPhys_Interface
python run_sample.py --grn S_Grn.idat --red S_Red.idat --engine v3 \
  --specimen "whole blood" --array-type EPIC_v1 --sex F --age 52 --id S001 --out S001.html
```
Isolated neutrophils: `--specimen "isolated neutrophils"`.
Tare, once ≥ 3 healthy references on the same slide have been read: add `--slide-ref-A 0.951,0.957,0.962`.
That list holds the references' untared A values.
Output: `S001.html` and `S001_bundle.json`.

A batch, with the tare done automatically: `MethylPhys/chain_tests/run_chain_acceptance.py`. Pass 1 reads every specimen. Pass 2 re-runs
each whole-blood specimen with the others in its batch as references.

## 5. What has been shown (development)

| test | result |
|---|---|
| Held-out purified neutrophils (6 physical arrays; each read against the other 5, sites re-chosen) | 0.983-1.045, SD 0.020 |
| End to end, 22 IDAT pairs | 22/22 processed. Isolated neutrophils 6/6 Normal (in-floor). Known mixtures tared 6/6 Normal. AML remission blood from another lab tared 5/5 Normal, 5 withheld as not dominant |
| Known damage, 2 % neutrophil pattern loss in mixtures (tared) | 6/6 above 1.05 (shift +0.061) |

Not yet shown: real healthy whole blood, repeat pairs, any disease. The C-score band is not set.
