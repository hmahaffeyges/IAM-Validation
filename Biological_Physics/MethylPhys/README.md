# MethylPhys — the cell instrument (chain v3)

**Status: development build, not commissioned, not a diagnostic test.** Scope: one cell type, neutrophils, on Illumina EPIC v1
arrays (Met-A) and on single-molecule bisulfite or EM-seq reads (IAM-A). Further cell types are added one at a time, each after it
passes pure-cell precision, mixture recovery and known damage.

The operator procedure is [`sop/MethylPhys_CPG_SOP_v3.md`](sop/MethylPhys_CPG_SOP_v3.md). The physics, the gauge and every frozen value
are explained in Part 4 of the book ([`docs/book`](../../docs/book)).

## What it reads

One gauge for every reading: **A = 1 is the same cell type when healthy**, in the middle; the thermal floor lies to the left; more
copy error lies to the right. Normal is 0.95–1.05. No other band is printed until it has been measured on this scale.

| reading | data | definition | status |
|---|---|---|---|
| **Met-A** | EPIC v1 array | mean H(β) over the cell's 6,000 identity sites ÷ the same quantity in purified healthy neutrophils on the same platform | reference CALIBRATED (6 physical arrays) |
| **Met-A C-score** | EPIC v1 array | clustering of the per-site residual along the genome (50-site blocks) ÷ the healthy median | baseline CALIBRATED; band not set |
| **IAM-A** | single molecules | H(ε) ÷ (P · H(ε₀)); ε = isolated copy errors on qualifying molecules; ε₀ = 1/(1+e^{E_hold/kT}) = 0.032 | form DERIVED, E_hold = 3.41 kT MEASURED, P = 1.099 MEASURED |
| **IAM-A C-score** | single molecules | per-region copy-error map | defined, not built |

Far ends of the neutrophil gauge (every identity site at a coin flip): Met-A 3.03, IAM-A 4.45. IAM-A floor: 1/P = 0.910.
Breach and the cancer region on this gauge are still to be measured.

## Stages and code

| stage | what it does | code | frozen input |
|---|---|---|---|
| 0 Intake | manifest, hashes, controls, detection p, bead count, call rate, sex check, decision gate | [`chain/stage_0_intake.py`](chain/stage_0_intake.py) | `chain/Runtime Matrices/Intake/intake_thresholds_v1.json` |
| 1 Calibration | IDAT → noob β | [`chain/stage_1_idat_calibration.py`](chain/stage_1_idat_calibration.py) | Illumina manifest |
| A Composition | whole blood only: 8 blood groups by NNLS on 963 markers (none is a neutrophil identity site) | [`chain/conductor_v3.py`](chain/conductor_v3.py) | `chain/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json` |
| M Met-A | isolated neutrophils against their own reference; whole blood against a composition-matched healthy expectation | [`chain/stage_m_met_a.py`](chain/stage_m_met_a.py) | `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json`, `chain/Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv` |
| MC C-score | residual map in genomic order | `chain/conductor_v3.py` | `chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json` |
| T Tare | against healthy references run on the same slide or batch | `chain/conductor_v3.py` | `chain/Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json` |
| Q IAM-A | copy error on single molecules | [`chain/stage_q_iam_a.py`](chain/stage_q_iam_a.py) | `chain/Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` |
| Report | one HTML page and a JSON bundle | [`chain/MethylPhys_Interface/report_v3.py`](chain/MethylPhys_Interface/report_v3.py) | — |

Rules the chain enforces: in whole blood a reading is made only when neutrophils are at least 50 %; whole blood must be tared; the
reference, the profiles and the specimen must be on the same platform; a Stage 0 quarantine stops the run; nothing in a reading
depends on a cohort, a classifier or a disease label.

## One command

```
cd Biological_Physics/MethylPhys/chain/MethylPhys_Interface
python run_sample.py --grn S_Grn.idat --red S_Red.idat --engine v3 \
  --specimen "whole blood" --array-type EPIC_v1 --sex F --age 52 --id S001 --out S001.html
```

Isolated neutrophils: `--specimen "isolated neutrophils"`. A batch with the tare done automatically:
[`chain_tests/run_chain_acceptance.py`](chain_tests/run_chain_acceptance.py).

## What has been measured (development)

| measurement | result | record |
|---|---|---|
| held-out purified neutrophils, each read against the other five | 0.983–1.045, SD 0.020 | [`chain_tests/CHAIN_V3_ACCEPTANCE_RUN3.md`](chain_tests/CHAIN_V3_ACCEPTANCE_RUN3.md) |
| constructed whole blood, ≥ 50 % neutrophils: healthy / 2 % pattern loss | 0.982–1.016 / 1.049–1.079 | [`doors/PROC_WB_NEUT_01_OUTCOME.md`](doors/PROC_WB_NEUT_01_OUTCOME.md) |
| DNMT1 blocked in three leukaemia lines (arrays) | vehicle 0.97–1.05; active drug ≥ 80 nM 1.16–1.85, methylated channel; inactive analog unchanged | [`doors/PROC_DNMT_01_PARTA_OUTCOME.md`](doors/PROC_DNMT_01_PARTA_OUTCOME.md) |
| DNMT1 blocked (single molecules, IAM-A input) | 1.65–1.97, 8/8; conversion failure unchanged | [`doors/PROC_DNMT_01_PARTB_OUTCOME.md`](doors/PROC_DNMT_01_PARTB_OUTCOME.md) |
| early-onset colorectal tumour vs the same patient's normal tissue (copy error) | higher in 6/6 pairs, median ratio 1.148 | [`doors/PROC_TUMOUR_01_OUTCOME.md`](doors/PROC_TUMOUR_01_OUTCOME.md) |

Not yet shown: real healthy whole blood, repeat draws, any disease. These are the commissioning tests now being prepared.

## Folders

| folder | contents |
|---|---|
| `chain/` | the chain code and `Runtime Matrices/` |
| `chain_tests/` | acceptance runs and their scripts |
| `doors/` | pre-registrations and outcome records (`PROC_*`, `DEV_*`), including those that failed |
| `atlas/v2/` | the methylation atlas the composition and identity sites are built from |
| `sop/` | the operator procedure (`MethylPhys_CPG_SOP_v3.md`) |
| `../RETIRED_2026-09/`, `../RETIRED_2026-10/` | earlier chain versions, class floors, reports and manuals, kept unchanged with an index |
