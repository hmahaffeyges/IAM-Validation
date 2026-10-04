# Chain v3 - run manifest and cold start

The chain reads raw Illumina EPIC v1 IDAT pairs (or a calibrated `cpg_id,beta` table, or single-molecule reads) and writes one report,
one bundle and one ledger row per specimen. Chain v3 is the only engine; the class-floor engine (v2) that this file used to describe was
retired on 2026-10-03 and is archived privately.

## 1. Environment

Python 3.11 with the pinned versions in [`requirements.txt`](requirements.txt): methylprep 1.7.1 (Stage 1; it needs pandas 1.5.3),
numpy 1.26.4, scipy. methylprep downloads its array manifests into `~/.methylprep_manifest_files` on first use; on a machine without
that access, point `HOME` at a folder that already holds `HumanMethylationEPIC_manifest_v2.csv.gz` (and the 450K manifest).
reportlab builds the manual; healpy, matplotlib and pyarrow are needed only by toolkit modules.

## 2. Frozen inputs (read at run time)

| file | read by | what |
|---|---|---|
| `Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json` | `stage_m_met_a.py`, `run_sample.py` (Stage 0.7b coverage) | the EPIC neutrophil floor and its 6,000 sites |
| `Runtime Matrices/Met_A_Floors/metA_floors_v1_3_loo.csv` | `stage_m_met_a.py` | leave-one-out record of the floor, printed with each reading |
| `Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json` | `conductor_v3.py` | site order, per-site H mean and SD, C-score block |
| `Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json` | `conductor_v3.py` | 963 composition markers, 8 group profiles, profiles at the neutrophil sites |
| `Runtime Matrices/Met_A_Floors/noise_sites_EPIC_v1.json` | `conductor_v3.py` | the 48,528 noise sites |
| `Runtime Matrices/Met_A_Floors/noise_gate_EPIC_v1.json` | `conductor_v3.py` | N_max |
| `Runtime Matrices/IAM_A_Positions/iama_positions_v1.json` | `stage_q_iam_a.py` | eps0 and the neutrophil position P with its pipeline |
| `Runtime Matrices/Intake/intake_thresholds_v1.json` | `stage_0_intake.py` | the intake gate's thresholds |

sha256 of each: [`FROZEN_INPUTS_v3.json`](FROZEN_INPUTS_v3.json), checked by [`release_check_v3.py`](release_check_v3.py).

## 3. Run one specimen

```
cd chain/MethylPhys_Interface
python3 run_sample.py --grn S_Grn.idat.gz --red S_Red.idat.gz --specimen "whole blood" --array-type EPIC_v1 \
  --sex F --age 52 --id S001 --out S001.html                       # --sex, --age optional (NOT_DECLARED when not given)
python3 run_sample.py ... --slide-ref-A 0.951,0.957,0.962        # whole blood: tare against >= 3 same-run healthy references
python3 run_sample.py --betas S.csv --id S001 --out S001.html      # a beta table from this chain's Stage 1 (Stage 0 does not run)
python3 run_sample.py --pat S.pat.gz --id S001 --out S001.html     # Stage Q (IAM-A) from a wgbstools .pat file
```

Exit 2 = Stage 0 quarantine (nothing scored). A batch is a loop over `run_sample.py` in two passes:
[`../chain_tests/run_chain_acceptance.py`](../chain_tests/run_chain_acceptance.py) is the batch runner the v3 development runs used.

## 4. Check before trusting a build

```
python3 chain/release_check_v3.py        # exit 0 only if every check passes; result table in kit/results/release_check.json
```
