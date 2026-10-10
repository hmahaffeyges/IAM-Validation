# DEV-LINK-IAMA-METAA-02 — outcome (2026-10-10; development)

Scored by `doors/data/DEV_LINK_IAMA_METAA_02/score_iama_dose_02.py` (committed f531c2d before any EM-seq reading) on the six Box Run 2
session 5 runs; rows `iama_dose_02_rows.csv`. Window sealed ec6efcb (03:25 UTC) before any treated run was read.

| run | treatment | ε | IAM-A_rel | molecules read (÷ vehicle) | Met-A_rel (EPIC, same series) |
|---|---|---|---|---|---|
| SRR25322252 / 51 | vehicle | 0.0508 / 0.0502 | 1.0048 / 0.9950 | 1.001 / 0.999 | 0.9986 / 1.0014 |
| SRR25322250 / 49 | 30 nM | 0.0588 / 0.0589 | 1.1182 / 1.1206 | **0.602 / 0.623** | 1.3944 / 1.4565 |
| SRR25322248 / 47 | 300 nM | 0.0550 / 0.0529 | 1.0647 / 1.0347 | **0.336 / 0.360** | 1.6425 / 1.6995 |

**Result by the sealed rule: no result.** Both doses read fewer than 70 % of the vehicle's molecules, so both are outside the relation's range.
The test is neither passed nor failed. The two vehicle libraries agree within 0.01 (repeatability as expected).

**What the data show (development observation, after reading).** In silico, a scattered loss that raises IAM-A to 1.12 leaves 95 % of molecules
readable; measured, 60 %. A loss that removes 40 % of molecules would raise IAM-A to 1.51 if it were scattered; measured, 1.12. So decitabine
does not raise the scattered (isolated) copy error that IAM-A measures. It strips methylation from whole molecules or long stretches, which then
fall below Stage Q's 80 %-methylated rule and leave the reading. The survivors carry only a small rise. At 300 nM, IAM-A falls back (1.05/1.03)
while fewer molecules survive: the survivors are the cells' most faithful copies. That fits how decitabine works: a DNMT1 molecule trapped on the
DNA leaves the rest of that stretch uncopied, so the loss comes in runs, not as single errors. The derivation (DEV-LINK-IAMA-METAA-01) models
scattered loss, and decitabine is a different kind of damage.

**What follows.** IAM-A and Met-A see different things here, by design: IAM-A the scattered copy error of molecules still held, Met-A all loss
at identity sites. The share of molecules read is itself a measurement of run-type loss. A test of the derivation needs scattered damage
(a lowered SAM supply, DEV-SAM-LEVER-01, or a DNMT1 hypomorph), not a trapping drug. A molecule-level reading of run-type loss (share read and
run length) is a new instrument to define and simulate before any data.
