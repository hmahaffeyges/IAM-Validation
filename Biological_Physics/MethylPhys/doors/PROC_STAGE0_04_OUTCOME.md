# PROC-STAGE0-04 - outcome: Stage 0's control-probe limits set from healthy arrays of three laboratories (2026-10-21)

**Not a milestone:** an intake limit set on healthy arrays, as the procedure requires before a gate may stop a sample; no reading changes.

**The instrument's job:** a healthy array passes Stage 0's control-probe checks, and an array whose chemistry failed is stopped.

**What was read.** 337 healthy whole-blood EPIC v1 arrays from three laboratories ([`stage0_healthy_epic_samples.csv`](data/PROC_STAGE0_04/stage0_healthy_epic_samples.csv):
GSE318173 278, GSE315366 35, GSE184269 24), read through Stage 0 on GitHub's machines (exploration run 39611932552) after the two fixes of
2026-10-20: each BS Conversion I pair read in its own color channel, and the hybridization check on the spread between the high and low
controls with the medium one between them. Per-array values: [`stage0_healthy_epic.csv`](data/PROC_STAGE0_04/stage0_healthy_epic.csv); the table below is written by [`stage0_limits_summary.py`](data/PROC_STAGE0_04/stage0_limits_summary.py) ([`stage0_limits_summary.txt`](data/PROC_STAGE0_04/stage0_limits_summary.txt)).

| check | GSE318173 (min / median / max) | GSE315366 | GSE184269 | limit set | healthy arrays stopped |
|---|---|---|---|---|---|
| bisulfite conversion | 0.8846 / 0.9692 / 0.9804 | 0.8991 / 0.9277 / 0.9410 | 0.5762 / 0.9559 / 0.9746 | at least 0.80 | 1 of 337 |
| hybridization spread | 3.010 / 3.464 / 3.891 | 2.828 / 3.006 / 3.140 | 2.690 / 3.115 / 3.645 | at least 2.0 | 0 of 337 |
| extension balance (green over red) | 0.450 / 0.946 / 1.174 | 0.929 / 1.009 / 1.218 | 0.652 / 0.879 / 1.186 | 0.2 to 5.0 | 0 of 337 |

**The rule** (pre-registration, rules 1 and 2): set from healthy arrays only, from more than one laboratory, as a percentile of the healthy
distribution; each laboratory counted equally so the largest set does not decide it alone. **The author chose 0.80** for the bisulfite line
(2026-10-21; rule: "The author decides the number"): the 2.5th percentile with each laboratory counted equally is 0.8045. It stops one healthy
array, GSM5583015 (0.5762), which reads low on every pair in its own channel: a real low conversion. Four other arrays of the same slide
(200607130067) read 0.80 to 0.84 and pass. The hybridization and extension lines already stopped no healthy array and are kept as calibrated.

**In the chain** (rule 3): the limits are in `chain/Runtime Matrices/Intake/intake_thresholds_v2.json` (frozen, FROZEN_INPUTS_v4.json), which
Stage 0 reads; `BS_THRESHOLD_CALIBRATED` follows the file and is True; `intake_thresholds_v1.json` stays as the record of the earlier limits, and
every earlier reading keeps its PROVISIONAL status in its own record. Every stop on these checks prints its measured value, the limit, this
source, what it means and what to do. Re-gated with the calibrated limits, the 337 arrays give exactly one stop, GSM5583015.

**Still open from this procedure:** the detection and call-rate lines were measured on Stage 1's statistic (poobah) and are applied to Stage 0's
own (Chain Decoder finding F14).
