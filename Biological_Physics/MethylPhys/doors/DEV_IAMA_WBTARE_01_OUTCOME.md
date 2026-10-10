# DEV-IAMA-WBTARE-01 — outcome (2026-10-10; development)

Read by the bars as written (`doors/data/DEV_IAMA_WBTARE_01/score_wbtare_01.py`; rows `wbtare_01_rows.csv`; Box Run 2 session 4 bundles).

**Intake.** 9 of 14 runs were refused at Q0.3 (bisulfite conversion ≥ 0.98, the ENCODE limit fixed before reading). Every Swift
library sits at the limit: 0.9792–0.9799, with Sample3 Swift rep2 passing at 0.98001. The TruSeq repeat libraries are below it
(0.9700–0.9738). The TruSeq first libraries pass (0.9894–0.9911). The limit is not changed after reading.

| bar | as written | read | result |
|---|---|---|---|
| 1 healthy reads Normal after the tare | 8 of 8 rep1 within 0.95–1.05 | 4 TruSeq rep1: 1.0242, 0.9723, 0.9789, 1.0215 (untared 1.077–1.110 on the neutrophil scale) | 4 of 4 readable within; 4 Swift rep1 not read |
| 2 kit offset removed | 4 donors, both kits | no Swift rep1 passed intake | not assessable |
| 3 repeatability | 6 rep2/rep1 pairs | one rep2 read (Sample3 Swift), no readable rep1 partner | not assessable |
| 4 a real change survives | in silico, both kits | — | not run |

**Decision: incomplete.** The four TruSeq donors agree within ±0.03 after the tare. That is what the tare predicts, but on one kit and
half the runs. The kit offset in whole blood is not tested by this set. The conversion limit decides which libraries are read, and every
Swift library sits on it, so whether 0.98 is right for this chemistry is now an open question for the intake (DEV-IAMA-INTAKE-01). It must
be decided on files not used here.
