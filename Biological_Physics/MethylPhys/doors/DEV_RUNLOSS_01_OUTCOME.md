# DEV-RUNLOSS-01 — outcome (2026-10-10; development)

Read by `data/DEV_RUNLOSS_01/runloss_01.py` (output `runloss_01_read.txt`), by the prediction sealed in 428d88d.

| run | dose | territory molecules | K | L | d_eff | L_scat | excess run loss |
|---|---|---|---|---|---|---|---|
| SRR25322251 | vehicle 2 | 478,805 | 0.8649 | 0.0028 | 0.0008 | 0.0029 | −0.0001 |
| SRR25322250 | 30 nM | 400,274 | 0.4032 | 0.1418 | 0.3054 | 0.0203 | +0.1215 |
| SRR25322249 | 30 nM | 416,060 | 0.4137 | 0.1340 | 0.2945 | 0.0184 | +0.1156 |
| SRR25322248 | 300 nM | 431,937 | 0.2127 | 0.5339 | 0.6308 | 0.2589 | +0.2750 |
| SRR25322247 | 300 nM | 427,491 | 0.2275 | 0.5166 | 0.6145 | 0.2331 | +0.2835 |

| bar | predicted | read | met |
|---|---|---|---|
| 1 d_eff = array loss ± 0.05 | 0.22 / 0.39 | 0.30 / 0.62 | no |
| 2 excess ≥ 0.5 d_eff | ≥ 0.15 / ≥ 0.31 | 0.12 / 0.28 (0.39–0.46 of d_eff) | no |

**What it shows.** Run-type loss is present and dose-graded: the excess is 0.12 and 0.28, against −0.0001 for the second vehicle and ~0 for
scattered loss in simulation. But it is about 40–45 % of the total loss, not most of it. The EM-seq molecules lose more than the arrays show at
both doses (0.30 against 0.22; 0.62 against 0.39). Array β is compressed away from 0 and 1, and the site sets differ; neither is tested here.
The reading is a development instrument. The next test needs a second experiment with molecules and arrays on the same cells.
