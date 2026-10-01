# DEV-CHAIN-V3-RUN2 — chain v3 end to end with the new read rules (development, 2026-10-01; commit d5873bd, floors v1.2)

**Run.** 644 EPIC whole bloods from GEO (FACS-counted healthy GSE112618 6; technical replicates GSE250556 64; COVID-19 GSE179325 574),
IDAT → intake → calibration → composition → Met-A → C-score → same-batch tare → report, through `run_sample.py --engine v3`.

**What the chain did.**
- 639/644 produced a bundle; 5 arrays failed with no bundle (download or IDAT issue, to trace).
- Neutrophil Met-A was read on 637/639 (all with neutrophils ≥ 0.20). Two were withheld: fractions 0.000 and 0.179.
- Every reading now carries its detection limit: the smallest loss of the neutrophil pattern this specimen could show.

| set | neutrophils | n | spread of tared A | median detection limit (% pattern loss) |
|---|---|---|---|---|
| FACS-counted healthy | 0.2–0.7 | 6 | 0.010–0.038 | 1.8–2.8 |
| technical replicates | 0.2–0.5 | 62 | 0.037 | 3.6–5.5 |
| COVID-19 | 0.2–0.4 | 20 | 0.065 | 8.9 |
| COVID-19 | 0.7–1.0 | 276 | 0.063 | 3.3 |

- No specimen was past the entropy ceiling (methylated-site mean β 0.870–0.938).
- COVID-19 by group (tared, median): negative 0.999, mild 0.998, severe 1.030. The severe group also has more neutrophils (0.79 vs 0.65), which
  DEV-NOISE-02 showed is most of this difference.

**What this teaches.**
1. The read line and detection limit work end to end, and the detection limit is honest: in the COVID set the same-batch spread (0.05–0.06) is
   3× the FACS set's. That is the array-noise term from DEV-NOISE-01/02, and the chain does not yet remove it.
2. Next chain change: put the noise-index correction (DEV-NOISE-02) into Stage T, then re-run this set. The detection limit should fall toward
   the FACS set's 2–3 %.
3. Floors moved to v1.3 after this run (duplicate arrays removed; value unchanged). The next run uses v1.3.
