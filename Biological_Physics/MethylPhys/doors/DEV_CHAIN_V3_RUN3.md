# DEV-CHAIN-V3-RUN3 — chain v3 with the audit fixes and the noise-corrected tare, end to end on real IDATs (development, 2026-10-02)

**Run.** 692 EPIC arrays from GEO (FACS-counted healthy whole blood GSE112618 6; isolated neutrophils GSE247193/5 48; technical replicates GSE250556 64;
COVID-19 whole blood GSE179325 574), IDAT → intake (gate before calibration) → calibration → composition → Met-A → C-score → same-batch tare → report,
`run_sample.py --engine v3`, two passes (pass 2 tares against the batch's healthy references with `--slide-ref-table`). 690 processed; 2 produced no report.

**What it shows.**
| set | read | tared | tared A_rel SD | in Normal |
|---|---|---|---|---|
| FACS healthy (6, median tare: < 20 references) | 6 | 6 | 0.026 | 6/6 |
| technical replicates (noise-corrected tare) | 63 | 63 | 0.013 | 62/63 |
| COVID-19 NEGATIVE (noise-corrected) | 99 | 99 | 0.022 | 96 % |
| COVID-19 MILD | 356 | 356 | 0.029 | 92 % |
| COVID-19 SEVERE | 113 | 113 | 0.024 | 96 % |
Before the noise correction (DEV-CHAIN-V3-RUN2) the COVID batch spread was 0.057–0.062 and NEGATIVE read 76 % Normal; with it, 0.022 and 96 %.
The healthy spread in this batch is now close to the purified-array floor precision (SD 0.020).
Severe COVID-19 reads above Normal no more often than NEGATIVE (2.7 % vs 3.0 %): once noise and fraction are removed, neutrophil Met-A on whole-blood arrays
does not register COVID-19 severity here. Median detection limit 1.4–1.8 % pattern loss. No specimen past the entropy ceiling.
Isolated neutrophils from the second lab (GSE247193/5): untared A median 1.08 (0.86–1.29); not tared, because the series has no same-run healthy
references. They need a same-lab reference set before they can be read.

**Chain change.** The audit-fix patch (intake gate before calibration, platform refusal, report gauge on the tared value, Stage Q pipeline required,
noise-corrected Stage T with `--slide-ref-table`) is adopted as tested here.
