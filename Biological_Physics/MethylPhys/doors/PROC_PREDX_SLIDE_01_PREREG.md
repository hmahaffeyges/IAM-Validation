# PROC-PREDX-SLIDE-01 — pre-registration (written 2026-10-01, before the 329 GSE51057 arrays are looked at under this rule)

**Origin (exploratory, disclosed).** In the 516 held-out EPIC-Italy arrays, the array slide explains 52 % of neutrophil Met-A variance and the sex
gap vanishes within slides (+0.007, p = 0.12). Read relative to the other controls on the same slide (leave-one-out median; ≥ 3 controls),
controls read 88 % Normal, and breast cases > 8 y before diagnosis read outside Normal more often than female controls (11/40 vs 17/162,
p = 0.008; all breast 22/89, p = 0.003; same-slide paired +0.009, 13/16 slides). This was found after looking; it is not evidence.

**Test set.** The 329 arrays also in GSE51057 (all women; 146 breast cases), read with exactly the same code already run. These arrays were used
in the author's earlier VALs, so they are not naive to the cohort, but they have not been looked at under this rule.

**Rule (fixed).** A_rel = A_neu / median A_neu of the other controls on the same slide (all 845 arrays' controls eligible as slide references;
≥ 3 required). Outside Normal = A_rel outside 0.95–1.05.

**Predictions.** S1: controls ≥ 80 % in Normal. S2: breast cases > 8 y outside Normal more often than controls (one-sided Fisher, p < 0.05).
S3: all breast cases, same, p < 0.05. Descriptive: by lead-time bin.
