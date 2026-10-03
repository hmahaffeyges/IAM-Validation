# PROC-REPL-V3-01 — counting rules fixed before any array was read (2026-10-03, written before the box job was submitted)

These fill in what the pre-registration leaves to the operator. They were written before the job ran and are not changed after.

- **Read end to end:** the pass-2 run (the array with its same-run references) wrote a report and bundle with a numeric Met-A.
  If pass 2 cannot run because pass 1 gave no A, the array is not read; the reason is taken from its run log
  (Stage 0 quarantine flags, refusal text, or the Met-A reason).
- **Tared:** pass-2 bundle carries a numeric A_rel (Stage T median tare, >= 3 references, the array itself excluded).
- **Reference set (Stage T):** the other GSE250556 arrays with a pass-1 A on the same slide when there are >= 3; else all other GSE250556
  arrays with an A. All 64 arrays are healthy-donor whole blood, so every one is a reference for the others.
- **Withheld by the noise gate:** pass-2 Met-A state begins "withheld". Also counted: arrays with noise index N above N_max = 0.149
  (the gate would withhold them if untared).
- **Person:** the subject label in the GEO series-matrix sample title (`subjectA`..`subjectD`), cross-checked against any
  characteristics field naming the subject. All replicates of one person are pooled whether "pooled" or "unpooled" DNA preparation.
- **Within-person SD (bar 2):** pooled within-person SD of tared A_rel = sqrt( sum_p sum_i (x_pi - mean_p)^2 / sum_p (n_p - 1) ).
  Per-person SDs (ddof = 1) are also reported. Bar 2 is judged on the pooled value, rounded to 3 decimals as reported.
- **SD over all:** SD (ddof = 1) of all tared A_rel.
- **Normal:** 0.95 <= A_rel <= 1.05. Bar 3: (number in Normal) / (number tared) >= 0.95.
- **RUN3 comparison:** `doors/data/chain_v3_dev3_readings.csv`, GSE250556 rows, column `A_rel_tared`, same statistics.
