# DEV-CSCORE-TARE-01 — the Met-A C-score is tared the same way as Met-A (development)

**DEVELOPMENT - not commissioned.** Data: Box Run 1 job C (`s3://…/results/BOXRUN1/C/C_arrays.csv`), 615 healthy arrays, 19 series.

**Question.** Job C showed the untared healthy C-score median differs by laboratory (series medians 0.82 to 1.47). Is that the cell or the
laboratory? If it is the laboratory, the C-score needs the same same-run tare Met-A already uses before any healthy band can be set.

**Checks.**
1. Does C track the array's noise index, its neutrophil fraction or its site count? Correlations: N 0.056, f_neu −0.068, n_sites −0.039.
   None does, so the spread is not explained by noise or composition.
2. Tare: C_rel = C / median C of the other healthy arrays of the same specimen type on the same slide (else the same series), at least 3,
   leaving the array itself out (the Met-A Stage T rule).
3. Held-out test: set a 2.5–97.5 % band on a random half of the laboratories, count how many arrays of the other half fall inside;
   200 random splits.

**Results.**
| | 2.5 % | median | 97.5 % | spread of series medians | held-out coverage (median, 5th percentile of splits) |
|---|---|---|---|---|---|
| untared C | 0.722 | 1.066 | 1.820 | 0.824–1.473 | 0.903, 0.733 |
| tared C_rel | 0.751 | 1.000 | 1.409 | 0.987–1.007 | **0.951, 0.874** |

**Reading.** The laboratory difference is removed by the same-run tare (series medians collapse to 0.99–1.01), and a band set on half the
laboratories holds on the other half at the intended 95 %. The C-score should therefore be read tared, exactly like Met-A.

**Proposed (to wire after the author's review, then commission):** print C_rel beside C; the healthy band is set on tared C_rel from
held-out laboratories (on these data 0.75–1.41), and checked on laboratories not used to set it before it is used. Same for the IAM-A
C-score once enough single-molecule files exist. Per-array values: `data/DEV_CSCORE_TARE_01_arrays.csv`.
