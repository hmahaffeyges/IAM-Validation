# DEV-TOOLKIT-ADDED-02 - trace cell (3b), foreign cell (3c), surface brightness (11b) on each array's own noise; IAM-A versions (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Author decision K.** Each line is set on the array's own noise; no line comes from other arrays; beta scale = Stage 1 noob beta (no map);
a v3 per-site uncertainty.

**v3 per-site uncertainty.** On one array, for each probe design (I, II) and state (low, high): the SD of beta over the array's own fixed sites of
that design and state (noise sites for type I; the DEV-SELFTARE-02 type II sets), each site measured against its frozen reference value.
s_i = the value for site i's design, the low-state value where the reference value of site i is < 0.5 and the high-state value otherwise.

**3b trace cell (contamination of an isolated-neutrophil specimen).** At the 963 composition markers, residual r = beta - mu_NEU; for each other
blood group c, the weighted regression of r on d_c = mu_c - mu_NEU (weights 1/s_i^2) gives f_c and its standard error from this array's own fit
residual scatter. Called when f_c > 3 SE. Constructed test: other-laboratory purified neutrophil arrays (GSE247195) with the betas of another
laboratory's purified cell (GSE122244 monocyte, B and T arrays) added at f = 0, 1, 2, 5, 10 % (beta mixed linearly). Bars: at 5 % the spiked cell is
called on >= 95 % of constructed specimens; of the calls at 5 %, >= 95 % name the spiked cell. The f = 0 call rate is recorded (the true contamination
of the unspiked arrays is not known).

**3c foreign cell (non-blood material in whole blood).** The bucket holds one placenta series (GSE271697, 93 arrays). Template: the mean of the first
half of its arrays (GSM order); spike material: the other half (never in the template). Sites: the 963 markers plus up to 500 sites where the template differs from every blood group by >= 0.25. Fit:
non-negative least squares on [8 blood groups, template], weights 1/s_i^2; the template's standard error from this array's own fit residual scatter.
Called when f > 3 SE. Specimens: GSE250556 replicates (no foreign material) with placenta added at f = 0, 1, 2, 5, 10 %. Bars: at f = 0 <= 5 % called;
at 5 % >= 95 % called.

**11b surface brightness (interval on Met-A).** 200 draws beta_i + N(0, s_i^2), clipped to (0, 1), A recomputed each time; 95 % interval.
Bar: on GSE250556 pooled-replicate pairs of the same person, |A1 - A2| <= sqrt(w1^2 + w2^2) (w = half-widths) in >= 95 % of pairs.

**IAM-A versions (design).** A read's own noise line needs, per read, the bisulfite / enzymatic conversion failure (unconverted non-CpG C on that
read) and its base-call error (base qualities): the isolated copy errors those two alone would make on that read are the read's noise line, and
the copy error is read above it. A `.pat` file keeps neither (no sequence, no quality), so the IAM-A versions of 3b, 3c and 11b cannot be tested on
the single-molecule data available; they need BAM input. Written into the SOP as designed, not built.

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box jobs 9bcbcf99 (3b, 3c) and 3003767d (11b; the first 11b run, job 9bcbcf99, returned no interval because sites without an expectation were not
dropped - fixed in `dev_stages.brightness`, then re-run). Runtime file for 3c: `chain/Runtime Matrices/Development/dev_foreign_placenta_EPIC_v1.json`
(963 markers + 500 placenta sites). Records: `data/DEV_TOOLKIT_ADDED_02/`.

**v3 per-site uncertainty** (\measured, GSE250556 pooled replicates): SD at this array's own fixed sites, type I 0.004-0.006, type II 0.017-0.019.

**3b trace cell** (24 GSE247195 purified neutrophil arrays x 15 GSE122244 spike arrays):

| added fraction | 0 | 1 % | 2 % | 5 % | 10 % |
|---|---|---|---|---|---|
| spiked cell called | 0 % | 11 % | 34 % | 82.5 % | 93.1 % |

- \measured At 5 %: **82.5 %** called (bar 95 %, outside); of the calls, **98.3 %** name the spiked cell (bar 95 %, within). B 100 %, T 71.7 %, monocytes 75.8 %.
- \measured Unspiked: 0 of 360 called.
- \observed Five of the 15 spike arrays are not pure by Stage 2 (GSM3462458 "monocyte" reads 0.76 neutrophil, GSM3462461 "T" 0.54 neutrophil, three T arrays
  0.62-0.71 CD4T). Read beside the bar: with spike arrays whose own group is >= 0.90 by Stage 2, 97.9 % called at 5 % (240 constructed); with the other five,
  51.7 %. The shortfall sits in the spike material, which is also recorded in DEV-NILC-01.

**3c foreign cell** (63 GSE250556 arrays, placenta from the second half of GSE271697):

| added fraction | 0 | 1 % | 2 % | 5 % | 10 % |
|---|---|---|---|---|---|
| called | 100 % | 100 % | 100 % | 100 % | 100 % |
| fraction read (median) | 0.034 | 0.045 | 0.056 | 0.088 | 0.137 |

- \measured With no placenta added, every array is called (bar <= 5 %, outside): the template takes 0.023-0.048 of a healthy whole blood from another laboratory,
  against a standard error of ~0.004. At 5 %, 100 % called (bar 95 %, within).
- \observed The read fraction moves with the added fraction (0.034 -> 0.137 for 0 -> 10 %); the zero is offset, not the slope. This array's own standard error does
  not hold the laboratory difference between this blood and the purified profiles; the line needs a same-run zero (the same tare logic as Stage T).

**11b surface brightness** (31 pooled replicates, 105 same-person pairs):
- \measured Interval half-width median **0.0034**; same-person |A1 - A2| median 0.032; pairs inside the combined interval
  **3.8 %** (bar 95 %, outside).
- \observed Per-site noise averaged over ~5,600 sites is ten times smaller than the replicate spread: what moves A between replicates is array-wide (the offsets
  the self-tare removes, DEV-SELFTARE-02), not site noise.

**IAM-A versions.** Designed as written above; not built (.pat carries no sequence or quality).

**Wiring.** 3b, 3c and 11b behind `--dev-trace`, `--dev-foreign`, `--dev-brightness`. Not part of the reading.
