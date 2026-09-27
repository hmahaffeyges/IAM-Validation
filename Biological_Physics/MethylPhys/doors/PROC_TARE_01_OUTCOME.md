# PROC-TARE-01 — outcome: NOT COMMISSIONED. The array's SNP probes see a real compression, and it carries almost no information about where a cell sits on the gauge.

**Sealed 2026-09-27** against the bars in [`PROC_TARE_01_PREREG.md`](PROC_TARE_01_PREREG.md), fixed before any SNP probe was
read for this purpose. 768 arrays: the full 732-array GSE87571 calibration plus the 12-array panels of GSE42861, GSE111629
and GSE125105 (raw IDATs fetched from GEO for this procedure). Evidence: PROC_TARE_01.json, PROC_TARE_01_per_array.parquet,
PROC_TARE_01_diag.json in the kit results folder; PROC_TARE_01.py in the kit; plate PROC_TARE_01.png.

**Run record.** The first pass started before the panel fetch had finished and its panel arm carried 1 + 12 + 0 pairs; the
GSE87571 arm (732) was complete and is kept. The panel arm was re-run alone on 2026-09-27 with the fetch complete (36 arrays)
and merged; the bars were then scored once, on all 768. Nothing was changed after results were visible.

## The construction, as pre-registered

65 SNP probes on the 450K array read β = 0, 0.5 or 1 by genotype. Per array: cluster to the nearest ideal (two passes),
fit a linear tare β' = (β − T_offset) / T_scale that puts the three clusters back on 0 / 0.5 / 1, apply it to the whole array
before the pipeline map, and read the immune identity gauge raw and tared.

## Result

| laboratory | n | median A raw | median A tared | sd raw -> tared | T_offset | T_scale | z_lab commissioned / measured here |
|---|---|---|---|---|---|---|---|
| GSE87571 | 732 | 0.9918 | 0.9475 | 0.0242 -> 0.0362 | -0.0030 | 0.9390 | -0.0117 / -0.0093 |
| GSE42861 | 12 | 1.0156 | 0.9886 | 0.0233 -> 0.0187 | +0.0055 | 0.9205 | +0.0084 / +0.0096 |
| GSE111629 | 12 | 0.9613 | 0.9706 | 0.0200 -> 0.0341 | +0.0285 | 0.9267 | -0.0673 / -0.0478 |
| GSE125105 | 12 | 1.0043 | 1.0192 | 0.0121 -> 0.0520 | +0.0425 | 0.8902 | -0.0346 / +0.0025 |

| bar | requirement | result |
|---|---|---|
| B1 | the raw path reproduces each laboratory's commissioned zero to ±0.005 | **FAILED** - met on GSE87571 (Δ 0.002) and GSE42861 (Δ 0.001); not on GSE111629 (Δ 0.020) or GSE125105 (Δ 0.037), each measured on 12 arrays against a 40-array commissioning |
| B2 | the tare moves every laboratory's median toward 1.00 and shrinks the worst offset by ≥ 50 % | **FAILED** - GSE87571 0.9918 → 0.9475 and GSE125105 1.0043 → 1.0192 move away; worst shrinks 24 % |
| B3 | all four medians inside NORMAL after the tare | **FAILED** - GSE87571 0.9475 |
| B4 | spread tightens or holds on ≥ 3 of 4 | **FAILED** - widens on three (GSE87571 0.024 → 0.036, GSE111629 0.020 → 0.034, GSE125105 0.012 → 0.052) |
| B5 | tare parameters independent of age and sex (\|r\| < 0.10) | **FAILED** - r_age = −0.18 (T_scale), r_sex = −0.04 |
| B6 | per-chip term shrinks ≥ 20 % | **FAILED** - sd of chip medians 0.012 → 0.019 (worse) |
| B7 | instrument unchanged on the 11 commissioning arrays | NOT ASSESSED (nothing in the chain was changed by this procedure) |

## Why, measured after the bars (diagnostics, not bars)

- **The tare is a constant shift.** It moves A by a median of −0.044 on 94 % of arrays regardless of where they sat: 95 % of
  arrays above 1.00 moved toward it and 94 % of arrays below 1.00 moved away. A correction that shifts everyone the same way is
  an offset, and the instrument already has one (the pipeline map); it does not tare anything.
- **T_scale is not a chip property.** Median 0.939 (the array compresses β by ~6 % at the extremes - real, and seen on every
  laboratory: 0.89-0.94), but within-chip SD 0.0098 exceeds between-chip SD 0.0062 over 62 chips. The chip term of row 5b is
  not what the SNP probes measure.
- **It does not predict A.** corr(T_scale, A_raw) = −0.14; corr(T_offset, A_raw) = −0.07 on 732 arrays. Compression at β = 0
  and β = 1 says little about the array's response at β ≈ 0.7, where the identity loci sit.

## What this decides

The question was whether the instrument can be zeroed on a known input carried by every array, instead of on a healthy panel.
With a **linear** tare from the **SNP probes**, no. Two routes remain and are recorded, not started: (i) a nonlinear response
model - the compression is a saturation and a straight rescale is the wrong form for it; the tri-modal SNP clusters constrain
only three points of a curve, so this needs the array's control probes as well; (ii) a physical reference material run
through the laboratory's own pipeline. Until one of these is measured, the pipeline map onto the atlas scale is the instrument's
calibration, and a laboratory zero remains on record and unapplied.

The statement stands as it was before this procedure: healthy is A = 1.00 with the tier scale as tolerance, and no population
defines it. What PROC-TARE-01 failed to do is replace the panel-derived instrument constants with an on-array one.
