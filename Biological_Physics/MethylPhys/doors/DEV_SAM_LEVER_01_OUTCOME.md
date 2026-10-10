# DEV-SAM-LEVER-01 — outcome (2026-10-10; development)

Scored by `data/DEV_SAM_LEVER_01/score_sam_01.py` (committed f7d7b56 before any knockout file was read; output `score_sam_01_output.txt`;
`sam_01_runs.csv`, `sam_01_mice.csv`). GSE77079 mouse liver RRBS, 19 mice, 38 runs, Box Run 6 (Trim Galore --rrbs, bwa-meth 0.2.0, mm10,
wgbstools bam2pat), Stage Q. Window sealed 49ab1e6 on the first wild-type mouse; power stated ec5d0de on all 8 wild types.

| group | mice | IAM-A (÷ wild-type median) | median |
|---|---|---|---|
| wild type | 8 | 0.952–1.096 | 1.000 |
| Mat1a knockout, vehicle | 6 | 1.106–1.171 | **1.124** |
| Mat1a knockout + SAMe | 5 | 1.074–1.141 | 1.098 |

**Prediction 1 (knockout vehicle ÷ wild type inside 1.016–1.135, p < 0.05): PASS.** Ratio 1.124, one-sided Mann-Whitney p = 0.0003; every
knockout mouse above 7 of the 8 wild types. The reading sits in the upper part of the window (the calculation's cells with liver SAM 30 µM and
most cells renewed, or 60 µM with all renewed).
**Prediction 2 (SAMe lowers it): direction met, not significant** (1.098 vs 1.124, p = 0.089).

**Checks for a technical cause (after scoring, recorded):** all 38 runs one instrument model (HiScanSQ), 50-base reads, one deposit; all mice
10 months. Copy error does not track depth within any group (Spearman ρ −0.12, +0.09, −0.20); wild-type runs at the lowest depth read ~3 %
higher than the deepest, against a 12 % effect.

**What this does not settle.** Mat1a knockout livers develop steatohepatitis (the series' own summary): injury, hepatocyte turnover and immune
cells are downstream of the same SAMe loss and could raise copy error by another route (more divisions, a different cell mix). SAMe
supplementation reverses part of both, so prediction 2 cannot separate them. Mouse identifiers carry cohort letters (wild type all K; knockout
vehicle J and G; knockout SAMe K, G and I) that GEO does not explain.

**Milestone:** listed in `Biological_Physics/README.md`, Advancements.
