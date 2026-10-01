# PROC-DNMT-01 — pre-registration (written 2026-10-01, before any array or read of these datasets is read by us)

**Question.** When the enzyme that copies methylation (DNMT1) is blocked by a known amount, do the instruments read it, in proportion to the
dose and the time, and not when an inactive look-alike compound is given? This is the physics test: a known cause, a known direction, pure cells.

**Part A — Met-A and C-score on arrays (GSE135205; Pappalardi et al. 2021, Nature Cancer).** EPIC arrays from three AML cell lines (MV4-11, NOMO-1, THP-1):
- DMSO vehicle on days 1, 2, 4, 6;
- GSK3685032 (DNMT1-selective, reversible): 400 nM on days 1, 2, 4, 6, plus a day-4 dose series of 3.2, 16, 80, 400, 2,000 and 10,000 nM;
- GSK3484862 (active, 1 µM) on days 2 and 4;
- GSK3510477 (inactive analog, 10 µM) on days 2 and 4.

One array per condition. Raw IDATs through our Stage 1.

- **Reference = the same line's own DMSO arrays** (4 per line). Sites by the canon site rule on those arrays: SD ≤ 0.05; mean β 0.75–0.95 (methylated channel)
  or 0.05–0.25 (unmethylated channel); ≤ 3,000 per channel, chosen by the smallest SD. Reference floor = mean over the DMSO arrays of the mean H(β) at those sites.
  Met-A = mean H(β) / reference floor. Each DMSO array is read against the other three (leave-one-out). Noise index N is reported per array.
- **P1 (vehicle):** all 12 DMSO arrays read in Normal (0.95–1.05) against their own line's other DMSO arrays.
- **P2 (inactive analog):** GSK3510477, 10 µM, reads in Normal on day 2 and day 4 in all 3 lines (6/6).
- **P3 (dose):** in each line, day-4 Met-A rises with GSK3685032 dose from 0 to 2,000 nM (Spearman ρ ≥ 0.9 over the 6 points 0, 3.2, 16, 80, 400, 2,000).
  The 10,000 nM point is descriptive: entropy falls again if the methylated sites lose almost all methylation (β < 0.15). Reported, not predicted.
- **P4 (time):** at 400 nM, Met-A on day 6 > day 4 > day 2 > day 1 in at least 2 of 3 lines.
- **P5 (active compound 2):** GSK3484862, 1 µM, reads above Normal on day 4 in 3/3 lines.
- **P6 (which channel):** the rise is carried by the methylated channel (its H rises); the unmethylated channel changes by less than a third as much. This is the physics:
  blocked maintenance lets methylated sites drift toward 0.5, while unmethylated sites have nothing to lose.
- Descriptive: the C-score (genome-order clustering of the per-site residual against the DMSO arrays) at each dose.

**Part B — IAM-A on single molecules (GSE329728; EM-seq, MV4-11, DNMT1 inhibitor 100 nM GSK3685032 for 7 days vs DMSO, 4 genotypes × 2 replicates).**
Copy error ε on qualifying molecules (as PROC-TUMOUR-01, our alignment). Reference = the same genotype's DMSO libraries. The pipeline is the same within this dataset,
so the ratio cancels the pipeline (P_cell for the EM-seq pipeline is not needed for this ratio).
- **Q1:** H(ε) treated / H(ε) DMSO > 1.05 in every genotype (4/4), both replicates.
- **Q2 (instrument):** the conversion-failure difference between treated and DMSO is < 0.005 in each pair.
- Descriptive: the size of the rise relative to the Part A Met-A at day 6, 400 nM.

**Stated limits now.** Cancer cell lines, one array per condition in Part A, one lab. The compounds also slow proliferation (cytostatic from day 3), and fewer divisions means
less passive loss. This works against P3 and P4. A pass shows the instruments read a known change in maintenance fidelity; it does not show a disease reading.
