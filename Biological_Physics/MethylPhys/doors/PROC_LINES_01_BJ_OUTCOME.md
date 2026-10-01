# PROC-LINES-01, first series — BJ fibroblasts (GSE91069), 31 arrays, our Stage 1; every stage read against the same cells' early passage

A_direct = mean per-locus entropy on loci stable across early passage / early-passage floor (the gauge's definition). A_model = one-parameter
contraction toward 0.5. toward / away = fraction of loci moving > 0.05 toward β = 0.5 / away from it.

| stage | n | A_direct | A_model | toward | away |
|---|---|---|---|---|---|
| early passage (held out) | 6 | 1.009 (0.957–1.028) | 1.011 | 0.016 | 0.007 |
| near-senescent | 3 | 0.955 | 1.013 | 0.088 | 0.121 |
| senescent | 3 | 0.916 | 0.993 | 0.084 | 0.152 |
| oncogene-induced senescence (10 days) | 3 | 1.003 | 1.005 | 0.020 | 0.021 |
| hTERT | 3 | 0.974 | 0.994 | 0.080 | 0.089 |
| hTERT + SV40 | 3 | 0.976 | 1.020 | 0.095 | 0.113 |
| hTERT + SV40 + HRAS (transformed) | 3 | 0.995–1.036 | 1.051–1.107 | 0.118 | 0.099 |
| xenograft tumours | 3 | 0.937–1.015 | 1.081–1.178 | 0.110–0.172 | 0.147–0.178 |
| near-senescent + EV / hTERT / SV40 / HRAS | 1 each | 0.888–0.944 | 1.027–1.042 | ~0.09 | 0.13–0.18 |

- Senescence LOWERS A (0.92): more loci sharpen away from 0.5 than blur toward it. Oncogene-induced senescence changes nothing (as the
  source paper reports: no methylation change in 10 days) — a clean negative control.
- Transformation and tumours move MANY loci both ways at once (11–17 % toward, 10–18 % away). The mean entropy barely moves (1.00–1.04);
  the contraction fit, which follows the reference pattern, reads 1.05–1.18. The single-number A hides the transformation.
- Xenograft arrays contain mouse stroma and one has call rate 0.774 (below intake).

**Reading.** On this series, where a line would sit depends on which number is read. The mean per-locus entropy does not cross a line at
transformation; the pattern-following contraction and the two-way fractions do. First series only; a second is needed before any line is set.

## Second series (PROC-LINES-02): IMR90 lung fibroblasts, WGBS (GSE48580), 1.65 M CpGs at depth ≥ 10 in all nine samples
| state | A (depth-corrected mean entropy vs proliferating) | toward 0.5 | away from 0.5 |
|---|---|---|---|
| proliferating (held out) | 1.005–1.008 | 0.26–0.27 | 0.23–0.25 |
| replicative senescent | **0.883–0.929** | 0.22–0.25 | 0.29–0.31 |
| SV40-immortalised | 0.960–0.971 | 0.27–0.29 | 0.26–0.28 |
**Senescence lowers A in a second cell line, on a second platform: 0.88–0.93 here, 0.92 in BJ.** More loci sharpen away from 0.5 than blur
toward it. (Toward/away are high even for held-out proliferating replicates because single-locus WGBS noise at depth 10 exceeds 0.05; they are
read against that baseline.)

## A as an error measure, read by channel (2026-09-30)
On an identity locus the entropy is the entropy of the error fraction (β = 1 − ε at methylated loci, β = ε at unmethylated loci), so
A = H(error now) / H(error in the cell's own standard). Read separately on methylated (A_hi) and unmethylated (A_lo) identity loci, each as H(mean β):

| series / state | A_hi (β) | A_lo (β) | per-locus A |
|---|---|---|---|
| IMR90 proliferating | 1.04–1.05 (0.84) | 1.04–1.06 (0.19–0.20) | 1.005–1.008 |
| IMR90 senescent | **1.20–1.23** (0.82–0.83) | **0.81–0.85** (0.10–0.11) | 0.88–0.93 |
| IMR90 SV40 | 1.28–1.38 | 0.72–0.78 | 0.96–0.97 |
| BJ early passage | 1.010 (0.880) | 1.004 (0.113) | 1.009 |
| BJ senescent | 1.001 | 0.989 | 0.916 |
| BJ HRAS transformed | **1.081** (0.866) | 1.017 | 1.035 |
| BJ xenograft tumours | 1.00–1.06 | **1.06–1.21** (0.12–0.16) | 0.94–1.02 |

IMR90 senescence carries two opposite errors (methylated sites losing marks, unmethylated sites wiped cleaner); a single average hides them. In BJ
the channel averages barely move with senescence (the per-locus sharpening is spread across loci), while transformation raises error on methylated
loci and tumour formation raises it on unmethylated loci. Report per channel, not one averaged number.

## PROC-CHANNEL-01 first pass: 56 cell types, read-level WGBS (first 60 MB of each file = the same stretch of chr1)
Neighbour agreement, isolated copy errors, erasure runs, de novo errors, coherent gains, PDR. Best grouping: 2 clusters (silhouette 0.65;
progenitors apart, high single-molecule mixing); ARI to the draft classes 0.019 (relabelling null p95 0.014, p = 0.018). Genome-wide sampling next.

## Third series (PROC-LINES-03): HBEC-3KT bronchial epithelium, 450K, GSE101673 (31 arrays, all pass Stage 1)
Read against (a) the 1-month control and (b) the time-matched control. Medians per stage.

| stage | A_meth vs 1-month | A_unmeth vs 1-month | A_meth vs matched | A_unmeth vs matched | per-locus vs matched |
|---|---|---|---|---|---|
| control 1 month | 1.002 | 1.004 | — | — | — |
| control 6 / 10 / 15 months | 1.03 / 1.03 / 1.00 | **1.07 / 1.15 / 1.24** | — | — | — |
| smoke 6 / 10 / 15 months | 1.06 / 1.03 / 1.03 | 1.05 / 1.09 / 1.15 | 1.049 / 1.040 / **1.057** | 1.006 / 0.991 / 0.972 | 1.019 / 0.972 / 0.979 |
| smoke 15 m + empty vector | 1.078 | 1.160 | **1.087** | 0.934 | 0.923 |
| smoke 15 m + KRAS | 1.000 | 1.194 | 1.004 | 0.973 | 0.924 |
| xenograft tumours (7) | 1.064 | 1.182 | **1.080** | 0.973 | 0.930 |

1. **Long culture alone moves the unmethylated channel by +24 % in 15 months** (control cells, no smoke). Any series read only against its early
   passage mixes culture drift into the reading. The time-matched control is required; BJ and IMR90 have no time-matched controls.
2. **Against matched controls, smoke and tumour read on the methylated channel: 1.04–1.06 with smoke, 1.08–1.09 transformed / tumour.** BJ HRAS
   read 1.081 on the same channel. Two lineages, two oncogenic routes, the same channel, within 0.01.
3. The unmethylated channel against matched controls falls slightly (0.93–0.97) and the per-locus A drops to 0.92–0.93: sharpening, as in senescence.
