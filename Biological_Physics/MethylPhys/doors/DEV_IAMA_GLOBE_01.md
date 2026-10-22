# DEV-IAMA-GLOBE-01: IAM-A on the globe (development; to-do step 29)

**The instrument's job:** on the IAM-A globe, a healthy sample shows no departures from its own predicted held error, and a stretch of genome
where held marks are lost more often lights up there.

**The reading (written 2026-10-22, before any run).** `chain/stage_q2_globe.py`: chain Stage Q2's own count, kept per place (the held
error on odd lines, the methylation calls on even lines, as `count_pat` splits them); each region's errors against the errors the sample's
own loose share predicts on the self-reference line, in the region's own counting noise; blocks of 10 neighboring sites, changed beyond the
5 % family-wise line (the other globes' rule). No other person enters the reading.

**Where it is drawn (author 2026-10-22: "Could the full genome globe help with creating the IAM-A globe?").** IAM-A counts held error
only on molecules at least 80 % methylated, which lie mostly in the long methylated stretches where the arrays have few probes. On the
array layout (IAM-Atlas v2.1's 813,951 sites) every CpG is given to its nearest site, so every molecule counts, but a site in a sparse
region carries a long stretch of genome. The full-genome globe (every CpG, or IAM-A's own 2,000-CpG tiles, placed by the same fixed
genome-address rule) is IAM-A's home; the counting is the same on either layout. The array-layout version is kept for side-by-side
comparison with the array globes.

**Synthetic test 1** (`data/DEV_IAMA_GLOBE_01/synthetic_iama_globe_01.py`, committed before it runs; on GitHub): a synthetic .pat over chr1
and chr2, healthy and with the loss rate doubled over 50,000 CpGs; reported: the per-site sums against `count_pat`'s totals, changed blocks
on the healthy copy, and how much of the planted stretch lights up.

**The full-genome layout (2026-10-22).** `chain/stage_q2_globe.py` (count_pat_regions, self_globe, pair_globe): hg19's 26,472,355 autosomal
CpGs in the .pat index order, regions of 100 CpGs, region i of n at NESTED pixel floor(i x NPIX / n), NSIDE 256 (the atlas globe's rule,
applied to every CpG). The X and Y are left off, as IAM-A's loose share leaves them off. A region is drawn when it holds at least 200
held-error opportunities. Checked on 40,000 synthetic molecules before any real run: the region sums equal count_pat's own held-error totals
exactly (6,097 errors over 164,034 opportunities).

**Step 2, real molecules (pictures, nothing scored).** `data/DEV_IAMA_GLOBE_01/genome_globes_02.py`: two healthy CD8 T-cell samples of
the committed line (spent; they built the line, so the pictures show the method and test nothing): one against its own prediction; the same
one with held-mark loss planted in one stretch of chr2 (constructed); the two people against each other. On GitHub (explore.yml).
