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

**Step 2, run once (2026-10-22; explore run 39804153955, commit f2701c95).** Output: `data/DEV_IAMA_GLOBE_01/genome_globes_02_summary.txt`.
Two people against each other: 20 of 99,221 regions changed, a quiet globe. One healthy sample against its own prediction: 337 of 116,311
changed (314 more loss), not dark; and the planted stretch lit only 13 of its 1,233 regions. Why, both: a region of 100 CpGs holds only a
few hundred chances to lose a held mark, so counting noise hides a doubled loss there; and healthy regions differ from one another by nature
(the genome-wide prediction is one number for every region), which counting noise alone does not cover. Two people share that natural
pattern, so their comparison is quiet.

**Step 3, written before it runs.** `genome_globes_03.py`: regions of 1,000 CpGs (about ten times the chances per region), and each
departure scaled by the sample's own spread of regions (the robust standard deviation from the median absolute deviation, never below counting noise), as the array globes scale by the
sample's own spread; still centered on the sample's own predicted held error, so a genome-wide loss still shows. The constructed plant is
made stronger (probability 0.08, about triple the held error) so the picture shows what a clear change looks like; it remains a constructed
picture, not a test.

**Synthetic test 1, run once (2026-10-22; explore run 39802161540, commit babac554).** Output:
`data/DEV_IAMA_GLOBE_01/synthetic_iama_globe_01_summary.txt` (array layout, blocks of 10 sites). The per-site sums equal count_pat's own
totals on both copies (healthy 939,299 errors over 26,002,890 opportunities). Healthy: 2 of 14,212 blocks changed. Planted: 52 of the 127
blocks wholly inside the stretch changed (41 %), 3 of 5 edge blocks, and 2 outside it. The counting is right and a doubled loss lights where
it was put. One thing the construction got wrong: its loss rate was meant to give the predicted held error 0.0403 and gave 0.0361 (the
neighbor rule removes more chances than the formula allowed); a genome-wide offset of that size lit nothing at this block size, which is
recorded, not adjusted.

**Step 3, run once (2026-10-22; explore run 39807826992, commit f8a21efb).** Output: `data/DEV_IAMA_GLOBE_01/genome_globes_03_summary.txt`.
Regions of 1,000 CpGs; 25,595 drawn. One sample against its own prediction: 33 changed (the sample's own spread of regions 2.467 counting
wobbles: healthy regions do differ from one another by nature). Two people: 6 of 25,555 changed (their spread 1.187: their regional pattern
is shared). The plant (probability 0.08) lit 11 of its 299 regions: still too weak against the spread to show clearly.

**Step 4, written before it runs.** `genome_globes_04.py` adds the view that matters for one individual over time: the same sample read twice,
its molecules split in two halves by a seeded coin, nothing changed, and then with a loss planted (probability 0.15, constructed) in one half
only; plus the two earlier views. Two halves of one sample differ only by counting, so this is how one individual sampled twice looks.

**Step 4, run once (2026-10-22; explore run 39812593849, commit 4d639a4d).** Output: `data/DEV_IAMA_GLOBE_01/genome_globes_04_summary.txt`.
Regions hold a median 1,933 held-error chances (one half 970). **The same sample read twice, nothing changed: 6 of 25,302 regions** (the
pair's own spread 1.179, close to pure counting). **The same sample read twice with a change planted between: 148 of the 293 regions inside
the stretch lit, 6 outside it.** Two healthy people: 6 of 25,555. One sample against its genome-wide prediction: 33 of 25,595, as in step 3.
The pictures drew one pixel per region, a sparse field of dots: `stage_q2_globe._page_layers` now gives every pixel of a region's span its
value (tested on synthetic counts: the planted stretch draws as a solid patch). **Step 5** (`genome_globes_05.py`) is step 4 redrawn that
way, with the region counts saved.
