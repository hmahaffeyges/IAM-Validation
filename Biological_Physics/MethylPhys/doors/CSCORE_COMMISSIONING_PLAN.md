# Plan: commissioning the Met-A C-score (neutrophils, EPIC v1) — written 2026-10-09, before the data below were read

**What the C-score is for.** Met-A says *how much* a cell has drifted from its healthy pattern. The C-score says *how* — whether the drift
is scattered at random over the genome (C near 1) or clustered in stretches of neighbouring sites (C well above 1), the way a structured
change (a silenced region, a copy change) would look. To commission it, three things must be shown, the same three Met-A had to show.

**Where it stands.** The same-run tare removes the laboratory offset (DEV-CSCORE-TARE-01). On a new laboratory (GSE226298) 23 of 26 healthy
granulocytes fell inside the band, below the 95 % bar; all three misses were low. Checked 2026-10-09 on 615 healthy arrays: tare scope
(same slide only vs same series) does not change the spread (SD 0.176 vs 0.172), so the misses are not a tare-scope artefact.

## Step 1 — the band, set by leaving one laboratory out (no box)
For each of the 21 laboratories with healthy arrays read (job C's 20 series + GSE226298): set the 2.5-97.5 % band on the tared C of the
other 20, count the held-out laboratory's arrays inside. **Bar:** pooled over all held-out arrays ≥ 95 % inside, and no laboratory below 85 %.
The commissioned band is then the 2.5-97.5 % of all 21 laboratories together.

## Step 2 — it sees what it is meant to see (constructed positive control, no box)
Real healthy arrays (GSE247195 purified neutrophils, GSE110530 whole blood), each read against its unchanged same-run references, with a
loss of pattern put into a fraction F of the C-score's sites:
- **clustered:** F as contiguous stretches in genomic order (the C-score's own block order);
- **scattered:** the same number of sites chosen at random.
F = 10 % of sites, loss 10 % toward β = 0.5 at those sites, 20 random placements per array.
**Bars:** clustered → tared C above the band's upper edge on ≥ 95 % of placements; scattered → inside the band on ≥ 95 %.
The response at F = 5 % and 20 %, and loss 5 % and 20 %, is recorded (no bar) to give the C-score's detection limit, as for Met-A.

## Step 3 — repeatability (no box)
Same-person replicate arrays (GSE250556, job A): the within-person spread of tared C must be smaller than the healthy between-person spread
(**bar:** within-person SD ≤ half the healthy SD).

If steps 1-3 meet their bars, the C-score is proposed for commissioning with its band and detection limit printed on every report. If step 1
fails, the band is not widened to fit; the cause (array quality, site coverage, specimen) is found first and tested on a laboratory not
used to find it.

---
## Results, 2026-10-09 (nothing above the line changed)

**Step 1 — leave one laboratory out (19 laboratories with ≥ 3 references each, 641 healthy arrays).** Pooled **604 / 641 inside (94.2 %;
bar 95 %)**; GSE225544 **84.0 %** (bar 85 %), GSE142512 88.2 %, GSE226298 88.5 %; every other laboratory 94-100 %. **Not met** (narrowly).
The all-laboratory band is 0.750-1.409, the same as before. Table: `data/CSCORE_COMMISSIONING/cscore_step1_leave_one_lab_out.csv`.

**Step 2 — constructed positive control (12 + 12 healthy arrays, 1,248 constructed readings).**

| | purified neutrophils (GSE247195) | whole blood (GSE110530) |
|---|---|---|
| clustered, F 10 %, loss 10 %: above the band | **100 %** (median C_rel 18.5) | **100 %** (median 7.5) |
| scattered, same sites count: inside the band | **97.5 %** (bar 95 %: met) | **88.3 %** (not met) |
| clustered at F 5 % / loss 5 % | 100 % / 100 % above | 100 % / 100 % above |

\measured The C-score sees clustered change at every size tried, down to 5 % of its sites with a 5 % loss. Scattered change does not raise
it; in whole blood it lowers it slightly (mean −0.046), because scattered loss adds site-to-site variance (the denominator). The misses are
arrays that already sat near the lower edge (0.77-0.83 unchanged).

**Step 3 — repeatability (GSE250556, 4 people × 15-16 arrays).** Within-person SD of tared C **0.138**; healthy SD across all
laboratories 0.175; ratio **0.79 (bar ≤ 0.5): not met.**

## Diagnosis (corrected 2026-10-09 after the block-size test below)
\calculated The C-score is a variance of block means over 6,000 sites / 50 per block = **120 blocks**; a variance estimated from 120 values
has a relative sampling error of up to √(2 / 119) = 0.130. \measured On the same-person replicates the untared C varies by 0.089 within a
person; the tared C by 0.138, so roughly half of the tared spread is the tare's own noise (dividing by a median of a few references that
carry the same counting noise). Both come from the small number of blocks. Earlier wording that the replicate spread "equals" 0.130 was
wrong and is replaced by this paragraph.

## Development test: block size (2026-10-09, same data; the chain is unchanged)
| block (sites) | blocks | expected error √(2/(n−1)) | measured within-person SD (untared, 63 arrays, 4 people) | clustered loss (stretches of 10 or 50 sites, F 10 %, loss 10 %) above 1 + 3 × error | scattered loss above 1 + 3 × error |
|---|---|---|---|---|---|
| 50 (current) | 120 | 0.130 | 0.089 | 100 % / 100 % | 0 % / 0 % |
| 25 | 240 | 0.091 | 0.071 | 100 % / 100 % | 0 % / 0 % |
| 10 | 600 | 0.058 | **0.043** | 100 % / 100 % | 0 % / 0 % |

\measured Blocks of 10 halve the repeat noise and still detect clustered change every time, with no scattered false alarms (60 constructed
readings per cell of the table, 12 healthy purified-neutrophil arrays). Files: `data/CSCORE_COMMISSIONING/cscore_blocks_*.csv`.

## Proposed next step (for the author)
1. Change the C-score block from 50 to 10 sites (a method change, so the author decides). The healthy baseline is re-measured the same way
   it was (the 6 Salas reference neutrophils, leave-one-out), nothing fitted.
2. Rerun steps 1-3 with blocks of 10 on every healthy array (one short box job: job C again with the new block), with the same bars.
3. If the tare still adds noise there, read the C-score against the healthy baseline directly (untared) where the tare's references are
   fewer than 6; this is tested in the same run.

---
## Results with blocks of 10 (2026-10-09, author approved the change; bars unchanged)
Chain: `neutrophil_reference_v1_2.json`; every healthy array re-read from its stored betas (641 arrays, 19 laboratories incl. GSE226298
re-read on the box). Met-A unchanged (GSE226298 26/26 Normal, identical values).

| step | bar | blocks of 50 | **blocks of 10** | met |
|---|---|---|---|---|
| band (all laboratories, 2.5-97.5 %) | – | 0.750-1.409 | **0.877-1.152** (2.4× narrower) | – |
| 1. leave one laboratory out | pooled ≥ 95 %, none < 85 % | 94.2 %; one lab 84 % | **94.5 %**; GSE110530 8/12, GSE226298 20/26 | no |
| 2. clustered loss (F 10 %, loss 10 %) above the band | ≥ 95 % | 100 % | **100 %** (median C_rel 4.0-4.1) | yes |
| 2. scattered loss inside the band (purified, 120 readings) | ≥ 95 % | 97.5 % | **94.2 %** (misses all below the band) | no |
| 3. within-person SD ÷ healthy SD | ≤ 0.5 | 0.79 | **0.85** (0.062 ÷ 0.073) | no |

\measured Blocks of 10 make the C-score 2.4× more precise and it still detects clustered change every time. The three misses are all near
the bars and all on the low side or in the ratio.
\calculated Why step 3 fails even though the noise fell: within-person SD (0.062) is nearly the whole healthy SD (0.073), i.e. healthy
people do not differ from each other in C; the healthy spread *is* the repeat noise. A ratio bar (≤ 0.5) assumes real between-person
differences, which a healthy reference built to read 1 for everyone does not have. Recorded as not met; not changed.

## For the author (no change made)
1. **One-sided reading.** Clustering is the signal; a C below the band means *less* clustered than healthy, which has no known meaning.
   Counting only C above the band as a departure, the held-out laboratories read 624 of 641 not above (97.3 %) and scattered loss never
   rises above it (0 of 120). This changes the definition, so it needs your approval and then a test on laboratories not used to decide it.
2. **Repeatability bar.** Replace "within ÷ between ≤ 0.5" by an absolute bar tied to detection: within-person SD ≤ one-quarter of the
   smallest clustered change to be detected (here C_rel ≥ 1.5 at F 5 %). Needs your approval; tested on new same-person replicates.
