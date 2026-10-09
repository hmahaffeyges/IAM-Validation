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

**Step 2 — constructed positive control (12 + 12 healthy arrays, 1,920 constructed readings).**

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

## Diagnosis
\calculated The C-score is a variance of block means taken over 6,000 sites / 50 per block = **120 blocks**. A variance estimated from
n = 120 values has a relative sampling error of √(2 / (n − 1)) = **0.130**. The measured same-person spread is 0.138. The healthy band is
therefore set almost entirely by the statistic's own counting noise, not by cells or laboratories: that is why it is wide (±40 %), why a
new laboratory's misses fall at random on its edges, and why step 3 fails. The tare cannot remove this; only more blocks can.

## Proposed next step (for the author)
Read the C-score over more sites so it has many more blocks — for example the 48,528 noise sites the chain already measures, in genomic
order: ≈ 970 blocks of 50, expected sampling error √(2/969) = 0.045, about a third of today's. Then steps 1-3 rerun unchanged on that
version. This changes which sites the C-score reads (a method change), so it waits for the author's decision.
