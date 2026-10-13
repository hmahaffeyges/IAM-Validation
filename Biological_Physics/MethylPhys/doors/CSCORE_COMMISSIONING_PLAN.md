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

## Author decision 2026-10-09 and the definitions now in force (development until tested)
The author: "do whatever is best for IAM and is the most defensible." Adopted, both written before any new data are read:
1. **One-sided reading.** Clustering is the signal: a C-score *above* the band is a departure; below the band is recorded as "less
   clustered than healthy" and is not a departure. Upper limit 1.152 (blocks of 10).
2. **Repeat bar tied to detection.** Same-person within-SD of C_rel ≤ one-quarter of the smallest clustered change to be detected
   (C_rel 1.5 at 5 % of sites clustered → ≤ 0.125). On the existing replicates: 0.062 (met).
**Test on laboratories not used to decide (bars):** (a) healthy arrays from new laboratories: ≥ 95 % not above 1.152 — first set:
GSE315366's 35 clonal-haematopoiesis-negative bloods (Box Run 3), then each further healthy EPIC v1 laboratory read; (b) new same-person
replicates when a set is found: within-SD ≤ 0.125. Clustered detection (step 2) already met at 100 %. If (a) and (b) are met, the C-score
is proposed for commissioning with its band and detection limit printed.

## Test (a) on the first laboratory not used to decide: MET (2026-10-13)

**Read from** `results/BOXRUN3/GSE315366_rows.csv` (Box Run 3, read once on 2026-10-09); rows and the scored summary copied into
`data/CSCORE_COMMISSIONING/band_test_GSE315366_rows.csv` and `band_test_GSE315366.json`.

**Bar, fixed before these readings existed:** at least 95 % of healthy arrays not above the band's upper limit 1.152 (one-sided
reading, blocks of 10). **Result: 34 of 35 not above, 97.1 %. MET.** The one above reads C_rel 1.195; median 0.988, range
0.853-1.195. Tare scope: 31 arrays tared against their series, 4 against their slide (3-34 references each).

**Independence.** The band was decided on 617 arrays from 19 series (`data/DEV_CSCORE_TARE_01_arrays.csv`); GSE315366 is not
among them. **Order, by commit timestamp on 2026-10-09:** the blocks-of-10 band 06:48 (b6c03e0c), the one-sided reading and the
repeat bar tied to detection 07:04 (642818bd), these readings 07:14 (b77610ed). The definitions were fixed before the data were
read, and the timestamps are the evidence.

**Scope, stated plainly.** These are peripheral whole bloods. That matches the band's own population (555 whole blood, 62
isolated neutrophils of the 617) and extends no commissioned scope: Met-A's commissioned scope remains isolated neutrophils.

**What is still open for the C-score.** (a) each further healthy EPIC v1 laboratory, since the plan asks for more than the first;
(b) new same-person replicates from a set that had no part in deciding the repeat bar (it is met at 0.062 against <= 0.125 on the
existing replicates). Candidate for (b) from the gold-mine map: GSE318669 (NIEHS, 54 arrays, repeat measures, 450K - platform has
to be checked against the band before it can serve).

**Not a milestone:** one bar of a development commissioning met; the C-score is not proposed for commissioning until (a) on
further laboratories and (b) on new replicates are met.

## Every number in this note recomputed from committed inputs (2026-10-13)

`data/CSCORE_COMMISSIONING/plan_numbers.py` (output `plan_numbers_output.txt`) recomputes the spread and repeatability
figures from the committed arrays file and a committed person map for GSE250556, and prints the earlier runs' figures with
their run and status from `historical_numbers.csv`. Three agree to the digit: the replicate spread 0.089 (0.0891, C tared to
series), and the derived block-count sampling errors 0.130 and 0.058 (0.1291, 0.0577). Two do not: **the committed arrays file
holds 617 arrays where step 3 says 641**, and step 3's spread figures (within-person 0.138, healthy 0.175) belong to that
larger blocks-of-50 set, which the blocks-of-10 run superseded. Recomputed on the committed file: healthy SD 0.1707 tared to
series and 0.1764 to slide; same-person within-SD 0.0891 tared to series, 0.1294 to slide, 0.0742 untared.

**Consequence for the repeat bar.** The bar in force is an absolute one, same-person within-SD of C_rel <= 0.125. On the
committed file that reads 0.0891 tared to series (met) and 0.1294 tared to slide (not met). Which tare a report uses is
therefore load-bearing, and the bar is recorded as **met on series-tared readings only**, pending the new replicates.

**Synthetic test:** DEV_CSCORE_MOSS_01/insilico_moss_01.py, DEV_CSCORE_MOSS_01/insilico_moss_01_output.txt, DEV_CSCORE_MOSS_01/insilico_moss_01_rows.csv, CSCORE_COMMISSIONING/plan_numbers.py, CSCORE_COMMISSIONING/plan_numbers_output.txt

Step 2's constructed positive control is the synthetic test: clustered loss planted at a known size must read above the band,
scattered loss of the same size must read inside it.

**Prior reading:** the band's own 617 arrays (19 series, including GSE110530, GSE142512, GSE225544, GSE226298, GSE247195,
GSE250556) built and decided the band and are not tests of it; GSE315366 was read once in Box Run 3 for clonal haematopoiesis
(a different question) and is used here only as the held-out test of the band, which it had no part in deciding.
GSE123914 (blood over a one-year period) was read earlier and is named in this note only as a set the search CANNOT use; it is not
read again here and no reading or seal rests on it.

## Test (a) on a second laboratory: pre-registered 2026-10-13, before any array is fetched

**Set.** GSE184269 (NIA-IRP, NIH; the GESTALT healthy-donor series; EPIC v1, raw IDATs for every sample). Two strata, listed in
`data/CSCORE_COMMISSIONING/band_test_GSE184269_samples.csv`: **granulocytes (10 arrays)**, which is Met-A's commissioned
specimen scope, and **whole blood (24 arrays)**, which matches the band's majority population (555 of 617). Healthy donors,
ages and sexes recorded in that file.

**Independence.** None of its eight authors appears on any of the 19 series that decided the band (`CANON/geo_contributors.csv`,
completed for all 19 today). The set has never been read by any version of this work (`CANON/data_spent.csv`).

**Bar, fixed now.** Pooled across the two strata (34 arrays): at least 95 % not above the band's upper limit 1.152, one-sided
reading, blocks of 10 - so at most 1 array above. Each stratum is also reported on its own, and the granulocyte stratum is
labelled indicative rather than decisive, because with 10 arrays the proportion can only move in steps of 10 %.

**Reading.** The chain unchanged: Stage 1 from the IDATs, then the C-score with the same-run tare (references of the same
specimen type within this series). Any array the chain refuses is reported with its refusal and excluded from the proportion,
with the count stated.

**If it is met** together with the first laboratory, what remains for the C-score is new same-person replicates for the repeat
bar, which the recomputation above shows is met on series-tared readings only.

**In plain words.** We have two meters. The first says how far a cell has drifted from healthy. This one, the C-score, says
*how* it drifted: in clumps, or scattered. Clumps are what cancer leaves. To trust it we measured the healthy range on 617
blood samples from 19 studies, then asked whether that range holds on healthy people from a laboratory that had no hand in
setting it. It did: 34 of 35 read inside, where we allowed up to 1 in 20 outside. **What it would take to finish.** The same
check on more healthy laboratories, and a set where one person was measured several times, to confirm that two readings of
the same person differ by less than a quarter of the smallest change we want to catch. On the samples we have, that holds
when a person is compared with others measured in the same batch and does not hold when the comparison is only within one
chip, so a report has to say which comparison it used.

### Amendment to that pre-registration, written before any array was scored (2026-10-13)

The chain refuses **granulocytes** at intake: its accepted blood specimens are whole blood and isolated, sorted or purified
neutrophils, and granulocytes - which also contain eosinophils and basophils - have no reference of their own
(`chain/stage_0_intake.py`, `ACCEPTED_SPECIMENS`). GEO labels this series' sorted fraction "Granulocytes", so those 10 arrays
will be refused, and **relabelling them as neutrophils to get a number would be routing around a scope rule, which we do not
do.** I should have checked the accepted list before naming that stratum; the refusal was predictable from the chain's own
code rather than discovered.

**The bar therefore rests on the 24 whole bloods** (the band's majority population): at least 95 % not above 1.152, so at
most 1 of 24. The 10 granulocyte arrays are reported with their refusal and excluded from the proportion, as the
pre-registration already required for refusals. Their calibrated betas are kept, so they can be read the day granulocytes have
a reference - which is now a job on the road ahead, not a relabelling.

## Test (a) on the second laboratory not used to decide: MET, on series-tared readings, by one array (2026-10-13)

**Read from** the raw IDATs of GSE184269 (NIA-IRP, NIH; GESTALT healthy donors; EPIC v1), fetched after the bar was committed and
calibrated by the chain's Stage 1 unchanged, then read by `conductor_v3.run_neutrophil` and the C-score with the tare copied verbatim
from the first laboratory's driver (`read_band_test.py` refuses to run if that copy has drifted). Scripts and outputs:
`data/CSCORE_COMMISSIONING/fetch_band_test.py`, `score_band_test.py`, `read_band_test.py`, `band_test_GSE184269_rows.csv`,
`band_test_GSE184269.json`.

| | |
|---|---|
| bar, committed before the arrays were fetched | at least 95 % of the whole bloods not above 1.152, pooled, one-sided |
| whole bloods listed | 24 |
| withheld by the chain | 1 (GSM5583025: only 5325 of 6000 neutrophil sites measured, 5400 required - the coverage rule of 2026-10-04) |
| scored | 23 |
| above 1.152 | 1 |
| not above | 22 of 23 = **0.9565**, so the bar is **MET** |
| spread | C_rel median 0.9992, range 0.871 to 1.1815 |
| granulocytes | 10 refused at intake, as the amendment said they would be; not relabelled |

**Three things that must travel with this result.**

1. **It is met by one array.** Had the withheld array read above the limit, the proportion would be 22 of 24 = 0.9167 and the bar would
   **not** be met. The scorer now computes and prints that sensitivity (`verdict_if_every_withheld_array_were_above`) so it cannot be
   left out of a summary.
2. **Series-tared, not same-run.** GESTALT ran one array per slide (24 slides for 24 arrays), so a same-run tare was impossible: every
   reading is tared to the series. The first laboratory's repeat bar is likewise met on series-tared readings only, so this test does not
   settle the slide-scope question - it inherits it.
3. **Whole blood, not the commissioned specimen.** These are whole bloods, consistent with the band's own majority population (555 of 617)
   and not an extension of Met-A's commissioned neutrophil scope.

**What it would take to finish test (a).** Each further healthy EPIC v1 laboratory gets read as the plan says, not only the first two;
and at least one of them must have several arrays per slide, so the slide-scope tare can be tested rather than inherited.

**In plain words.** Our second meter says how a cell's drift is arranged - in clumps, which is what cancer leaves, or scattered, which is
ordinary wear. Before trusting it on a patient we have to know it does not cry wolf on healthy people it has never seen. We set the pass
mark first - at most 1 in 20 healthy samples may read outside the healthy range - and then read 24 healthy blood samples from a second
laboratory, the National Institute on Aging, that had no hand in setting that range. One sample the instrument refused to read, because too
few of its sites came through; of the 23 it did read, 22 read inside the range. That clears the mark, but only just: if the refused sample
had read high, it would have failed. **What it would take to be convincing:** more healthy laboratories, and one where several samples share
a chip, so we can check the stricter chip-level comparison instead of inheriting it.

**Not a milestone:** one of several healthy laboratories; test (a) is not complete until the further laboratories in the plan are read and
the slide-scope tare is tested rather than inherited.

## Search for the last open check: same-person replicate arrays (metadata only, 2026-10-13)

What is still open for the C-score is the repeatability bar on a set that had no part in building the band. Searched GEO
(E-utilities, `gds`) with four query families: technical replicates on EPIC v1; blood with reproducibility, repeatability,
test-retest or intraclass wording; blood with same-individual or repeated-measurement wording; and blood, longitudinal or
two-time-point, healthy or control, on EPIC v1 and 450K. Every accession already in `CANON/data_spent.csv` (164 of them) was
excluded automatically. Nothing was downloaded.

**The obvious set is already spent.** GSE250556 ("Measuring variability in Illumina MethylationEPIC microarrays") is exactly the
right design, and it is the set the band's repeatability estimate was built on. GSE123914 (blood over a one-year period) is also
spent. Neither can serve as the held-out test.

**Candidates found, each with the caveat that decides it:**

| set | n | platform | design | caveat |
|---|---|---|---|---|
| GSE112611 | 402 | EPIC v1 | Crohn's disease and controls, longitudinal | the control subset must be checked for same-person pairs; disease cohort, so healthy status needs reading per sample |
| GSE184202 | 24 | EPIC v1 | healthy volunteers, blood before and after an intervention | 12 pairs at most, and an intervention sits between the two draws |
| GSE61151 | 188 | 450K | white blood cells, temporal stability - the ideal design | wrong platform for the band, which is EPIC v1; usable only when the 450K band is commissioned |
| GSE182919 | 112 | EPIC v1 | true technical replicate pairs with IDATs | cultured cells with a drug treatment, not donor blood: specimen mismatch |

**Next step, in order.** Read GSE112611's sample metadata and count healthy-control donors with two or more arrays. If that yields at
least seven pairs, it is the held-out test set and gets written into this plan before any array is fetched. If it does not,
GSE184202 is the fallback and its intervention is declared in advance as a caveat, not discovered afterwards. GSE61151 is reserved
for the 450K band.

**Note on what a "replicate" buys here.** A technical replicate (same DNA, two arrays) tests the instrument alone. Two draws from the
same healthy person weeks apart is a STRICTER test, since it includes real biological variation, and passing it would be the stronger
claim. Whichever is used, which kind it is gets stated with the result.

### Checked from sample metadata, 2026-10-13 (nothing fetched): both front-runners are out

- **GSE112611 is unusable at any priority: it has NO raw arrays.** 402 samples, zero IDAT files - submitter beta tables only. The
  chain's Stage 1 calibrates from IDATs, so this set cannot be read by our instrument at all. (Its design was otherwise right:
  74 non-IBD controls, with baseline and follow-up arrays.)
- **GSE184202 has the right shape and the wrong donors.** 24 arrays, all with IDATs, EPIC v1, blood DNA, exactly 12 subjects with
  2 arrays each - but the diagnosis field reads **Burnout for 22 of 24 and control for 2**. So it is a patient cohort with an
  intervention between draws, and only one control donor has a pair. Not a healthy-donor replicate set.
- **GSE291053** (32 arrays, IDATs, EPIC v1) is dialysis patients' mononuclear cells. Not healthy, and not the commissioned fraction.

**Where that leaves the last open check.** No healthy-donor, EPIC v1, raw-array, same-person replicate set has been found yet. Three
honest options, in the order they should be tried:
1. **Widen the search terms rather than the standards** - plate-level designs often carry unlabelled repeat arrays (the same donor
   appearing twice in a sample sheet), which no summary-level query will find. That means reading sample sheets of healthy EPIC v1
   blood series directly, which is free but slower.
2. **Commission the 450K band**, which makes GSE61151 (188 arrays, white blood cells, temporal stability - the ideal design) usable.
   This is the option that buys the most, since it also opens the larger 450K archive to the C-score.
3. **Accept a technical-replicate set with a declared specimen caveat** (e.g. GSE182919, cultured cells with treatments). Weakest,
   and only worth doing if the first two fail, with the caveat stated in advance.

Until one of these yields a set, the C-score's repeatability bar remains MET ONLY on data that helped build the band, and that is
how it is reported.

