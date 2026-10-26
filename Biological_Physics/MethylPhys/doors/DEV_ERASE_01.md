# DEV-ERASE-01 — the half and half: one half writing, one half erasing (prediction sealed 2026-10-14, before any data set is searched)

**The idea (author, 2026-10-14).** "What is the half and half in DNA methylation and now I know, one is erase and one is write." It is the
virial pair of IAM's Law as Part I states it: when a system binds, the well splits into two equal halves, the kinetic it keeps and the heat
it writes (K = Q = |V|/2). In the cell's copy channel the loss has two routes, the writer's own misses and active erasure (TET enzymes); the
author's statement is that they are the two equal halves.

**Why it is worth testing.** The writer alone, measured outside the cell (DNMT1, DEV-WRITER-01), would hold a copy error of 0.0155;
healthy cells hold 0.032. Two equal halves of loss lower the holding by exactly one bit, kT ln 2, which from the writer's measured range
gives 3.17-3.69 kT around the measured 3.41. That match was found after the fact on a known number, so it does not count. What counts is the
prediction below, which no data here has been read for.

**Prediction.** Remove the eraser and the copy error halves. Written as the ratio of the copy error in eraser-removed cells to that in the
same laboratory's control cells, the ratio cancels laboratory, species and instrument offsets. Computed in
`data/DEV_ERASE_01/prediction_erase_01.py` (output `prediction_erase_01.txt`): equal halves give about 0.51; the two virial alternatives
(eraser twice or half the writer) give about 0.34 and 0.67; no eraser share gives 1.

**Rule, fixed now.** MET if the ratio is within 0.40-0.60. NOT MET if it is 0.85 or above (the extra loss is not erasure; missed restores
or something else). Anything else is INCONCLUSIVE and reported as such.

**Before any data is opened.** (1) Metadata-only search for single-molecule sequencing of eraser-removed cells with same-laboratory
controls: TET1/2/3 triple knockout, and IDH1/2-mutant cells or tumors (2-hydroxyglutarate blocks TET); every set found is declared in
`CANON/data_spent.csv` before reading. (2) The copy error is read by Stage Q, the same reading behind 0.032; the IAM-A instrument is still
on hold at commissioning, so any result is development, not commissioned, until it passes. (3) Caveats written now: knockout cells are often
stem cells and adapt over time; IDH mutation also changes other enzymes; mouse cells are another species (the ratio form is chosen for this).

**Synthetic test:** DEV_ERASE_01/synthetic_erase_01.py, DEV_ERASE_01/synthetic_erase_01.txt
Simulated molecules with copy errors planted at 0.032 and at half that, read by Stage Q's own counting: ratio 0.5232 against the 0.517
expected from counting isolated errors only, bar 0.45-0.55: BAR MET. Disclosed: the bar was in the file before it ran, but the file was
committed after (same session, 2026-10-14). A planted check on real healthy molecules is still to do before any real set is read.

**Planted check on real healthy molecules (committed 2026-10-14 before it runs):** `data/DEV_ERASE_01/planted_real_erase_01.py` and
`.sh`. Two healthy Loyfer files already spent for development (granulocytes GSM5652313, CD4 T cells GSM5652281). Arms: unchanged; half of the
real isolated errors turned back (the eraser-removed picture); about as many new ones added (the boosted-eraser picture). Bars: halved/base
0.45-0.55 and the base file's two halves within 0.95-1.05; doubled/base reported, no bar (amended before any real file was read: a
code smoke test on a small made-up file read a doubling as 1.77, because a new error beside an old one leaves neither isolated).

**Planted check on real healthy molecules, run once (job 091bf539): MET.** Granulocytes (33.9 M opportunities): halved/base 0.5316, halves
1.0005. CD4 T cells (29.3 M): halved/base 0.5285, halves 0.9995. A true halving reads about 0.53 on real molecules, inside the sealed 0.40-0.60.
The doubled arm read 1.5962 and 1.6142 (reported, no bar): on real reads an added error often lands beside another, so a boost is read compressed.
That matters for the vitamin C side arm, which is reported only.

**The sets, found by a metadata-only search and declared before any is read (2026-10-14).** GEO titles, summaries and sample fields only
(`data/DEV_ERASE_01/erase_search_geo_metadata.json`, `erase_sets_runs.csv`); no data file opened.

PRIMARY units: constitutive TET1/2/3 removal in cell lines, whole-genome bisulfite reads, wild-type controls from the same laboratory:
| unit | set | eraser removed | control |
|---|---|---|---|
| U1 human ES cells (HUES8) | GSE89727 + GSE126958 (one laboratory; pooled per genotype) | TET1/2/3 knockout | HUES8 wild type |
| U2 mouse ES cells | GSE223576 day 0, 3 + 3 replicates | TET knockout | TET wild type |
| U3 mouse ES cells | GSE248933 | Tet triple knockout | Tet wild type |
| U4 mouse neural stem cells | GSE248933 | Tet triple knockout | Tet wild type |
| U5 mouse ES cells (E14) | GSE56986 | Tet triple knockout | E14 wild type |

SECONDARY, reported, not counted in the verdict: tissue conditional knockouts, where the deleted fraction of cells is unknown and dilutes
the ratio toward 1 (GSE288100 retina 3 + 3; GSE174047 skeletal 1 + 1); and the side arms, where the eraser is only turned down or up:
IDH1-mutant cholangiocarcinoma cells with the IDH1 inhibitor AG-120 against DMSO (GSE265854; prediction: inhibitor/DMSO above 1) and
IDH1-R132H cells with vitamin C against untreated (GSE87623; prediction: vitamin C/untreated above 1). Not used: GSE85355 (no wild type),
GSE129470 (targeted capture, not whole genome), GSE55049 / GSE55574 (no bisulfite reads), and every other candidate listed in the metadata file.

READING. Each sample: 10,000,000 read pairs taken as 20 equal blocks spread evenly through the run (not the head of the file: the cause test
found a head-of-file effect), aligned with bwameth to the box's hg19 or mm10 index, converted to .pat with wgbstools as in the cause test,
and read by Stage Q's counting (pat_site_table). A unit's ratio is the pooled copy error of its knockout samples (sum of isolated errors over
sum of opportunities) over that of its wild-type samples.

VERDICT, fixed now. Each primary unit gets the sealed rule's verdict (MET 0.40-0.60; NOT MET 0.85 or above; otherwise INCONCLUSIVE).
DEV-ERASE-01 is MET if at least 3 of the 5 primary units are MET and none is NOT MET; NOT MET if at least 3 are NOT MET; otherwise
INCONCLUSIVE. A unit with fewer than 100,000 opportunities per genotype is UNREADABLE and is reported as such, not counted. Caveats, written
now: cultured stem cells adapt to knockouts over passages; mouse is another species (the ratio form is chosen for this); a knockout can change
how many sites are methylated, which changes which molecules qualify, so the qualifying share is reported beside every ratio.

**In plain words.** We think methylation's half and half is writing and erasing. If that is right, cells whose eraser is switched off should
make half as many copy errors as normal cells from the same laboratory. We wrote down the number and the pass and fail lines before looking
at any such data. What it would take to finish: the planted check on real healthy molecules, at least one eraser-removed set with same-laboratory controls has been found and declared, and its ratio has been read once against the lines above, whatever it says.

Related: PHY-040 (the author's half-law), PHY-006 / HYP-002 / SCO-033 (this test's earlier rows), DEV-WRITER-01/-02, LEV-019/-044, LEV-046.

## Outcome, primary units read once (2026-10-15): NOT MET

All five primary units complete; read by the committed `erase_verdict.py` (`data/DEV_ERASE_01/erase_primary_units.csv`,
`erase_primary_verdict.txt`). The side arms (A1, A2, S1, S2) are reported only and are still reading; the full verdict file is written once,
by the job, when they land. They cannot change the primary verdict.

| unit | eraser removed / control | verdict |
|---|---|---|
| U1 | 0.9608 | NOT MET |
| U2 | 0.7930 | INCONCLUSIVE |
| U3 | 0.8560 | NOT MET |
| U4 | 0.8938 | NOT MET |
| U5 | 0.8514 | NOT MET |

Primary units MET 0, NOT MET 4 of 5: **DEV-ERASE-01 NOT MET** by the rule committed before any read.

**What it rules out.** A per-site half: removing the eraser does not halve the held error on the counted molecules. The step
"the eraser's loss equals the writer's" (r = 1) is not supported where we count. The middle mark stays MEASURED (3.41 kT), as the foundation
file already says; its derivation is open.

**What it shows.** In all five units the held error falls when the eraser is removed, by 4 to 21 percent. The eraser carries a small share of
the loss on the counted molecules, not half. This agrees with the overnight whiteboard note (development/WHITEBOARD_2026-10-15.md): on the sites
every cell keeps fully methylated, the held error sits inside the writer-alone range.

**In plain words.** Switching off the eraser lowered the error a little in every set, never by half, so the half-and-half rule does not hold
site by site where we count. The middle line itself is unchanged.

## Side arms (reported only, 2026-10-15)

The job's own verdict file (`data/DEV_ERASE_01/erase_verdict.txt`, written once by the job after every sample) gives the primary units exactly
as recorded above, and the side arms, which carry no bar:

| arm | set | change | eraser treated / control |
|---|---|---|---|
| A1 | GSE265854 | IDH1 inhibitor (AG120), eraser freed in IDH-mutant cells | 1.1046 |
| A2 | GSE87623 | vitamin C, eraser boosted | 0.9970 |
| S1 | GSE288100 | eraser removed (secondary set) | 0.8923 |
| S2 | GSE174047 | eraser removed (secondary set) | 0.9181 |

The secondary removals read like the primary units (a small fall, not a halving). Freeing or boosting the eraser did not lower the held error.

**The one-sample human timing test (2026-10-23; erase_test run 39911507208, commit 09c50ab3; GSM2387332; `data/DEV_ERASE_01/timing_01/timing_01_log.txt`).** On one GitHub machine: the hg19 index built in 6,814 s, 10,000,000 read pairs fetched in 135 s and aligned in 6,827 s, molecules written in 306 s. The count did not run: the workflow installed only numpy, pandas and scipy, and run_sample.py stopped on a missing plotting package. Fixed (the workflow installs the chain's requirements.txt). So 10 million pairs fit one machine in about 4 hours, the index being half of it; a sample's full read count decides whether it needs splitting (plan 14.2).

**The count, from the saved molecules (erase_test run 39988716983; `data/DEV_ERASE_01/timing_01/count_01_q2.json`).** run_sample.py --dev-count-only on the molecules run 39911507208 saved, with no second alignment: held error 0.04198 over 343,973 chances, loose share 0.0765, 11,263 tiles. IAM-A's intake refused a reading, as it should (cultured cells have no healthy shape reference), and the count is recorded, flagged. So a human sequencing sample now goes from raw reads to its count through the one command on GitHub's machines: about 4 hours to align 10 million pairs, then the count. Two reruns before it stopped at setup (lesson C39: preflight, the pinned Python version, healpy, the chain's own Python).

## The recount on the commissioned counter

Each sample's molecules counted by `run_sample.py` (Stage Q2) on GitHub's machines (erase_test runs 39994306888 to 39994446012, commit 23daa9f4;
counts in `data/DEV_ERASE_01/recount_01/`, the units in `recount_01/recount_units.csv`), pooled per genotype as the sealed rule pools them
(errors summed over chances). The sealed rule's bands applied to the new ratios; the earlier verdict stands, sealed, as recorded above.

| unit | eraser removed / control, recount | earlier count | the rule on the recount |
|---|---|---|---|
| U1 (first pair only: GSM2387332, GSM2387333) | 0.9925 | 0.9608 | NOT MET |
| U2 | 0.7911 | 0.7930 | INCONCLUSIVE |
| U3 | 0.8642 | 0.8560 | NOT MET |
| U4 | 0.9027 | 0.8938 | NOT MET |
| U5 | 0.8518 | 0.8514 | NOT MET |

U1's second pair (GSM3618718, GSM3618720) stopped at the machines' 6-hour limit while aligning, nothing saved; it waits on a split run (plan
14.5). **In plain words.** On the commissioned counter the result is the same: removing the eraser lowers the held error by a tenth to a fifth,
never by half, so the half-and-half rule is not supported where we count. The derivation of the firm rate needs another route.

## A maintenance knockout: UHRF1 (written and committed before it runs; plan 14.10)

**Prior reading:** GSE116229 (the maintenance knockout): only the Reh control and UHRF1 knockout counts were read for this record; the series is spent for the leukemia replication (DEV_ALL_INFO_01.md).

**The question:** UHRF1 brings DNMT1 to each newly copied strand; without it the cell cannot hold a mark through division. The published
information-theory study of pediatric leukemia (Nature Biomedical Engineering 2021) knocked it out in the Reh leukemia line and found
methylation lost and entropy raised genome-wide. What does IAM-A's count see on the same molecules?

**Data spent (author: "Knockout pair only"):** GEO GSE116229, WGBS of Reh with a non-targeting control (GSM4914798, SRR13085130) and with
UHRF1 knocked out (GSM4914799, SRR13085131); the study's leukemias and normal B-cell precursors stay unread. `data/DEV_ERASE_01/uhrf1_samples.csv`,
the eraser recount's pipeline unchanged (`erase_gh.sh`, the same fixed 10 million read pairs from each), counted by `run_sample.py`
(Stage Q2, count only: a cultured line has no healthy shape reference).

**Written before it runs, both outcomes informative:**
- **The held error rises in the knockout:** maintenance failure appears as holes in held molecules, the route the firm-rate derivation
  needs.
- **The loose share rises and the held error barely moves:** maintenance failure removes whole molecules from the held class (as the
  decitabine test found), and the floor's derivation must take the loose route.
Nothing scored.

**The maintenance knockout, read** (`data/DEV_ERASE_01/uhrf1_01/uhrf1_summary.txt` and both bundles; scored by `uhrf1_score.py`, written
before either count existed): **the first outcome.** Held error 0.03216 in the control, 0.09407 in the knockout (2.925 times); loose share
0.0490 in the control, 1.0 in the knockout (every tile loose). Maintenance failure makes holes in the molecules that are still held, the route
the firm-rate derivation needs, and it also removes most molecules from the held class: the knockout's held opportunities are 121,693 against
the control's 1,879,870.

**What it does and does not settle:** missed restores during maintenance are a real source of held error, so they are the leading candidate
for the second term of the firm rate (the eraser supplied only a small part of it). A total knockout cannot size that term in a healthy
cell. Sizing it needs a graded reduction of maintenance, or the division route (replication error, plan 14.3) measured on one cell type
resting and dividing.

## The division route: dividing and non-dividing cells of one type (written and committed before any count exists; plan 14.3)

**Prior reading:** GSE48580 was read by the earlier work (PROC-LINES-02, methylation channels); it was never counted by Stage Q2.

**The question:** the firm rate is about twice the writer's own miss rate; erasure supplies only a small part of the second half, and missed
restores during maintenance do make held error (the UHRF1 knockout). Does dividing add to it (replication error, the author's division
route)? **Data:** GSE48580, IMR90 lung fibroblasts by whole-genome bisulfite sequencing: proliferating (3), replicatively senescent, no
longer dividing (3), and dividing past senescence with SV40 large T antigen (3); `data/DEV_ERASE_01/division_samples.csv`. Each counted by
the eraser recount's pipeline unchanged at 10 million read pairs (`erase_gh.sh`, `run_sample.py --dev-count-only`); scorer
`division_score.py`. **Written before, both outcomes informative:**
- **Held error lower in the senescent cells than the proliferating ones, loose share aside:** dividing adds held error, and the difference
  sizes the division term, a candidate for the firm rate's second half.
- **Held error the same within the replicates' spread:** dividing adds none at steady state; the second half is maintenance itself.
**Stated before:** senescent cells reached senescence by dividing, and senescence reorganizes chromatin; IAM-A reads the reorganization in the
loose share, not the held error. SV40 also disables p53 and Rb, so its reading is reported, not used to size the term. Nothing scored.

**The prebuilt index and the molecule step:** the first jobs to use the index from the release (the division counts and the Reh control's split alignment) aligned and sorted, then stopped at bam2pat (exit 34); the jobs reported success because the script's exit code was only posted to issue 32. Fixed: bam2pat's log is kept; on a failure the genome dictionary is rebuilt from the fetched FASTA the committed way (init_genome, the CpG count checked) and bam2pat is run once more; the exit code is uploaded with each job. No count was made on the failed jobs; all were sent again.

**The division route, read** (`data/DEV_ERASE_01/division_01/division_summary.txt`; all 9 samples, exit 0; the molecule step ran after the
dictionary was rebuilt, as above): **the first outcome as written, and small.** Held error pooled 0.05070 in the proliferating cells (0.04960,
0.05180, 0.05070) and 0.04828 in the senescent cells (0.04725, 0.04945, 0.04813); the two sets of replicates do not overlap. The difference is
about a twentieth of the held error, so dividing adds at most a small part of it, far less than the firm rate's second half. **Reported, not
used to size the term (stated before):** the SV40 cells, which keep dividing, read 0.04825, the senescent level, so the difference may come
from the culture's state rather than division itself. The loose share rises with senescence (median 0.5807 to 0.6353) and further past it
with SV40 (0.6877), as stated before for the chromatin's reorganization. **What it settles:** replication error is not the firm rate's
missing second half; it is at most about a twentieth of the held error at steady state. The second half stays with maintenance itself (the
UHRF1 knockout's route).

## The graded maintenance test (written and committed before any count exists; plan 14.1)

**The instrument's job:** IAM-A's counter reads the held error of cells whose maintenance is turned down by a known amount, and the way it
rises says which term of the law the maintenance supplies.

**Prior reading:** none. GSE184116 (mouse embryonic stem cells, bisulfite sequencing, 18 runs) has only had its sample labels read.
**Data:** the DNMT1 inhibitor GSK-3484862 at 0 (DMSO), 2 and 10 uM, from two suppliers, on days 6 and 14; 5-azacytidine at 100 and 300 nM
with its own DMSO control; Dnmt1/3a/3b triple knockouts (no methylation) on days 6 and 14; three deep runs (DMSO day 14, 2 uM days 6 and
14). `data/DEV_ERASE_01/graded_samples.csv`. Each run counted by the eraser recount's pipeline unchanged (`erase_gh.sh`, mm10), at most 10
million read pairs (most runs hold fewer), the deep runs in full by the split alignment.

**What the law says, stated before:** a held mark's error is e = 1/(1 + exp(E)), E = phi M (the canon's E_hold_meth, measured on healthy cells' held error). Turning
maintenance down lowers E, and two physically different ways of doing so are told apart without fitting anything:
- **R, restoring slowed:** maintenance restores a lost mark at a rate; cutting the rate by a factor f lowers E by ln(1/f) everywhere, the
  same amount in every region. Across 2,000-CpG tiles, E at a dose against E in the control has **slope 1 and a negative intercept**.
- **Phi, less of the drive spent holding:** E = phi M with phi lowered by a factor g; every region's E shrinks in proportion to its own E.
  Across tiles, **slope g below 1 and intercept 0**.

**Bars (at most three):**
1. **Synthetic first, through the one command:** constructed molecules over mm10 tiles whose E is shifted uniformly (by 0.7) and, separately,
   scaled (by 0.8) are read; the scorer recovers each slope and intercept within 0.05 and calls each correctly.
2. **The counter reads maintenance:** pooled held error rises with dose, DMSO < 2 uM < 10 uM, on both days and for both suppliers.
3. **The discriminating reading, on the deep pair (DMSO day 14 against 2 uM day 14):** slope and intercept of E(dose) on E(control) across
   tiles, the control's E taken from one half of its molecules and the slope's errors-in-variables bias removed with the other half (both
   halves independent); **R** if the slope's 95 % interval holds 1 and the intercept's lies below 0; **Phi** if the slope's interval lies
   below 1 and the intercept's holds 0; otherwise neither, recorded as such.

**Reported, not scored:** the triple knockouts (held error undefined without held molecules), 5-azacytidine (a different mechanism: it is
built into the DNA and traps the enzyme), day 6 against day 14 at each dose (equal within intervals would mean a steady state). Mouse
embryonic stem cells run at 37 C like human cells, so M is the canon's; the control is each series' own DMSO cells (self-reference).

**Synthetic test:** `DEV_ERASE_01/graded_synthetic.py` (through the one command)

Scored by `data/DEV_ERASE_01/graded_score.py`; output `data/DEV_ERASE_01/graded_01/graded_synthetic.txt`. New in the chain for this test: `stage_q2_selfref.count_pat_tile_errors` and
`run_sample.py --dev-count-tiles` (the held error tile by tile, in two independent halves of the molecules), written into the counting call
of `erase_gh.sh`.

**Graded test, bar 1 (synthetic), read: met.** Three runs, all disclosed: the first two stopped on faults in the synthetic itself, not in the
counter or scorer. Run 1 planted E from 2.0, so a shift of -0.7 asked for more lone errors than a molecule can carry. Run 2 planted each
call's error rate as p (1 - p)^2 = e, which ignores that the counter only counts molecules it holds as marked, and it read back slopes off
by 0.06 and 0.2. Run 3 plants each rate through the counter's own held-molecule rule, enumerated over every pattern, and recovers both cases
within 0.05: the uniform shift reads R (slope and intercept as in the output), the proportional scaling reads Phi. What run 2 teaches is
recorded for the real counts: the lone-error rate depends on which molecules count as held.

**The mouse index, built once:** the graded test's three deep runs are aligned by the split alignment, 20 machines each. The mm10 bwa-meth
index and wgbstools' mm10 dictionary are now built once by `build_index.yml` through `erase_gh.sh`'s own index block and published as release
files with the stream's SHA-256, as the hg19 index is; each job streams them in and checks them, and builds them itself if the fetch or the
check fails. Bar 3 reads the deep pair (DMSO day 14 against 2 uM day 14): on the small runs a dosed sample keeps too few held-molecule
opportunities for the tile reading (only 2 tiles pass the stated minimums).

**Graded test, bar 2, read: not met as written** (`data/DEV_ERASE_01/graded_02/graded_summary.txt`, 15 of 18 runs counted; the three deep
runs wait for bar 3). The held error roughly doubles from the DMSO control to 2 uM in every series (day 6 and day 14, both suppliers), and the
two suppliers agree. From 2 uM to 10 uM it barely moves: it rises in three of the four series and falls in one (day 6, ChemieTek), by less
than about two standard errors of counting at these opportunity counts, so the bar's strict ordering fails on that one. Read as written:
the held error saturates by 2 uM. 5-azacytidine, a different route to the same loss, rises with dose at both doses (100 and 300 nM).
Bar 3 (the deep pair, tile by tile) is what separates a slowed restoring rate from a smaller holding fraction; it runs once the mouse index
release is built.

**Bar 3 amended before the deep pair is read (disclosed):** tiles of 50,000 CpGs (the counter's 2,000-CpG tiles summed in genome order,
25 at a time), the same minimums applied to the summed tiles. Reason, from the small runs' own tile counts scaled to the deep runs' read
depth: about 14 tiles of 2,000 CpGs would pass at 2 uM, too few for a slope; about 290 of 50,000 CpGs would. The control's tile-to-tile
spread in held-mark energy survives at that size (about 0.10 in log, against counting noise that falls below it at the deep runs' depth).
Everything else in bar 3 is unchanged. **Synthetic test:** DEV_ERASE_01/graded_synthetic_v2.py, DEV_ERASE_01/graded_03/graded_synthetic_v2.txt
(met: the planted shift read as slope 1.0128, intercept -0.7391, called R; the planted scaling as slope 0.8065, intercept -0.0185, called
Phi). If bar 3 is not met, plan 14.1 moves to the human degron series; IAM-A's baseline stays calibrated either way.

**The deep dose sample and GitHub's six-hour limit:** the deep 2 uM sample has 373,587,918 read pairs, 5.1 times the deep control's
72,835,211, whose 20 chunks took 1.1 to 1.5 hours each; its chunks are expected to take 5.5 to 7.4 hours, past the job limit. The split
alignment now re-sends a chunk as its two halves (input split N:OFF: chunk k of 20 is exactly chunks 2k and 2k + 1 of 40, since the
planner cuts at k x total / n; checked on this sample's own pair count) and merges chunks of both splits (keepchunks k or k/N). Chunks that
finish are kept; any that time out are re-sent as halves.

**Bar 3, read: neither** (`data/DEV_ERASE_01/graded_04/graded_deep_summary.txt`; the deep control's 20 chunks, 32,157,060 molecules; the
deep 2 uM sample's 17 chunks with its 3 timed-out chunks re-aligned as halves, 23 pieces, 169,382,384 molecules). Across 430 tiles of
50,000 CpGs: **slope 0.2582 [0.1620, 0.3624], intercept +1.4171 [+1.1228, +1.6903]**. Neither R (slope 1) nor Phi (intercept 0): under
the inhibitor the tiles' holding energies move most of the way to one common value, so the regional differences the control holds are
largely lost, rather than shifted or scaled. **Caveat:** at 2 uM most marks are gone and the counter reads only molecules still held, the
survivors, which by itself pulls tiles toward a common level; the reading cannot tell that from the biology. **What follows, as stated
before:** plan 14.1 moves to the human degron series (maintenance turned off over days, so the counted molecules stay many); IAM-A's
baseline stays calibrated.

## The degron series: maintenance turned off over days (written and committed before any of its reads are aligned; plan 14.1)

**Prior reading:** GSE236026 (HCT116 with UHRF1 or DNMT1 tagged for degradation, whole-genome bisulfite, days 0 to 12 in triplicate):
labels and read counts only. **Spent for development:** replicate 1 of the UHRF1-AID and DNMT1-AID lines at days 0, 2 and 4 (6 samples,
`data/DEV_ERASE_01/degron_samples.csv`). **Reserved, unread:** replicates 2 and 3 of the same, for the confirmation; every other sample.

**The instrument's job:** say whether turning maintenance off lowers every region's holding energy by the same amount (R, restoring slowed)
or in proportion (Phi, less of the drive spent holding), as the graded test asked; here maintenance falls over days, so at day 2 most marks
are still held and the counted molecules are not only the survivors of a near-complete loss (the graded test's caveat).
**Synthetic test:** DEV_ERASE_01/graded_synthetic_v2.py, DEV_ERASE_01/graded_03/graded_synthetic_v2.txt
(the same scorer, `graded_score.py`, unchanged; met: a planted shift read R, a planted scaling read Phi).

**Bars (at most three):**
1. The synthetic, met above.
2. Held error at day 2 and at day 4 above day 0, in both lines (the one command's count).
3. Day 2 against day 0 in each line, tiles of 50,000 CpGs, read by `graded_score.py` with the rule stated before: R if the slope's 95 %
   interval holds 1 and the intercept's lies below 0; Phi if the slope's lies below 1 and the intercept's holds 0; otherwise neither.
   Day 4 is read the same way and reported, not scored. Pairs: UHRF1-AID day 2: GSM7516959 against GSM7516962; UHRF1-AID day 4: GSM7516959 against GSM7516965; DNMT1-AID day 2: GSM7516977 against GSM7516980; DNMT1-AID day 4: GSM7516977 against GSM7516983.

**Expected outcomes, stated before:** R in both lines says maintenance sets the restoring rate, the firm rate's missing half; Phi says the
holding fraction; neither, with the regions converging as in the graded test, says the convergence is not only survivor selection. The
confirmation reads replicates 2 and 3 the same way. IAM-A's baseline stays calibrated whatever this reads.

**The degron libraries are PBAT:** the first sample's 20 chunks ran out of memory (15.8 GB and growing swap, machines shut down 25 minutes
in), and one chunk on 2 threads ran past five and a half hours without finishing. GSE236026's own description gives the cause: its libraries
are tPBAT (post-bisulfite adaptor tagging), whose first mate is read from the strand a directional library reads with its second, so the
aligner was fitting every read against the wrong strand. The alignment script now gives the mates in swapped order for a sample listed
with library `pbat` (`degron_samples.csv`); directional libraries, every earlier sample, are unchanged. One chunk is re-sent to confirm.
