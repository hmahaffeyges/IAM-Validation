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
