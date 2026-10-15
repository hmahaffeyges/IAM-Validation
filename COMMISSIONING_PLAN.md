# Commissioning plan: the IAM instrument catalogue (written 2026-10-15, overnight)

The author's mandate: commission IAM-A first and foremost, then the C-Score, Met-A on its own loose share, and Group-A; build the A-score
database; derivations only where they improve accuracy now. Every step below is a Master Testing Manifest row; every test runs synthetic
first, writes its rule before its data, and is pushed through the gate. Times are Pacific and are targets: box jobs set the pace.

## Decisions waiting for the author (nothing below is blocked on them)

1. **COM-CSCORE-013:** the C-Score bars narrowly unmet. Should the healthy band be read one-sided (only "above the band" counts, as cancer
   moves toward the ceiling), and should the repeatability bar be absolute (0.10) rather than series-tared? Recommendation: one-sided,
   because the gauge flags only rightward moves; absolute repeatability kept, because a tared bar can hide array-to-array noise.
2. **COM-CSCORE-012 against COM-CSCORE-014:** for the C-Score's last check, keep searching for unspent same-person EPIC v1 repeats, or
   commission the 450K band instead (opens the 450K archive). Recommendation: the repeat search first (metadata only); 450K only if it
   comes back empty.
3. **ATL-059 (parked):** sites measured only by arrays, in a later atlas version? Recommendation: measure first; no v2.2 is planned.

## Running now

| Machine | Job | Watcher (every 10 min, watch.json in S3) | Expected |
|---|---|---|---|
| methylphys-cpu-01 (128 cores) | atlas v2.1 fit (ATL-005): 160 of 700 blocks at 7:05 AM, 24 blocks per hour, 0 divergences, worst R-hat 1.008 | STALLED if no block in 100 min or no workers | about 5:30-6:00 AM Oct 16; about 30 h, about 220-230 dollars |
| methylphys-fit-01 (32 cores) | tumor pairs on both sides (PHY-043): 9 of 12 samples | STALLED if no sample in 75 min or no workers | about 8:50 AM Oct 15 |

If a watcher reads STALLED: diagnose; restart the block runner if a block died (finished blocks are skipped); stop the machine if the job
cannot continue, so it does not bill idle.

## Order of work

Correction (author, 2026-10-15): the Loyfer and Moss use tests were first placed after the atlas lands; they must be sealed before any atlas
value is opened, so they are now step 0.

### Stage 1: tonight and this morning (laptop and the second machine)

| # | Row | Step | Needs |
|---|---|---|---|
| 0 | ATL-060 | **First, before the fit lands:** write, synthetic-test and seal the Loyfer and Moss use tests (markers against their published lists; the wide-area mixture with a fair same-ground arm; extras shown). Only the fit's run log (speed, R-hat, divergences) has been read; no atlas value has been opened | laptop |
| 1 | PHY-049 | Open-sea own-read loose share: an hg19 open-sea map (islands, shores and shelves out, UCSC cpgIslandExt), synthetic, then the 153 Loyfer samples streamed from GEO | the second machine |
| 2 | PHY-043 | Score the tumor pairs on both sides by the committed rule (DEV-BACKBONE-01 island arm) | the job's output |
| 3 | COM-METAA-039 | Met-A against its own loose share: synthetic, then healthy neutrophil arrays already spent (development) | the box after the fit, or the second machine |
| 4 | COM-CSCORE-012 | Metadata search for unspent healthy same-person EPIC v1 repeat arrays (sample sheets read directly) | laptop |
| 5 | OTH-051 | The A-score database generator and its gate check | laptop |
| 6 | PHY-050 | Meter correction design: sequencing error measured in ENCODE's sorted immune alignments (development use) | laptop, then the second machine |

### Stage 2: when the atlas lands (about 6 AM Oct 16)

| # | Row | Step |
|---|---|---|
| 7 | ATL-004, ATL-045 | Release checks: every declared output present; distinctness and agreement with v2 on all 700 blocks |
| 8 | ATL-001, ATL-035, ATL-034 | Commissioning bars sealed 2026-10-14 (L15) scored on the held-out sets: U1 profiles on independent labs, U2 composition-blind sites, U3 blood composition |
| 10 | COM-METAA-008, COM-METAA-033 | Met-A second-laboratory neutrophil reference from the atlas (ends the six-array, all-male reference) |

### Stage 3: commissioning IAM-A atlas-free (the main goal)

| # | Row | Step |
|---|---|---|
| 11 | PHY-047 | Second laboratory: ENCODE sorted immune cells (development); does the line hold there? |
| 12 | PHY-050 | Meter correction applied; does the firm rate move toward the writer's measured accuracy? |
| 13 | PHY-055 | Clean-room rebuild of the 2026-10-15 results from public data only |
| 14 | COM-IAMA-052 | **The grand sealed test** on fresh data (candidates GSE144691 and GSE98837: healthy sorted CD8 T-cell subsets, unopened): healthy samples on their own line within a stated band; memory subsets above naive by their loose share; tumors off the line toward the ceiling (tumor arm from unopened tumor/normal pairs, named before the seal) |
| 15 | COM-IAMA-011 | Commissioning note written from the grand test, scope stated exactly |

### Stage 4: the rest of the catalogue

| # | Row | Step |
|---|---|---|
| 16 | COM-CSCORE-005, COM-CSCORE-003 | C-Score last check, then its commissioning note |
| 17 | COM-METAA-013 | Met-A cell by cell (blood cells, then breast, colon, prostate, lung), easier if COM-METAA-039 works |
| 18 | COM-METAA-032 | Whole blood back into Met-A scope with a counted-cell truth set |
| 19 | GRP-004, GRP-011 | Group-A, one cancer at a time, patients and controls in the same series |
| 20 | PHY-064, PHY-065 | Whole blood and cfDNA on IAM-A (the atlas assigns molecules, the line reads them) |
| 21 | PHY-069 | The concern line (where a reading first leaves the healthy line) from the A-score database |

## Derivations

Only three improve accuracy now, and they are in the plan as measurement steps: the meter correction (PHY-050), the open-sea count
(PHY-049) and the bend of the line at very loose cells (PHY-049). Everything else (the shield height, the temperature law's test,
cosmology, the papers) waits until the catalogue is commissioned, unless the author calls it.
