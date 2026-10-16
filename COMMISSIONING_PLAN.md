# Commissioning plan: the IAM instrument catalogue (written 2026-10-15, overnight)

The author's mandate: commission IAM-A first and foremost, then the C-Score, Met-A on its own loose share, and Group-A; build the A-score
database; derivations only where they improve accuracy now. Every step below is a Master Testing Manifest row; every test runs synthetic
first, writes its rule before its data, and is pushed through the gate. Times are Pacific and are targets: box jobs set the pace.

## To-do now, in order (updated 2026-10-15 evening; each item a manifest row)

Commissioning first; documents while the machines run.

1. **IAM-A gate: second laboratory** (PHY-047): first scored 3 of 5, corrected to NOT MET like for like (DEV-SECONDLAB-01). Next: make the loose share read the same in two laboratories. Meter error measured (PHY-050).
1b. **IAM-A sample-shape check** (COM-IAMA-053): no reading for a sample outside the healthy shape; ENCODE's B cell must be refused. **Instrument agreement check** (OTH-058).
2. **IAM-A gate passed? then** the clean-room rebuild of the 2026-10-15 results (PHY-055), then the grand sealed test (COM-IAMA-052).
3. **Backbone, scale:** the healthy firm-territory range counted on the same counter (PHY-071).
4. **Backbone, field effect:** tumor-adjacent colon against truly healthy colon (PHY-072).
5. **Probe-position check:** own reads counted only at the arrays' probe positions (PHY-070).
6. **C-Score on real tumors** with a sealed bar, which gives its tumor line (OTH-054).
7. **Document audit** of the last week (OTH-055), while machines run.
8. **Operations-manual PDF rebuilt** (OTH-056), while machines run.
9. **Atlas v2.1 lands** about 8 AM Pacific Friday: release checks, sealed bars, the Loyfer and Moss use tests.
10. Stage T: the temperature law, T2 synthetic then the seal (PHY-057).
11. **IAM-A's self-reference in the chain** (OTH-057): Stage Q2 written, synthetic MET, gate rule 23b live. Next: rebuild the line through the chain from Loyfer's 153 samples. Every IAM-A test from now on runs through it.

## Decisions waiting for the author (nothing below is blocked on them)

1. ~~COM-CSCORE-013~~ **Withdrawn 2026-10-15: already decided.** The author adopted both C-Score definitions on 2026-10-09 (one-sided
   band, upper limit 1.152; repeat bar within-person SD of C_rel <= 0.125; CSCORE_COMMISSIONING_PLAN.md). This plan wrongly listed them as
   open. The last check, test (b), is now sealed on new same-person repeats (GSE147740), with a third-laboratory band test on the same arrays.
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
| 0 | ATL-060 | **DONE 2026-10-15:** the Loyfer and Moss use tests sealed before the fit landed (DEV-ATLAS-USE-01; synthetic run 1 NOT MET and fixed, run 2 MET). Real run after the release | laptop |
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

## Stage T: the temperature law, right after the current items (author, 2026-10-15)

The author: the instruments prove the tools detect disease; the temperature test proves the physics is real "in a way that matters to
physicists". Both are needed. Order: the current items finish first (open-sea own reads, tumor pairs scored, C-score test b, the atlas
landing with its sealed bars); then this stage; then logging the wins and documenting the Warburg and tumor lines (PHY-069, from ASCORES/).

| step | row | what | where |
|---|---|---|---|
| T1 | PHY-057 | Data search, metadata only: one species at two body temperatures on single-molecule reads (torpor against active, or two rearing temperatures), at least 20,000 qualifying molecules per temperature, never a fish set already scored | laptop |
| T2 | PHY-057 | Seal the sample list and the reading against the numbers already sealed (log-odds ratio 1.115 for 5 against 37 C; 1.0343 for 18 against 28 C); synthetic already MET (run 2) | laptop |
| T3 | PHY-057 | Read once with the committed counter; report the measured ratio with its error against the sealed number | box or spot |
