# DEV-BASE-CHAIN-01 — base chain v3 on every EPIC array in the bucket (development, checks written 2026-10-03 before any array was read)

**Why.** Commissioning order, base chain first (SOP v3 section 2b): stages 0, 1, 2, 5, 6, 8, 9, 13 on EPIC arrays, and stage 7 on the
bundled single-molecule test data. This note states the checks before the run. The outcome is added below the line after the run;
nothing above the line is changed after reading.

**Data.** `downloads/G_chain_tests/` in the project bucket: 56 GEO series, 5,149 arrays (EPIC v1 4,801; 450K 276; EPIC v2 72), listed one
row per array in `doors/data/DEV_BASE_CHAIN_01/manifest.csv` (series, GSM, IDAT name, slide, platform from the GEO platform id,
specimen, declared sex and age, and whether the series record labels the array a healthy reference). The manifest was built from the
series matrices before any array was read. Specimen is taken from the series record as written (whole blood, isolated neutrophils,
sorted cells, PBMC, bone marrow, cell line, placenta, constructed DNA mixture, unspecified); it is passed to `--specimen` unchanged.
No disease label enters a run; the only label used is "healthy reference", and only to choose same-run tare references (SOP rule 2).

**Run.** Chain v3 at the commit recorded in the outcome (repository main 3e03b37 plus the report sections and the `--save-betas`
option added on 2026-10-03, below), on ssh:methylphys-cpu-01. Every array goes through `run_sample.py --engine v3` with
`--array-type` from the GEO platform, `--sex`/`--age` from the series record. Constructed DNA mixtures (72; no donor) run with
`--no-intake`, as `chain_tests/run_chain_acceptance.py` did. Pass 2 (Stage 8 tare): every array read in pass 1 is re-run with
`--slide-ref-table` when >= 3 other arrays of the same series and specimen, labelled healthy reference and read in pass 1, are on its
slide (else in its series = batch); the array itself is never its own reference. Nothing is fitted; no floor, bar or line is changed.

**Code added before the run (no physics).** `report_v3.py`: the sections SOP 2b lists as required for the v3 report (red flags
STOP/WITHHELD/CAUTION/NOTE, also written to the bundle; safeguards: rendered-claim scan, formula self-test, anchors, deconvolver
conformance, atlas separability; troubleshooting; integrity; file inventory; run it yourself; stage table; toolkit table with
PASS/FAIL/NOT_RUN/NOT_BUILT), each with an HTML id. `run_sample.py`: `--save-betas` (writes the Stage 1 vector; used by the toolkit
checks that follow) and the command line in the bundle.

## Checks (set now)

**(a) No crash.** Every array of every series either runs end to end (exit 0, HTML and bundle written) or stops at Stage 0 with a
named reason (exit 2 and a `QUARANTINE` line naming the hard failure or the refusal). Anything else is a crash: another exit code, a
Python traceback, exit 0 without report and bundle, exit 2 without a named Stage 0 reason. **Bar: 0 crashes of 5,149.**
A crash is a code bug; it is fixed (crash, path, platform handling only) and the affected arrays re-run; both runs are recorded.

**(b) Purified healthy neutrophils read Normal.** Set: isolated neutrophils labelled healthy in the record — GSE110554 (6, the
floor's own arrays), GSE167998 (6) and GSE181034 (6) (re-deposits of the floor arrays), GSE118144 controls (13), GSE122244 healthy
controls (5), GSE247193 (24), GSE247195 (24). Reading: tared A_rel where pass 2 ran, and the untared own-floor state. **Bar: every
array of the set that reaches Stage 5 reads Normal (0.95-1.05) on its tared A_rel.** In-floor arrays and other laboratories are
reported separately. Arrays the set loses at Stage 0 are counted with their reason, not as Normal or not Normal.

**(c) Technical replicates, GSE250556 (64 arrays, 4 people).** Within-person SD of tared A_rel after the median tare (pooled:
sqrt(sum of squared deviations from each person's mean / sum of (n-1)), the DEV-REPL-V3-01 rule); the noise gate's withhold rate
(arrays whose printed state is withheld by the gate, of those read) and the count with N > N_max that would be withheld untared.
Targets carried unchanged from `DEV_REPL_V3_01_PLAN.md`: within-person SD <= 0.020; >= 95 % of tared readings in Normal.

**(d) Report on every array that passes Stage 0.** For every array with Stage 0 PROCEED or PROCEED_WITH_PENALTY (and every
constructed mixture), the HTML page and the JSON bundle exist, the bundle carries `red_flags`, and the page carries every section
SOP 2b requires: Stage 0 intake (with the Stage 1 line), composition, Met-A, tare, noise gate, C-score, red flags, the five named
safeguards, troubleshooting, integrity, file inventory, run it yourself, stage table, toolkit table showing NOT_BUILT and
NOT_RUN/PASS rows. **Bar: 100 % of those arrays.** Safeguard results (e.g. claim-scan hits) are recorded as found, not as part of the bar.

**(e) Stage 7 IAM-A on the bundled single-molecule test data.** The repository bundles no real single-molecule file; the bundled
test data are the constructed site table and `.pat` of `release_check_v3.py` E4. Through `run_sample.py`: a site table whose copy
error equals the healthy position reads IAM-A 1 +/- 0.005, Normal; the same table under another pipeline is refused; the `.pat`
extractor returns the 30 constructed isolated errors. **Bar: all three.**

**Diagnostic arm (not part of any bar).** Arrays that stop at Stage 0 only because the series record publishes no age or no sex
(Stage 0.2 `MANIFEST_INVALID`) are also run with `--no-intake`, so stages 1-13 are exercised on them and the toolkit checks have
their beta vectors. Their readings are reported separately and labelled intake not run.

---
## Outcome (recorded 2026-10-03 after the run; nothing above the line was changed)

**Runs.** Box ssh:methylphys-cpu-01 (128 cores, 96 workers). Run 1: job 5182527a, chain commit 63d55fa on the box = main 058b646 + the
pre-run code above + the Stage 0 environment status (lead's fix 3), 14:58-16:37 UTC. Run 2 (after the crash fixes below): job fddbcd69, commit
ffd401d, the 72 EPIC v2 arrays of GSE286313 and the one unreadable IDAT of GSE255057 re-run; the 72 refusal reports re-rendered at 9593d66 after the
Stage 1 line fix (jobs 385b3ede, 3c27bc06). Per-array records: `data/DEV_BASE_CHAIN_01/readings_all.csv`
(one row per array and arm), check summary `data/DEV_BASE_CHAIN_01/outcome.json`; every report, bundle and log copied to S3
`results/DEV_BASE_CHAIN_01/chunks/`, Stage 1 vectors to `results/DEV_BASE_CHAIN_01/betas/`.
The manifest's 5,149 rows are **5,069 physical arrays**: GSE181034 is a SuperSeries that re-lists the GSMs of GSE167998 and GSE182379 (80).

| check | bar | result |
|---|---|---|
| (a) no crash | 0 of every array | **Run 1: FAIL** - 26 crashes (25 EPIC v2 arrays crashed in Stage 1, methylprep "Unknown array type (1,105,209 probes)"; 1 truncated IDAT crashed Stage 1 in the --no-intake arm, EOFError) and 38 EPIC v2 arrays wrongly quarantined for "sex" (read by EPIC v1 decode rules). 7 EPIC v2 arrays stopped as ENVIRONMENT_MISSING_MANIFEST (the methylprep EPIC v2 manifest failed to load on the first concurrent attempts): the new status did its job. **Run 2 after the fixes: PASS** - 5,069 of 5,069 arrays end to end or stopped with a named reason; 0 crashes in any arm |
| (b) purified healthy neutrophils Normal | every array reaching Stage 5 Normal on tared A_rel | **FAIL** - floor's own arrays (GSE110554) 6/6 Normal (A_rel 0.991-1.008; re-deposit GSE167998, diagnostic arm, 6/6). Other laboratories 42/49 Normal (A_rel 0.810-1.068): GSE247193 17/21, GSE247195 22/24, GSE122244 3/4. Untared, 45 of those 49 read above Normal against the own floor (A 0.862-1.286): the other-laboratory offset the tare is there to remove. Not read: 3 GSE247193 (identity-site coverage < 90 % on high-noise arrays), 1 GSE122244 (4,905 of 6,000 sites). Lost at Stage 0: 19 (13 GSE118144 controls, no age in the record; 6 GSE167998, no sex). Diagnostic arm, GSE118144 controls: 8/13 Normal (0.913-1.065) |
| (c) GSE250556 replicates | targets: within-person SD <= 0.020; >= 95 % Normal | read 63/64 (GSM7981500: Stage 0 `ctrl_qc`), tared 63/63; **within-person SD 0.0369** (A 0.030, B 0.035, C 0.040, D 0.041), SD over all 0.040; **48/63 Normal** (8 below, 7 above), A_rel 0.920-1.077. Noise gate: withheld 0 of 63 on the tared readings; N > 0.149 on 59 of 64 (N 0.137-0.195), i.e. 59 would be withheld untared. Both targets **not met**; numbers reproduce DEV-REPL-V3-01 |
| (d) report on every array past Stage 0 | 100 % | **Run 1: 5,481 of 5,553 reports complete; FAIL** - the 72 EPIC v2 refusal reports (run 2) had no Stage 1 line because Stage 1 never ran; the report now prints "Stage 1: not run - <reason>". **After that fix (re-render, commit 9593d66, jobs 385b3ede/3c27bc06): 5,553 of 5,553 PASS.** Every report: claim scan PASS (5,553/5,553), formula self-test PASS, anchors PASS, deconvolver conformance PASS 4,978 / NOT_RUN 575 (no composition: isolated neutrophils, refusals), atlas separability NOT_RUN (stage 3 not wired) |
| (e) Stage 7 IAM-A, constructed single-molecule data | all three | **PASS** - site table at the healthy position IAM-A 1.0 (Normal, eps 0.03619); other pipeline refused; .pat extractor 30 of 30 isolated errors; the report carries the IAM-A section. No real single-molecule file is bundled in the repository |

**Stage 0 stops (5,069 arrays, run 2).** 1,569 incomplete manifest (the series record gives no age and/or no sex; Stage 0.1 requires both),
276 450K arrays at 0.7b (`hm450_coverage`), 52 sex mismatch, 37 `ctrl_qc`, 2 multiple. Platform refusals with a report: 72 EPIC v2, 15 EPIC v1
vectors under 700,000 probes. Diagnostic arm (--no-intake on the 1,569): 1,568 read end to end, 1 unreadable IDAT (named stop).

**Fixed (code, no physics; run 2 and later).** `run_sample.py`: an EPIC v2 array is refused before the QC decode with a report (it had crashed in
Stage 1 or been quarantined for "sex"); an IDAT that Stage 1 cannot read is a named stop; a methylprep manifest that cannot load is
ENVIRONMENT_MISSING_MANIFEST (exit 3), never QUARANTINE_CORRUPT_IDAT. `report_v3.py`: "Stage 1: not run" line. `conductor_v3.py`: the noise-gate
state is applied only when A exists (3 GSE247193 arrays printed "withheld ... A printed as a number" with no A).

**Development findings.** The median tare does not bring other-laboratory purified neutrophils into Normal on every array (42/49), and does not
bring replicate precision to 0.020 (0.037). Age is a required Stage 0 field although no v3 stage reads it: it stopped 1,569 arrays (31 %),
including 13 healthy purified neutrophils (author decision). No threshold, floor or bar was changed.
