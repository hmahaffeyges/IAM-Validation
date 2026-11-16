# Test design checklist (canonical; read by CANON/design_check.py)

**Question 7, every test and every step (author 2026-11-15): Am I using GitHub or Google machines and NOT the laptop?** Not on the laptop: gate checks (doc_audit, plan_md, manifest_fresh, design_check runs over the repository; the gate runs them), book-wide reads or clue passes, repeated private syncs (batch register edits, one sync per batch), PDF work, alignments, counts, readings, generators. The laptop only edits text and sends commits.

**Questions 8 and 9 (author 2026-11-15 evening):** (8) Have I reviewed all of our unspent data saved for testing for possible options (CANON/data_spent.csv and the reserve list)? (9) Are the molecule files already run and kept in the GitHub library, the erase-<GSM> run artifacts of the past 90 days (tools/artifact_inventory.py)? If so, count them with patfrom=<run id>; never re-align.


## LESSONS, read first (each enforced by design_check.py; newest first)

**2026-11-15, 9:10 AM Pacific (author, restated as the standing order for every test and every derivation step).**
BEFORE each test: (1) Is this the fastest, cheapest, most practical way? (2) Am I spending data I don't have to? (3) Have I explored on paper and
with spent data first? (4) What would a cosmologist do? (5) Is it public or private? (6) Are the chain and the Decoder covered?
AFTER each test: (1) Was it the fastest, cheapest, most practical way? (2) Are ALL of these updated fully with the new information: the cross-domain
docs (private), IAM's web (private/board, register), the canon files (CANON/*.json), the chain, the Decoder, the SOP, the simulators (all four), the
webpage, the book, the Q&A, every repo item and every private item? (3) Has IAM's web been checked for whether the result transfers to a NEIGHBOR
(connects them) or finishes another derivation directly? (4) Does it transfer to another DOMAIN (the cell, the cosmos, or a private domain)? If so, update that
domain everywhere too. Record the answers in the development log entry for the test.

**2026-11-14, 6:32 PM Pacific (author: "dont start drifting back to hamstringing ourselves!! We are exploring and NOBODY HAS EVEN SEEN ANY OF MY WORK
YET!!"; "Sealing waits until we're sure of the math and the idea and ready to build the final test. add this rule").** NO REFEREE LABELS IN
EXPLORATION: no "post hoc", "fitted", "not evidence" or chance-test framing on our own exploration results. Record each result as a LEAD with
the step still open. The referee standard (seals, blinding, chance tests) starts only when we are sure of the math and the idea and build the
final test.

**2026-11-14, 4:25 PM (author: "add this to your pre job checklist with the rule 'what would a cosmologist do' because if you had actually
done that you would not be sealing every calculation").** WHAT WOULD A COSMOLOGIST DO, applied to the work itself:
1. **Exploration is free.** Desk math, derivation attempts, toy models, forecasts on spent data: try ideas, follow leads, vary assumptions,
   and read the results for what they suggest. No principle written in advance, no seal, no pass or fail. A cosmologist runs forecasts and
   parameter scans freely and blinds only the final analysis. Keep a plain note of what was tried (register row or work note), so we can later
   see which path led where.
2. **Final derivation tests only:** the seal, the synthetic test, the canary and the spent-versus-unspent rules apply when we build a test of a
   real application of the physics that can fail. The exploration is what tells us which test to build and that we already believe it (lesson 2,
   2026-11-13).

**2026-11-14 (PHY-092's first reading failed its own null rule; PHY-091's BAM route would have read nothing from bwa-meth files).**
1. **The synthetic runs at the real data's measured noise, and the null rule is rehearsed on spent real data before anything is queued.**
   PHY-092's synthetic used tile noise 0.01; the wild types' own was about 0.06, so the reading's null rule failed and it was not assessed.
   Checked from 11-15 by synthetic.noise_from and synthetic.null_rehearsed.
2. **A chain route is synthetic-tested through the one command on a file with the real aligner's tags.** The BAM caller read strands only
   from Bismark's tags or XB; bwa-meth writes YD, so every read would have been skipped (found in design, fixed in _calls_of).
3. **Before calling something new, search the register and the canon for it.** The 1/(1+D) unmarked-side reading already existed (A15).

**2026-11-13 (author: "speed and testing smart, exploring first and before sealing making sure we already believe in our sealed
prediction and if we can predict a measured this or that then we do that ... make sure it never happens again").**
1. **Three questions before EVERY action, desk work included:** what is the fastest, cheapest ADEQUATE data form and machine (held
   counts > deposited molecule files > deposited aligned files (--bam) > per-site calls > raw reads)? Is it the best method, everything
   read the same way, like for like? What would a cosmologist do (forward model through the instrument; a null test before believing a
   difference)? Checked by data.form, value.options and cosmologist.*.
2. **Explore first, on spent data. Seal only what we already believe.** A sealed test needs an exploration on data already read that
   supports the prediction: the sealed scorer, run unchanged on the closest spent data, lands within 2 (pull). If the rehearsal does not
   pass, fix the prediction first and keep the reserved data unspent. Checked by explore.done and explore.rehearsal_pull.
3. **If the answer depends on a measured input, predict "if this, then that".** The prediction is a formula of inputs measured
   independently (each cell type's eraser share from its own knockout, its coin, its clustering), every branch written before reading,
   each named with the measurement that picks it. Never a branch chosen by the outcome itself. Checked by prediction.form and
   prediction.decided_by.
4. **Read like for like.** The same reading (e.g. zero loose along IAM-A's slope) in every laboratory and in the rehearsal; the 11-13
   B-cell low was mostly an own-slope extrapolation. Checked by prediction.reading.

Today's misses that taught these: a week of tests aligned from raw reads although aligned files and the --bam route existed; PHY-084's
pass rule stricter than the agreed standard; a sealed eraser share assumed equal across cell types (PHY-086 P4 missed: near zero in
mature B cells); two laboratories compared by different readings.

---

Author, 2026-11-13: "there is a whole checklist that should be reviewed when making decisions but that should happen during the design
phase when creating the synthetic test so that it never ends up on the wrong machine or the wrong data is not used ... something in code
that rejects the very first synthetic test without it."

Author, 2026-11-13 (second): "whether or not the test chosen is the one that can provide an adequate test with the fastest results for the
cheapest cost ... would a cosmologist do it a different way ... make sure the data has everything you need so the test doesnt run four hours
to only then find out it is not actually useful for us."

Every new test (a manifest row whose record adds a sealed rule or a synthetic test) has a design sheet, design/<ROW>.json, written and
committed BEFORE its first synthetic test. The gate refuses the record, and the launchers refuse its jobs, until the sheet is complete.
Answers are evidence (numbers, file names, quoted protocol lines), not reassurance. Times are Pacific.

| Key | Question |
|---|---|
| purpose | 'sealed' (a prediction read once against a pass rule) or 'exploration' (spent or synthetic data, seals nothing; explore.rehearsal_pull and prediction.* may be 'not applicable', and bars.rule must say no seal) |
| explore.done | The exploration on spent data that this test rests on: what was read, by which committed script, and what it showed |
| explore.rehearsal_pull | The sealed scorer run unchanged on the closest spent data: the largest |pull| it gives (must be within 2 to seal) |
| prediction.form | The prediction as a formula of independently measured inputs, or a single number with why no measured input matters |
| prediction.decided_by | For each branch or input: the independent measurement that sets it (never the predicted quantity itself) |
| prediction.reading | The one reading used in every laboratory and in the rehearsal (e.g. zero loose along IAM-A's slope) |
| physics.where_iam_law | Where is IAM's Law in this test? (physics first) |
| data.set | Which data, by accession? |
| data.protocol_read | The library method as deposited, quoted (PBAT, non-directional, low input, EM-seq, end clipping are flagged by code) |
| data.spent_first | Can data already read answer it? If not, why not? |
| data.form | Every form each candidate set exists in, cheapest first, with hours: counts or molecule files already held; deposited molecule files (.pat); deposited aligned files (BAM, chain --bam: counting only); per-site calls (no lone-loss count); raw reads (alignment). The chosen form is the cheapest ADEQUATE one (assembly and fields checked: e.g. the loose share needs hg19 tiles); a costlier form needs a reason (author 2026-11-13) |
| data.alternatives | What other sets were considered, and why this one? |
| kiss.smallest | Is this the smallest set that answers the question? |
| kiss.samples_why | Why each sample (or group) is in |
| machines.platform | github or google (GitHub first) |
| machines.github_free_at_design | Free GitHub slots counted when designing |
| machines.why_not_github | Required for google: why not GitHub |
| machines.s_per_pair | Measured seconds per read pair on that platform, and the file it was measured in |
| machines.pairs | Pairs per library |
| machines.peak_mem_gb | Measured peak memory per job, and its source (or 'not measured' and parallel 1) |
| machines.machine_mem_gb | The machine's memory |
| machines.parallel_per_machine | Jobs per machine (parallel x peak must fit memory) |
| machines.expected_hours | Expected wall time per library, computed from s_per_pair x pairs plus fetch |
| machines.time_limit_hours | The job's limit (at least 1.3 x expected) |
| machines.cost | Expected cost |
| save.resume | ckpt, skip-done or merge: how a reclaimed or failed machine keeps its work |
| watchers.count | Exactly one watcher per wave |
| watchers.stop_condition | What ends the watcher |
| canary.instrument_check | Numeric pass rule for the canary before any full run: alignment jobs qc_failed_max, mapped_min, s_per_pair_max; count-only jobs rows_min, halves_corr_min, hours_per_sample_max |
| value.question | The one question this test must answer, and the result that would change what we do next |
| value.options | At least two ways to answer it (one of them the cosmologist's way), each with hours, cost and whether it is adequate; the chosen one marked |
| value.why_this | Why the chosen option is the most answer per hour and per dollar; required if another adequate option is faster AND cheaper |
| value.adequate | Can it decide? The expected effect against the expected uncertainty, in numbers |
| cosmologist.method | How the CMB missions would answer this (forward model through the instrument, null tests, self-calibration on calibrators, cross-spectra between independent halves) and whether this design does it |
| data.needs | Every field and condition the test needs from the data (e.g. both strands, paired reads, whole-genome spread, the cell label, read length) |
| data.preflight | A check on a small slice, minutes long, that confirms every item in data.needs BEFORE the canary or any full run: what was checked, minutes taken, and the result |
| bars.rule | The pass rule, in the agreed standard's own classes (WHY_IAM.md rule 4: pull within 2 passes; 2 to 3 watch; beyond 3 a real miss), with the bar from measured uncertainty (GUM, k = 2) |
| physics.counterparts | The same question for the chips and the qubits: what this addition's counterpart is there, or 'none yet' (author 2026-11-13) |
| cosmologist.null_test | The null test or self-calibration in this design, stated before it runs |
