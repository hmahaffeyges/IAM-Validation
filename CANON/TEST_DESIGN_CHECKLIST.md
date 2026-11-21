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
| bars.rule | The pass rule, in the agreed standard's own classes (TEST_DESIGN_CHECKLIST.md, setting test bars, rule 4: pull within 2 passes; 2 to 3 watch; beyond 3 a real miss), with the bar from measured uncertainty (GUM, k = 2) |
| physics.counterparts | The same question for the chips and the qubits: what this addition's counterpart is there, or 'none yet' (author 2026-11-13) |
| cosmologist.null_test | The null test or self-calibration in this design, stated before it runs |

## The testing rules (moved verbatim from WHY_IAM.md, retired 2026-11-20)

**Before any test or run, the four questions, answered in writing.**
1. Is this the best data for it, chosen from everything already in hand (kept values, bundles, count files, molecule files) before anything new?
2. Is this the most efficient and expedient route: the cheapest data, free machines first, the smallest set that answers?
3. Is this the best test, the one a physicist and a geneticist cannot tear apart?
4. What would a cosmologist do? (Calibrate each detector on its own data; forward-model the instrument onto the expectation; split null tests; noise maps.)

**Data.** Spent data first. Fresh data only when necessary, and spent once, on a sealed test whose form exploration has already settled.

**Judging results.** Exploration is judged by what the data shows; strict sealed bars are for commissioning and referees. Act on named risks before reading, not only note them.

**Setting test bars (agreed 2026-11-12; never set a bar by choice).** A bar comes from measured uncertainty, following metrology's guide (GUM):
1. Combined uncertainty: sigma_total^2 = (the prediction's own uncertainty)^2 + (the meter)^2 + (the natural spread)^2. When the test is on a mean or median of n samples, the natural spread enters divided by n. Coverage k = 2 (about 95 %).
2. Say which question is tested. The baseline (is the healthy level where the law predicts?) is tested on the median, with the median's uncertainty. Each sample (is every sample within X?) uses X = the natural spread itself, and about 1 healthy sample in 20 is expected outside by chance.
3. Write the formula and the bar before any data are read (sealed), so a referee can recompute the bar.
4. Report the pull, (observed - predicted) / sigma_total: within 2 passes; 2 to 3, watch; beyond 3, a real miss. One result just past 2 is not a refutation; a miss counts when it is large or repeats.
5. A claim of new physics needs about 5 sigma; agreement tests use k = 2.
6. Measured components so far (healthy lone-loss rate, 150 Loyfer non-neuron samples, PHY-079): meter under 1 %; same cell type 2.9 %; between cell types 6.7 %; total 7.5 %. Per-cell-type bars become possible as the cell-type part is explained.
7. The lesson behind it: PHY-050's third test held each sample to +/-10 % while the body's own spread is 7.5 % (about 82 % of healthy samples fall inside by nature; the bar asked 83 %; it got 81.5 %). On the baseline question it passes at about 1 sigma (median 0.987 of the prediction, uncertainty about 1.3 %). The original sealed verdicts stay in the record beside the rescored ones.

**The chain.** If the chain didn't produce it, it didn't happen; every chain change goes into the Decoder in the same commit; synthetic first; nothing fitted; healthy always from the atlases; the Deconvolver decides what is in a sample.

## The author's testing checklist, testing plan, methods and machine rules (2026-11-20)

### Testing Plan

Author's request (2026-11-16 ~1:30 AM): 'go through and run through the checklist on each one and fill out a testing plan for each one of them and notate everything we checked and what is needed to test it ... Each one a step by step plan with all things to check including the simulators and web and decoder and searching the book and writing the synthetic test and so forth. That way we have a stack of derivation items ready to actually go.' Rules: the one spend of unseen cell data is the sealed COMMISSIONING test; each derivation enters it as an if-then line (method 5). Derivation work uses methods 1-3 only (math in code; recover a known result; spent or published data, labelled).
0. What it is and where it stands
* Register row(s) and current status (quoted from the row's FULL text, its notes not only its status column):
* Instruments that read this quantity (IAM-A, Met-A sequencing, Met-A arrays, coupling, coupling-A, IAM-A C-score, territories, Group-A, the Deconvolver, IAM-A globe): one line each: how it enters, or why it does not.
* What is DERIVED / MEASURED and stays measured (true input, why) / MEASURED NOW but derivable (open target) / FITTED / OPEN:
1. Checklist (all 40 questions, answered for THIS item; label first, then a real reason)
2. Checked so far (each with what was found and where)
* IAM's web: reviewed links touching the row, two steps out; the physics they carry; gaps
* The book: every passage naming it, read in full; its tag
* The simulators: what physics each holds for it (cell, cosmos); does it compute correctly
* The chain and the Decoder: code that reads it; Decoder entry
* Docs and earlier records (doors/, day logs, development log): already done or redundant?
* Established neighbours (E rows) and the first-principles path
3. Step-by-step plan
1. Paper derivation (equations, checked in code on GitHub, never the laptop)
2. Simulator run that answers it first (which model, which settings, expected output)
3. Spent data that settles or recovers a known result (exact files; kept molecule files or counts; nothing unseen)
4. Synthetic test at the real data's measured noise through the one command; the null rule rehearsed on spent data
5. Canary (smallest set) before any full run
6. Chain change and its Decoder entry (if any)
7. What each step's result would mean (if X then ..., if Y then ...)
4. Commissioning line (the if-then this item contributes to the ONE sealed test)
* Prediction, inputs, bars (GUM: measured uncertainty, never a chosen number), which instrument reads it, which data in the sealed set
5. Finish list (findings ledger columns): register, development log, day summary, book, canon, chain, Decoder, simulators, IAM's web, memory
6. Cost and order
* Machines (GitHub/Google), time, data spent (none before commissioning), what it unblocks next

### Checklist

**for AI before each test:**

CHECKLIST, printed from private/checklist.py (2026-11-16). Every answer starts with its label and gives a reason for THIS item;
bare yes/n/a, a copied answer under more than two questions, or a plan answer without an existing plan file is refused.

**BEFORE an item starts:**

1. [fastest] Is this the fastest possible way to perform this test?
2. [cheapest] Is this the cheapest way to perform this test?
3. [practical] Is this the most practical way?
4. [pen and paper] Have you done the pen and paper calculations first?
5. [strict] Are you making the tests too strict again and hamstringing ourselves? (exploration needs no sealed prediction)
6. [spent data] Have you exhausted all options, including spent data, first?
7. [IAM's web] Have you checked IAM's web for connections to neighbors, established physics or other domains? (evidence below)
8. [machines] Am I using GitHub or Google machines and NOT the laptop?
9. [simulators] Have you checked and/or tested the simulators for clues or answers? (evidence below)
10. [docs] Have you searched the book, the repository's docs AND the gate's own rules (CANON/repro_check.py, CANON/chain_gate.py, CANON/kiss_check.py) to see whether the answer already exists or is already ruled out? (evidence below)
11. [spending] Am I spending data I don't have to?
12. [explored] Have I explored on paper and with spent data first?
13. [cosmologist] What would a cosmologist do?
14. [public] Is it public or private?
15. [chain] Is the chain and the Decoder covered?
16. [unspent] Have you reviewed all of our unspent data saved for testing for possible options? (CANON/data_spent.csv)
17. [molecule files] Are the molecule files already run and kept in the GitHub library (past 90 days)? (tools/artifact_inventory.py)
18. [bars] Are the bars from the agreed scientific standard (GUM, measured uncertainty), not too high or too narrow?
19. [brief] Have you read the item's brief and every book section it names IN FULL (no truncated previews)?
20. [IAM's Law] Where is IAM's Law in this item? (physics first)
21. [like for like] Is every comparison like for like (the same reading in every laboratory and in the rehearsal)?
22. [sealed] If anything is sealed: is the prediction 'if this, then that', each branch set by an independent measurement, with the rehearsal pull within 2?
23. [library] Is the data's library method read from the deposit (PBAT, low input, EM-seq, end clipping) and does it fit the reading?
24. [synthetic] Does the synthetic run at the real data's measured noise, through the one command, with the null rule rehearsed on spent data?
25. [memory] Is anything taken from memory? Every number, citation and accession checked against its source or file?
26. [IAM's terms] Am I answering in IAM's terms first, giving the standard view only as the older lens?
27. [gated] Is any data behind an application wall (EGA, dbGaP, registries)? It must not be.
28. [hotspot] Will anything larger than a few KB come over the owner's hotspot? (results are read on GitHub; only summaries come back)
29. [design sheet] For a new or sealed test: is the design sheet complete before the first synthetic (design_check)?
30. [reserve] Is the reserve ring-fenced (no unspent data used before commissioning, except as agreed)?
31. [fitted] Does any value used come from a fit or a substitution? Is it inside the valid range it was set in (lever rule)?
32. [names] Am I using the chain's and the Decoder's own names (no renames without the author's words)?
33. [next machine] Is the next machine item prepared, so the machines are not left idle?
34. [time] Are the time labels from the clock, in Pacific?
35. [method] Which accepted method settles this row: 1 derivation from accepted premises, 2 recovering a known result, 3 published or spent data read after the fact, 4 blind test, 5 joint test? Is there a cheaper accepted one?
36. [one spend] Does this row need unseen data? For cell rows: can its prediction ride on the one commissioning test instead of its own spend?
37. [established] Which established results (E-rows: Landauer, Szilard, Bennett, Berut, Bekenstein-Hawking, Jacobson/Cai-Kim, Zurek, Sartori-Pigolotti, Hopfield...) does this row stand on, and does a derivation from them settle it without a test? Where is the one identification that still needs a downstream check?
38. [path] Trace the path: which first principle (Boltzmann, detailed balance, Landauer, ...) and which step of IAM's Law does this rest on, through which established rows? Is that path drawn on IAM's web? (name the rows)
39. [two steps] Two steps out on IAM's web: which neighbours, and neighbours of neighbours, does this touch, and does the new result fill, change or connect any of them? (name them)
40. [simulators] Does the result have a place in each simulator (cell, cosmos), and in the chain and Decoder if it is a cell result? (one line each)
41. [chain] GATE: Does every reading in this item come from the chain's one command (run_sample.py) and its stage functions, with NOTHING computed in a cell or script from chain counts? (name the command and stage; author 2026-11-16: ALWAYS USE THE CHAIN)
42. [laptop text] GATE: Is every command this item runs on the laptop text-only and kilobyte-sized (no sub-agents, no repo pulls, no parsing large files, no runs)? (list each laptop command; everything else goes to GitHub or Google)
43. [sources read] GATE: For every term, number and status this item uses, which book passage and register row did you READ today (file:line), not recall? (author 2026-11-16: go read the damn book)
44. [google price] GOOGLE (if used): today's price check run (gcp_prices.py, prices file dated today) and the cheapest VERIFIED zone chosen, spot only? (name zone, machine, price; or 'not used: <why>')
45. [google safety] GOOGLE (if used): does every machine self-delete at job end and at the time limit, and save results to the bucket in pieces as it runs, saying how it resumes? (name the job's resume rule)
46. [google watch] GOOGLE (if used): is relaunch_watch.py started with the job, the machines checked every 30 minutes (logs in UTC), and every 'GAVE UP <job> rc <code>' logged so a failure never looks like a finish?
47. [google spread] GOOGLE (if used): is the work spread across machines within the 32-CPU quota (e.g. 8 x 4 CPUs), and the disk sized to the job (a big alignment chunk fills 60 GB)? (name the split and disk size)
48. [physics] CHECK THE PHYSICS: on IAM's web and in each simulator, is the physics this item touches actually there, does it compute correctly, and does the new result agree or contradict it? (not counts or names: say what physics you found and whether it works)
49. [instruments] Which instruments can use this derivation (IAM-A, Met-A sequencing, Met-A arrays, coupling, coupling-A, C-score, territories, Group-A, Deconvolver, globe, tumor gauge, library floor)? Every one that can use it must be included: name each and how, or why not
50. [commissioning] Commissioning: is this a type test (once, in the one sealed run) or an installation check (each sample's own intake and meter)? Is the use outside the commissioned range (e.g. neurons, tumor tissue), needing a verification point?
51. [plan] Testing plan: name the plan file private/plans/<ITEM>_PLAN.md with all sections (0 status from the FULL row; 1 checklist; 2 checked; 3 steps; 4 commissioning line; 5 finish list; 6 cost), or 'not a test: <why>'
52. [chaining] Chaining: does this row follow from a DERIVED row by accepted physics only (it inherits that standing, no separate test), or does it make new contact with data (the one place a test is spent)?
53. (scoring) Are you using the full chain with one button and letting the chain decide???

**AFTER the result is in:**

1. [practical] Was this the fastest, cheapest, most practical way?
2. [updated] Updated: IAM's web, canon files, chain, Decoder, SOP, simulators, webpage, book, Q&A, repo items, private items? (name each)
3. [web result] Checked IAM's web: does the result connect a neighbor or finish another derivation? (name the rows)
4. [other domains] Does it transfer directly to another domain (cosmos)? If so, updated everywhere for that domain? (name them)
5. [recorded] Findings recorded the same day: development log, day summary, register, memory, Universe document, work list? (name each)
6. [next queued] Was the next machine item queued when this one finished?
7. [ledger] Findings ledger: is there a row for this result with every column (register, devlog, daysummary, book, canon, chain, decoder, simulators, iams_web, memory) set to done/PENDING/n/a with a reason?
8. [expectation] Reported to the author against expectation: positive or negative, against which predicted number, and was every reading the chain's own (not computed outside it)?
9. Please confirm these are all updated in detail elaborately and the math was tested especially in the book but all the math via code?
10. The private plan-document merge, line by line. The checklist gate gets changed so it accepts a plan section instead of a plan file.
11. The Decoder, the book's IAM-A chapter and the webpage, one GitHub edit each.
12. The Q&A, the M-number paper and the rolling log.
13. The canon entries, glossary check and generators.
14. The private web and the simulators.
15. The daily report at midnight.
16. CHECK THE IAM-WEB AND SIMULATORS MEANS CHECK THAT THE  PHYSICS IS THERE AND WORKS AND LOOK FOR CONNECTIONS!!!

### Examples And Methods For Testing

There are several accepted ways to establish a result, and only some of them need a test run on new data. Roughly from cheapest to strongest:
| Method | What it is | Needs a prediction test? | Our examples |
|---|---|---|---|
| 1. Derivation from accepted premises | A mathematical proof that starts only from established physics. The referee checks the premises and the math. | No. We check the math in code (symbolic algebra, high precision); no data needed. | Every settled surface holds ½ of what it paid (from Bekenstein–Hawking plus the first law); a paid pump keeps at most ⅓ (X3, 50 digits) |
| 2. Recovering a known result | Show that IAM gives an established result in some limit | No. It shows consistency, and referees expect it, though it's weak as evidence because the formula was built to fit | The floor is Sartori–Pigolotti's one-step copy limit; the record term rides the Jacobson/Cai–Kim first law |
| 3. Comparison with published measurements | A derived number compared with values already measured and published | A check against existing data, with no run on our machines. It counts only if nothing was tuned, and it needs a chance test | Matter's expansion rate against SH0ES; the growth deficit against fσ8; Λ's factor with its own chance test (3 of 414 forms) |
| 4. Blind test on unseen data | Prediction written first, read once | Yes. This is the strongest method, and the only one that commissions an instrument | PHY-084, and A6 next |
| 5. Joint test | One data set tests several connected rows at once | One test for many pieces | The fσ8 ramp pins the rate exponent, and through it c and the shape of E(a) together |

On chaining (your question about neighbors): yes. If A is derived and B follows from A using only accepted physics, B inherits A's standing, and B's derivation needs no separate test. The test is needed only where a row says something new about the world. So build long chains on paper with method 1, and spend one test at the point where the chain makes contact with data. One well-chosen data set can then decide a whole chain.
Where a test can't be avoided:
* Identifications. Proving that the horizon holds ½ is mathematics. Claiming that the horizon's holding is IAM's record, and that it drives matter's expansion rate, is an identification. A referee will accept it only if it predicts something checkable somewhere downstream. That check can often be method 3, published cosmology data, rather than a computer run.
* Instruments. A methylation gauge has to be commissioned on unseen data (method 4), because the claim is that it reads real patients correctly. That's why the cell side is where the data runs happen, and why the unseen data should go only there.
* Independent routes giving the same number (e.g. Ω_m/2 from the black-hole half and from the virial half). This is strong support only if the two routes really share no premise. A referee will check exactly that.
**** Unseen data is now reserved for one sealed commissioning test, which will also carry each cell row's derived prediction as a pre-registered line. Nothing gets queued against unseen data until that test is assembled.

Does each cell type need its own commissioning? No. IAM-A is one line for every cell type and is commissioned once. Metrology handles this the way your protection relays are handled:
* Type testing, done once. For us that's the one sealed run, across a spread of cell types that covers the range IAM-A will be used on.
* A check at every installation. For us that's every sample's own intake and meter checks: coverage, conversion failure, false marks. The chain already runs these on every library.
Two cautions:
* Cell types outside the commissioned range, such as neurons or tumor tissue, need a verification point before IAM-A can be trusted there.
* Met-A is different because it reads against the atlas, which is measured type by type. Its per-type references are measured, not separately commissioned, and the one run checks the method on a sample of types.

### Google Machines Rules

1. Cheapest only, verified that day. Price-check the zones and regions daily before launching (gcp_prices.py), and pick the cheapest that's actually verified, not assumed. Spot machines count against the regular quota.
2. Self-delete when done. Every machine shuts itself down and deletes itself when its job ends, with a hard time limit as a backstop (the launcher's maximum run time). Nothing is left running or billing.
3. Save to the bucket as it runs. Results are copied to the storage bucket in pieces while the job runs, so a recalled spot machine loses nothing. It relaunches itself and resumes from what's already in the bucket.
4. Spread the work across machines to finish faster, within the quota: 32 CPUs in total, so 8 machines of 4 CPUs.
5. Check the machines every 30 minutes, not hourly, and log anything a dispatcher gives up on ("GAVE UP <job> rc <code>") so a failure never looks like a finish. Remember the machine logs are in UTC.
6. Disk: a big alignment chunk fills the default 60 GB disk, so size the disk to the job.
7. The launcher's own checks: a job needs its design sheet and checklist before it launches, and it must define everything it uses (the $BUCKET lesson from November 7).
Private work fits there because those machines aren't public. The web rebuild and the planners' book reading are small, so one cheapest machine at a time is enough, with each output written to the bucket as it's made.
Before launching anything, I'll read the launcher's code and today's prices to confirm each of these is actually enforced, not take it from memory. I'll tell you which are built in and which I have to add.

### Launcher’S Code

| Rule | Enforced in code? | How |
|---|---|---|
| Cheapest, verified that day | Yes | Refuses to launch without today's price check (the prices file dated today); refuses a machine not offered in that region; refuses a price above twice today's spot price. Spot machines only. |
| Self-delete when done | Yes, three ways | When the job ends for any reason, including an error, the machine deletes itself. A timer deletes it at the time limit. Google's own maximum run time is set as a backstop. The disk is deleted with it. |
| Save to the bucket while it runs | Yes | The log goes to the bucket at every step and results go up as they're written. The launcher refuses a job that doesn't save as it goes (it has to say how it resumes). The shutdown script saves the log if Google reclaims the machine. |
| Relaunch when reclaimed | A separate watcher (relaunch_watch.py) | Not in the launcher. It has to be started with the job. Its November 10 lesson: it lost its connection once and only logged errors, so it gets checked. |
| Spread across machines | Partly | The launcher caps the total at the quota (32 CPUs, plus Toronto's N2 limit). Splitting the work into pieces is up to each job's dispatcher. |
| Design sheet and checklist first | Yes | It refuses without them. |
| Every launch recorded | Yes | Time in UTC, machine, zone, price, time limit and purpose (gcp_ledger.csv). |
