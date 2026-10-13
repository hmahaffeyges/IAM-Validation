# DEV-IAMA-READACROSS-01 — how a sequencing run is sampled before IAM-A reads it (design, 2026-10-13; NOT sealed)

**Why.** DEV-IAMA-CAUSE-01 (scored 2026-10-13) found that three EMSeq libraries from one laboratory read 11 to 14 % low in copy
error at the START of their files and recover to within 1 to 4 % deeper in, while two control libraries read the same at all three
positions. The commissioning run had read every library from the head of its file, so the across-laboratories check compared a
biased sample in one laboratory against an unbiased one in the other. The gap was therefore mostly an artifact of WHERE in the run
we read, and the remedy is a sampling rule, not a change to any reading.

**The rule proposed (not yet sealed).** A library's copy error is read from THREE equally spaced blocks of the run - at 0, 40 and
80 % of the file - pooled as counts before the error is formed:

    eps(library) = (sum of err_A + err_B over the three blocks) / (sum of opp_A + opp_B over the three blocks)

Counts are pooled, not the three error rates averaged, so a block with fewer usable molecules carries proportionally less weight.
Nothing in the chain's reading changes: Stage Q is called exactly as now, three times instead of once, and the sum is what it reads.

**Why three, and why those positions.** Three is the smallest number that can show a trend rather than a difference, and these are
the positions already read for all five libraries of the cause test, so the rule is being fixed on positions whose behaviour is
already documented rather than chosen to flatter a result. Equal spacing avoids weighting the start, which is the part the cause
test showed is least settled.

**What must happen before it is sealed, in order.**
1. **Planted test (this note's synthetic step).** Build runs whose true copy error is known, give the head of each run a bias of
   the size the cause test measured (11 to 14 % low in A), and check that the pooled three-block rule recovers the truth while a
   head-only read does not. Bar, written before running: pooled estimate within 1 % relative of the planted truth at every planted
   bias, and the head-only estimate outside 5 % at the largest bias. If the pooled rule cannot recover a planted bias, it is not a
   remedy and this note says so.
2. **One full-file reading, as ground truth.** One library read end to end, so the three-block rule can be compared against the
   whole run rather than against a simulation only. Costed before it is launched.
3. **Seal**, then re-read the WHOLE stage on the new rule. The five libraries of the cause test are spent on the diagnosis and
   cannot be the evidence for the fix.

**Synthetic test:** DEV_IAMA_READACROSS_01/planted_readacross_01.py, DEV_IAMA_READACROSS_01/planted_readacross_01_output.txt

**In plain words.** A sequencing machine's first cycles are its least settled, so the beginning of a data file is not quite like the
rest. We had been reading each file from its beginning, which made one laboratory's libraries look better than they were. The fix is
to read three places in every file and add the counts together. Before we trust that fix we plant a known answer in made-up runs,
bend the beginning of each one the way the real files were bent, and check that the three-place rule still finds the right answer
while the old head-only way does not. **What it would take to finish:** that test, one real file read end to end as a check, a seal,
and then the whole stage read again on the new rule.

## Outcome of the planted test: the pooled rule FAILED its bar, and the arithmetic says why (2026-10-13)

Run once, bar unchanged (`planted_readacross_01.py`, output `planted_readacross_01_output.txt`):

| head bias planted in A | head-only estimate | pooled three-block estimate | 95th percentile of pooled error |
|---|---|---|---|
| 0.110 | -14.39 % off the truth | **-4.86 %** | 6.59 % |
| 0.125 | -16.33 % | **-5.39 %** | 7.40 % |
| 0.140 | -18.20 % | **-6.05 %** | 8.24 % |

**Why it failed, and it is not a tuning problem.** Pooling counts over three blocks still COUNTS the biased block. With three blocks
of similar weight it divides the bias by about three and no more: a 15 % error becomes a 5 % error. The commissioning tolerance is
3 %, so a diluted bias is still a failing bias. No choice of positions or weights fixes this, because the quantity being averaged is
contaminated rather than noisy. **Adding more samples cannot remove a bias.**

**What this rules out, and what it leaves.** Reading more places in the run is not the remedy. The remedy must EXCLUDE the part of
the run that is not settled, and the exclusion must be decided by something measured INDEPENDENTLY of IAM-A - otherwise we would be
discarding the data whose answer we dislike, which is the one thing a reviewer should refuse to accept. The per-cycle base quality
and the conversion rate reported by `alignment_qc.py` are such independent diagnostics: they are properties of the sequencing run,
not of the reading.

**The revised rule to design and test next (DEV-IAMA-TRIM-01, not yet written).** Decide, from per-cycle quality and conversion
alone, how many leading cycles of a run are unsettled; trim those cycles for EVERY library by the same measured rule; then read.
Its planted test must show that the trim recovers a planted head bias to within 1 %, that it changes an unbiased library's reading
by less than the 3 % tolerance (so trimming is not itself a new bias), and that the trim length is chosen by the quality diagnostic
rather than by the reading it feeds.

**In plain words, corrected.** Our first idea was to read three places in each file instead of one and add them up. We tested that
idea on made-up runs with a known answer and bent beginnings, and it failed: averaging in a bad measurement makes it smaller, not
absent - a 15 % error became a 5 % error, where our tolerance is 3 %. So the fix cannot be to read more of the file; it has to be to
stop reading the part of the file that is not trustworthy, with the decision made from the machine's own quality numbers rather than
from our result. That is the next thing to design, and this idea is recorded as tried and failed.

## What the independent diagnostics say, from the 15 committed quality files (2026-10-13)

`DEV_IAMA_READACROSS_01/head_diagnostics_01.csv` holds every part's conversion rate with its binomial standard error, duplicate
fraction, mapped reads and the number of reads the quality step scanned. Two findings, and the second changes the plan.

**1. The head of a file really is different, and it is measurable without IAM-A.** Conversion at the head averages 0.99803
(median SE 0.000006, from about 50 million CHH calls) against 0.99040 deeper in (median SE 0.00022, from 85,000 to 330,000 calls).
The difference, 0.0076, is **34 standard errors of a deeper part** - real, and not an artifact of the quality step scanning fewer
reads deeper in. So a position-resolved diagnostic measured independently of our reading CAN detect that the head of a run is
anomalous.

**2. But the head effect runs in OPPOSITE directions in the two groups, so no correction can fix it.** All five libraries show the
conversion difference, controls included, so conversion does not distinguish the suspects. And in the reading itself:

| library | role | eps at the head | eps deeper | head effect |
|---|---|---|---|---|
| SRR13051128 | suspect | 0.04561 | 0.04831, 0.04465 | reads LOW |
| SRR13051256 | suspect | 0.04676 | 0.04821, 0.04493 | reads LOW |
| SRR13051268 | suspect | 0.04800 | 0.04884, 0.04641 | reads LOW |
| SRR13051136 | control | 0.05600 | 0.05058, 0.04837 | reads HIGH |
| SRR13051263 | control | 0.05619 | 0.04884, 0.04607 | reads HIGH |

The three suspects read low at the head and both controls read high. So "the head of a file is unreliable" is the right conclusion
and "the head is biased by amount X" is the wrong one: the sign depends on the library. That is why the sealed decision returned
MIXED rather than a clean file-position verdict, and it rules out any approach that estimates the head's bias and corrects for it.
**The head has to be excluded, not corrected** - which is what the revised rule above already requires, now with a measured reason.

**What this settles about the next rule (DEV-IAMA-TRIM-01).** How much to exclude must be decided by a position-resolved diagnostic
that needs neither our reading nor an alignment, so it can be computed cheaply for every library and audited by anyone: the per-cycle
base composition and the apparent C-to-T fraction of the RAW READS, in windows along the file. Alignment-based conversion would also
work but costs a full alignment per window, and the cause run's BAMs went with the machine, so position-resolved conversion cannot be
recomputed from what was kept - only the per-part values in this file survive.

**In plain words.** We checked whether the sequencing machine's own quality numbers, which have nothing to do with our reading, can
tell that the beginning of a data file is untrustworthy. They can: the conversion rate at the start differs from the rest by
thirty-four times its own uncertainty. But the beginning does not push every library the same way - three read too low there and two
read too high - so there is no correction to apply, only a part to leave out. The next rule therefore decides how much of the start to
leave out using a measurement taken straight from the raw reads, before any alignment: cheap enough to do for every library, and
simple enough for anyone to check.

**Not a milestone:** a proposed sampling rule that failed its own planted test, with the next rule named but unwritten; nothing is sealed and no reading has changed.
