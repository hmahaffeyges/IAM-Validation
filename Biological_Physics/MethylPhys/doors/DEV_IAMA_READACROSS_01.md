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

## Diagnostic 2, run once (2026-10-13): duplicates at the head separate suspects from controls - a lead, not a finding

After the chemistry fingerprint failed to separate them (DEV-IAMA-TRIM-01), `head_quality_02.py` (bar committed first) compared the
head's duplicate fraction with deeper in the same file, from the cause run's committed quality files (`head_quality_02_output.txt`).
**Both controls carry about 4 points more duplicates at the head (+0.045, +0.041); the three suspects do not (-0.001, +0.022, -0.005).**
By the bar, that separates them. But with 3 suspects and 2 controls a meaningless measure does this 1 time in 5, so it is a lead to
test on libraries we have not used, not an explanation. A second measure (mapped reads per read scanned) was dropped as malformed
after its first run: the denominator counts only the reads sampled for the conversion check.

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
fraction, mapped reads and reads scanned; `head_summary_01.py` computes every number below from it and from the cause run's readings.
Three findings, and the third is the mechanism.

**1. The head of a file is different on a quantity that has nothing to do with our reading.** Conversion at the head averages
0.99803 (median standard error 0.000006, from about 50 million CHH calls) against 0.99040 deeper in (median standard error 0.000224,
from 85,000 to 330,000 calls). The difference, 0.0076, is **34 standard errors of a deeper part** - real, and not an artifact of the
quality step scanning fewer reads deeper in (2,000,001 reads at the head against 3,925 to 13,836 deeper). Duplicate fraction shifts
the same way, 0.2201 at the head against 0.1998 deeper. This holds in **all five libraries, controls included**, so it is a position
effect common to the platform and not something that distinguishes the suspect laboratory.

**2. Two different framings, and only one of them answers the question.** In ABSOLUTE copy error the head is higher than the mean
of the deeper parts in **four of the five libraries**; the fifth, SRR13051128, reads LOW at the head (0.04561 against a deeper mean of
0.04648), and the per-library signs are listed in `head_summary_01_output.txt`. So the absolute framing is not even consistent in
sign across libraries - which is exactly why an earlier draft of this section, written from the absolute values, had the per-library
pattern wrong. The sealed decision does not compare a part against other parts of the same library; its quantity is
**r = A / (mean A of the two normal runs AT THAT PART)**, which cancels the common position effect by construction.

| library | role | r at the head | r at p40 | r at p80 |
|---|---|---|---|---|
| SRR13051128 | suspect | **0.8578** | 0.9791 | 0.9591 |
| SRR13051256 | suspect | **0.8740** | 0.9775 | 0.9635 |
| SRR13051268 | suspect | **0.8912** | 0.9870 | 0.9872 |
| SRR13051136 | control | 0.9988 | 1.0130 | 1.0182 |
| SRR13051263 | control | 1.0012 | 0.9870 | 0.9818 |

So the finding is sharper than "the head is unreliable": **at the head the suspect libraries read 11 to 14 % below the controls read
at the same position**, and deeper in they come back to within 1 to 4 %. The deficit is specific to those libraries at that position,
on top of the position effect everyone shares.

**3. The mechanism, and why the commissioning failed.** The decision script's own check reports that **part 0 reproduces each
library's sealed Stage B reading to 0.0 % in all five cases** (`p0_check: PASSED`). Stage B therefore read the head of each file - so
the sealed commissioning values inherited exactly the part of the file where the suspects are depressed. That is the whole chain of
events: Stage B read the head, the head is where those three libraries are 11 to 14 % low against the controls, and the
across-laboratory check failed on them.

**What this settles about the next rule (DEV-IAMA-TRIM-01).** Excluding the head is right, and the exclusion cannot be replaced by a
correction: the common position effect could be corrected, but the suspects' extra deficit at the head is library-specific and there
is nothing to predict it from. How much to exclude must be decided by a position-resolved diagnostic that needs neither our reading
nor an alignment - the per-cycle base composition and apparent C-to-T fraction of the RAW READS, in windows along the file.
Alignment-based conversion would work too, but it costs an alignment per window, and the cause run's BAMs went with the terminated
machine: only the per-part values in `head_diagnostics_01.csv` survive.

**In plain words.** We asked whether the sequencing machine's own numbers, which know nothing about our physics, can show that the
beginning of a data file is untrustworthy. They can: at the beginning, the chemistry's conversion rate differs from the rest of the
file by thirty-four times its own uncertainty, and that is true of every library we looked at, including the two we used as controls.
Then the sharper question: our failed check compared each library against the two control libraries read at the same place in their
files - and at the beginning, the three libraries that failed read eleven to fourteen percent lower than the controls, while deeper
in they came back to within a few percent. And the reason this mattered at all is that our original reading had been taken from the
beginning of each file: the check inherited the worst place to look. So the fix is to stop reading there, and to decide how much to
skip from a measurement taken off the raw reads before any alignment - cheap, position by position, and checkable by anyone.

**Not a milestone:** a proposed sampling rule that failed its own planted test, with the next rule named but unwritten; nothing is sealed and no reading has changed.
