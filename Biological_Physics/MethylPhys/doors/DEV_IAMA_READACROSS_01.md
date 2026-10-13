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

**Not a milestone:** a proposed sampling rule that failed its own planted test, with the next rule named but unwritten; nothing is sealed and no reading has changed.
