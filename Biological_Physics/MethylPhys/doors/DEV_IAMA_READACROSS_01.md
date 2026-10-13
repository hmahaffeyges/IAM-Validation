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

**Not a milestone:** a proposed sampling rule with its test unwritten; nothing is sealed and no reading has changed.
