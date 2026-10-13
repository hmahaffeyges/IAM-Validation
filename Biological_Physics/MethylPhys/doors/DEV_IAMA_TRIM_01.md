# DEV-IAMA-TRIM-01 — the trim rule for the sequencing instrument (design and planted test, 2026-10-13; NOT sealed)

**Why.** The across-laboratory commissioning check failed on the one kit made in two laboratories. The cause test
(`DEV_IAMA_CAUSE_01.md`) and the diagnostics review (`DEV_IAMA_READACROSS_01.md`) established three things: the head of a run
differs on a quantity independent of our reading (conversion, by 34 standard errors, in every library); the three failing libraries
read 11 to 14 % below the controls read at the SAME position; and part 0 reproduces each library's sealed Stage B value to 0.0 %, so
**Stage B had read the head**. The first proposed remedy - read several blocks and pool - failed its own planted test, because
pooling divides a bias rather than removing it. The head must be excluded.

**The constraint that makes this defensible.** How much to exclude must be decided by something measured independently of IAM-A.
Trimming until the answer improves is tuning, and a reviewer should refuse it.

## The diagnostic: the raw reads' own chemistry, no alignment and no reference

In a bisulfite library almost every cytosine outside a CpG is converted to thymine, so in the READ SEQUENCE the fraction of C among
C+T at positions not followed by G is near zero and is a property of the chemistry, not of the genome and not of our physics:

    d(window) = count(C not followed by G) / count((C or T) not followed by G)

It needs no alignment, so it costs pennies to compute in windows along a file, and anyone can check it in a few lines.

## The rule, fixed before any real file is trimmed (`trim_rule_01.py`)

1. read the file in windows of 50,000 reads;
2. the reference level is the MEDIAN d over the windows in the last half of the file, with a robust standard deviation (1.4826 x MAD);
3. walk from the start, skipping windows while `|d - reference| > 5 robust SD`;
4. never skip more than 25 % of the file - above that the library is **REFUSED**, not trimmed, because a library anomalous
   throughout is not a trimming problem;
5. the reading is then the chain's Stage Q, unchanged, on what the rule admits.

`K_SD = 5` and `MAX_TRIM = 25 %` are fixed now and are not to be adjusted after seeing a reading.

## The planted test: six bars, written before it ran (`planted_trim_01.py`), ALL MET

| bar | what it plants | result |
|---|---|---|
| 1 recovery | head anomaly over 10 % of the file, head eps biased -12 % | TRIM, 4 of 40 windows; read -0.14 % off truth (from -1.02 %) |
| 2 no harm | an unbiased library | READ ALL, skipped nothing, reading moved 0.0000 % |
| 3 refusal | chemistry off across the whole first half | **REFUSE** at 50 %, as required above 25 % |
| 4 sign blindness | the same chemistry anomaly with eps biased -12 %, then +12 % | 4 windows skipped in both cases - identical |
| 5 declared limit | head chemistry NORMAL, head reading biased -12 % | trims NOTHING, as required |
| 6 the real configuration | the reading taken from the HEAD ALONE, as Stage B took it | head-only read **-12.25 %** off truth; after the rule **-0.21 %** |

**Bar 6 was added before the test ran, because bar 1 was too easy.** Diluting a -12 % head bias across a whole file leaves only a
-1 % error, which is already inside the sealed 3 % tolerance - so bar 1 alone would not have exercised the failure we actually had.
Bar 6 reads the head alone, as Stage B did, so the bias arrives undiluted.

**Bar 5 is the rule's honest limit, declared rather than hidden.** If a library's chemistry looks normal at the head but its reading
is biased there, this rule trims nothing and the bias survives - by design, because the rule may not act on a bias it has no
independent evidence for. Such a library is caught by the commissioning check, not by the trim.

## What has NOT been done

Nothing real has been trimmed. Calibrating this on the EpiQC libraries needs the diagnostic computed from their raw reads, which is
one cheap box read (no alignment), and the across-laboratory check then has to be rerun on the WHOLE stage, not only the five
libraries used for the diagnosis - those five are spent.

## Calibration on all 71 libraries, run once (2026-10-13): the rule works, and it does NOT explain the failed laboratory check

`calib_trim_all_01.sh` (committed before it ran; box run 2026-10-13, the code's checksums in `calib_trim_all_01_code_checksums.txt`
match the commit), merged by `merge_trim_all_01.py`; outputs `calib_trim_all_01_windows.csv`, `calib_trim_all_01_decisions.csv`,
`calib_trim_all_01_summary.txt`. 1,420 windows read, none failed; K_SD and MAX_TRIM unchanged.

- **Where the rule acts:** only in TrueMethylBS (laboratory 1): 6 libraries trimmed, 1 refused as anomalous through half its file.
  No EMSeq, MethylSeq, SPLAT or TruSeq library is trimmed.
- **The three suspect EMSeq libraries are not trimmed.** Their head chemistry sits within 3 robust SD of the rest of the same file
  (bar 5 needs more than 5), the same as the two controls. Whatever made their heads read 11-14 % low, it is not the conversion
  chemistry this rule measures.

**What it means.** This is bar 5, the limit we declared before the run: the rule will not act on a bias it has no independent evidence
for. So the trim rule cannot release the instrument. The across-laboratory check stays NOT MET and IAM-A stays held. The trims it
does make in TrueMethylBS are recorded, but none of those libraries was part of the failed check.

**Next, before any more box money:** look for an independent diagnostic that does separate the suspects from the controls at the
head, using the quality files the cause run already saved (read quality, duplicates, insert size by position), with its bar written
first. If none separates them, the laboratory gap is treated as real and EMSeq's cross-laboratory scope is reported as failed.

**Prior reading:** the 71 EpiQC Stage B libraries are already read by DEV-IAMA-COMMISSION-01 and DEV-IAMA-CAUSE-01; this pass reads
only their raw-read chemistry, not the physics, and none of them may later serve as a held-out test of IAM-A.

**In plain words.** The beginning of a sequencing file read wrong, and that is where our original reading had been taken. We cannot
just average more of the file, because averaging in a bad measurement only shrinks the error instead of removing it. So we skip the
bad beginning - and to keep that honest, how much to skip is decided by the sequencing chemistry's own fingerprint in the raw reads,
which knows nothing about our physics, rather than by whether our answer improves. We then tested the rule on made-up files where we
knew the truth: it recovered a planted 12 % error to within a fifth of a percent, left a clean file untouched, refused a file that was
bad throughout, behaved identically whether the planted error pushed up or down, and - importantly - did nothing at all when the
chemistry looked fine, which is a limit we are declaring rather than hiding. **What it would take to finish:** compute that chemistry fingerprint
from the raw reads of the EpiQC libraries - one cheap reading pass on the rented machine with no alignment step - then rerun the
failed laboratory check on the whole set of libraries rather than the five we already used up.

**Synthetic test:** DEV_IAMA_TRIM_01/planted_trim_01.py, DEV_IAMA_TRIM_01/planted_trim_01_summary.txt

**Not a milestone:** a rule, its planted test and its calibration; the failed check is not explained by it and the instrument stays held.
