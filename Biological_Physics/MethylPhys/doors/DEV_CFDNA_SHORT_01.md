# DEV-CFDNA-SHORT-01: IAM-A's counter on cell-free DNA-length molecules (TIS-053), synthetic

**In plain words.** Cell-free DNA in blood comes in short pieces, about 167 letters long. This test planted pieces of that length, with a known
rate of lost marks, and checked that IAM-A's counter reads that rate back. The first run read about 10 percent low on short and long pieces alike,
which exposed a flaw in one correction step of the counter; that step was fixed in the chain and the second run read the planted rate within
0.3 percent. What it would take to succeed: the short pieces read within 3 percent of the planted rate.

**The instrument's job:** on cell-free DNA-length molecules, IAM-A's counter returns the held error that was planted.

**Synthetic test:** DEV_CFDNA_SHORT_01/cfdna_short_synthetic.py

**Design (written 2026-10-17 before it ran):** planted human-like molecules with methylated open-sea stretches (1 % and 3 % CpG density) and
unmethylated islands (10 %), a planted per-site held error of 0.045 and sequencer error of 0.004. Two arms with the same planted rates:
150-letter reads, and plasma cfDNA fragment lengths (mode 167, 10-letter periodicity). Counted by the chain's counter, read through its response
curve, corrected for the sequencer error. One bar: the cfDNA arm within 3 % of the planted value.

**Run 1, NOT MET (2026-10-17; numbers in the development log):** both arms read about 10 % low. Cause: the response curve was weighted by the length
mix of every molecule, including unmethylated island molecules that can never qualify. The plant synthetic (T2) had no unmethylated regions, so it
could not show this. The same step sits in the temperature test's correction, where it moves the cold/warm ratio by 10-15 % of the effect sought (PRED_TEMPERATURE_01/counter_bias_check.json; LESSONS C23).

**Fix, in the chain:** the counter records the length mix of qualifying molecules only (stage_q2_selfref), and the response curve is computed
exactly from the binomial distribution instead of by simulation (stage_temperature.response).

**Run 2, MET (2026-10-18):** cfDNA arm 0.04515 against 0.045 (0.3 %); long-read arm
0.04436 (1.4 %). The plant synthetic T2 still passes (counter within 1.2 %; law and no-effect designs MET).

**Recorded, not scored:** qualifying molecules per million fragments: 30,700 (cfDNA) and 24,396 (150-letter reads) on this
planted genome. The raw counter output differs by 1.5 % between the two arms for the same planted rate, so cfDNA can be read
through the per-site form; whether IAM-A's line, built on whole-genome reads, holds on cfDNA directly is for the first real cfDNA test.
