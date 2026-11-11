# PHY-050, the per-call firm rate against the M number: outcome NOT MET

Sealed in `PHY_050_PERCALL_PREREG.md` (commit 3a44cf72; addendum f48f3905, both before any count was opened). Each of the 153 healthy Loyfer
2023 samples (GSE186458, 56 cell types) was read by the one command (`run_sample.py --geo-pat GSM --dev-count-only`) on GitHub's machines
(40 shards, one sample first as a canary) and scored by `DEV_CHAIN_Q2_01/percall_score.py` (`data/PHY_050_PERCALL/percall_summary.txt` and `outcome_numbers.txt`; counts in `data/DEV_CHAIN_Q2_01/percall_out/`).

**Not a milestone:** the sealed prediction was not met.

| Bar | Result | Met |
|---|---|---|
| 1. the per-call intercept's 95 % interval contains 1/(1+M) = 0.04558 (M 20.9405) | intercept 0.05048, interval 0.04930 to 0.05165 | no |
| 2. the interval's half-width at most 0.003 | 0.00117 | yes |

**In plain words.** With the counter's own selection bias removed, a fully held region of a healthy cell loses 5.05 of every 100 marks,
not the 4.56 the M number predicts: 10.8 % above it, 8.2 standard errors, in this one laboratory.

**Read as fixed in advance (the addendum).** The miss is above the prediction by 0.0049, more than the 0.003 the copier's measured error
alone can add (0.1 to 0.3 % per substitution); by the sealed reading, the cell loses more than M alone allows, or this laboratory's meter is
larger than the measured range. No per-sample copier error was measured here; that is the next sealed test, on laboratories not used.

**Recorded beside the verdict, not scored (from the same summary):** the counter's held error intercept 0.04018 (0.03946 to 0.04090);
the length-corrected value (the chance that a site is a lone loss between two held calls, e(1-e)^2, the counter's own axis) intercept
0.04558 (0.04465 to 0.04651); a fit on the cell types' medians 0.04977 (0.04773 to 0.05180); the median per-call holding energy over cell
types 2.789 kT against ln M 3.042. The length-corrected value equals 1/(1+M) to four digits, but it was not the sealed quantity and was
seen after the verdict; it is a lead for a new sealed test on data not used here, not a result. (A per-call rate of 0.0506 gives a lone-loss
chance of 0.04558.)

**Prior reading:** GSE186458 was read before to build IAM-A's line (DEV-CHAIN-Q2-01, `CANON/data_spent.csv`); the per-call quantity was not
computed on it before this test.

**What it would take to succeed.** A prediction for the firm rate, derived from physics and sealed before reading, whose interval holds on
healthy samples from at least two laboratories with each library's own copier error measured and removed. The next sealed tests: the same
quantity with each library's copier error measured from its own calibrators; and the lone-loss rate against 1/(1+M) on laboratories not
used here.
