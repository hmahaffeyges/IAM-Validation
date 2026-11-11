# PHY-050: the per-call firm rate against the M number (sealed before any sample is counted)

**Question.** IAM-A's healthy line, held error = firm rate + slope x loose share, has two measured constants. The counter's held error is a
construct of its own rules: it scores only molecules of at least 6 calls that are at least 80 % marked and counts only isolated losses, so a
molecule carrying a second loss is dropped. Stage Q2 already records the per-call loss rate with that rule's bias removed (`held_e_call`,
`stage_q2_selfref.held_err_length`, exact, nothing fitted), the quantity Boltzmann's occupation 1/(1+e^E) describes. This test asks whether the
per-call firm rate of a fully held region equals the share set by the M number alone.

**Prediction (sealed).** The intercept of `held_e_call` on the loose share, across the healthy Loyfer 2023 samples, equals 1/(1+M) with
M = dG_ATP / RT = 54 kJ/mol / (R x 310.15 K) = 20.9405 (CANON/iam_canon.json): **0.04558**. Equivalently, the holding energy of a fully held
region is ln M = 3.042 kT.

**Data.** The 153 samples of `DEV_CHAIN_Q2_01/loyfer_samples.csv` (Loyfer 2023, GSE186458; 56 healthy cell types), each read by the one command
(`run_sample.py --geo-pat GSM --dev-count-only`: the chain fetches the sample's hg19 .pat.gz from GEO and Stage Q2 counts it on the
commissioned line's tiles), driven by `DEV_CHAIN_Q2_01/percall_count.py` on GitHub's free machines (explore.yml), one shard first as a canary.

**Bars (`DEV_CHAIN_Q2_01/percall_score.py`).** 1. The intercept's 95 % interval (ordinary least squares, one row per sample) contains 0.04558.
2. The interval's half-width is at most 0.003; otherwise the verdict is INCONCLUSIVE. MET needs both.
Reported beside and not scored: the same fit on the counter's held error and on `held_err_length`; the median per-call holding energy by cell
type against ln M; the fit on cell-type medians.

**What it would mean.** Met: the firm rate of a fully held region is set by the cell's energy coin against thermal kicks, with nothing
measured, and IAM-A's middle has its first derived constant (the slope stays measured). Not met: M is not the firm rate on the physical scale,
and the size and sign of the delta is what the next step must explain. Neither result changes the commissioned instrument.

**Known limits, stated in advance.** The per-call correction assumes each call is lost independently. The rate still carries each library's
sequencing and conversion error (the meter, about a tenth of the held error in earlier counts); removing it would lower the rate. All 153
samples come from one laboratory.

**Prior reading:** these 153 samples were counted to build the line (DEV-CHAIN-Q2-01) on the counter's held error; their per-call rate has
never been computed. On 18 other spent healthy libraries (three laboratories, doors/data/DEV_ERASE_01/position_01) the per-call intercept
read 0.0473 +/- 0.0027 today, which is what prompted this test; the 1/(1+M) candidate was itself proposed after the counter-scale numbers
were seen (IAMA_MIDDLE_DERIVATION_STATUS, 2026-11-10).

## Addendum, written before any of the 153 counts was opened (the bars above unchanged)

**The measuring stick, checked on synthetic molecules (offline design check; the counter's rules and `held_err_length` on the real length
mix of a healthy library's qualifying molecules, 200,000 molecules per case, true per-call loss 0.04558):** truth alone, per-call reads
0.04540 (-0.4 %) and the counter 0.03679; a copier error of 0.5 % on marked calls, per-call 0.05030 (+10 %); 1 %, 0.05538 (+22 %); losses
clumped (a loss makes its neighbour lost with 20 % extra chance), 0.03619 (-21 %); clumped and a 0.5 % copier, 0.04059 (-11 %). So the
per-call correction recovers the truth; a copier error adds to it about one for one; clumped losses pull it down.

**How a miss will be read (fixed now).** Let d = per-call intercept - 0.04558.
- d within the interval: met (bar 1).
- d above, at most 0.003: within what the copier's measured error alone can add (PROC-CHANNEL-01: 0.1 to 0.3 % per substitution, ENCODE
  alignments); the copier is then the first candidate, and a copier-corrected prediction is tested next on laboratories not used here.
- d above, more than 0.003: more than the copier's measured error can add; the cell loses more than M alone allows.
- d below: the cell holds better than M alone allows, or its losses clump; the counts' own clumping record (`coupling_pair_share`) is
  read beside.
Nothing will be adjusted to match after the counts are opened.
**Range of the data:** the synthetic used a true per-call loss of 0.04558, copier errors of 0 to 1 % and clumping of 0 to 20 %, on molecules of
6 to 43 calls (median 7); in the real data, the 18 spent healthy libraries read per-call 0.0401 to 0.0763 (counter 0.0333 to 0.0546) on molecules of the same
length mix, and their mirror error on molecules held unmarked (`gain_err`) 1.4 % in the library whose length mix was used.

