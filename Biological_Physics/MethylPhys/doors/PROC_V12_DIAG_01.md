# PROC-V12-DIAG-01 — diagnostic (not a bar): is the V12 scatter selection noise or donor spread?

**Written 2026-09-29 before it runs.** Reports only; changes nothing in the chain; no bar.

For every cell with ≥ 4 samples, **leave-one-out**: choose loci from the other n − 1 samples (every one inside b* ± 0.05, on the atlas
scale), read the left-out sample; repeat for each sample. Two selection variants on the same splits:
- **L-plain:** as above.
- **L-stable:** additionally require the n − 1 samples' own spread at the locus (SD) ≤ 0.03 — a locus the cell holds at its floor in
  every donor, not only on average.
Reported per cell: median |A − 1| and fraction in NORMAL, against the number of samples used to choose.
If |A − 1| falls as the number of choosing samples rises, and L-stable is tighter than L-plain, the V12 scatter is mostly selection
noise and a production rule using all samples plus donor stability is the fix (a new procedure, PROC-V12-IDENTITY-03, pre-registered
before its build). If it does not fall, the spread is real and is reported to the author as a physics question, not tuned away.
