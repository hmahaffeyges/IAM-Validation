# Task: remove the opening chapter, add the six-step derivation as an appendix

Branch off `main`, one pull request, do not merge. The author reads it before it goes in.

## 1. Remove Chapter 1 (`part1/p1_01_encoding_surfaces.tex`, `ch:surfaces`)
The author's ruling (2026-10-05): it is a poor opening. It uses terms and results from later Parts before the reader has met them,
and it uses "floor" the wrong way round.
- Take its `\input` out of `main.tex` and move the file to `development/archive/`. Do not delete it.
- About 51 `\ref{ch:surfaces}` (and refs to its sections and figures) point at it. Repoint each to where the topic is developed now:
  encoding surfaces and the cost per bit go to `ch:law` (p1_02); black holes go to Part III; the cell goes to Part VI; the full chain
  goes to the new appendix. Where a sentence only makes sense with the old chapter, change it as little as possible.
- Figures used only by that chapter go to `development/archive/` with it. Remove their checks from `verify_book.py`.

## 2. New appendix: the six-step derivation
Working title "Saturation: one cost from the horizon to the cell". The author picks the final title.
- **Steps 1-5** (Bekenstein, Hawking, Landauer, Jacobson, IAM) are in `docs/tasks/sources/saturation_steps_1_to_5.txt`, the author's
  own text. Carry them near word for word, in LaTeX, with the book's equations and macros. Give every equation its status label:
  established results are OBSERVED or DERIVED, as elsewhere in the book; Step 5 is the one new identification.
  Cross-reference each step to the chapter that develops it.
- **Opening statement**, in the author's voice: a system reaches the limit of what it can maintain when its rate of irreversible
  information production saturates the capacity of its nearest encoding surface. At stellar scale this is the black hole. At cell
  scale it is the cell's record leaving its maintained state. Do NOT write that entropy or IAM drives cosmic expansion: in IAM the
  background does not change.
- **Floor and ceiling (essential):** the floor is H_min, the minimum entropy a healthy maintained record holds, measured on purified
  healthy cells. The black hole is the opposite end: the surface saturated at its full capacity, the ceiling. Never call the star's
  collapse a "floor breach", and do not use "floor breach" anywhere.
- **Step 6, the cell (write new from the current chain, not from any older report):**
  the encoding surface is the methylation record; T = 310.15 K; N = the CpG sites in the record; cost per bit k_B T ln 2 =
  2.97 x 10^-21 J (CALCULATED). H_min is the floor measured on purified healthy cells through chain v3, with nothing fitted. Met-A
  (arrays) and IAM-A (single-molecule sequencing) read how far a cell's record sits above that floor. The identification of the
  cell's approach to its ceiling with the black hole's is CONJECTURE.
  No tier words, no thresholds such as 1.05 or 1.10, no clinical or cancer-prediction claims, no development readings.
- **Structural identity table**, black hole against cell, with these rows: encoding surface; temperature (T_H against T_body);
  bit count (A/4 l_P^2 against N sites); cost per bit (k_B T ln 2 at each temperature); floor (none for the horizon; H_min for the
  cell); the ceiling (horizon saturation; full disorder of the record); what is read (the horizon; Met-A, IAM-A).
  Close with the author's line: the same equation N k_B T ln 2 at both ends, with T and N differing by many orders of magnitude.
- Leave out the source paper's "empirical confirmation" section: it held pre-chain results.

## 3. Rules
- Every new number gets a check in `verify_book.py` with its control. verify_book must stay at 0 FAIL, and the book must compile.
- Do not change mu0, beta_m, either H0 or any chain result. Write in the author's positive, inviting tone. No population or cohort
  wording. Do not name any private product or place.
- Also, in the same PR: full SOP §9 item 9 (self-tare II, done 2026-10-04) moves out of "Pending changes" into the main text, the
  same way the noise gate did.
- List every repointed reference in the PR (old target, new target).
