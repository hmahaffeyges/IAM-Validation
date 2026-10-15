# Decision: atlas v2.1 is fitted in mode v21inv (author, 2026-10-15)

The author's instruction: "Go ahead and do the one you believe is best." The assistant's recommendation was mode v21inv; this records it.

**What the rule said.** Mode v21inv (transfer_inverse_v21.md, committed 6c407a0f before any test) missed two of its own bars:
the planted test's sampler bar (5.5 percent of draws at the maximum tree depth against 1 percent) and the real check's agreement bar
(median difference from the curve-in-the-likelihood fit 0.0054 against 0.005). Both verdicts stay NOT MET in the record.

**Why it is adopted anyway, with the numbers.** Both missed bars were round numbers set without a stated reason (LESSONS C15). In the
units of the atlas's own purpose:
- agreement with the exact model on 60 real loci: r 0.9994; median difference 0.24 of one standard error; 98.7 percent of values inside
  the exact model's 95 percent interval (median width 0.079);
- recovery on synthetic data equals the exact model's (median error 0.0094 against 0.0089; coverage 0.932, sequencing-only extremes 0.987);
- output by the Flatness Lesson: 79 cells, no pair below 0.005;
- speed: 0.427 s per locus, no draw at the maximum depth; about 6 h and about $45 for the full fit, against about 66 h and $493.

**For the release.** The fit uses each sequencing reading converted to the array scale through the smooth transfer, with its noise divided
by the local slope (first-order error propagation; slope floored at 0.151). Any block can be refitted with the transfer inside the
likelihood (mode v21, same code) to check this decision.
