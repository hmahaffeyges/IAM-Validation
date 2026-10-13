"""DEV-IAMA-READACROSS-01 planted test (written 2026-10-13, before it was run): does pooling three blocks of a run recover a copy
error when the head of the run is biased, where a head-only read does not?

# NO CHAIN: no reading is re-derived here. The quantity under test is the SAMPLING rule - which blocks of a run are counted - and
# it is applied to per-block COUNTS of the form Stage Q produces (err_A, err_B, opp_A, opp_B). The chain's own reading is what
# produces those counts on real files (read_parts_01.py calls stage_q_iam_a.pat_site_table); this file plants counts with a known
# answer and checks the arithmetic of pooling them, so there is nothing for the chain to do.

Design, fixed before running:
  - 2000 synthetic runs. Each has a TRUE copy error drawn in 0.040-0.056 (the range the five cause-test libraries span) and three
    blocks at 0, 40 and 80 % of the file with 400,000 to 1,000,000 opportunities each (the cause test's blocks held 0.49M to 0.99M
    sites), so blocks differ in weight as they really do.
  - The head block is biased LOW in copy error by a factor drawn from the measured range: A (binary entropy) 11 to 14 % low, which
    is what DEV-IAMA-CAUSE-01 measured; the other two blocks are unbiased.
  - Estimators compared: HEAD-ONLY (block 0 alone, the commissioning run's rule) and POOLED (counts summed over three blocks, the
    proposed rule).
  - BAR, written now: the pooled estimate must land within 1 % relative of the planted truth at every planted bias, and the
    head-only estimate must be outside 5 % at the largest bias. If pooling cannot recover the planted truth, it is not a remedy.
Writes planted_readacross_01_output.txt. Usage: python3 planted_readacross_01.py [OUTDIR]
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
rg = np.random.default_rng(20261013)
N = 2000
H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e))
L = ["DEV-IAMA-READACROSS-01 planted test: can pooling three blocks recover a copy error when the head of the run is biased?", ""]
rows = []
for a_bias in (0.11, 0.125, 0.14):                       # A is this fraction LOW in the head block, as measured
    head_rel, pooled_rel = [], []
    for _ in range(N):
        eps_true = rg.uniform(0.040, 0.056)
        opp = rg.integers(400_000, 1_000_001, 3)
        # the head block's A is (1 - a_bias) x the true A; invert H numerically for the eps that gives it
        a_head = H(eps_true) * (1 - a_bias)
        grid = np.linspace(1e-4, 0.5, 20001); eps_head = float(grid[np.argmin(np.abs(H(grid) - a_head))])
        err = rg.binomial(opp, [eps_head, eps_true, eps_true])
        head = err[0] / opp[0]
        pooled = err.sum() / opp.sum()
        head_rel.append(head / eps_true - 1); pooled_rel.append(pooled / eps_true - 1)
    h, p = np.array(head_rel), np.array(pooled_rel)
    rows.append((a_bias, float(np.median(h)), float(np.median(p)), float(np.percentile(np.abs(p), 95))))
    L.append(f"head biased {a_bias:.3f} in A | HEAD-ONLY estimate {np.median(h) * 100:+.2f} % off the truth "
             f"| POOLED {np.median(p) * 100:+.3f} % off, 95th percentile of |error| {np.percentile(np.abs(p), 95) * 100:.3f} %")
pool_ok = all(abs(r[2]) <= 0.01 and r[3] <= 0.01 for r in rows)
head_bad = abs(rows[-1][1]) > 0.05
L += ["", f"BAR: pooled within 1 % at every bias: {'yes' if pool_ok else 'NO'} | head-only outside 5 % at the largest bias: "
      f"{'yes' if head_bad else 'NO'}",
      "PLANTED TEST " + ("PASSED - pooling three blocks recovers a planted truth that a head-only read does not"
                         if (pool_ok and head_bad) else "FAILED - the pooled rule is not a remedy for this bias"),
      "", "This is a statement about the sampling rule, not about any library or any laboratory."]
open(os.path.join(OUT, "planted_readacross_01_output.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
