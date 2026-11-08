"""Scores the THIRD new-people test (written and committed 2026-11-08 before it runs). Bars chosen with their false-fail rate computed first: for
an instrument whose Normal holds 95 % of healthy people, bar 1 (at least 108 of all 120 new people inside) fails by chance 0.7 % of the time and
bar 2 (at least 16 of 20 in every cell and spread) 1.6 %; the second test's bar (18 of 20 in each of 6) failed a perfect instrument 37 % of
the time. The second test's text follows. Scores the SECOND new-people test (written and committed 2026-11-08 before it runs; the first was not met for B cells at site-by-site spread,
14 of 20, because its 20 donors read narrower than one person read 20 times). Normal: the prediction interval for one new reading from n donors,
median +/- t(0.975, n - 1) x sd x sqrt(1 + 1/n), the textbook rule for a width measured on a finite set of donors. Usage: DONORS_DIR NEW_DIR [REPEAT_DIR] OUT.
The first test's text follows. Scores the new-people test of the measured width (written and committed 2026-11-08 before any new person is read).
The instrument's job: a healthy person reads Normal on the gauge, where Normal is the width measured on healthy donors read the same way.
Normal for each cell and spread: the 20 donors' readings (the width synthetic with the wide line, width_wide_*.json), median +/- 1.96 x their
standard deviation. Bars (each cell and spread): 1. at least 18 of the 20 new healthy people (width_new_*.json) read inside Normal;
2. every new person's main cell found is the planted cell. Reported beside it, not scored: the donors' spread against the truth's and against
the instrument's own repeat spread (width_diag3_repeat_*.json, one person read 20 times), sqrt(truth^2 + repeat^2).
Usage: python3 width_new_score.py DONORS_DIR NEW_DIR [REPEAT_DIR] OUT.txt"""
import sys, os, glob, json
import numpy as np, pandas as pd
dd, nd = sys.argv[1], sys.argv[2]; rd = sys.argv[3] if len(sys.argv) > 4 else None; out = sys.argv[-1]
load = lambda d, pat: pd.DataFrame([r for f in sorted(glob.glob(os.path.join(d, pat))) for r in json.load(open(f))])
Dn, Nw = load(dd, "*.json"), load(nd, "*.json"); Dn, Nw = Dn[Dn.rc == 0], Nw[Nw.rc == 0]
Rp = load(rd, "*.json") if rd else None
TOT = [0, 0]; L = [f"third new-people test (prediction interval): {len(Dn)} donors, {len(Nw)} new people read by the one command"]; ok1 = ok2 = True
for (c, rho), d in Dn.groupby(["cell", "rho"]):
    n = Nw[(Nw.cell == c) & (Nw.rho == rho)]; md, sd = d.reading.median(), d.reading.std(); from scipy.stats import t as _t; k_ = _t.ppf(0.975, len(d) - 1) * np.sqrt(1 + 1 / len(d)); lo, hi = md - k_ * sd, md + k_ * sd
    inside = int(((n.reading >= lo) & (n.reading <= hi)).sum()); found = int((n.main_cell == c).sum())
    rp = Rp[(Rp.cell == c) & (Rp.rho == rho)].reading.std() if Rp is not None and len(Rp) else float("nan")
    L.append(f"  {c:12} rho {rho} | Normal {lo:.4f} to {hi:.4f} (donors {len(d)}, spread {sd:.5f}; truth {d.truth.std():.5f}; sqrt(truth^2 + repeat^2) "
             f"{np.sqrt(d.truth.std() ** 2 + rp ** 2):.5f}) | new people inside {inside} of {len(n)} | main cell found {found} of {len(n)}")
    ok1 &= len(n) == 20 and inside >= 16; ok2 &= found == len(n); TOT[0] += inside; TOT[1] += len(n)
okp = TOT[0] >= 108 and TOT[1] == 120
L += [f"bar 1 (at least 108 of all 120 new healthy people inside Normal): {TOT[0]} of {TOT[1]}, {'met' if okp else 'not met'}",
      f"bar 2 (at least 16 of 20 inside in every cell and spread): {'met' if ok1 else 'not met'}",
      f"bar 3 (every main cell found is the planted cell): {'met' if ok2 else 'not met'}", f"test: {'met' if okp and ok1 and ok2 else 'not met'}"]
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
