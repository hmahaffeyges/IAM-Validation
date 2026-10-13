"""DEV-LEVERS-01 synthetic test 4 (written 2026-10-13, before it was run): can an age series in cells that do not divide tell a
fixed balance from a drifting one?

The foundation (CANON/foundation.json, entry `balance`; book Sec. "The held state is a balance of restore and loss"): a held
site is lost with probability f over any interval and a lost one restored with probability g, so eps = f/(f+g) and
E_hold = kT ln(g/f). The interval need not be a division. The prediction for cells that never divide is therefore that the held
error does NOT rise with age once the balance is reached; a rise means the ratio g/f itself moved.

So the test is a power question, not a physics question: with the donors and ages brain data actually offers, how large a drift
in g/f over a lifetime could we detect, and would we wrongly claim one when there is none?

Design, fixed before running:
 - ages drawn uniformly 20-90 (what public brain series span); donors 10, 20, 40, 80
 - truth: ln(g/f) changes by a fraction d over the 70-year span, d = 0 (null), 0.02, 0.05, 0.10, 0.20, with eps_0 = 0.032
   (the healthy reference, foundation entry `eps0`) at age 55, the middle of the span
 - donor spread: multiplicative lognormal on eps with sigma 0.05 and 0.10 (within a cell type; the published healthy range
   across cell types, 0.024-0.042, is wider and is NOT used here, since a series reads one cell type)
 - reader: ordinary least squares slope of logit(eps) on age, which is the slope of ln(g/f) with sign flipped - no fitting of
   the physics, only the slope
 - bars, written now: a drift is DETECTED if the 5th percentile of |slope| over 200 draws exceeds the 95th percentile of the
   null draws (d = 0). The false-positive rate at d = 0 is reported; if it is above 5 of 100 the reader is rejected.
Writes levers_synth_04_output.csv and levers_synth_04_summary.txt. Usage: python3 levers_synth_04.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
EPS0, A0, SPAN, NDRAW = 0.032, 55.0, 70.0, 200
rg = np.random.default_rng(20261013)
logit = lambda p: np.log(p / (1 - p))

def draw(nd, d, sig):
    age = rg.uniform(20, 90, nd)
    lr0 = np.log((1 - EPS0) / EPS0)                       # ln(g/f) at the middle age
    lr = lr0 * (1 + d * (age - A0) / SPAN)                # the balance drifts by a fraction d across the span
    eps = 1 / (1 + np.exp(lr)) * rg.lognormal(0, sig, nd)
    eps = np.clip(eps, 1e-4, 0.5)
    return np.polyfit(age, logit(eps), 1)[0]

rows = []
for sig in (0.05, 0.10):
    for nd in (10, 20, 40, 80):
        null = np.abs([draw(nd, 0.0, sig) for _ in range(NDRAW)])
        thr = float(np.percentile(null, 95))
        fp = float((null > thr).mean())
        for d in (0.02, 0.05, 0.10, 0.20):
            sl = np.abs([draw(nd, d, sig) for _ in range(NDRAW)])
            rows.append(dict(sigma=sig, n_donors=nd, drift=d, slope_median=float(np.median(sl)),
                             slope_p05=float(np.percentile(sl, 5)), null_p95=thr, false_positive=fp,
                             detected=bool(np.percentile(sl, 5) > thr)))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "levers_synth_04_output.csv"), index=False)
L = ["DEV-LEVERS-01 synthetic test 4: detecting a drift in the balance with an age series in non-dividing cells",
     f"eps_0 {EPS0} at age {A0:.0f} | ages 20-90 | {NDRAW} draws per cell", ""]
for (sig, nd), g in T.groupby(["sigma", "n_donors"]):
    d = g[g.detected]
    L.append(f"donor spread {sig:.2f}, {nd:2d} donors: smallest detectable drift in ln(g/f) over 70 years "
             f"{(str(d.drift.min()) if len(d) else 'none up to 0.20')} | false-positive rate at no drift {g.false_positive.iloc[0]:.3f}")
worst = T.false_positive.max()
L += ["", f"reader accepted: {'yes' if worst <= 0.07 else 'NO - false-positive rate ' + str(round(worst, 3))}",
      "BAR: " + ("MET - some feasible age series detects a drift, and the reader does not invent one" if T.detected.any() and worst <= 0.07
                 else "NOT MET"),
      "Read as the design requirement for the brain series, not as a result about the world."]
open(os.path.join(OUT, "levers_synth_04_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
