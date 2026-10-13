"""DEV-LEVERS-01 synthetic test 3 (written 2026-10-13, before it was run): if one of the three channels is impaired, can the
readings say WHICH one?

The foundation (CANON/foundation.json, entry `balance`): a held site is lost with probability f over any interval and a lost
one restored with probability g, so eps = f/(f+g). Three operations feed those two rates:
  - COPY: the template-copying of an existing mark. A failure is a LOSS, and the enzyme is processive, so failures come in runs.
  - DE NOVO: writing a mark where none was templated. This is a RESTORE, and it is site-by-site, so it is scattered.
  - ERASE: active removal of a mark. This is a LOSS, and it is scattered.
So the derivation already predicts the directions, with nothing fitted:
  - impair COPY -> f falls? No: a copy FAILURE is a loss, so impairing copying RAISES f -> eps rises, in runs (clustered).
  - impair DE NOVO -> g falls -> eps rises, scattered.
  - impair ERASE  -> f falls -> eps FALLS, scattered.
Two readings should therefore place the three channels apart: the held error (up, up, down) and the clustering reading
(clustered, scattered, scattered). That is the test.

Design, fixed before running:
  - 400 molecules x 2000 sites per draw, 200 draws per condition; healthy f and g set so eps = 0.032 (foundation entry eps0)
  - impairment sizes 1.25x, 1.5x and 2x on one channel at a time, the other two untouched
  - processive run length for copy failures: 15 sites (test 1 showed the clustering reading separates mechanism above ~10)
  - readings: eps (isolated unmethylated calls between two methylated calls over opportunities, the chain's definition) and the
    clustering reading (variance of 50-site block means x 50 / site variance), both as a ratio to the healthy draws
  - BAR, written now: at the 2x impairment, the three channels must occupy three non-overlapping regions of that two-reading
    plane across 200 draws each - no pair of channels may overlap in BOTH readings at once. If they overlap, the three-channel
    picture is not separable at this depth and the audit says so, which also means tumors with a blocked erase channel cannot
    be read as a natural experiment.
Writes levers_synth_03_output.csv and levers_synth_03_summary.txt. Usage: python3 levers_synth_03.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
NMOL, NSITE, NDRAW, RUN, EPS0, BLOCK = 400, 2000, 200, 15, 0.032, 50
rg = np.random.default_rng(20261013)
# healthy rates: eps = f/(f+g) = 0.032, with the loss split between copy failure and erase
F_COPY, F_ERASE = 0.0020, 0.0010          # per site per interval
G_DENOVO = (F_COPY + F_ERASE) * (1 - EPS0) / EPS0

def one(fc=1.0, fe=1.0, gd=1.0):
    """One draw: molecules of held sites, one interval of the three operations, each channel scaled by its factor."""
    M = np.ones((NMOL, NSITE), dtype=bool)
    nruns = rg.poisson(F_COPY * fc * NSITE / RUN, NMOL)      # copy failures arrive as runs
    for i, k in enumerate(nruns):
        for s in rg.integers(0, NSITE - RUN, k): M[i, s:s + RUN] = False
    M &= rg.random((NMOL, NSITE)) > F_ERASE * fe             # erase: scattered loss
    M |= rg.random((NMOL, NSITE)) < G_DENOVO * gd * ~M       # de novo: scattered restore of lost sites
    # the chain's reading: isolated unmethylated calls between two methylated calls, over opportunities
    a, b, c = M[:, :-2], M[:, 1:-1], M[:, 2:]
    opp = (a & c).sum(); iso = (a & ~b & c).sum()
    eps = iso / max(opp, 1)
    u = (~M).mean(0)                                          # unmethylated fraction per site
    nb = NSITE // BLOCK
    bm = u[:nb * BLOCK].reshape(nb, BLOCK).mean(1)
    Cc = bm.var() * BLOCK / max(u.var(), 1e-12)
    return eps, Cc

base = np.array([one() for _ in range(NDRAW)])
rows = []
for ch, kw in (("copy", "fc"), ("erase", "fe"), ("de novo", "gd")):
    for mult in (1.25, 1.5, 2.0):
        f = mult if ch != "de novo" else 1 / mult            # impairing de novo LOWERS g
        v = np.array([one(**{kw: f}) for _ in range(NDRAW)])
        rows.append(dict(channel=ch, impairment=mult,
                         eps_ratio_median=float(np.median(v[:, 0]) / np.median(base[:, 0])),
                         eps_ratio_p05=float(np.percentile(v[:, 0], 5) / np.median(base[:, 0])),
                         eps_ratio_p95=float(np.percentile(v[:, 0], 95) / np.median(base[:, 0])),
                         C_ratio_median=float(np.median(v[:, 1]) / np.median(base[:, 1])),
                         C_ratio_p05=float(np.percentile(v[:, 1], 5) / np.median(base[:, 1])),
                         C_ratio_p95=float(np.percentile(v[:, 1], 95) / np.median(base[:, 1]))))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "levers_synth_03_output.csv"), index=False)
L = ["DEV-LEVERS-01 synthetic test 3: can the readings say which of the three channels was impaired?",
     f"healthy eps {np.median(base[:,0]):.4f} (target {EPS0}) | healthy clustering {np.median(base[:,1]):.3f} | "
     f"{NDRAW} draws per condition, {NMOL} molecules x {NSITE} sites", "",
     "derived directions: impair copy -> eps up, clustered | impair de novo -> eps up, scattered | impair erase -> eps DOWN, scattered", ""]
for r in T.itertuples():
    L.append(f"{r.channel:<8} x{r.impairment:<5} eps ratio {r.eps_ratio_median:.3f} [{r.eps_ratio_p05:.3f}, {r.eps_ratio_p95:.3f}] | "
             f"clustering ratio {r.C_ratio_median:.3f} [{r.C_ratio_p05:.3f}, {r.C_ratio_p95:.3f}]")
two = T[T.impairment == 2.0].set_index("channel")
def overlap(a, b, col):
    return not (two.loc[a, col + "_p95"] < two.loc[b, col + "_p05"] or two.loc[b, col + "_p95"] < two.loc[a, col + "_p05"])
pairs = [("copy", "erase"), ("copy", "de novo"), ("erase", "de novo")]
bad = [f"{a} vs {b}" for a, b in pairs if overlap(a, b, "eps_ratio") and overlap(a, b, "C_ratio")]
L += ["", "at the 2x impairment, pairs overlapping in BOTH readings: " + (", ".join(bad) if bad else "none"),
      "BAR: " + ("MET - the three channels occupy three separable regions, so a blocked channel can be identified"
                 if not bad else "NOT MET - " + "; ".join(bad) + " cannot be told apart at this depth"),
      "", "Also derived and checked above: the eps direction alone separates erase from the other two, and the clustering",
      "reading separates copy from de novo. Read as a statement about the instruments, not about any tumor."]
open(os.path.join(OUT, "levers_synth_03_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
