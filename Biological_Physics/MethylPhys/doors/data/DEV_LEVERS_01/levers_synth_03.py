"""DEV-LEVERS-01 synthetic test 3, version 3 (2026-10-13: the healthy balance is set with the RESTORE rate; see the note below) (written 2026-10-13, before it was run; v1 kept as levers_synth_03_v1_INVALID.*).

Why v1 was thrown away, before its verdict was read: it ran ONE interval from a perfectly methylated start, so the held error
never reached its balance - the healthy draw read eps 0.0009 against its own target of 0.032, a factor of 35 - and it scaled
the erase rate UP to represent an impaired erase channel, when impairing a channel means running it SLOWER. Its verdict
("copy and de novo cannot be told apart") is therefore about a simulation that was not at balance and is not reported as a
finding.

The question (unchanged): if one of the three channels is impaired, can the readings say WHICH one?

The foundation (CANON/foundation.json, entry `balance`): eps = f/(f+g), reached in about 1/(f+g) intervals. Three operations
feed the two rates:
  - COPY failure: a LOSS, from a processive enzyme, so losses arrive in runs.
  - DE NOVO: a RESTORE, site by site, scattered.
  - ERASE: a LOSS, site by site, scattered.
Impairing a channel means running it slower, which gives the derived directions with nothing fitted:
  - impair COPY (more failures per copy) -> f rises -> eps rises, clustered.
  - impair DE NOVO (slower restore)      -> g falls -> eps rises, scattered.
  - impair ERASE (slower active removal)  -> f falls -> eps FALLS, scattered.

Design, fixed before running:
  - 400 molecules x 2000 sites; the healthy rates are iterated for 300 intervals to reach the balance, and the script REFUSES
    to score unless the healthy draw sits within 10 % of eps_0 = 0.032 (foundation entry eps0). That check is what v1 lacked.
  - the impairment is then applied for a further 300 intervals, so the readings are of a new balance, not of a transient.
  - impairment factors 1.25x, 1.5x, 2x on one channel at a time; copy failures arrive in runs of 15 sites (test 1 showed the
    clustering reading separates mechanism above about 10).
  - readings: eps by the chain's definition (isolated unmethylated calls between two methylated calls over opportunities) and
    the clustering reading (variance of 50-site block means x 50 / site variance), both as ratios to the healthy draws.
  - BAR: at the 2x impairment, no pair of channels may overlap in BOTH readings across 200 draws each. If a pair does, the
    three-channel picture is not separable at this depth, and tumors with a blocked erase channel cannot serve as a natural
    experiment - which the audit would then say.
Writes levers_synth_03_output.csv and levers_synth_03_summary.txt. Usage: python3 levers_synth_03.py [OUTDIR]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
# The readings come from the CHAIN, not from a second implementation of the same physics (rule 23; v1 of this test re-derived
# them and got the erase channel's direction backwards). _H and the clustering are the chain's own functions.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain")))
from stage_q_iam_a import _H                      # the chain's entropy
from conductor_v3 import _clustering              # the chain's clustering reading (blocks in genomic order)
# COST DECISION 2026-10-13, taken BEFORE any result of this version was seen (the script prints only at the end, and the first
# attempt at 400 molecules x 200 draws was stopped at 70 minutes with hours still to run): molecules 400 -> 200 and draws per
# condition 200 -> 60. Nothing about the test's logic, rates, bars or conditions changed. The spread of each reading is reported
# alongside its median, so a reader can see whether 60 draws was enough to separate the channels; if any pair of channels is
# closer than its spread, the verdict is 'cannot separate at this precision' and the draws go back up rather than the bar moving.
NMOL, NSITE, NDRAW, RUN, EPS0, BLOCK, NSTEP = 200, 2000, 60, 15, 0.032, 50, 300
rg = np.random.default_rng(20261013)
F_COPY, F_ERASE = 0.0020, 0.0010                     # per site per interval, healthy (scaled below)
G_DENOVO = (F_COPY + F_ERASE) * (1 - EPS0) / EPS0    # the restore rate that puts the STATE balance at eps_0
# WHY v3 CHANGES THE CALIBRATION (v2 kept as *_v2_REFUSED.*). The chain's held-error reading matches an ISOLATED loss: a
# methylated call on BOTH sides of an unmethylated one. A run of two or more consecutive losses therefore contributes EXACTLY
# ZERO to it - not even its ends. That is a real property of the instrument, read out of the code, and it is why the held-error
# meter and the clustering meter are complementary rather than redundant: the first is blind to clustered loss, the second is
# built for it. Consequence here: most loss in this test is planted in runs of 15, so raising the LOSS rate cannot raise the
# reading - v2's bisection ran to the top of its bracket and still read 0.0172 against the 0.032 target, and refused to score,
# correctly. v3 sets the healthy balance with the RESTORE rate instead: it is the rate that defines the balance and it moves
# the reading monotonically. Both the state balance and the chain's reading are reported, so the gap is documented.

def evolve(M, steps, fc=1.0, fe=1.0, gd=1.0):
    for _ in range(steps):
        k = rg.poisson(F_COPY * LOSS * fc * NSITE / RUN, NMOL)       # copy failures, in runs
        for i, kk in enumerate(k):
            for s in rg.integers(0, NSITE - RUN, kk): M[i, s:s + RUN] = False
        M &= rg.random((NMOL, NSITE)) > F_ERASE * LOSS * fe                 # erase: scattered loss
        lost = ~M
        M |= lost & (rg.random((NMOL, NSITE)) < G_DENOVO * GD * gd)       # de novo: scattered restore
    return M

def read(M):
    """The chain's two readings: the held error by its definition (isolated unmethylated calls between two methylated calls
    over the opportunities), and the chain's own clustering reading on the per-site residual map."""
    a, b, c = M[:, :-2], M[:, 1:-1], M[:, 2:]
    eps = (a & ~b & c).sum() / max((a & c).sum(), 1)
    u = pd.Series((~M).mean(0))                                  # per-site unmethylated fraction
    z = (u - u.mean()) / (u.std() or 1.0)                        # the residual map the chain clusters
    cl = _clustering(z)
    assert cl is not None, 'the chain refused the clustering reading: too few blocks'
    return eps, float(cl), _H(min(max(eps, 1e-9), 0.5))

def draw(**kw):
    M = evolve(np.ones((NMOL, NSITE), dtype=bool), NSTEP)            # to the healthy balance
    if kw: M = evolve(M, NSTEP, **kw)                                # then to the impaired balance
    return read(M)

LOSS = 1.0
GD = 1.0
def _healthy_read(k, n=24):
    global GD; GD = k
    return float(np.median([draw()[0] for _ in range(n)]))
# bisect the RESTORE multiplier so the chain READS eps_0; a LOWER restore rate raises the reading, so the bracket is inverted
lo, hi = 0.02, 2.0
_rl, _rh = _healthy_read(lo), _healthy_read(hi)
assert _rh < EPS0 < _rl, f"the bracket does not contain eps_0: reading {_rl:.4f} at restore x{lo} and {_rh:.4f} at x{hi}"
for _ in range(10):
    mid = (lo + hi) / 2
    if _healthy_read(mid) > EPS0: lo = mid
    else: hi = mid
GD = (lo + hi) / 2
base = np.array([draw() for _ in range(NDRAW)])      # columns: eps, chain clustering, chain entropy H(eps)
hm = float(np.median(base[:, 0]))
L = ["DEV-LEVERS-01 synthetic test 3 (v2): can the readings say which of the three channels was impaired?",
     f"healthy eps at balance {hm:.4f} (target {EPS0}) | healthy clustering {np.median(base[:,1]):.3f} | "
     f"{NDRAW} draws, {NMOL} molecules x {NSITE} sites, {NSTEP} intervals to balance | restore multiplier calibrated to {GD:.4f} so the CHAIN READS eps_0; state balance f/(f+g) = {(F_COPY + F_ERASE) / (F_COPY + F_ERASE + G_DENOVO * GD):.4f}", ""]
if not (0.9 * EPS0 <= hm <= 1.1 * EPS0):
    L += [f"REFUSED TO SCORE: the healthy draw is not at the balance the foundation sets ({hm:.4f} vs {EPS0}); the rates or the",
          "number of intervals are wrong, and nothing read from an off-balance simulation means anything."]
    open(os.path.join(OUT, "levers_synth_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L)); sys.exit(0)
rows = []
for ch, kw, inv in (("copy", "fc", False), ("erase", "fe", True), ("de novo", "gd", True)):
    for mult in (1.25, 1.5, 2.0):
        v = np.array([draw(**{kw: (1 / mult if inv else mult)}) for _ in range(NDRAW)])
        rows.append(dict(channel=ch, impairment=mult,
                         eps_ratio_median=float(np.median(v[:, 0]) / hm), eps_ratio_p05=float(np.percentile(v[:, 0], 5) / hm),
                         eps_ratio_p95=float(np.percentile(v[:, 0], 95) / hm),
                         C_ratio_median=float(np.median(v[:, 1]) / np.median(base[:, 1])),
                         H_ratio_median=float(np.median(v[:, 2]) / np.median(base[:, 2])),
                         C_ratio_p05=float(np.percentile(v[:, 1], 5) / np.median(base[:, 1])),
                         C_ratio_p95=float(np.percentile(v[:, 1], 95) / np.median(base[:, 1]))))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "levers_synth_03_output.csv"), index=False)
L += ["derived directions: impair copy -> eps up, clustered | impair de novo -> eps up, scattered | impair erase -> eps DOWN, scattered", ""]
for r in T.itertuples():
    L.append(f"{r.channel:<8} x{r.impairment:<5} eps ratio {r.eps_ratio_median:.3f} [{r.eps_ratio_p05:.3f}, {r.eps_ratio_p95:.3f}] | "
             f"clustering ratio {r.C_ratio_median:.3f} [{r.C_ratio_p05:.3f}, {r.C_ratio_p95:.3f}]")
two = T[T.impairment == 2.0].set_index("channel")
ov = lambda a, b, c: not (two.loc[a, c + "_p95"] < two.loc[b, c + "_p05"] or two.loc[b, c + "_p95"] < two.loc[a, c + "_p05"])
bad = [f"{a} vs {b}" for a, b in (("copy", "erase"), ("copy", "de novo"), ("erase", "de novo")) if ov(a, b, "eps_ratio") and ov(a, b, "C_ratio")]
dirs = {r.channel: ("up" if r.eps_ratio_median > 1 else "down") for r in T[T.impairment == 2.0].itertuples()}
L += ["", "measured directions at 2x: " + ", ".join(f"{k} {v}" for k, v in dirs.items()),
      "pairs overlapping in BOTH readings: " + (", ".join(bad) if bad else "none"),
      "BAR: " + ("MET - the three channels are separable, so a blocked channel can be identified" if not bad
                 else "NOT MET - " + "; ".join(bad) + " cannot be told apart at this depth"),
      "Read as a statement about the instruments, not about any tumor."]
open(os.path.join(OUT, "levers_synth_03_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
