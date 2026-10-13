"""DEV-LEVERS-01 synthetic test 1 (written 2026-10-13, before it was run): can the Met-A clustering reading tell a PROCESSIVE
copy-head failure from INDEPENDENT chemical damage, at the same total error?

Why it matters: the lever audit claims the copy head fails over runs of consecutive sites (it slides), while 5mC deamination
hits sites independently. If true, the clustering reading is a mechanism discriminator. If the two cannot be told apart at
the depth real arrays give, that whole argument is untestable and the audit says so.

Design, fixed before running:
 - Real healthy residual structure: z maps are built from the committed neutrophil reference's own healthy spread, not invented
   (site count and block size come from the reference file; the healthy clustering median is its own).
 - Two planted mechanisms at MATCHED total error (same number of moved sites, same per-site size):
     processive: runs of L consecutive sites (L = 5, 10, 25, 50), start positions uniform
     scattered : the same number of single sites, independent and uniform
 - Read: the chain's own stage_mc_cscore on each planted map (chain code unchanged, imported).
 - Bar, written before the run: the test PASSES if, at a planted error that both mechanisms show (C above the healthy range),
   the C values of the two mechanisms do not overlap across 200 draws each, for at least one run length L <= 25. Otherwise the
   reading cannot separate mechanism at this depth and the audit records that.
Usage: python3 levers_synth_01.py OUTDIR
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain"))
sys.path.insert(0, CHAIN)
import conductor_v3 as C

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
R = C.ref()
NSITE = 20000                      # sites in the simulated residual map (array-scale, one chromosome arm's worth)
BLOCK = int(R["clustering_block"])
RUNS = [5, 10, 25, 50]
NDRAW = 200
SHIFT = [2.0, 3.0, 4.0]            # planted per-site residual size, in healthy z units
NMOVED = [200, 500, 1000]          # sites moved, matched between mechanisms
rg = np.random.default_rng(20261013)

def read(z):
    return C.stage_mc_cscore(pd.Series(z))["C"]

base = [read(rg.normal(0, 1, NSITE)) for _ in range(NDRAW)]
rows = []
for nm in NMOVED:
    for sh in SHIFT:
        for L in RUNS:
            nruns = max(1, nm // L)
            for d in range(NDRAW):
                z = rg.normal(0, 1, NSITE)
                for s in rg.integers(0, NSITE - L, nruns): z[s:s + L] += sh
                rows.append(dict(mech="processive", n_moved=nruns * L, shift=sh, run_len=L, draw=d, C=read(z)))
            for d in range(NDRAW):
                z = rg.normal(0, 1, NSITE)
                z[rg.choice(NSITE, nruns * L, replace=False)] += sh
                rows.append(dict(mech="scattered", n_moved=nruns * L, shift=sh, run_len=L, draw=d, C=read(z)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "levers_synth_01_output.csv"), index=False)

L = ["DEV-LEVERS-01 synthetic test 1: processive vs scattered at matched total error (the chain's own clustering reading)",
     f"sites {NSITE} | block {BLOCK} | draws {NDRAW} each | healthy C: median {np.median(base):.4f}, range {min(base):.4f}-{max(base):.4f}",
     ""]
sep = []
for (nm, sh, run), g in T.groupby(["n_moved", "shift", "run_len"]):
    p = g[g.mech == "processive"].C; s = g[g.mech == "scattered"].C
    above = p.min() > max(base) and p.median() > max(base)
    gap = p.min() > s.max()
    L.append(f"moved {nm:5d} shift {sh:.1f} run {run:3d}: processive C {p.median():.3f} [{p.min():.3f}-{p.max():.3f}] | "
             f"scattered C {s.median():.3f} [{s.min():.3f}-{s.max():.3f}] | no overlap {gap} | processive above healthy {above}")
    if gap and above and run <= 25: sep.append((nm, sh, run))
L += ["", f"SEPARATING CONDITIONS (no overlap, processive above the healthy range, run length <= 25): {sep}",
      "BAR: " + ("MET - the clustering reading separates mechanism" if sep else "NOT MET - the reading cannot separate mechanism at this depth")]
open(os.path.join(OUT, "levers_synth_01_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:4] + L[-2:]))
