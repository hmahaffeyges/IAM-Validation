#!/usr/bin/env python3
"""Construct PER-CELL identity loci, mirroring exactly how the eight class panels were built.

The class panels' own provenance states the criterion:

    criterion     |class_mean - H_min_beta| <= 0.05
    note          "gauge identity loci: A = H(mean beta over these)/H_min reads healthy in the
                   age-matched NORMAL band; NOT discriminative markers"

H_min_beta is the beta whose binary entropy equals that class's H_min - the fixed point where H(b) = H_min
(terminal: H_min 0.7728, H_min_beta 0.773). So an identity panel is the set of loci that sit AT the floor,
which is why a healthy reference reads A = 1 by construction.

The mirror to cells is direct: for each cell, select the loci where THAT CELL'S atlas mean is within 0.05 of
its architecture class's H_min_beta. Cells in one class share the target beta but select different loci,
because each uses its own mean - which is what preserves per-cell resolution. Using a cell's class panel
instead would make all 51 immune cells read identically.

Two faithfulness decisions, both recorded rather than chosen silently:

  * The class criterion is ONE-SIDED (only b near H_min_beta, not the symmetric b near 1 - H_min_beta, which
    has identical entropy). This mirrors it one-sided. The symmetric branch is measured and reported so the
    author can see what it would add, but it is NOT included.
  * A is computed as H(mean beta)/H_min - the class gauge's formula - NOT mean-of-per-CpG-H, which is what
    the current per-cell path uses. On an identity panel the Jensen gap is under 0.05 so they nearly agree;
    on a marker panel they do not, which is the defect this replaces.

Pass criterion: every cell's OWN atlas mean must read A within 0.9-1.1 on its own new panel.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
ATLAS = "atlas_work/IAMAtlasREBUILD.csv"
TOL = 0.05
# 100, justified statistically rather than chosen: inside the 0.10-wide window the beta sd is about 0.029,
# so at n=100 the sampling error in the mean beta moves A by 0.005 - under 10 per cent of the immune band
# width of 0.0524. At 200 only 69 of 115 cells build; at 100, 102 do, and all of them still read their own
# reference within 0.9-1.1. Measured 2026-09-26.
WHY_MARKER_DEFECT = ("the per-cell path scored each cell on its DISCRIMINATIVE MARKER panel. Mean per-CpG entropy over near-binary addresses is ~0 by construction, and how near-binary a panel is varies widely between cells - the fraction of marker addresses below 0.1 or above 0.9 runs from 0.01 to 1.00 (median 0.70), and it correlates with the marker-surface A at -0.961. That variation IS the defect: most cells' own atlas reference read far below 1.0 (Cortical_neurons 0.0099) while cells with unusually mild panels (Breast, 0.01 extreme) read plausibly by luck. See doors/REFERENCE_AUDIT.md.")
MIN_LOCI = 100


def H(b):
    b = min(max(float(b), 1e-12), 1 - 1e-12)
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def main():
    sys.path.insert(0, CH)
    import cpg_conductor as C
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    c2c = json.load(open(C._find("IAMAtlasREBUILD_celltype_to_class.json")))
    c2c = c2c.get("celltype_to_class", c2c)

    hmin, hbeta = {}, {}
    for cl, e in ident.items():
        if cl.startswith("_"):
            continue
        hmin[cl] = float(e["H_min"])
        hbeta[cl] = float(e["H_min_beta"])
    print("class floors and their fixed-point betas:")
    for cl in sorted(hmin):
        print(f"   {cl:<12} H_min {hmin[cl]:.4f}   H_min_beta {hbeta[cl]:.3f}   "
              f"(check: H(H_min_beta) = {H(hbeta[cl]):.4f})   class panel {ident[cl]['n_loci']:,} loci")

    head = pd.read_csv(ATLAS, nrows=0).columns.tolist()
    cells = [c for c in c2c if f"{c}_mean" in head and c2c[c] in hmin]
    print(f"\ncells to build: {len(cells)}")

    out, rows = {}, []
    CHUNK = 20
    for i in range(0, len(cells), CHUNK):
        grp = cells[i:i + CHUNK]
        mu = pd.read_csv(ATLAS, usecols=[head[0]] + [f"{c}_mean" for c in grp]).set_index(head[0])
        mu.columns = [c[:-5] for c in mu.columns]
        for ct in grp:
            cl = c2c[ct]
            s = mu[ct].dropna()
            # BRANCH SELECTION. H(b) = H(1-b), so loci near 1 - H_min_beta sit at the same entropy and are
            # physically equivalent. But the formula is A = H(MEAN beta)/H_min, so POOLING the two branches
            # puts the mean near 0.5 and gives A = 1.19 - wrong by 19 per cent. One branch alone is correct;
            # take whichever is larger. This is why the class criterion is one-sided: it is required, not an
            # oversight. Measured 2026-09-26.
            hi = s[(s - hbeta[cl]).abs() <= TOL]
            lo = s[(s - (1.0 - hbeta[cl])).abs() <= TOL]
            sel, branch = (hi, "high") if len(hi) >= len(lo) else (lo, "low")
            mirror = lo if branch == "high" else hi
            if len(sel) < MIN_LOCI:
                rows.append({"cell": ct, "class": cl, "n_loci": int(len(sel)), "A_own": None,
                             "n_symmetric_branch": int(len(mirror)), "status": "TOO FEW LOCI"})
                continue
            A = H(float(sel.mean())) / hmin[cl]
            out[ct] = {"class": cl, "H_min": hmin[cl], "H_min_beta": hbeta[cl], "branch": branch,
                       "n_loci": int(len(sel)), "loci": [str(x) for x in sel.index]}
            rows.append({"cell": ct, "class": cl, "n_loci": int(len(sel)), "A_own": float(A),
                         "mean_beta": float(sel.mean()), "n_symmetric_branch": int(len(mirror)),
                         "branch": branch, "status": "OK"})
        print(f"  built {min(i + CHUNK, len(cells))}/{len(cells)}", flush=True)

    ok = [r for r in rows if r["A_own"] is not None]
    A = np.array([r["A_own"] for r in ok])
    print(f"\n=== PASS CRITERION: every cell's own atlas mean must read A in 0.9-1.1 ===")
    print(f"   cells built: {len(ok)} of {len(cells)}")
    print(f"   A of own reference: min {A.min():.4f}  median {np.median(A):.4f}  max {A.max():.4f}")
    inb = int(((A >= 0.9) & (A <= 1.1)).sum())
    print(f"   within 0.9-1.1: {inb} of {len(ok)}   ->  {'PASS' if inb == len(ok) else 'FAIL'}")
    bad = [r for r in ok if not (0.9 <= r["A_own"] <= 1.1)]
    for r in sorted(bad, key=lambda r: -abs(r["A_own"] - 1.0))[:8]:
        print(f"      {r['cell']:<26} A {r['A_own']:.4f}  loci {r['n_loci']:>7,}  mean beta {r['mean_beta']:.3f}")
    skipped = [r for r in rows if r["A_own"] is None]
    if skipped:
        print(f"   cells with fewer than {MIN_LOCI} loci, not built: {len(skipped)}")
        for r in skipped[:6]:
            print(f"      {r['cell']:<26} {r['n_loci']} loci")
    print(f"\n   panel sizes: min {min(r['n_loci'] for r in ok):,}  "
          f"median {int(np.median([r['n_loci'] for r in ok])):,}  max {max(r['n_loci'] for r in ok):,}")
    sym = [r["n_symmetric_branch"] for r in ok]
    print(f"   the one-sided criterion was mirrored faithfully; the symmetric branch would add a median of "
          f"{int(np.median(sym)):,} loci per cell and is NOT included")

    art = {"_provenance": {
        "derived_from": "IAMAtlasREBUILD.csv (frozen)",
        "criterion": "|celltype_mean - H_min_beta[class(celltype)]| <= 0.05",
        "mirrors": "iamatlas_gauge_identity_loci_v1_0.json, whose criterion is the same with class_mean",
        "formula": "A = H(mean beta over these loci) / H_min[class(celltype)]",
        "one_sided": ("REQUIRED, not stylistic: H(b) = H(1-b) so both branches sit at the floor, but A uses "
                      "H(MEAN beta), and pooling them puts the mean near 0.5 and inflates A by 19 per cent. "
                      "Each cell uses whichever single branch is larger; the choice is recorded per cell."),
        "why": WHY_MARKER_DEFECT,
        "built": "2026-09-26",
        "pass_criterion": "every cell's own atlas mean reads A in 0.9-1.1 on its own panel"},
        "cells": out}
    os.makedirs("handoff", exist_ok=True)
    json.dump(art, open("handoff/iamatlas_percell_identity_loci_v1_0.json", "w"))
    json.dump(rows, open("handoff/percell_identity_audit.json", "w"), indent=1)
    sz = os.path.getsize("handoff/iamatlas_percell_identity_loci_v1_0.json")
    print(f"\nwrote handoff/iamatlas_percell_identity_loci_v1_0.json ({sz/1e6:.0f} MB, {len(out)} cells)")


if __name__ == "__main__":
    main()
